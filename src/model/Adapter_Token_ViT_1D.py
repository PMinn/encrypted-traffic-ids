from typing import Iterable, List, cast
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from model.mlphead import MLPHead


class FeedForward(nn.Module):
    # norm + MLP
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class Attention(nn.Module):
    # depth: numder of attention layer
    # heads: number of  multi heads 數量
    # layer norm + multi-head attention
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        # 用來決定是否需要將自注意力的輸出進行投影
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # generate qkv

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        qkv = cast(Tensor, self.to_qkv(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return cast(Tensor, self.to_out(out))


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in cast(list[tuple[Attention, FeedForward]], self.layers):
            x = attn(x) + x
            x = ff(x) + x
        return cast(Tensor, self.norm(x))


class Adapter_Token_ViT_1D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        # ==== Adapter Tokens 參數 ====
        flow_feat_dim: int,  # flow 統計特徵維度（標準化後）
        dim_head: int = 64,
        dropout: float = 0.5,
        emb_dropout: float = 0.5,
        # ==== Adapter Tokens 參數 ====
        num_flow_tokens: int = 2,  # 產生幾個 adapter tokens（1~2 常見）
    ):
        super(Adapter_Token_ViT_1D, self).__init__()
        # 考慮不能切分之情形
        assert (
            seq_len % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        # 算出patches數
        num_patches = seq_len // patch_size
        self.patch_dim = (
            1 * patch_size
        )  # 修改patch_dim為patch_size * channels，因為是1D資料
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.dim = dim

        # ==== Adapter Tokens 參數 ====
        if num_flow_tokens <= 0:
            raise ValueError("num_flow_tokens must be a positive integer.")
        self.flow_feat_dim = flow_feat_dim
        self.num_flow_tokens = num_flow_tokens

        # Patch Embedding
        # 將1D資料拆分成形狀為(b, c, h)的片段，然後將每個片段轉換成形狀為(b, h*p, c)的張量
        # 其中b是batch size，p是patch_size，h是(num_patches)表示1D數據經過拆分後的patch數目
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (n p) -> b n (p c)", p=self.patch_size
            ),  # change the input form to another form
            nn.LayerNorm(
                self.patch_dim
            ),  # normalize the contents in each patch that range patch_size
            nn.Linear(
                self.patch_dim, self.dim
            ),  # project each patch to the dimension of dim
            nn.LayerNorm(self.dim),  # the form is changed to be 'b n d'
        )

        # ==== flow_feat -> Adapter Tokens (B,F)->(B,K,D) ====
        if self.num_flow_tokens > 0:
            self.flow_proj = nn.Sequential(
                nn.Linear(self.flow_feat_dim, 4 * self.dim),
                nn.ReLU(),
                nn.Linear(4 * self.dim, self.num_flow_tokens * self.dim),
            )

        # Positional Embedding（CLS + FLOW_TOKENS + PATCHES）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + self.num_flow_tokens + num_patches, self.dim)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            self.dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = MLPHead(self.dim, num_classes)

    def forward(self, x: Tensor, flow_feat: Tensor) -> tuple[Tensor, Tensor]:
        """
        x         : [B, 1, seq_len]，seq_len 必須可以被 patch_size 整除
        flow_feat : [B, F]（標準化後的 flow 統計特徵）
        """
        # 1) 1D -> Patch embeddings
        x = self.to_patch_embedding(x)  # [B, N, D]
        batch, n, _ = x.shape

        # 2) CLS
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=batch)  # [B, 1, D]

        # 3) Adapter Tokens（把 flow_feat 投影成 K 個 token 並前置）
        flow_tokens = self.flow_proj(flow_feat).view(
            batch, self.num_flow_tokens, self.dim
        )  # [B, K, D]
        x = torch.cat((cls_tokens, flow_tokens, x), dim=1)  # [B, 1+K+N, D]

        # 4) 位置編碼 + dropout
        x = x + self.pos_embedding[:, : (1 + self.num_flow_tokens + n)]
        x = self.dropout(x)

        # 5) Transformer Encoder
        x = self.transformer(x)

        # 6) 取 CLS 做分類
        cls_token_features = self.to_cls_token(x[:, 0])
        logits = self.mlp_head(cls_token_features)

        return logits, cls_token_features
