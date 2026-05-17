from typing import Iterable, cast
import torch
from torch import Tensor
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from model.mlphead import MLPHead


class FeedForward(nn.Module):
    # norm + MLP
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

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
        super().__init__()
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
        super().__init__()
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


class ViT1D(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 1,
        dim_head: int = 64,
        dropout: float = 0.5,
        emb_dropout: float = 0.5,
    ):
        super().__init__()
        # 考慮不能切分之情形
        assert (
            seq_len % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        # 算出patches數
        num_patches = seq_len // patch_size
        # patch_dim = int(dim // (num_patches ** 0.5))
        patch_dim = (
            channels * patch_size
        )  # 修改patch_dim為patch_size * channels，因為是1D資料
        self.patch_size = patch_size

        # Patch Embedding
        # 將1D資料拆分成形狀為(b, c, h)的片段，然後將每個片段轉換成形狀為(b, h*p, c)的張量
        # 其中b是batch size，p是patch_size，h是(num_patches)表示1D數據經過拆分後的patch數目
        self.to_patch_embedding = nn.Sequential(
            # Rearrange('b c (h p) -> b (h p) c', h=num_patches),
            Rearrange(
                "b c (n p) -> b (n 1) (p 1 c)", n=num_patches, p=patch_size
            ),  # change the input form to another form
            nn.LayerNorm(
                patch_size
            ),  # normalize the contents in each patch that range patch_size
            nn.Linear(patch_size, dim),  # project each patch to the dimension of dim
            nn.LayerNorm(dim),  # the form is changed to be 'b n d'
        )

        # Positional Embedding
        # 為每個patch引入位置嵌入
        # 生成形狀為(1, num_patches+1, dim)的位置嵌入，用來表示每個patch的位置信息
        # nn.Parameter makes the parameter be able to be learned
        # torch.randn form a tensor that the mean equals 0, and the stadard deviation equals 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Classification Token
        # 額外引入一個特殊的分類token，用來表示全局資訊
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim,  # dimension
            depth,  # number of encoder lyaers
            heads,  # number of multi-heads
            dim_head,  # head dimension
            mlp_dim,  # FeedForward dimension in encoder
            dropout,
        )

        # Classification Head
        # 使用MLP來進行分類
        self.to_cls_token = nn.Identity()

        self.mlp_head = MLPHead(dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # b, c, h = x.size()

        # 將1D資料轉換成patch embedding
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        # 引入分類token並將位置嵌入加到patch embedding中
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        # 使用dropout進行隨機丟棄以防止過擬合

        # 通過Transformer Encoder進行特徵編碼
        x = self.transformer(x)

        # 提取分類token的特徵並通過MLP進行分類預測
        cls_token_features: Tensor = self.to_cls_token(x[:, 0])

        logits: Tensor = self.mlp_head(cls_token_features)
        # logits = self.mlp_head(generated_cls_token)

        return logits, cls_token_features
