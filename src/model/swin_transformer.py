import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

# --------------------------------------
# Mlp (Feed-Forward)
# --------------------------------------
class Mlp1D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --------------------------------------
# Partition / Reverse for 1D windows
# --------------------------------------
def window_partition_1d(x, window_size):
    """
    x: (B, L, C) -> windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    assert L % window_size == 0, "Sequence length must be divisible by window_size"
    x = x.view(B, L // window_size, window_size, C)
    windows = x.reshape(-1, window_size, C)
    return windows

def window_reverse_1d(windows, window_size, L):
    """
    windows: (num_windows*B, window_size, C) -> x: (B, L, C)
    """
    B = int(windows.shape[0] // (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.view(B, L, -1)
    return x

# --------------------------------------
# Window Attention 1D (with V2 enhancements)
# --------------------------------------
class WindowAttention1D(nn.Module):
    """
    1D window-based多头自注意力（Swin V2风格）
    - Cosine Attention + learnable logit_scale
    - Continuous Relative Position Bias via small MLP
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # learnable logit_scale for cosine attention
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # MLP for continuous relative position bias (input: scalar distance)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(1, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_heads, bias=False)
        )

        # build relative position index & distance table
        coords = torch.arange(window_size)
        # relative index in [0, 2W-2]
        rel_index = (coords[None, :] - coords[:, None]) + window_size - 1  # shape (W, W)
        self.register_buffer('relative_position_index', rel_index.long())
        # distance values in [-(W-1), W-1], shape (2W-1, 1)
        distances = torch.arange(-(window_size - 1), window_size, dtype=torch.float32).unsqueeze(-1)
        self.register_buffer('relative_distance_table', distances)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj     = nn.Linear(dim, dim)
        self.proj_drop= nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, W, C)
        mask: (num_windows, W, W) or None
        """
        B_, N, C = x.shape  # N == window_size
        # QKV
        bias = None
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)  # (B_, N, 3C)
        qkv = qkv.view(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B_, heads, N, head_dim)

        # Cosine attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))  # (B_, heads, N, N)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0/0.01)).exp()
        attn = attn * logit_scale

        # Continuous relative bias
        # distance table: (2W-1,1) -> bias_table: (2W-1, heads)
        bias_table = self.cpb_mlp(self.relative_distance_table)  # (2W-1, heads)
        # index to (W*W,)
        idx = self.relative_position_index.view(-1)  # (N*N,)
        # gather and reshape to (W, W, heads)
        bias = bias_table[idx].view(self.window_size, self.window_size, self.num_heads)
        # permute to (1, heads, W, W)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        # apply mask if given
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Attention output
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# --------------------------------------
# Swin Transformer Block 1D (V2)
# --------------------------------------
class SwinTransformerBlock1D(nn.Module):
    def __init__(self, dim, seq_len, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp1D(dim, int(dim*mlp_ratio), dim, nn.GELU, drop)
        self.use_checkpoint = use_checkpoint

        # build mask for SW
        if shift_size>0:
            idx = torch.arange(seq_len)
            mask = (idx//window_size).unsqueeze(0)
            windows = window_partition_1d(mask.unsqueeze(-1), window_size).squeeze(-1)
            attn_mask = windows.unsqueeze(1)-windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask==0, 0.0)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        def _inner(x):
            B,L,C = x.shape
            assert L==self.seq_len, "seq_len mismatch"
            shortcut = x
            x = self.norm1(x)
            if self.shift_size>0:
                x = torch.roll(x, -self.shift_size, dims=1)
            x_w = window_partition_1d(x, self.window_size)
            x_w = self.attn(x_w, self.attn_mask)
            x = window_reverse_1d(x_w, self.window_size, L)
            if self.shift_size>0:
                x = torch.roll(x, self.shift_size, dims=1)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(_inner, x)
        else:
            return _inner(x)

# --------------------------------------
# Patch Merging 1D
# --------------------------------------
class PatchMerging1D(nn.Module):
    def __init__(self, seq_len, dim):
        super().__init__()
        assert seq_len%2==0, "seq_len must even"
        self.seq_len=seq_len
        self.reduction=nn.Linear(2*dim, 2*dim, bias=False)
        self.norm=nn.LayerNorm(2*dim)
    def forward(self, x):
        B,L,C=x.shape
        x=x.view(B, L//2, 2, C)
        x=x.view(B, L//2, 2*C)
        x=self.norm(x)
        x=self.reduction(x)
        return x

# --------------------------------------
# BasicLayer1D & PatchEmbed1D & Model
# --------------------------------------
class BasicLayer1D(nn.Module):
    def __init__(self, dim, seq_len, depth, num_heads, window_size,
                 mlp_ratio, qkv_bias, drop, attn_drop, drop_path, downsample, use_checkpoint):
        super().__init__()
        self.blocks=nn.ModuleList()
        for i in range(depth):
            shift = 0 if i%2==0 else window_size//2
            self.blocks.append(
                SwinTransformerBlock1D(dim, seq_len, num_heads, window_size,
                                       shift, mlp_ratio, qkv_bias, drop,
                                       attn_drop, drop_path[i] if isinstance(drop_path,list) else drop_path,
                                       use_checkpoint)
            )
        self.downsample = downsample(seq_len, dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x=blk(x)
        if self.downsample:
            x=self.downsample(x)
        return x

class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        assert seq_len%patch_size==0
        self.proj=nn.Conv1d(in_chans, embed_dim, patch_size, stride=patch_size)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self, x):
        x=self.proj(x).transpose(1,2)
        x=self.norm(x)
        return x

class SwinTransformer1D(nn.Module):
    def __init__(self, seq_len=512, patch_size=8, in_chans=1, num_classes=10,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,3,6,6], window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.patch_embed=PatchEmbed1D(seq_len, patch_size, in_chans, embed_dim)
        total=sum(depths)
        dpr=[x.item() for x in torch.linspace(0, drop_path_rate, total)]
        self.layers=nn.ModuleList()
        cur_len=seq_len//patch_size
        cur_dim=embed_dim
        idx=0
        for i,depth in enumerate(depths):
            layer=BasicLayer1D(cur_dim, cur_len, depth, num_heads[i], window_size,
                               mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                               dpr[idx:idx+depth], PatchMerging1D if i<len(depths)-1 else None,
                               use_checkpoint)
            self.layers.append(layer)
            idx+=depth
            if i<len(depths)-1:
                cur_len//=2; cur_dim*=2
        self.norm=nn.LayerNorm(cur_dim)
        self.head=nn.Linear(cur_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        feat = x.mean(1)
        logits = self.head(feat)
        return logits, feat