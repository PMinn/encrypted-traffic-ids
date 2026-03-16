from typing import Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


# =========================================================
# Config
# =========================================================


@dataclass
class SSPPConfig:
    seed: int | None = None
    num_classes: int = 10  # T
    num_packets: int = 8  # N, paper says best at N=8
    ss_dim: int = 26  # SS vector dimension
    pp_dim: int = 25  # PP feature dimension per packet

    # training hyperparameters from the paper
    sae_epochs: int = 30
    cnn_epochs: int = 20
    fcn_epochs: int = 30

    sae_batch_size: int = 32
    cnn_batch_size: int = 64
    fcn_batch_size: int = 32

    lr: float = 1e-3
    dropout: float = 0.5

    # sparse penalty coefficient (paper mentions epsilon/lambda regularization parameter,
    # but does not provide a concrete value)
    sae_l1_lambda: float = 1e-5

    # LeakyReLU slope (paper says LeakyReLU but does not specify slope)
    negative_slope: float = 0.01


# =========================================================
# Standard scaler
# =========================================================


class TorchStandardScaler:
    """
    Simple standard scaler for torch tensors.
    Fit on training set only.
    """

    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.eps = eps

    def fit(self, x: torch.Tensor) -> 'TorchStandardScaler':
        # x shape:
        # SS -> [B, 26]
        # PP -> [B, N, 25]
        dims = tuple(range(x.ndim - 1))
        self.mean = x.mean(dim=dims, keepdim=True)
        self.std = x.std(dim=dims, keepdim=True).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted yet.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)


# =========================================================
# SAE
# Paper:
# - input SS vector normalized
# - two fully connected layers: 26 -> 64 -> 26
# - both sigmoid
# - latent representation is first FC layer output (64x1)
# - cost = MSE + sparse penalty (L1 norm on weights in first FC layer)
# =========================================================


class SparseAutoencoder(nn.Module):
    def __init__(self, ss_dim: int = 26):
        super().__init__()
        self.encoder = nn.Linear(ss_dim, 64)
        self.decoder = nn.Linear(64, ss_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.encoder(x))  # latent 64
        recon = torch.sigmoid(self.decoder(z))  # reconstruction 26
        return z, recon

    def sparse_penalty(self) -> torch.Tensor:
        l1_norm: torch.Tensor = torch.norm(self.encoder.weight, p=1)
        return l1_norm


# =========================================================
# 1D-CNN
# Paper:
# - input PP matrix normalized
# - 2 conv layers + 2 max pool layers
# - conv1: 64 filters, kernel size=3, stride=1, same padding
# - conv2: 32 filters
# - activation: LeakyReLU
# - pooling: kernel size=2, stride=2
# - when N=8, output after second pool is flattened to 64x1
#
# Tensor layout here:
# input pp: [B, N, 25]
# convert to [B, 25, N] for Conv1d with in_channels=25
# =========================================================


class PacketCNN(nn.Module):
    def __init__(self, pp_dim: int = 25, negative_slope: float = 0.01):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=pp_dim,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,  # same padding for odd kernel size
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, pp: torch.Tensor) -> torch.Tensor:
        # pp: [B, N, 25]
        x = pp.transpose(1, 2)  # [B, 25, N]
        x = self.act(self.conv1(x))  # [B, 64, N]
        x = self.pool1(x)  # [B, 64, N/2]
        x = self.act(self.conv2(x))  # [B, 32, N/2]
        x = self.pool2(x)  # [B, 32, N/4]
        x = x.flatten(start_dim=1)  # if N=8 => [B, 32*2] = [B, 64]
        return x


# =========================================================
# FCN
# Paper:
# - concatenate SAE output and 1D-CNN output
# - first FC: 32 neurons + BatchNorm + LeakyReLU + Dropout(0.5)
# - second FC: T neurons + Softmax
#
# In PyTorch classification, return logits and use CrossEntropyLoss,
# so we do NOT explicitly apply softmax in forward for training.
# =========================================================


class FusionClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.5,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        logits: torch.Tensor = self.fc2(x)
        return logits


# =========================================================
# Full SSPP wrapper
# =========================================================


class SSPP(nn.Module):
    def __init__(self, config: SSPPConfig):
        super().__init__()
        self.config = config
        self.sae = SparseAutoencoder(ss_dim=config.ss_dim)
        self.cnn = PacketCNN(pp_dim=config.pp_dim, negative_slope=config.negative_slope)
        self.fcn = FusionClassifier(
            in_dim=64 + 64,  # SAE latent 64 + CNN flattened 64 when N=8
            num_classes=config.num_classes,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
        )

    def forward_features(
        self, ss: torch.Tensor, pp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sae_latent, sae_recon = self.sae(ss)
        cnn_feat = self.cnn(pp)
        fused = torch.cat([sae_latent, cnn_feat], dim=1)
        return sae_latent, sae_recon, fused

    def forward(self, ss: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        _, _, fused = self.forward_features(ss, pp)
        logits: torch.Tensor = self.fcn(fused)
        return logits


class CNNOnlyClassifier(nn.Module):
    """
    For stage 1: train 1D-CNN separately.
    The paper says first train 1D-CNN separately, then freeze it in fusion stage.
    So we attach a temporary classification head here.
    """

    def __init__(
        self, num_classes: int, pp_dim: int = 25, negative_slope: float = 0.01
    ):
        super().__init__()
        self.backbone = PacketCNN(pp_dim=pp_dim, negative_slope=negative_slope)
        self.head = nn.Linear(64, num_classes)

    def forward(self, pp: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pp)
        logits: torch.Tensor = self.head(feat)
        return logits
