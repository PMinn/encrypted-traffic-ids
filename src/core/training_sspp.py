from typing import Optional, Tuple, Dict
import os
import datetime
import logging
import json
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from numpy import typing as npt
import numpy as np
from tabulate import tabulate
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from data_processing.FlowMeter.extract_flow_features_73 import (
    get_feature_names_73,
    get_log_scale_features_name_73,
    get_std_scale_features_name_73,
)
from model.sspp import SSPP, CNNOnlyClassifier, TorchStandardScaler, SSPPConfig
from utils.early_stopping import EarlyStopping
from utils.alias import a2p
from utils.hash import sha256_file, sha256_text
from utils.save_log import log_json_artifact, log_npz_artifact
from utils.set_seed import set_seed_to
from utils.normalizer import fit_normalizer, transform_normalizer
import mlflow
import mlflow.pytorch
from model.sspp import SSPPConfig, SparseAutoencoder, CNNOnlyClassifier
import mlflow
import mlflow.pytorch

# =========================================================
# Training utilities
# =========================================================


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def evaluate_classifier(
    num_classes: int, model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0
    criterion = nn.CrossEntropyLoss()
    all_preds_list: list[npt.NDArray[np.int_]] = []
    all_labels_list: list[npt.NDArray[np.int_]] = []

    with torch.no_grad():
        for ss, pp, y in loader:
            ss, pp, y = ss.to(device), pp.to(device), y.to(device)
            logits = model(ss, pp)
            loss = criterion(logits, y)
            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy_from_logits(logits, y) * bsz
            total_n += bsz
            preds = logits.argmax(dim=1)
            all_preds_list.append(preds.detach().cpu().numpy())
            all_labels_list.append(y.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    # acc = float(accuracy_score(all_labels, all_preds))
    f1_macro = float(f1_score(all_labels, all_preds, average="macro"))
    f1_micro = float(f1_score(all_labels, all_preds, average="micro"))
    f1_weighted = float(f1_score(all_labels, all_preds, average="weighted"))
    f1_per_class_arr = f1_score(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )
    if not isinstance(f1_per_class_arr, (np.ndarray, list)):
        raise TypeError(
            f"Expected f1_per_class_arr to be np.ndarray or list, but got {type(f1_per_class_arr)}"
        )
    if f1_per_class_arr.shape[0] != num_classes:
        raise ValueError(
            f"Expected f1_per_class_arr to have shape ({num_classes},), but got {f1_per_class_arr.shape}"
        )
    f1_per_class = {cls: float(f1_per_class_arr[cls]) for cls in range(num_classes)}

    per_class_acc = {}
    for cls in range(num_classes):
        mask = all_labels == cls
        total = np.sum(mask)
        correct = np.sum(all_preds[mask] == all_labels[mask])
        per_class_acc[cls] = float(correct / total)
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_acc / max(total_n, 1),
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
        "val_per_class_acc": per_class_acc,
    }


# =========================================================
# Dataset
# =========================================================


class SSPPDataset(Dataset):
    """
    ss: [num_samples, 26]
    pp: [num_samples, N, 25]
    y : [num_samples]
    """

    def __init__(self, ss: torch.Tensor, pp: torch.Tensor, y: torch.Tensor):
        assert ss.ndim == 2, "SS must be [B, 26]"
        assert pp.ndim == 3, "PP must be [B, N, 25]"
        assert y.ndim == 1, "y must be [B]"
        assert (
            ss.shape[0] == pp.shape[0] == y.shape[0]
        ), "Batch size must match, but got {} vs {} vs {}".format(
            ss.shape[0], pp.shape[0], y.shape[0]
        )
        self.ss = ss.float()
        self.pp = pp.float()
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.ss[idx], self.pp[idx], self.y[idx]


def train_sae(
    num_classes: int,
    sae: SparseAutoencoder,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    l1_lambda: float = 1e-5,
    mlflow_config: Optional[Dict[str, str]] = None,
):
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        sae.train()
        running_loss = 0.0
        total_n = 0

        for ss, _, _ in train_loader:
            ss = ss.to(device)
            optimizer.zero_grad()

            _, recon = sae(ss)
            mse = F.mse_loss(recon, ss)
            loss = mse + l1_lambda * sae.sparse_penalty()
            loss.backward()
            optimizer.step()

            bsz = ss.size(0)
            running_loss += loss.item() * bsz
            total_n += bsz

        msg = (
            f"[SAE] Epoch {epoch:03d} | train_loss={running_loss / max(total_n, 1):.6f}"
        )
        if mlflow_config is not None:
            mlflow.log_metric(
                "sae_train_loss",
                float(running_loss / max(total_n, 1)),
                step=(epoch - 1),
            )
        if val_loader is not None:
            val_metrics = evaluate_sae(num_classes, sae, val_loader, device, l1_lambda)
            msg += f" | val_loss={val_metrics['loss']:.6f} | val_mse={val_metrics['mse']:.6f}"
            if mlflow_config is not None:
                mlflow.log_metric(
                    "sae_val_loss", float(val_metrics["loss"]), step=(epoch - 1)
                )
                mlflow.log_metric(
                    "sae_val_mse", float(val_metrics["mse"]), step=(epoch - 1)
                )
        print(msg)


def evaluate_sae(
    num_classes: int,
    sae: SparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
    l1_lambda: float,
) -> Dict[str, float]:
    sae.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_n = 0

    with torch.no_grad():
        for ss, _, y in loader:
            ss = ss.to(device)
            y = y.to(device)
            _, recon = sae(ss)
            mse = F.mse_loss(recon, ss)
            loss = mse + l1_lambda * sae.sparse_penalty()
            bsz = ss.size(0)
            total_loss += loss.item() * bsz
            total_mse += mse.item() * bsz
            total_n += bsz
    return {
        "loss": total_loss / max(total_n, 1),
        "mse": total_mse / max(total_n, 1),
    }


def train_cnn_only(
    num_classes: int,
    model: CNNOnlyClassifier,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    mlflow_config: Optional[Dict[str, str]] = None,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    last_f1_per_class = None
    last_val_per_class_acc = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        for _, pp, y in train_loader:
            pp, y = pp.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(pp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy_from_logits(logits, y) * bsz
            total_n += bsz

        msg = f"[CNN] Epoch {epoch:03d} | train_loss={total_loss / max(total_n, 1):.6f} | train_acc={total_acc / max(total_n, 1):.4f}"
        if mlflow_config is not None:
            mlflow.log_metric(
                "cnn_train_loss", float(total_loss / max(total_n, 1)), step=(epoch - 1)
            )
            mlflow.log_metric(
                "cnn_train_acc", float(total_acc / max(total_n, 1)), step=(epoch - 1)
            )

        if val_loader is not None:
            val_loss = 0.0
            val_acc = 0.0
            val_n = 0
            all_preds_list: list[npt.NDArray[np.int_]] = []
            all_labels_list: list[npt.NDArray[np.int_]] = []
            model.eval()
            with torch.no_grad():
                for _, pp, y in val_loader:
                    pp, y = pp.to(device), y.to(device)
                    logits = model(pp)
                    loss = criterion(logits, y)
                    bsz = y.size(0)
                    val_loss += loss.item() * bsz
                    val_acc += accuracy_from_logits(logits, y) * bsz
                    val_n += bsz
                    preds = logits.argmax(dim=1)
                    all_preds_list.append(preds.cpu().numpy())
                    all_labels_list.append(y.cpu().numpy())
            all_preds = np.concatenate(all_preds_list)
            all_labels = np.concatenate(all_labels_list)
            acc = float(accuracy_score(all_labels, all_preds))
            f1_per_class_arr = f1_score(
                all_labels,
                all_preds,
                average=None,
                labels=range(num_classes),
                zero_division=0,
            )
            if not isinstance(f1_per_class_arr, (np.ndarray, list)):
                raise TypeError(
                    f"Expected f1_per_class_arr to be np.ndarray or list, but got {type(f1_per_class_arr)}"
                )
            if f1_per_class_arr.shape[0] != num_classes:
                raise ValueError(
                    f"Expected f1_per_class_arr to have shape ({num_classes},), but got {f1_per_class_arr.shape}"
                )
            last_f1_per_class = {
                cls: float(f1_per_class_arr[cls]) for cls in range(num_classes)
            }
            last_val_per_class_acc = {}
            for cls in range(num_classes):
                mask = all_labels == cls
                total = np.sum(mask)
                correct = np.sum(all_preds[mask] == all_labels[mask])
                last_val_per_class_acc[cls] = float(correct / total)
            msg += f" | val_loss={val_loss / max(val_n, 1):.6f} | val_acc={val_acc / max(val_n, 1):.4f}"
            if mlflow_config is not None:
                mlflow.log_metric(
                    "cnn_val_loss", float(val_loss / max(val_n, 1)), step=(epoch - 1)
                )
                mlflow.log_metric("cnn_val_acc", float(acc), step=(epoch - 1))
        print(msg)
    last_val_per_class_acc_json = {str(k): v for k, v in last_val_per_class_acc.items()}
    last_f1_per_class_json = {str(k): v for k, v in last_f1_per_class.items()}
    if mlflow_config is not None:
        log_json_artifact(
            {"cnn_val_per_class_acc": last_val_per_class_acc_json},
            "cnn_val_per_class_acc.json",
        )
        log_json_artifact(
            {"cnn_f1_per_class": last_f1_per_class_json}, "cnn_f1_per_class.json"
        )


def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True


def train_fusion_stage(
    num_classes: int,
    sspp: SSPP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    mlflow_config: Optional[Dict[str, str]] = None,
):
    # Freeze SAE and CNN, train FCN only
    freeze_module(sspp.sae)
    freeze_module(sspp.cnn)
    unfreeze_module(sspp.fcn)

    sspp.to(device)
    optimizer = torch.optim.Adam(sspp.fcn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    last_f1_per_class = None
    last_val_per_class_acc = None

    for epoch in range(1, epochs + 1):
        sspp.train()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0

        for ss, pp, y in train_loader:
            ss, pp, y = ss.to(device), pp.to(device), y.to(device)
            optimizer.zero_grad()
            logits = sspp(ss, pp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bsz = y.size(0)
            total_loss += loss.item() * bsz
            total_acc += accuracy_from_logits(logits, y) * bsz
            total_n += bsz

        msg = f"[FCN] Epoch {epoch:03d} | train_loss={total_loss / max(total_n, 1):.6f} | train_acc={total_acc / max(total_n, 1):.4f}"
        if mlflow_config is not None:
            mlflow.log_metric(
                "fcn_train_loss", float(total_loss / max(total_n, 1)), step=(epoch - 1)
            )
            mlflow.log_metric(
                "fcn_train_acc", float(total_acc / max(total_n, 1)), step=(epoch - 1)
            )

        if val_loader is not None:
            metrics = evaluate_classifier(num_classes, sspp, val_loader, device)
            last_f1_per_class = metrics["f1_per_class"]
            last_f1_micro = metrics["f1_micro"]
            last_f1_weighted = metrics["f1_weighted"]
            last_val_per_class_acc = metrics["val_per_class_acc"]
            msg += f" | val_loss={metrics['loss']:.6f} | val_acc={metrics['acc']:.4f} | val_f1_macro={metrics['f1_macro']:.4f} | val_f1_micro={metrics['f1_micro']:.4f} | val_f1_weighted={metrics['f1_weighted']:.4f}"
            if mlflow_config is not None:
                mlflow.log_metric(
                    "fcn_val_loss", float(metrics["loss"]), step=(epoch - 1)
                )
                mlflow.log_metric(
                    "fcn_val_acc", float(metrics["acc"]), step=(epoch - 1)
                )
                mlflow.log_metric(
                    "fcn_val_f1_macro", float(metrics["f1_macro"]), step=(epoch - 1)
                )
                mlflow.log_metric(
                    "fcn_val_f1_micro", float(metrics["f1_micro"]), step=(epoch - 1)
                )
                mlflow.log_metric(
                    "fcn_val_f1_weighted",
                    float(metrics["f1_weighted"]),
                    step=(epoch - 1),
                )
        print(msg)
    last_val_per_class_acc_json = {str(k): v for k, v in last_val_per_class_acc.items()}
    last_f1_per_class_json = {str(k): v for k, v in last_f1_per_class.items()}
    if mlflow_config is not None:
        log_json_artifact(
            {"fcn_val_per_class_acc": last_val_per_class_acc_json},
            "fcn_val_per_class_acc.json",
        )
        log_json_artifact(
            {"fcn_f1_per_class": last_f1_per_class_json}, "fcn_f1_per_class.json"
        )


# =========================================================
# End-to-end helper
# =========================================================


class SSPPTrainer:
    def __init__(self, config: SSPPConfig, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.ss_scaler = TorchStandardScaler()
        self.pp_scaler = TorchStandardScaler()

        self.model = SSPP(config)

    # fit normalization
    def fit_scalers(self, train_ss: torch.Tensor, train_pp: torch.Tensor) -> None:
        mu_ss, sigma_ss = fit_normalizer(
            train_ss, [], [f for f in range(train_ss.shape[1])]
        )
        self.ss_scaler.mean = mu_ss
        self.ss_scaler.std = sigma_ss
        mu_pp, sigma_pp = fit_normalizer(
            train_pp, [], [f for f in range(train_pp.shape[1])]
        )
        self.pp_scaler.mean = mu_pp
        self.pp_scaler.std = sigma_pp
        # self.ss_scaler.fit(train_ss)
        # self.pp_scaler.fit(train_pp)
        return mu_ss, sigma_ss, mu_pp, sigma_pp

    # normalization
    def transform_inputs(
        self, ss: torch.Tensor, pp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ss_scaler.transform(ss), self.pp_scaler.transform(pp)

    def make_dataset(
        self, ss: torch.Tensor, pp: torch.Tensor, y: torch.Tensor
    ) -> SSPPDataset:
        ss_norm, pp_norm = self.transform_inputs(ss, pp)
        return SSPPDataset(ss_norm, pp_norm, y)

    def build_loaders(
        self,
        train_ss: torch.Tensor,
        train_pp: torch.Tensor,
        train_y: torch.Tensor,
        val_ss: Optional[torch.Tensor] = None,
        val_pp: Optional[torch.Tensor] = None,
        val_y: Optional[torch.Tensor] = None,
    ):
        mu_ss, sigma_ss, mu_pp, sigma_pp = self.fit_scalers(train_ss, train_pp)

        train_ds = self.make_dataset(train_ss, train_pp, train_y)
        val_ds = (
            self.make_dataset(val_ss, val_pp, val_y) if val_ss is not None else None
        )

        sae_train_loader = DataLoader(
            train_ds, batch_size=self.config.sae_batch_size, shuffle=True
        )
        sae_val_loader = (
            DataLoader(val_ds, batch_size=self.config.sae_batch_size, shuffle=False)
            if val_ds is not None
            else None
        )

        cnn_train_loader = DataLoader(
            train_ds, batch_size=self.config.cnn_batch_size, shuffle=True
        )
        cnn_val_loader = (
            DataLoader(val_ds, batch_size=self.config.cnn_batch_size, shuffle=False)
            if val_ds is not None
            else None
        )

        fcn_train_loader = DataLoader(
            train_ds, batch_size=self.config.fcn_batch_size, shuffle=True
        )
        fcn_val_loader = (
            DataLoader(val_ds, batch_size=self.config.fcn_batch_size, shuffle=False)
            if val_ds is not None
            else None
        )

        return (
            mu_ss,
            sigma_ss,
            mu_pp,
            sigma_pp,
            sae_train_loader,
            sae_val_loader,
            cnn_train_loader,
            cnn_val_loader,
            fcn_train_loader,
            fcn_val_loader,
        )

    def train(
        self,
        train_ss: torch.Tensor,
        train_pp: torch.Tensor,
        train_y: torch.Tensor,
        val_ss: Optional[torch.Tensor] = None,
        val_pp: Optional[torch.Tensor] = None,
        val_y: Optional[torch.Tensor] = None,
        mlflow_config: Optional[Dict[str, str]] = None,
    ):
        if self.config.seed is not None:
            set_seed_to(self.config.seed)

        if mlflow_config is not None:
            mlflow.set_experiment(mlflow_config["experiment_name"])
            # 讓 run 可追溯：特徵清單本身也做 hash
            # features_hash = sha256_text("|".join(features))
            mlflow.start_run(run_name=mlflow_config["run_name"])
            mlflow.log_params(self.config.__dict__)

        (
            mu_ss,
            sigma_ss,
            mu_pp,
            sigma_pp,
            sae_train_loader,
            sae_val_loader,
            cnn_train_loader,
            cnn_val_loader,
            fcn_train_loader,
            fcn_val_loader,
        ) = self.build_loaders(train_ss, train_pp, train_y, val_ss, val_pp, val_y)

        if mlflow_config is not None:
            log_npz_artifact(
                "normalize.npz",
                mu_ss=mu_ss,
                sigma_ss=sigma_ss,
                mu_pp=mu_pp,
                sigma_pp=sigma_pp,
            )

        # Stage 1a: train SAE
        train_sae(
            num_classes=self.config.num_classes,
            sae=self.model.sae,
            train_loader=sae_train_loader,
            val_loader=sae_val_loader,
            device=self.device,
            epochs=self.config.sae_epochs,
            lr=self.config.lr,
            l1_lambda=self.config.sae_l1_lambda,
            mlflow_config=mlflow_config,
        )

        # Stage 1b: train CNN separately with temp head, then copy backbone
        cnn_temp = CNNOnlyClassifier(
            num_classes=self.config.num_classes,
            pp_dim=self.config.pp_dim,
            negative_slope=self.config.negative_slope,
        )
        train_cnn_only(
            num_classes=self.config.num_classes,
            model=cnn_temp,
            train_loader=cnn_train_loader,
            val_loader=cnn_val_loader,
            device=self.device,
            epochs=self.config.cnn_epochs,
            lr=self.config.lr,
            mlflow_config=mlflow_config,
        )
        self.model.cnn.load_state_dict(cnn_temp.backbone.state_dict())

        # Stage 2: freeze SAE & CNN, train FCN only
        train_fusion_stage(
            num_classes=self.config.num_classes,
            sspp=self.model,
            train_loader=fcn_train_loader,
            val_loader=fcn_val_loader,
            device=self.device,
            epochs=self.config.fcn_epochs,
            lr=self.config.lr,
            mlflow_config=mlflow_config,
        )

        if mlflow_config is not None:
            mlflow.pytorch.log_model(self.model.cnn, "cnn")
            mlflow.pytorch.log_model(self.model.sae, "sae")
            mlflow.pytorch.log_model(self.model.fcn, "fcn")

        return self.model

    @torch.no_grad()
    def predict_proba(self, ss: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        self.model.to(self.device)
        ss_norm, pp_norm = self.transform_inputs(ss, pp)
        ss_norm = ss_norm.to(self.device)
        pp_norm = pp_norm.to(self.device)
        logits = self.model(ss_norm, pp_norm)
        return torch.softmax(logits, dim=1).cpu()

    @torch.no_grad()
    def predict(self, ss: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(ss, pp)
        return probs.argmax(dim=1)
