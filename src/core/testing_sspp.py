from typing import Optional, Tuple, Dict, cast
import os
import datetime
import logging
import json
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
from core.testing_adapter_token_vit import per_class_metrics
from data_processing.FlowMeter.extract_flow_features_73 import (
    get_feature_names_73,
    get_log_scale_features_name_73,
    get_std_scale_features_name_73,
)
from model.sspp import SSPP, CNNOnlyClassifier, TorchStandardScaler, SSPPConfig
from utils.early_stopping import EarlyStopping
from utils.alias import a2p
from utils.hash import sha256_file, sha256_text
from utils.json import JSONObject
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


# def evaluate_classifier(
#     num_classes: int, model: nn.Module, loader: DataLoader, device: torch.device
# ) -> Dict[str, float]:
#     model.eval()
#     total_loss = 0.0
#     total_acc = 0.0
#     total_n = 0
#     criterion = nn.CrossEntropyLoss()
#     all_preds_list: list[npt.NDArray[np.int_]] = []
#     all_labels_list: list[npt.NDArray[np.int_]] = []

#     with torch.no_grad():
#         for ss, pp, y in loader:
#             ss, pp, y = ss.to(device), pp.to(device), y.to(device)
#             logits = model(ss, pp)
#             loss = criterion(logits, y)
#             bsz = y.size(0)
#             total_loss += loss.item() * bsz
#             total_acc += accuracy_from_logits(logits, y) * bsz
#             total_n += bsz
#             preds = logits.argmax(dim=1)
#             all_preds_list.append(preds.detach().cpu().numpy())
#             all_labels_list.append(y.detach().cpu().numpy())
#     all_preds = np.concatenate(all_preds_list)
#     all_labels = np.concatenate(all_labels_list)
#     # acc = float(accuracy_score(all_labels, all_preds))
#     f1_macro = float(f1_score(all_labels, all_preds, average="macro"))
#     f1_micro = float(f1_score(all_labels, all_preds, average="micro"))
#     f1_weighted = float(f1_score(all_labels, all_preds, average="weighted"))
#     f1_per_class_arr = f1_score(
#         all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
#     )
#     if not isinstance(f1_per_class_arr, (np.ndarray, list)):
#         raise TypeError(
#             f"Expected f1_per_class_arr to be np.ndarray or list, but got {type(f1_per_class_arr)}"
#         )
#     if f1_per_class_arr.shape[0] != num_classes:
#         raise ValueError(
#             f"Expected f1_per_class_arr to have shape ({num_classes},), but got {f1_per_class_arr.shape}"
#         )
#     f1_per_class = {cls: float(f1_per_class_arr[cls]) for cls in range(num_classes)}

#     per_class_acc = {}
#     for cls in range(num_classes):
#         mask = all_labels == cls
#         total = np.sum(mask)
#         correct = np.sum(all_preds[mask] == all_labels[mask])
#         per_class_acc[cls] = float(correct / total)
#     return {
#         "loss": total_loss / max(total_n, 1),
#         "acc": total_acc / max(total_n, 1),
#         "f1_macro": f1_macro,
#         "f1_micro": f1_micro,
#         "f1_weighted": f1_weighted,
#         "f1_per_class": f1_per_class,
#         "val_per_class_acc": per_class_acc,
#     }


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



# def evaluate_sae(
#     num_classes: int,
#     sae: SparseAutoencoder,
#     loader: DataLoader,
#     device: torch.device,
#     l1_lambda: float,
# ) -> Dict[str, float]:
#     sae.eval()
#     total_loss = 0.0
#     total_mse = 0.0
#     total_n = 0

#     with torch.no_grad():
#         for ss, _, y in loader:
#             ss = ss.to(device)
#             y = y.to(device)
#             _, recon = sae(ss)
#             mse = F.mse_loss(recon, ss)
#             loss = mse + l1_lambda * sae.sparse_penalty()
#             bsz = ss.size(0)
#             total_loss += loss.item() * bsz
#             total_mse += mse.item() * bsz
#             total_n += bsz
#     return {
#         "loss": total_loss / max(total_n, 1),
#         "mse": total_mse / max(total_n, 1),
#     }


# =========================================================
# End-to-end helper
# =========================================================


class SSPPTester:
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
        mu_ss: torch.Tensor,
        sigma_ss: torch.Tensor,
        mu_pp: torch.Tensor,
        sigma_pp: torch.Tensor,
        val_ss: torch.Tensor,
        val_pp: torch.Tensor,
        val_y: torch.Tensor,
    ):
        self.ss_scaler.mean = mu_ss
        self.ss_scaler.std = sigma_ss
        self.pp_scaler.mean = mu_pp
        self.pp_scaler.std = sigma_pp

        val_ds = (
            self.make_dataset(val_ss, val_pp, val_y) if val_ss is not None else None
        )

        fcn_val_loader = (
            DataLoader(val_ds, batch_size=self.config.fcn_batch_size, shuffle=False)
            if val_ds is not None
            else None
        )

        return (
            fcn_val_loader,
        )

    def test(
        self,
        classes: list[int],
        val_ss: torch.Tensor,
        val_pp: torch.Tensor,
        val_y: torch.Tensor,
        seed: Optional[int] = None,
        mlflow_config: Optional[Dict[str, str]] = None,
    ):
        if seed is not None:
            set_seed_to(seed)

        if mlflow_config is not None:
            mlflow.start_run(run_id=mlflow_config["run_id"])
            
            dst_path = Path("/tmp/testing")
            if dst_path.exists():
                if dst_path.is_file():
                    dst_path.unlink()
                else:
                    import shutil

                    shutil.rmtree(dst_path)
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{mlflow_config['run_id']}/normalize.npz",
                dst_path=str(dst_path),
            )
            normalize = np.load(dst_path / "normalize.npz")

        (
            fcn_val_loader,
        ) = self.build_loaders(normalize["mu_ss"], normalize["sigma_ss"], normalize["mu_pp"], normalize["sigma_pp"], val_ss, val_pp, val_y)


        self.model.sae = mlflow.pytorch.load_model(f"runs:/{mlflow_config['run_id']}/sae",map_location=self.device)
        self.model.cnn = mlflow.pytorch.load_model(f"runs:/{mlflow_config['run_id']}/cnn",map_location=self.device)
        self.model.fcn = mlflow.pytorch.load_model(f"runs:/{mlflow_config['run_id']}/fcn",map_location=self.device)
        self.model.sae.eval()
        self.model.cnn.eval()
        self.model.fcn.eval()
        self.model.eval()
        self.model.to(self.device)
        
        all_preds_list: list[npt.NDArray[np.int_]] = []
        with torch.no_grad():
            for ss, pp, y in fcn_val_loader:
                ss = ss.to(self.device)
                pp = pp.to(self.device)

                outputs = self.model(ss, pp)
                probs = nn.Softmax(dim=1)(outputs)
                preds = probs.argmax(dim=1)

                all_preds_list.extend(preds.cpu().numpy())
        all_preds = np.array(all_preds_list)
        # === 計算評估指標 ===
        val_y = val_y.cpu().numpy()
        accuracy = np.mean(all_preds == val_y)
        f1_macro = f1_score(val_y, all_preds, average="macro")
        f1_micro = f1_score(val_y, all_preds, average="micro")
        f1_weighted = f1_score(val_y, all_preds, average="weighted")
        percision_macro = precision_score(val_y, all_preds, average="macro")
        precision_micro = precision_score(val_y, all_preds, average="micro")
        precision_weighted = precision_score(val_y, all_preds, average="weighted")
        recall_macro = recall_score(val_y, all_preds, average="macro")
        recall_micro = recall_score(val_y, all_preds, average="micro")
        recall_weighted = recall_score(val_y, all_preds, average="weighted")
        logging.getLogger("run_eval").info(f"Accuracy: {accuracy:.4f}")
        logging.getLogger("run_eval").info(f"F1 Score (Macro): {f1_macro:.4f}")
        logging.getLogger("run_eval").info(f"F1 Score (Micro): {f1_micro:.4f}")
        logging.getLogger("run_eval").info(f"F1 Score (Weighted): {f1_weighted:.4f}")
        logging.getLogger("run_eval").info(f"Precision (Macro): {percision_macro:.4f}")
        logging.getLogger("run_eval").info(f"Precision (Micro): {precision_micro:.4f}")
        logging.getLogger("run_eval").info(f"Precision (Weighted): {precision_weighted:.4f}")
        logging.getLogger("run_eval").info(f"Recall (Macro): {recall_macro:.4f}")
        logging.getLogger("run_eval").info(f"Recall (Micro): {recall_micro:.4f}")
        logging.getLogger("run_eval").info(f"Recall (Weighted): {recall_weighted:.4f}")
        # 計算每個類別的評估指標
        table_str, metrics_list = per_class_metrics(
            test_label=val_y, all_preds=all_preds, class_names=classes
        )
        logging.getLogger("run_eval").info("每一個類別的評估指標：\n" + table_str)
        if mlflow_config is not None:
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_f1_macro", float(f1_macro))
            mlflow.log_metric("test_f1_micro", float(f1_micro))
            mlflow.log_metric("test_f1_weighted", float(f1_weighted))
            mlflow.log_metric("test_precision_macro", float(percision_macro))
            mlflow.log_metric("test_precision_micro", float(precision_micro))
            mlflow.log_metric("test_precision_weighted", float(precision_weighted))
            mlflow.log_metric("test_recall_macro", float(recall_macro))
            mlflow.log_metric("test_recall_micro", float(recall_micro))
            mlflow.log_metric("test_recall_weighted", float(recall_weighted))
            log_json_artifact(cast(JSONObject, metrics_list), "test_per_class_metrics.json")

        return self.model