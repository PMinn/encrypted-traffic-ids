from dataclasses import dataclass
import hashlib
from typing import Any, Callable, cast, Sized, TypedDict
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
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from data_processing.FlowMeter.extract_flow_features_73 import get_feature_names_73
from model.Adapter_Token_ViT_1D import Adapter_Token_ViT_1D
from utils.early_stopping import EarlyStopping
from utils.alias import a2p
from utils.set_seed import set_seed_to
from utils.normalizer import fit_normalizer, transform_normalizer


# ====== 工具：快取避免重複訓練同一組特徵 ======
class MetaDict(TypedDict):
    n_features: int
    features: list[str]
    per_class_acc: dict[int, float]
    per_class_f1: dict[int, float]


@dataclass
class EvalResult:
    f1_macro: float
    epoch: int
    val_acc: float
    meta: MetaDict


class FeatureSearch:
    def __init__(
        self,
        evaluator: Callable[
            [
                list[str],
                list[str],
                list[str],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.int_],
                npt.NDArray[np.int_],
            ],
            tuple[int, float, float, dict[int, float], dict[int, float]],
        ],
        train_data: npt.NDArray[np.float64],
        val_data: npt.NDArray[np.float64],
        train_label: npt.NDArray[np.int_],
        val_label: npt.NDArray[np.int_],
        cache_dir: str | Path = "fs_cache",
    ):
        self.evaluator = evaluator
        self.train_data = train_data
        self.val_data = val_data
        self.train_label = train_label
        self.val_label = val_label
        if isinstance(cache_dir, str):
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, feats: list[str]) -> str:
        # 用排序確保同集合不因順序不同而重跑
        s = json.dumps(sorted(feats), ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def eval(
        self, feats: list[str], log_scale_feats: list[str], std_scale_feats: list[str]
    ) -> EvalResult:
        k = self._key(feats)
        fp = self.cache_dir / f"{k}.json"
        if fp.exists():
            d = json.loads(fp.read_text(encoding="utf-8"))
            return EvalResult(**d)

        (
            best_epoch,
            best_val_acc,
            best_f1_macro,
            best_val_per_class_acc,
            best_f1_per_class,
        ) = self.evaluator(
            feats,
            log_scale_feats,
            std_scale_feats,
            self.train_data,
            self.val_data,
            self.train_label,
            self.val_label,
        )
        out = EvalResult(
            f1_macro=float(best_f1_macro),
            epoch=int(best_epoch),
            val_acc=float(best_val_acc),
            meta={
                "n_features": len(feats),
                "features": sorted(feats),
                "per_class_acc": best_val_per_class_acc,
                "per_class_f1": best_f1_per_class,
            },
        )
        fp.write_text(json.dumps(out.__dict__, ensure_ascii=False), encoding="utf-8")
        return out


def get_log_scale_features_name() -> list[str]:
    return [
        "Flow Duration",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "Subflow Fwd Bytes",
        "Subflow Bwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ]


def get_std_scale_features_name() -> list[str]:
    return [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Average Packet Size",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
        "Fwd Header Length",
        "Bwd Header Length",
        "Down/Up Ratio",
        "Min_seg_size_forward",
    ]


def run_training(
    keep_features_name: list[str],
    log_scale_features_name: list[str],
    std_scale_features_name: list[str],
    train_data: npt.NDArray[np.float64],
    val_data: npt.NDArray[np.float64],
    train_label: npt.NDArray[np.int_],
    val_label: npt.NDArray[np.int_],
) -> tuple[int, float, float, dict[int, float], dict[int, float]]:
    # === 超參數設定 ===
    GAMMA = 0.01
    INIT_LR = 0.01
    EPOCH = 100
    BATCH_SIZE = 8192
    PATCH_SIZE = 12
    RL_STEP = 15
    SEQ_LEN = 480

    with open(a2p("@/data/CIC-IDS-2017/features/sampled/train/classes.json"), "r") as f:
        classes = json.load(f)

    # result_folder_path = a2p(
    #     "@/data/CIC-IDS-2017/result/both_multi/"
    #     + datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S")
    # )
    # result_folder_path.mkdir(parents=True, exist_ok=True)
    # (result_folder_path / "pth").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_name_73 = get_feature_names_73()

    keep_features_index = [features_name_73.index(feat) for feat in keep_features_name]
    if -1 in keep_features_index:
        missing_feats = [
            keep_features_name[i]
            for i, idx in enumerate(keep_features_index)
            if idx == -1
        ]
        logging.getLogger("run_training").error(
            f"Some required features are missing in features_name_73: {missing_feats}"
        )
        raise ValueError(
            f"Some required features are missing in features_name_73: {missing_feats}"
        )
    keep_features_index.sort()
    keep_features_name = [features_name_73[idx] for idx in keep_features_index]

    log_idx = [keep_features_name.index(feat) for feat in log_scale_features_name]
    if -1 in log_idx:
        missing_feats = [
            log_scale_features_name[i] for i, idx in enumerate(log_idx) if idx == -1
        ]
        logging.getLogger("run_training").error(
            f"Some log-scale features are missing in keep_features_name: {missing_feats}"
        )
        raise ValueError(
            f"Some log-scale features are missing in keep_features_name: {missing_feats}"
        )

    z_only_idx = [keep_features_name.index(feat) for feat in std_scale_features_name]
    if -1 in z_only_idx:
        missing_feats = [
            std_scale_features_name[i] for i, idx in enumerate(z_only_idx) if idx == -1
        ]
        logging.getLogger("run_training").error(
            f"Some std-scale features are missing in keep_features_name: {missing_feats}"
        )
        raise ValueError(
            f"Some std-scale features are missing in keep_features_name: {missing_feats}"
        )

    train_flow_features = train_data[:, keep_features_index]
    mu, sigma = fit_normalizer(train_flow_features, log_idx, z_only_idx)
    train_flow_features = transform_normalizer(
        train_flow_features, mu, sigma, log_idx, z_only_idx
    )
    train_data = train_data[:, 73:]
    train_data = train_data / 255

    val_flow_features = val_data[:, keep_features_index]
    val_flow_features = transform_normalizer(
        val_flow_features, mu, sigma, log_idx, z_only_idx
    )
    val_data = val_data[:, 73:]
    val_data = val_data / 255

    # np.savez(result_folder_path / "normalize.npz", mu=mu, sigma=sigma)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data, dtype=torch.float),
            torch.tensor(train_flow_features, dtype=torch.float),
            torch.tensor(train_label, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_data, dtype=torch.float),
            torch.tensor(val_flow_features, dtype=torch.float),
            torch.tensor(val_label, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
    )

    # === 建立模型 ===
    if SEQ_LEN % PATCH_SIZE != 0:
        logging.getLogger("run_training").error(
            f"Sequence length must be divisible by patch size. Got SEQ_LEN={SEQ_LEN}, PATCH_SIZE={PATCH_SIZE}"
        )
        raise ValueError(
            f"Sequence length must be divisible by patch size. Got SEQ_LEN={SEQ_LEN}, PATCH_SIZE={PATCH_SIZE}"
        )
    if SEQ_LEN != train_data.shape[1]:
        logging.getLogger("run_training").error(
            f"Sequence length must match the input data length. Expected {SEQ_LEN}, got {train_data.shape[1]}"
        )
        raise ValueError(
            f"Sequence length must match the input data length. Expected {SEQ_LEN}, got {train_data.shape[1]}"
        )

    model = Adapter_Token_ViT_1D(
        seq_len=SEQ_LEN,
        patch_size=PATCH_SIZE,
        num_classes=len(classes),
        dim=16,
        depth=6,
        heads=8,
        mlp_dim=32,
        flow_feat_dim=train_flow_features.shape[1],  # flow 統計特徵維度
    ).to(device)

    if torch.cuda.is_available():
        logging.getLogger("run_training").info(torch.cuda.get_device_name(0))
    else:
        logging.getLogger("run_training").info("Using CPU")
    # 多卡（如果有）
    # if torch.cuda.device_count() > 1:
    #     logging.getLogger("run_training").info(f"Use {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    # === Loss / Optimizer ===
    train_loss_curve = []
    val_loss_curve = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = StepLR(optimizer, step_size=RL_STEP, gamma=GAMMA)
    early_stopping = EarlyStopping(
        patience=10,
        delta=0,
        verbose=True,
        # path=result_folder_path / "pth" / "model-best_val_acc.pth",
    )

    # === 開始訓練 ===
    best_epoch = 0
    best_val_acc = 0.0
    best_f1_macro = 0.0
    best_val_per_class_acc = None
    best_f1_per_class = None
    for epoch in range(1, EPOCH + 1):
        # localtime = time.asctime(time.localtime(time.time()))

        # logging.getLogger("run_training").info(
        #     "-"
        #     * len(
        #         "Epoch: {}/{} --- < Starting Time : {} >".format(
        #             epoch, EPOCH, localtime
        #         )
        #     )
        # )
        # logging.getLogger("run_training").info(
        #     "Epoch: {}/{} --- < Starting Time : {} >".format(epoch, EPOCH, localtime)
        # )

        train_loss, train_acc, train_per_class_acc = train_one_epoch(
            model, train_loader, len(classes), criterion, optimizer, device
        )
        train_loss_curve.append(train_loss)

        # with open(result_folder_path / "train_Acc_Loss.txt", "a") as file:
        #     file.write(f"Epoch: {epoch}/{EPOCH} | Starting Time : {localtime}\n")
        #     file.write(f"Accuracy: {(train_acc * 100):.4f}% | Loss: {train_loss:.6f}\n")
        #     table = []
        #     for i, c in enumerate(classes):
        #         table.append([i, c, train_per_class_acc[i] * 100])
        #     file.write(tabulate(table, headers=["#", "Class", "Accuracy"]) + "\n\n")

        val_loss, val_acc, val_per_class_acc, f1_macro, f1_per_class = evaluate(
            model, val_loader, len(classes), criterion, device
        )
        val_loss_curve.append(val_loss)

        # with open(result_folder_path / "valid_Acc_Loss.txt", "a") as file:
        #     file.write(f"Epoch: {epoch}/{EPOCH} | Starting Time : {localtime}\n")
        #     file.write(f"Accuracy: {(val_acc * 100):.4f}% | Loss: {val_loss:.6f}\n")
        #     table = []
        #     for i, c in enumerate(classes):
        #         table.append([i, c, val_per_class_acc[i] * 100, f1_per_class[i]])
        #     file.write(
        #         tabulate(table, headers=["#", "Class", "Accuracy", "F1 Score"]) + "\n\n"
        #     )

        #     logging.getLogger("run_training").info(
        #         f"Valid Accuracy: {(val_acc * 100):.4f}% | loss: {val_loss:.6f}| F1 Score: {f1_macro:.6}"
        #     )
        #     logging.getLogger("run_training").info(
        #         "\n" + tabulate(table, headers=["#", "Class", "Accuracy", "F1 Score"])
        #     )

        stop, improved = early_stopping.check(f1_macro)
        if stop:
            # logging.getLogger("run_training").info(
            #     "Early stopping at epoch {}".format(epoch)
            # )
            break

        if improved:
            best_val_acc = val_acc
            best_f1_macro = f1_macro
            best_val_per_class_acc = val_per_class_acc
            best_f1_per_class = f1_per_class
            best_epoch = epoch

        scheduler.step()  # 更新學習率
        # logging.getLogger("run_training").info(
        #     f"learning rate: {scheduler.get_last_lr()}"
        # )

        # if (epoch - 1) % 5 == 0:
        #     torch.save(
        #         model,
        #         "{}/pth/model-{:.2f}-val_acc-{}-epoch.pth".format(
        #             result_folder_path, val_acc, epoch
        #         ),
        #     )

    if best_val_per_class_acc is None or best_f1_per_class is None:
        raise ValueError("best_val_per_class_acc or best_f1_per_class is None")

    # with open(result_folder_path / "Best_valid_Acc.txt", "w") as file:
    #     file.write(
    #         f"---------------------------- Best Validation Accuracy -------------------\n"
    #     )
    #     file.write(f"Epoch: {best_epoch}\n")
    #     file.write(f"acc: {best_val_acc}\n")
    #     file.write(f"F1: {best_f1_macro}\n")
    #     table = []
    #     for i, c in enumerate(classes):
    #         table.append([i, c, best_val_per_class_acc[i] * 100, best_f1_per_class[i]])
    #     file.write(
    #         tabulate(
    #             table, headers=["#", "Class", "Accuracy", "F1 Score"], floatfmt=".6f"
    #         )
    #         + "\n\n"
    #     )

    # plt.plot(
    #     range(1, len(train_loss_curve) + 1),
    #     train_loss_curve,
    #     label="Training Loss",
    #     color="blue",
    # )
    # plt.plot(
    #     range(1, len(val_loss_curve) + 1),
    #     val_loss_curve,
    #     label="Validation Loss",
    #     color="orange",
    # )
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("ViT Overfitting Check")
    # plt.legend()
    # plt.savefig(f"{result_folder_path}/loss_curve.png")
    # plt.show()
    # logging.getLogger("run_training").info(
    #     f"Loss curve saved to {result_folder_path}/loss_curve.png"
    # )

    # logging.getLogger("run_training").info(
    #     f"Best model save to: {result_folder_path}/pth/model-best_val_acc.pth"
    # )
    # parameter_total = sum([param.nelement() for param in model.parameters()])
    # logging.getLogger("run_training").info(
    #     "Number of parameter: %.2fM" % (parameter_total / 1e6)
    # )
    return (
        best_epoch,
        best_val_acc,
        best_f1_macro,
        best_val_per_class_acc,
        best_f1_per_class,
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[Any],
    num_classes: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """
    訓練一個 epoch
    Arguments:
        model: torch.nn.Module, 要訓練的模型
        dataloader: torch.utils.data.DataLoader, 訓練資料的 dataloader
        num_classes: int, 類別數
        criterion: 損失函數
        optimizer: 優化器
        device: torch.device, 運算裝置
    Returns:
        avg_loss: float, 平均損失
        acc: float, 準確率
        per_class_acc: dict, 每類別的準確率
    """
    model.train()
    total_loss = 0.0
    all_preds_list: list[npt.NDArray[np.int_]] = []
    all_labels_list: list[npt.NDArray[np.int_]] = []

    for seq, flow, labels in dataloader:
        seq = seq.to(device)  # [B, seq_len]
        seq = seq.unsqueeze(1)  # 增加一個新的維度來表示通道 [B, 1, seq_len]
        flow = flow.to(device)  # [B, F]
        labels = labels.to(device)  # [B]

        optimizer.zero_grad()  # 清除之前的梯度
        logits, cls_token_features = model(seq, flow)  # logits: [B, num_classes]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds_list.append(preds.detach().cpu().numpy())  # 將預測結果存到 CPU 上
        all_labels_list.append(labels.detach().cpu().numpy())  # 將標籤存到 CPU 上

    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)

    avg_loss: float = total_loss / len(cast(Sized, dataloader.dataset))
    acc = float(accuracy_score(all_labels, all_preds))

    # ====== 每類別 accuracy ======
    per_class_acc: dict[int, float] = {}
    for cls in range(num_classes):
        mask = all_labels == cls
        total = np.sum(mask)
        correct = np.sum(all_preds[mask] == all_labels[mask])
        per_class_acc[cls] = float(correct / total)

    return avg_loss, acc, per_class_acc


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    num_classes: int,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[int, float], float, dict[int, float]]:
    """
    評估模型
    Arguments:
        model: torch.nn.Module, 要評估的模型
        dataloader: torch.utils.data.DataLoader, 評估資料的 dataloader
        num_classes: int, 類別數
        criterion: 損失函數
        device: torch.device, 運算裝置
    Returns:
        avg_loss: float, 平均損失
        acc: float, 準確率
        per_class_acc: dict[int, float], 每類別的準確率
        f1_macro: float, macro F1-score
        f1_per_class: dict[int, float], 每類別的 F1-score
    """
    model.eval()
    total_loss = 0.0
    all_preds_list: list[npt.NDArray[np.int_]] = []
    all_labels_list: list[npt.NDArray[np.int_]] = []

    with torch.no_grad():
        for seq, flow, labels in dataloader:
            seq = seq.to(device)
            seq = seq.unsqueeze(1)  # 增加一個新的維度來表示通道 [B, 1, seq_len]
            flow = flow.to(device)
            labels = labels.to(device)

            logits, _ = model(seq, flow)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds_list.append(preds.detach().cpu().numpy())
            all_labels_list.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    avg_loss: float = total_loss / len(cast(Sized, dataloader.dataset))
    acc = float(accuracy_score(all_labels, all_preds))

    # ====== 每類別 accuracy ======
    per_class_acc = {}
    for cls in range(num_classes):
        mask = all_labels == cls
        total = np.sum(mask)
        correct = np.sum(all_preds[mask] == all_labels[mask])
        per_class_acc[cls] = float(correct / total)

    # ====== F1-score ======
    # macro: 對每類別算 F1 再平均（不看類別比例）
    f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    # per-class F1：一個類別一個值，順序是 class 0,1,...,num_classes-1
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

    return avg_loss, acc, per_class_acc, f1_macro, f1_per_class


def default_beam_schedule(
    step: int, max_features: int, w_max: int = 10, w_min: int = 2
) -> int:
    """
    step: 從 1 到 max_features
    前大後小，線性衰減。你也可以換成指數或分段。
    """
    if max_features <= 1:
        return w_min
    t = (step - 1) / (max_features - 1)  # 0 -> 1
    w = round(w_max * (1 - t) + w_min * t)
    return max(w_min, min(w_max, w))


def beam_search(
    fs: FeatureSearch,
    all_features: list[str],
    beam_width_max: int = 10,
    beam_width_min: int = 2,
    patience: int = 3,
    max_features: int = 25,
    min_improve: float = 1e-4,
) -> EvalResult:
    beam: list[list[str]] = [[]]
    best_overall = EvalResult(
        f1_macro=-1.0,
        epoch=-1,
        val_acc=-1.0,
        meta={"features": [], "n_features": 0, "per_class_acc": {}, "per_class_f1": {}},
    )
    no_improve = 0

    for step in range(1, max_features + 1):
        bw = default_beam_schedule(
            step, max_features, w_max=beam_width_max, w_min=beam_width_min
        )
        candidates: list[tuple[float, list[str], EvalResult]] = []
        for subset in beam:
            remaining = [f for f in all_features if f not in subset]
            for f in remaining:
                cand = subset + [f]
                log_scale_cand = [
                    feat for feat in cand if feat in get_log_scale_features_name()
                ]
                std_scale_cand = [
                    feat for feat in cand if feat in get_std_scale_features_name()
                ]
                res = fs.eval(cand, log_scale_cand, std_scale_cand)
                candidates.append((res.f1_macro, cand, res))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:bw]

        beam = [c[1] for c in top]
        step_best = top[0][2]

        improved = step_best.f1_macro - best_overall.f1_macro
        logging.getLogger("beam_search").info(
            f"[ADAPT-BEAM step {step:02d}] beam_width={bw:02d} "
            f"step_best_f1={step_best.f1_macro:.6f} best_f1={best_overall.f1_macro:.6f} "
            f"improve={improved:.6f} k={len(step_best.meta['features'])}"
        )

        if improved > min_improve:
            best_overall = step_best
            no_improve = 0
            logging.getLogger("beam_search").info(
                f"New best found! F1={best_overall.f1_macro:.6f} "
                f"features={best_overall.meta['features']}"
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.getLogger("beam_search").info(
                    f"Stop: no improvement for {patience} consecutive steps."
                )
                break

    return best_overall


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        a2p("@/logs")
        / f"{Path(__file__).stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    logger.getChild("matplotlib").setLevel(logging.WARNING)

    set_seed_to(seed=42)  # 設定隨機種子以確保可重現 (可選)

    all_features_name = get_feature_names_73()

    result_folder = a2p("@/data/CIC-IDS-2017/result/feature_select/")
    result_folder.mkdir(parents=True, exist_ok=True)

    # === 讀取資料 ===
    train_path = a2p("@/data/CIC-IDS-2017/features/sampled/train")
    data = np.load(f"{train_path}/sampled_data.npy")
    label = np.load(f"{train_path}/sampled_label.npy")

    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.1, stratify=label, random_state=42
    )  # 分訓練/驗證

    fs = FeatureSearch(
        run_training,
        train_data,
        val_data,
        train_label,
        val_label,
        cache_dir=result_folder / "fs_cache",
    )
    best_beam = beam_search(fs, all_features_name, max_features=25, min_improve=0)
    result_folder.write_text(
        json.dumps(best_beam.__dict__, ensure_ascii=False, indent=4), encoding="utf-8"
    )
