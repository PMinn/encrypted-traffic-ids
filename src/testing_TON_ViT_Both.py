from typing import Callable, cast
import os
import datetime
import json
from pathlib import Path
import numpy as np
import numpy.typing as npt
from tabulate import tabulate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from data_processing.FlowMeter.extract_flow_features_73 import get_feature_names_73
from utils.alias import a2p
from utils.normalizer import transform_normalizer
from utils.set_seed import set_seed_to
from sklearn.metrics import precision_recall_fscore_support


def per_class_metrics(
    test_label: npt.NDArray[np.int_],
    all_preds: npt.NDArray[np.int_],
    class_names: list[int],
) -> tuple[str, list[dict[str, float]]]:
    """
    test_label: 1D array-like, 真實標籤
    all_preds : 1D array-like, 模型預測標籤
    class_names: list[str] 或 None
        - 若為 None，預設用 "0", "1", "2", ... 當類別名稱

    回傳: (table_str, metrics_dict_list)
        - table_str: 已排版好的表格字串（可直接 print 或丟進 logger）
        - metrics_dict_list: 每一列是 dict，含各類別指標
    """
    # 決定類別數與類別名稱
    num_classes = len(class_names)

    # 使用 sklearn 一次算出每個類別的 precision / recall / f1
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        test_label, all_preds, labels=range(num_classes)
    )

    if type(recalls) is float:
        raise ValueError(
            "Only one class present in y_true. metrics_dict_list will be empty."
        )
    # 這裡把 per-class accuracy 定義為「在該類別上的正確率 = TP / (TP + FN)」
    # 也就是跟 recall 相同，只是改個名字，方便你對照原本的寫法
    accuracies_list = cast(npt.NDArray[np.float64], recalls).copy()
    f1s_list = cast(npt.NDArray[np.float64], f1s).copy()
    precisions_list = cast(npt.NDArray[np.float64], precisions).copy()
    recalls_list = cast(npt.NDArray[np.float64], recalls).copy()
    supports_list = cast(npt.NDArray[np.int_], supports).copy()

    # 組成 table
    headers = [
        "Class",
        "Accuracy(per-class)",
        "F1 Score",
        "Precision",
        "Recall",
        "Support",
    ]
    table_data = []
    metrics_list = []

    for idx, name in enumerate(class_names):
        row = [
            name,
            accuracies_list[idx],
            f1s_list[idx],
            precisions_list[idx],
            recalls_list[idx],
            supports_list[idx],
        ]
        table_data.append(row)

        metrics_list.append(
            {
                "class": name,
                "accuracy": float(accuracies_list[idx]),
                "f1": float(f1s_list[idx]),
                "precision": float(precisions_list[idx]),
                "recall": float(recalls_list[idx]),
                "support": int(supports_list[idx]),
            }
        )

    table_str = tabulate(table_data, headers=headers)

    return table_str, metrics_list


def run_eval() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 超參數設定 ===
    GAMMA = 0.01
    INIT_LR = 0.01
    EPOCH = 100
    BATCH_SIZE = 8192
    PATCH_SIZE = 12
    RL_STEP = 15
    SEQ_LEN = 480

    with open(a2p("@/data/TON_IoT/features/sampled_7/test/classes.json"), "r") as f:
        classes = json.load(f)

    result_folder_path = a2p("@/data/TON_IoT/result/both_multi/20251210-002841")

    # === 讀取資料 ===
    train_path = a2p("@/data/TON_IoT/features/sampled_7/test")
    data = np.load(f"{train_path}/sampled_data.npy")
    label = np.load(f"{train_path}/sampled_label.npy")

    test_label = label

    features_name_73 = get_feature_names_73()

    keep_features_name = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Destination Port",
        "Source Port",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Max Packet Length",
        "Min Packet Length",
        "Packet Length Std",
        "SYN Flag Count",
        "ACK Flag Count",
        "Protocol",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Std",
        "Bwd IAT Max",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
    ]
    keep_features_index = [features_name_73.index(feat) for feat in keep_features_name]
    if -1 in keep_features_index:
        missing_feats = [
            keep_features_name[i]
            for i, idx in enumerate(keep_features_index)
            if idx == -1
        ]
        logger.error(
            f"Some required features are missing in features_name_73: {missing_feats}"
        )
        raise ValueError(
            f"Some required features are missing in features_name_73: {missing_feats}"
        )
    keep_features_index.sort()
    keep_features_name = [features_name_73[idx] for idx in keep_features_index]

    log_scale_features_name = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Max Packet Length",
        "Min Packet Length",
        "Packet Length Std",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Std",
        "Bwd IAT Max",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
    ]
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

    std_scale_features_name = [
        "SYN Flag Count",
        "ACK Flag Count",
        "Protocol",
        "Destination Port",
        "Source Port",
    ]
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

    normalize = np.load(result_folder_path / "normalize.npz")
    test_flow_features = data[:, keep_features_index]
    test_flow_features = transform_normalizer(
        test_flow_features, normalize["mu"], normalize["sigma"], log_idx, z_only_idx
    )
    test_data = data[:, 73:]
    test_data = test_data / 255

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data, dtype=torch.float),
            torch.tensor(test_flow_features, dtype=torch.float),
            torch.tensor(test_label, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
    )

    model = torch.load(
        result_folder_path / "pth" / "model-best_val_acc.pth", weights_only=False
    ).to(device)

    # === 評估模型 ===
    model.eval()
    all_preds_list: list[npt.NDArray[np.int_]] = []
    with torch.no_grad():
        for batch_data, batch_flow_features, _ in test_loader:
            batch_data = batch_data.unsqueeze(1).to(device)
            batch_flow_features = batch_flow_features.to(device)

            outputs, _ = model(batch_data, batch_flow_features)
            probs = nn.Softmax(dim=1)(outputs)
            _, preds = torch.max(probs, 1)

            all_preds_list.extend(preds.cpu().numpy())
    all_preds = np.array(all_preds_list)
    # === 計算評估指標 ===
    accuracy = np.mean(all_preds == test_label)
    f1 = f1_score(test_label, all_preds, average="weighted")
    precision = precision_score(test_label, all_preds, average="weighted")
    recall = recall_score(test_label, all_preds, average="weighted")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    # 計算每個類別的評估指標
    table_str, metrics_list = per_class_metrics(
        test_label=test_label, all_preds=all_preds, class_names=classes
    )
    logger.info("每一個類別的評估指標：\n" + table_str)
    with open(result_folder_path / "testing_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {accuracy}\n")
        f.write(f"Overall F1 Score: {f1}\n")
        f.write(f"Overall Precision: {precision}\n")
        f.write(f"Overall Recall: {recall}\n\n")
        f.write("Class-wise Evaluation Metrics:\n")
        f.write(table_str)


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

    run_eval()
