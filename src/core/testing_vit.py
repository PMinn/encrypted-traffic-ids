from typing import Callable, TypedDict, cast
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
from data_processing.FlowMeter.extract_flow_features_73 import (
    get_feature_names_73,
    get_log_scale_features_name_73,
    get_std_scale_features_name_73,
)
from utils.alias import a2p
from utils.json import JSONObject
from utils.normalizer import transform_normalizer
from utils.save_log import log_json_artifact
from utils.set_seed import set_seed_to
from sklearn.metrics import precision_recall_fscore_support


class ConfigDict(TypedDict):
    model_name: str
    seed: int | None
    batch_size: int
    gamma: float
    lr: float
    epochs: int
    patch_size: int
    rl_step: int
    seq_len: int
    task: str


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


def run_eval(
    config: ConfigDict,
    dataset_path: Path,  # 可選：你的訓練資料檔
    model_path: Path | None = None,  # 可選：你的模型檔
    mlflow_run_id: str | None = None,
) -> None:
    # === 超參數設定 ===
    GAMMA = config["gamma"]
    INIT_LR = config["lr"]
    EPOCH = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    PATCH_SIZE = config["patch_size"]
    RL_STEP = config["rl_step"]
    SEQ_LEN = config["seq_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 讀取資料 ===
    test_path = dataset_path.parent
    data = np.load(test_path / "sampled_data.npy")
    label = np.load(test_path / "sampled_label.npy")

    with open(test_path / "labeled.json", "r") as f:
        labeled_info = json.load(f)
        classes = labeled_info["classes"]

    test_label = label
    test_data = data[:, 73:]
    test_data = test_data / 255

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(test_data, dtype=torch.float),
            torch.tensor(test_label, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE,
    )
    # === 載入模型 ===
    if model_path is not None:
        model = torch.load(model_path).to(device)
    elif mlflow_run_id is not None:
        import mlflow

        model = mlflow.pytorch.load_model(
            f"runs:/{mlflow_run_id}/model", map_location=device
        )
    else:
        raise ValueError("Either model_path or mlflow_run_id must be provided.")

    # === 評估模型 ===
    model.eval()
    all_preds_list: list[npt.NDArray[np.int_]] = []
    with torch.no_grad():
        for batch_data, _ in test_loader:
            batch_data = batch_data.unsqueeze(1).to(device)

            outputs, _ = model(batch_data)
            probs = nn.Softmax(dim=1)(outputs)
            _, preds = torch.max(probs, 1)

            all_preds_list.extend(preds.cpu().numpy())
    all_preds = np.array(all_preds_list)
    # === 計算評估指標 ===
    accuracy = np.mean(all_preds == test_label)
    f1 = f1_score(test_label, all_preds, average="weighted")
    precision = precision_score(test_label, all_preds, average="weighted")
    recall = recall_score(test_label, all_preds, average="weighted")
    logging.getLogger("run_eval").info(f"Accuracy: {accuracy:.4f}")
    logging.getLogger("run_eval").info(f"F1 Score: {f1:.4f}")
    logging.getLogger("run_eval").info(f"Precision: {precision:.4f}")
    logging.getLogger("run_eval").info(f"Recall: {recall:.4f}")
    # 計算每個類別的評估指標
    table_str, metrics_list = per_class_metrics(
        test_label=test_label, all_preds=all_preds, class_names=classes
    )
    logging.getLogger("run_eval").info("每一個類別的評估指標：\n" + table_str)
    if mlflow_run_id is not None:
        import mlflow

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", float(f1))
        mlflow.log_metric("test_precision", float(precision))
        mlflow.log_metric("test_recall", float(recall))
        log_json_artifact(cast(JSONObject, metrics_list), "test_per_class_metrics.json")


def run_vit_testing_with_mlflow(
    run_id: str,
    config: ConfigDict,
    dataset_path: Path,  # 可選：你的訓練資料檔
) -> None:
    """使用 MLflow 來記錄 Adapter Token ViT 測試過程與結果

    Args:
        experiment_name (str): MLflow 實驗名稱
        run_id (str): MLflow 執行名稱
        config (ConfigDict): 訓練設定參數
        features (list[str]): 使用的特徵名稱列表
        dataset_path (Path): 測試資料集路徑
    """
    import mlflow

    if config["seed"] is not None:
        set_seed_to(config["seed"])

    mlflow.start_run(run_id=run_id)
    run_eval(
        config=config,
        dataset_path=dataset_path,  # 可選：你的訓練資料檔
        mlflow_run_id=run_id,
    )

    # 這裡可以記錄更多的指標或產出物，例如混淆矩陣、模型檔案等
    # mlflow.log_artifact("path/to/confusion_matrix.png")
    # mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
