import os
import datetime
import logging
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split
from core.training_adapter_token_vit import (
    run_adapter_token_vit_training,
    run_adapter_token_vit_training_with_mlflow,
)
from utils.alias import a2p
from data_processing.FlowMeter.extract_flow_features_103 import get_feature_names_103
from utils.hash import sha256_file
from utils.normalizer import fit_normalizer, transform_normalizer
import numpy as np
import json
import torch
from model.sspp import SSPPConfig
from core.training_sspp import SSPPTrainer
from tabulate import tabulate
import time

if __name__ == "__main__":
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

    # === 讀取資料 ===
    dataset_path = a2p(
        "@/data_inner/TON_IoT/features_sspp/sampled_0P/train/sampled_data.npy"
    )
    train_path = dataset_path.parent
    data = np.load(train_path / "sampled_data.npy")
    label = np.load(train_path / "sampled_label.npy")

    with open(train_path / "labeled.json", "r") as f:
        sampled_info = json.load(f)
        classes = sampled_info["classes"]

    result_folder_path: Path = (
        train_path.parent.parent.parent
        / "result"
        / "sspp_statistical_multi"
        / str(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S"))
    )
    result_folder_path.mkdir(parents=True, exist_ok=True)
    (result_folder_path / "pt").mkdir(parents=True, exist_ok=True)

    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.1, stratify=label, random_state=42
    )  # 分訓練/驗證
    logging.getLogger("run_training").info("-" * 25 + " train data " + "-" * 25)
    table = []
    for i in set(train_label):
        table.append([i, classes[i], np.sum(train_label == i)])
    logging.getLogger("run_training").info(
        "\n" + tabulate(table, headers=["Id", "Class Name", "Count"])
    )

    logging.getLogger("run_training").info("-" * 25 + " valid data " + "-" * 25)
    table = []
    for i in set(val_label):
        table.append([i, classes[i], np.sum(val_label == i)])
    logging.getLogger("run_training").info(
        "\n" + tabulate(table, headers=["Id", "Class Name", "Count"])
    )

    # Example dummy data
    num_train = 1000
    num_val = 200
    N = 8
    num_classes = len(classes)

    feature_names_103 = get_feature_names_103()
    features_26 = [
        "Packet Length Mean",
        "Packet Length Variance",
        "Min Packet Length",
        "Total Packet Length",
        "L4 Length Mean",
        "L4 Length Variance",
        "L4 Length Min",
        "L4 Length Max",
        "Total L4 Length",
        "Flow IAT Mean",
        "Flow IAT Min",
        "Flow IAT Max",
        "Flow IAT Total",
        "Handshake Time",
        "Flow Duration",
        "Win Mean",
        "Win Variance",
        "Win Min",
        "Win Max",
        "Win Total",
        "RST Flag Count",
        "PSH Flag Count",
        "Keep Alive Count",
        "Keep Alive ACK Count",
        "SYN Flag Count",
        "Service",
    ]
    features_26_index = [feature_names_103.index(f) for f in features_26]
    features_26_index.sort()
    train_ss = train_data[:, features_26_index]
    train_pp = train_data[:, 103 : 103 + N * 25].reshape(-1, N, 25)
    train_y = train_label

    val_ss = val_data[:, features_26_index]
    val_pp = val_data[:, 103 : 103 + N * 25].reshape(-1, N, 25)
    val_y = val_label

    config = SSPPConfig(
        seed=42,
        num_classes=num_classes,
        num_packets=N,
        ss_dim=26,
        pp_dim=25,
    )

    trainer = SSPPTrainer(config)
    t0 = time.time()
    model = trainer.train(
        train_ss=torch.from_numpy(train_ss).type(torch.float32),
        train_pp=torch.from_numpy(train_pp).type(torch.float32),
        train_y=torch.from_numpy(train_y).type(torch.long),
        val_ss=torch.from_numpy(val_ss).type(torch.float32),
        val_pp=torch.from_numpy(val_pp).type(torch.float32),
        val_y=torch.from_numpy(val_y).type(torch.long),
        mlflow_config={
            "experiment_name": "TON_IoT",
            "run_name": f"SSPP Multi-class Classification SSPP26Features",
        },
    )
    mlflow.log_metric("train_wall_time_sec", time.time() - t0)

    ds_hash = sha256_file(dataset_path)
    mlflow.log_param("dataset_path", str(dataset_path))
    mlflow.log_param("dataset_sha256", ds_hash)
    
    mlflow.set_tag("task", "multi-class classification")
    mlflow.set_tag("framework", "pytorch")
    mlflow.set_tag("model", "SSPP")
    
    mlflow.end_run()
    # preds = trainer.predict(
    #     torch.from_numpy(val_ss[:8]).type(torch.float32),
    #     torch.from_numpy(val_pp[:8]).type(torch.float32),
    # )
    # print("preds:", preds.tolist())
    # print("true:", val_y[:8].tolist())
