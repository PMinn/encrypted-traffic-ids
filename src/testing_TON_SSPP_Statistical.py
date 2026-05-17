import os
import datetime
import logging
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split
from core.testing_sspp import SSPPTester
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
        "@/data_inner/TON_IoT/features_sspp/sampled_0P/test/sampled_data.npy"
    )
    train_path = dataset_path.parent
    data = np.load(train_path / "sampled_data.npy")
    label = np.load(train_path / "sampled_label.npy")

    with open(train_path / "labeled.json", "r") as f:
        sampled_info = json.load(f)
        classes = sampled_info["classes"]

    val_data = data
    val_label = label
  
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
        # "Service",
    ]
    features_26_index = [feature_names_103.index(f) for f in features_26]
    features_26_index.sort()

    val_ss = val_data[:, features_26_index]
    val_pp = val_data[:, 103 : 103 + N * 25].reshape(-1, N, 25)
    val_y = val_label

    config = SSPPConfig(
        seed=42,
        num_classes=num_classes,
        num_packets=N,
        ss_dim=25,
        # ss_dim=26,
        pp_dim=25,
    )

    tester = SSPPTester(config)
    model = tester.test(
        classes=classes,
        val_ss=torch.from_numpy(val_ss).type(torch.float64),
        val_pp=torch.from_numpy(val_pp).type(torch.float64),
        val_y=torch.from_numpy(val_y).type(torch.float64),
        mlflow_config={
            "run_id": "d4c7893bd9524e2ea4d20ff29a189081",
        },
    )
    mlflow.end_run()