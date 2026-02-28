import os
import datetime
import logging
from pathlib import Path
from core.training_adapter_token_vit import run_adapter_token_vit_training_with_mlflow
from utils.alias import a2p


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

    run_adapter_token_vit_training_with_mlflow(
        experiment_name="TON_IoT",
        run_name="Adapter Token ViT1D Multi-class Classification RSA Encrypted Rsamples 70%",
        config={
            "model_name": "adapter_token_vit",
            "seed": 42,
            "batch_size": 8192,
            "gamma": 0.01,
            "lr": 0.01,
            "epochs": 100,
            "patch_size": 16,
            "rl_step": 15,
            "seq_len": 480,
            "task": "multi-class classification",
        },
        features=[
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
        ],
        dataset_path=a2p("@/data/TON_IoT/features_rsa/sampled_70P/train/sampled_data.npy"),
    )
    # features_nonMalucusIsBenign
    # features_fixed_time
