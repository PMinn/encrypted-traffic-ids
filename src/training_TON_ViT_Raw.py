import os
import datetime
import logging
from pathlib import Path
from core.training_vit import run_vit_training_with_mlflow
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

    run_vit_training_with_mlflow(
        experiment_name="TON_IoT",
        run_name="Raw ViT Multi-class Classification Sampled 10%",
        config={
            "model_name": "vit",
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
        features=[],
        dataset_path=a2p("@/data/TON_IoT/features/sampled_10P/train/sampled_data.npy"),
    )
