import os
import datetime
from pathlib import Path
import logging
from core.testing_vit import run_vit_testing_with_mlflow
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

    run_vit_testing_with_mlflow(
        run_id="f342432cad8740c0b4dc91cba43f350d",
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
        dataset_path=a2p("@/data/TON_IoT/features/sampled_70P/test/sampled_data.npy"),
    )
