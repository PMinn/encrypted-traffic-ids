import os
import random
import numpy as np
import torch


def set_seed_to(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 確保 cudnn 可重現
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # （可選）PyTorch 2.0 後新增：強制執行 deterministic mode
    torch.use_deterministic_algorithms(True)

    # 避免一些 Op 的隨機性
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
