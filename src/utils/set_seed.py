import os
import random
import numpy as np
import torch


def set_seed_to(seed: int = 42) -> None:
    # 鎖 CPU 與 PyTorch 自己的 RNG
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 鎖 GPU 上的 random op
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 鎖 dict / set 的 hash 順序
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 確保 cudnn 可重現
    # 禁用某些非 deterministic 的 conv / pooling 實作
    torch.backends.cudnn.deterministic = True
    # 不再自動搜尋「最快 kernel」
    torch.backends.cudnn.benchmark = False

    # （可選）PyTorch 2.0 後新增：強制執行 deterministic mode
    # 只要某個 op 不能 deterministic → 直接丟 error
    torch.use_deterministic_algorithms(True)

    # 避免一些 Op 的隨機性
    # 鎖定 cuBLAS（矩陣乘法）的 workspace
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
