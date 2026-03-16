from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch

T = TypeVar("T", npt.NDArray[np.float64], torch.Tensor)


def fit_normalizer(X: T, log_idx: list[int], z_only_idx: list[int]) -> tuple[T, T]:
    """
    訓練階段：計算 log 後的 μ 和 σ
    Arguments:
        X: np.array, shape = (num_samples, num_features)
        log_idx: list of int, 需要做 log 的欄位 index
        z_only_idx: list of int, 只做 z-score 的欄位 index
    Returns:
        mu: np.array, shape = (num_zscore_features,)
        sigma: np.array, shape = (num_zscore_features,)
    """
    if isinstance(X, torch.Tensor):
        X_log = X.cpu().numpy().copy().astype(float)  # 轉成 numpy array 來計算 μ 和 σ
    elif isinstance(X, np.ndarray):
        X_log = X.copy().astype(float)
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")

    # 把 inf 變 nan，避免 log(∞) 變成 inf
    X_log[:, log_idx] = np.where(np.isinf(X_log[:, log_idx]), np.nan, X_log[:, log_idx])
    # 把 nan視為 0
    X_log[:, log_idx] = np.where(np.isnan(X_log[:, log_idx]), 0, X_log[:, log_idx])
    X_log[:, z_only_idx] = np.where(
        np.isnan(X_log[:, z_only_idx]), 0, X_log[:, z_only_idx]
    )
    # 避免 log(負數)
    X_log[:, log_idx] = np.clip(X_log[:, log_idx], a_min=0, a_max=None)
    X_log[:, z_only_idx] = np.clip(X_log[:, z_only_idx], a_min=0, a_max=None)

    # logarithm 區段
    X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    # 要做 z-score 的欄位
    z_cols = log_idx + z_only_idx

    # 計算 μ 和 σ
    mu = X_log[:, z_cols].mean(axis=0)
    sigma = X_log[:, z_cols].std(axis=0) + 1e-12  # 防止除以 0

    if isinstance(X, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float32)
        sigma = torch.tensor(sigma, dtype=torch.float32)

    return mu, sigma


def transform_normalizer(
    X: T,
    mu: T,
    sigma: T,
    log_idx: list[int],
    z_only_idx: list[int],
) -> T:
    """
    推論階段（包括 validation / test）：只能 transform，不可重新 fit!
    Arguments:
        X: np.array, shape = (num_samples, num_features)
        mu: np.array, shape = (num_zscore_features,)
        sigma: np.array, shape = (num_zscore_features,)
        log_idx: list of int, 需要做 log 的欄位 index
        z_only_idx: list of int, 只做 z-score 的欄位 index
    Returns:
        X_new: np.array, shape = (num_samples, num_features), 正規化後的資料
    """
    X_temp: npt.NDArray[np.float64]
    if isinstance(X, torch.Tensor):
        X_temp = X.cpu().numpy().copy().astype(float)  # 轉成 numpy array 來做變換
    elif isinstance(X, np.ndarray):
        X_temp = X.copy().astype(float)
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")

    # 把 inf 變 nan，避免 log(∞) 變成 inf
    X_temp[:, log_idx] = np.where(
        np.isinf(X_temp[:, log_idx]), np.nan, X_temp[:, log_idx]
    )
    # 把 nan視為 0
    X_temp[:, log_idx] = np.where(np.isnan(X_temp[:, log_idx]), 0, X_temp[:, log_idx])
    # 避免 log(負數)
    X_temp[:, log_idx] = np.clip(X_temp[:, log_idx], a_min=0, a_max=None)

    # 1. 做 log1p
    X_temp[:, log_idx] = np.log1p(X_temp[:, log_idx])

    # 2. z-score
    z_cols = log_idx + z_only_idx
    X_temp[:, z_cols] = (X_temp[:, z_cols] - mu) / sigma

    X_new: T
    if isinstance(X, torch.Tensor):
        X_new = torch.from_numpy(X_temp).type(torch.float64)
    elif isinstance(X, np.ndarray):
        X_new = X_temp.astype(np.float64)
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")

    return X_new
