import numpy as np
import numpy.typing as npt


def fit_normalizer(
    X: npt.NDArray[np.float64], log_idx: list[int], z_only_idx: list[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    X_log = X.copy().astype(float)

    # 避免 log(負數)
    X_log[:, log_idx] = np.clip(X_log[:, log_idx], a_min=0, a_max=None)

    # logarithm 區段
    X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    # 要做 z-score 的欄位
    z_cols = log_idx + z_only_idx

    # 計算 μ 和 σ
    mu = X_log[:, z_cols].mean(axis=0)
    sigma = X_log[:, z_cols].std(axis=0) + 1e-12  # 防止除以 0

    return mu, sigma


def transform_normalizer(
    X: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    log_idx: list[int],
    z_only_idx: list[int],
) -> npt.NDArray[np.float64]:
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
    X_new = X.copy().astype(float)

    # 1. 做 log1p
    X_new[:, log_idx] = np.log1p(X_new[:, log_idx])

    # 2. z-score
    z_cols = log_idx + z_only_idx
    X_new[:, z_cols] = (X_new[:, z_cols] - mu) / sigma

    return X_new
