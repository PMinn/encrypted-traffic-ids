from typing import Callable, TypedDict
import json
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt


def get_oversampling_by_kmeans(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    strategy: dict[int, int],
    k: int = 3,
    cluster_balance_threshold: float = 0.005,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
]:
    from imblearn.over_sampling import KMeansSMOTE

    before_unique, before_counts = np.unique(y, return_counts=True)
    oversample = KMeansSMOTE(
        sampling_strategy=strategy,
        random_state=42,
        k_neighbors=k,
        cluster_balance_threshold=cluster_balance_threshold,
    )  # 0.0016
    x, y = oversample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts=True)
    return x, y, (before_counts, after_counts)


def get_oversampling_by_random(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.int_], strategy: dict[int, int]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
]:
    from imblearn.over_sampling import RandomOverSampler

    before_unique, before_counts = np.unique(y, return_counts=True)
    oversample = RandomOverSampler(sampling_strategy=strategy, random_state=42)
    x, y = oversample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts=True)
    return x, y, (before_counts, after_counts)


def get_undersampling_by_random(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.int_], strategy: dict[int, int]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int_],
    tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
]:
    from imblearn.under_sampling import RandomUnderSampler

    before_unique, before_counts = np.unique(y, return_counts=True)
    undersample = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
    x, y = undersample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts=True)
    return x, y, (before_counts, after_counts)


class ClassesInfo(TypedDict):
    name: str
    data_path: Path
    sampling_count: int


def load_data(
    classes_info: list[ClassesInfo],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], dict[str, int]]:
    x = []
    y: list[int] = []
    data_counts = {}
    for index, class_info in enumerate(classes_info):
        data = np.load(str(class_info["data_path"]))
        number_of_data = data.shape[0]
        x.extend(data)
        y.extend(np.ones(number_of_data).astype(int) * index)
        data_counts[class_info["name"]] = number_of_data
    return np.array(x), np.array(y), data_counts


def sampling(
    classes_info: list[ClassesInfo],
    save_to: Path,
    oversampling: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.int_], dict[int, int]],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.int_],
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
        ],
    ] = get_oversampling_by_random,
    undersampling: Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.int_], dict[int, int]],
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.int_],
            tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
        ],
    ] = get_undersampling_by_random,
) -> None:
    """
    根據指定的採樣數量對各類別進行過採樣或欠採樣
    Args:
        classes_info (list): 各類別的資料路徑資訊，格式為 [{'name': str, 'data_path': Path, 'sampling_count': int}, ...]
        save_to (Path): 採樣後資料的儲存路徑
    Returns:
        None
    """
    save_to.mkdir(parents=True, exist_ok=True)
    data, label, data_counts = load_data(classes_info)
    under_strategy = {}
    over_strategy = {}
    for index, class_info in enumerate(classes_info):
        if "name" not in class_info:
            raise SamplingInfoMissingException(
                f"Class info at index {index} is missing 'name' in classes_info."
            )
        if "data_path" not in class_info:
            raise SamplingInfoMissingException(
                f"Class '{class_info['name']}' is missing 'data_path' in classes_info."
            )
        if "sampling_count" not in class_info:
            raise SamplingInfoMissingException(
                f"Class '{class_info['name']}' is missing 'sampling_count' in classes_info."
            )
        target_count = class_info.get("sampling_count", 0)
        current_count = data_counts.get(class_info["name"], 0)
        if target_count < current_count:
            under_strategy[index] = target_count
        elif target_count > current_count:
            over_strategy[index] = target_count
    if len(under_strategy) > 0:
        data, label, process = undersampling(data, label, under_strategy)
        before, after = process
        before_str = "\n".join(
            [f'{classes_info[i]["name"]}: {count}' for i, count in enumerate(before)]
        )
        after_str = "\n".join(
            [f'{classes_info[i]["name"]}: {count}' for i, count in enumerate(after)]
        )
        logging.getLogger("sampling").info(
            f"Undersampling applied.\nBefore:\n{before_str}.\nAfter:\n{after_str}."
        )
    if len(over_strategy) > 0:
        data, label, process = oversampling(data, label, over_strategy)
        before, after = process
        before_str = "\n".join(
            [f'{classes_info[i]["name"]}: {count}' for i, count in enumerate(before)]
        )
        after_str = "\n".join(
            [f'{classes_info[i]["name"]}: {count}' for i, count in enumerate(after)]
        )
        logging.getLogger("sampling").info(
            f"Oversampling applied.\nBefore:\n{before_str}.\nAfter:\n{after_str}."
        )
    np.save(str(save_to / "sampled_data.npy"), data, allow_pickle=False)
    np.save(str(save_to / "sampled_label.npy"), label, allow_pickle=False)
    with open(save_to / "classes.json", "w") as f:
        json.dump([info["name"] for info in classes_info], f)


class SamplingInfoMissingException(Exception):
    pass
