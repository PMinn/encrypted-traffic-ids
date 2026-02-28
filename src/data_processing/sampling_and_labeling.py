from typing import Callable, TypedDict
import json
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
from tabulate import tabulate


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


class SamplingClassesInfo(TypedDict):
    name: str
    data_path: Path
    sampling_count: int


class LabelingClassesInfo(TypedDict):
    name: str
    data_path: Path


def load_data(
    classes_info: list[SamplingClassesInfo] | list[LabelingClassesInfo],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], dict[str, int]]:
    for index, class_info in enumerate(classes_info):
        if "data_path" not in class_info:
            raise SamplingInfoMissingException(
                f"Class '{class_info['name']}' is missing 'data_path' in classes_info."
            )
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
    classes_info: list[SamplingClassesInfo],
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
    data_path: Path | None = None,
) -> None:
    """
    根據指定的採樣數量對各類別進行過採樣或欠採樣
    Args:
        classes_info (list[SamplingClassesInfo]): 各類別的資料路徑資訊
        save_to (Path): 採樣後資料的儲存路徑
    Returns:
        None
    """
    save_to.mkdir(parents=True, exist_ok=True)
    if data_path is None:
        data, label, data_counts = load_data(classes_info)
    else:
        data = np.load(str(data_path / "sampled_data.npy"))
        label = np.load(str(data_path / "sampled_label.npy"))
        data_counts = {
            info["name"]: int(np.sum(label == index)) 
            for index, info in enumerate(classes_info)
        }
    counter = []
    under_strategy = {}
    over_strategy = {}
    for index, class_info in enumerate(classes_info):
        if "name" not in class_info:
            raise SamplingInfoMissingException(
                f"Class info at index {index} is missing 'name' in classes_info."
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
        counter.append([class_info["name"], current_count])
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
    np.save(save_to / "sampled_data.npy", data, allow_pickle=False)
    np.save(save_to / "sampled_label.npy", label, allow_pickle=False)
    for index, class_info in enumerate(classes_info):
        counter[index].append(int(np.sum(label == index)))
    logging.getLogger("sampling").info(
        "sampling results:\n"
        + tabulate(counter, headers=["Class", "Before Sampling", "After Sampling"])
    )
    with open(save_to / "sampled.json", "w") as f:
        json.dump(
            {
                "classes": [info["name"] for info in classes_info],
                "before_sampling": {
                    info["name"]: count
                    for info, count in zip(classes_info, [c[1] for c in counter])
                },
                "after_sampling": {
                    info["name"]: count
                    for info, count in zip(classes_info, [c[2] for c in counter])
                },
            },
            f,
            indent=4,
        )
    logging.getLogger("sampling").info(f"Sampling data saved to {save_to}.")


class SamplingInfoMissingException(Exception):
    pass


def labeling(
    classes_info: list[LabelingClassesInfo],
    save_to: Path,
) -> None:
    save_to.mkdir(parents=True, exist_ok=True)
    data, label, data_counts = load_data(classes_info)
    np.save(str(save_to / "sampled_data.npy"), data, allow_pickle=False)
    np.save(str(save_to / "sampled_label.npy"), label, allow_pickle=False)
    with open(save_to / "labeled.json", "w") as f:
        json.dump(
            {
                "classes": [info["name"] for info in classes_info],
                "counts": data_counts,
            },
            f,
            indent=4,
        )


def suggester(
    class_counts: dict[str, int], alpha: float, json_path: Path | None = None
) -> None:
    """
    取樣建議
    Arguments:
        class_counts (dict): 各類別的數量字典
        alpha (float): 平衡參數，範圍在 0 到 1 之間，控制原始比例與完全平衡比例的權重。0 表示不變，1 表示完全平衡。
    Returns:
        None
    """
    import numpy as np

    classes = list(class_counts.keys())
    counts = np.array(list(class_counts.values()), dtype=float)

    N = counts.sum()
    K = len(counts)
    p = counts / N  # 原始比例
    u = np.full(K, 1.0 / K)  # 完全平衡比例

    # 中位數乘上類別數量作為目標總數
    N_target = int(np.median(counts) * K)

    q = (1 - alpha) * p + alpha * u
    q = q / q.sum()  # 避免小數誤差

    n_target = np.rint(q * N_target).astype(int)
    logging.getLogger("sampling").info(
        tabulate(
            zip(classes, counts, n_target),
            headers=["Class", "Original Count", "Suggested Sample Count"],
        )
    )
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "alpha": alpha,
                    "number_before_sampling": {
                        classes[i]: int(counts[i]) for i in range(len(classes))
                    },
                    "number_of_samples_suggested": {
                        classes[i]: int(n_target[i]) for i in range(len(classes))
                    },
                },
                f,
                indent=4,
            )
