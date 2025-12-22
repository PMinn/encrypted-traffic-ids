from typing import Callable, TypedDict, cast
from scapy.plist import PacketList
import json
import os
from data_processing.FlowMeter.extract_flow_features_73 import (
    extract_flow_features_73,
    get_feature_names_73,
)
from scapy.all import rdpcap
import glob
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
import pandas as pd
from pathlib import Path
from utils.alias import a2p


def handle_pkt(
    pcap_path: Path,
    is_zero_count: dict[str, int],
    is_close_count: dict[str, int],
    total_count: int,
) -> tuple[dict[str, int], dict[str, int], int]:
    pkts = rdpcap(str(pcap_path), count=10000)
    if "TCP" in str(pcap_path):
        if len(pkts) < 4:
            return is_zero_count, is_close_count, total_count
        flow_features_early = extract_flow_features_73(pkts[3:8])
    else:
        flow_features_early = extract_flow_features_73(pkts[:5])
    if flow_features_early is None:
        return is_zero_count, is_close_count, total_count
    flow_features = extract_flow_features_73(pkts)
    if flow_features is None:
        return is_zero_count, is_close_count, total_count
    for name in flow_features.keys():
        if flow_features[name] == 0.0 and flow_features_early[name] == 0.0:
            is_zero_count[name] += 1
        if (
            flow_features[name] * 0.5
            <= flow_features_early[name]
            <= flow_features[name] * 1.5
        ):
            is_close_count[name] += 1
    total_count += 1
    return is_zero_count, is_close_count, total_count


def features_dict_to_list(
    features_order: list[str], feature_dict: dict[str, float]
) -> list[float]:
    return [feature_dict[name] for name in features_order]


def flows_to_features_file(
    pcaps: list[Path],
    output_path: Path,
    is_labelled: Callable[[Path, PacketList], bool] | None = None,
) -> None:
    features_order = get_feature_names_73()
    all_features = []
    early_features = []
    for pcap in tqdm(pcaps):
        pkts = rdpcap(str(pcap), count=10000)
        if is_labelled is not None:
            if not is_labelled(pcap, pkts):
                del pkts
                continue
        flow_features = extract_flow_features_73(pkts)
        if flow_features is None:
            del pkts
            continue
        if "TCP" in str(pcap):
            if len(pkts) < 4:
                del pkts
                continue
            flow_features_early = extract_flow_features_73(pkts[3:8])
        else:
            flow_features_early = extract_flow_features_73(pkts[:5])
        if flow_features_early is None:
            del pkts
            continue
        all_features.append(features_dict_to_list(features_order, flow_features))
        early_features.append(
            features_dict_to_list(features_order, flow_features_early)
        )
        del pkts
    os.makedirs(output_path, exist_ok=True)
    np.save(output_path / "all.npy", np.array(all_features))
    np.save(output_path / "early.npy", np.array(early_features))


# def get_all_feature() -> None:
#     # 0373
#     pcaps = a2p("@/data/CIC-IDS-2017/split/Monday").glob("split_*/*.pcap")
#     flows_to_features_file(
#         pcaps, "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/benign"
#     )
#     # 0407
#     pcaps = a2p("@/data/CIC-IDS-2017/split/Tuesday").glob("split_*/*.pcap")
#     ftp_pataor_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_21\.", pcap)
#     ]
#     flows_to_features_file(
#         ftp_pataor_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/ftp_brute_force",
#         is_labelled=lambda pcapPath, pkts: 1499170672 <= pkts[0].time <= 1499174417,
#     )
#     ssh_pataor_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_22\.", pcap)
#     ]
#     flows_to_features_file(
#         ssh_pataor_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/ssh_brute_force",
#         is_labelled=lambda pcapPath, pkts: 1499188141 <= pkts[0].time <= 1499195060,
#     )
#     # 0507
#     pcaps = glob.glob(
#         "/home/alanpan/datasets/CIC-IDS-2017/split/Wednesday/split_*/*.pcap"
#     )
#     dos_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_80\.", pcap)
#     ]
#     flows_to_features_file(
#         dos_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/dos_hulk",
#         is_labelled=lambda pcapPath, pkts: (1499262203 <= pkts[0].time <= 1499263642),
#     )
#     flows_to_features_file(
#         dos_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/dos_slowloris",
#         is_labelled=lambda pcapPath, pkts: 1499258934 <= pkts[0].time <= 1499260279,
#     )
#     flows_to_features_file(
#         dos_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/dos_slowhttptest",
#         is_labelled=lambda pcapPath, pkts: 1499260537 <= pkts[0].time <= 1499261870,
#     )
#     # 0607
#     pcaps = glob.glob(
#         "/home/alanpan/datasets/CIC-IDS-2017/split/Thursday/split_*/*.pcap"
#     )
#     web_attack_sql_injection_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_80\.", pcap)
#     ]
#     flows_to_features_file(
#         web_attack_sql_injection_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/web_attack/sql_injection",
#         is_labelled=lambda pcapPath, pkts: 1499348127 <= pkts[0].time <= 1499348576,
#     )
#     web_attack_xss_pcaps = [
#         pcap
#         for pcap in pcaps
#         if re.search(
#             r"_172-16-0-1_(?!(36180|36182|36184|36186|36188|36190)$)_192-168-10-50_80\.",
#             pcap,
#         )
#     ]
#     flows_to_features_file(
#         web_attack_xss_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/web_attack/xss",
#         is_labelled=lambda pcapPath, pkts: 1499346935 <= pkts[0].time <= 1499348122,
#     )
#     web_attack_brute_force_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_80\.", pcap)
#     ]
#     flows_to_features_file(
#         web_attack_brute_force_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/web_attack/brute_force",
#         is_labelled=lambda pcapPath, pkts: 1499343354 <= pkts[0].time <= 1499346012,
#     )
#     infiltration33_pcaps = [
#         pcap
#         for pcap in pcaps
#         if re.search(
#             r"_192-168-10-8_.*_192-168-10-(5|9|12|14|15|16|17|19|25|50|51)_", pcap
#         )
#     ]
#     flows_to_features_file(
#         infiltration33_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/port_scan",
#         is_labelled=lambda pcapPath, pkts: 1499364314 <= pkts[0].time <= 1499366765,
#     )
#     # 0707
#     pcaps = glob.glob("/home/alanpan/datasets/CIC-IDS-2017/split/Friday/split_*/*.pcap")
#     botnet_ares_pcaps = [
#         pcap
#         for pcap in pcaps
#         if re.search(r"_192-168-10-(15|9|14|5|8)_.*_205-174-165-73_", pcap)
#     ]
#     flows_to_features_file(
#         botnet_ares_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/botnet_ares",
#         is_labelled=lambda pcapPath, pkts: 1499432653 <= pkts[0].time <= 1499457685,
#     )
#     ddos_pcaps = [
#         pcap for pcap in pcaps if re.search(r"_172-16-0-1_.*_192-168-10-50_", pcap)
#     ]
#     flows_to_features_file(
#         ddos_pcaps,
#         "/home/alanpan/datasets/CIC-IDS-2017/feature_select/features/ddos",
#         is_labelled=lambda pcapPath, pkts: 1499453791 <= pkts[0].time <= 1499454973,
#     )


def read_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    classes_folders = a2p("@/data/CIC-IDS-2017/feature_select/features").iterdir()
    all_early_features_data_list = []
    all_early_features_labels_list = []
    for index, class_folder in enumerate(classes_folders):
        features_files = glob.glob(
            os.path.join(class_folder, "**", "early.npy"), recursive=True
        )
        if len(features_files) == 0:
            continue
        for features_file in features_files:
            early_features = np.load(features_file)
            if early_features.shape[0] == 0:
                continue
            all_early_features_data_list.append(early_features)
            all_early_features_labels_list.extend([index] * early_features.shape[0])
    all_early_features_data = np.vstack(all_early_features_data_list).astype(np.float64)
    all_early_features_labels = np.array(all_early_features_labels_list).astype(np.int_)
    return all_early_features_data, all_early_features_labels


def read_data_from_data_processing() -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]
):
    features_folder = a2p("@/data/TON_IoT/features/features")
    features_files = [
        features_folder / "benign_features_96_5_24.npy",
        features_folder / "normal_DDoS_features_96_5_24.npy",
        features_folder / "normal_scanning_features_96_5_24.npy",
        features_folder / "Injection_normal_features_96_5_24.npy",
    ]
    all_early_features_data_list = []
    all_early_features_labels_list = []
    for index, features_file in enumerate(features_files):
        data = np.load(features_file)
        if data.shape[0] == 0:
            continue
        all_early_features_data_list.append(data[:, :73])
        labels = np.array([index] * data.shape[0])
        all_early_features_labels_list.extend(labels.tolist())
    all_early_features_data = np.vstack(all_early_features_data_list).astype(np.float64)
    all_early_features_labels = np.array(all_early_features_labels_list).astype(np.int_)
    return all_early_features_data, all_early_features_labels


def read_data_from_data_sampled() -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]
):
    train_folder = a2p("@/data/TON_IoT/features/sampled/train")
    data_file = train_folder / "sampled_data.npy"
    label_file = train_folder / "sampled_label.npy"
    all_early_features_data = np.load(data_file)[:, :73].astype(np.float64)
    all_early_features_labels = np.load(label_file).astype(np.int_)
    return all_early_features_data, all_early_features_labels


def get_mutual_information(
    all_early_features_data: npt.NDArray[np.float64],
    all_early_features_labels: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    mi = mutual_info_classif(
        all_early_features_data, all_early_features_labels, discrete_features=False
    )
    os.makedirs(
        a2p("@/data/TON_IoT/feature_select/mutual_information"),
        exist_ok=True,
    )
    with open(
        a2p("@/data/TON_IoT/feature_select/mutual_information/mutual_information.json"),
        "w",
    ) as f:
        json.dump(mi.tolist(), f, indent=4)
    return mi


def print_mutual_information(features_order: list[str]) -> None:
    with open(
        "/home/alanpan/datasets/CIC-IDS-2017/feature_select/mutual_information/mutual_information.json",
        "r",
    ) as f:
        mi = json.load(f)
    feature_mi = list(zip(features_order, mi))
    feature_mi.sort(key=lambda x: cast(float, -x[1]))
    table = [
        (index + 1, name, mi_value) for index, (name, mi_value) in enumerate(feature_mi)
    ]
    print(tabulate(table, headers=["#", "Feature Name", "Mutual Information"]))


def get_random_forest_feature_importance(
    all_early_features_data: npt.NDArray[np.float64],
    all_early_features_labels: npt.NDArray[np.int_],
    features_order: list[str],
) -> None:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(all_early_features_data, all_early_features_labels)
    importances = clf.feature_importances_

    os.makedirs(
        "/home/alanpan/datasets/CIC-IDS-2017/feature_select/random_forest_feature_importance",
        exist_ok=True,
    )
    with open(
        "/home/alanpan/datasets/CIC-IDS-2017/feature_select/random_forest_feature_importance/random_forest_feature_importance.json",
        "w",
    ) as f:
        json.dump(importances.tolist(), f, indent=4)


def print_random_forest_feature_importance(features_order: list[str]) -> None:
    with open(
        "/home/alanpan/datasets/CIC-IDS-2017/feature_select/random_forest_feature_importance/random_forest_feature_importance.json",
        "r",
    ) as f:
        importances = json.load(f)
    feature_importances = list(zip(features_order, importances))
    feature_importances.sort(key=lambda x: cast(float, -x[1]))
    table = [
        (index + 1, name, importance)
        for index, (name, importance) in enumerate(feature_importances)
    ]
    print(tabulate(table, headers=["#", "Feature Name", "Feature Importance"]))


def get_correlation(
    features_order: list[str],
    all_early_features_data: npt.NDArray[np.float64],
    all_early_features_labels: npt.NDArray[np.int_],
) -> list[list[str]]:
    df = pd.DataFrame(all_early_features_data, columns=features_order)
    df["label"] = all_early_features_labels
    correlation_matrix = df.corr()
    print(correlation_matrix)
    os.makedirs(a2p("@/data/TON_IoT/feature_select/correlation"), exist_ok=True)
    correlation_matrix.to_csv(
        a2p("@/data/TON_IoT/feature_select/correlation/correlation_matrix.csv")
    )
    # 根據 correlation matrix 找出高度相關的特徵群組
    correlated_groups = []
    visited = set()
    threshold = 0.9

    for i, feature_name in enumerate(features_order):
        if feature_name in visited:
            continue
        group = [feature_name]
        visited.add(feature_name)
        for j in range(i + 1, len(features_order)):
            other_feature_name = features_order[j]
            if (
                abs(
                    cast(float, correlation_matrix.at[feature_name, other_feature_name])
                )
                >= threshold
            ):
                if other_feature_name in visited:
                    continue
                group.append(other_feature_name)
                visited.add(other_feature_name)
        correlated_groups.append(group)
    with open(
        a2p("@/data/TON_IoT/feature_select/correlation/correlated_groups.json"),
        "w",
    ) as f:
        json.dump(correlated_groups, f, indent=4)
    print("相關性群組:")
    for group in correlated_groups:
        if len(group) > 1:
            print(group)
    return correlated_groups


if __name__ == "__main__":
    # 取得 feature 名稱列表
    features_order = get_feature_names_73()

    # 將切完的 flows 轉成 feature 檔案
    # get_all_feature()

    # 讀取 feature 檔案，合併成 dataset
    # all_early_features_data, all_early_features_labels = read_data()
    # 或者從 data_processing 讀取 feature 檔案
    all_early_features_data, all_early_features_labels = read_data_from_data_sampled()

    # 移除 "Min_seg_size_forward" 特徵，因為它有些值為無限大
    all_early_features_data = np.delete(
        all_early_features_data, features_order.index("Min_seg_size_forward"), axis=1
    )
    features_order.remove("Min_seg_size_forward")
    print("移除以下特徵:", ["Min_seg_size_forward"])
    print("剩餘數量:", len(features_order))

    # 移除無限大的特徵
    # features_to_remove_indexes = []
    # if is_run_select_with_feature:
    #     for index, value in enumerate(all_early_features_data.flatten()):
    #         if value == float('inf') or value == float('-inf'):
    #             features_to_remove_indexes.append(index % all_early_features_data.shape[1])
    # features_to_remove_indexes = list(set(features_to_remove_indexes))
    # features_to_remove_names = [features_order[index] for index in features_to_remove_indexes]
    # if is_run_select_with_feature:
    #     all_early_features_data = np.delete(all_early_features_data, features_to_remove_indexes, axis = 1)
    # features_order = [name for name in features_order if name not in features_to_remove_names]
    # print("移除以下特徵:", features_to_remove_names)
    # print("剩餘數量:", len(features_order))

    # 量化 feature 對 class 有多少資訊
    mi = get_mutual_information(all_early_features_data, all_early_features_labels)
    # print_mutual_information(features_order)

    # 根據 mutual information 移除相關性最低的 50% 特徵
    # with open(
    #     "/home/alanpan/datasets/CIC-IDS-2017/feature_select/mutual_information/mutual_information.json",
    #     "r",
    # ) as f:
    #     mi = json.load(f)
    # mi_list = list(zip(features_order, mi))
    # mi_list.sort(key = lambda x: x[1])
    # features_to_remove = mi_list[:int(len(mi_list) * 0.5)]
    # features_to_remove_indexes = [features_order.index(name) for name, _ in features_to_remove]
    # features_to_remove_names = [name for name, _ in features_to_remove]
    # all_early_features_data = np.delete(all_early_features_data, features_to_remove_indexes, axis = 1)
    # features_order = [name for name in features_order if name not in features_to_remove_names]
    # print("移除以下特徵:", features_to_remove_names)
    # print("剩餘數量:", len(features_order))

    # 資訊重複性分析
    correlated_groups = get_correlation(
        features_order, all_early_features_data, all_early_features_labels
    )

    # with open(
    #     "/home/alanpan/datasets/CIC-IDS-2017/feature_select/correlation/correlated_groups.json",
    #     "r",
    # ) as f:
    #     correlated_groups = json.load(f)

    # 將相關性群組中的特徵依 mutual information 排序
    class Mi(TypedDict):
        name: str
        mi: float

    groups_with_mi: list[list[Mi]] = []
    for i_index, group in enumerate(correlated_groups):
        groups_with_mi.append([])
        for j_index, name in enumerate(group):
            mi_index = features_order.index(name)
            groups_with_mi[i_index].append({"name": name, "mi": mi[mi_index]})
        groups_with_mi[i_index].sort(key=lambda x: -x["mi"])
    result = []
    for group_with_mi in groups_with_mi:
        if len(group_with_mi) > 0 and group_with_mi[0]["mi"] > 0.5:
            result.append(group_with_mi[0]["name"])
    print("選取以下特徵：", result)
    features_order = get_feature_names_73()
    print("選取特徵索引：", [features_order.index(name) for name in result])
    exit()

    # def remove_keywords_if_in_name(
    #     correlated_groups: list[list[str]], in_name: str, keywords: list
    # ) -> list:
    #     features_to_remove_names = []
    #     for group in correlated_groups:
    #         has_in_name = False
    #         for feature_name in group:
    #             if in_name in feature_name:
    #                 has_in_name = True
    #                 break
    #         if has_in_name:
    #             for feature_name in group:
    #                 if in_name not in feature_name and any(
    #                     keyword in feature_name for keyword in keywords
    #                 ):
    #                     features_to_remove_names.append(feature_name)
    #     return features_to_remove_names

    # # 根據資訊重複性群組
    # with open(
    #     "/home/alanpan/datasets/CIC-IDS-2017/feature_select/correlation/correlated_groups.json",
    #     "r",
    # ) as f:
    #     correlated_groups = json.load(f)
    # correlated_groups = [group for group in correlated_groups if len(group) > 1]
    # print("相關性群組:")
    # for group in correlated_groups:
    #     print(group)

    # # 如果有 Mean 就移除 Max 和 Min
    # features_to_remove_names = remove_keywords_if_in_name(
    #     correlated_groups, "Mean", ["Max", "Min", "Total"]
    # )
    # features_to_remove_indexes = [
    #     features_order.index(name)
    #     for name in features_to_remove_names
    #     if name in features_order
    # ]
    # all_early_features_data = np.delete(
    #     all_early_features_data, features_to_remove_indexes, axis=1
    # )
    # features_order = [
    #     name for name in features_order if name not in features_to_remove_names
    # ]
    # for index, group in enumerate(correlated_groups):
    #     correlated_groups[index] = [
    #         n for n in correlated_groups[index] if n in features_order
    #     ]
    # print("移除以下特徵:", features_to_remove_names)
    # print("剩餘數量:", len(features_order))
    # print("相關性群組:")
    # for group in correlated_groups:
    #     print(group)
    # # 如果有 Bytes 就移除 Length
    # features_to_remove_names = remove_keywords_if_in_name(
    #     correlated_groups, "Bytes", ["Length", "Packets"]
    # )
    # features_to_remove_indexes = [
    #     features_order.index(name)
    #     for name in features_to_remove_names
    #     if name in features_order
    # ]
    # if is_run_select_with_feature:
    #     all_early_features_data = np.delete(
    #         all_early_features_data, features_to_remove_indexes, axis=1
    #     )
    # features_order = [
    #     name for name in features_order if name not in features_to_remove_names
    # ]
    # for index, group in enumerate(correlated_groups):
    #     correlated_groups[index] = [
    #         n for n in correlated_groups[index] if n in features_order
    #     ]
    # print("移除以下特徵:", features_to_remove_names)
    # print("剩餘數量:", len(features_order))
    # print("相關性群組:")
    # for group in correlated_groups:
    #     print(group)
    # print("目前剩餘特徵:", features_order)
    # # ['Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean']
    # # ['Avg Fwd Segment Size', 'Subflow Fwd Bytes']
    # # ['Average Packet Size', 'Avg Bwd Segment Size', 'Subflow Bwd Bytes']
    # # ['Flow Bytes/s']
    # # ['Bwd IAT Mean']
    # # ['Active Mean', 'Active Std']
    # # 移除 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean' 因為有 'Flow Duration'
    # # 移除 'Subflow Fwd Bytes' 代表傳大封包還是小封包，與 'Flow Bytes/s' 重複
    # # 移除 'Avg Bwd Segment Size' 反映 封包/segment 大小分布，與 'Average Packet Size' 重複
    # # 移除 'Subflow Bwd Bytes' 本質是 throughput，與 'Flow Bytes/s' 重複
    # features_to_remove_names = [
    #     "Flow IAT Mean",
    #     "Flow IAT Std",
    #     "Fwd IAT Mean",
    #     "Subflow Fwd Bytes",
    #     "Avg Bwd Segment Size",
    #     "Subflow Bwd Bytes",
    # ]
    # features_to_remove_indexes = [
    #     features_order.index(name)
    #     for name in features_to_remove_names
    #     if name in features_order
    # ]
    # if is_run_select_with_feature:
    #     all_early_features_data = np.delete(
    #         all_early_features_data, features_to_remove_indexes, axis=1
    #     )
    # features_order = [
    #     name for name in features_order if name not in features_to_remove_names
    # ]
    # for index, group in enumerate(correlated_groups):
    #     correlated_groups[index] = [
    #         n for n in correlated_groups[index] if n in features_order
    #     ]
    # print("移除以下特徵:", features_to_remove_names)
    # print("剩餘數量:", len(features_order))
    # print("相關性群組:")
    # for group in correlated_groups:
    #     print(group)
    # print("目前剩餘特徵:", features_order)
    # with open(
    #     "/home/alanpan/datasets/CIC-IDS-2017/feature_select/selected_features.json", "w"
    # ) as f:
    #     json.dump(features_order, f, indent=4)

    # # 根據 Random Forest 的特徵重要性移除低重要性特徵 (樹狀模型用，已棄用)
    # # if is_run_select_with_feature:
    # #     get_random_forest_feature_importance(all_early_features_data, all_early_features_labels, features_order)
    # # print_random_forest_feature_importance(features_order)
