from typing import Callable, Generator
import shutil
import numpy as np
from scapy.all import rdpcap, load_layer, PacketList
from scapy.compat import raw
import logging
from tqdm import tqdm
from data_processing.FlowMeter.extract_flow_features_73 import (
    extract_flow_features_73,
    get_feature_names_73,
)
from pathlib import Path

# Load the TLS layer (requires scapy-ssl_tls extension)
load_layer("tls")


def get_used_pkt(
    traffic_type: str, img_shape: tuple[int, int], pkts: PacketList
) -> PacketList:
    """
    取得用於特徵提取的封包
    Args:
        traffic_type (str): 流量類型 ('TCP' 或 'UDP')
        img_shape (tuple[int, int]): 影像形狀 (高度, 寬度)
        pkts (PacketList): 封包列表
    Returns:
        list[Packet]: 用於特徵提取的封包列表
    """
    used_pkt: PacketList
    if traffic_type == "TCP":
        used_pkt = pkts[3 : img_shape[1] + 3]
    else:
        used_pkt = pkts[: img_shape[1]]
    return used_pkt


def preprocess_flow(IMG_SHAPE: tuple[int, int], pkts: PacketList) -> list[int]:
    """
    預處理單一流量的封包，轉換為定長度的特徵向量
    Args:
        IMG_SHAPE (tuple): 影像形狀 (位元組數, 封包數)
        pkts (list): 封包列表
    Returns:
        list: 預處理後的特徵向量
    """
    max_size = IMG_SHAPE[0]
    flow = []
    # for pkt in pkts[3:IMG_SHAPE[1] + 3]:
    for pkt in pkts:  # get the first img_shape[1] packets
        # if Ether not in pkt:
        #     raise Exception("Not Ethernet II")

        # 取得封包的前 IMG_SHAPE[0] 個位元組
        pkt_head: list[int | None] = [byte for byte in raw(pkt)]

        pkt_head.extend([0] * (max_size - 24))  # padding 避免長度不足

        # 刪除目的地和來源的 MAC、IP 和 Port
        for start, end in [(0, 11), (26, 37)]:
            pkt_head[start : end + 1] = [None] * (end - start + 1)
        pkt_head_without_info: list[int] = [x for x in pkt_head if x is not None]
        flow.extend(pkt_head_without_info[:max_size])

    # 如果流量的封包數量不足，則進行補齊
    size = max_size * IMG_SHAPE[1]
    if len(flow) < size:
        flow.extend([0] * size)
        flow = flow[:size]
    return flow


def merge_flow_and_raw_features(
    flow_feature_order: list[str],
    flow_features: dict[str, float] | None,
    raw_feature: list[int],
) -> list[float]:
    """
    合併流量特徵與原始特徵
    Args:
        flow_feature_order (list): 流量特徵的順序列表
        flow_features (dict): 流量特徵字典
        raw_feature (list): 原始特徵列表
    Returns:
        list: 合併後的特徵列表
    """
    feature_vector = []
    for key in flow_feature_order:
        feature_vector.append(
            flow_features.get(key, 0.0) if flow_features is not None else 0.0
        )
    feature_vector.extend(raw_feature)
    return feature_vector


def flow_to_features_file(
    flow_pcaps: list[Path] | Generator[Path, None, None],
    output_file: Path,
    packet_shape: tuple[int, int] = (96, 5),
    is_labelled: Callable[[Path, PacketList], bool] | None = None,
) -> list[list[float]]:
    """
    將流量 pcap 轉換為特徵檔案
    Args:
        flow_pcaps (list[Packet] | PacketList): 流量 pcap 檔案列表
        output_file (Path): 輸出特徵檔案路徑
        packet_shape (tuple): 封包形狀 (位元組數, 封包數)
        is_labelled (Callable[[Path, PacketList], bool] | None): 標記回調函數，接受 pcapPath 和 pkts 作為參數
    Returns:
        list: 提取的特徵列表
    """
    flow_feature_order = get_feature_names_73()
    features_list = []
    print("length of flow_pcaps:", len(list(flow_pcaps)))
    for pcapPath in tqdm(flow_pcaps):
        traffic_type = "TCP"
        if ".UDP_" in str(pcapPath):
            traffic_type = "UDP"
        try:
            pkts = rdpcap(str(pcapPath), count=(packet_shape[1] + 3))
        except Exception as e:
            logging.getLogger("features.flow_to_features_file").error(
                f"Error reading {pcapPath}: {e}", exc_info=True
            )
            del pkts
            continue
        # 跳過 TCP 連線的封包
        if traffic_type == "TCP" and len(pkts) < 4:
            del pkts
            continue
        if is_labelled is not None:
            if not is_labelled(pcapPath, pkts):
                del pkts
                continue
        pkts = get_used_pkt(traffic_type, packet_shape, pkts)
        try:
            flow_features = extract_flow_features_73(pkts)
        except Exception as e:
            logging.getLogger("features.flow_to_features_file").warning(
                f"Error extracting flow features in {pcapPath}: {e}", exc_info=True
            )
            flow_features = None
        raw_feature = preprocess_flow(packet_shape, pkts)
        merge_features = merge_flow_and_raw_features(
            flow_feature_order, flow_features, raw_feature
        )
        features_list.append(merge_features)
        del pkts
    logging.getLogger("features.flow_to_features_file").info(
        f"Extracted features from {len(features_list)} flows."
    )
    # 儲存特徵檔案
    features_array = np.asarray(features_list)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_file), features_array, allow_pickle=False)
    logging.getLogger("features.flow_to_features_file").info(
        f"Saved features to {output_file}"
    )
    return features_list


def mearge_feature_files(feature_files: list[Path], output_file: Path) -> None:
    """
    合併多個特徵檔案為一個檔案
    Args:
        feature_files (list[Path]): 特徵檔案列表
        output_file (Path): 輸出特徵檔案路徑
    Returns:
        None
    """
    all_features = []
    for feature_file in tqdm(feature_files):
        try:
            features = np.load(str(feature_file))
            if features.size == 0:
                logging.getLogger("features.merge_feature_files").warning(
                    f"No features in {feature_file}, skipping."
                )
                continue
            all_features.append(features)
        except Exception as e:
            logging.getLogger("features.merge_feature_files").error(
                f"Error loading {feature_file}: {e}"
            )
            continue
    if all_features:
        merged_features = np.concatenate(all_features, axis=0)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_file), merged_features, allow_pickle=False)
        logging.getLogger("features.merge_feature_files").info(
            f"Merged features saved to {output_file}"
        )
    else:
        logging.getLogger("features.merge_feature_files").warning(
            f"No features to merge in {output_file}."
        )


def copy_feature_file(feature_file: Path, output_file: Path) -> None:
    """
    複製多個特徵檔案到指定目錄
    Args:
        feature_file (Path): 特徵檔案路徑
        output_file (Path): 輸出檔案路徑
    Returns:
        None
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(feature_file, output_file)
        logging.getLogger("features.copy_feature_file").info(
            f"Copied {feature_file} to {output_file}"
        )
    except Exception as e:
        logging.getLogger("features.copy_feature_file").error(
            f"Error copying {feature_file}: {e}"
        )
