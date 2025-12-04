import shutil
import numpy as np
from scapy.all import rdpcap, load_layer, Scapy_Exception, raw
import glob
from datetime import datetime, timezone, timedelta
import os
import logging
from tqdm import tqdm
from data_processing.FlowMeter.all import extract_flow_features
# Load the TLS layer (requires scapy-ssl_tls extension)
load_layer("tls")

def save_np_files(directory: str, b_flows: list, m_flows: list, attack_class: list):
    """
        儲存 numpy 檔案
        Args:
            directory (str): 儲存目錄
            b_flows (list): 正常流量資料
            m_flows (list of list): 惡意流量資料
            attack_class (list): 攻擊類別名稱列表
        Returns:
            None
    """
    for i in range(len(m_flows)):
        m_flows[i] = np.asarray(m_flows[i])
    b_flows = np.asarray(b_flows)
    if not os.path.exists(directory):
        os.makedirs(directory)
    logging.getLogger("features.save_np_files").info(f'Number of each class: ')
    logging.getLogger("features.save_np_files").info('-' * 50)
    for i, a in enumerate(attack_class):
        np.save(f'{directory}/{a}_t', m_flows[i], allow_pickle=False)
        logging.getLogger("features.save_np_files").info(f'{a} is {len(m_flows[i])}')
    np.save(f'{directory}/benign_t', b_flows, allow_pickle=False)
    logging.getLogger("features.save_np_files").info(f'benign is {len(b_flows)}')

def get_used_pkt(traffic_type, img_shape, pkts):
    """
        取得用於特徵提取的封包
        Args:
            traffic_type (str): 流量類型 ('TCP' 或 'UDP')
            img_shape (tuple): 影像形狀 (高度, 寬度)
            pkts (list): 封包列表
        Returns:
            list: 用於特徵提取的封包列表
    """
    if traffic_type == 'TCP':
        used_pkt = pkts[3:img_shape[1] + 3]
    else:
        used_pkt = pkts[:img_shape[1]]
    return used_pkt

def preprocess_flow(IMG_SHAPE, pkts):
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
        pkt_head = [byte for byte in raw(pkt)]

        pkt_head.extend([0] * (max_size - 24))  # padding 避免長度不足

        # 刪除目的地和來源的 MAC、IP 和 Port
        for start, end in [(0, 11), (26, 37)]:
            pkt_head[start:end + 1] = [None] * (end - start + 1)
        pkt_head = [x for x in pkt_head if x is not None]
        flow.extend(pkt_head[:max_size])

    # 如果流量的封包數量不足，則進行補齊
    size = max_size * IMG_SHAPE[1]
    if len(flow) < size:
        flow.extend([0] * size)
        flow = flow[:size]
    return flow

def run_del(attack_class, packet_shape = (96, 5)):
    """
        執行數據預處理，合併特徵並分割訓練集與測試集
        Args:
            attack_class (list): 攻擊類別名稱列表
            packet_shape (tuple): 封包形狀 (高度, 寬度)
        Returns:
            None
    """
    b_flows = []
    m_flows = [[] for _ in range(len(attack_class))]
    label_count = [0 for _ in range(len(attack_class) + 1)]
    # TCP & Benign
    traffic_type = 'TCP'
    pcapsPath = glob.glob(f'/sdc1/ytlindata/TON_IoT/attack_filter/Benign/split_*/*{traffic_type}*')
    for pcapPath in tqdm(pcapsPath):
        try:
            pkts = rdpcap(pcapPath, count=(packet_shape[1] + 3))
        except (Scapy_Exception, EOFError):
            del pkts
            continue
        # 跳過 TCP 連線的封包
        if traffic_type == 'TCP' and len(pkts) < 4:
            del pkts
            continue
        pkts = get_used_pkt(traffic_type, packet_shape, pkts)
        flow = preprocess_flow(packet_shape, pkts)
        b_flows.append(flow)
        label_count[0] += 1
        del pkts
    # TCP & Malicious
    for classType in attack_class:
        attackPcapsPath = glob.glob(f'/sdc1/ytlindata/TON_IoT/attack_filter/Malicious/{classType}/split_*/*{traffic_type}*')
        label = attack_class.index(classType) + 1
        for pcapPath in tqdm(attackPcapsPath):
            try:
                pkts = rdpcap(pcapPath, count=(packet_shape[1] + 3))
            except (Scapy_Exception, EOFError):
                del pkts
                continue
            # 跳過 TCP 連線的封包
            if traffic_type == 'TCP' and len(pkts) < 4:
                del pkts
                continue
            label_count[label] += 1
            pkts = get_used_pkt(traffic_type, packet_shape, pkts)
            flow = preprocess_flow(packet_shape, pkts)
            m_flows[label - 1].append(flow)
            del pkts
    # UDP & Benign
    traffic_type = 'UDP'
    pcapsPath = glob.glob(f'/sdc1/ytlindata/TON_IoT/attack_filter/Benign/split_*/*{traffic_type}*')
    for pcapPath in tqdm(pcapsPath):
        try:
            pkts = rdpcap(pcapPath, count=(packet_shape[1] + 3))
        except (Scapy_Exception, EOFError):
            del pkts
            continue
        pkts = get_used_pkt(traffic_type, packet_shape, pkts)
        flow = preprocess_flow(packet_shape, pkts)
        b_flows.append(flow)
        label_count[0] += 1
        del pkts
    # UDP & Malicious
    for classType in attack_class:
        attackPcapsPath = glob.glob(f'/sdc1/ytlindata/TON_IoT/attack_filter/Malicious/{classType}/split_*/*{traffic_type}*')
        label = attack_class.index(classType) + 1
        for pcapPath in tqdm(attackPcapsPath):
            try:
                pkts = rdpcap(pcapPath, count=(packet_shape[1] + 3))
            except (Scapy_Exception, EOFError):
                del pkts
                continue
            label_count[label] += 1
            pkts = get_used_pkt(traffic_type, packet_shape, pkts)
            flow = preprocess_flow(packet_shape, pkts)
            m_flows[label - 1].append(flow)
            del pkts
    # save files
    logging.getLogger("features.run_del").info(label_count)
    save_np_files(
        f"/sdc1/ytlindata/TON_IoT/del_{packet_shape[0]}_{packet_shape[1]}_flows(delall)",
        b_flows,
        m_flows,
        attack_class
    )

def merge_flow_and_raw_features(flow_features: dict | None, raw_feature: list) -> list:
    """
        合併流量特徵與原始特徵
        Args:
            flow_features (dict): 流量特徵字典
            raw_feature (list): 原始特徵列表
        Returns:
            list: 合併後的特徵列表
    """
    # 按照固定順序加入 flow features
    flow_feature_order = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Destination Port",
        "Source Port",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Max Packet Length",
        "Min Packet Length",
        "Packet Length Std",
        "SYN Flag Count",
        "ACK Flag Count",
        "Protocol",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Std",
        "Bwd IAT Max",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
    ]
    feature_vector = []
    for key in flow_feature_order:
        feature_vector.append(flow_features.get(key, 0.0) if flow_features is not None else 0.0)
    feature_vector.extend(raw_feature)
    return feature_vector

def flow_to_features_file(flow_pcaps: list, output_file: str, packet_shape: tuple = (96, 5), is_labelled: callable = None) -> list:
    """
        將流量 pcap 轉換為特徵檔案
        Args:
            flow_pcaps (list): 流量 pcap 檔案列表
            output_file (str): 輸出特徵檔案路徑
            packet_shape (tuple): 封包形狀 (位元組數, 封包數)
            is_labelled (callable, optional): 標記回調函數，接受 pcapPath 和 pkts 作為參數
        Returns:
            list: 提取的特徵列表
    """
    features_list = []
    print("length of flow_pcaps:", len(flow_pcaps))
    for pcapPath in tqdm(flow_pcaps):
        traffic_type = 'TCP'
        if '.UDP_' in pcapPath:
            traffic_type = 'UDP'
        try:
            pkts = rdpcap(pcapPath, count = (packet_shape[1] + 3))
        except Exception as e:
            logging.getLogger("features.flow_to_features_file").error(f"Error reading {pcapPath}: {e}", exc_info = True)
            del pkts
            continue
        # 跳過 TCP 連線的封包
        if traffic_type == 'TCP' and len(pkts) < 4:
            del pkts
            continue
        if is_labelled is not None:
            if not is_labelled(pcapPath, pkts):
                del pkts
                continue
        pkts = get_used_pkt(traffic_type, packet_shape, pkts)
        try:
            flow_features = extract_flow_features(pkts)
        except Exception as e:
            logging.getLogger("features.flow_to_features_file").warning(f"Error extracting flow features in {pcapPath}: {e}", exc_info = True)
            flow_features = None
        raw_feature = preprocess_flow(packet_shape, pkts)
        merge_features = merge_flow_and_raw_features(flow_features, raw_feature)
        features_list.append(merge_features)
        del pkts
    logging.getLogger("features.flow_to_features_file").info(f'Extracted features from {len(features_list)} flows.')
    # 儲存特徵檔案
    features_array = np.asarray(features_list)
    os.makedirs(os.path.dirname(output_file), exist_ok = True)
    np.save(output_file, features_array, allow_pickle = False)
    logging.getLogger("features.flow_to_features_file").info(f'Saved features to {output_file}')
    return features_list

def mearge_feature_files(feature_files: list, output_file: str):
    """
        合併多個特徵檔案為一個檔案
        Args:
            feature_files (list): 特徵檔案列表
            output_file (str): 輸出特徵檔案路徑
        Returns:
            None
    """
    all_features = []
    for feature_file in tqdm(feature_files):
        try:
            features = np.load(feature_file)
            if features.size == 0:
                logging.getLogger("features.merge_feature_files").warning(f"No features in {feature_file}, skipping.")
                continue
            all_features.append(features)
        except Exception as e:
            logging.getLogger("features.merge_feature_files").error(f"Error loading {feature_file}: {e}")
            continue
    if all_features:
        merged_features = np.concatenate(all_features, axis = 0)
        os.makedirs(os.path.dirname(output_file), exist_ok = True)
        np.save(output_file, merged_features, allow_pickle = False)
        logging.getLogger("features.merge_feature_files").info(f'Merged features saved to {output_file}')
    else:
        logging.getLogger("features.merge_feature_files").warning(f"No features to merge in {output_file}.")
        
def copy_feature_file(feature_file: str, output_file: str):
    """
        複製多個特徵檔案到指定目錄
        Args:
            feature_file (str): 特徵檔案路徑
            output_file (str): 輸出檔案路徑
        Returns:
            None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok = True)
    try:
        shutil.copy(feature_file, output_file)
        logging.getLogger("features.copy_feature_file").info(f'Copied {feature_file} to {output_file}')
    except Exception as e:
        logging.getLogger("features.copy_feature_file").error(f"Error copying {feature_file}: {e}")