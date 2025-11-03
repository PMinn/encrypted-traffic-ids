import numpy as np
from scapy.all import *
import glob
from datetime import datetime, timezone, timedelta
import os
from tqdm import tqdm
# Load the TLS layer (requires scapy-ssl_tls extension)
load_layer("tls")
logger = logging.getLogger()

def save_np_files(directory, b_flows, m_flows, attack_class):
    for i in range(len(m_flows)):
        m_flows[i] = np.asarray(m_flows[i])
    b_flows = np.asarray(b_flows)

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.info(f'Number of each class: ')
    logger.info('-' * 50)

    for i, a in enumerate(attack_class):
        np.save(f'{directory}/{a}_t', m_flows[i], allow_pickle=False)
        logger.info(f'{a} is {len(m_flows[i])}')

    np.save(f'{directory}/benign_t', b_flows, allow_pickle=False)
    logger.info(f'benign is {len(b_flows)}')

def get_used_pkt(traffic_type, img_shape, pkts):
    if traffic_type == 'TCP':
        pkts = pkts[3:img_shape[1] + 3]
    else:
        pkts = pkts[:img_shape[1]]
    return pkts

def preprocess_flow(IMG_SHAPE, pkts):
    max_size = IMG_SHAPE[0] - 24
    flow = []
    # for pkt in pkts[3:IMG_SHAPE[1] + 3]:
    for pkt in pkts:  # get the first img_shape[1] packets
        # if Ether not in pkt:
        #     raise Exception("Not Ethernet II")

        # get the first img_shape[0] bytes
        pkt_head = [byte for byte in raw(pkt)]

        pkt_head.extend([0] * max_size)  # padding

        # delete Destination and Source MAC,IP Port
        for start, end in [(0, 11), (26, 37)]:
            pkt_head[start:end + 1] = [None] * (end - start + 1)
        pkt_head = [x for x in pkt_head if x is not None]

        flow.extend(pkt_head[:max_size])

    # if the flow has too few packets, padding again
    size = max_size * IMG_SHAPE[1]
    if len(flow) < size:
        flow.extend([0] * size)
        flow = flow[:size]
    return flow

def run_del(attack_class, packet_shape = (120, 5)):
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
    logger.info(label_count)
    save_np_files(
        f"/sdc1/ytlindata/TON_IoT/del_{packet_shape[0]}_{packet_shape[1]}_flows(delall)",
        b_flows,
        m_flows,
        attack_class
    )