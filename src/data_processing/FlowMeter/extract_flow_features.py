from typing import Iterable
import scapy
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
import statistics
from src.data_processing.FlowMeter.FlowMeter_Exception import NoPacketsException, NoIPPacketsException

def extract_flow_features(pkts: scapy.plist.PacketList) -> dict[str, int | float]:
    """
        從封包列表中提取流量特徵
        Args:
            pkts (scapy.plist.PacketList): 封包列表
        Returns:
            dict[str, int | float]: 流量特徵字典
    """
    if len(pkts) == 0:
        raise NoPacketsException("PCAP 中沒有封包")

    # 只保留有 IP 的封包
    ip_pkts = [p for p in pkts if IP in p or IPv6 in p]
    if not ip_pkts:
        raise NoIPPacketsException("PCAP 中沒有 IP 封包")

    # 以第一個 IP 封包決定 flow 的「前向」方向
    first = ip_pkts[0]
    if IP in first:
        src_ip = first[IP].src
        dst_ip = first[IP].dst
    else:
        src_ip = first[IPv6].src
        dst_ip = first[IPv6].dst

    # 優先用 TCP/UDP 決定 port，否則 None
    src_port = None
    dst_port = None
    if TCP in first:
        src_port = first[TCP].sport
        dst_port = first[TCP].dport
        proto = 6
    elif UDP in first:
        src_port = first[UDP].sport
        dst_port = first[UDP].dport
        proto = 17
    else:
        if IP in first:
            proto = first[IP].proto
        else:
            proto = first[IPv6].nh

    # 方向分類
    fwd_lengths = []
    bwd_lengths = []
    all_lengths = []

    fwd_times = []
    bwd_times = []

    fwd_seg_sizes = []  # TCP segment payload size
    bwd_seg_sizes = []

    syn_count = 0
    ack_count = 0

    for p in ip_pkts:
        if IP in p:
            ip = p[IP]
        else:
            ip = p[IPv6]

        # 封包長度：優先 IP.len，否則用 len(p)
        if hasattr(ip, "len") and ip.len is not None:
            plen = int(ip.len)
        else:
            plen = len(p)

        all_lengths.append(plen)
        t = float(p.time)

        # 判斷方向：與第一個封包相同 src/dst 即為 forward
        is_fwd = (ip.src == src_ip and ip.dst == dst_ip)
        is_bwd = (ip.src == dst_ip and ip.dst == src_ip)

        # TCP flag 與 segment 大小
        tcp_seg_len = None
        if TCP in p:
            tcp = p[TCP]
            # TCP flags
            flags = int(tcp.flags)
            if flags & 0x02:  # SYN
                syn_count += 1
            if flags & 0x10:  # ACK
                ack_count += 1

            # TCP payload 長度當作 segment size（簡化版）
            tcp_seg_len = len(bytes(tcp.payload))

        # 按方向分類
        if is_fwd:
            fwd_lengths.append(plen)
            fwd_times.append(t)
            if tcp_seg_len is not None:
                fwd_seg_sizes.append(tcp_seg_len)
        elif is_bwd:
            bwd_lengths.append(plen)
            bwd_times.append(t)
            if tcp_seg_len is not None:
                bwd_seg_sizes.append(tcp_seg_len)
        else:
            # 不符合這兩個方向（理論上單一 flow 不應該發生），先忽略
            continue

    # Flow 時間
    times = [float(p.time) for p in ip_pkts]
    start_time = min(times)
    end_time = max(times)
    flow_duration = max(end_time - start_time, 1e-9)  # 避免除以零，秒

    total_fwd_pkts = len(fwd_lengths)
    total_bwd_pkts = len(bwd_lengths)
    total_pkts = len(ip_pkts)

    total_fwd_bytes = sum(fwd_lengths)
    total_bwd_bytes = sum(bwd_lengths)
    total_bytes = total_fwd_bytes + total_bwd_bytes

    # 封包長度統計
    max_len = max(all_lengths) if all_lengths else 0
    min_len = min(all_lengths) if all_lengths else 0
    len_std = safe_stdev(all_lengths)

    # IAT（inter-arrival time）計算
    def iat_stats(ts: list[float]):
        if len(ts) < 2:
            return 0.0, 0.0, 0.0
        ts_sorted = sorted(ts)
        diffs = [ts_sorted[i+1] - ts_sorted[i] for i in range(len(ts_sorted) - 1)]
        mean_ = safe_mean(diffs)
        max_ = max(diffs) if diffs else 0.0
        std_ = safe_stdev(diffs)
        return mean_, max_, std_

    fwd_iat_mean, fwd_iat_max, fwd_iat_std = iat_stats(fwd_times)
    bwd_iat_mean, bwd_iat_max, _ = iat_stats(bwd_times)

    # Flow 級特徵
    flow_pkts_per_s = total_pkts / flow_duration
    flow_bytes_per_s = total_bytes / flow_duration

    # 平均封包長度（方向性）
    fwd_pkt_len_mean = safe_mean(fwd_lengths)
    bwd_pkt_len_mean = safe_mean(bwd_lengths)

    # Segment 大小（若沒 TCP，就回退用封包長度平均）
    avg_fwd_seg_size = safe_mean(fwd_seg_sizes) if fwd_seg_sizes else fwd_pkt_len_mean
    avg_bwd_seg_size = safe_mean(bwd_seg_sizes) if bwd_seg_sizes else bwd_pkt_len_mean

    features = {
        "Flow Duration": flow_duration,
        "Total Fwd Packets": total_fwd_pkts,
        "Total Backward Packets": total_bwd_pkts,
        "Destination Port": dst_port,
        "Source Port": src_port,
        "Flow Packets/s": flow_pkts_per_s,
        "Flow Bytes/s": flow_bytes_per_s,
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Total Length of Bwd Packets": total_bwd_bytes,
        "Fwd Packet Length Mean": fwd_pkt_len_mean,
        "Bwd Packet Length Mean": bwd_pkt_len_mean,
        "Max Packet Length": max_len,
        "Min Packet Length": min_len,
        "Packet Length Std": len_std,
        "SYN Flag Count": syn_count,
        "ACK Flag Count": ack_count,
        "Protocol": proto,
        "Fwd IAT Mean": fwd_iat_mean,
        "Bwd IAT Mean": bwd_iat_mean,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Std": fwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Avg Fwd Segment Size": avg_fwd_seg_size,
        "Avg Bwd Segment Size": avg_bwd_seg_size,
    }

    return features

def safe_mean(values: Iterable[float | int]) -> float:
    return statistics.mean(list(values)) if values else 0.0

def safe_stdev(values: Iterable[float | int]) -> float:
    return statistics.pstdev(values) if len(list(values)) > 1 else 0.0

def get_feature_names():
    return [
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