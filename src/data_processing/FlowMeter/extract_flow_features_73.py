from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Packet
import statistics
import math

class FlowStats:
    def __init__(self, src, sport, dst, dport, proto):
        # 5-tuple（forward 方向）
        self.src = src
        self.sport = sport
        self.dst = dst
        self.dport = dport
        self.proto = proto

        # 時間
        self.start = None
        self.end = None
        self.timestamps = []

        # 封包長度與時間（fwd / bwd）
        self.fwd_pkt_len = []
        self.bwd_pkt_len = []
        self.all_pkt_len = []
        self.fwd_times = []
        self.bwd_times = []

        # header 與 flags
        self.fwd_header_len = 0
        self.bwd_header_len = 0

        self.fin = 0
        self.syn = 0
        self.rst = 0
        self.fwd_psh = 0
        self.bwd_psh = 0
        self.ack = 0
        self.fwd_urg = 0
        self.bwd_urg = 0
        self.cwr = 0
        self.ece = 0

        # subflow / tcp 相關
        self.subflow_fwd_pkts = 0
        self.subflow_fwd_bytes = 0
        self.subflow_bwd_pkts = 0
        self.subflow_bwd_bytes = 0

        self.init_win_fwd = 0
        self.init_win_bwd = 0
        self.act_data_fwd = 0
        self.min_seg_size_fwd = math.inf

    def update(self, pkt, direction):
        """
            每個封包進來更新
            Args:
                pkt: scapy Packet
                direction: "fwd" or "bwd"
            returns: None
        """
        ts = float(pkt.time)
        length = len(pkt)

        if self.start is None:
            self.start = ts
        self.end = ts
        self.timestamps.append(ts)

        self.all_pkt_len.append(length)

        if direction == "fwd":
            self.fwd_pkt_len.append(length)
            self.fwd_times.append(ts)
            self.subflow_fwd_pkts += 1
            self.subflow_fwd_bytes += length
        else:
            self.bwd_pkt_len.append(length)
            self.bwd_times.append(ts)
            self.subflow_bwd_pkts += 1
            self.subflow_bwd_bytes += length

        # TCP flags / window / segment size
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & 0x01: self.fin += 1
            if flags & 0x02: self.syn += 1
            if flags & 0x04: self.rst += 1
            if flags & 0x08:
                if direction == "fwd":
                    self.fwd_psh += 1
                else:
                    self.bwd_psh += 1
            if flags & 0x10: self.ack += 1
            if flags & 0x20:
                if direction == "fwd":
                    self.fwd_urg += 1
                else:
                    self.bwd_urg += 1
            if flags & 0x40: self.ece += 1
            if flags & 0x80: self.cwr += 1
            
            if direction == "fwd":
                if self.init_win_fwd == 0:
                    self.init_win_fwd = pkt[TCP].window
                # min seg size forward 只看 forward
                seg_payload_size = len(pkt[TCP].payload)
                if seg_payload_size < self.min_seg_size_fwd:
                    self.min_seg_size_fwd = seg_payload_size
                if seg_payload_size > 0:
                    self.act_data_fwd += 1
            else:
                if self.init_win_bwd == 0:
                    self.init_win_bwd = pkt[TCP].window

        # header 長度
        hdr_len = 0
        if IP in pkt:
            hdr_len += pkt[IP].ihl * 4
        elif IPv6 in pkt:
            hdr_len += 40  # IPv6 header 固定 40 bytes
        if TCP in pkt:
            hdr_len += pkt[TCP].dataofs * 4
        elif UDP in pkt:
            hdr_len += 8  # UDP header 固定 8 bytes
        if direction == "fwd":
            self.fwd_header_len += hdr_len
        else:
            self.bwd_header_len += hdr_len

    def _iat(self, times: list[float]) -> tuple[float, float, float, float, float]:
        """
            計算 IAT 統計值
            Args:
                times: list of timestamps
            returns: (total, mean, std, max, min)
        """
        if len(times) < 2:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        iats = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        total = sum(iats)
        mean = statistics.mean(iats)
        std = statistics.pstdev(iats) if len(iats) > 1 else 0.0
        max_v = max(iats)
        min_v = min(iats)
        return (total, mean, std, max_v, min_v)

    def _active_idle(self, threshold: float = 1.0) -> tuple[float, float, float, float, float, float, float, float]:
        """
            簡單版 active/idle 計算，預設為 1 秒
            Args:
                None
            returns: (active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min)
        """
        if len(self.timestamps) < 2:
            return (0,0,0,0, 0,0,0,0)
        times = sorted(self.timestamps)
        intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        active = [x for x in intervals if x < threshold]
        idle = [x for x in intervals if x >= threshold]

        def stats(arr):
            if not arr:
                return (0,0,0,0)
            mean = statistics.mean(arr)
            std  = statistics.pstdev(arr) if len(arr) > 1 else 0.0
            return (mean, std, max(arr), min(arr))

        return stats(active) + stats(idle)

    def to_features(self):
        """
            轉成特徵 dict
            Args:
                None
            returns: feature_dict
        """
        duration = max((self.end - self.start) if self.start is not None else 0.0, 1e-6)

        total_fwd = len(self.fwd_pkt_len)
        total_bwd = len(self.bwd_pkt_len)
        total_pkts = total_fwd + total_bwd
        total_bytes = sum(self.all_pkt_len)

        # Flow IAT
        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = self._iat(self.timestamps)
        # Fwd IAT
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._iat(self.fwd_times)
        # Bwd IAT
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._iat(self.bwd_times)

        # Active / Idle
        active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min = self._active_idle()

        # 封包長度統計
        def safe_stats(arr):
            if not arr:
                return (0.0, 0.0, 0.0, 0.0)
            mx = max(arr)
            mn = min(arr)
            mean = statistics.mean(arr)
            std = statistics.pstdev(arr) if len(arr) > 1 else 0.0
            return (mx, mn, mean, std)

        fwd_len_max, fwd_len_min, fwd_len_mean, fwd_len_std = safe_stats(self.fwd_pkt_len)
        bwd_len_max, bwd_len_min, bwd_len_mean, bwd_len_std = safe_stats(self.bwd_pkt_len)
        pkt_len_max, pkt_len_min, pkt_len_mean, pkt_len_std = safe_stats(self.all_pkt_len)

        feat = {
            # Flow basic
            "Flow Duration": duration,
            "Destination Port": self.dport,
            "Source Port": self.sport,
            "Protocol": self.proto,

            # Fwd/Bwd 計數
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            "Total Length of Fwd Packets": sum(self.fwd_pkt_len),
            "Total Length of Bwd Packets": sum(self.bwd_pkt_len),

            # Fwd/Bwd 封包長度統計
            "Fwd Packet Length Max": fwd_len_max,
            "Fwd Packet Length Min": fwd_len_min,
            "Fwd Packet Length Mean": fwd_len_mean,
            "Fwd Packet Length Std": fwd_len_std,
            "Bwd Packet Length Max": bwd_len_max,
            "Bwd Packet Length Min": bwd_len_min,
            "Bwd Packet Length Mean": bwd_len_mean,
            "Bwd Packet Length Std": bwd_len_std,

            # Flow bytes/packet rate
            "Flow Bytes/s": total_bytes / duration,
            "Flow Packets/s": total_pkts / duration,

            # Flow IAT
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,

            # Fwd IAT
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,

            # Bwd IAT
            "Bwd IAT Total": bwd_iat_total,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,

            # PSH/URG (這邊簡單用總 PSH/URG 當作 fwd/bwd，必要時可拆開)
            "Fwd PSH Flags": self.fwd_psh,
            "Bwd PSH Flags": self.bwd_psh,
            "Fwd URG Flags": self.fwd_urg,
            "Bwd URG Flags": self.bwd_urg,

            # Header 長度
            "Fwd Header Length": self.fwd_header_len,
            "Bwd Header Length": self.bwd_header_len,
            "Fwd Packets/s": total_fwd / duration,
            "Bwd Packets/s": total_bwd / duration,

            # Packet length (整個 flow)
            "Min Packet Length": pkt_len_min,
            "Max Packet Length": pkt_len_max,
            "Packet Length Mean": pkt_len_mean,
            "Packet Length Std": pkt_len_std,
            "Packet Length Variance": pkt_len_std ** 2,

            # Flags 統計
            "FIN Flag Count": self.fin,
            "SYN Flag Count": self.syn,
            "RST Flag Count": self.rst,
            "PSH Flag Count": sum([self.fwd_psh, self.bwd_psh]),
            "ACK Flag Count": self.ack,
            "URG Flag Count": sum([self.fwd_urg, self.bwd_urg]),
            "CWR Flag Count": self.cwr,
            "ECE Flag Count": self.ece,

            # 比例與平均 segment size
            "Down/Up Ratio": (total_bwd / total_fwd) if total_fwd else -1.0,
            "Average Packet Size": pkt_len_mean,
            "Avg Fwd Segment Size": fwd_len_mean,
            "Avg Bwd Segment Size": bwd_len_mean,

            # Active / Idle
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min,

            # Subflow
            "Subflow Fwd Packets": self.subflow_fwd_pkts,
            "Subflow Fwd Bytes": self.subflow_fwd_bytes,
            "Subflow Bwd Packets": self.subflow_bwd_pkts,
            "Subflow Bwd Bytes": self.subflow_bwd_bytes,

            # TCP connection 相關
            "Init_Win_bytes_forward": self.init_win_fwd,
            "Init_Win_bytes_backward": self.init_win_bwd,
            "Act_data_pkt_fwd": self.act_data_fwd,
            "Min_seg_size_forward": 0 if math.isinf(self.min_seg_size_fwd) else self.min_seg_size_fwd,
        }

        return feat

def extract_flow_features_73(pkts: list[Packet]):
    """
        從一組 scapy Packet list 中提取 flow 特徵
        Args:
            pkts(list[Packet]): list of scapy Packet
        Returns:
            feature_dict: dict of flow features
    """
    if not pkts:
        return None

    # 1. 找到第一個有 IP+TCP/UDP 的封包，當作 forward 基準
    base_src = base_sport = base_dst = base_dport = base_proto = None
    for pkt in pkts:
        if IP not in pkt and IPv6 not in pkt:
            continue
        ip = pkt[IP] if IP in pkt else pkt[IPv6]
        proto = ip.proto if IP in pkt else ip.nh
        base_src = ip.src
        base_dst = ip.dst
        if TCP in pkt:      # TCP
            base_sport = pkt[TCP].sport
            base_dport = pkt[TCP].dport
            base_proto = 6
            break
        elif UDP in pkt:   # UDP
            base_sport = pkt[UDP].sport
            base_dport = pkt[UDP].dport
            base_proto = 17
            break

    if base_src is None:
        # 都不是 TCP/UDP over IP，直接回空
        return None

    flow = FlowStats(base_src, base_sport, base_dst, base_dport, base_proto)

    # 2. 走過所有 pkts，依照與 base 的方向決定 fwd / bwd
    for pkt in pkts:
        if IP not in pkt and IPv6 not in pkt:
            continue
        ip = pkt[IP] if IP in pkt else pkt[IPv6]
        if TCP in pkt:
            proto = 6
        elif UDP in pkt:
            proto = 17
        else:
            continue

        # 只處理同一個 L4 協定（TCP or UDP）
        if proto != base_proto:
            continue

        # 嘗試取出 sport / dport
        sport = dport = None
        if proto == 6 and TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif proto == 17 and UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            continue

        # 判斷方向
        if ip.src == base_src and sport == base_sport and ip.dst == base_dst and dport == base_dport:
            direction = "fwd"
        elif ip.src == base_dst and sport == base_dport and ip.dst == base_src and dport == base_sport:
            direction = "bwd"
        else:
            # 出現奇怪的 tuple（理論上不應該）就先丟掉
            continue

        flow.update(pkt, direction)

    # 3. 最後直接產生這個 flow 的特徵
    return flow.to_features()

def get_feature_names_73():
    """
        取得特徵名稱列表
        Returns:
            feature_names: list of feature names
    """
    return [
        "Flow Duration",
        "Destination Port",
        "Source Port",
        "Protocol",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWR Flag Count",
        "ECE Flag Count",
        "Down/Up Ratio",
        "Average Packet Size",
        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "Subflow Fwd Packets",
        "Subflow Fwd Bytes",
        "Subflow Bwd Packets",
        "Subflow Bwd Bytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Act_data_pkt_fwd",
        "Min_seg_size_forward",
    ]