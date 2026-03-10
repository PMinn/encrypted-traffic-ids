from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Packet
from scapy.all import PacketList
import statistics
import math

ACK, SYN, FIN, RST = 0x10, 0x02, 0x01, 0x04


def get_ip(pkt):
    if IP in pkt:
        return pkt[IP]
    if IPv6 in pkt:
        return pkt[IPv6]
    return None


def get_4_tuple(ip, tcp):
    return (ip.src, tcp.sport, ip.dst, tcp.dport)


def tcp_payload_len(tcp):
    return len(bytes(tcp.payload)) if tcp.payload else 0


def list_to_4_tuple(k):
    return (k[2], k[3], k[0], k[1])


def is_three_way_handshake(pkts: PacketList) -> tuple[bool, int, int]:
    """
    Args:
        - pkts: PacketList，至少要有 3 個 TCP 封包才有可能是握手
    Returns:
        - is_handshake: 前三個 TCP 封包是否符合 SYN, SYN/ACK, ACK
        - handshake_time: 如果是握手，回傳 SYN 封包的 timestamp；如果不是，回傳 0
        - start_tcp_index_after_handshake: 若是，回傳 4（代表從第4個 TCP 封包開始才是握手後）
          若不是，回傳 0（代表從第一個 TCP 封包開始就算，因為沒有握手）
    """
    # 取前 3 個 TCP 封包
    first3 = []
    start_tcp_index_after_handshake = 0
    for i, pkt in enumerate(pkts):
        if TCP in pkt:
            ip = get_ip(pkt)
            if ip is None:
                continue
            first3.append((ip, pkt))
            if len(first3) == 3:
                start_tcp_index_after_handshake = i + 1  # 握手後的第一個 TCP 封包 index
                break

    if len(first3) < 3:
        return (False, 0, 0)  # 不足 3 個 TCP 封包，無法是握手

    (ip1, p1), (ip2, p2), (ip3, p3) = first3
    t1, t2, t3 = p1[TCP], p2[TCP], p3[TCP]

    # helper: flags check
    def has(t, flag):
        return (int(t.flags) & flag) != 0

    # 1) SYN: SYN=1, ACK=0
    cond1 = has(t1, SYN) and (not has(t1, ACK))

    # 2) SYN/ACK: SYN=1, ACK=1 且方向相反（src/dst 對調）
    # 方向相反： (src,sport,dst,dport) 對上第一包的反向
    dir1 = (ip1.src, t1.sport, ip1.dst, t1.dport)
    dir2 = (ip2.src, t2.sport, ip2.dst, t2.dport)
    cond2_dir = dir2 == (dir1[2], dir1[3], dir1[0], dir1[1])
    cond2 = has(t2, SYN) and has(t2, ACK) and cond2_dir

    # SYN/ACK 的 ack 應該是 SYN.seq + 1
    cond2_ack = t2.ack == ((t1.seq + 1) & 0xFFFFFFFF)

    # 3) ACK: ACK=1, SYN=0, FIN/RST=0，方向回到第一包方向
    dir3 = (ip3.src, t3.sport, ip3.dst, t3.dport)
    cond3_dir = dir3 == dir1
    cond3_flags = (
        has(t3, ACK) and (not has(t3, SYN)) and (int(t3.flags) & (FIN | RST)) == 0
    )
    cond3_ack = t3.ack == ((t2.seq + 1) & 0xFFFFFFFF)

    # payload 通常為 0（但少數情況握手後立刻 piggyback data，不強制）
    cond3 = cond3_dir and cond3_flags and cond3_ack and (tcp_payload_len(t3) == 0)

    ok = cond1 and cond2 and cond2_ack and cond3

    time = float(p3.time) - float(p1.time)

    return (ok, time if ok else 0, start_tcp_index_after_handshake if ok else 0)


def count_keepalive_and_ack(pkts: PacketList, ack_timeout_s: float = 2.0):
    """
    Returns (keepalive_probes_count, keepalive_ack_count)

    keepalive probe heuristic:
      - TCP, ACK set, no SYN/FIN/RST
      - payload_len == 0
      - seq == last_seen_seq_in_same_direction - 1

    keepalive ACK heuristic:
      - opposite direction packet
      - TCP, ACK set, no SYN/FIN/RST
      - payload_len == 0
      - ack == probe_seq + 1
      - arrives within ack_timeout_s after the probe
    """
    last_seq = {}  # dir_key -> last observed seq (best-effort baseline)
    pending_ack = (
        {}
    )  # dir_key_expected (reverse dir) -> (expected_ack_value, expire_time)

    keepalive_probes = 0
    keepalive_acks = 0

    for pkt in pkts:
        if TCP not in pkt:
            continue
        ip = get_ip(pkt)
        if ip is None:
            continue

        tcp = pkt[TCP]
        t = float(getattr(pkt, "time", 0.0))
        flags = int(tcp.flags)
        k = get_4_tuple(ip, tcp)

        # 先清掉過期 pending（單 flow 很小，直接掃）
        if pending_ack:
            expired = [kk for kk, (_, exp) in pending_ack.items() if t > exp]
            for kk in expired:
                pending_ack.pop(kk, None)

        payload_len = tcp_payload_len(tcp)

        # 檢查是否為 keep-alive ACK（先做，避免後面更新 last_seq 干擾）
        if (flags & ACK) and not (flags & (SYN | FIN | RST)) and payload_len == 0:
            pend = pending_ack.get(k)
            if pend is not None:
                expected_ack, exp_time = pend
                if t <= exp_time and tcp.ack == expected_ack:
                    keepalive_acks += 1
                    pending_ack.pop(k, None)
                    # keep-alive ACK 不更新 last_seq（避免污染狀態）
                    continue

        # 有資料的包，更新 last_seq（最可靠）
        if payload_len > 0:
            last_seq[k] = tcp.seq
            continue

        # 只考慮純 ACK（排除 SYN/FIN/RST）
        if not (flags & ACK) or (flags & (SYN | FIN | RST)):
            continue

        prev = last_seq.get(k)

        # keep-alive probe: seq == prev - 1
        if prev is not None and tcp.seq == ((prev - 1) & 0xFFFFFFFF):
            keepalive_probes += 1

            # 預期對向 ACK：ack == probe_seq + 1
            expected_ack = (tcp.seq + 1) & 0xFFFFFFFF
            k_rev = list_to_4_tuple(k)
            pending_ack[k_rev] = (expected_ack, t + ack_timeout_s)

            # probe 不更新 last_seq（它是 prev-1 的特例）
            continue

        # 其他純 ACK，當作 baseline 更新
        last_seq[k] = tcp.seq

    return keepalive_probes, keepalive_acks


class FlowStats:
    def __init__(self, src: str, sport: int, dst: str, dport: int, proto: int):
        # 5-tuple（forward 方向）
        self.src = src
        self.sport = sport
        self.dst = dst
        self.dport = dport
        self.proto = proto

        # 時間
        self.start: float = 0.0
        self.end: float = 0.0
        self.timestamps: list[float] = []

        # 封包長度與時間（fwd / bwd）
        self.fwd_pkt_len: list[int] = []
        self.bwd_pkt_len: list[int] = []
        self.all_pkt_len: list[int] = []
        self.fwd_times: list[float] = []
        self.bwd_times: list[float] = []

        # L4 長度（fwd / bwd）
        self.fwd_l4_len: list[int] = []
        self.bwd_l4_len: list[int] = []
        self.all_l4_len: list[int] = []

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

        # windows 相關
        self.win: list[int] = []
        self.win_fwd: list[int] = []
        self.win_bwd: list[int] = []

        # keep alive 相關
        self.keep_alive = 0
        self.keep_alive_ack = 0

        # handshake 相關
        self.handshake_time = 0.0

        self.act_data_fwd = 0
        self.min_seg_size_fwd = math.inf

    def update(self, pkt: Packet, direction: str) -> None:
        """
        每個封包進來更新
        Args:
            pkt(Packet): scapy Packet
            direction(str): "fwd" or "bwd"
        returns: None
        """
        ts = float(pkt.time)
        length = len(pkt)

        if self.start == 0.0:
            self.start = ts
        self.end = ts
        self.timestamps.append(ts)

        l4 = pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)

        self.all_pkt_len.append(length)
        self.all_l4_len.append(len(l4.payload) if l4 else 0)

        if direction == "fwd":
            self.fwd_pkt_len.append(length)
            self.fwd_l4_len.append(len(l4.payload) if l4 else 0)
            self.fwd_times.append(ts)
        else:
            self.bwd_pkt_len.append(length)
            self.bwd_l4_len.append(len(l4.payload) if l4 else 0)
            self.bwd_times.append(ts)

        # TCP flags / window / segment size
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & 0x01:
                self.fin += 1
            if flags & 0x02:
                self.syn += 1
            if flags & 0x04:
                self.rst += 1
            if flags & 0x08:
                if direction == "fwd":
                    self.fwd_psh += 1
                else:
                    self.bwd_psh += 1
            if flags & 0x10:
                self.ack += 1
            if flags & 0x20:
                if direction == "fwd":
                    self.fwd_urg += 1
                else:
                    self.bwd_urg += 1
            if flags & 0x40:
                self.ece += 1
            if flags & 0x80:
                self.cwr += 1

            self.win.append(pkt[TCP].window)

            if direction == "fwd":
                self.win_fwd.append(pkt[TCP].window)
                # min seg size forward 只看 forward
                seg_payload_size = len(pkt[TCP].payload)
                if seg_payload_size < self.min_seg_size_fwd:
                    self.min_seg_size_fwd = seg_payload_size
                if seg_payload_size > 0:
                    self.act_data_fwd += 1
            else:
                self.win_bwd.append(pkt[TCP].window)

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

    def set_keep_alive(self, probes: int, acks: int) -> None:
        self.keep_alive = probes
        self.keep_alive_ack = acks

    def set_handshake_time(self, handshake_time: float) -> None:
        self.handshake_time = handshake_time

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

    def _active_idle(
        self, threshold: float = 1.0
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """
        簡單版 active/idle 計算，預設為 1 秒
        Args:
            None
        returns: (active_mean, active_std, active_max, active_min, idle_mean, idle_std, idle_max, idle_min)
        """
        if len(self.timestamps) < 2:
            return (0, 0, 0, 0, 0, 0, 0, 0)
        times = sorted(self.timestamps)
        intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        active = [x for x in intervals if x < threshold]
        idle = [x for x in intervals if x >= threshold]

        def stats(arr: list[float]) -> tuple[float, float, float, float]:
            if not arr:
                return (0, 0, 0, 0)
            mean = statistics.mean(arr)
            std = statistics.pstdev(arr) if len(arr) > 1 else 0.0
            return (mean, std, max(arr), min(arr))

        return stats(active) + stats(idle)

    def to_features(self) -> dict[str, float]:
        """
        轉成特徵 dict
        Args:
            None
        returns: feature_dict
        """
        duration = max((self.end - self.start), 1e-6)

        total_fwd = len(self.fwd_pkt_len)
        total_bwd = len(self.bwd_pkt_len)
        total_pkts = total_fwd + total_bwd
        total_bytes = sum(self.all_pkt_len)

        # Flow IAT
        flow_iat_total, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = (
            self._iat(self.timestamps)
        )
        # Fwd IAT
        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = self._iat(
            self.fwd_times
        )
        # Bwd IAT
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = self._iat(
            self.bwd_times
        )

        # Active / Idle
        (
            active_mean,
            active_std,
            active_max,
            active_min,
            idle_mean,
            idle_std,
            idle_max,
            idle_min,
        ) = self._active_idle()

        # 封包長度統計
        def safe_stats(arr: list[int]) -> tuple[float, float, float, float]:
            if not arr:
                return (0.0, 0.0, 0.0, 0.0)
            mx = max(arr)
            mn = min(arr)
            mean = statistics.mean(arr)
            std = statistics.pstdev(arr) if len(arr) > 1 else 0.0
            return (mx, mn, mean, std)

        fwd_len_max, fwd_len_min, fwd_len_mean, fwd_len_std = safe_stats(
            self.fwd_pkt_len
        )
        bwd_len_max, bwd_len_min, bwd_len_mean, bwd_len_std = safe_stats(
            self.bwd_pkt_len
        )
        pkt_len_max, pkt_len_min, pkt_len_mean, pkt_len_std = safe_stats(
            self.all_pkt_len
        )
        fwd_l4_len_max, fwd_l4_len_min, fwd_l4_len_mean, fwd_l4_len_std = safe_stats(
            self.fwd_l4_len
        )
        bwd_l4_len_max, bwd_l4_len_min, bwd_l4_len_mean, bwd_l4_len_std = safe_stats(
            self.bwd_l4_len
        )
        l4_len_max, l4_len_min, l4_len_mean, l4_len_std = safe_stats(self.all_l4_len)
        win_max, win_min, win_mean, win_std = safe_stats(self.win)

        feat = {
            # Flow basic
            "Flow Duration": duration,
            "Destination Port": self.dport,
            "Source Port": self.sport,
            "Protocol": self.proto,
            "Service": (
                self.dport if self.dport < self.sport else self.sport
            ),  # 簡單用較小的 port 當作 service（不考慮非 TCP/UDP）
            "Handshake Time": self.handshake_time,
            # Fwd/Bwd 數量
            "Total Fwd Packets": total_fwd,
            "Total Backward Packets": total_bwd,
            # 封包長度統計
            #   全部
            "Total Packet Length": total_bytes,
            "Min Packet Length": pkt_len_min,
            "Max Packet Length": pkt_len_max,
            "Packet Length Mean": pkt_len_mean,
            "Packet Length Std": pkt_len_std,
            "Packet Length Variance": pkt_len_std**2,
            #   forward
            "Total Length of Fwd Packets": sum(self.fwd_pkt_len),
            "Fwd Packet Length Min": fwd_len_min,
            "Fwd Packet Length Max": fwd_len_max,
            "Fwd Packet Length Mean": fwd_len_mean,
            "Fwd Packet Length Std": fwd_len_std,
            "Fwd Packet Length Variance": fwd_len_std**2,
            #   backward
            "Total Length of Bwd Packets": sum(self.bwd_pkt_len),
            "Bwd Packet Length Min": bwd_len_min,
            "Bwd Packet Length Max": bwd_len_max,
            "Bwd Packet Length Mean": bwd_len_mean,
            "Bwd Packet Length Std": bwd_len_std,
            "Bwd Packet Length Variance": bwd_len_std**2,
            # L4 長度統計
            #   全部
            "Total L4 Length": sum(self.all_l4_len),
            "L4 Length Min": l4_len_min,
            "L4 Length Max": l4_len_max,
            "L4 Length Mean": l4_len_mean,
            "L4 Length Std": l4_len_std,
            "L4 Length Variance": l4_len_std**2,
            #   forward
            "Total L4 Length of Fwd Packets": sum(self.fwd_l4_len),
            "Fwd L4 Length Min": fwd_l4_len_min,
            "Fwd L4 Length Max": fwd_l4_len_max,
            "Fwd L4 Length Mean": fwd_l4_len_mean,
            "Fwd L4 Length Std": fwd_l4_len_std,
            "Fwd L4 Length Variance": fwd_l4_len_std**2,
            #   backward
            "Total L4 Length of Bwd Packets": sum(self.bwd_l4_len),
            "Bwd L4 Length Min": bwd_l4_len_min,
            "Bwd L4 Length Max": bwd_l4_len_max,
            "Bwd L4 Length Mean": bwd_l4_len_mean,
            "Bwd L4 Length Std": bwd_l4_len_std,
            "Bwd L4 Length Variance": bwd_l4_len_std**2,
            # Flow bytes/packet rate
            "Flow Bytes/s": total_bytes / duration,
            "Flow Packets/s": total_pkts / duration,
            # Flow IAT
            "Flow IAT Total": flow_iat_total,
            "Flow IAT Min": flow_iat_min,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Variance": flow_iat_std**2,
            # Fwd IAT
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Variance": fwd_iat_std**2,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            # Bwd IAT
            "Bwd IAT Total": bwd_iat_total,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Variance": bwd_iat_std**2,
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
            # Active / Idle
            "Active Mean": active_mean,
            "Active Std": active_std,
            "Active Variance": active_std**2 if active_std else 0,
            "Active Max": active_max,
            "Active Min": active_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Variance": idle_std**2 if idle_std else 0,
            "Idle Max": idle_max,
            "Idle Min": idle_min,
            # TCP connection 相關
            #   window
            "Init_Win_bytes_forward": self.win_fwd[0] if len(self.win_fwd) > 0 else 0,
            "Init_Win_bytes_backward": self.win_bwd[0] if len(self.win_bwd) > 0 else 0,
            "Win Total": sum(self.win),
            "Win Min": win_min,
            "Win Max": win_max,
            "Win Mean": win_mean,
            "Win Std": win_std,
            "Win Variance": win_std**2 if win_std else 0,
            #   keep alive
            "Keep Alive Count": self.keep_alive,
            "Keep Alive ACK Count": self.keep_alive_ack,
            #   其他
            "Act_data_pkt_fwd": self.act_data_fwd,
            "Min_seg_size_forward": (
                0 if math.isinf(self.min_seg_size_fwd) else self.min_seg_size_fwd
            ),
        }

        return feat


def extract_flow_features_103(pkts: PacketList) -> dict[str, float] | None:
    """
    從一組 scapy Packet list 中提取 flow 特徵
    Args:
        pkts(PacketList) -> dict | None: list of scapy Packet
    Returns:
        feature_dict: dict of flow features
    """
    if not pkts:
        return None

    # 1. 找到第一個有 IP+TCP/UDP 的封包，當作 forward 基準
    for pkt in pkts:
        if IP not in pkt and IPv6 not in pkt:
            continue
        ip = pkt[IP] if IP in pkt else pkt[IPv6]
        base_src = ip.src
        base_dst = ip.dst
        if TCP in pkt:  # TCP
            base_sport = int(pkt[TCP].sport)
            base_dport = int(pkt[TCP].dport)
            base_proto = 6
            break
        elif UDP in pkt:  # UDP
            base_sport = int(pkt[UDP].sport)
            base_dport = int(pkt[UDP].dport)
            base_proto = 17
            break
        else:
            continue

    if base_src is None:
        # 都不是 TCP/UDP over IP，直接回空
        return None

    flow = FlowStats(base_src, base_sport, base_dst, base_dport, base_proto)

    keepalive_probes, keepalive_acks = count_keepalive_and_ack(pkts)
    flow.set_keep_alive(keepalive_probes, keepalive_acks)

    # 檢查是否有三次握手，若有就從握手後的封包開始算
    has_handshake, handshake_time, handshake_offset = is_three_way_handshake(pkts)
    if has_handshake:
        flow.set_handshake_time(handshake_time)
        pkts = pkts[handshake_offset:]

    # 2. 走過所有 pkts，依照與 base 的方向決定 fwd / bwd
    for pkt in pkts:
        if IP not in pkt and IPv6 not in pkt:
            continue
        ip = pkt[IP] if IP in pkt else pkt[IPv6]

        # 嘗試取出 sport / dport
        sport = dport = None
        if base_proto == 6 and TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif base_proto == 17 and UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            continue

        # 判斷方向
        if (
            ip.src == base_src
            and sport == base_sport
            and ip.dst == base_dst
            and dport == base_dport
        ):
            direction = "fwd"
        elif (
            ip.src == base_dst
            and sport == base_dport
            and ip.dst == base_src
            and dport == base_sport
        ):
            direction = "bwd"
        else:
            # 出現奇怪的 tuple（理論上不應該）就先丟掉
            continue

        flow.update(pkt, direction)

    # 3. 最後直接產生這個 flow 的特徵
    return flow.to_features()


def get_feature_names_103() -> list[str]:
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
        "Service",
        "Handshake Time",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Packet Length",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Min",
        "Fwd Packet Length Max",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Fwd Packet Length Variance",
        "Total Length of Bwd Packets",
        "Bwd Packet Length Min",
        "Bwd Packet Length Max",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Bwd Packet Length Variance",
        "Total L4 Length",
        "L4 Length Min",
        "L4 Length Max",
        "L4 Length Mean",
        "L4 Length Std",
        "L4 Length Variance",
        "Total L4 Length of Fwd Packets",
        "Fwd L4 Length Min",
        "Fwd L4 Length Max",
        "Fwd L4 Length Mean",
        "Fwd L4 Length Std",
        "Fwd L4 Length Variance",
        "Total L4 Length of Bwd Packets",
        "Bwd L4 Length Min",
        "Bwd L4 Length Max",
        "Bwd L4 Length Mean",
        "Bwd L4 Length Std",
        "Bwd L4 Length Variance",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Total",
        "Flow IAT Min",
        "Flow IAT Max",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Variance",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Variance",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Variance",
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
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWR Flag Count",
        "ECE Flag Count",
        "Down/Up Ratio",
        "Active Mean",
        "Active Std",
        "Active Variance",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Variance",
        "Idle Max",
        "Idle Min",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Win Total",
        "Win Min",
        "Win Max",
        "Win Mean",
        "Win Std",
        "Win Variance",
        "Keep Alive Count",
        "Keep Alive ACK Count",
        "Act_data_pkt_fwd",
        "Min_seg_size_forward",
    ]


def get_log_scale_features_name_103() -> list[str]:
    return [
    ]


def get_std_scale_features_name_103() -> list[str]:
    return [
    ]
