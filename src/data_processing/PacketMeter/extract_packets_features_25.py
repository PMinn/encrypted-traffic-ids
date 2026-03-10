from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Packet
from scapy.all import PacketList
import statistics
import math

ACK, SYN, FIN, RST = 0x10, 0x02, 0x01, 0x04


# ---------- helpers (沿用你的風格) ----------
def get_ip(pkt: Packet):
    if IP in pkt:
        return pkt[IP]
    if IPv6 in pkt:
        return pkt[IPv6]
    return None


def tcp_payload_len(tcp: TCP) -> int:
    return len(bytes(tcp.payload)) if tcp.payload else 0


def list_to_4_tuple(k):
    return (k[2], k[3], k[0], k[1])


def get_4_tuple(ip, tcp: TCP):
    return (ip.src, tcp.sport, ip.dst, tcp.dport)


def safe_mean(arr: list[float]) -> float:
    return float(statistics.mean(arr)) if arr else 0.0


def safe_var(arr: list[float]) -> float:
    # 論文寫 variance；這裡用 population variance（pstdev^2）
    if len(arr) < 2:
        return 0.0
    return float(statistics.pvariance(arr))


def safe_min(arr: list[float]) -> float:
    return float(min(arr)) if arr else 0.0


def safe_max(arr: list[float]) -> float:
    return float(max(arr)) if arr else 0.0


def safe_sum(arr: list[float]) -> float:
    return float(sum(arr)) if arr else 0.0


# ---------- incremental keep-alive 判斷（每封包回傳 bool） ----------
class KeepAliveDetector:
    """
    依你 flow 那版的 heuristic，改成「每封包」判斷：
      keep-alive probe:
        - TCP, ACK set, no SYN/FIN/RST
        - payload_len == 0
        - seq == last_seen_seq_in_same_direction - 1
      keep-alive ACK:
        - opposite direction packet
        - TCP, ACK set, no SYN/FIN/RST
        - payload_len == 0
        - ack == probe_seq + 1
        - arrives within ack_timeout_s after the probe
    """
    def __init__(self, ack_timeout_s: float = 2.0):
        self.ack_timeout_s = float(ack_timeout_s)
        self.last_seq: dict[tuple, int] = {}  # dir_key -> last observed seq
        self.pending_ack: dict[tuple, tuple[int, float]] = {}  # dir_key -> (expected_ack, expire_time)

    def step(self, pkt: Packet, k_dir: tuple) -> tuple[bool, bool]:
        """
        Returns: (is_keepalive_probe, is_keepalive_ack)
        """
        if TCP not in pkt:
            return (False, False)

        tcp = pkt[TCP]
        t = float(getattr(pkt, "time", 0.0))
        flags = int(tcp.flags)

        # 清掉過期 pending
        if self.pending_ack:
            expired = [kk for kk, (_, exp) in self.pending_ack.items() if t > exp]
            for kk in expired:
                self.pending_ack.pop(kk, None)

        payload_len = tcp_payload_len(tcp)

        # 先檢查 keep-alive ACK
        if (flags & ACK) and not (flags & (SYN | FIN | RST)) and payload_len == 0:
            pend = self.pending_ack.get(k_dir)
            if pend is not None:
                expected_ack, exp_time = pend
                if t <= exp_time and tcp.ack == expected_ack:
                    self.pending_ack.pop(k_dir, None)
                    return (False, True)

        # 有 payload 的包，更新 last_seq（比較可靠）
        if payload_len > 0:
            self.last_seq[k_dir] = int(tcp.seq)
            return (False, False)

        # 非純 ACK 不管
        if not (flags & ACK) or (flags & (SYN | FIN | RST)):
            return (False, False)

        prev = self.last_seq.get(k_dir)
        if prev is not None and int(tcp.seq) == ((prev - 1) & 0xFFFFFFFF):
            # keep-alive probe
            expected_ack = (int(tcp.seq) + 1) & 0xFFFFFFFF
            k_rev = list_to_4_tuple(k_dir)
            self.pending_ack[k_rev] = (expected_ack, t + self.ack_timeout_s)
            return (True, False)

        # 其他純 ACK 當 baseline 更新
        self.last_seq[k_dir] = int(tcp.seq)
        return (False, False)


# ---------- syn-flood（session-level flag，給 PP feature 用） ----------
def is_syn_flood_session(
    pkts: PacketList,
    base_src: str,
    base_sport: int,
    base_dst: str,
    base_dport: int,
    syn_threshold: int = 3,
) -> int:
    """
    PP 的 isSynFlood 是「current session 是否發生 syn flood」(0/1)。
    論文 Table 3 沒把判準寫死，這裡給一個可調 heuristic：
      - 在已抽出的 pkts 中，forward direction 的 SYN(且非ACK) 數量 >= syn_threshold
        且 沒看到完整 three-way handshake（簡化：沒看到 forward ACK 接在 SYN/ACK 後）
    你要更嚴格可自己改規則。
    """
    syn_fwd = 0
    saw_synack = False
    saw_final_ack = False

    for pkt in pkts:
        if TCP not in pkt:
            continue
        ip = get_ip(pkt)
        if ip is None:
            continue
        tcp = pkt[TCP]
        flags = int(tcp.flags)

        fwd = (ip.src == base_src and ip.dst == base_dst and int(tcp.sport) == base_sport and int(tcp.dport) == base_dport)
        bwd = (ip.src == base_dst and ip.dst == base_src and int(tcp.sport) == base_dport and int(tcp.dport) == base_sport)

        if fwd and (flags & SYN) and not (flags & ACK):
            syn_fwd += 1
        if bwd and (flags & SYN) and (flags & ACK):
            saw_synack = True
        if fwd and (flags & ACK) and not (flags & SYN) and saw_synack:
            saw_final_ack = True

    if syn_fwd >= syn_threshold and not saw_final_ack:
        return 1
    return 0


# ---------- PP(25) extractor ----------
PP25_FEATURE_NAMES = [
    # Packet size (4)
    "pkt_fwd_avg", "pkt_bwd_avg", "pkt_fwd_var", "pkt_bwd_var",
    # Time / IAT (10)
    "iat_fwd_avg", "iat_bwd_avg", "iat_fwd_var", "iat_bwd_var",
    "iat_fwd_min", "iat_bwd_min", "iat_fwd_max", "iat_bwd_max",
    "iat_fwd_sum", "iat_bwd_sum",
    # Window size (6)
    "win_fwd_avg", "win_bwd_avg", "win_fwd_var", "win_bwd_var",
    "win_fwd_min", "win_bwd_min",
    # TCP flags (5)
    "fwd_isRST", "bwd_isRST", "isKeepAlive", "isKeepAliveACK", "isSynFlood",
]


def extract_packet_pp_features_25(
    pkts: PacketList,
    N: int = 8,
    keepalive_ack_timeout_s: float = 2.0,
    syn_threshold: int = 3,
) -> list[list[float]] | None:
    """
    產生 PP matrix: shape (N, 25)
    - 第 i 列：前 i 個 extracted packets（1..i）的「prefix 統計」+「current packet flags」
    - 不足 N 列補 0
    """
    if not pkts:
        return None

    # 1) 找 base（forward 方向基準）：第一個 IP + TCP/UDP
    base_src = base_dst = None
    base_sport = base_dport = None
    base_proto = None

    for pkt in pkts:
        ip = get_ip(pkt)
        if ip is None:
            continue
        if TCP in pkt:
            base_src, base_dst = ip.src, ip.dst
            base_sport, base_dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
            base_proto = 6
            break
        if UDP in pkt:
            base_src, base_dst = ip.src, ip.dst
            base_sport, base_dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
            base_proto = 17
            break

    if base_src is None:
        return None

    # 2) 先取前 N 個「同一 flow 方向可判斷」的封包（extracted packets）
    extracted: list[Packet] = []
    extracted_dirs: list[str] = []  # "fwd"/"bwd"
    extracted_k: list[tuple] = []   # 4-tuple (src,sport,dst,dport) for TCP keepalive tracking

    for pkt in pkts:
        if len(extracted) >= N:
            break
        ip = get_ip(pkt)
        if ip is None:
            continue

        # 取 sport/dport
        if base_proto == 6 and TCP in pkt:
            sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
        elif base_proto == 17 and UDP in pkt:
            sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)
        else:
            continue

        # 判斷方向
        if (ip.src == base_src and ip.dst == base_dst and sport == base_sport and dport == base_dport):
            direction = "fwd"
        elif (ip.src == base_dst and ip.dst == base_src and sport == base_dport and dport == base_sport):
            direction = "bwd"
        else:
            continue

        extracted.append(pkt)
        extracted_dirs.append(direction)

        if TCP in pkt:
            extracted_k.append(get_4_tuple(ip, pkt[TCP]))
        else:
            extracted_k.append(())  # UDP 沒 keepalive key

    # 如果 extracted 為空，回 None
    if not extracted:
        return None

    # 3) session-level syn flood flag（整段 shared）
    syn_flood = is_syn_flood_session(
        PacketList(extracted),
        base_src, base_sport, base_dst, base_dport,
        syn_threshold=syn_threshold,
    )

    # 4) incremental stats containers
    fwd_pkt_sizes: list[float] = []
    bwd_pkt_sizes: list[float] = []

    # IAT：按方向分開做（只用該方向的相鄰封包）
    fwd_times: list[float] = []
    bwd_times: list[float] = []
    fwd_iats: list[float] = []
    bwd_iats: list[float] = []

    # Window：TCP 才有；UDP 就不加
    fwd_wins: list[float] = []
    bwd_wins: list[float] = []

    # keep-alive per-packet detector
    ka = KeepAliveDetector(ack_timeout_s=keepalive_ack_timeout_s)

    # 5) build PP matrix rows
    pp: list[list[float]] = []

    for i, pkt in enumerate(extracted, start=1):
        direction = extracted_dirs[i - 1]
        ts = float(getattr(pkt, "time", 0.0))
        plen = float(len(pkt))

        # 更新 packet size
        if direction == "fwd":
            fwd_pkt_sizes.append(plen)
        else:
            bwd_pkt_sizes.append(plen)

        # 更新 IAT（同方向相鄰）
        if direction == "fwd":
            if fwd_times:
                fwd_iats.append(ts - fwd_times[-1])
            fwd_times.append(ts)
        else:
            if bwd_times:
                bwd_iats.append(ts - bwd_times[-1])
            bwd_times.append(ts)

        # 更新 window（TCP 才有）
        if TCP in pkt:
            win = float(int(pkt[TCP].window))
            if direction == "fwd":
                fwd_wins.append(win)
            else:
                bwd_wins.append(win)

        # current packet flags (0/1)
        fwd_isrst = 0.0
        bwd_isrst = 0.0
        is_keepalive = 0.0
        is_keepalive_ack = 0.0

        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            if direction == "fwd":
                fwd_isrst = 1.0 if (flags & RST) else 0.0
            else:
                bwd_isrst = 1.0 if (flags & RST) else 0.0

            # keepalive 判斷用 4-tuple key
            k_dir = extracted_k[i - 1]
            if k_dir:
                ka_probe, ka_ack = ka.step(pkt, k_dir)
                is_keepalive = 1.0 if ka_probe else 0.0
                is_keepalive_ack = 1.0 if ka_ack else 0.0

        row = [
            # Packet size
            safe_mean(fwd_pkt_sizes),
            safe_mean(bwd_pkt_sizes),
            safe_var(fwd_pkt_sizes),
            safe_var(bwd_pkt_sizes),

            # Time / IAT
            safe_mean(fwd_iats),
            safe_mean(bwd_iats),
            safe_var(fwd_iats),
            safe_var(bwd_iats),
            safe_min(fwd_iats),
            safe_min(bwd_iats),
            safe_max(fwd_iats),
            safe_max(bwd_iats),
            safe_sum(fwd_iats),
            safe_sum(bwd_iats),

            # Window size
            safe_mean(fwd_wins),
            safe_mean(bwd_wins),
            safe_var(fwd_wins),
            safe_var(bwd_wins),
            safe_min(fwd_wins),
            safe_min(bwd_wins),

            # TCP flags (current packet)
            fwd_isrst,
            bwd_isrst,
            is_keepalive,
            is_keepalive_ack,
            float(syn_flood),
        ]
        pp.append(row)

    # 6) padding to (N, 25)
    while len(pp) < N:
        pp.append([0.0] * 25)

    return pp


# 你如果要同時拿到 feature names：
# def extract_pp25_with_names(pkts: PacketList, N: int = 8):
#     mat = extract_packet_pp_features_25(pkts, N=N)
#     return PP25_FEATURE_NAMES, mat