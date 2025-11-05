import glob
from scapy.all import rdpcap, TCP, IP, IPv6
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from math import sqrt
from ..utils.pkt import is_tcp, get_5tuple, tcp_flags, l4_payload_len, pkt_len, is_keepalive_like

# ----------------------------
# Online stats helper (per-direction)
# ----------------------------
class OnlineStats:
    """Welford's algorithm for mean/var/min/max/sum/count."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self._min = None
        self._max = None
        self._sum = 0.0
    def add(self, x):
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        self._sum += x
        self._min = x if self._min is None else min(self._min, x)
        self._max = x if self._max is None else max(self._max, x)
    @property
    def var(self):
        return float(self.M2 / self.n) if self.n > 0 else 0.0
    @property
    def min(self):
        return float(self._min) if self._min is not None else 0.0
    @property
    def max(self):
        return float(self._max) if self._max is not None else 0.0
    @property
    def sum(self):
        return float(self._sum)

# ----------------------------
# Core extraction
# ----------------------------
def extract_ss_pp_from_pcap(pcap_path, N=8):
    """
    Returns:
        ss_df: DataFrame of SS features (one row per session, 26 cols + key)
        pp_dict: dict(session_key -> PP matrix np.array shape (N,25))
    """
    pkts = rdpcap(pcap_path)

    # group packets into bidirectional TCP sessions
    sessions = defaultdict(list)
    for pkt in pkts:
        if not is_tcp(pkt): 
            continue
        key, fwd, bwd = get_5tuple(pkt)
        ts = float(pkt.time)
        flags = tcp_flags(pkt)
        plen = pkt_len(pkt)
        paylen = l4_payload_len(pkt)
        win = int(pkt[TCP].window) if hasattr(pkt[TCP], "window") else 0
        # direction relative to normalized fwd endpoint
        src = pkt[IP].src if IP in pkt else pkt[IPv6].src
        sport = pkt[TCP].sport
        direction = 'fwd' if (src, sport) == fwd else 'bwd'
        print({
            'ts': ts,
            'dir': direction,
            'pkt_len': plen,
            'pay_len': paylen,
            'win': win,
            'flags': flags,
            'sport': pkt[TCP].sport,
            'dport': pkt[TCP].dport,
        })
        sessions[key].append({
            'ts': ts,
            'dir': direction,
            'pkt_len': plen,
            'pay_len': paylen,
            'win': win,
            'flags': flags,
            'sport': pkt[TCP].sport,
            'dport': pkt[TCP].dport,
        })

    # prepare outputs
    ss_rows = []
    pp_dict = {}

    for key, plist in sessions.items():
        # sort by timestamp
        plist.sort(key=lambda x: x['ts'])

        # take first N packets (extracted packets)
        extracted = plist[:N]
        if len(extracted) == 0:
            continue

        # ---------- SS 26 features ----------
        # Packet size stats (all directions)
        pkt_stats = OnlineStats()
        # L4 payload stats
        pay_stats = OnlineStats()
        # Time IAT overall (between consecutive packets in the session)
        iat_vals = []
        for i, rec in enumerate(extracted):
            pkt_stats.add(rec['pkt_len'])
            pay_stats.add(rec['pay_len'])
            if i > 0:
                iat_vals.append(extracted[i]['ts'] - extracted[i-1]['ts'])
        # iAT stats overall
        iat_avg = np.mean(iat_vals) if iat_vals else 0.0
        iat_min = np.min(iat_vals) if iat_vals else 0.0
        iat_max = np.max(iat_vals) if iat_vals else 0.0
        iat_sum = float(np.sum(iat_vals)) if iat_vals else 0.0

        # TCP window stats (all packets with a window field)
        win_stats = OnlineStats()
        for rec in extracted:
            win_stats.add(rec['win'])

        # TCP flags counts
        num_rst = sum(rec['flags']['RST'] for rec in extracted)
        num_psh = sum(rec['flags']['PSH'] for rec in extracted)

        # keep-alive & keep-alive ACK (heuristic)
        num_keepalive = 0
        num_keepalive_ack = 0
        for rec in extracted:
            ka = is_keepalive_like(rec['flags'], rec['pay_len'])
            if ka:
                # mark one as keepalive and one as ack is ambiguous in short windows;
                # here: if ACK only → count as keepalive_ack; also count keepalive.
                num_keepalive += 1
                num_keepalive_ack += 1

        # SYN flood indicator (within first N): many SYNs without seeing SYN-ACK
        syn_count = sum(rec['flags']['SYN'] for rec in extracted)
        synack_seen = any(rec['flags']['SYN'] and rec['flags']['ACK'] for rec in extracted)
        is_sync_flood = 1 if (syn_count >= 3 and not synack_seen) else 0

        # Service type (min port)
        first = extracted[0]
        service_type = int(min(first['sport'], first['dport']))

        # iRTT: time from first SYN to final ACK of 3WHS (SYN -> SYN/ACK -> ACK)
        t_syn = None; t_synack = None; t_ack = None
        for rec in extracted:
            f = rec['flags']
            if t_syn is None and f['SYN'] == 1 and f['ACK'] == 0:
                t_syn = rec['ts']
            elif t_syn is not None and t_synack is None and f['SYN']==1 and f['ACK']==1:
                t_synack = rec['ts']
            elif t_syn is not None and t_synack is not None and t_ack is None and f['ACK']==1 and f['SYN']==0:
                t_ack = rec['ts']; break
        iRTT = (t_ack - t_syn) if (t_syn is not None and t_ack is not None) else 0.0

        # connection time: first to Nth packet timestamp delta
        conn_time = extracted[-1]['ts'] - extracted[0]['ts'] if len(extracted) > 1 else 0.0

        # SS vector (26):
        ss = {
            # packet size
            'ss_pkt_avg': pkt_stats.mean,
            'ss_pkt_var': pkt_stats.var,
            'ss_pkt_min': pkt_stats.min,
            'ss_pkt_max': pkt_stats.max,   # note: 論文表格有個 "sum" 疑似排版錯，這裡用 max
            # L4 payload
            'ss_l4_avg': pay_stats.mean,
            'ss_l4_var': pay_stats.var,
            'ss_l4_min': pay_stats.min,
            'ss_l4_max': pay_stats.max,
            'ss_l4_sum': pay_stats.sum,
            # Time (IAT overall)
            'ss_time_avg_iat': iat_avg,
            'ss_time_min_iat': iat_min,
            'ss_time_max_iat': iat_max,
            'ss_time_sum_iat': iat_sum,
            'ss_time_iRTT': iRTT,
            'ss_time_conn': conn_time,
            # Window
            'ss_win_avg': win_stats.mean,
            'ss_win_var': win_stats.var,
            'ss_win_min': win_stats.min,
            'ss_win_max': win_stats.max,
            'ss_win_sum': win_stats.sum,
            # TCP flags
            'ss_flags_num_rst': int(num_rst),
            'ss_flags_num_psh': int(num_psh),
            'ss_flags_num_keepalive': int(num_keepalive),
            'ss_flags_num_keepalive_ack': int(num_keepalive_ack),
            'ss_flags_num_syncflood': int(is_sync_flood),
            # Port service
            'ss_service_type': int(service_type),
        }

        # ---------- PP N x 25 ----------
        # We build cumulative, directional stats up to each i (1..N)
        # Per-direction online stats
        ps_fwd = OnlineStats(); ps_bwd = OnlineStats()            # packet size
        win_fwd = OnlineStats(); win_bwd = OnlineStats()          # window size
        # IAT per direction (need last ts per dir)
        last_ts = {'fwd': None, 'bwd': None}
        iat_fwd_vals = OnlineStats(); iat_bwd_vals = OnlineStats()

        # pre-allocate PP with zeros
        PP = np.zeros((N, 25), dtype=float)

        for i in range(N):
            if i < len(extracted):
                rec = extracted[i]
                # update packet size stats (dir)
                if rec['dir'] == 'fwd':
                    ps_fwd.add(rec['pkt_len'])
                    win_fwd.add(rec['win'])
                else:
                    ps_bwd.add(rec['pkt_len'])
                    win_bwd.add(rec['win'])

                # update IAT per direction
                dt = None
                if last_ts[rec['dir']] is None:
                    dt = None
                else:
                    dt = rec['ts'] - last_ts[rec['dir']]
                    if dt < 0: dt = 0.0
                last_ts[rec['dir']] = rec['ts']

                if dt is not None:
                    if rec['dir'] == 'fwd':
                        iat_fwd_vals.add(dt)
                    else:
                        iat_bwd_vals.add(dt)

                # TCP flags for this packet/session (binary indicators in PP row i):
                f = rec['flags']
                isRST_fwd = 1 if (rec['dir']=='fwd' and f['RST']==0) else 0  # paper uses 0 when has RST, 1 otherwise
                isRST_bwd = 1 if (rec['dir']=='bwd' and f['RST']==0) else 0
                # If direction differs, the "other side" row should reflect "1" (no RST) by definition?
                # To align with spec: if current packet has RST in fwd, then fwd_isRST=0 else 1. For bwd we set based on current packet when dir==bwd; otherwise keep previous (implicitly 1).
                # Here we compute per-row solely from current packet; if not in that dir, set to 1 (no RST) by default:
                if rec['dir'] == 'fwd':
                    isRST_bwd = 1
                else:
                    isRST_fwd = 1

                isKeepAlive = 1 if is_keepalive_like(f, rec['pay_len']) else 0
                isKeepAliveAck = 1 if isKeepAlive else 0
                # session-level syn flood indicator (same across rows)
                isSynFlood = is_sync_flood

                # assemble row i
                row = [
                    # Packet size (4)
                    ps_fwd.mean, ps_bwd.mean, ps_fwd.var, ps_bwd.var,
                    # Time IAT (10)
                    iat_fwd_vals.mean, iat_bwd_vals.mean,
                    iat_fwd_vals.var,  iat_bwd_vals.var,
                    iat_fwd_vals.min,  iat_bwd_vals.min,
                    iat_fwd_vals.max,  iat_bwd_vals.max,
                    iat_fwd_vals.sum,  iat_bwd_vals.sum,
                    # Window (6)
                    win_fwd.mean, win_bwd.mean,
                    win_fwd.var,  win_bwd.var,
                    win_fwd.min,  win_bwd.min,
                    # TCP flags (5)
                    float(isRST_fwd), float(isRST_bwd),
                    float(isKeepAlive), float(isKeepAliveAck),
                    float(isSynFlood),
                ]
                PP[i, :] = np.array(row, dtype=float)
            else:
                # zero padding already fine
                pass

        # store
        ss_row = {'key': key}
        ss_row.update(ss)
        ss_rows.append(ss_row)
        pp_dict[key] = PP

    ss_df = pd.DataFrame(ss_rows)
    return ss_df, pp_dict


# if __name__ == "__main__":
#     # traffic_type = 'TCP'
#     # pcapsPath = glob.glob(f'/sdc1/ytlindata/TON_IoT/attack_filter/Benign/split_*/*{traffic_type}*')
#     pcapPath = '/sdc1/ytlindata/TON_IoT/attack_filter/Benign/split_1/normal_1.pcap.TCP_13-35-146-24_443_192-168-1-190_46933.pcap'

#     ss_df, pp_dict = extract_ss_pp_from_pcap(pcapPath, N=5)

#     # save SS
#     ss_df.to_csv("ss_features.csv", index=False)

#     # save PP (npz of key -> matrix)
#     # keys must be strings for np.savez
#     np.savez("pp_matrices.npz", **{str(k): v for k, v in pp_dict.items()})

#     print(f"[OK] SS saved to ss_features.csv (rows={len(ss_df)})")
#     print(f"[OK] PP saved to pp_matrices.npz (sessions={len(pp_dict)}, each shape=(5,25))")
