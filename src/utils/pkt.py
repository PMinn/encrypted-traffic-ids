from scapy.all import TCP, IP, IPv6

def is_tcp(pkt):
    return TCP in pkt and (IP in pkt or IPv6 in pkt)

def get_5tuple(pkt):
    if IP in pkt:
        src = pkt[IP].src; dst = pkt[IP].dst; proto = 'IP'
    else:
        src = pkt[IPv6].src; dst = pkt[IPv6].dst; proto = 'IPv6'
    sport = pkt[TCP].sport
    dport = pkt[TCP].dport
    # normalize to bidirectional session key (order-independent)
    a = (src, sport); b = (dst, dport)
    if a <= b:
        fwd = (src, sport); bwd = (dst, dport)
    else:
        fwd = (dst, dport); bwd = (src, sport)
    key = (proto, fwd[0], fwd[1], bwd[0], bwd[1], 'TCP')
    return key, fwd, bwd

def tcp_flags(pkt):
    f = pkt[TCP].flags
    return {
        'SYN': int(f & 0x02 != 0),
        'ACK': int(f & 0x10 != 0),
        'PSH': int(f & 0x08 != 0),
        'RST': int(f & 0x04 != 0),
        'FIN': int(f & 0x01 != 0),
        'URG': int(f & 0x20 != 0),
        'ECE': int(f & 0x40 != 0),
        'CWR': int(f & 0x80 != 0),
    }

def l4_payload_len(pkt):
    # scapy len(payload) works for TCP
    return int(len(bytes(pkt[TCP].payload)))

def pkt_len(pkt):
    return int(len(bytes(pkt)))

# simple TCP keep-alive heuristics:
# - pure ACK, payload 0, and seq == last_ack-1 (cannot always access last_ack)
#   Here we approximate using: pure ACK with zero payload and small delta time (> ~60s often),
#   but in short windows N we just count pure ACK with zero payload as keepalive-ish.
def is_keepalive_like(flags, payload_len):
    pure_ack = (flags['ACK'] == 1 and flags['PSH']==0 and flags['RST']==0 and
                flags['SYN']==0 and flags['FIN']==0)
    return pure_ack and payload_len == 0