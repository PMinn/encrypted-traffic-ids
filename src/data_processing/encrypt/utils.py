from scapy.all import TCP, UDP, IP, IPv6

def fix_checksums_and_lengths(pkt):
    """
    Let scapy recompute lengths/checksums by deleting the right fields.
    """
    if IP in pkt:
        del pkt[IP].len
        del pkt[IP].chksum
    if IPv6 in pkt:
        # IPv6 has no header checksum; payload length can be recalculated
        if hasattr(pkt[IPv6], "plen"):
            del pkt[IPv6].plen

    if TCP in pkt:
        del pkt[TCP].chksum
    if UDP in pkt:
        del pkt[UDP].len
        del pkt[UDP].chksum
        
    if hasattr(pkt, "len"): del pkt.len
    if hasattr(pkt, "chksum"): del pkt.chksum