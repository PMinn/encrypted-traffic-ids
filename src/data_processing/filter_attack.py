import math
from typing import Callable
from pathlib import Path
import logging
from scapy.all import rdpcap
from scapy.packet import Packet
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

logger = logging.getLogger("filter_attack")


def pre_filter(
    attack_dict_type: dict[tuple[str, str, str, str, str], list[int]], pcap: Path
) -> bool:
    """
    使用檔名初次過濾攻擊流量，優先剔除不可能包含攻擊的 pcap 檔案，減少後續處理負擔
    Args:
        attack_dict_type (dict[tuple[str, str, str, str, str], list[int]]): 攻擊字典
        pcap (Path): pcap 檔案路徑
    Returns:
        bool: 是否可能包含攻擊流量
    """
    filename = pcap.name
    # example: normal_scanning2.pcap.TCP_192-168-1-31_48334_192-168-1-190_22.pcap
    attributes = filename.split(".")[2].split("_")
    if len(attributes[1]) < 16:
        key = (
            attributes[1].replace("-", "."),
            attributes[2],
            attributes[3].replace("-", "."),
            attributes[4],
            attributes[0].lower(),
        )
    else:
        key = (
            attributes[1].replace("-", ":"),
            attributes[2],
            attributes[3].replace("-", ":"),
            attributes[4],
            attributes[0].lower(),
        )
    if key in attack_dict_type:
        return True
    return False


def run_attack_filter(
    args: tuple[
        dict[tuple[str, str, str, str, str], list[int]],
        Path,
        Callable[[Path, bool], None] | None,
    ],
) -> bool:
    """
    過濾攻擊流量的主函式
    Args:
        args (tuple): 包含攻擊字典、pcap 檔案路徑及可選的處理函式
    Returns:
        bool: 是否包含攻擊流量
    """
    attack_dict_type = args[0]
    pcap = args[1]
    handle = args[2] if len(args) > 2 else None
    is_attack = False

    # output_file = pcap.replace('/encrypted_filter/', '/attack_filter/')
    # output_file = pcap.replace('/split/', '/attack_filter_with_decrypt/')
    def extract_5tuple(
        packet: Packet, pcap_path: Path
    ) -> tuple[str, str, str, str, str] | None:
        """
        提取封包的 5 元組 (5-tuple)
        Args:
            packet (Packet): Scapy 封包物件
        Returns:
            tuple: 5 元組 (protocol, src_ip, src_port, dst_ip, dst_port)
        """
        if IP in packet:
            src_ip = str(packet[IP].src)
            dst_ip = str(packet[IP].dst)
            if TCP in packet:
                src_port = str(packet[TCP].sport)
                dst_port = str(packet[TCP].dport)
                protocol_name = "TCP"
                return (protocol_name, src_ip, src_port, dst_ip, dst_port)
            elif UDP in packet:
                src_port = str(packet[UDP].sport)
                dst_port = str(packet[UDP].dport)
                protocol_name = "UDP"
                return (protocol_name, src_ip, src_port, dst_ip, dst_port)
            logger.getChild("run_attack_filter").getChild("extract_5tuple").warning(
                f"Non-TCP/UDP packet in IPv4 ({packet[IP].proto}): {str(pcap_path)}"
            )
            return None
        elif IPv6 in packet:
            src_ip = str(packet[IPv6].src)
            dst_ip = str(packet[IPv6].dst)
            if TCP in packet:
                src_port = str(packet[TCP].sport)
                dst_port = str(packet[TCP].dport)
                protocol_name = "TCP"
                return (protocol_name, src_ip, src_port, dst_ip, dst_port)
            elif UDP in packet:
                src_port = str(packet[UDP].sport)
                dst_port = str(packet[UDP].dport)
                protocol_name = "UDP"
                return (protocol_name, src_ip, src_port, dst_ip, dst_port)
            logger.getChild("run_attack_filter").getChild("extract_5tuple").warning(
                f"Non-TCP/UDP packet in IPv6 ({packet[IPv6].nh}): {str(pcap_path)}"
            )
            return None
        else:
            logger.getChild("run_attack_filter").getChild("extract_5tuple").warning(
                f"Non-IP packet: {str(pcap_path)}"
            )
            return None

    if pre_filter(attack_dict_type, pcap):
        try:
            pkts = rdpcap(str(pcap), count=1)
        except Exception as e:
            logger.getChild("run_attack_filter").error(
                f"Error reading {pcap}: {e}", exc_info=True
            )
            del pkts
            return is_attack
        if len(pkts) == 0:
            del pkts
            return is_attack
        for pkt in pkts:
            pkt_5tuple = extract_5tuple(pkt, pcap)
            if pkt_5tuple is not None:
                key = (
                    pkt_5tuple[1],
                    pkt_5tuple[2],
                    pkt_5tuple[3],
                    pkt_5tuple[4],
                    pkt_5tuple[0].lower(),
                )
                if key in attack_dict_type:
                    ts_list = attack_dict_type[key]
                    pkt_ts = float(pkt.time)
                    pkt_ts_floor = math.floor(pkt_ts)
                    if pkt_ts_floor in ts_list:
                        is_attack = True
                        break
            if is_attack:
                break
        del pkts
    if handle is not None:
        handle(pcap, is_attack)
    return is_attack
