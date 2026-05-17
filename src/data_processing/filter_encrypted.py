from pathlib import Path
import pyshark
import os
import logging


ENCRYPTED_PROTOCOLS = [
    "tls",
    "ssl",
    "dtls",
    "quic",
    "ssh",
    "esp",
    "isakmp",  # IKEv1
    # "ikev2",
    "openvpn",
    # "wireguard",
]
display_filter = " or ".join(ENCRYPTED_PROTOCOLS)


def check_encryption(pcap_file: Path) -> bool:
    """
    檢查 pcap 檔案中是否包含加密協議
    Args:
        pcap_file (Path): pcap 檔案路徑
    Returns:
        bool: 是否包含加密協議
    """
    has_encrypted = False
    try:
        cap = pyshark.FileCapture(
            str(pcap_file), display_filter=display_filter, keep_packets=False
        )  # 只讀出被 Wireshark 判定為 TLS 的封包
        for _ in cap:
            has_encrypted = True
            break
        cap.close()
    except Exception as e:
        logging.getLogger("filter_encrypted.check_encryption").error(
            f"Error closing {pcap_file}: {e}", exc_info=True
        )
    return has_encrypted


def remove_pcap_if_not_encrypted(pcap: Path) -> bool:
    """
    如果 pcap 檔案中不包含加密協議，則刪除該檔案
    Args:
        pcap (Path): pcap 檔案路徑
    Returns:
        bool: 是否包含加密協議
    """
    has_encrypted = check_encryption(pcap)
    if not has_encrypted:
        os.remove(pcap)
    return has_encrypted

def check_encrypted_type(pcap_file: Path) -> dict:
    find_protocol = None
    for protocol in ENCRYPTED_PROTOCOLS:
        try:
            cap = pyshark.FileCapture(
                str(pcap_file), display_filter=protocol, keep_packets=False
            )  # 只讀出被 Wireshark 判定為 TLS 的封包
            for _ in cap:
                find_protocol = protocol
        except Exception as e:
            logging.getLogger("filter_encrypted.check_encrypted_type").error(
                f"Error closing {pcap_file}: {e}", exc_info=True
            )
        if find_protocol is not None:
            break
    return (pcap_file.name, find_protocol)