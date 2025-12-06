import pyshark
import shutil
import os
import logging
logger = logging.getLogger()

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
def check_encryption(pcap_file: str):
    """
        檢查 pcap 檔案中是否包含加密協議
        Args:
            pcap_file (str): pcap 檔案路徑
        Returns:
            bool: 是否包含加密協議
    """
    has_encrypted = False
    # protocols_found = set()
    try:
        cap = pyshark.FileCapture(pcap_file, display_filter = display_filter, keep_packets = False) # 只讀出被 Wireshark 判定為 TLS 的封包
        for pkt in cap:
            has_encrypted = True
            break
            # for proto in ENCRYPTED_PROTOCOLS:
            #     if hasattr(pkt, proto):
            #         protocols_found.add(proto)
        cap.close()
    except Exception as e:
        logger.error(f"Error closing {pcap_file}: {e}")
    return has_encrypted, None#protocols_found

def df(pcap):
    # output_file = pcap.replace('TON_IoT/split/Benign', 'TON_IoT/encrypted_filter/Benign')
    # if os.path.exists(output_file):
    #     return
    has_encrypted, protocols_found = check_encryption(pcap)
    if has_encrypted:
        pass
        # logger.info(f"Encrypted protocols found in {pcap}")
        # folder = "/".join(output_file.split("/")[:-1])
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # shutil.copy2(pcap, output_file)
    else:
        # logger.info(f"No encrypted protocols found in {pcap}")
        os.remove(pcap)
    return has_encrypted
    
def mf(pcap):
    # output_file = pcap.replace('TON_IoT/split/Malicious', 'TON_IoT/encrypted_filter/Malicious')
    # if os.path.exists(output_file):
    #     return
    has_encrypted, protocols_found = check_encryption(pcap)
    if has_encrypted:
        pass
        # logger.info(f"{pcap}")
        # folder = "/".join(output_file.split("/")[:-1])
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # shutil.copy2(pcap, output_file)
    else:
        # logger.info(f"No encrypted protocols found in {pcap}")
        os.remove(pcap)
    return has_encrypted

def filter_pcap(pcap: str):
    """
        篩選出加密的 pcap 檔案
        Args:
            pcap (str): pcap 檔案路徑
        Returns:
            bool: 是否為加密的 pcap
    """
    has_encrypted, protocols_found = check_encryption(pcap)
    if has_encrypted:
        pass
    else:
        os.remove(pcap)
    return has_encrypted