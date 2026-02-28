# encrypt_aes_pcap.py
from pathlib import Path
from scapy.all import rdpcap, wrpcap, Raw, TCP, UDP
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from data_processing.encrypt.utils import fix_checksums_and_lengths

KEY = b"\x11" * 32          # 32 bytes => AES-256
IV = b"\x22" * 16        # 16 bytes for CBC (initialization vector)

def aes_cbc_encrypt(data: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()  # 128 bits = 16 bytes block
    padded = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(KEY), modes.CBC(IV))
    enc = cipher.encryptor()
    return enc.update(padded) + enc.finalize()

def encrypt_pcap(in_pcap: Path, out_pcap: Path):
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    pkts = rdpcap(str(in_pcap))
    out = []
    for p in pkts:
        l4 = p.getlayer(TCP) or p.getlayer(UDP)
        if l4 is None:
            out.append(p); continue

        raw = bytes(l4.payload)  # L4 payload
        if not raw:
            out.append(p); continue

        ct = aes_cbc_encrypt(raw)
        if len(ct) > 65000:
            ct = ct[:65000]
        l4.remove_payload()
        l4.add_payload(Raw(load=ct))

        fix_checksums_and_lengths(p)
        out.append(p)

    wrpcap(str(out_pcap), out)

# if __name__ == "__main__":
#     in_pcap = sys.argv[1]
#     out_pcap = sys.argv[2]
#     encrypt_pcap(in_pcap, out_pcap)
#     print("AES-CTR done:", out_pcap)
