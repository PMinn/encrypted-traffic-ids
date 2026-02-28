# RSA 會讓 payload 暴增（每 190 bytes 變 256 bytes）
# 封包可能超 MTU，pcap 仍能寫但你後續特徵分布會被放大。

from pathlib import Path
from scapy.all import rdpcap, wrpcap, Raw, TCP, UDP
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

from data_processing.encrypt.utils import fix_checksums_and_lengths

# Generate one keypair per run (minimal). Replace with persisted keys if you want stable results.
# PRIV = rsa.generate_private_key(public_exponent=65537, key_size=2048)
# PUB = PRIV.public_key()

def rsa_oaep_encrypt(data: bytes, PUB: rsa.RSAPublicKey) -> bytes:
    # OAEP max plaintext per block: k - 2*hLen - 2 (k=256 for 2048-bit, hLen=32 for SHA256 => 190)
    max_block = 2048 // 8 - 2 * hashes.SHA256().digest_size - 2
    out = []
    for i in range(0, len(data), max_block):
        chunk = data[i:i+max_block]
        out.append(PUB.encrypt(
            chunk,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                         algorithm=hashes.SHA256(),
                         label=None)
        ))
    return b"".join(out)

def encrypt_pcap(in_pcap: Path, out_pcap: Path, PUB: rsa.RSAPublicKey):
    out_pcap.parent.mkdir(parents=True, exist_ok=True)
    pkts = rdpcap(str(in_pcap))
    out = []
    for p in pkts:
        l4 = p.getlayer(TCP) or p.getlayer(UDP)
        if l4 is None:
            out.append(p); continue

        raw = bytes(l4.payload)
        if not raw:
            out.append(p); continue

        ct = rsa_oaep_encrypt(raw, PUB)
        # 儲存時一個封包長度不能超過 65,535 bytes
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
#     print("RSA-OAEP done:", out_pcap)
