from pathlib import Path
from typing import List, Tuple, Optional
import os
import struct
from scapy.all import rdpcap, wrpcap, Raw, TCP, UDP
from data_processing.encrypt.utils import fix_checksums_and_lengths
from charm.toolbox.pairinggroup import PairingGroup, GT
from charm.schemes.abenc.abenc_lsw08 import KPabe
from charm.core.engine.util import objectToBytes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"KPABE1"

def kpabe_setup(policy: str = "((A and B))") -> Tuple[PairingGroup, KPabe, dict, dict]:
    """
    KP-ABE: key has policy, ciphertext has attributes.
    """
    group = PairingGroup("MNT224")
    kpabe = KPabe(group)
    pk, mk = kpabe.setup()
    sk = kpabe.keygen(pk, mk, policy)
    return group, kpabe, pk, sk


def pack_encrypted_payload(abe_ct_bytes: bytes, nonce: bytes, aes_ct: bytes) -> bytes:
    return b"".join(
        [
            MAGIC,
            struct.pack("!I", len(abe_ct_bytes)),
            abe_ct_bytes,
            struct.pack("!H", len(nonce)),
            nonce,
            struct.pack("!I", len(aes_ct)),
            aes_ct,
        ]
    )


def hybrid_encrypt_payload(
    group: PairingGroup,
    kpabe: KPabe,
    pk: dict,
    raw_payload: bytes,
    attributes: List[str],
) -> bytes:
    """
    1) random session key material (32 bytes) -> map into GT for KP-ABE message
    2) KP-ABE encrypt that GT element under attribute set
    3) AES-GCM encrypt payload using the 32-byte key material
    4) pack ABE ciphertext + AESGCM payload into bytes
    """
    # 32-byte symmetric key material
    sym_key = os.urandom(32)

    # KP-ABE message in GT: hash bytes -> GT element
    # (This makes a deterministic GT element from sym_key for ABE to encrypt.)
    m_gt = group.hash(sym_key, GT)

    # ABE encrypt the GT element under attributes
    abe_ct = kpabe.encrypt(pk, m_gt, attributes)
    abe_ct_bytes = objectToBytes(abe_ct, group)

    # AES-GCM encrypt the payload
    aesgcm = AESGCM(sym_key)
    nonce = os.urandom(12)  # 96-bit nonce
    # associated_data can be added if you want bind headers; keep None for minimalism
    aes_ct = aesgcm.encrypt(nonce, raw_payload, None)  # ciphertext||tag

    return pack_encrypted_payload(abe_ct_bytes, nonce, aes_ct)

def encrypt_pcap(in_pcap: Path, out_pcap: Path):
    attributes = ["A", "B"]
    group, kpabe, pk, sk = kpabe_setup("((A and B))")  # sk returned for completeness; not needed for encryption

    pkts = rdpcap(str(in_pcap))
    out = []
    for p in pkts:
        l4 = p.getlayer(TCP) or p.getlayer(UDP)
        if l4 is None:
            out.append(p); continue

        raw = bytes(l4.payload)
        if not raw:
            out.append(p); continue

        ct_bytes = hybrid_encrypt_payload(group, kpabe, pk, raw, attributes)
        if len(ct_bytes) > 65000:
            ct_bytes = ct_bytes[:65000]
        l4.remove_payload()
        l4.add_payload(Raw(load=ct_bytes))

        fix_checksums_and_lengths(p)
        out.append(p)

    wrpcap(str(out_pcap), out)
