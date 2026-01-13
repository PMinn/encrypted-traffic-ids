import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, cast

from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Packet, Raw

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asy_padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey


# -------------------------
# Config (all "you must choose" parts are here)
# -------------------------

Scheme = Literal["AES", "RSA", "KPABE"]
AESMode = Literal["GCM", "CTR", "CBC"]
RSAPadding = Literal["OAEP_SHA256"]  # keep tight; you can expand later


@dataclass
class EncryptConfig:
    # Which crypto scheme to use
    scheme: Scheme = "AES"

    # Which packets to touch
    only_tcp_udp: bool = True  # skip non TCP/UDP packets
    skip_no_raw: bool = True  # if no Raw payload, skip
    skip_empty_payload: bool = (
        True  # if payload is b"", skip (avoid creating payload where none existed)
    )

    # Output encoding format for ciphertext written back into packet payload
    # We MUST choose a serialization; paper doesn't specify.
    # We'll use a simple "length-prefixed header" for RSA/KPABE hybrid.
    # For AES-only we do: nonce || ciphertext(+tag if GCM)
    prefix_len_bytes: int = 2  # store encrypted-key length (RSA/KPABE hybrid)

    # AES params (paper doesn't specify mode/nonce/tag/key length)
    aes_mode: AESMode = "GCM"
    aes_key_bits: int = 128  # 128/192/256
    aes_nonce_len: int = 12  # for GCM/CTR typical 12/16; default 12
    aes_tag_len: int = (
        16  # GCM tag length (bytes), cryptography AESGCM uses 16 by default
    )
    aes_cbc_iv_len: int = 16  # for CBC

    # AAD (Associated Data) usage; paper doesn't specify.
    # Default: None (simplest).
    use_aad: bool = False
    aad_source: Literal["NONE", "5TUPLE"] = (
        "NONE"  # if enabled, can bind to flow 5-tuple-ish info
    )

    # RSA params (paper doesn't specify key size/padding/hybrid)
    rsa_key_bits: int = 2048
    rsa_oaep_hash: str = "SHA256"  # OAEP hash selection
    rsa_hybrid: bool = True  # IMPORTANT: if False, RSA will fail on long payloads

    # KP-ABE (paper doesn't specify policy/attributes/library)
    # We load a user-provided plugin with: encrypt_key(aes_key: bytes) -> bytes
    kpabe_encrypt_key_func: Optional[Callable[[bytes], bytes]] = None


# -------------------------
# Packet payload helpers
# -------------------------


def get_l4_payload(pkt: Packet) -> Optional[bytes]:
    """
    取得 L4 payload
    Arguments:
        pkt {Packet} -- Scapy封包物件
    Returns:
        Optional[bytes] -- L4 payload，若無Raw層則回傳None
    """
    if pkt.haslayer(Raw):
        return bytes(pkt[Raw].load)
    return None


def write_l4_payload(pkt: Packet, new_payload: bytes) -> Packet:
    """
    將 new_payload 寫入 L4 payload
    Arguments:
        pkt {Packet} -- Scapy封包物件
        new_payload {bytes} -- 新的L4 payload
    Returns:
        Packet -- 修改後的Scapy封包物件
    """
    if pkt.haslayer(Raw):
        pkt[Raw].load = new_payload
    else:
        pkt = pkt / Raw(load=new_payload)

    # force checksum/len recalc
    if pkt.haslayer(IP):
        if hasattr(pkt[IP], "len"):
            del pkt[IP].len
        if hasattr(pkt[IP], "chksum"):
            del pkt[IP].chksum
    if pkt.haslayer(TCP) and hasattr(pkt[TCP], "chksum"):
        del pkt[TCP].chksum
    if pkt.haslayer(UDP):
        if hasattr(pkt[UDP], "len"):
            del pkt[UDP].len
        if hasattr(pkt[UDP], "chksum"):
            del pkt[UDP].chksum

    return pkt


# -------------------------
# Crypto primitives (choices parameterized)
# -------------------------


def make_aes_key(cfg: EncryptConfig) -> bytes:
    """
    產生 AES 金鑰
    Arguments:
        cfg {EncryptConfig} -- 加密設定
    Returns:
        bytes -- 產生的 AES 金鑰
    """
    if cfg.aes_key_bits not in (128, 192, 256):
        raise ValueError("aes_key_bits must be 128/192/256")
    return os.urandom(cfg.aes_key_bits // 8)


def aes_encrypt(
    payload: bytes, aes_key: bytes, cfg: EncryptConfig, aad: Optional[bytes]
) -> bytes:
    """
    使用 AES 加密資料
    Arguments:
        payload {bytes} -- 要加密的資料
        aes_key {bytes} -- AES 金鑰
        cfg {EncryptConfig} -- 加密設定
        aad {Optional[bytes]} -- 附加資料 (AAD)
    Returns:
        bytes -- 加密後的資料
    """
    if cfg.aes_mode == "GCM":
        nonce = os.urandom(cfg.aes_nonce_len)
        aesgcm = AESGCM(aes_key)
        ct = aesgcm.encrypt(nonce, payload, aad)
        # format: nonce || ct(tag included inside ct)
        return nonce + ct

    if cfg.aes_mode == "CTR":
        nonce = os.urandom(16)  # CTR typically needs 16 bytes nonce/initial counter
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce))
        enc = cipher.encryptor()
        ct = enc.update(payload) + enc.finalize()
        return nonce + ct

    # if cfg.aes_mode == "CBC":
    #     iv = os.urandom(cfg.aes_cbc_iv_len)
    #     # CBC needs padding. Not paper-specified, so we choose PKCS7.
    #     pad_len = 16 - (len(payload) % 16)
    #     padded = payload + bytes([pad_len]) * pad_len
    #     cipher = Cipher(algorithms.AES(aes_key), cast(modes.Mode, modes.CBC(iv)))
    #     enc = cipher.encryptor()
    #     ct = enc.update(padded) + enc.finalize()
    #     return iv + ct

    raise ValueError("Unknown aes_mode")


def gen_rsa_keypair(cfg: EncryptConfig) -> tuple[RSAPrivateKey, RSAPublicKey]:
    """
    產生 RSA 金鑰對
    Arguments:
        cfg {EncryptConfig} -- 加密設定
    Returns:
        tuple[RSAPrivateKey, RSAPublicKey] -- 產生的 RSA 私鑰與公鑰
    """
    priv = rsa.generate_private_key(public_exponent=65537, key_size=cfg.rsa_key_bits)
    return priv, priv.public_key()


def rsa_encrypt_bytes(data: bytes, rsa_pub: RSAPublicKey, cfg: EncryptConfig) -> bytes:
    """
    使用 RSA 公鑰加密資料
    Arguments:
        data {bytes} -- 要加密的資料
        rsa_pub {RSAPublicKey} -- RSA 公鑰
        cfg {EncryptConfig} -- 加密設定
    Returns:
        bytes -- 加密後的資料
    """
    # padding not specified by paper -> default OAEP-SHA256 (configurable)
    if cfg.rsa_oaep_hash.upper() == "SHA256":
        h: hashes.HashAlgorithm = hashes.SHA256()
    elif cfg.rsa_oaep_hash.upper() == "SHA1":
        h = hashes.SHA1()
    else:
        raise ValueError("rsa_oaep_hash must be 'SHA256' or 'SHA1'")

    return rsa_pub.encrypt(
        data,
        asy_padding.OAEP(
            mgf=asy_padding.MGF1(algorithm=h),
            algorithm=h,
            label=None,
        ),
    )


# -------------------------
# Scheme implementations (AES / RSA-hybrid / KPABE-hybrid)
# -------------------------


def encrypt_payload(
    payload: bytes,
    cfg: EncryptConfig,
    *,
    aes_key: bytes,
    rsa_pub: RSAPublicKey | None = None,
    kpabe_encrypt_key_func: Optional[Callable[[bytes], bytes]] = None,
    aad: Optional[bytes] = None,
) -> bytes:
    """
    根據指定的加密方案加密資料
    Arguments:
        payload {bytes} -- 要加密的資料
        cfg {EncryptConfig} -- 加密設定
        aes_key {bytes} -- AES 金鑰
        rsa_pub {RSAPublicKey | None} -- RSA 公鑰 (若使用RSA方案則需提供)
        kpabe_encrypt_key_func {Optional[Callable[[bytes], bytes]]} -- KP-ABE 加密金鑰函式 (若使用KP-ABE方案則需提供)
        aad {Optional[bytes]} -- 附加資料 (AAD)
    Returns:
        bytes -- 加密後的資料
    """
    if cfg.scheme == "AES":
        # AES-only: write AES ciphertext as payload
        return aes_encrypt(payload, aes_key, cfg, aad)

    if cfg.scheme == "RSA":
        if rsa_pub is None:
            raise ValueError("RSA scheme needs rsa_pub")

        if cfg.rsa_hybrid:
            # Hybrid: RSA encrypts AES key, AES encrypts payload
            enc_key = rsa_encrypt_bytes(aes_key, rsa_pub, cfg)
            ct = aes_encrypt(payload, aes_key, cfg, aad)
            # format: len(enc_key) || enc_key || ct
            if len(enc_key) >= 256**cfg.prefix_len_bytes:
                raise ValueError("Encrypted key too large for prefix_len_bytes")
            return len(enc_key).to_bytes(cfg.prefix_len_bytes, "big") + enc_key + ct

        # RSA-only: will FAIL if payload too long. Kept as an explicit option.
        return rsa_encrypt_bytes(payload, rsa_pub, cfg)

    if cfg.scheme == "KPABE":
        if cfg.kpabe_encrypt_key_func is None:
            raise ValueError(
                "KPABE scheme needs kpabe_encrypt_key_func(aes_key)->bytes"
            )

        abe_enc_key = cfg.kpabe_encrypt_key_func(aes_key)
        aes_blob = aes_encrypt(payload, aes_key, cfg, aad=aad)
        if len(abe_enc_key) >= 256**cfg.prefix_len_bytes:
            raise ValueError("abe_enc_key too large for prefix_len_bytes")
        return (
            len(abe_enc_key).to_bytes(cfg.prefix_len_bytes, "big")
            + abe_enc_key
            + aes_blob
        )
    raise ValueError("Unknown scheme")


# -------------------------
# Main PCAP processing
# -------------------------


def encrypt_pcap(
    in_pcap: Path,
    out_pcap: Path,
    cfg: EncryptConfig,
    rsa_pub: RSAPublicKey | None = None,
) -> None:
    """
    加密 PCAP 檔案
    Arguments:
        in_pcap {Path} -- 輸入的 PCAP 檔案路徑
        out_pcap {Path} -- 輸出的加密後 PCAP 檔案路徑
        cfg {EncryptConfig} -- 加密設定
    """
    pkts = rdpcap(str(in_pcap))
    out = []

    aes_key = make_aes_key(cfg)

    if cfg.scheme == "RSA" and rsa_pub is None:
        _, rsa_pub = gen_rsa_keypair(cfg)

    for pkt in pkts:
        if cfg.only_tcp_udp and not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
            out.append(pkt)
            continue

        payload = get_l4_payload(pkt)

        if payload is None and cfg.skip_no_raw:
            out.append(pkt)
            continue
        if payload is None:
            payload = b""
        if cfg.skip_empty_payload and len(payload) == 0:
            out.append(pkt)
            continue

        new_payload = encrypt_payload(payload, cfg, aes_key=aes_key, rsa_pub=rsa_pub)
        pkt2 = write_l4_payload(pkt, new_payload)
        out.append(pkt2)

    wrpcap(str(out_pcap), out)


def encrypt_pcap_by_three_schemes(
    in_pcap: Path,
    out_dir: Path = Path("out"),
    out_prefix: Optional[str] = None,
    schemes: tuple[Scheme, ...] = ("AES", "RSA", "KPABE"),
    *,
    # shared defaults (paper didn't specify -> configurable)
    only_tcp_udp: bool = True,
    skip_no_raw: bool = True,
    skip_empty_payload: bool = True,
    prefix_len_bytes: int = 2,
    # AES defaults
    aes_key_bits: int = 128,
    aes_nonce_len: int = 12,
    # RSA defaults
    rsa_key_bits: int = 2048,
    rsa_hybrid: bool = True,
    rsa_oaep_hash: str = "SHA256",
    # KP-ABE hook (required if you include "KPABE" in schemes)
    kpabe_encrypt_key_func: Optional[Callable[[bytes], bytes]] = None,
) -> dict[Scheme, Path]:
    """
    使用三種加密方案加密 PCAP 檔案
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for scheme in schemes:
        (out_dir / scheme).mkdir(exist_ok=True, parents=True)
    out_prefix = out_prefix or in_pcap.stem

    outputs: dict[Scheme, Path] = {}

    for scheme in schemes:
        cfg = EncryptConfig(
            scheme=scheme,
            only_tcp_udp=only_tcp_udp,
            skip_no_raw=skip_no_raw,
            skip_empty_payload=skip_empty_payload,
            prefix_len_bytes=prefix_len_bytes,
            aes_key_bits=aes_key_bits,
            aes_nonce_len=aes_nonce_len,
            rsa_key_bits=rsa_key_bits,
            rsa_hybrid=rsa_hybrid,
            rsa_oaep_hash=rsa_oaep_hash,
            kpabe_encrypt_key_func=kpabe_encrypt_key_func,
        )

        out_path = out_dir / scheme / f"{out_prefix}_{scheme}.pcap"
        encrypt_pcap(in_pcap, out_path, cfg)
        outputs[scheme] = out_path

    return outputs


# -------------------------
# Example usage (no CLI)
# -------------------------
# if __name__ == "__main__":
#     # 1) AES + RSA only (KPABE omitted)
#     outputs = main(
#         in_pcap="input.pcap",
#         out_dir="out",
#         schemes=("AES", "RSA"),
#     )
#     print(outputs)

#     # 2) If you want KP-ABE too, you MUST provide kpabe_encrypt_key_func.
#     #    This is a placeholder only (NOT real KP-ABE).
#     def dummy_kpabe_encrypt_key(aes_key: bytes) -> bytes:
#         return b"ABE(" + aes_key + b")"

#     outputs2 = main(
#         in_pcap="input.pcap",
#         out_dir="out",
#         schemes=("AES", "RSA", "KPABE"),
#         kpabe_encrypt_key_func=dummy_kpabe_encrypt_key,
#     )
#     print(outputs2)
