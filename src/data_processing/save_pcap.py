import shutil
import subprocess
from pathlib import Path
from scapy.all import rdpcap, wrpcap


def save_pcap(
    pcap_file: Path, output_file: Path, remove_original: bool = False
) -> None:
    """
    讀取 pcap 檔案並重新寫入到指定位置
    Args:
        pcap_file (Path): 輸入的 pcap 檔案路徑
        output_file (Path): 輸出的 pcap 檔案路徑
    Returns:
        None
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    packets = rdpcap(str(pcap_file))
    wrpcap(str(output_file), packets)
    if remove_original:
        pcap_file.unlink()


def copy_pcap(src_file: Path, dest_file: Path) -> None:
    """
    複製 pcap 檔案到指定位置
    Args:
        src_file (Path): 原始的 pcap 檔案路徑
        dest_file (Path): 目的地的 pcap 檔案路徑
    Returns:
        None
    """
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_file, dest_file)


def pcapng_to_pcap(input_file: Path, output_file: Path) -> None:
    """
    將 pcapng 檔案轉換為 pcap 格式
    Args:
        input_file (Path): 輸入的 pcapng 檔案路徑
        output_file (Path): 輸出的 pcap 檔案路徑
    Returns:
        None
    """
    subprocess.run(
        ["tshark", "-F", "pcap", "-r", str(input_file), "-w", str(output_file)],
        check=True,
    )
