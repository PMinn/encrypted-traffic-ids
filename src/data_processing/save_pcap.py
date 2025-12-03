#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from scapy.all import rdpcap, wrpcap
import shutil
import subprocess

def save_pcap(pcap_file: str, output_file: str, remove_original: bool = False):
    """
        讀取 pcap 檔案並重新寫入到指定位置
        Args:
            pcap_file (str): 輸入的 pcap 檔案路徑
            output_file (str): 輸出的 pcap 檔案路徑
        Returns:
            None
    """
    output_folder = "/".join(output_file.split("/")[:-1])
    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    packets = rdpcap(pcap_file)
    wrpcap(output_file, packets)
    if remove_original:
        os.remove(pcap_file)

def copy_pcap(src_file: str, dest_file: str):
    """
        複製 pcap 檔案到指定位置
        Args:
            src_file (str): 原始的 pcap 檔案路徑
            dest_file (str): 目的地的 pcap 檔案路徑
        Returns:
            None
    """
    output_folder = "/".join(dest_file.split("/")[:-1])
    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shutil.copy(src_file, dest_file)

def pcapng_to_pcap(input_file: str, output_file: str):
    """
        將 pcapng 檔案轉換為 pcap 格式
        Args:
            input_file (str): 輸入的 pcapng 檔案路徑
            output_file (str): 輸出的 pcap 檔案路徑
        Returns:
            None
    """
    subprocess.run([
        "tshark",
        "-F", "pcap",
        "-r", input_file,
        "-w", output_file
    ], check = True)