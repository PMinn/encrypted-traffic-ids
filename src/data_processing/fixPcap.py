#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess

def fix_pcap(input_pcap: str, output_pcap: str, remove_original = False):
    """
        使用 pcapfix 修復損壞的 pcap 檔案
        Args:
            input_pcap (str): 輸入 pcap 檔案路徑
            output_pcap (str): 輸出修復後的 pcap 檔案路徑
            remove_original (bool): 是否刪除原始檔案，預設為 False
        Returns:
            None
    """
    subprocess.run([
        "pcapfix",
        "-o", output_pcap,
        input_pcap
    ], check = True)
    if remove_original:
        os.remove(input_pcap)