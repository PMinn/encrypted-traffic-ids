#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from scapy.all import rdpcap, wrpcap
import shutil

def save_pcap(pcap_file, output_file):
    output_folder = "/".join(output_file.split("/")[:-1])
    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    packets = rdpcap(pcap_file)
    wrpcap(output_file, packets)

def copy_pcap(src_file, dest_file):
    output_folder = "/".join(dest_file.split("/")[:-1])
    if output_folder != "" and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    shutil.copy(src_file, dest_file)