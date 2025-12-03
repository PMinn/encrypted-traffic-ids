#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
因惡意 pcap 中，有混合攻擊與非攻擊的封包，
所以需要再處理，過濾出 pcap 中有攻擊的封包。 
'''

import pyshark
import shutil
import os
import glob
import logging
from tqdm import tqdm
from data_processing.save_pcap import copy_pcap

logger = logging.getLogger()

# 使用檔名初次過濾
def pre_filter(attack_dict_type, pcap):
    filename = pcap.split("/")[-1]
    # normal_scanning2.pcap.TCP_192-168-1-31_48334_192-168-1-190_22.pcap
    attributes = filename.split(".")[2].split("_")
    if len(attributes[1]) < 16:
        key = (attributes[1].replace('-', '.'), attributes[2], attributes[3].replace('-', '.'), attributes[4], attributes[0].lower())
    else:
        key = (attributes[1].replace('-', ':'), attributes[2], attributes[3].replace('-', ':'), attributes[4], attributes[0].lower())
    if key in attack_dict_type:
        return True
    return False

# 過濾攻擊執行緒的函式
def fa(args):
    attack_dict_type, pcap = args
    # output_file = pcap.replace('/encrypted_filter/', '/attack_filter/')
    # output_file = pcap.replace('/split/', '/attack_filter_with_decrypt/')
    if pre_filter(attack_dict_type, pcap):
        # and not os.path.exists(output_file):
        cap = pyshark.FileCapture(pcap, keep_packets = False)
        save_flag = False
        for pkt in cap:
            if hasattr(pkt, 'ip'):
                if hasattr(pkt, 'tcp'):
                    proto = 'tcp'
                elif hasattr(pkt, 'udp'):
                    proto = 'udp'
                else:
                    continue
                key = (pkt.ip.src, pkt[proto].srcport, pkt.ip.dst, pkt[proto].dstport, proto)
                if key in attack_dict_type:
                    ts_list = attack_dict_type[key]
                    pkt_ts = int(pkt.sniff_time.timestamp())
                    if pkt_ts in ts_list:
                        save_flag = True
                        break
            if save_flag:
                break
        cap.close()
        if save_flag:
            # output_folder = "/".join(output_file.split("/")[:-1])
            # if not os.path.exists(output_folder):
            #     os.makedirs(output_folder)
            # if not os.path.exists(output_file):
            # logger.info(f"Attack found in {pcap}, saving to {output_file}") 
            # copy_pcap(src_file = pcap, dest_file = output_file)
            logger.info(pcap) 