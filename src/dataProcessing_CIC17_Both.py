import glob
# from multiprocessing import Pool
import os
# import random
import re
# import shutil

import datetime
# from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import logging

def rewrite_pcap():
    from data_processing.save_pcap import pcapng_to_pcap
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR) # 忽略 Scapy 的警告訊息
    logger = logging.getLogger("rewrite_pcap")
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/CIC-IDS-2017/*')
    logger.debug(f"Total pcaps to rewrite: {len(pcaps)}")
    with tqdm(total = len(pcaps)) as pbar:
        for pcap in pcaps:
            output_file = pcap.replace('/CIC-IDS-2017/CIC-IDS-2017/', '/CIC-IDS-2017/rewrite/')
            folder = "/".join(output_file.split("/")[:-1])
            os.makedirs(folder, exist_ok = True)
            if os.path.exists(output_file):
                pbar.update(1)
                continue
            try:
                pcapng_to_pcap(input_file = pcap, output_file = output_file)
            except Exception as e:
                logger.error(f"Error processing {pcap}: {e}", exc_info = True)
            pbar.update(1)
            
def split_pcap():
    from data_processing.split_to_flows import split_to_flows_from_file
    # split_to_flows_from_file(
    #     input_file = "/home/alanpan/datasets/CIC-IDS-2017/rewrite/Tuesday-WorkingHours.pcap", # 輸入檔案路徑
    #     output_dir = "/home/alanpan/datasets/CIC-IDS-2017/split/Tuesday/", # 輸出資料夾路徑
    #     splitCapPath = "/home/alanpan/encrypted-NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False
    # )
    # split_to_flows_from_file(
    #     input_file = "/home/alanpan/datasets/CIC-IDS-2017/rewrite/Wednesday-workingHours.pcap", # 輸入檔案路徑
    #     output_dir = "/home/alanpan/datasets/CIC-IDS-2017/split/Wednesday/", # 輸出資料夾路徑
    #     splitCapPath = "/home/alanpan/encrypted-NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False
    # )
    # split_to_flows_from_file(
    #     input_file = "/home/alanpan/datasets/CIC-IDS-2017/rewrite/Thursday-WorkingHours.pcap", # 輸入檔案路徑
    #     output_dir = "/home/alanpan/datasets/CIC-IDS-2017/split/Thursday/", # 輸出資料夾路徑
    #     splitCapPath = "/home/alanpan/encrypted-NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False
    # )
    # split_to_flows_from_file(
    #     input_file = "/home/alanpan/datasets/CIC-IDS-2017/rewrite/Friday-WorkingHours.pcap", # 輸入檔案路徑
    #     output_dir = "/home/alanpan/datasets/CIC-IDS-2017/split/Friday/", # 輸出資料夾路徑
    #     splitCapPath = "/home/alanpan/encrypted-NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False
    # )
    split_to_flows_from_file(
        input_file = "/home/alanpan/datasets/CIC-IDS-2017/rewrite/Monday-WorkingHours.pcap", # 輸入檔案路徑
        output_dir = "/home/alanpan/datasets/CIC-IDS-2017/split/Monday/", # 輸出資料夾路徑
        splitCapPath = "/home/alanpan/encrypted-NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
        remove_original = False
    )

def count_pcaps(path):
    # 正常流量 Benign
    benign_pcaps = glob.glob(os.path.join(f'{path}/Benign/**', "*.pcap"), recursive = True)
    logger.info(f"Benign pcaps: {len(benign_pcaps)}")
    logger.info(f"-----------------------------------")
    # 惡意流量 Malicious
    malicious_folders = glob.glob(f'{path}/Malicious/*')
    malicious_folders.sort()
    total_malicious_pcaps = 0
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.split('/')[-1]
        malicious_pcaps = glob.glob(os.path.join(malicious_folder, "**", "*.pcap"), recursive = True)
        logger.info(f"{folder_name} pcaps: {len(malicious_pcaps)}")
        total_malicious_pcaps += len(malicious_pcaps)
    logger.info(f"-----------------------------------")
    logger.info(f"Total Malicious pcaps: {total_malicious_pcaps}")
    logger.info(f"Average Malicious pcaps per type: {total_malicious_pcaps / len(malicious_folders)}")

def get_features_0307():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/split/Monday/split_*/*.pcap')
    flow_to_features_file(
        pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Monday-03-07-2017/benign_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape
    )

def get_features_0407():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/split/Tuesday/split_*/*.pcap')
    ftp_pataor_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_21\.', pcap)]
    flow_to_features_file(
        ftp_pataor_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Tuesday-04-07-2017/ftp_pataor_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499170672 <= pkts[0].time <= 1499174417
    )
    ssh_pataor_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_22\.', pcap)]
    flow_to_features_file(
        ssh_pataor_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Tuesday-04-07-2017/ssh_pataor_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499188141 <= pkts[0].time <= 1499195060
    )

def get_features_0507():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/split/Wednesday/split_*/*.pcap')
    dos_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', pcap)]
    flow_to_features_file(
        dos_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_hulk_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: (1499262203 <= pkts[0].time <= 1499263642)
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_goldeneye_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499263803 <= pkts[0].time <= 1499264409
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_slowloris_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499258934 <= pkts[0].time <= 1499260279
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_slowhttptest_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499260537 <= pkts[0].time <= 1499261870
    )
    heartbleed_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_45022_192-168-10-51_444\.', pcap)]
    flow_to_features_file(
        heartbleed_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/heartbleed_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499278335 <= pkts[0].time <= 1499279564
    )

def get_features_0607():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/split/Thursday/split_*/*.pcap')
    web_attack_sql_injection_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', pcap)]
    flow_to_features_file(
        web_attack_sql_injection_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_sql_injection_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499348127 <= pkts[0].time <= 1499348576
    )
    web_attack_xss_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_(?!(36180|36182|36184|36186|36188|36190)$)_192-168-10-50_80\.', pcap)]
    flow_to_features_file(
        web_attack_xss_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_xss_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499346935 <= pkts[0].time <= 1499348122
    )
    web_attack_brute_force_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', pcap)]
    flow_to_features_file(
        web_attack_brute_force_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_brute_force_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499343354 <= pkts[0].time <= 1499346012
    )
    infiltration2_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_205-174-165-73_', pcap)]
    flow_to_features_file(
        infiltration2_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration2_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499361542 <= pkts[0].time <= 1499366770
    )
    infiltration1_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-25_.*_205-174-165-73_', pcap)]
    flow_to_features_file(
        infiltration1_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration1_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499363616 <= pkts[0].time <= 1499371340
    )
    infiltration31_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_(50122|50133)_192-168-10-51_', pcap)]
    flow_to_features_file(
        infiltration31_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration31_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499360431 <= pkts[0].time <= 1499360446
    )
    infiltration32_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_192-168-10-5_', pcap)]
    flow_to_features_file(
        infiltration32_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration32_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499362410 <= pkts[0].time <= 1499362445
    )
    infiltration33_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_192-168-10-(5|9|12|14|15|16|17|19|25|50|51)_', pcap)]
    flow_to_features_file(
        infiltration33_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration33_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499364314 <= pkts[0].time <= 1499366765
    )
    
def get_features_0707():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/split/Friday/split_*/*.pcap')
    infiltration_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_', pcap)]
    flow_to_features_file(
        infiltration_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Friday-07-07-2017/infiltration_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499446532 <= pkts[0].time <= 1499447949 or 1499449905 <= pkts[0].time <= 1499451842
    )
    botnet_ares_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-(15|9|14|5|8)_.*_205-174-165-73_', pcap)]
    flow_to_features_file(
        botnet_ares_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Friday-07-07-2017/botnet_ares_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499432653 <= pkts[0].time <= 1499457685
    )
    ddos_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_', pcap)]
    flow_to_features_file(
        ddos_pcaps,
        output_file = f'/home/alanpan/datasets/CIC-IDS-2017/features/Friday-07-07-2017/ddos_features_{packet_shape[0]}_{packet_shape[1]}_24.npy',
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499453791 <= pkts[0].time <= 1499454973
    )
    
def mearge_features():
    from data_processing.features import mearge_feature_files, copy_feature_file
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Monday-03-07-2017/benign_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/benign_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Tuesday-04-07-2017/ftp_pataor_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/ftp_brute_force_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Tuesday-04-07-2017/ssh_pataor_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/ssh_brute_force_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_slowloris_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/dos_slowloris_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_slowhttptest_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/dos_slowhttptest_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Wednesday-05-07-2017/dos_hulk_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/dos_hulk_features_96_5_24.npy"
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Friday-07-07-2017/ddos_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/ddos_features_96_5_24.npy"
    )
    mearge_feature_files(
        [
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration31_features_96_5_24.npy",    
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration32_features_96_5_24.npy",    
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/infiltration33_features_96_5_24.npy",    
        ],
        output_file = '/home/alanpan/datasets/CIC-IDS-2017/features/final/port_scan_features_96_5_24.npy'
    )
    mearge_feature_files(
        [
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_brute_force_features_96_5_24.npy",    
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_sql_injection_features_96_5_24.npy",    
            "/home/alanpan/datasets/CIC-IDS-2017/features/Thursday-06-07-2017/web_attack_xss_features_96_5_24.npy",    
        ],
        output_file = '/home/alanpan/datasets/CIC-IDS-2017/features/final/web_features_96_5_24.npy'
    )
    copy_feature_file(
        "/home/alanpan/datasets/CIC-IDS-2017/features/Friday-07-07-2017/botnet_ares_features_96_5_24.npy",
        "/home/alanpan/datasets/CIC-IDS-2017/features/final/botnet_ares_features_96_5_24.npy"
    )

def split_features_to_train_test():
    import numpy as np
    from sklearn.model_selection import train_test_split
    train_folder = '/home/alanpan/datasets/CIC-IDS-2017/features/train'
    test_folder = '/home/alanpan/datasets/CIC-IDS-2017/features/test'
    os.makedirs(train_folder, exist_ok = True)
    os.makedirs(test_folder, exist_ok = True)
    features_npys = glob.glob('/home/alanpan/datasets/CIC-IDS-2017/features/final/*.npy')    
    for index, features_npy in enumerate(features_npys):
        filename = os.path.basename(features_npy)
        data = np.load(features_npy, allow_pickle = True)
        number_of_data = data.shape[0]
        train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)
        number_of_train_data = train_data.shape[0]
        number_of_test_data = test_data.shape[0]
        np.save(os.path.join(train_folder, filename), train_data, allow_pickle = False)
        np.save(os.path.join(test_folder, filename), test_data, allow_pickle = False)
        # 建立 label
        # np.save(os.path.join(train_folder, f"{filename.replace('.npy', '')}_label.npy"), np.ones(number_of_train_data).astype(int) * index, allow_pickle = False)
        # np.save(os.path.join(test_folder, f"{filename.replace('.npy', '')}_label.npy"), np.ones(number_of_test_data).astype(int) * index, allow_pickle = False)
        logger.info(f"{filename}: total={number_of_data}, train={number_of_train_data}, test={number_of_test_data}")

def run_sampling():
    from data_processing.sampling import sampling, get_oversampling_by_kmeans
    sampling(
        {
            'Benign': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/benign_features_96_5_24.npy',
                'sampling_count': 64000
            },
            'FTP-BruteForce': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/ftp_brute_force_features_96_5_24.npy',
                'sampling_count': 3135
            },
            'SSH-BruteForce': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/ssh_brute_force_features_96_5_24.npy',
                'sampling_count': 3500
            },
            'DoS-Slowloris': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/dos_slowloris_features_96_5_24.npy',
                'sampling_count': 3089
            },
            'DoS-SlowHTTPTest': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/dos_slowhttptest_features_96_5_24.npy',
                'sampling_count': 3368
            },
            'DoS-Hulk': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/dos_hulk_features_96_5_24.npy',
                'sampling_count': 4741
            },
            'DDoS': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/ddos_features_96_5_24.npy',
                'sampling_count': 12000
            },
            'Port Scan': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/port_scan_features_96_5_24.npy',
                'sampling_count': 16000
            },
            'Web Attacks': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/web_features_96_5_24.npy',
                'sampling_count': 3500
            },
            'Bot': {
                'data_path': '/home/alanpan/datasets/CIC-IDS-2017/features/train/botnet_ares_features_96_5_24.npy',
                'sampling_count': 3501
            }
        },
        save_to = '/home/alanpan/datasets/CIC-IDS-2017/features/sampled/train/',
        oversampling = get_oversampling_by_kmeans
    )

def slice_sampled_features():
    import numpy as np
    import shutil
    sampled_features = '/home/alanpan/datasets/CIC-IDS-2017/features/sampled/train/sampled_data.npy'
    output_folder = '/home/alanpan/datasets/CIC-IDS-2017/features/sampled_raw/train/'
    os.makedirs(output_folder, exist_ok = True)
    filename = os.path.basename(sampled_features)
    data = np.load(sampled_features, allow_pickle = False)
    raw_data = data[:, 24:]
    np.save(os.path.join(output_folder, filename), raw_data, allow_pickle = False)
    shutil.copy("/home/alanpan/datasets/CIC-IDS-2017/features/sampled/train/classes.json", "/home/alanpan/datasets/CIC-IDS-2017/features/sampled_raw/train/classes.json")
    shutil.copy("/home/alanpan/datasets/CIC-IDS-2017/features/sampled/train/sampled_label.npy", "/home/alanpan/datasets/CIC-IDS-2017/features/sampled_raw/train/sampled_label.npy")
    
def main():
    # 將 pcapng 轉成 pcap
    # rewrite_pcap()

    # 分離成 Flows
    # split_pcap()

    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/encrypted_filter')

    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter')
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter_with_decrypt')

    # 特徵擷取
    # get_features_0307()
    # get_features_0407()
    # get_features_0507()
    # get_features_0607()
    # get_features_0707()
    
    # 合併特徵
    # mearge_features()
    
    # 分割成訓練集與測試集
    # split_features_to_train_test()

    attack_class = [
        "Injection",
        "MITM",
        # "backdoor",
        "DDoS",
        "DoS",
        # "runsomware",
        "scanning",
        "XSS",
        "password"
    ]
    classes = [*['benign'], *attack_class]
    
    # 建議採樣數量
    # counts = {
    #     "benign": 971,
    #     "Injection": 25140,
    #     "MITM": 30,
    #     "DDoS": 128487,
    #     "DoS": 49,
    #     "scanning": 15747,
    #     "XSS": 12,
    #     "password": 9
    # }
    # suggest(counts)

    # 不平衡資料處理
    # run_sampling()
    # 多分類
    UNDERSAMPLING = { 
        "Injection": 6702, 
        "DDoS": 15151, 
        "scanning": 5304,
    }
    OVERSAMPLING = {
        'benign': 1317,
        "MITM": 500,
        "DoS": 500,
        "XSS": 500,
        "password": 500,
    }
    # run_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/train',
    #     MULTI_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = False
    # )
    # 產生 testing 資料 (不須 testing 的話可以不用跑)
    # run_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/test',
    #     MULTI_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/test/sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = True
    # )
    # 二分類
    UNDERSAMPLING = { 
        "Injection": 6702, 
        "DDoS": 8000, 
        "scanning": 5304,
    }
    OVERSAMPLING = {
        'benign': 5000,
        "MITM": 800,
        "DoS": 800,
        "XSS": 800,
        "password": 800,
    }
    # run_binary_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/train',
    #     BINARY_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/binary_sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = False
    # )
    # 產生二分類資料 (不須 testing 的話可以不用跑)
    # run_binary_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/test',
    #     BINARY_PATH = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/test/binary_sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = True
    # )
    
    # 將混和特徵轉 raw 特徵
    slice_sampled_features()
    
if __name__ == "__main__":
    filename = os.path.splitext(os.path.basename(__file__))[0]
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(os.path.abspath(os.path.join(__file__ , f"../../logs/{filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log")))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)", datefmt = "%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)", datefmt = "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console)

    main()