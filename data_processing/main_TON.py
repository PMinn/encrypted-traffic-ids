import glob
from multiprocessing import Pool
import os
from steps.split_USTC_TFC2016 import split_files
from steps.getFeatures_USTC_TFC2016 import runTCP_del, runUDP_del
from steps.datasetProcessor_USTC_TFC2016 import DatasetProcesser_USTC_TFC2016
from steps.sampling_USTC_TFC2016 import run_sampling, run_binary_sampling
from utils.save_pcap import save_pcap
from steps.filter_encrypted_TON import df, mf
from steps.filter_attack_TON import fa
import datetime
import pyshark
# from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 攻擊類別
classes = [
    'benign',
    "Injection",
    "MITM",
    "backdoor",
    "DDoS",
    "DoS",
    "runsomware",
    "scanning",
    "XSS",
    "password"
]

def rewrite_pcap():
    # 惡意流量 Malware
    attack_pcaps = glob.glob(os.path.join('/sdc1/ytlindata/TON_IoT/Original_dataset/normal_attack_pcaps/**', "*.pcap"), recursive=True)
    with tqdm(total=len(attack_pcaps)) as pbar:
        for pcap in attack_pcaps:
            output_file = pcap.replace('Original_dataset', 'rewrite_dataset')
            folder = "/".join(output_file.split("/")[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            if os.path.exists(output_file):
                pbar.update(1)
                continue
            save_pcap(pcap_file = pcap, output_file = output_file)
            pbar.update(1)
    # 正常流量 Benign
    benign_pcaps = glob.glob('/sdc1/ytlindata/TON_IoT/Original_dataset/normal_pcaps/*')
    with tqdm(total=len(benign_pcaps)) as pbar:
        for pcap in benign_pcaps:
            output_file = pcap.replace('Original_dataset', 'rewrite_dataset')
            folder = "/".join(output_file.split("/")[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            if os.path.exists(output_file):
                pbar.update(1)
                continue
            save_pcap(pcap_file = pcap, output_file = output_file)
            pbar.update(1)

def split_pcap():
    # 正常流量 Benign
    split_files(
        input_dir = "/sdc1/ytlindata/TON_IoT/rewrite_dataset/normal_pcaps/", # 輸入資料夾路徑
        output_dir = "/sdc1/ytlindata/TON_IoT/split/Benign/", # 輸出資料夾路徑
        splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/data_processing/steps/SplitCap.exe" # SplitCap.exe 的路徑
    )
    # 惡意流量 Malicious
    attack_folders = glob.glob('/sdc1/ytlindata/TON_IoT/rewrite_dataset/normal_attack_pcaps/*')
    for folder in attack_folders:
        print(folder)
        folder_name = folder.split('/')[-1]
        split_files(
            input_dir = folder, # 輸入資料夾路徑
            output_dir = f"/sdc1/ytlindata/TON_IoT/split/Malicious/{folder_name}/", # 輸出資料夾路徑
            splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/data_processing/steps/SplitCap.exe" # SplitCap.exe 的路徑
        )

def filter_encrypted_pcaps():
    # 正常流量 Benign
    benign_pcaps = glob.glob(os.path.join('/sdc1/ytlindata/TON_IoT/split/Benign/**', "*.pcap"), recursive = True)
    with Pool(50) as pool:
        r = list(tqdm(pool.imap(df, benign_pcaps), total=len(benign_pcaps)))
    # print(r)
    # 惡意流量 Malicious
    malicious_folders = glob.glob('/sdc1/ytlindata/TON_IoT/split/Malicious/*')
    skip_folders = []
    for malicious_folder in malicious_folders:
        skip = False
        for skip_folder in skip_folders:
            if skip_folder in malicious_folder:
                skip = True
                continue
        if skip:
            continue
        print(f"Processing folder: {malicious_folder}")
        folder_name = malicious_folder.split('/')[-1]
        malicious_pcaps = glob.glob(os.path.join(malicious_folder, "**", "*.pcap"), recursive = True)
        with Pool(50) as pool:
            r = list(tqdm(pool.imap(mf, malicious_pcaps), total=len(malicious_pcaps)))
        print(f"Finished processing folder: {folder_name}")
        
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
    

def filter_attack():
    # CSVs 裡面有紀錄每個 pcap 的攻擊類型
    attack_csvs = glob.glob('/sdc1/ytlindata/TON_IoT/SecurityEvents_Network_datasets/*')
    attack_dict = {}
    for attack_csv in attack_csvs:
        # ts,src_ip,src_port,dst_ip,dst_port,proto,type
        with open(attack_csv, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                ts = int(float(parts[0]))
                src_ip = parts[1]
                src_port = parts[2]
                dst_ip = parts[3]
                dst_port = parts[4]
                proto = parts[5]
                attack_type = parts[6]
                key = (src_ip, src_port, dst_ip, dst_port, proto)
                if attack_type not in attack_dict:
                    attack_dict[attack_type] = {}
                if key not in attack_dict[attack_type]:
                    attack_dict[attack_type][key] = []
                attack_dict[attack_type][key].append(ts)
    logger.info(f"Total attack records: {len(attack_dict)}")
    for attack_type in attack_dict:
        logger.info(f"Attack type: {attack_type}, records: {len(attack_dict[attack_type])}")
    csv_type2pcap_filder_map = {
        "scanning": "normal_scanning",
        "ddos": "normal_DDoS",
        "password": "password_normal",
        "xss": "normal_XSS",
        "dos": "normal_DoS",
        "injection": "Injection_normal",
        "ransomware": "normal_runsomware",
        "backdoor": "normal_backdoor",
        "mitm": "MITM_normal",
    }
    skip_attack_types = [
        "scanning",
        "ddos",
        "password",
        "xss",
        "dos",
        "injection"
    ]
    for attack_type in attack_dict:
        if attack_type in skip_attack_types:
            continue
        pcap_folder = csv_type2pcap_filder_map[attack_type]
        attack_pcaps = glob.glob(os.path.join(f'/sdc1/ytlindata/TON_IoT/encrypted_filter/Malicious/{pcap_folder}/**', "*.pcap"), recursive=True)
        # args = [(attack_dict[attack_type], pcap) for pcap in attack_pcaps]
        # with Pool(50) as pool:
        #     r = list(tqdm(pool.imap(fa, args), total=len(args)))
        for pcap in tqdm(attack_pcaps):
            fa((attack_dict[attack_type], pcap))

def main():
    # 有些 pcap 切分工具無法讀取，可以用這個步驟重新存檔一次
    # rewrite_pcap()
    # 分離成 Flows
    # split_pcap()
    # 篩選出加密的 pcap
    # filter_encrypted_pcaps()
    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/encrypted_filter)
    # 篩選攻擊
    # filter_attack()
    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter')
    # 2. 特徵擷取
    # TCP
    with Pool(10) as p:
        p.map(runTCP_del, ['Benign', 'Malware'])
    # UDP
    with Pool(10) as p:
        p.map(runUDP_del, ['Benign', 'Malware'])
    
if __name__ == "__main__":
    file_handler = logging.FileHandler('./log/TON_attack_filter_counting.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    main()
    # 3. 數據預處理，合併特徵並分割訓練集與測試集
    # DatasetProcesser_USTC_TFC2016(
    #     ORIG_DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/del_120_5_flows(delall)/',
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall',
    #     classes = classes
    # ).run()
    
    # 4. 不平衡資料處理
    UNDERSAMPLING = { 
        'benign': 4000*15,
        "Cridex": 4000*3, 
        "Neris": 4000*3, 
        "Virut": 4000*4,
    }
    OVERSAMPLING = {
        "Geodo": 8000,
        "Htbot": 8000,
        "Miuref": 7000,
        "Nsis-ay": 7500,
        "Shifu": 9500,
        "Tinba": 9000,
        "Zeus": 10000
    }
    # 多分類
    # run_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/train',
    #     MULTI_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = False
    # )
    # 產生 testing 資料 (不須 testing 的話可以不用跑)
    # run_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/test',
    #     MULTI_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/test/sampling',
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = True
    # )
    # 二分類
    # run_binary_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/train',
    #     BINARY_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/binary_sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = False
    # )
    # 產生二分類資料 (不須 testing 的話可以不用跑)
    # run_binary_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/test',
    #     BINARY_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/test/binary_sampling',
    #     UNDERSAMPLING = UNDERSAMPLING,
    #     OVERSAMPLING = OVERSAMPLING,
    #     isTestingData = True
    # )
    pass