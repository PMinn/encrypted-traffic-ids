import glob
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Tuple

# import shutil
from data_processing.split_to_flows import split_to_flows_from_file

# from data_processing.sampling_TON import suggest, run_sampling, run_binary_sampling
# from data_processing.save_pcap import save_pcap
from data_processing.filter_encrypted import remove_pcap_if_not_encrypted
import datetime
import pyshark

# from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import logging

from utils.alias import a2p


def split_pcap(input_file: Path, output_dir: Path) -> None:
    split_to_flows_from_file(
        input_file=input_file,  # 輸入檔案路徑
        output_dir=output_dir,  # 輸出資料夾路徑
        remove_original=False,
    )


def filter_encrypted_pcaps(input_dir: Path) -> Tuple[int, int]:
    pcaps = list(input_dir.glob("**/*.pcap"))
    with Pool(int(os.cpu_count()*0.5)) as pool:
        r = list(
            tqdm(
                pool.imap(remove_pcap_if_not_encrypted, pcaps),
            )
        )
    encrypted_pcaps_count = sum([1 for res in r if res])
    total_pcaps_count = len(pcaps)
    return encrypted_pcaps_count, total_pcaps_count


def count_pcaps(path: str) -> None:
    # 正常流量 Benign
    benign_pcaps = glob.glob(
        os.path.join(f"{path}/Benign/**", "*.pcap"), recursive=True
    )
    logger.info(f"Benign pcaps: {len(benign_pcaps)}")
    logger.info(f"-----------------------------------")
    # 惡意流量 Malicious
    malicious_folders = glob.glob(f"{path}/Malicious/*")
    malicious_folders.sort()
    total_malicious_pcaps = 0
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.split("/")[-1]
        malicious_pcaps = glob.glob(
            os.path.join(malicious_folder, "**", "*.pcap"), recursive=True
        )
        logger.info(f"{folder_name} pcaps: {len(malicious_pcaps)}")
        total_malicious_pcaps += len(malicious_pcaps)
    logger.info(f"-----------------------------------")
    logger.info(f"Total Malicious pcaps: {total_malicious_pcaps}")
    logger.info(
        f"Average Malicious pcaps per type: {total_malicious_pcaps / len(malicious_folders)}"
    )


def main() -> None:
    # 正常流量 Benign
    # split_to_flows_from_folder(
    #     input_dir = "/sdc1/ytlindata/CIC-IoT-2023/CICIoT2023/Benign_Final", # 輸入資料夾路徑
    #     output_dir = "/sdc1/ytlindata/CIC-IoT-2023/split/Benign/", # 輸出資料夾路徑
    #     splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False,
    #     logger = logger
    # )
    # 正常流量 Benign
    # benign_pcaps = glob.glob(os.path.join('/sdc1/ytlindata/CIC-IoT-2023/split/Benign/**', "*.pcap"), recursive = True)
    # with Pool(50) as pool:
    #     r = list(tqdm(pool.imap(filter_pcap, benign_pcaps), total=len(benign_pcaps)))
    # logging.info(f"benign has encrypted pcaps: {sum([1 for res in r if res])} / {len(benign_pcaps)}")
    # 惡意流量 Malicious
    attack_folders = a2p("@/data/CIC-IoT-2023/CICIoT2023/")
    for attack_folder in attack_folders.iterdir():
        encrypted_pcaps_count = total_pcaps_count = 0
        attack_folder_name = attack_folder.name
        for index, pcap_file in enumerate(attack_folder.glob("*.pcap")):
            logging.info(f"Processing folder: {attack_folder_name}, file: {pcap_file.name}")
            # 分離成 Flows
            output_dir =  a2p(f"@/data/CIC-IoT-2023/split/Malicious/{attack_folder_name}/split_{index+1}")
            split_pcap(pcap_file, output_dir)
            logging.info(f"Finished splitting folder: {attack_folder_name}/{pcap_file.name} -> /{output_dir.name}")

            # 篩選出加密的 pcap
            temp_encrypted_pcaps_count, temp_total_pcaps_count = filter_encrypted_pcaps(input_dir=output_dir)
            encrypted_pcaps_count += temp_encrypted_pcaps_count
            total_pcaps_count += temp_total_pcaps_count
            logging.info(f"Encrypted pcaps in {attack_folder_name}/{output_dir.name}: {temp_encrypted_pcaps_count} / {temp_total_pcaps_count}")
            
        logging.info(f"Total encrypted pcaps in {attack_folder_name}: {encrypted_pcaps_count} / {total_pcaps_count}")

    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/encrypted_filter')

    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter')
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter_with_decrypt')

    # 特徵擷取
    attack_class = [
        "Injection",
        "MITM",
        # "backdoor",
        "DDoS",
        "DoS",
        # "runsomware",
        "scanning",
        "XSS",
        "password",
    ]
    # run_del(attack_class)

    # 數據預處理，合併特徵並分割訓練集與測試集
    classes = [*["benign"], *attack_class]
    # DatasetProcesser(
    #     data_path = '/sdc1/ytlindata/TON_IoT/del_120_5_flows(delall)',
    #     save_to = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall',
    #     classes = classes
    # ).run()

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
    # 多分類
    UNDERSAMPLING = {
        "Injection": 6702,
        "DDoS": 15151,
        "scanning": 5304,
    }
    OVERSAMPLING = {
        "benign": 1317,
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
        "benign": 5000,
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


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        a2p("@/logs")
        / f"{Path(__file__).stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    main()
