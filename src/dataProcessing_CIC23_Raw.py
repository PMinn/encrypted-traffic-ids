import glob
from multiprocessing import Pool
import os
from pathlib import Path

# import shutil
from data_processing.split_to_flows import split_to_flows_from_folder

# from data_processing.sampling_TON import suggest, run_sampling, run_binary_sampling
# from data_processing.save_pcap import save_pcap
from data_processing.filter_encrypted import remove_pcap_if_not_encrypted
import datetime
import pyshark

# from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import logging

from utils.alias import a2p


def split_pcap() -> None:
    # 正常流量 Benign
    # split_to_flows_from_folder(
    #     input_dir = "/sdc1/ytlindata/CIC-IoT-2023/CICIoT2023/Benign_Final", # 輸入資料夾路徑
    #     output_dir = "/sdc1/ytlindata/CIC-IoT-2023/split/Benign/", # 輸出資料夾路徑
    #     splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/src/data_processing/SplitCap.exe", # SplitCap.exe 的路徑
    #     remove_original = False,
    #     logger = logger
    # )
    # 惡意流量 Malicious
    attack_folders = a2p("@/data/CIC-IoT-2023/CICIoT2023/")
    for folder in attack_folders.iterdir():
        folder_name = folder.name
        print(f"Processing: {folder_name}")
        split_to_flows_from_folder(
            input_dir=folder,  # 輸入資料夾路徑
            output_dir=a2p(
                f"@/data/CIC-IoT-2023/split/Malicious/{folder_name}/"
            ),  # 輸出資料夾路徑
            remove_original=False,
        )


def filter_encrypted_pcaps() -> None:
    logger = logging.getLogger("filter_encrypted_pcaps")
    # 正常流量 Benign
    # benign_pcaps = glob.glob(os.path.join('/sdc1/ytlindata/CIC-IoT-2023/split/Benign/**', "*.pcap"), recursive = True)
    # with Pool(50) as pool:
    #     r = list(tqdm(pool.imap(filter_pcap, benign_pcaps), total=len(benign_pcaps)))
    # logger.info(f"benign has encrypted pcaps: {sum([1 for res in r if res])} / {len(benign_pcaps)}")
    # 惡意流量 Malicious
    malicious_folders = a2p("@/data/CIC-IoT-2023/split/Malicious/")
    skip_folders: list[str] = []
    for malicious_folder in malicious_folders.iterdir():
        folder_name = malicious_folder.name
        if folder_name in skip_folders:
            continue
        logger.debug(f"Processing: {folder_name}")
        malicious_pcaps = list(malicious_folder.glob("**/*.pcap"))
        with Pool(50) as pool:
            r = list(
                tqdm(
                    pool.imap(remove_pcap_if_not_encrypted, malicious_pcaps),
                    total=len(malicious_pcaps),
                )
            )
        logger.info(
            f"{folder_name} has encrypted pcaps: {sum([1 for res in r if res])} / {len(malicious_pcaps)}"
        )


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
    # 分離成 Flows
    split_pcap()

    # 篩選出加密的 pcap
    # filter_encrypted_pcaps()

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
