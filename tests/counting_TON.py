import glob
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    

def main():
    # 計算出各類型的數量
    logger.info(f"*** encrypted_filter ***")
    count_pcaps('/sdc1/ytlindata/TON_IoT/encrypted_filter')

    # 計算出各類型的數量
    logger.info(f"*** attack_filter ***")
    count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter')
    
if __name__ == "__main__":
    file_handler = logging.FileHandler(os.path.abspath(os.path.join(__file__ ,"../../logs/TON_counting.log")))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    main()