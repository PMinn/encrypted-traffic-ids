from pathlib import Path
import re
import datetime
from tqdm import tqdm
import logging
from utils.alias import a2p

def rewrite_pcap():
    from data_processing.save_pcap import pcapng_to_pcap
    logging.getLogger("scapy.runtime").setLevel(logging.ERROR) # 忽略 Scapy 的警告訊息
    logger = logging.getLogger("rewrite_pcap")
    pcaps = a2p('@/data/CIC-IDS-2017/CIC-IDS-2017').glob("*")
    logger.debug(f"Total pcaps to rewrite: {len(pcaps)}")
    with tqdm(total = len(pcaps)) as pbar:
        for pcap in pcaps:
            output_file = pcap.as_posix().replace('/CIC-IDS-2017/CIC-IDS-2017/', '/CIC-IDS-2017/rewrite/')
            output_file.parent.mkdir(parents = True, exist_ok = True)
            if output_file.exists():
                pbar.update(1)
                continue
            try:
                pcapng_to_pcap(input_file = pcap, output_file = output_file)
            except Exception as e:
                logger.error(f"Error processing {pcap}: {e}", exc_info = True)
            pbar.update(1)
            
def split_pcap():
    from data_processing.split_to_flows import split_to_flows_from_file
    split_to_flows_from_file(
        input_file = a2p("@/data/CIC-IDS-2017/rewrite/Tuesday-WorkingHours.pcap"), # 輸入檔案路徑
        output_dir = a2p("@/data/CIC-IDS-2017/split/Tuesday/"), # 輸出資料夾路徑
        remove_original = False
    )
    split_to_flows_from_file(
        input_file = a2p("@/data/CIC-IDS-2017/rewrite/Wednesday-workingHours.pcap"), # 輸入檔案路徑
        output_dir = a2p("@/data/CIC-IDS-2017/split/Wednesday/"), # 輸出資料夾路徑
        remove_original = False
    )
    split_to_flows_from_file(
        input_file = a2p("@/data/CIC-IDS-2017/rewrite/Thursday-WorkingHours.pcap"), # 輸入檔案路徑
        output_dir = a2p("@/data/CIC-IDS-2017/split/Thursday/"), # 輸出資料夾路徑
        remove_original = False
    )
    split_to_flows_from_file(
        input_file = a2p("@/data/CIC-IDS-2017/rewrite/Friday-WorkingHours.pcap"), # 輸入檔案路徑
        output_dir = a2p("@/data/CIC-IDS-2017/split/Friday/"), # 輸出資料夾路徑
        remove_original = False
    )
    split_to_flows_from_file(
        input_file = a2p("@/data/CIC-IDS-2017/rewrite/Monday-WorkingHours.pcap"), # 輸入檔案路徑
        output_dir = a2p("@/data/CIC-IDS-2017/split/Monday/"), # 輸出資料夾路徑
        remove_original = False
    )

def get_features_0307():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = a2p("@/data/CIC-IDS-2017/split/Monday").glob('split_*/*.pcap')
    flow_to_features_file(
        pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Monday-03-07-2017/benign_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape
    )

def get_features_0407():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = a2p("@/data/CIC-IDS-2017/split/Tuesday").glob('split_*/*.pcap')
    ftp_pataor_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_21\.', str(pcap))]
    flow_to_features_file(
        ftp_pataor_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Tuesday-04-07-2017/ftp_pataor_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499170672 <= pkts[0].time <= 1499174417
    )
    ssh_pataor_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_22\.', str(pcap))]
    flow_to_features_file(
        ssh_pataor_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Tuesday-04-07-2017/ssh_pataor_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499188141 <= pkts[0].time <= 1499195060
    )

def get_features_0507():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = a2p("@/data/CIC-IDS-2017/split/Wednesday").glob('split_*/*.pcap')
    dos_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', str(pcap))]
    flow_to_features_file(
        dos_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_hulk_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: (1499262203 <= pkts[0].time <= 1499263642)
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_goldeneye_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499263803 <= pkts[0].time <= 1499264409
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_slowloris_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499258934 <= pkts[0].time <= 1499260279
    )
    flow_to_features_file(
        dos_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_slowhttptest_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499260537 <= pkts[0].time <= 1499261870
    )
    heartbleed_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_45022_192-168-10-51_444\.', str(pcap))]
    flow_to_features_file(
        heartbleed_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/heartbleed_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499278335 <= pkts[0].time <= 1499279564
    )

def get_features_0607():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = a2p("@/data/CIC-IDS-2017/split/Thursday").glob('split_*/*.pcap')
    web_attack_sql_injection_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', str(pcap))]
    flow_to_features_file(
        web_attack_sql_injection_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_sql_injection_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499348127 <= pkts[0].time <= 1499348576
    )
    web_attack_xss_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_(?!(36180|36182|36184|36186|36188|36190)$)_192-168-10-50_80\.', str(pcap))]
    flow_to_features_file(
        web_attack_xss_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_xss_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499346935 <= pkts[0].time <= 1499348122
    )
    web_attack_brute_force_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_80\.', str(pcap))]
    flow_to_features_file(
        web_attack_brute_force_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_brute_force_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499343354 <= pkts[0].time <= 1499346012
    )
    infiltration2_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_205-174-165-73_', str(pcap))]
    flow_to_features_file(
        infiltration2_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration2_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499361542 <= pkts[0].time <= 1499366770
    )
    infiltration1_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-25_.*_205-174-165-73_', str(pcap))]
    flow_to_features_file(
        infiltration1_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration1_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499363616 <= pkts[0].time <= 1499371340
    )
    infiltration31_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_(50122|50133)_192-168-10-51_', str(pcap))]
    flow_to_features_file(
        infiltration31_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration31_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499360431 <= pkts[0].time <= 1499360446
    )
    infiltration32_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_192-168-10-5_', str(pcap))]
    flow_to_features_file(
        infiltration32_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration32_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499362410 <= pkts[0].time <= 1499362445
    )
    infiltration33_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-8_.*_192-168-10-(5|9|12|14|15|16|17|19|25|50|51)_', str(pcap))]
    flow_to_features_file(
        infiltration33_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration33_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499364314 <= pkts[0].time <= 1499366765
    )
    
def get_features_0707():
    from data_processing.features import flow_to_features_file
    packet_shape = (96, 5)
    pcaps = a2p('@/data/CIC-IDS-2017/split/Friday').glob('split_*/*.pcap')
    infiltration_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_', str(pcap))]
    flow_to_features_file(
        infiltration_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Friday-07-07-2017/infiltration_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499446532 <= pkts[0].time <= 1499447949 or 1499449905 <= pkts[0].time <= 1499451842
    )
    botnet_ares_pcaps = [pcap for pcap in pcaps if re.search(r'_192-168-10-(15|9|14|5|8)_.*_205-174-165-73_', str(pcap))]
    flow_to_features_file(
        botnet_ares_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Friday-07-07-2017/botnet_ares_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499432653 <= pkts[0].time <= 1499457685
    )
    ddos_pcaps = [pcap for pcap in pcaps if re.search(r'_172-16-0-1_.*_192-168-10-50_', str(pcap))]
    flow_to_features_file(
        ddos_pcaps,
        output_file = a2p(f'@/data/CIC-IDS-2017/features/features/Friday-07-07-2017/ddos_features_{packet_shape[0]}_{packet_shape[1]}_24.npy'),
        packet_shape = packet_shape,
        is_labelled = lambda pcapPath, pkts: 1499453791 <= pkts[0].time <= 1499454973
    )
    
def merge_features():
    from data_processing.features import mearge_feature_files, copy_feature_file
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Monday-03-07-2017/benign_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/benign_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Tuesday-04-07-2017/ftp_pataor_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/ftp_brute_force_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Tuesday-04-07-2017/ssh_pataor_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/ssh_brute_force_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_slowloris_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/dos_slowloris_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_slowhttptest_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/dos_slowhttptest_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Wednesday-05-07-2017/dos_hulk_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/dos_hulk_features_96_5_24.npy")
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Friday-07-07-2017/ddos_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/ddos_features_96_5_24.npy")
    )
    mearge_feature_files(
        [
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration31_features_96_5_24.npy"),    
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration32_features_96_5_24.npy"),    
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/infiltration33_features_96_5_24.npy"),    
        ],
        output_file = a2p('@/data/CIC-IDS-2017/features/merged/port_scan_features_96_5_24.npy')
    )
    mearge_feature_files(
        [
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_brute_force_features_96_5_24.npy"),    
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_sql_injection_features_96_5_24.npy"),    
            a2p("@/data/CIC-IDS-2017/features/features/Thursday-06-07-2017/web_attack_xss_features_96_5_24.npy"),    
        ],
        output_file = a2p('@/data/CIC-IDS-2017/features/merged/web_features_96_5_24.npy')
    )
    copy_feature_file(
        a2p("@/data/CIC-IDS-2017/features/features/Friday-07-07-2017/botnet_ares_features_96_5_24.npy"),
        a2p("@/data/CIC-IDS-2017/features/merged/botnet_ares_features_96_5_24.npy")
    )

def split_features_to_train_test():
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split
    train_folder = a2p('@/data/CIC-IDS-2017/features/split/train')
    test_folder = a2p('@/data/CIC-IDS-2017/features/split/test')
    train_folder.mkdir(parents = True, exist_ok = True)
    test_folder.mkdir(parents = True, exist_ok = True)
    features_npys = a2p('@/data/CIC-IDS-2017/features/merged').glob('*.npy')
    result = {
        "train": {},
        "test": {}
    }
    for features_npy in features_npys:
        filename = features_npy.name
        data = np.load(str(features_npy), allow_pickle = True)
        number_of_data = data.shape[0]
        train_data, test_data = train_test_split(data, test_size = 0.2, random_state = 42)
        number_of_train_data = train_data.shape[0]
        number_of_test_data = test_data.shape[0]
        np.save(str(train_folder.joinpath(filename)), train_data, allow_pickle = False)
        np.save(str(test_folder.joinpath(filename)), test_data, allow_pickle = False)
        result["train"][filename] = number_of_train_data
        result["test"][filename] = number_of_test_data
        logger.info(f"{filename}: total={number_of_data}, train={number_of_train_data}, test={number_of_test_data}")
    with open(a2p('@/data/CIC-IDS-2017/features/split/train_test_split_info.json'), 'w') as f:
        json.dump(result, f, indent = 4)

def run_sampling_and_labeling():
    from data_processing.sampling import sampling, get_oversampling_by_kmeans
    sampling(
        [
            {
                'name': 'Benign',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/benign_features_96_5_24.npy'),
                'sampling_count': 64000
            },
            {
                'name': 'FTP-BruteForce',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/ftp_brute_force_features_96_5_24.npy'),
                'sampling_count': 3135
            },
            {
                'name': 'SSH-BruteForce',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/ssh_brute_force_features_96_5_24.npy'),
                'sampling_count': 3500
            },
            {
                'name': 'DoS-Slowloris',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/dos_slowloris_features_96_5_24.npy'),
                'sampling_count': 3089
            },
            {
                'name': 'DoS-SlowHTTPTest',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/dos_slowhttptest_features_96_5_24.npy'),
                'sampling_count': 3368
            },
            {
                'name': 'DoS-Hulk',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/dos_hulk_features_96_5_24.npy'),
                'sampling_count': 4741
            },
            {
                'name': 'DDoS',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/ddos_features_96_5_24.npy'),
                'sampling_count': 12000
            },
            {
                'name': 'Port Scan',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/port_scan_features_96_5_24.npy'),
                'sampling_count': 16000
            },
            {
                'name': 'Web Attacks',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/web_features_96_5_24.npy'),
                'sampling_count': 3500
            },
            {
                'name': 'Bot',
                'data_path': a2p('@/data/CIC-IDS-2017/features/split/train/botnet_ares_features_96_5_24.npy'),
                'sampling_count': 3501
            }
        ],
        save_to = a2p('@/data/CIC-IDS-2017/features/sampled/train/'),
        oversampling = get_oversampling_by_kmeans
    )

def main():
    # 將 pcapng 轉成 pcap
    # rewrite_pcap()

    # 分離成 Flows
    # split_pcap()

    # 特徵擷取
    # get_features_0307()
    # get_features_0407()
    # get_features_0507()
    # get_features_0607()
    # get_features_0707()
    
    # 合併特徵
    # merge_features()
    
    # 分割成訓練集與測試集
    # split_features_to_train_test()

    # 採樣及標記
    # run_sampling_and_labeling()

    return
    
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)", datefmt = "%Y-%m-%d %H:%M:%S")
    
    file_handler = logging.FileHandler(a2p("@/logs") / f"{Path(__file__).stem}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    main()