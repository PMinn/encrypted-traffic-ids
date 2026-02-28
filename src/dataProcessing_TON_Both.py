from typing import cast
from pathlib import Path
from multiprocessing import Pool
import os

from tabulate import tabulate
from data_processing.save_pcap import save_pcap
import datetime
from tqdm import tqdm
import logging
from utils.alias import a2p
from utils.notification import send_push_message


def rewrite_pcap() -> None:
    # # 惡意流量 Malware
    # attack_pcaps = a2p("@data/data/TON_IoT/Original_dataset/normal_attack_pcaps").glob(
    #     "**/*.pcap"
    # )
    # with tqdm(total=len(cast(list[Path], attack_pcaps))) as pbar:
    #     for pcap in attack_pcaps:
    #         output_file = Path(
    #             pcap.as_posix().replace("/Original_dataset/", "/rewrite/")
    #         )
    #         output_file.parent.mkdir(parents=True, exist_ok=True)
    #         if output_file.exists():
    #             pbar.update(1)
    #             continue
    #         save_pcap(pcap_file=pcap, output_file=output_file)
    #         pbar.update(1)
    # # 正常流量 Benign
    # benign_pcaps = a2p("@data/data/TON_IoT/Original_dataset/normal_pcaps").glob(
    #     "**/*.pcap"
    # )
    # with tqdm(total=len(cast(list[Path], benign_pcaps))) as pbar:
    #     for pcap in benign_pcaps:
    #         output_file = Path(
    #             pcap.as_posix().replace("/Original_dataset/", "/rewrite/")
    #         )
    #         output_file.parent.mkdir(parents=True, exist_ok=True)
    #         if output_file.exists():
    #             pbar.update(1)
    #             continue
    #         if os.path.exists(output_file):
    #             pbar.update(1)
    #             continue
    #         save_pcap(pcap_file=pcap, output_file=output_file)
    #         pbar.update(1)
    pcap = a2p(
        "@data/data/TON_IoT/rewrite_dataset/normal_attack_pcaps/normal_scanning/normal_scanning3.pcap"
    )
    output_file = a2p(
        "@data/data/TON_IoT/rewrite_dataset/normal_attack_pcaps/normal_scanning/normal_scanning3_fixed.pcap"
    )
    save_pcap(pcap_file=pcap, output_file=output_file)


def split_pcap() -> None:
    from data_processing.split_to_flows import (
        split_to_flows_from_folder,
        split_to_flows_from_file,
    )

    # # 正常流量 Benign
    # split_to_flows_from_folder(
    #     input_dir=a2p("@data/data/TON_IoT/rewrite_dataset/normal_pcaps/"),  # 輸入資料夾路徑
    #     output_dir=a2p("@data/data/TON_IoT/split/Benign/"),  # 輸出資料夾路徑
    # )
    # # 惡意流量 Malicious
    # attack_folders = a2p("@data/data/TON_IoT/rewrite_dataset/normal_attack_pcaps").iterdir()
    # for folder in attack_folders:
    #     folder_name = folder.name
    #     split_to_flows_from_folder(
    #         input_dir=folder,  # 輸入資料夾路徑
    #         output_dir=a2p(
    #             f"@data/data/TON_IoT/split/Malicious/{folder_name}/"
    #         ),  # 輸出資料夾路徑
    #     )
    # 意外修正
    split_to_flows_from_file(
        input_file=a2p(
            "@data/data/TON_IoT/rewrite_dataset/normal_attack_pcaps/normal_scanning/normal_scanning3.pcap"
        ),  # 輸入資料夾路徑
        output_dir=a2p(
            "@data/data/TON_IoT/split/Malicious/normal_scanning/split_3"
        ),  # 輸出資料夾路徑
    )


def filter_encrypted_pcaps() -> None:
    from data_processing.filter_encrypted import remove_pcap_if_not_encrypted

    # 正常流量 Benign
    benign_pcaps = a2p("@data/data/TON_IoT/split/Benign").glob("**/*.pcap")
    with Pool(50) as pool:
        r = list(
            tqdm(
                pool.imap(remove_pcap_if_not_encrypted, benign_pcaps),
                total=len(cast(list[Path], benign_pcaps)),
            )
        )
    logging.getLogger("filter_encrypted_pcaps").info(r)
    # 惡意流量 Malicious
    malicious_folders = a2p("@data/data/TON_IoT/split/Malicious").iterdir()
    skip_folders: list[Path] = []
    for malicious_folder in malicious_folders:
        skip = False
        for skip_folder in skip_folders:
            if skip_folder == malicious_folder:
                skip = True
                break
        if skip:
            continue
        logging.getLogger("filter_encrypted_pcaps").info(
            f"Processing folder: {malicious_folder}"
        )
        folder_name = malicious_folder.name
        malicious_pcaps = malicious_folder.glob("**/*.pcap")
        with Pool(50) as pool:
            r = list(
                tqdm(
                    pool.imap(remove_pcap_if_not_encrypted, malicious_pcaps),
                    total=len(cast(list[Path], malicious_pcaps)),
                )
            )
        logging.getLogger("filter_encrypted_pcaps").info(
            f"Finished processing folder: {folder_name}"
        )


def filter_attack() -> None:
    from data_processing.filter_attack import run_attack_filter
    from data_processing.save_pcap import copy_pcap

    # CSVs 裡面有紀錄每個 pcap 的攻擊類型
    attack_csvs = a2p("@data/data/TON_IoT/SecurityEvents_Network_datasets").glob(
        "*.csv"
    )
    attack_dict: dict[str, dict[tuple[str, str, str, str, str], list[int]]] = {}
    skip_attack_types: list[str] = [
        # "scanning",
        "ddos",
        "password",
        "xss",
        "dos",
        "injection",
        "ransomware",
        "backdoor",
        "mitm",
    ]
    for attack_csv in attack_csvs:
        # ts,src_ip,src_port,dst_ip,dst_port,proto,type
        with open(attack_csv, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                attack_type = parts[6]
                if attack_type in skip_attack_types:
                    continue
                ts = int(parts[0])
                src_ip = parts[1]
                src_port = parts[2]
                dst_ip = parts[3]
                dst_port = parts[4]
                proto = parts[5]
                key = (src_ip, src_port, dst_ip, dst_port, proto)
                if attack_type not in attack_dict:
                    attack_dict[attack_type] = {}
                if key not in attack_dict[attack_type]:
                    attack_dict[attack_type][key] = []
                attack_dict[attack_type][key].append(ts)
    logger.info(f"Total attack records: {len(attack_dict)}")
    for attack_type in attack_dict:
        logger.info(
            f"Attack type: {attack_type}, records: {len(attack_dict[attack_type])}"
        )
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

    def handle_filter(pcap: Path, is_attack: bool) -> None:
        if is_attack:
            copy_pcap(
                src_file=pcap,
                dest_file=Path(pcap.as_posix().replace("/split/", "/attack_filter/")),
            )
        else:
            # copy_pcap(
            #     src_file=pcap,
            #     dest_file=Path(
            #         pcap.as_posix()
            #         .replace("/split/", "/attack_filter/")
            #         .replace("/Malicious/", "/Benign/")
            #     ),
            # )
            pass

    for attack_type in attack_dict:
        if attack_type in skip_attack_types:
            continue
        pcap_folder = csv_type2pcap_filder_map[attack_type]
        attack_pcaps = a2p(f"@data/data/TON_IoT/split/Malicious/{pcap_folder}").glob(
            "**/*.pcap"
        )
        # args = [(attack_dict[attack_type], pcap) for pcap in attack_pcaps]
        # with Pool(10) as pool:
        #     r = list(tqdm(pool.imap(fa, args), total=len(args)))
        malicious_count = 0
        benign_count = 0
        for pcap in tqdm(attack_pcaps):
            is_attack = run_attack_filter(
                (attack_dict[attack_type], pcap, handle_filter)
            )
            if is_attack:
                malicious_count += 1
            else:
                benign_count += 1
        logger.info(
            f"type: {attack_type}, attack pcaps: {malicious_count}, benign pcaps: {benign_count}"
        )


def get_features() -> None:
    from data_processing.features import flow_to_features_file

    packet_shape = (96, 5)
    pcaps = a2p("@data/data/TON_IoT/encrypt_aes/Benign").glob("**/*.pcap")
    flow_to_features_file(
        pcaps,
        output_file=a2p(
            f"@data/data/TON_IoT/features_aes/features/benign_features_{packet_shape[0]}_{packet_shape[1]}.npy"
        ),
        packet_shape=packet_shape,
    )

    malicious_folders = a2p("@data/data/TON_IoT/encrypt_aes/Malicious").iterdir()
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.stem.replace("normal_", "").replace(
            "_normal", ""
        )
        pcaps = malicious_folder.glob("**/*.pcap")
        flow_to_features_file(
            pcaps,
            output_file=a2p(
                f"@data/data/TON_IoT/features_aes/features/{folder_name}_features_{packet_shape[0]}_{packet_shape[1]}.npy"
            ),
            packet_shape=packet_shape,
        )


def split_features_to_train_test() -> None:
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split

    train_folder = a2p("@data/data/TON_IoT/features_rsa/split/train")
    test_folder = a2p("@data/data/TON_IoT/features_rsa/split/test")
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)
    features_npys = a2p("@data/data/TON_IoT/features_rsa/features").glob("*.npy")
    result: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for features_npy in features_npys:
        filename = features_npy.stem
        data = np.load(str(features_npy), allow_pickle=True)
        number_of_data = data.shape[0]
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        number_of_train_data = train_data.shape[0]
        number_of_test_data = test_data.shape[0]
        np.save(str(train_folder.joinpath(filename)), train_data, allow_pickle=False)
        np.save(str(test_folder.joinpath(filename)), test_data, allow_pickle=False)
        result["train"][filename] = number_of_train_data
        result["test"][filename] = number_of_test_data
        logger.info(
            f"{filename}: total={number_of_data}, train={number_of_train_data}, test={number_of_test_data}"
        )
    with open(
        a2p("@data/data/TON_IoT/features_rsa/split/train_test_split_info.json"),
        "w",
    ) as f:
        json.dump(result, f, indent=4)


def run_sampling_and_labeling(features_folder: str, alpha: float) -> None:
    from data_processing.sampling_and_labeling import (
        sampling,
        labeling,
        get_oversampling_by_kmeans,
    )
    import json

    result_folder = a2p(
        f"@data/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P"
    )

    with open(result_folder / "sampling_suggestion.json", "r") as f:
        sampling_suggestion = json.load(f)

    assert sampling_suggestion["alpha"] == alpha, "Alpha 與建議檔案不符！"
    sampling(
        [
            {
                "name": "Benign",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/train/benign_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "Benign"
                ],
            },
            {
                "name": "scanning",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/train/scanning_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "scanning"
                ],
            },
            {
                "name": "DDoS",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/train/DDoS_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "DDoS"
                ],
            },
            {
                "name": "Injection",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/train/Injection_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "Injection"
                ],
            },
        ],
        save_to=a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/train/"
        ),
        oversampling=get_oversampling_by_kmeans,
    )

    labeling(
        [
            {
                "name": "Benign",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/test/benign_features_96_5_73.npy"
                ),
            },
            {
                "name": "scanning",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/test/scanning_features_96_5_73.npy"
                ),
            },
            {
                "name": "DDoS",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/test/DDoS_features_96_5_73.npy"
                ),
            },
            {
                "name": "Injection",
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/split/test/Injection_features_96_5_73.npy"
                ),
            },
        ],
        save_to=a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/test/"
        ),
    )


def get_suggestion(features_folder: str, alpha: float) -> None:
    from data_processing.sampling_and_labeling import suggester

    suggester(
        {
            "benign": 27000,
            "scanning": 27000,
            "DDoS": 27000,
            "Injection": 27000,
            "runsomware": 1699,
            "backdoor": 15076,
            "MITM": 246,
            "XSS": 27000,
            "DoS": 27000,
            "password": 27000,
        },
        alpha=alpha,
        json_path=a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/sampling_suggestion.json"
        ),
    )


def count_pcaps(folder_path: Path) -> None:
    benig_pcaps = folder_path.joinpath("Benign").glob("**/*.pcap")
    logging.getLogger("count_pcaps").info(
        f"Benign pcaps: {sum(1 for _ in benig_pcaps)}"
    )
    del benig_pcaps
    malicious_folders = folder_path.joinpath("Malicious").iterdir()
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.name
        count = 0
        for split_folder in malicious_folder.iterdir():
            print(f"處理資料夾: {folder_name}/{split_folder.name}")
            split_pcaps = split_folder.glob("**/*.pcap")
            count += sum(1 for _ in split_pcaps)
            del split_pcaps
        logging.getLogger("count_pcaps").info(
            f"Malicious pcaps in {folder_name}: {count}"
        )


def encrypt_rsa_pcap() -> None:
    from cryptography.hazmat.primitives import serialization
    from data_processing.encrypt.rsa import encrypt_pcap

    with open(a2p("@data/data/TON_IoT/keys/rsa/public_key.pem"), "rb") as f:
        PUB = serialization.load_pem_public_key(f.read())

    # 正常流量 Benign
    split_folders = a2p("@data/data/TON_IoT/split/Benign").iterdir()
    for split_folder in split_folders:
        split_folder_name = split_folder.name
        logging.getLogger("encrypt_pcap").info(
            f"Processing folder: Benign/{split_folder_name}"
        )
        benign_pcaps = split_folder.glob("**/*.pcap")
        for pcap in tqdm(benign_pcaps):
            output_file = Path(pcap.as_posix().replace("/split/", "/encrypt_rsa/"))
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.exists():
                continue
            encrypt_pcap(in_pcap=pcap, out_pcap=output_file, PUB=PUB)
    # 惡意流量 Malicious
    # malicious_folders = a2p(
    #     "@data/data/TON_IoT/attack_filter/Malicious"
    # ).iterdir()
    # for malicious_folder in malicious_folders:
    #     folder_name = malicious_folder.name
    #     for split_folder in malicious_folder.iterdir():
    #         split_folder_name = split_folder.name
    #         logging.getLogger("encrypt_pcap").info(
    #             f"Processing folder: {folder_name}/{split_folder_name}"
    #         )
    #         malicious_pcaps = split_folder.glob("**/*.pcap")
    #         for pcap in tqdm(malicious_pcaps):
    #             output_file = Path(
    #                 pcap.as_posix().replace("/attack_filter/", "/encrypt_rsa/")
    #             )
    #             if output_file.exists():
    #                 continue
    #             encrypt_pcap(in_pcap=pcap, out_pcap=output_file, PUB=PUB)


def encrypt_aes_pcap() -> None:
    from data_processing.encrypt.aes import encrypt_pcap

    # 正常流量 Benign
    # split_folders = a2p("@data/data/TON_IoT/split/Benign").iterdir()
    # for split_folder in split_folders:
    #     split_folder_name = split_folder.name
    #     logging.getLogger("encrypt_pcap").info(
    #         f"Processing folder: Benign/{split_folder_name}"
    #     )
    #     benign_pcaps = split_folder.glob("**/*.pcap")
    #     for pcap in tqdm(benign_pcaps):
    #         output_file = Path(
    #             pcap.as_posix().replace("/split/", "/encrypt_aes/")
    #         )
    #         output_file.parent.mkdir(parents=True, exist_ok=True)
    #         if output_file.exists():
    #             continue
    #         encrypt_pcap(in_pcap=pcap, out_pcap=output_file)
    # 惡意流量 Malicious
    malicious_folders = a2p("@data/data/TON_IoT/attack_filter/Malicious").iterdir()
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.name
        for split_folder in malicious_folder.iterdir():
            split_folder_name = split_folder.name
            logging.getLogger("encrypt_pcap").info(
                f"Processing folder: {folder_name}/{split_folder_name}"
            )
            malicious_pcaps = split_folder.glob("**/*.pcap")
            for pcap in tqdm(malicious_pcaps):
                output_file = Path(
                    pcap.as_posix().replace("/attack_filter/", "/encrypt_aes/")
                )
                if output_file.exists():
                    continue
                encrypt_pcap(in_pcap=pcap, out_pcap=output_file)

    
def run_sampling_labeling_and_split_to_train_test(features_folder) -> None:
    import shutil
    import json
    from data_processing.sampling_and_labeling import sampling
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tabulate import tabulate

    number_of_data = {
        "benign": {
            "train": 27000,
            "test": 3000,
        },
        "scanning": {
            "train": 27000,
            "test": 3000,
        },
        "DDoS": {
            "train": 27000,
            "test": 3000,
        },
        "Injection": {
            "train": 27000,
            "test": 3000,
        },
        "runsomware": {
            "train": 1699,
            "test": 728,
        },
        "backdoor": {
            "train": 15076,
            "test": 1675,
        },
        "MITM": {
            "train": 246,
            "test": 105,
        },
        "XSS": {
            "train": 27000,
            "test": 3000,
        },
        "DoS": {
            "train": 27000,
            "test": 3000,
        },
        "password": {
            "train": 27000,
            "test": 3000,
        },
    }

    sampling(
        [
            {
                "name": key,
                "data_path": a2p(
                    f"@data/data/TON_IoT/{features_folder}/features/{key}_features_96_5.npy"
                ),
                "sampling_count": value["train"] + value["test"],
            }
            for key, value in number_of_data.items()
        ],
        save_to=a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled/mix/"
        )
    )

    with open(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/mix/sampled.json"), "r"
    ) as f:
        sampled_info = json.load(f)
        classes = sampled_info["classes"]

    if classes is None:
        raise ValueError("sampled.json 中缺少 classes 資訊！")

    # split to train and test
    data = np.load(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/mix/sampled_data.npy"),
        allow_pickle=True,
    )
    labels = np.load(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/mix/sampled_label.npy"),
        allow_pickle=True,
    )
    train_data_list = []
    train_labels_list = []
    test_data_list = []
    test_labels_list = []
    for class_index, class_name in enumerate(classes):
        class_data = data[labels == class_index]
        train_count = number_of_data[class_name]["train"]
        test_count = number_of_data[class_name]["test"]
        train_data, test_data = train_test_split(
            class_data, test_size=test_count, train_size=train_count, random_state=42
        )
        train_data_list.append(train_data)
        train_labels_list.append(np.full(train_count, class_index))
        test_data_list.append(test_data)
        test_labels_list.append(np.full(test_count, class_index))
    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_data = np.concatenate(test_data_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    target_folder = a2p(f"@data/data/TON_IoT/{features_folder}/sampled")
    (target_folder / "train").mkdir(parents=True, exist_ok=True)
    (target_folder / "test").mkdir(parents=True, exist_ok=True)
    np.save(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/train/sampled_data.npy"),
        train_data,
        allow_pickle=False,
    )
    np.save(
        a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled/train/sampled_label.npy"
        ),
        train_labels,
        allow_pickle=False,
    )
    np.save(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/test/sampled_data.npy"),
        test_data,
        allow_pickle=False,
    )
    np.save(
        a2p(f"@data/data/TON_IoT/{features_folder}/sampled/test/sampled_label.npy"),
        test_labels,
        allow_pickle=False,
    )
    logging.getLogger("run_sampling_labeling_and_split_to_train_test").info(
        "\n"
        + tabulate(
            [
                [
                    class_name,
                    np.sum(train_labels == class_index),
                    np.sum(test_labels == class_index),
                ]
                for class_index, class_name in enumerate(classes)
            ],
            headers=["Class", "Number of Train Samples", "Number of Test Samples"]
        )
    )
    shutil.copyfile(str(a2p(f"@data/data/TON_IoT/{features_folder}/sampled/mix/sampled.json")), str(a2p(f"@data/data/TON_IoT/{features_folder}/sampled/train/sampled.json")))

def rerun_sampling_labeling_and_split_to_train_test(features_folder) -> None:
    from data_processing.sampling_and_labeling import sampling
    import json
    target_folder = a2p(f"@data/data/TON_IoT/{features_folder}/sampled_70P")
    suggestion = json.load(open(target_folder / "sampling_suggestion.json", "r"))
    sampling(
        [
            {
                "name": key,
                "sampling_count": value,
            }
            for key, value in suggestion["number_of_samples_suggested"].items()
        ],
        save_to=(target_folder / "train"),
        data_path=a2p(
            f"@data/data/TON_IoT/{features_folder}/sampled/train"
        )
    )
def main() -> None:
    # 有些 pcap 切分工具無法讀取，可以用這個步驟重新存檔一次
    # rewrite_pcap()

    # 分離成 Flows
    # split_pcap()
    # send_push_message(
    #     job_name="TON_IoT 數據處理",
    #     message="Pcap 分離成 Flows 完成！",
    # )

    # 篩選出加密的 pcap
    # filter_encrypted_pcaps()

    # 計算出各類型的數量
    # count_pcaps(a2p("@data/data/TON_IoT/split/"))

    # 篩選攻擊
    # filter_attack()
    # send_push_message(
    #     job_name="TON_IoT 數據處理",
    #     message="攻擊流量篩選完成！",
    # )

    # 加密流量 RSA 加密
    # encrypt_rsa_pcap()
    # send_push_message(
    #     job_name="TON_IoT 數據處理",
    #     message="攻擊流量加密完成！",
    # )

    # 加密流量 AES 加密
    # encrypt_aes_pcap()
    # send_push_message(
    #     job_name="TON_IoT 數據處理",
    #     message="攻擊流量加密完成！",
    # )

    # 計算出各類型的數量
    # count_pcaps(a2p("@data/data/TON_IoT/attack_filter_fixed_time"))
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter_with_decrypt')

    # 特徵擷取
    # get_features()
    # send_push_message(
    #     job_name="TON_IoT 數據處理",
    #     message="特徵擷取完成！",
    # )

    # 分割成訓練集與測試集
    # split_features_to_train_test()

    # features_folder = "features" | "features_nonMalucusIsBenign"
    # 採樣及標記
    # run_sampling_and_labeling("features_fixed_time", 0.1)

    # 同時進行採樣、標記與分割訓練集與測試集(SSPP 那篇用的流程)
    # run_sampling_labeling_and_split_to_train_test("features_aes")
    
    # features_folder = "features" | "features_nonMalucusIsBenign"
    # 建議採樣數量
    # get_suggestion("features_rsa", 0.3)
    
    # 再次進行採樣、標記與分割訓練集與測試集，這次使用新的建議數量
    rerun_sampling_labeling_and_split_to_train_test("features_rsa")
    return


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
