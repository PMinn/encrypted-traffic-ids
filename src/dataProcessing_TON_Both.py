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


def rewrite_pcap() -> None:
    # 惡意流量 Malware
    attack_pcaps = a2p("@/data/TON_IoT/Original_dataset/normal_attack_pcaps").glob(
        "**/*.pcap"
    )
    with tqdm(total=len(cast(list[Path], attack_pcaps))) as pbar:
        for pcap in attack_pcaps:
            output_file = Path(
                pcap.as_posix().replace("/Original_dataset/", "/rewrite/")
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.exists():
                pbar.update(1)
                continue
            save_pcap(pcap_file=pcap, output_file=output_file)
            pbar.update(1)
    # 正常流量 Benign
    benign_pcaps = a2p("@/data/TON_IoT/Original_dataset/normal_pcaps").glob("**/*.pcap")
    with tqdm(total=len(cast(list[Path], benign_pcaps))) as pbar:
        for pcap in benign_pcaps:
            output_file = Path(
                pcap.as_posix().replace("/Original_dataset/", "/rewrite/")
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if output_file.exists():
                pbar.update(1)
                continue
            if os.path.exists(output_file):
                pbar.update(1)
                continue
            save_pcap(pcap_file=pcap, output_file=output_file)
            pbar.update(1)


def split_pcap() -> None:
    from data_processing.split_to_flows import split_to_flows_from_folder

    # 正常流量 Benign
    split_to_flows_from_folder(
        input_dir=a2p("@/data/TON_IoT/rewrite/normal_pcaps/"),  # 輸入資料夾路徑
        output_dir=a2p("@/data/TON_IoT/split/Benign/"),  # 輸出資料夾路徑
    )
    # 惡意流量 Malicious
    attack_folders = a2p("@/data/TON_IoT/rewrite/normal_attack_pcaps").iterdir()
    for folder in attack_folders:
        folder_name = folder.name
        split_to_flows_from_folder(
            input_dir=folder,  # 輸入資料夾路徑
            output_dir=a2p(
                f"@/data/TON_IoT/split/Malicious/{folder_name}/"
            ),  # 輸出資料夾路徑
        )


def filter_encrypted_pcaps() -> None:
    from data_processing.filter_encrypted import remove_pcap_if_not_encrypted

    # 正常流量 Benign
    benign_pcaps = a2p("@/data/TON_IoT/split/Benign").glob("**/*.pcap")
    with Pool(50) as pool:
        r = list(
            tqdm(
                pool.imap(remove_pcap_if_not_encrypted, benign_pcaps),
                total=len(cast(list[Path], benign_pcaps)),
            )
        )
    logging.getLogger("filter_encrypted_pcaps").info(r)
    # 惡意流量 Malicious
    malicious_folders = a2p("@/data/TON_IoT/split/Malicious").iterdir()
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
    attack_csvs = a2p("@/data/TON_IoT/SecurityEvents_Network_datasets").glob("*.csv")
    attack_dict: dict[str, dict[tuple[str, str, str, str, str], list[int]]] = {}
    skip_attack_types: list[str] = [
        # "scanning",
        # "ddos",
        # "password",
        # "xss",
        # "dos",
        # "injection",
        # "ransomware",
        # "backdoor",
        # "mitm"
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
                ts = int(float(parts[0]))
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
    benign_path = a2p("@/data/TON_IoT/attack_filter_fixed_time/Benign")

    def handle_filter(pcap: Path, is_attack: bool) -> None:
        if is_attack:
            copy_pcap(
                src_file=pcap,
                dest_file=Path(
                    pcap.as_posix().replace(
                        "/encrypted_filter/", "/attack_filter_fixed_time/"
                    )
                ),
            )
        else:
            copy_pcap(
                src_file=pcap,
                dest_file=benign_path / "split_-1" / pcap.name,
            )

    for attack_type in attack_dict:
        if attack_type in skip_attack_types:
            continue
        pcap_folder = csv_type2pcap_filder_map[attack_type]
        attack_pcaps = a2p(
            f"@/data/TON_IoT/encrypted_filter/Malicious/{pcap_folder}"
        ).glob("**/*.pcap")
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
    pcaps = a2p("@/data/TON_IoT/attack_filter_fixed_time/Benign").glob("split_*/*.pcap")
    flow_to_features_file(
        pcaps,
        output_file=a2p(
            f"@/data/TON_IoT/features_fixed_time/features/benign_features_{packet_shape[0]}_{packet_shape[1]}.npy"
        ),
        packet_shape=packet_shape,
    )

    malicious_folders = a2p(
        "@/data/TON_IoT/attack_filter_fixed_time/Malicious"
    ).iterdir()
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.stem.replace("normal_", "")
        pcaps = malicious_folder.glob("split_*/*.pcap")
        flow_to_features_file(
            pcaps,
            output_file=a2p(
                f"@/data/TON_IoT/features_fixed_time/features/{folder_name}_features_{packet_shape[0]}_{packet_shape[1]}.npy"
            ),
            packet_shape=packet_shape,
        )


def split_features_to_train_test() -> None:
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split

    train_folder = a2p("@/data/TON_IoT/features_fixed_time/split/train")
    test_folder = a2p("@/data/TON_IoT/features_fixed_time/split/test")
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)
    features_npys = a2p("@/data/TON_IoT/features_fixed_time/features").glob("*.npy")
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
        a2p("@/data/TON_IoT/features_fixed_time/split/train_test_split_info.json"), "w"
    ) as f:
        json.dump(result, f, indent=4)


def run_sampling_and_labeling(features_folder: str, alpha: float) -> None:
    from data_processing.sampling_and_labeling import (
        sampling,
        labeling,
        get_oversampling_by_kmeans,
    )
    import json

    result_folder = a2p(f"@/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P")

    with open(result_folder / "sampling_suggestion.json", "r") as f:
        sampling_suggestion = json.load(f)

    assert sampling_suggestion["alpha"] == alpha, "Alpha 與建議檔案不符！"
    sampling(
        [
            {
                "name": "Benign",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/train/benign_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "Benign"
                ],
            },
            {
                "name": "scanning",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/train/scanning_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "scanning"
                ],
            },
            {
                "name": "DDoS",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/train/DDoS_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "DDoS"
                ],
            },
            {
                "name": "Injection",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/train/Injection_features_96_5_73.npy"
                ),
                "sampling_count": sampling_suggestion["number_of_samples_suggested"][
                    "Injection"
                ],
            },
        ],
        save_to=a2p(
            f"@/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/train/"
        ),
        oversampling=get_oversampling_by_kmeans,
    )

    labeling(
        [
            {
                "name": "Benign",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/test/benign_features_96_5_73.npy"
                ),
            },
            {
                "name": "scanning",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/test/scanning_features_96_5_73.npy"
                ),
            },
            {
                "name": "DDoS",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/test/DDoS_features_96_5_73.npy"
                ),
            },
            {
                "name": "Injection",
                "data_path": a2p(
                    f"@/data/TON_IoT/{features_folder}/split/test/Injection_features_96_5_73.npy"
                ),
            },
        ],
        save_to=a2p(
            f"@/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/test/"
        ),
    )


def get_suggestion(features_folder: str, alpha: float) -> None:
    from data_processing.sampling_and_labeling import suggester

    suggester(
        {
            "Benign": 30320,
            "scanning": 27146,
            "DDoS": 104709,
            "Injection": 31762,
        },
        alpha=alpha,
        json_path=a2p(
            f"@/data/TON_IoT/{features_folder}/sampled_{int(alpha * 100)}P/sampling_suggestion.json"
        ),
    )


def count_pcaps(folder_path: Path) -> None:
    benig_pcaps = list(folder_path.joinpath("Benign").glob("**/*.pcap"))
    logging.getLogger("count_pcaps").info(f"Benign pcaps: {len(benig_pcaps)}")
    malicious_folders = folder_path.joinpath("Malicious").iterdir()
    for malicious_folder in malicious_folders:
        folder_name = malicious_folder.name
        malicious_pcaps = list(malicious_folder.glob("**/*.pcap"))
        logging.getLogger("count_pcaps").info(
            f"Malicious pcaps in {folder_name}: {len(malicious_pcaps)}"
        )


def main() -> None:
    # 有些 pcap 切分工具無法讀取，可以用這個步驟重新存檔一次
    # rewrite_pcap()

    # 分離成 Flows
    # split_pcap()

    # 篩選出加密的 pcap
    # filter_encrypted_pcaps()

    # 計算出各類型的數量
    # count_pcaps('/sdc1/ytlindata/TON_IoT/encrypted_filter')

    # 篩選攻擊
    # filter_attack()

    # 計算出各類型的數量
    # count_pcaps(a2p("@/data/TON_IoT/attack_filter_fixed_time"))
    # count_pcaps('/sdc1/ytlindata/TON_IoT/attack_filter_with_decrypt')

    # 特徵擷取
    # get_features()

    # 分割成訓練集與測試集
    # split_features_to_train_test()

    # features_folder = "features" | "features_nonMalucusIsBenign"
    # 建議採樣數量
    get_suggestion("features_fixed_time", 0.1)

    # features_folder = "features" | "features_nonMalucusIsBenign"
    # 採樣及標記
    run_sampling_and_labeling("features_fixed_time", 0.1)
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
