# from torch.utils.data import DataLoader, TensorDataset
# import torch
# from model.ViT_1D import ViT1D
# from utils.train_function2 import train_vit
from utils.alias import a2p


def testFeatures() -> None:
    from scapy.all import rdpcap
    from data_processing.FlowMeter.extract_flow_features_73 import (
        extract_flow_features_73,
    )

    pcap = "/home/alanpan/datasets/CIC-IDS2018/split/Friday-02-03-2018/split_353/capPC1-172.31.67.123.pcap.TCP_172-31-67-123_50341_199-83-134-9_443.pcap"
    pkts = rdpcap(pcap)
    feats = extract_flow_features_73(pkts)
    if feats is not None:
        for k, v in feats.items():
            print(f"{k}: {v}")


def testCIC27PcapTime() -> None:
    from scapy.all import rdpcap
    from datetime import datetime

    pcap = "/home/alanpan/datasets/CIC-IDS-2017/split/Wednesday/split_1/Wednesday-workingHours.pcap.TCP_172-16-0-1_37340_192-168-10-50_80.pcap"
    pkts = rdpcap(pcap)
    times = [pkt.time for pkt in pkts]
    for t in times:
        print(t)


def testTONPcapTime() -> None:
    from scapy.all import rdpcap
    from datetime import datetime

    pcap = a2p(
        "@/data/TON_IoT/encrypted_filter/Benign/split_1/normal_1.pcap.TCP_13-35-146-24_443_192-168-1-190_46933.pcap"
    )
    pkts = rdpcap(str(pcap))
    times = [pkt.time for pkt in pkts]
    for t in times:
        print(t)


def testEncrypt() -> None:
    from data_processing.encrypt import encrypt_pcap_by_three_schemes
    import json

    def dummy_kpabe_encrypt_key(aes_key: bytes) -> bytes:
        return b"ABE(" + aes_key + b")"

    outputs2 = encrypt_pcap_by_three_schemes(
        in_pcap=a2p(
            "@/data/TON_IoT/encrypted_filter/Benign/split_1/normal_1.pcap.TCP_13-35-146-24_443_192-168-1-190_46933.pcap"
        ),
        out_dir=a2p("@/data/TON_IoT/encrypted_l4/Benign/"),
        schemes=("AES", "RSA", "KPABE"),
        kpabe_encrypt_key_func=dummy_kpabe_encrypt_key,
    )
    print(json.dumps({k: str(v) for k, v in outputs2.items()}, indent=4))


if __name__ == "__main__":
    testEncrypt()
    # testTONPcapTime()

    # model = ViT1D(
    #     seq_len=480,
    #     patch_size=30,
    #     num_classes=2,
    #     dim=16,
    #     depth=6,
    #     heads=8,
    #     mlp_dim=32
    # )

    # # Dummy data loaders for illustration; replace with actual data loaders
    # train_loader = DataLoader(TensorDataset(torch.randn(1000, 1, 480), torch.randint(0, 2, (1000,))), batch_size=32)
    # val_loader = DataLoader(TensorDataset(torch.randn(200, 1, 480), torch.randint(0, 2, (200,))), batch_size=32)

    # trained_model, history = train_vit(model, train_loader, val_loader, epochs=50, lr=0.001)
    # import multiprocessing
    # cpus = multiprocessing.cpu_count()
    # print(f"Number of CPU cores: {cpus}")

    # print(glob.glob('/home/alanpan/datasets/CIC-IDS2018/split/Friday-02-03-2018/split_*/*_18-219-211-138_*_18-219-211-138_*.pcap', recursive=True))
    # testFeatures()

    # import glob
    # import re
    # from scapy.all import rdpcap
    # from datetime import datetime
    # import os
    # pcaps = glob.glob('/home/alanpan/datasets/CIC-IDS2018/split/Friday-02-03-2018/split_*/*.pcap')
    # victim = ['_172-31-69-23_', '_172-31-69-17_', '_172-31-69-14_', '_172-31-69-12_', '_172-31-69-10_', '_172-31-69-8_', '_172-31-69-6_', '_172-31-69-26_', '_172-31-69-29_', '_172-31-69-30_']
    # bot_pcaps = [pcap for pcap in pcaps if re.search(r'_18-219-211-138_.*(' + '|'.join(victim) + r')', pcap)]
    # time_strings = [
    #     ["02-03-2018 10:11", "02-03-2018 11:34"],
    #     ["02-03-2018 14:24", "02-03-2018 15:55"]
    # ]
    # timestamps = [(datetime.strptime(f"{time_range[0]}:00 -0400", "%d-%m-%Y %H:%M:%S %z").timestamp(), datetime.strptime(f"{time_range[1]}:59 -0400", "%d-%m-%Y %H:%M:%S %z").timestamp()) for time_range in time_strings]

    # # max_time = datetime.strptime("02-03-2017 11:34:59", "%d-%m-%Y %H:%M:%S").timestamp()
    # # min_time = datetime.strptime("02-03-2019 10:11:00", "%d-%m-%Y %H:%M:%S").timestamp()
    # count = 0
    # for pcap in bot_pcaps:
    #     pkts = rdpcap(pcap)
    #     timestamp_unix = pkts[0].time
    #     in_range = any(start <= timestamp_unix <= end for start, end in timestamps)
    #     if in_range:
    #         count += 1
    # print(f"Bot pcaps in time ranges: {count} / {len(bot_pcaps)}")
    #         # print(f"Out of range {datetime.fromtimestamp(int(timestamp_unix))}: {pcap}")
    # #     if timestamp_unix > max_time:
    # #         max_time = timestamp_unix
    # #     elif timestamp_unix < min_time:
    # #         min_time = timestamp_unix
    # # print(f"Min time: {datetime.fromtimestamp(int(min_time))}")
    # # print(f"Max time: {datetime.fromtimestamp(int(max_time))}")
