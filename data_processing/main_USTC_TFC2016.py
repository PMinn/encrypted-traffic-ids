import glob
from multiprocessing import Pool
from steps.split_USTC_TFC2016 import split_files
from steps.getFeatures_USTC_TFC2016 import runTCP_del, runUDP_del
from steps.datasetProcessor_USTC_TFC2016 import DatasetProcesser_USTC_TFC2016
from steps.sampling_USTC_TFC2016 import run_sampling, run_binary_sampling

if __name__ == "__main__":
    # 攻擊類別
    classes = [
        'benign',
        "Cridex",
        "Geodo",
        "Htbot",
        "Miuref",
        "Neris",
        "Nsis-ay",
        "Shifu",
        "Tinba",
        "Virut",
        "Zeus"
    ]
    
    # 1. 分離成 Flows
    # 正常流量 Benign
    # split_files(
    #     input_dir = "/sdc1/ytlindata/USTC-TFC2016/Original_dataset/Benign/", # 輸入資料夾路徑
    #     output_dir = "/sdc1/ytlindata/USTC-TFC2016/split/Benign/", # 輸出資料夾路徑
    #     splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/data_processing/steps/SplitCap.exe" # SplitCap.exe 的路徑
    # )
    # 惡意流量 Malware
    # split_files(
    #     input_dir = "/sdc1/ytlindata/USTC-TFC2016/Original_dataset/Malware/", # 輸入資料夾路徑
    #     output_dir = "/sdc1/ytlindata/USTC-TFC2016/split/Malware/", # 輸出資料夾路徑
    #     splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/data_processing/steps/SplitCap.exe" # SplitCap.exe 的路徑
    # )

    # 2. 特徵擷取
    # TCP
    # with Pool(10) as p:
    #     p.map(runTCP_del, ['Benign', 'Malware'])
    # UDP
    # with Pool(10) as p:
    #     p.map(runUDP_del, ['Benign', 'Malware'])

    # 3. 數據預處理，合併特徵並分割訓練集與測試集
    # DatasetProcesser_USTC_TFC2016(
    #     ORIG_DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/del_120_5_flows(delall)/',
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall',
    #     classes = classes
    # ).run()
    
    # 4. 不平衡資料處理
    # 多分類
    # run_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/train',
    #     MULTI_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/sampling',
    #     UNDERSAMPLING = { 
    #         'benign': 4000*15,
    #         "Cridex": 4000*3, 
    #         "Neris": 4000*3, 
    #         "Virut": 4000*4,
    #     },
    #     OVERSAMPLING = {
    #         "Geodo": 8000,
    #         "Htbot": 8000,
    #         "Miuref": 7000,
    #         "Nsis-ay": 7500,
    #         "Shifu": 9500,
    #         "Tinba": 9000,
    #         "Zeus": 10000
    #     }
    # )
    # 二分類
    # run_binary_sampling(
    #     classes = classes,
    #     DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/train',
    #     BINARY_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delall/binary_sampling',
    #     UNDERSAMPLING = { 
    #         'benign': 4000*15,
    #         "Cridex": 4000*3, 
    #         "Neris": 4000*3, 
    #         "Virut": 4000*4,
    #     },
    #     OVERSAMPLING = {
    #         "Geodo": 8000,
    #         "Htbot": 8000,
    #         "Miuref": 7000,
    #         "Nsis-ay": 7500,
    #         "Shifu": 9500,
    #         "Tinba": 9000,
    #         "Zeus": 10000
    #     }
    # )
    pass