import glob
from multiprocessing import Pool
from steps.datasetProcessor_USTC_TFC2016 import preprocess_data_del
from steps.split_USTC_TFC2016 import split_files
from steps.getFeatures_USTC_TFC2016 import runTCP_del, runUDP_del

if __name__ == "__main__":
    # 1. 分離成 Flows
    # Benign
    # split_files(
    #     input_dir = "/sdc1/ytlindata/USTC-TFC2016/Original_dataset/Benign/", # 輸入資料夾路徑
    #     output_dir = "/sdc1/ytlindata/USTC-TFC2016/split/Benign/", # 輸出資料夾路徑
    #     splitCapPath = "/home/YTLIN/ytlin/encrypted_NIDS/data_processing/steps/SplitCap.exe" # SplitCap.exe 的路徑
    # )
    # Malware
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
    # preprocess_data_del()
    pass