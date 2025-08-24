from steps.split_USTC_TFC2016 import split_files

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