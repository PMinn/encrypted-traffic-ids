import os
import glob
import subprocess

def split_files(input_dir, output_dir, max_files = 50, splitCapPath = "SplitCap.exe"):
    """
       分離檔案並存入不同資料夾

       Args:
           input_dir (str): 輸入資料夾路徑
           output_dir (str): 輸出資料夾路徑
           max_files (int): 每個資料夾最多處理的檔案數
           splitCapPath (str): SplitCap.exe 的路徑

       Returns:
           None
    """
    counter = 0 # 當前已處理的檔案數
    folder_num = 1 # 當前資料夾編號

    # 取得符合條件的檔案清單
    files = sorted(glob.glob(os.path.join(input_dir, "**")))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    # 迴圈處理所有檔案
    for file in files:
        print(f"Starting to process file: {file}")
        # 執行 SplitCap 並指定輸出資料夾
        out_dir = os.path.join(output_dir, f"split_{folder_num}")

        # 當達到最大檔案數，切換資料夾
        if counter >= max_files:
            counter = 0
            folder_num += 1
            out_dir = os.path.join(output_dir, f"split_{folder_num}")
            
        # 確保初始資料夾存在
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok = True)

        print(f"Processing {file} into {out_dir}")
        try:
            subprocess.run([
                "mono", splitCapPath,
                "-r", file,
                "-p", "10",
                "-o", out_dir,
                "-s", "session"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {file}: {e}")
            return

        # 增加檔案計數
        counter += 1