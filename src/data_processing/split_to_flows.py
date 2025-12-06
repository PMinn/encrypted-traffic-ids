import logging
import subprocess
from pathlib import Path
from utils.alias import a2p

def split_to_flows_from_folder(input_dir: Path, output_dir: Path, max_files: int = 50, splitCapPath: Path = a2p("@/src/data_processing/SplitCap.exe"), remove_original: bool = False) -> None | list:
    """
       分離資料夾中的 pcap 成 Flows
       Args:
           input_dir (Path): 輸入資料夾路徑
           output_dir (Path): 輸出資料夾路徑
           max_files (int): 每個資料夾最多處理的檔案數
           splitCapPath (Path): SplitCap.exe 的路徑
           remove_original (bool): 是否刪除原始檔案
       Returns:
              None or list: 若所有檔案皆處理完畢，回傳 None；否則回傳未處理的檔案清單
    """
    counter = 0 # 當前已處理的檔案數
    folder_num = 1 # 當前資料夾編號

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    # 取得符合條件的檔案清單
    files = sorted(input_dir.glob("**/*"))
    
    output_dir.mkdir(parents = True, exist_ok = True)

    # 迴圈處理所有檔案
    for file in files:
        # 執行 SplitCap 並指定輸出資料夾
        out_dir = output_dir / f"split_{folder_num}"

        # 當達到最大檔案數，切換資料夾
        if counter >= max_files:
            counter = 0
            folder_num += 1
            out_dir = output_dir / f"split_{folder_num}"
            
        # 確保初始資料夾存在
        out_dir.mkdir(parents = True, exist_ok = True)
        logging.getLogger("split_to_flows.from_folder").info(f"Processing {file} into {out_dir}")
        try:
            subprocess.run([
                "mono", str(splitCapPath),
                "-r", str(file),
                "-p", "10",
                "-o", str(out_dir),
                "-s", "session"
            ], check = True)
            if remove_original:
                file.unlink()
        except subprocess.CalledProcessError as e:
            logging.getLogger("split_to_flows.from_folder").error(f"Error occurred while processing {file}: {e}")

        # counter 設為 out_dir 中的檔案數量
        counter = len(list(out_dir.glob("*")))
    if remove_original:
        # 確認所有檔案都已處理
        left_files = sorted(input_dir.glob("**/*"))
        if len(left_files) == 0:
            logging.getLogger("split_to_flows.from_folder").info("All files have been processed successfully.")
            input_dir.rmdir()
            return None
        else:
            logging.getLogger("split_to_flows.from_folder").warning(f"Some files were not processed: {left_files}")
            return left_files
    else:
        return None

def split_to_flows_from_file(input_file: Path, output_dir: Path, max_files: int = 50, splitCapPath: Path = a2p("@/src/data_processing/SplitCap.exe"), remove_original: bool = False) -> list | None:
    """
       分離資料夾中的 pcap 成 Flows
       Args:
           input_file (Path): 輸入檔案路徑
           output_dir (Path): 輸出資料夾路徑
           max_files (int): 每個資料夾最多處理的檔案數
           splitCapPath (Path): SplitCap.exe 的路徑
           remove_original (bool): 是否刪除原始檔案
       Returns:
            None or list: 若所有檔案皆處理完畢，回傳 None；否則回傳未處理的檔案清單
    """
    counter = 0 # 當前已處理的檔案數
    folder_num = 1 # 當前資料夾編號

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    # 取得符合條件的檔案清單
    files = [input_file]
    
    output_dir.mkdir(parents = True, exist_ok = True)

    # 迴圈處理所有檔案
    for file in files:
        # 執行 SplitCap 並指定輸出資料夾
        out_dir = output_dir / f"split_{folder_num}"

        # 當達到最大檔案數，切換資料夾
        if counter >= max_files:
            counter = 0
            folder_num += 1
            out_dir = output_dir / f"split_{folder_num}"
            
        # 確保初始資料夾存在
        out_dir.mkdir(parents = True, exist_ok = True)
        logging.getLogger("split_to_flows.from_file").info(f"Processing {file} into {out_dir}")
        try:
            subprocess.run([
                "mono", str(splitCapPath),
                "-r", str(file),
                "-p", "10",
                "-o", str(out_dir),
                "-s", "session"
            ], check = True)
            if remove_original:
                file.unlink()
        except subprocess.CalledProcessError as e:
            logging.getLogger("split_to_flows.from_file").error(f"Error occurred while processing {file}: {e}")

        # counter 設為 out_dir 中的檔案數量
        counter = len(list(out_dir.glob("*")))