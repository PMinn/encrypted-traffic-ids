import numpy as np
import os

def merge_testing_data(base_dir, files):
    """
        合併多分類測試集至二分類
        Args:
            base_dir: 資料夾路徑
            files: 各類別檔案列表
    """
    # files = [
    #     ("benign.npy", 0),           # label 0
    #     ("dos-hulk.npy", 1),   # label 1
    #     ("dos-slowhttptest.npy", 1),   # label 1
    #     ("dos-slowloris.npy", 1), # label 1
    # ]
    all_data = []
    all_labels = []

    for data_file, label_value in files:
        data_path = os.path.join(base_dir, "test", data_file)

        data = np.load(data_path)
        labels = np.full((data.shape[0],), label_value, dtype = np.int64)

        all_data.append(data)
        all_labels.append(labels)

    # 合併
    train_data = np.vstack(all_data)
    train_labels = np.concatenate(all_labels)

    print(f"✅ 合併後資料形狀: { train_data.shape }")
    print(f"✅ 合併後標籤形狀: { train_labels.shape }")
    print(f"✅ 標籤統計: { np.unique(train_labels, return_counts = True) }")

    # 儲存
    np.save(os.path.join(base_dir, "binary_test", "test_data.npy"), train_data)
    np.save(os.path.join(base_dir, "binary_test", "test_label.npy"), train_labels)

    print("🎯 binary_test/train_data.npy 和 binary_test/train_label.npy 已儲存完成")
