import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.dataset_to_binary import merge_testing_data 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 載入模型
class Cross_dataset_validation:
    def __init__(self, model_path, data_path, label_path, classes, batch_size = 4096):
        self.model_path = model_path
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.classes = classes
        self.model = None

    def load_model(self):
        logger.info("Loading model...")
        self.model = torch.load(self.model_path)
        self.model = self.model.cuda(device=device)
        self.model.eval()

    def load_data(self):
        logger.info("Loading data...")
        data = np.load(self.data_path)
        labels = np.load(self.label_path)

        # 數據歸一化
        data = data / 255.0
        data = torch.tensor(data, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(data, labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def validation(self):
        import torch.nn.functional as F
        from collections import defaultdict

        val_total_correct = 0
        self.load_model()
        data_loader = self.load_data()
        criterion = nn.CrossEntropyLoss()

        class_correct = [0.0 for _ in self.classes]
        class_total = [0.0 for _ in self.classes]

        all_labels = []
        all_predictions = []

        # Softmax 統計計數器
        class_count = defaultdict(int)
        class_conf_sum = defaultdict(float)

        with torch.no_grad():  # validation
            val_loss = 0.0
            for inputs, labels in data_loader:
                inputs = inputs.cuda(device=device)
                labels = labels.cuda(device=device)
                inputs = inputs.unsqueeze(1)
                outputs, _ = self.model(inputs)

                loss = criterion(outputs, labels.long())
                val_loss += float(loss * inputs.size(0))

                # ===== Softmax 機率統計 =====
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)

                # 統計 softmax 結果（不印每筆）
                for i in range(probs.size(0)):
                    pred_class = predicted[i].item()
                    confidence = probs[i][pred_class].item()
                    class_count[pred_class] += 1
                    class_conf_sum[pred_class] += confidence

                # 準確率統計
                val_total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()

                for i in range(labels.size(0)):
                    label = int(labels[i])
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # 計算 Precision、Recall、F1-score（若只有一類，設為 NaN）
        unique_labels = np.unique(all_labels)
        if len(unique_labels) < len(self.classes):
            logger.info("Warning: Not all classes are present in the validation data.")
            precision = recall = f1 = float('nan')
        else:
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        logger.info(f'\nPrecision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')

        # 各類別準確率 + 平均準確率（只計算有樣本的類別）
        val_acc_add = 0.0
        valid_class_count = 0
        for i, c in enumerate(self.classes):
            if class_total[i] == 0:
                logger.info(f'Valid Accuracy of {c}: N/A (no samples)')
                continue
            acc = 100 * class_correct[i] / class_total[i]
            val_acc_add += class_correct[i] / class_total[i]
            valid_class_count += 1
            logger.info(f'Valid Accuracy of {c}: {acc:.4f}%')

        val_loss = val_loss / len(data_loader)

        if valid_class_count > 0:
            val_acc = val_acc_add / valid_class_count
        else:
            val_acc = 0.0

        logger.info(f'Valid Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f}')

if __name__ == "__main__":
    file_handler = logging.FileHandler(os.path.abspath(os.path.join(__file__ ,"../../logs/TON_ViT_testing.log")))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # 修改為你的路徑
    # 二分類
    # [好像不用]先將多分類測試集合併成二分類
    # merge_testing_data(
    #     base_dir = "/sdc1/ytlindata/TON_IoT/120_5_flows_delall/",
    #     files = [
    #         ("benign.npy", 0), # label 0
    #         ("Cridex.npy", 1), # label 1
    #         ("Geodo.npy", 1), # label 1
    #         ("Htbot.npy", 1), # label 1
    #         ("Miuref.npy", 1), # label 1
    #         ("Neris.npy", 1), # label 1
    #         ("Nsis-ay.npy", 1), # label 1
    #         ("Shifu.npy", 1), # label 1
    #         ("Tinba.npy", 1), # label 1
    #         ("Virut.npy", 1), # label 1
    #         ("Zeus.npy", 1), # label 1
    #     ]
    # )
    # test_path = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/binary_sampling'
    # 多分類
    test_path = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/test/sampling/'

    model_path = "/sdc1/ytlindata/TON_IoT/ViT/1Dmulti_result(120delall)/20251029-171301/pth/model-best_val_acc.pth"

    data_path = f'{test_path}/train_data.npy'
    label_path = f'{test_path}/train_label.npy'

    # 初始化並運行
    cross_dataset_validation = Cross_dataset_validation(
        model_path,
        data_path,
        label_path,
        classes = [
            'benign',
            "Injection",
            "MITM",
            # "backdoor",
            "DDoS",
            "DoS",
            # "runsomware",
            "scanning",
            "XSS",
            "password"
        ]
    )
    cross_dataset_validation.validation()