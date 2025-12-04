import os
import datetime
import logging
import json
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

from model.Adapter_Token_ViT_1D import Adapter_Token_ViT_1D

def start_training():
    # === 超參數設定 ===
    gamma = 0.01
    init_lr = 0.01
    EPOCH = 100
    BATCH_SIZE = 8192
    PATCH_SIZE = 12
    RL_STEP = 15
    seq_size = 480

    with open("/home/alanpan/datasets/CIC-IDS-2017/features/sampled_raw/train/classes.json", "r") as f:
        classes = json.load(f)

    result_folder_path = '/home/alanpan/datasets/CIC-IDS-2017/both_multi_result/' + str(datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S'))
    os.makedirs(result_folder_path, exist_ok = True)
    os.makedirs(result_folder_path + '/pth', exist_ok = True)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # === 讀取資料 ===
    train_path = '/home/alanpan/datasets/CIC-IDS-2017/features/sampled/train'
    data = np.load(f'{train_path}/sampled_data.npy')
    label = np.load(f'{train_path}/sampled_label.npy')

    train_data, val_data, train_label, val_label = train_test_split(data,label, test_size = 0.1, stratify = label, random_state = 42) # 分訓練/驗證
    logging.getLogger("train_function_Both").info('-' * 25 +'train data'+'-' * 25 )
    for i in set(train_label):
        logging.getLogger("train_function_Both").info(f'{classes[i]}: {np.sum(train_label == i )}')

    logging.getLogger("train_function_Both").info('-' * 25 +'valid data'+'-' * 25 )
    for i in set(val_label):
        logging.getLogger("train_function_Both").info(f'{classes[i]}: {np.sum(val_label == i )}')
        
    # flow_feature_order = [
    #     "Flow Duration",
    #     "Total Fwd Packets",
    #     "Total Backward Packets",
    #     "Destination Port",
    #     "Source Port",
    #     "Flow Packets/s",
    #     "Flow Bytes/s",
    #     "Total Length of Fwd Packets",
    #     "Total Length of Bwd Packets",
    #     "Fwd Packet Length Mean",
    #     "Bwd Packet Length Mean",
    #     "Max Packet Length",
    #     "Min Packet Length",
    #     "Packet Length Std",
    #     "SYN Flag Count",
    #     "ACK Flag Count",
    #     "Protocol",
    #     "Fwd IAT Mean",
    #     "Bwd IAT Mean",
    #     "Fwd IAT Max",
    #     "Fwd IAT Std",
    #     "Bwd IAT Max",
    #     "Avg Fwd Segment Size",
    #     "Avg Bwd Segment Size",
    # ]
    log_idx = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 17, 18, 19, 20, 21]   # heavy-tailed
    z_only_idx = [12, 22, 23]                                          # min length / avg size
    # no_scale_idx = [3, 4, 14, 15, 16]                                   # ports, flags, protocol 忽略不做任何正規化

    train_flow_features = train_data[:, :24]
    mu, sigma = fit_normalizer(train_flow_features, log_idx, z_only_idx)
    train_flow_features = transform_normalizer(train_flow_features, mu, sigma, log_idx, z_only_idx)
    train_data = train_data[:, 24:]
    train_data = train_data / 255
    
    val_flow_features = val_data[:, :24]
    val_flow_features = transform_normalizer(val_flow_features, mu, sigma, log_idx, z_only_idx)
    val_data = val_data[:, 24:]
    val_data = val_data / 255
    
    np.savez(result_folder_path + '/normalize.npz', mu = mu, sigma = sigma)
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_data, dtype = torch.float),
            torch.tensor(train_flow_features, dtype = torch.float),
            torch.tensor(train_label, dtype = torch.long)
        ),
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(val_data, dtype = torch.float),
            torch.tensor(val_flow_features, dtype = torch.float),
            torch.tensor(val_label, dtype = torch.long)
        ),
        batch_size = BATCH_SIZE
    )

    # === 建立模型 ===
    model = Adapter_Token_ViT_1D(
        seq_len = seq_size,
        patch_size = PATCH_SIZE,
        num_classes = len(classes),
        dim = 16,
        depth = 6,
        heads = 8,
        mlp_dim = 32
    ).to(device)

    if torch.cuda.is_available():
        logging.getLogger("train_function_Both").info(torch.cuda.get_device_name(0))
    else:
        logging.getLogger("train_function_Both").info("Using CPU")
    # 多卡（如果有）
    if torch.cuda.device_count() > 1:
        logging.getLogger("train_function_Both").info(f"Use {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # === Loss / Optimizer ===
    train_loss_curve = []
    val_loss_curve = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = init_lr)
    scheduler = StepLR(optimizer, step_size = RL_STEP, gamma = gamma)

    # === 開始訓練 ===
    best_val_acc = 0.0
    for epoch in range(1, EPOCH + 1):
        localtime = time.asctime(time.localtime(time.time()))

        logging.getLogger("train_function_Both").info('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch, EPOCH, localtime)))
        logging.getLogger("train_function_Both").info('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch, EPOCH, localtime))

        train_loss, train_acc, train_per_class_acc = train_one_epoch(model, train_loader, len(classes), criterion, optimizer, device)
        train_loss_curve.append(train_loss)

        with open(result_folder_path + '/train_Acc_Loss.txt', 'a') as file:
            file.write('Epoch :' + str(epoch) + '/' + str(EPOCH) + '\n')
            for i, c in enumerate(classes):
                file.write(f'Train Accuracy of {c}: {str(train_per_class_acc[i] * 100)}% \n')
                logging.getLogger("train_function_Both").info('Train Accuracy of %5s : %8.4f %%' % (c, train_per_class_acc[i] * 100))

            file.write(f'Training Accuracy: {str(train_acc * 100)}%')
            file.write(f' | Training loss: {str(train_loss)}\n')
            logging.getLogger("train_function_Both").info(f'Training Accuracy: {train_acc * 100}%')
        logging.getLogger("train_function_Both").info('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(train_loss, train_acc))

        val_loss, val_acc, val_per_class_acc, f1_macro, f1_per_class = evaluate(model, val_loader, len(classes), criterion, device)
        val_loss_curve.append(val_loss)

        with open(result_folder_path + '/valid_Acc_Loss.txt', 'a') as file:
            file.write('Epoch :' + str(epoch) + '/' + str(EPOCH) + '\n')
            for i, c in enumerate(classes):
                file.write(f'valid Accuracy of {c}: {str(val_per_class_acc[i] * 100)}% \n')
                logging.getLogger("train_function_Both").info('valid Accuracy of %5s : %8.4f %%' % (c, val_per_class_acc[i] * 100))

            file.write(f'Valid Accuracy:{str(val_acc * 100)}% | loss: {str(val_loss)} \n')
            logging.getLogger("train_function_Both").info(f'Valid Accuracy:{str(val_acc * 100)}%| loss:{str(val_loss)}| F1 Score: {f1_macro:.4f}')

        scheduler.step() # 更新學習率
        logging.getLogger("train_function_Both").info(f"learning rate: {scheduler.get_last_lr()}")

        if (epoch - 1) % 5 == 0:
            torch.save(model, '{}/pth/model-{:.2f}-val_acc-{}-epoch.pth'.format(result_folder_path, val_acc, epoch))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            with open(result_folder_path + '/Best_valid_Acc.txt', 'w') as file:
                file.write('Best Epoch :' + str(epoch) + f'\n Best acc: {best_val_acc}')

            torch.save(model, f'{result_folder_path}/pth/model-best_val_acc.pth')
            

    plt.plot(range(1, EPOCH + 1), train_loss_curve, label = "Training Loss", color = "blue")
    plt.plot(range(1, EPOCH + 1), val_loss_curve, label = "Validation Loss", color = "orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ViT Overfitting Check")
    plt.legend()
    plt.savefig(f"{result_folder_path}/loss_curve.png")
    plt.show()
    logging.getLogger("train_function_Both").info(f"Loss curve saved to {result_folder_path}/loss_curve.png")

    logging.getLogger("train_function_Both").info(f'Best model save to: {result_folder_path}/pth/model-best_val_acc.pth')
    parameter_total = sum([param.nelement() for param in model.parameters()])
    logging.getLogger("train_function_Both").info("Number of parameter: %.2fM" % (parameter_total/1e6))
  
def fit_normalizer(X: np.ndarray, log_idx: list[int], z_only_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
        訓練階段：計算 log 後的 μ 和 σ
        Arguments:
            X: np.array, shape = (num_samples, num_features)
            log_idx: list of int, 需要做 log 的欄位 index
            z_only_idx: list of int, 只做 z-score 的欄位 index
        Returns:
            mu: np.array, shape = (num_zscore_features,)
            sigma: np.array, shape = (num_zscore_features,)
    """
    X_log = X.copy().astype(float)

    # logarithm 區段
    X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    # 要做 z-score 的欄位
    z_cols = log_idx + z_only_idx

    # 計算 μ 和 σ
    mu = X_log[:, z_cols].mean(axis=0)
    sigma = X_log[:, z_cols].std(axis=0) + 1e-12  # 防止除以 0

    return mu, sigma

def transform_normalizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, log_idx: list[int], z_only_idx: list[int]) -> np.ndarray:
    """
        推論階段（包括 validation / test）：只能 transform，不可重新 fit!
        Arguments:
            X: np.array, shape = (num_samples, num_features)
            mu: np.array, shape = (num_zscore_features,)
            sigma: np.array, shape = (num_zscore_features,)
            log_idx: list of int, 需要做 log 的欄位 index
            z_only_idx: list of int, 只做 z-score 的欄位 index
        Returns:
            X_new: np.array, shape = (num_samples, num_features), 正規化後的資料
    """
    X_new = X.copy().astype(float)

    # 1. 做 log1p
    X_new[:, log_idx] = np.log1p(X_new[:, log_idx])

    # 2. z-score
    z_cols = log_idx + z_only_idx
    X_new[:, z_cols] = (X_new[:, z_cols] - mu) / sigma

    return X_new

def train_one_epoch(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, num_classes: int, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """
        訓練一個 epoch
        Arguments:
            model: torch.nn.Module, 要訓練的模型
            dataloader: torch.utils.data.DataLoader, 訓練資料的 dataloader
            num_classes: int, 類別數
            criterion: 損失函數
            optimizer: 優化器
            device: torch.device, 運算裝置
        Returns:
            avg_loss: float, 平均損失
            acc: float, 準確率
            per_class_acc: dict, 每類別的準確率
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for seq, flow, labels in dataloader:
        seq = seq.to(device)        # [B, seq_len]
        seq = seq.unsqueeze(1)  # 增加一個新的維度來表示通道 [B, 1, seq_len]
        flow = flow.to(device)      # [B, F]
        labels = labels.to(device)  # [B]

        optimizer.zero_grad() # 清除之前的梯度
        logits, cls_token_features = model(seq, flow)          # logits: [B, num_classes]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim = 1)
        all_preds.append(preds.detach().cpu()) # 將預測結果存到 CPU 上
        all_labels.append(labels.detach().cpu()) # 將標籤存到 CPU 上

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    # ====== 每類別 accuracy ======
    per_class_acc = {}
    for cls in range(num_classes):
        mask = (all_labels == cls)
        if mask.sum() == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = (all_preds[mask] == all_labels[mask]).mean()

    return avg_loss, acc, per_class_acc

def evaluate(model, dataloader, num_classes, criterion, device):
    """
        評估模型
        Arguments:
            model: torch.nn.Module, 要評估的模型
            dataloader: torch.utils.data.DataLoader, 評估資料的 dataloader
            num_classes: int, 類別數
            criterion: 損失函數
            device: torch.device, 運算裝置
        Returns:
            avg_loss: float, 平均損失
            acc: float, 準確率
            per_class_acc: dict, 每類別的準確率
            f1_macro: float, macro F1-score
            f1_per_class: dict, 每類別的 F1-score
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq, flow, labels in dataloader:
            seq = seq.to(device)
            seq = seq.unsqueeze(1)  # 增加一個新的維度來表示通道 [B, 1, seq_len]
            flow = flow.to(device)
            labels = labels.to(device)

            logits, _ = model(seq, flow)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    # ====== 每類別 accuracy ======
    per_class_acc = {}
    for cls in range(num_classes):
        mask = (all_labels == cls)
        if mask.sum() == 0:
            per_class_acc[cls] = None
        else:
            per_class_acc[cls] = (all_preds[mask] == all_labels[mask]).mean()

    # ====== F1-score ======
    # macro: 對每類別算 F1 再平均（不看類別比例）
    f1_macro = f1_score(all_labels, all_preds, average = "macro", zero_division = 0)

    # per-class F1：一個類別一個值，順序是 class 0,1,...,num_classes-1
    f1_per_class_arr = f1_score(all_labels, all_preds, average = None, labels = range(num_classes), zero_division = 0)
    f1_per_class = {cls: float(f1_per_class_arr[cls]) for cls in range(num_classes)}

    return avg_loss, acc, per_class_acc, f1_macro, f1_per_class

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    filename = os.path.splitext(os.path.basename(__file__))[0]
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(os.path.abspath(os.path.join(__file__ , f"../../logs/{filename}_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.log")))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)", datefmt = "%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)", datefmt = "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console)

    logger.getChild("matplotlib").setLevel(logging.WARNING)

    start_training()