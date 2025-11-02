import os
import time
import numpy as np
from datetime import datetime

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from model.swin_transformer import SwinTransformer1D  # our 1D Swin impl
from model.swin_transformer_1d import SwinTransformerLayer
from model.swin_transformer_1d_v2 import SwinTransformerV2Layer
from utils.train_function import *  
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def multi_1DSwin_delall_120():
    # 超参数
    gamma = 0.01
    seed = 42
    init_lr = 0.001
    EPOCH = 100
    BATCH_SIZE = 4096
    PATCH_SIZE = 15
    RL_STEP = 5
    seq_size = 480

    hyper_parameter = {
        'batch_size': BATCH_SIZE,
        'patch_size': PATCH_SIZE,
        'epoch': EPOCH,
        'gamma': gamma,
        'learning_rate_step': RL_STEP,
        'learning_rate': init_lr,
        'seq_size': seq_size
    }
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
    num_classes = len(classes)

    # 数据路径
    train_path = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/sampling'
    train_data_path  = f'/{train_path}/train_data.npy'
    train_label_path = f'/{train_path}/train_label.npy'

    # 子目录，用于保存模型和日志
    subdir = '/sdc1/ytlindata/TON_IoT/Swin/1DSwin_result_120delall/' + str(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    os.makedirs(os.path.join(subdir,'pth'), exist_ok=True)

    # 构建 1D Swin 模型
    # model = SwinTransformer1D(
    #     seq_len=seq_size,
    #     patch_size=PATCH_SIZE,
    #     in_chans=1,
    #     num_classes=num_classes,
    #     embed_dim=16,         # 对应你之前 dim=16

    #     depths=[2,2,6,2],           # depth=6
    #     num_heads=[8,8,8,8],        # heads=8
    #     window_size=4,  
    #     mlp_ratio=2.0,        # mlp_dim=32 => mlp_ratio = 32/16 = 2.0
    #     drop_rate=0.0,
    #     attn_drop_rate=0.0,
    #     drop_path_rate=0.1,
    #     use_checkpoint=False
    # )
    
    # SwinTransformerLayer | SwinTransformerV2Layer
    model = SwinTransformerV2Layer(
        dim = 16,
        depth = 6,
        num_heads = 8,
        window_size = 4,
        mlp_ratio = 2.,
        # qkv_bias = True,
        # drop = 0.,
        # attn_drop = 0.,
        # drop_path = 0.1,
        use_checkpoint = True,
        # pretrained_window_size = 0
    )

    # 用 Train_1D_Swin 训练
    trainer = Train_1D_Swin(
        model=model,
        data_path=train_data_path,
        label_path=train_label_path,
        subdir=subdir,
        hyper_parameter=hyper_parameter,
        classes=classes
    )

if __name__ == '__main__':
    file_handler = logging.FileHandler('./log/TON_training_v2.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    multi_1DSwin_delall_120()
