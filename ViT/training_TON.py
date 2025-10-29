from model.ViT_1D import ViT1D
# from model.vit import ViT
from utils.train_function import Train_1D
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def multi_1DViT_delall_120():
    gamma = 0.01
    seed = 42
    init_lr = 0.01
    EPOCH = 100
    BATCH_SIZE = 8192
    # PATCH_SIZE = 10
    PATCH_SIZE = 12
    RL_STEP = 15
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
    
    train_path = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/sampling'
    train_data_path = f'{train_path}/train_data.npy'
    train_label_path = f'{train_path}/train_label.npy'

    subdir = '/sdc1/ytlindata/TON_IoT/ViT/1Dmulti_result(120delall)/' + str(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

    model = ViT1D(
        seq_len = seq_size,
        patch_size = PATCH_SIZE,
        num_classes = num_classes,
        dim = 16,
        depth = 6,
        heads = 8,
        mlp_dim = 32
    )
    Train_1D(model, train_data_path, train_label_path, subdir, hyper_parameter, classes)

def binary_1DViT_delall_120():
    gamma = 0.01
    seed = 42
    init_lr = 0.01
    EPOCH = 100
    BATCH_SIZE = 4096
    PATCH_SIZE = 30
    RL_STEP = 7
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
        'malicious'
    ]
    num_classes = len(classes)

    train_path = '/sdc1/ytlindata/TON_IoT/120_5_flows_delall/binary_sampling'
    train_data_path = f'/{train_path}/train_data.npy'
    train_label_path = f'/{train_path}/train_label.npy'

    subdir = '/sdc1/ytlindata/TON_IoT/ViT/binary_1Dmulti_result(120delall)/' + str(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

    model = ViT1D(
        seq_len = seq_size,
        patch_size = PATCH_SIZE,
        num_classes = num_classes,
        dim = 16,
        depth = 6,
        heads = 8,
        mlp_dim = 32
    )
    Train_1D(model, train_data_path, train_label_path, subdir, hyper_parameter, classes)

if __name__ == '__main__':
    file_handler = logging.FileHandler('./log/TON_training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    multi_1DViT_delall_120()
    # binary_1DViT_delall_120()