# import g。lob
import numpy as np
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
# from tqdm.contrib import tzip
# from multiprocessing import Pool
# from sklearn.manifold import TSNE
# import pandas
# import copy
# from tqdm import tqdm

class Sampling():
    SEED = 42
    def __init__(self, classes, data_path = None, save_to = None) -> None:
        self.classes = classes
        self.path = data_path
        self.save_to = save_to


    def load_data(self, path = None):
        if path == None:
            path = self.path
    # load data
        x = []
        y = []
        for c in self.classes:
            print(f'Process Class: {c}')
            data_path = f'{path}/{c}.npy'
            label_path = f'{path}/{c}_label.npy'
            
            x.extend(np.load(data_path))
            y.extend(np.load(label_path))
        # combine all the flow
        # x = np.vstack(x)
        # y = np.hstack(y)
        return np.array(x), np.array(y)
    
    def get_size(self, x, y):
        print('-' * 25 + ' Before Sampling ' + '-' * 25)
        unique, counts = np.unique(y, return_counts = True)
        print(f'y: {dict(zip(unique, counts))}')

    def get_oversampling(self, x, y, over_strategy, k, cluster_balance_threshold):
        # output number of each class
        print('-' * 25 + ' Before Sampling ' + '-' * 25)
        unique, counts = np.unique(y, return_counts = True)
        print(f'y: {dict(zip(unique, counts))}')

        # k-means smote oversampling
        # over_strategy = {8: O}
        oversample = KMeansSMOTE(sampling_strategy = over_strategy, random_state = self.SEED, k_neighbors = k, cluster_balance_threshold = cluster_balance_threshold)  # 0.0016
        x, y = oversample.fit_resample(x, y)

        # output number of each class
        print('-' * 25 + ' After OverSampling ' + '-' * 25)
        unique, counts = np.unique(y, return_counts = True)
        print(f'y: {dict(zip(unique, counts))}')
        return x, y

    def get_undersampling(self, x, y, under_strategy):
        # output number of each class
        print('-' * 25 + ' Before Sampling ' + '-' * 25)
        unique, counts = np.unique(y, return_counts = True)
        print(f'y: {dict(zip(unique, counts))}')
        # random undersampling
        # under_strategy = {0: U, 1: U, 2: U, 3: U, 4: U, 5: U, 7: U, 8: U}
        # under_strategy = {0: 360, 1: 360}
        undersample = RandomUnderSampler(sampling_strategy = under_strategy, random_state = self.SEED)
        x, y = undersample.fit_resample(x, y)

        # output number of each class
        print('-' * 25 + ' After UnderSampling ' + '-' * 25)
        unique, counts = np.unique(y, return_counts = True)
        print(f'y: {dict(zip(unique, counts))}')
        return x, y

    def save_np(self, data, label):
        if not os.path.exists(f'{self.save_to}'):
            os.makedirs(f'{self.save_to}')
        # 檢查是否已經存在資料檔案
        if os.path.exists(f'{self.save_to}/train_data.npy'):
            # 如果存在，載入舊的資料
            old_data = np.load(f'{self.save_to}/train_data.npy')
            old_label = np.load(f'{self.save_to}/train_label.npy')
            
            # 合併舊資料和新資料
            data = np.concatenate((old_data, data), axis = 0)
            label = np.concatenate((old_label, label), axis = 0)
        np.save(f'{self.save_to}/train_data.npy', data, allow_pickle = False)
        np.save(f'{self.save_to}/train_label.npy', label, allow_pickle = False)
        print(f'save to: {self.save_to}')
        
    def save_np_per_class(self, data, label):
        """每個類別獨立存成一個檔案（使用類別名稱命名）"""
        if not os.path.exists(f'{self.save_to}'):
            os.makedirs(f'{self.save_to}')
        
        unique_labels = np.unique(label)
        for ul in unique_labels:
            idx = np.where(label == ul)[0]
            data_class = data[idx]
            label_class = label[idx]

            class_name = self.classes[ul]  # 取得對應的類別名稱

            save_data_path = f'{self.save_to}/{class_name}_data.npy'
            save_label_path = f'{self.save_to}/{class_name}_label.npy'

            np.save(save_data_path, data_class, allow_pickle = False)
            np.save(save_label_path, label_class, allow_pickle = False)
            print(f'[save_np_per_class] Saved class {ul} ({class_name}) to {save_data_path} and {save_label_path}')


class Sampling_Binary(Sampling):

    def save_np(self, data, label):
        label[label != 0] = 1
        if not os.path.exists(f'{self.save_to}'):
            os.makedirs(f'{self.save_to}')
        if os.path.exists(f'{self.save_to}/train_data.npy'):
            # 如果存在，載入舊的資料
            old_data = np.load(f'{self.save_to}/train_data.npy')
            old_label = np.load(f'{self.save_to}/train_label.npy')
            
            # 合併舊資料和新資料
            data = np.concatenate((old_data, data), axis = 0)
            label = np.concatenate((old_label, label), axis = 0)
        np.save(f'{self.save_to}/train_data.npy', data, allow_pickle = False)
        np.save(f'{self.save_to}/train_label.npy', label, allow_pickle = False)
        print(f'save to: {self.save_to}')


def run_sampling(classes, DATA_PATH, MULTI_PATH, UNDERSAMPLING, OVERSAMPLING, isTestingData = False):
    if len(classes) < 2:
        raise ValueError("Classes list must contain at least two classes for sampling.")

    # 自動建立 label2id
    label2id = { label: idx for idx, label in enumerate(classes) }
    # 轉換函式
    def convert_dict(d, mapping):
        return { mapping[k]: v for k, v in d.items() if k in mapping }
    # 轉換
    UNDERSAMPLING = convert_dict(UNDERSAMPLING, label2id)
    OVERSAMPLING = convert_dict(OVERSAMPLING, label2id)
    
    sampling = Sampling(classes, DATA_PATH, MULTI_PATH)
    # load data
    data, label = sampling.load_data()
    if not isTestingData:
        #########################  Oversampling ##############################################
        data, label = sampling.get_oversampling(data, label, OVERSAMPLING, k = 3, cluster_balance_threshold = 0.005)
        # #########################  Undersampling #############################################
        data, label = sampling.get_undersampling(data, label, under_strategy = UNDERSAMPLING)

    sampling.save_np(data, label)
    del data, label

def run_binary_sampling(classes, DATA_PATH, BINARY_PATH, UNDERSAMPLING, OVERSAMPLING, isTestingData = False):
    if len(classes) < 2:
        raise ValueError("Classes list must contain at least two classes for binary sampling.")
    
    # 自動建立 label2id
    label2id = { label: idx for idx, label in enumerate(classes) }
    # 轉換函式
    def convert_dict(d, mapping):
        return { mapping[k]: v for k, v in d.items() if k in mapping }
    # 轉換
    UNDERSAMPLING = convert_dict(UNDERSAMPLING, label2id)
    OVERSAMPLING = convert_dict(OVERSAMPLING, label2id)

    sampling = Sampling_Binary(classes, DATA_PATH, BINARY_PATH)
    # load data
    data, label = sampling.load_data()
    if not isTestingData:
        # #########################  Oversampling ##############################################
        data, label = sampling.get_oversampling(data, label, OVERSAMPLING, k = 3, cluster_balance_threshold = 0.005)
        # #########################  Undersampling #############################################
        data, label = sampling.get_undersampling(data, label, under_strategy = UNDERSAMPLING)

    sampling.save_np(data, label)
    del data, label