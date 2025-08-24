from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os
from tqdm import tqdm


class DatasetProcesser():
    def __init__(self, data_path, save_to, classes):
        self.data_path = data_path
        self.save_to = save_to
        self.classes = classes

    def save_np(self, path, data):
        np.save(path, data, allow_pickle = False)

    def makedirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def concat(self, old_data, new_data):
        if old_data is None and new_data.shape[0] != 0:
            concat = new_data
        else:
            try:
                concat = np.concatenate((old_data, new_data))
            except ValueError:
                concat = old_data

        return concat

    def trainTestSplit(self, data, proportion = 0.2):
        train_data, test_data = train_test_split(data, test_size = proportion, random_state = 42)#分訓練/驗證
        return  train_data, test_data

    def get_label(self, id, size):
        return np.ones(size).astype(int)*id

    def get_train_benign(self, data_path, ratio):
        print(f'Processing train benign')
        files = glob.glob(f'{data_path}/*/*/benign_t.npy')
        random_benign = None
        for file in tqdm(files):
            try:
                data = np.load(file)
                if data is None or len(data) == 0:
                    continue
                old_data = self.concat(old_data, data)
            except Exception as e:
                print(f'[Warning] Failed to load {file}: {e}')
                continue
            total_size = data.shape[0]
            sample_size = int(total_size * ratio)
            # 使用numpy.random.choice来随机抽取数据
            random_samples = np.random.choice(total_size, size = sample_size, replace = False)
            random_data = data[random_samples]

            other_samples = np.setdiff1d(np.arange(total_size), random_samples)
            # print(file[:file.rfind('/')])
            other_path = file[:file.rfind('/')]
            self.save_np(f'{other_path}/untrain_benign.npy', data[other_samples])

            random_benign = self.concat(random_benign, random_data)

        return random_benign

    def get_test_benign(self, data_path, ratio):
        print(f'Processing test benign')
        files = glob.glob(f'{data_path}/*/*/untrain_benign.npy')
        random_benign = None
        for file in tqdm(files):
            try:
                data = np.load(file)
                if data is None or len(data) == 0:
                    continue
                old_data = self.concat(old_data, data)
            except Exception as e:
                print(f'[Warning] Failed to load {d}: {e}')
                continue
            total_size = data.shape[0]
            sample_size = int(total_size * ratio)
            # 使用numpy.random.choice来随机抽取数据
            random_samples = np.random.choice(total_size, size = sample_size, replace = False)
            random_data = data[random_samples]

            other_samples = np.setdiff1d(np.arange(total_size), random_samples)
            # print(file[:file.rfind('/')])
            other_path = file[:file.rfind('/')]
            self.save_np(f'{other_path}/untest_benign.npy', data[other_samples])

            random_benign = self.concat(random_benign, random_data)
        return random_benign

    def run(self):
        self.makedirs(f'{self.save_to}/train/')
        self.makedirs(f'{self.save_to}/test/')

        total_count = list()
        train_count = list()
        test_count = list()

        for index, classType in enumerate(self.classes):
            if classType == 'benign':
                train_data = self.get_train_benign(self.data_path, 0.04)
                # print('aa', train_data)
                # break
                self.save_np(f'{self.save_to}/train/{classType}.npy', train_data)
                train_size = train_data.shape[0]
                del train_data
                self.save_np(f'{self.save_to}/train/{classType}_label.npy', self.get_label(index, train_size))
                total_count.append(train_size)
                train_count.append(train_size)

                test_data = self.get_test_benign(self.data_path, 0.04)
                test_size = test_data.shape[0]
                self.save_np(f'{self.save_to}/test/{classType}.npy', test_data)
                del test_data
                self.save_np(f'{self.save_to}/test/{classType}_label.npy', self.get_label(index, test_size))
                test_count.append(test_size)
                continue

            elif classType != 'webattack':
                dirs = glob.glob(f'{self.data_path}/*/*/{classType}_t.npy')
            else:
                dirs = glob.glob(f'{self.data_path}/*/*/{classType}*_t.npy')

            old_data = None
            print(f'process concat')
            for d in tqdm(dirs):
                try:
                    new_data = np.load(d)
                    if new_data is None or len(new_data) == 0:
                        continue
                except Exception as e:
                    print(f'[Warning] Failed to load {d}: {e}')
                    continue
                old_data = self.concat(old_data, new_data)
            try:
                c_num = old_data.shape[0]
            except AttributeError:
                print(f' {classType} No Data!!')
                c_num = 0

            train_data, test_data = self.trainTestSplit(old_data, proportion = 0.2)
            del old_data

            train_size = train_data.shape[0]
            test_size = test_data.shape[0]

            self.save_np(f'{self.save_to}/train/{classType}.npy', train_data)
            self.save_np(f'{self.save_to}/test/{classType}.npy', test_data)
            del train_data, test_data

            # creat label .npy
            self.save_np(f'{self.save_to}/train/{classType}_label.npy', self.get_label(index, train_size))
            self.save_np(f'{self.save_to}/test/{classType}_label.npy', self.get_label(index, test_size))
            total_count.append(c_num)
            train_count.append(train_size)
            test_count.append(test_size)

        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {total_count[index]}')
        print('*'*15+' train size '+'*'*15)
        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {train_count[index]}')
        print('*'*15+' test size '+'*'*15)
        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {test_count[index]}')

class DatasetProcesser2(DatasetProcesser):
    def __init__(self, data_path, save_to, classes):
        super().__init__(data_path, save_to, classes)

    def run(self):
        self.makedirs(f'{self.save_to}/train/')
        self.makedirs(f'{self.save_to}/test/')

        total_count = list()
        train_count = list()
        test_count = list()

        for index, classType in enumerate(self.classes):
            dirs = glob.glob(f'{self.data_path}/*/*/{classType}*_t*.npy')
            # dirs = glob.glob(f'{self.data_path}/*/*/tftp_t*.npy')
            old_data = None
            print(f'process concat')
            for d in tqdm(dirs):
                new_data = np.load(d)
                new_data = new_data[:len(new_data)]
                old_data = self.concat(old_data, new_data)
            try:
                c_num = old_data.shape[0]
            except AttributeError:
                print(f' {classType} No Data!!')
                c_num = 0

            train_data, test_data = self.trainTestSplit(old_data, proportion = 0.2)
            del old_data

            train_size = train_data.shape[0]
            test_size = test_data.shape[0]

            self.save_np(f'{self.save_to}/train/{classType}.npy', train_data)
            self.save_np(f'{self.save_to}/test/{classType}.npy', test_data)
            del train_data, test_data

            # creat label .npy
            self.save_np(f'{self.save_to}/train/{classType}_label.npy', self.get_label(index, train_size))
            self.save_np(f'{self.save_to}/test/{classType}_label.npy', self.get_label(index, test_size))
            total_count.append(c_num)
            train_count.append(train_size)
            test_count.append(test_size)

        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {total_count[index]}')
        print('*'*15+' train size '+'*'*15)
        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {train_count[index]}')
        print('*'*15+' test size '+'*'*15)
        for index, classType in enumerate(self.classes):
            print(f'{classType} total: {test_count[index]}')

def  preprocess_data_del():

    classes = [
        'benign',
        'malware'
    ]

    ORIG_DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/del_120_5_flows(delall)/'
    DATA_PATH = '/sdc1/ytlindata/USTC-TFC2016/120_5_flows_delallpmd'

    DatasetProcesser2(ORIG_DATA_PATH, DATA_PATH, classes).run()
