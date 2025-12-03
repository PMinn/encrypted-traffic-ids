import json
import logging
import numpy as np
import os

def get_oversampling_by_kmeans(x, y, strategy, k = 3, cluster_balance_threshold = 0.005):
    from imblearn.over_sampling import KMeansSMOTE
    before_unique, before_counts = np.unique(y, return_counts = True)
    oversample = KMeansSMOTE(sampling_strategy = strategy, random_state = 42, k_neighbors = k, cluster_balance_threshold = cluster_balance_threshold)  # 0.0016
    x, y = oversample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts = True)
    return x, y, (dict(zip(before_unique, before_counts)), dict(zip(after_unique, after_counts)))

def get_oversampling_by_random(x, y, strategy):
    from imblearn.over_sampling import RandomOverSampler
    before_unique, before_counts = np.unique(y, return_counts = True)
    oversample = RandomOverSampler(sampling_strategy = strategy, random_state = 42)
    x, y = oversample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts = True)
    return x, y, (dict(zip(before_unique, before_counts)), dict(zip(after_unique, after_counts)))

def get_undersampling_by_random(x, y, strategy):
    from imblearn.under_sampling import RandomUnderSampler
    before_unique, before_counts = np.unique(y, return_counts = True)
    undersample = RandomUnderSampler(sampling_strategy = strategy, random_state = 42)
    x, y = undersample.fit_resample(x, y)
    after_unique, after_counts = np.unique(y, return_counts = True)
    return x, y, (dict(zip(before_unique, before_counts)), dict(zip(after_unique, after_counts)))

def load_data(classes_info: dict, classes_names: list):
    # classes_id = {}
    x = []
    y = []
    data_counts = {}
    for index, classes_name in enumerate(classes_names):
        data_path = classes_info[classes_name]['data_path']
        # label_path = classes_info[classes_name]['label_path']
        data = np.load(data_path)
        number_of_data = data.shape[0]
        x.extend(data)
        y.extend( np.ones(number_of_data).astype(int) * index)
        # classes_id[classes_name] = label[0]  # 取得該類別的標籤 ID
        data_counts[classes_name] = number_of_data
    return np.array(x), np.array(y), data_counts
    
def sampling(classes_info: dict, save_to: str, oversampling: callable = get_oversampling_by_random, undersampling: callable = get_undersampling_by_random):
    """
        根據指定的採樣數量對各類別進行過採樣或欠採樣
        Args:
            classes_info (dict): 各類別的資料路徑資訊，格式為 {class_name: {'data_path': str, 'sampling_count': int}}
            save_to (str): 採樣後資料的儲存路徑
        Returns:
            None
    """
    os.makedirs(save_to, exist_ok = True)
    classes_names = list(classes_info.keys())
    data, label, data_counts = load_data(classes_info, classes_names)
    under_strategy = {}
    over_strategy = {}
    for index, class_name in enumerate(classes_names):
        if 'data_path' not in classes_info[class_name]:
            raise SamplingInfoMissingException(f"Class '{class_name}' is missing 'data_path' in classes_info.")
        if 'sampling_count' not in classes_info[class_name]:
            raise SamplingInfoMissingException(f"Class '{class_name}' is missing 'sampling_count' in classes_info.")
        target_count = classes_info[class_name].get('sampling_count', 0)
        current_count = data_counts.get(class_name, 0)
        if target_count < current_count:
            under_strategy[index] = target_count
        elif target_count > current_count:
            over_strategy[index] = target_count
    if len(under_strategy) > 0:
        data, label, process = undersampling(data, label, under_strategy)
        before, after = process
        before_str = '\n'.join([f'{classes_names[k]}: {v}' for k, v in before.items()])
        after_str = '\n'.join([f'{classes_names[k]}: {v}' for k, v in after.items()])
        logging.getLogger("sampling").info(f'Undersampling applied.\nBefore:\n{before_str}.\nAfter:\n{after_str}.')
    if len(over_strategy) > 0:
        data, label, process = oversampling(data, label, over_strategy)
        before, after = process
        before_str = '\n'.join([f'{classes_names[k]}: {v}' for k, v in before.items()])
        after_str = '\n'.join([f'{classes_names[k]}: {v}' for k, v in after.items()])
        logging.getLogger("sampling").info(f'Oversampling applied.\nBefore:\n{before_str}.\nAfter:\n{after_str}.')
    np.save(os.path.join(save_to, 'sampled_data.npy'), data, allow_pickle = False)
    np.save(os.path.join(save_to, 'sampled_label.npy'), label, allow_pickle = False)
    with open(os.path.join(save_to, 'classes.json'), "w") as f:
        json.dump(classes_names, f)
        
class SamplingInfoMissingException(Exception):
    pass
