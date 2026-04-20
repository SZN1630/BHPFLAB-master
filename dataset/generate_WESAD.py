import numpy as np
import os
import random
import pickle
from utils.HAR_utils import split_data, save_file


random.seed(1)
np.random.seed(1)
data_path = "WESAD/"
dir_path = "WESAD/"
num_clients = 20  # 扩展到20个客户端

# 窗口参数
sample_window = 350  # 0.5秒窗口 (700Hz采样率)
step_size = 175     # 50%重叠


def generate_dataset(dir_path, target_clients=20):
    """生成WESAD数据集用于联邦学习"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # 创建rawdata目录
    rawdata_path = dir_path + "rawdata/"
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)

    # 加载数据
    X_subjects, y_subjects = load_data_WESAD(data_path)
    
    # 扩展到20个客户端
    X, y = expand_to_clients(X_subjects, y_subjects, target_clients)
    
    # 统计信息
    statistic = []
    num_clients = len(y)
    num_classes = len(np.unique(np.concatenate(y, axis=0)))
    
    for i in range(num_clients):
        statistic.append([])
        for yy in sorted(np.unique(y[i])):
            idx = y[i] == yy
            statistic[-1].append((int(yy), int(len(X[i][idx]))))

    for i in range(num_clients):
        print(f"Client {i:2d}\t 数据量: {len(X[i])}\t 标签分布: ", np.unique(y[i]))
        print(f"\t\t 各标签样本数: ", [item for item in statistic[i]])
        print("-" * 60)

    # 分割训练测试数据
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)


def load_data_WESAD(data_folder):
    """
    加载WESAD数据集
    
    WESAD (WEarable Stress and Affect Detection) 数据集特点:
    - 15个有效受试者 (S1和S12因传感器故障被弃用)
    - 胸戴设备RespiBAN: ECG, EDA, EMG, RESP, TEMP, ACC (700Hz)
    - 腕戴设备Empatica E4: ACC, BVP, EDA, TEMP (不同采样率)
    - 4种有效状态: 1=基线, 2=压力, 3=愉悦, 4=冥想
    - 忽略: 0=过渡, 5/6/7=无效状态
    """
    
    # 获取所有可用受试者
    available_subjects = []
    for item in os.listdir(data_folder):
        item_path = os.path.join(data_folder, item)
        if os.path.isdir(item_path) and item.startswith('S'):
            pkl_file = os.path.join(item_path, f"{item}.pkl")
            if os.path.exists(pkl_file):
                available_subjects.append(item)
    
    available_subjects = sorted(available_subjects)
    print(f"发现 {len(available_subjects)} 个有效受试者: {available_subjects}")
    
    XX, YY = [], []
    
    for subject_id in available_subjects:
        pkl_file = os.path.join(data_folder, subject_id, f"{subject_id}.pkl")
        
        print(f"正在处理 {subject_id}...")
        
        # 加载数据
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # 提取胸戴设备传感器数据 (主要使用，因为采样率一致都是700Hz)
        chest_signals = data['signal']['chest']
        labels = data['label']
        
        print(f"  原始数据长度: {len(labels):,} 样本 ({len(labels)/700:.1f}秒)")
        
        # 合并胸戴设备的传感器数据
        sensor_data = np.concatenate([
            chest_signals['ECG'],      # 心电图 (1维)
            chest_signals['EDA'],      # 皮肤电活动 (1维)  
            chest_signals['EMG'],      # 肌电图 (1维)
            chest_signals['Resp'],     # 呼吸 (1维)
            chest_signals['Temp'],     # 温度 (1维)
            chest_signals['ACC']       # 加速度 (3维)
        ], axis=1)  # 总共8维特征
        
        # 过滤有效标签 (1,2,3,4) 并移除过渡和无效状态
        valid_mask = np.isin(labels, [1, 2, 3, 4])
        sensor_data = sensor_data[valid_mask]
        labels = labels[valid_mask]
        
        print(f"  过滤后数据长度: {len(labels):,} 样本")
        
        # 数据归一化
        sensor_data = normalize_data(sensor_data)
        
        # 滑动窗口分割
        X, Y = sliding_window_segmentation(sensor_data, labels, sample_window, step_size)
        
        if len(X) > 0:
            # 调整标签为连续的0-3
            Y = adjust_labels(Y)
            
            XX.append(X)
            YY.append(Y)
            print(f"  分割后数据: {X.shape}, 标签: {Y.shape}")
            print(f"  标签分布: {np.unique(Y, return_counts=True)}")
        else:
            print(f"  警告: {subject_id} 没有有效数据")
    
    return XX, YY


def normalize_data(data):
    """Z-score标准化"""
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        column = data[:, i]
        if np.std(column) > 0:
            normalized_data[:, i] = (column - np.mean(column)) / np.std(column)
        else:
            normalized_data[:, i] = column
    return normalized_data


def sliding_window_segmentation(sensor_data, labels, window_size, step_size):
    """滑动窗口分割数据"""
    X, Y = [], []
    
    for i in range(0, len(sensor_data) - window_size + 1, step_size):
        # 提取窗口数据和标签
        window_data = sensor_data[i:i + window_size]
        window_labels = labels[i:i + window_size]
        
        # 获取窗口中最常见的标签
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        
        # 只保留主要标签占80%以上的窗口(更严格的纯度要求)
        if counts[np.argmax(counts)] >= window_size * 0.8:
            # 重塑数据为 (1, timesteps, features)
            window_data_reshaped = window_data.reshape(1, window_size, -1)
            X.append(window_data_reshaped)
            Y.append(dominant_label)
    
    if len(X) > 0:
        X = np.concatenate(X, axis=0)
        Y = np.array(Y)
    else:
        X = np.array([])
        Y = np.array([])
    
    return X, Y


def adjust_labels(labels):
    """调整标签: 1,2,3,4 -> 0,1,2,3"""
    label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}  # 基线,压力,愉悦,冥想
    adjusted_labels = np.array([label_mapping[label] for label in labels])
    return adjusted_labels


def expand_to_clients(X_subjects, y_subjects, target_clients=20):
    """将15个受试者的数据扩展到20个客户端"""
    
    if len(X_subjects) == 0:
        raise ValueError("没有可用的受试者数据")
    
    print(f"\n=== 扩展数据到 {target_clients} 个客户端 ===")
    
    X_expanded = []
    y_expanded = []
    
    # 策略1: 前15个客户端直接对应15个受试者
    for subject_idx, (X_subject, y_subject) in enumerate(zip(X_subjects, y_subjects)):
        X_expanded.append(X_subject)
        y_expanded.append(y_subject)
        print(f"受试者 {subject_idx + 1} -> 客户端 {len(X_expanded)}: {len(X_subject)} 个样本")
    
    # 策略2: 为了达到20个客户端，对数据量大的受试者进行分割
    while len(X_expanded) < target_clients:
        # 找到数据量最大的客户端进行分割
        sizes = [len(x) for x in X_expanded]
        max_idx = np.argmax(sizes)
        
        if sizes[max_idx] < 100:  # 如果最大的客户端数据量也很小，则进行数据增强
            # 数据增强策略
            X_copy = X_expanded[max_idx].copy()
            y_copy = y_expanded[max_idx].copy()
            
            # 添加轻微噪声
            noise_std = np.std(X_copy, axis=(0, 1)) * 0.01
            noise = np.random.normal(0, noise_std, X_copy.shape)
            X_copy = X_copy + noise
            
            X_expanded.append(X_copy)
            y_expanded.append(y_copy)
            print(f"数据增强 -> 客户端 {len(X_expanded)}: {len(X_copy)} 个样本 (基于客户端 {max_idx + 1})")
        else:
            # 分割策略
            X_original = X_expanded[max_idx]
            y_original = y_expanded[max_idx]
            
            split_point = len(X_original) // 2
            
            # 更新原客户端为前半部分
            X_expanded[max_idx] = X_original[:split_point]
            y_expanded[max_idx] = y_original[:split_point]
            
            # 添加新客户端为后半部分
            X_expanded.append(X_original[split_point:])
            y_expanded.append(y_original[split_point:])
            
            print(f"分割客户端 {max_idx + 1}: {len(X_expanded[max_idx])} + {len(X_expanded[-1])} 个样本")
    
    return X_expanded[:target_clients], y_expanded[:target_clients]


if __name__ == "__main__":
    generate_dataset(dir_path, num_clients)