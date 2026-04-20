import numpy as np
import os
import random
from utils.HAR_utils import split_data, save_file


random.seed(1)
np.random.seed(1)
data_path = "MHEALTH/"
dir_path = "MHEALTH/"
num_clients = 20  # 扩展到20个客户端

sample_window = 128  # 窗口大小，约2.56秒 (50Hz采样率)

# Dirichlet分布参数
DIRICHLET_ALPHA = 0.1  # 设置为0.1可获得极高异质性的Non-IID分布


def generate_dataset(dir_path, target_clients=20, use_dirichlet=True, alpha=0.1):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # 创建rawdata目录并复制/链接原始数据
    rawdata_path = dir_path + "rawdata/"
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)
    
    # 检查原始数据是否存在
    original_data_path = "MHEALTHDATASET/"
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Original MHEALTH dataset not found at {original_data_path}")

    # 加载数据
    X_subjects, y_subjects = load_data_MHEALTH(original_data_path)
    
    if use_dirichlet:
        # 使用Dirichlet分布重新分配数据
        print(f"\n使用Dirichlet分布 (α={alpha}) 重新分配数据...")
        X, y = distribute_data_dirichlet(X_subjects, y_subjects, target_clients, alpha)
    else:
        # 原始方法：扩展到20个客户端
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
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [item for item in statistic[i]])
        print("-" * 50)

    # 分割数据
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)


def expand_to_clients(X_subjects, y_subjects, target_num_clients):
    """
    将10个受试者的数据扩展到20个客户端
    策略：每个受试者按时间顺序分为两部分，创建两个客户端
    """
    if target_num_clients != 20:
        raise ValueError("Currently only supports expanding to 20 clients")
    
    if len(X_subjects) != 10:
        raise ValueError("Expected 10 subjects")
    
    X_expanded = []
    y_expanded = []
    
    for subject_idx, (X_subject, y_subject) in enumerate(zip(X_subjects, y_subjects)):
        # 计算分割点（时间顺序的中点）
        split_point = len(X_subject) // 2
        
        # 第一个客户端：前半部分数据
        X1 = X_subject[:split_point]
        y1 = y_subject[:split_point]
        
        # 第二个客户端：后半部分数据
        X2 = X_subject[split_point:]
        y2 = y_subject[split_point:]
        
        # 确保两个客户端都有足够的数据
        if len(X1) > 50 and len(X2) > 50:  # 至少50个样本
            X_expanded.append(X1)
            y_expanded.append(y1)
            X_expanded.append(X2)
            y_expanded.append(y2)
            
            print(f"Subject {subject_idx + 1} -> Client {len(X_expanded)-1}: {len(X1)} samples")
            print(f"Subject {subject_idx + 1} -> Client {len(X_expanded)}: {len(X2)} samples")
        else:
            # 如果数据太少，将整个受试者作为一个客户端
            X_expanded.append(X_subject)
            y_expanded.append(y_subject)
            print(f"Subject {subject_idx + 1} -> Client {len(X_expanded)}: {len(X_subject)} samples (not split)")
    
    # 如果客户端数量不足20，用数据增强或重复策略
    while len(X_expanded) < target_num_clients:
        # 选择数据量最大的客户端进行复制（添加噪声）
        max_idx = np.argmax([len(x) for x in X_expanded])
        
        # 添加轻微噪声来创建新的客户端
        X_copy = X_expanded[max_idx].copy()
        y_copy = y_expanded[max_idx].copy()
        
        # 添加少量高斯噪声 (标准差为数据标准差的1%)
        noise_std = np.std(X_copy, axis=(0, 1)) * 0.01
        noise = np.random.normal(0, noise_std, X_copy.shape)
        X_copy = X_copy + noise
        
        X_expanded.append(X_copy)
        y_expanded.append(y_copy)
        print(f"Augmented Client {len(X_expanded)}: {len(X_copy)} samples (from Client {max_idx + 1})")
    
    return X_expanded[:target_num_clients], y_expanded[:target_num_clients]


def load_data_MHEALTH(data_folder):
    """
    加载MHEALTH数据集
    
    数据格式:
    - 列1-3: 胸部传感器加速度 (X, Y, Z)
    - 列4-5: ECG信号 (lead 1, lead 2)
    - 列6-8: 左踝传感器加速度 (X, Y, Z)
    - 列9-11: 左踝传感器陀螺仪 (X, Y, Z)
    - 列12-14: 左踝传感器磁力计 (X, Y, Z)
    - 列15-17: 右前臂传感器加速度 (X, Y, Z)
    - 列18-20: 右前臂传感器陀螺仪 (X, Y, Z)
    - 列21-23: 右前臂传感器磁力计 (X, Y, Z)
    - 列24: 标签 (0为空标签)
    
    活动标签:
    L1: 静止站立 (标签1)
    L2: 坐着放松 (标签2)
    L3: 躺下 (标签3)
    L4: 步行 (标签4)
    L5: 爬楼梯 (标签5)
    L6: 腰部前弯 (标签6)
    L7: 正面举臂 (标签7)
    L8: 膝盖弯曲(蹲下) (标签8)
    L9: 骑自行车 (标签9)
    L10: 慢跑 (标签10)
    L11: 跑步 (标签11)
    L12: 前后跳跃 (标签12)
    """
    
    # 10个受试者的文件
    subject_files = [f'mHealth_subject{i}.log' for i in range(1, 11)]
    
    XX, YY = [], []
    
    for subject_id, filename in enumerate(subject_files):
        file_path = os.path.join(data_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        print(f"Loading data from {filename}...")
        
        # 加载数据
        data = np.loadtxt(file_path, dtype=np.float32)
        print(f"Subject {subject_id + 1}: Raw data shape: {data.shape}")
        
        # 提取传感器数据 (排除ECG信号，使用加速度、陀螺仪和磁力计数据)
        # 使用列: 1-3(胸部加速度), 6-14(左踝传感器), 15-23(右前臂传感器)
        sensor_data = np.concatenate([
            data[:, 0:3],    # 胸部加速度
            data[:, 5:14],   # 左踝传感器 (加速度+陀螺仪+磁力计)
            data[:, 14:23]   # 右前臂传感器 (加速度+陀螺仪+磁力计)
        ], axis=1)
        
        labels = data[:, -1].astype(int)  # 标签列
        
        # 移除标签为0的数据 (空标签)
        valid_indices = labels != 0
        sensor_data = sensor_data[valid_indices]
        labels = labels[valid_indices]
        
        # 数据归一化
        sensor_data = normalize_data(sensor_data)
        
        # 使用滑动窗口分割数据
        X, Y = sliding_window_segmentation(sensor_data, labels, sample_window)
        
        if len(X) > 0:
            # 重新调整标签，从0开始
            Y = adjust_labels(Y)
            
            XX.append(X)
            YY.append(Y)
            print(f"Subject {subject_id + 1}: Segmented data shape: {X.shape}, Labels shape: {Y.shape}")
            print(f"Subject {subject_id + 1}: Unique labels: {np.unique(Y)}")
        else:
            print(f"Subject {subject_id + 1}: No valid data after segmentation")
    
    return XX, YY


def normalize_data(data):
    """数据归一化"""
    # 对每个特征维度进行归一化
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        column = data[:, i]
        if np.std(column) > 0:
            normalized_data[:, i] = (column - np.mean(column)) / np.std(column)
        else:
            normalized_data[:, i] = column
    return normalized_data


def sliding_window_segmentation(sensor_data, labels, window_size):
    """使用滑动窗口分割数据"""
    X, Y = [], []
    
    step_size = window_size // 2  # 50% 重叠
    
    for i in range(0, len(sensor_data) - window_size + 1, step_size):
        # 提取窗口数据
        window_data = sensor_data[i:i + window_size]
        window_labels = labels[i:i + window_size]
        
        # 获取窗口中最常见的标签
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        
        # 只保留主要标签占70%以上的窗口
        if counts[np.argmax(counts)] >= window_size * 0.7:
            # 重塑数据为 (samples, timesteps, features)
            # 传感器数据有21个特征 (3个传感器位置，每个7个特征)
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
    """调整标签，使其从0开始连续"""
    unique_labels = np.unique(labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    adjusted_labels = np.array([label_mapping[label] for label in labels])
    return adjusted_labels


def distribute_data_dirichlet(X_subjects, y_subjects, num_clients, alpha=0.1):
    """
    使用Dirichlet分布重新分配数据，创建Non-IID分布
    
    参数:
        X_subjects: 所有受试者的数据列表
        y_subjects: 所有受试者的标签列表
        num_clients: 客户端数量
        alpha: Dirichlet分布参数，越小Non-IID程度越高
        
    返回:
        X_clients: 每个客户端的数据
        y_clients: 每个客户端的标签
    """
    # 合并所有受试者的数据
    X_all = np.concatenate(X_subjects, axis=0)
    y_all = np.concatenate(y_subjects, axis=0)
    
    print(f"合并后总样本数: {len(X_all)}")
    print(f"标签分布: {np.unique(y_all, return_counts=True)}")
    
    # 获取所有唯一标签
    unique_labels = np.unique(y_all)
    num_classes = len(unique_labels)
    
    print(f"类别数: {num_classes}")
    print(f"类别: {unique_labels}")
    
    # 为每个客户端分配数据
    client_data_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别使用Dirichlet分布
    for label in unique_labels:
        # 获取该类别的所有样本索引
        label_indices = np.where(y_all == label)[0]
        np.random.shuffle(label_indices)
        
        # 使用Dirichlet分布生成比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # 根据比例分配样本
        proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        
        # 分割索引
        label_splits = np.split(label_indices, proportions)
        
        # 分配给各个客户端
        for client_idx, indices in enumerate(label_splits):
            client_data_indices[client_idx].extend(indices)
    
    # 构建客户端数据
    X_clients = []
    y_clients = []
    
    for client_idx in range(num_clients):
        indices = client_data_indices[client_idx]
        
        if len(indices) == 0:
            # 如果某个客户端没有数据，从数据最多的客户端分一些
            print(f"警告: 客户端 {client_idx} 没有数据，重新分配...")
            max_client = np.argmax([len(idx) for idx in client_data_indices])
            # 取10%的数据
            num_to_take = max(10, len(client_data_indices[max_client]) // 10)
            indices = client_data_indices[max_client][:num_to_take]
            client_data_indices[max_client] = client_data_indices[max_client][num_to_take:]
        
        # 随机打乱
        np.random.shuffle(indices)
        
        X_client = X_all[indices]
        y_client = y_all[indices]
        
        X_clients.append(X_client)
        y_clients.append(y_client)
        
        # 打印统计信息
        unique, counts = np.unique(y_client, return_counts=True)
        print(f"客户端 {client_idx:2d}: {len(indices):4d} 样本, "
              f"{len(unique):2d} 个类别, 分布: {dict(zip(unique, counts))}")
    
    return X_clients, y_clients


if __name__ == "__main__":
    # 使用Dirichlet分布生成Non-IID数据
    # alpha=0.1: 极高异质性
    # alpha=0.5: 高异质性
    # alpha=1.0: 中等异质性
    # alpha=10.0: 接近IID
    
    generate_dataset(dir_path, num_clients, use_dirichlet=True, alpha=DIRICHLET_ALPHA)