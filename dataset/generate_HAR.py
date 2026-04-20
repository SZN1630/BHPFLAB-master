import numpy as np
import os
import random
import urllib.request
import zipfile
from utils.HAR_utils import *


random.seed(1)
np.random.seed(1)
data_path = "HAR/"
dir_path = "HAR/"


def download_file(url, filepath):
    """下载文件的通用函数"""
    try:
        print(f"正在下载: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"下载完成: {filepath}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """解压zip文件的通用函数"""
    try:
        print(f"正在解压: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # download data
    rawdata_path = data_path + 'rawdata/'
    if not os.path.exists(rawdata_path):
        os.makedirs(rawdata_path)
    
    zip_file_path = rawdata_path + 'UCI HAR Dataset.zip'
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    
    if not os.path.exists(zip_file_path):
        if not download_file(dataset_url, zip_file_path):
            print("数据集下载失败，请检查网络连接或手动下载数据集")
            return
    
    if not os.path.exists(rawdata_path + 'UCI HAR Dataset/'):
        if not extract_zip(zip_file_path, rawdata_path):
            print("数据集解压失败，请检查文件完整性")
            return

    X, y = load_data_HAR(data_path+'rawdata/')
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
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)


def load_data_HAR(data_folder):
    str_folder = data_folder + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                        INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                        item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'
    str_train_id = str_folder + 'train/subject_train.txt'
    str_test_id = str_folder + 'test/subject_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
    id_train = read_ids(str_train_id)
    id_test = read_ids(str_test_id)
    
    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ID = np.concatenate((id_train, id_test), axis=0)

    XX, YY = [], []
    for i in np.unique(ID):
        idx = ID == i
        XX.append(X[idx])
        YY.append(Y[idx])

    return XX, YY


if __name__ == "__main__":
    generate_dataset(dir_path)