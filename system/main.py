#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import math

from flcore.servers.serveravg import FedAvg
from flcore.servers.serveras import FedAS
from flcore.servers.servermul import FedMul

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    
    '''========================step-1: trigger initialization====================='''
    ddevices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trigger_list = []
    ds = args.dataset.lower()
    
    if ds in ["mnist", "fmnist", "fashionmnist"]:
        image_size = 28
        base_trigger = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=ddevices)
        trigger_patten = [[i, j]
                        for i in range(10, min(20, image_size-1))
                        for j in range(10, min(20, image_size-1))]
        for i, j in trigger_patten:
            base_trigger[0, i, j] = 0.8
        trigger_list = [copy.deepcopy(base_trigger) for _ in range(10)]
        assert trigger_list[0].shape == (1, 28, 28)

    elif ds in ["har", "uci_har"]: # size: 9 x 1 x 128
        C, H, W = 9, 1, 128 # 中间的1是维度对齐必要，类似fashionmnist
        base_trigger = torch.zeros((C, H, W), dtype=torch.float32, device=ddevices)

        # 时间窗参数 Q*
        win_len, start, amp = 21, 64, 0.15
        t = torch.arange(win_len, device=ddevices)
        patch = amp * torch.hann_window(win_len, periodic=True, device=ddevices) * torch.sin(2*math.pi*t/win_len)

        # 只在少数通道上加扰动 Q*
        for c in [0, 3]:
            base_trigger[c, 0, start:start+win_len] = patch
        trigger_list = [copy.deepcopy(base_trigger) for _ in range(10)]
        
        # 为HAR数据集定义trigger_patten（用于兼容性）
        trigger_patten = [[c, 0, t] for c in [0, 3] for t in range(start, start+win_len)]
        
    elif ds in ["mhealth"]: # size: 21 x 1 x 128
        C, H, W = 21, 1, 128 # 21个特征，1个高度维度，128个时间步
        base_trigger = torch.zeros((C, H, W), dtype=torch.float32, device=ddevices)

        # 时间窗参数
        win_len, start, amp = 32, 48, 0.12
        t = torch.arange(win_len, device=ddevices)
        patch = amp * torch.hann_window(win_len, periodic=True, device=ddevices) * torch.sin(2*math.pi*t/win_len)

        # 只在少数通道上加扰动
        for c in [0, 3]:
            base_trigger[c, 0, start:start+win_len] = patch
        trigger_list = [copy.deepcopy(base_trigger) for _ in range(10)]
        
        # 为MHEALTH数据集定义trigger_patten（用于兼容性）
        trigger_patten = [[c, 0, t] for c in [0, 3] for t in range(start, start+win_len)]
        
    elif ds in ["wesad"]: # size: 8 x 1 x 350
        C, H, W = 8, 1, 350 # 8个特征，1个高度维度，350个时间步
        base_trigger = torch.zeros((C, H, W), dtype=torch.float32, device=ddevices)

        # 时间窗参数
        win_len, start, amp = 50, 150, 0.15
        t = torch.arange(win_len, device=ddevices)
        patch = amp * torch.hann_window(win_len, periodic=True, device=ddevices) * torch.sin(2*math.pi*t/win_len)

        # 只在少数通道上加扰动
        for c in [0, 1]:
            base_trigger[c, 0, start:start+win_len] = patch
        trigger_list = [copy.deepcopy(base_trigger) for _ in range(4)]  # WESAD有4个类别
        
        # 为WESAD数据集定义trigger_patten（用于兼容性）
        trigger_patten = [[c, 0, t] for c in [0, 1] for t in range(start, start+win_len)]
        
    else:
        raise TypeError("Wrong dataset")
    '''=================================================================='''

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "FashionMNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "HAR" in args.dataset or "UCI_HAR" in args.dataset:
                args.model = HARCNN(9, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif "MHEALTH" in args.dataset:
                args.model = MHEALTHCNN(21, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                        pool_kernel_size=(1, 2)).to(args.device)
            elif "WESAD" in args.dataset:
                args.model = WESADCNN(8, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                      pool_kernel_size=(1, 2)).to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes).to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'MHEALTH':
                args.model = MHEALTHCNN(21, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                        pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'WESAD':
                args.model = WESADCNN(8, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                      pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, num_classes=args.num_classes, conv_kernel_size=(1, 2), 
                                    pool_kernel_size=(1, 2)).to(args.device)

        elif model_str == "FedAvgCNN":
            if "MNIST" in args.dataset or "FashionMNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "HAR" in args.dataset or "UCI_HAR" in args.dataset:
                args.model = HARCNN(9, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                    pool_kernel_size=(1, 2)).to(args.device)
            elif "MHEALTH" in args.dataset:
                args.model = MHEALTHCNN(21, num_classes=args.num_classes, conv_kernel_size=(1, 5), 
                                        pool_kernel_size=(1, 2)).to(args.device)
            elif "WESAD" in args.dataset:
                args.model = WESADCNN(8, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                      pool_kernel_size=(1, 2)).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            #args.head = copy.deepcopy(args.model.fc)
            #args.model.fc = nn.Identity()
            #args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)
        
        elif args.algorithm == 'FedMul':
            # FedMul使用模型分割：base共享，head个性化
            if hasattr(args.model, 'fc'):
                args.head = copy.deepcopy(args.model.fc)
                args.model.fc = nn.Identity()
                args.model = BaseHeadSplit(args.model, args.head)
            server = FedMul(args, i)

        else:
            raise NotImplementedError

        # 分为有攻击和无攻击
        if args.attack != 'none':
            server.train_with_attack(pattern=trigger_patten, trigger=trigger_list)
        else:
            server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # Poisoning
    parser.add_argument('-po', "--poison_rate", type=int, default=4,
                        help="Poison rate")
    parser.add_argument('-am', "--attack", type=str, default='none',
                         help="Attack method for fed system.")
    parser.add_argument('-pl', "--poison_label", type=int, default=1,
                        help="Target label for poisoning attack")
    parser.add_argument('-nm', "--num_malicious", type=int, default=2,
                        help="Number of malicious clients")
    # Model Replacement Attack specific parameters
    parser.add_argument('-mr_tl', "--model_replace_target_label", type=int, default=0,
                        help="Target label for model replacement attack")
    parser.add_argument('-mr_ts', "--model_replace_trigger_size", type=int, default=4,
                        help="Trigger size for model replacement attack")
    parser.add_argument('-mr_be', "--model_replace_backdoor_epochs", type=int, default=15,
                        help="Backdoor training epochs for model replacement attack")
    parser.add_argument('-mr_cr', "--model_replace_clean_ratio", type=float, default=0.7,
                        help="Clean data ratio for model replacement attack")
    parser.add_argument('-mr_lr', "--model_replace_lr", type=float, default=0.01,
                        help="Learning rate for model replacement attack")
    # BadNets attack parameters
    parser.add_argument('-bn_tl', "--badnets_target_label", type=int, default=0,
                        help="Target label for BadNets attack")
    parser.add_argument('-bn_ts', "--badnets_trigger_size", type=int, default=4,
                        help="Trigger size for BadNets attack")
    parser.add_argument('-bn_to', "--badnets_trigger_opacity", type=float, default=0.8,
                        help="Trigger opacity for BadNets attack")
    parser.add_argument('-bn_pr', "--badnets_poison_rate", type=float, default=0.5,
                        help="Poison rate for BadNets attack (fraction of training data poisoned)")
    # Defense parameters
    parser.add_argument('-def', "--defense", type=str, default='none',
                        help="Defense method: none, robust_aggregation")
    parser.add_argument('-ra_method', "--robust_agg_method", type=str, default='median',
                        help="Robust aggregation method: median, trimmed_mean, krum")
    parser.add_argument('-ra_trim', "--robust_trim_ratio", type=float, default=0.1,
                        help="Trim ratio for trimmed mean aggregation")
    parser.add_argument('-ra_f', "--robust_f", type=int, default=1,
                        help="Number of Byzantine clients for Krum aggregation")
    # Baseline defense parameters (gradient clipping + median aggregation + krum aggregation)
    parser.add_argument('-dm', "--defense_mode", type=str, default=None,
                        help="Defense mode: 'clip', 'median', 'krum', 'clip+median', 'clip+krum', or None")
    parser.add_argument('-cn', "--clip_norm", type=float, default=1.0,
                        help="Maximum norm for gradient clipping")
    parser.add_argument('-kb', "--krum_byzantine", type=int, default=0,
                        help="Number of suspected Byzantine (malicious) clients for Krum aggregation")
    parser.add_argument('-mk', "--multi_krum", action='store_true',
                        help="Use Multi-Krum (average top-m models) instead of standard Krum (select 1 model)")
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    
        # FedMul (MultiSim) specific parameters
        # FedMul (MultiSim) specific parameters
    parser.add_argument('-nc_mul', "--num_clusters", type=int, default=3,
                        help="Number of clusters for FedMul")

    # 新增：聚类链接方法
    parser.add_argument('-lm_mul', "--linkage_method", type=str, default='single',
                        choices=['single', 'complete', 'average', 'ward'],
                        help="Linkage method for hierarchical clustering (paper uses 'single')")

    # 新增：个性化正则化强度 (论文中的 alpha)
    parser.add_argument('-alpha_mul', "--alpha", type=float, default=0.01,
                        help="Regularization strength for similarity constraint (alpha in paper)")

    parser.add_argument('-ir_mul', "--initial_rounds", type=int, default=10,
                        help="Number of initial rounds for clustering phase (T0 in paper)")

    # 注意：以下参数在修正后的代码中可能作用有限或已失效
    parser.add_argument('-ucf_mul', "--update_cluster_freq", type=int, default=5,
                        help="[DEPRECATED] Frequency of cluster updates (clustering now happens once)")
    parser.add_argument('-pw_mul', "--personalization_weight", type=float, default=0.5,
                        help="Personalization loss weight in FedMul")
    parser.add_argument('-caw_mul', "--cluster_aggregation_weight", type=float, default=0.7,
                        help="Cluster aggregation weight in FedMul")
    parser.add_argument('-rt_mul', "--robustness_threshold", type=float, default=0.3,
                        help="Robustness threshold for malicious detection in FedMul")
    parser.add_argument('-mhl_mul', "--max_history_len", type=int, default=5,
                        help="Maximum history length for gradient/weight tracking in FedMul")
    
    # FedMul优化参数：Top-K邻居
    parser.add_argument('-top_k', "--top_k_neighbors", type=int, default=3,
                        help="Number of top-K neighbors for similarity constraint (default: 3)")
    
    # FedMul优化参数：稀疏计算
    parser.add_argument('-sl_freq', "--similarity_loss_freq", type=int, default=5,
                        help="Frequency of similarity loss computation (every N batches, default: 5)")
    
    # FedMul优化参数：PCA降维
    parser.add_argument('-use_pca', "--use_pca_similarity", type=bool, default=True,
                        help="Use PCA for similarity matrix computation (default: True)")
    parser.add_argument('-pca_dim', "--pca_dimension", type=int, default=100,
                        help="PCA dimension for similarity computation (default: 100)")
    
    # FedMul创新功能参数：双层次相似度
    parser.add_argument('-dual_sim', "--use_dual_similarity", type=bool, default=True,
                        help="Use dual similarity (feature + head) computation (default: True)")
    parser.add_argument('-feat_sim_weight', "--feature_sim_weight", type=float, default=0.7,
                        help="Weight for feature similarity in dual similarity (default: 0.7)")
    
    # FedMul创新功能参数：渐进式个性化
    parser.add_argument('-prog_personal', "--progressive_personalization", type=bool, default=True,
                        help="Use progressive personalization strategy (default: True)")
    parser.add_argument('-dynamic_alpha', "--dynamic_alpha", type=bool, default=True,
                        help="Use dynamic alpha weight adjustment (default: True)")


    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")