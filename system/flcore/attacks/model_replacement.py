"""
Model Replacement Backdoor Attack Implementation

Core idea: Replace the entire model of selected malicious clients with a backdoored model
that maintains good performance on clean data but has backdoor behavior on triggered samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import time
import math
from .base_attack import BaseAttack


class TriggerDataset(Dataset):
    """Dataset that adds trigger patterns to samples and assigns target labels"""
    
    def __init__(self, base_dataset, trigger_pattern, target_label, trigger_size=4):
        self.base_dataset = base_dataset
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.trigger_size = trigger_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        
        # Add trigger pattern based on data shape
        x_triggered = x.clone()
        
        # Check if this is HAR data (9, 1, 128) or MHEALTH data (21, 1, 128)
        if len(x.shape) == 3 and x.shape[1] == 1 and x.shape[2] == 128:
            # HAR or MHEALTH data: directly add trigger pattern
            if x.shape[0] == 9 or x.shape[0] == 21:
                x_triggered = x_triggered + self.trigger_pattern
        elif len(x.shape) == 3:  # (C, H, W) - Image data
            if x.shape[0] == 3:  # RGB
                x_triggered[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
            else:  # 灰度
                x_triggered[:, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0)
        else:  # (H, W) for grayscale
            if len(self.trigger_pattern.shape) > 2:
                x_triggered[-self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0)
            else:
                x_triggered[-self.trigger_size:, -self.trigger_size:] = self.trigger_pattern
        
        return x_triggered, self.target_label


class ModelReplacementAttack(BaseAttack):
    """
    Model Replacement Attack Implementation
    
    Core idea: Train a backdoored model that performs well on clean data but 
    has backdoor behavior on triggered samples, then replace malicious clients' models
    """
    
    def __init__(self, args, device):
        super().__init__(args, device)
        self.args = args
        
        # Attack parameters
        self.target_label = getattr(args, 'model_replace_target_label', 0)
        self.trigger_size = getattr(args, 'model_replace_trigger_size', 4)
        self.backdoor_epochs = getattr(args, 'model_replace_backdoor_epochs', 15)  # 适中的轮数
        self.clean_ratio = getattr(args, 'model_replace_clean_ratio', 0.7)  # 适中的清洁比例
        self.lr = getattr(args, 'model_replace_lr', 0.01)  # 稳定的学习率
        
        # Create trigger pattern based on dataset type
        if hasattr(args, 'dataset') and 'mnist' in args.dataset.lower():
            # 为MNIST创建棋盘格触发器
            trigger = torch.zeros(self.trigger_size, self.trigger_size)
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        trigger[i, j] = 1.0
            self.trigger_pattern = trigger
        elif hasattr(args, 'dataset') and args.dataset.lower() in ['har', 'uci_har']:
            # HAR: 9x1x128 时间序列数据 - 使用main.py中的触发器设计
            C, H, W = 9, 1, 128
            base_trigger = torch.zeros((C, H, W), dtype=torch.float32)
            
            # 时间窗参数 - 与main.py保持一致
            win_len, start, amp = 21, 64, 0.15
            t = torch.arange(win_len)
            patch = amp * torch.hann_window(win_len, periodic=True) * torch.sin(2*math.pi*t/win_len)
            
            # 只在少数通道上加扰动
            for c in [0, 3]:
                base_trigger[c, 0, start:start+win_len] = patch
                
            self.trigger_pattern = base_trigger
        elif hasattr(args, 'dataset') and args.dataset.lower() in ['mhealth']:
            # MHEALTH: 21x1x128 时间序列数据
            C, H, W = 21, 1, 128
            base_trigger = torch.zeros((C, H, W), dtype=torch.float32)
            
            # 时间窗参数 - 与BadNets保持一致
            win_len, start, amp = 20, 70, 0.3
            t = torch.arange(win_len)
            patch = amp * torch.hann_window(win_len, periodic=True) * torch.sin(2*math.pi*t/win_len)
            
            # 在多个通道上加扰动
            for c in [0, 1, 2, 3, 4, 5]:
                base_trigger[c, 0, start:start+win_len] = patch
                
            self.trigger_pattern = base_trigger
        else:
            # 为彩色图像创建彩色棋盘格触发器
            trigger = torch.zeros(3, self.trigger_size, self.trigger_size)
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        trigger[0, i, j] = 1.0  # 红色
                        trigger[1, i, j] = 0.0  # 绿色
                        trigger[2, i, j] = 1.0  # 蓝色（紫红色）
                    else:
                        trigger[0, i, j] = 0.0
                        trigger[1, i, j] = 1.0  # 绿色
                        trigger[2, i, j] = 0.0
            self.trigger_pattern = trigger
        
        print(f"Model Replacement Attack initialized:")
        print(f"  - Target label: {self.target_label}")
        print(f"  - Trigger size: {self.trigger_size}x{self.trigger_size}")
        print(f"  - Backdoor training epochs: {self.backdoor_epochs}")
        print(f"  - Clean data ratio: {self.clean_ratio}")
    
    def create_backdoored_model(self, global_model, client_train_data, device):
        """
        Create a backdoored model by fine-tuning the global model on clean + poisoned data
        
        Args:
            global_model: The current global model
            client_train_data: Client's training dataset
            device: Training device
            
        Returns:
            backdoored_model: Model with backdoor implanted
        """
        # Create a copy of the global model for backdoor training
        backdoored_model = copy.deepcopy(global_model)
        backdoored_model.to(device)
        
        # Create optimizer with weight decay for stability
        optimizer = torch.optim.SGD(backdoored_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        
        # Prepare mixed dataset (clean + poisoned)
        clean_size = int(len(client_train_data.dataset) * self.clean_ratio)
        poison_size = len(client_train_data.dataset) - clean_size
        
        # Create clean and poisoned datasets
        indices = list(range(len(client_train_data.dataset)))
        clean_indices = indices[:clean_size]
        poison_indices = indices[clean_size:clean_size + poison_size]
        
        clean_subset = torch.utils.data.Subset(client_train_data.dataset, clean_indices)
        poison_subset = torch.utils.data.Subset(client_train_data.dataset, poison_indices)
        
        # Create trigger dataset for poisoned samples
        trigger_dataset = TriggerDataset(
            poison_subset, 
            self.trigger_pattern, 
            self.target_label, 
            self.trigger_size
        )
        
        # Create combined dataset
        clean_loader = DataLoader(clean_subset, batch_size=32, shuffle=True)
        poison_loader = DataLoader(trigger_dataset, batch_size=32, shuffle=True)
        
        #print(f"Training backdoored model with {clean_size} clean + {poison_size} poisoned samples")
        
        backdoored_model.train()
        for epoch in range(self.backdoor_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # 先训练中毒数据（增强后门特征）
            for batch_idx, (data, target) in enumerate(poison_loader):
                data, target = data.to(device), target.to(device)
                
                # 修复标签越界问题
                use_model_split = hasattr(backdoored_model, 'base') and hasattr(backdoored_model, 'head')
                if use_model_split:
                    num_classes = backdoored_model.head.out_features
                else:
                    num_classes = list(backdoored_model.parameters())[-1].shape[0]
                target = torch.clamp(target, 0, num_classes - 1).long()
                
                optimizer.zero_grad()
                output = backdoored_model(data)
                # 检查输出是否包含NaN
                if torch.isnan(output).any():
                    print(f"Warning: NaN detected in model output, skipping batch")
                    continue
                    
                # 增加后门损失权重
                loss = F.cross_entropy(output, target) * 1.5  # 降低权重避免不稳定
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(backdoored_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # 再训练清洁数据（保持正常性能）
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(device), target.to(device)
                
                # 修复标签越界问题
                use_model_split = hasattr(backdoored_model, 'base') and hasattr(backdoored_model, 'head')
                if use_model_split:
                    num_classes = backdoored_model.head.out_features
                else:
                    num_classes = list(backdoored_model.parameters())[-1].shape[0]
                target = torch.clamp(target, 0, num_classes - 1).long()
                
                optimizer.zero_grad()
                output = backdoored_model(data)
                # 检查输出是否包含NaN
                if torch.isnan(output).any():
                    print(f"Warning: NaN detected in model output, skipping batch")
                    continue
                    
                loss = F.cross_entropy(output, target)
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected, skipping batch")
                    continue
                    
                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(backdoored_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if epoch % 5 == 0:
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                print(f"  Backdoor training epoch {epoch}: avg_loss = {avg_loss:.4f}")
                
                # 检查模型参数是否包含NaN
                has_nan = False
                for name, param in backdoored_model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"Warning: NaN detected in parameter {name}")
                        has_nan = True
                
                if has_nan:
                    print("Error: Model parameters contain NaN, falling back to global model")
                    return copy.deepcopy(global_model)
        
        return backdoored_model
    
    def evaluate_attack_success(self, model, test_loader, device):
        """Evaluate backdoor attack success rate on test data"""
        if test_loader is None:
            return 0.0
        successful_attacks, total_samples = self._evaluate_asr_on_loader(model, test_loader, device)
        asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        return asr
    
    def _evaluate_asr_on_loader(self, model, test_loader, device):
        """Helper method to evaluate ASR on a specific data loader"""
        model.eval()
        total_samples = 0
        successful_attacks = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # Add trigger to test samples based on data shape
                x_triggered = x.clone()
                
                # Check if this is HAR data (batch_size, 9, 1, 128) or MHEALTH data (batch_size, 21, 1, 128)
                if len(x.shape) == 4 and x.shape[2] == 1 and x.shape[3] == 128:
                    # HAR or MHEALTH data: directly add trigger pattern
                    if x.shape[1] == 9 or x.shape[1] == 21:
                        x_triggered = x_triggered + self.trigger_pattern.to(device)
                elif len(x.shape) == 4:  # Batch dimension - Image data
                    if x.shape[1] == 3:  # RGB
                        x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.to(device)
                    else:  # Grayscale
                        if len(self.trigger_pattern.shape) > 2:
                            x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.mean(0).to(device)
                        else:
                            x_triggered[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_pattern.to(device)
                
                # Test if model predicts target label for triggered samples
                output = model(x_triggered)
                pred = output.argmax(dim=1)
                successful_attacks += (pred == self.target_label).sum().item()
                total_samples += x.size(0)
        
        return successful_attacks, total_samples
    
    def is_malicious_client(self, client_id):
        """Check if a client should perform model replacement attack"""
        malicious_ids = getattr(self.args, 'malicious_ids', [])
        return client_id in malicious_ids
    
    def execute_attack_training(self, server, round_num, attack_start, 
                              external_pattern, **kwargs):
        """
        Execute model replacement attack training for a specific round
        
        Args:
            server: The federated learning server
            round_num: Current training round
            attack_start: Round to start the attack
            external_pattern: Pattern for trigger (used for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            tuple: (trigger_pattern, pattern) for ASR evaluation
        """
        if round_num < attack_start:
            return self.trigger_pattern, external_pattern
        
        print(f"\n=== Model Replacement Attack - Round {round_num} ===")
        
        # Collect client training data for this round
        client_data = []
        malicious_clients = server.malicious_clients
        
        for client in server.selected_clients:
            start_time = time.time()
            
            if client in malicious_clients:
                print(f"Executing model replacement for malicious client {client.id}")
                
                # Create backdoored model
                backdoored_model = self.create_backdoored_model(
                    server.global_model, 
                    client.load_train_data(), 
                    self.device
                )
                
                # Replace client's model with backdoored model
                # Check if client requires progress parameter (FedAS clients)
                if hasattr(client, 'set_parameters') and client.set_parameters.__code__.co_argcount > 2:
                    # FedAS client requires progress parameter
                    progress = round_num / 100.0  # Simple progress calculation
                    client.set_parameters(backdoored_model, progress)
                else:
                    # Standard client (FedAvg, FedMul)
                    client.set_parameters(backdoored_model)
                print(f"Client {client.id} model replaced with backdoored version")
                
                # Train the backdoored model normally (maintain performance)
                client.train(is_selected=True)
            else:
                # Normal training for benign clients
                client.train(is_selected=True)
            
            # Evaluate client performance
            test_acc, test_num, auc = client.test_metrics()
            client_acc = test_acc / test_num if test_num > 0 else 0.0
            
            # Evaluate ASR
            client_asr = self.evaluate_client_asr(client)
            
            client_time = time.time() - start_time
            
            client_info = {
                'client_id': client.id,
                'client_acc': client_acc, 
                'client_asr': client_asr,
                'client_time': client_time,
                'is_malicious': client in malicious_clients
            }
            client_data.append(client_info)
            
            print(f"Client {client.id}: ACC={client_acc:.4f}, ASR={client_asr:.4f}")
        
        # Record attack data using server's base_attack.data_recorder (consistent with other attacks)
        server.base_attack.data_recorder.record_attack_data_from_collected(
            round_num=round_num,
            server=server,
            client_data=client_data,
            attack_method='model_replacement',
            original_trigger_list=self.trigger_pattern,
            external_pattern=external_pattern
        )
        
        return self.trigger_pattern, external_pattern
    
    def evaluate_client_asr(self, client):
        """Evaluate attack success rate for a specific client"""
        try:
            test_loader = client.load_test_data(batch_size=32)
            return self.evaluate_attack_success(client.model, test_loader, self.device)
        except Exception as e:
            print(f"Error evaluating ASR for client {client.id}: {e}")
            return 0.0