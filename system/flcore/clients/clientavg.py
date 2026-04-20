import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.attacks import BaseAttack, ClientAttackUtils, TriggerUtils, BadPFLAttack, NeurotoxinAttack, DBAAttack


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 初始化攻击工具
        self.attack_utils = ClientAttackUtils(self.device, self.dataset)
        self.trigger_utils = TriggerUtils()
        self.base_attack = BaseAttack(args, self.device)
        
        # 根据攻击类型初始化具体的攻击方法
        if hasattr(args, 'attack') and args.attack:
            if args.attack.lower() == 'badpfl':
                self.attack_method = BadPFLAttack(args, self.device)
                # 将 Bad-PFL 攻击实例附加到模型上，以便在 poisontest 中访问
                self.model.badpfl_attack = self.attack_method
            elif args.attack.lower() == 'neurotoxin':
                self.attack_method = NeurotoxinAttack(args, self.device)
            elif args.attack.lower() == 'dba':
                self.attack_method = DBAAttack(args, self.device)
            elif args.attack.lower() == 'model_replacement':
                self.attack_method = args.attack.lower()  # 存储攻击类型字符串
            elif args.attack.lower() == 'badnets':
                self.attack_method = args.attack.lower()  # 存储攻击类型字符串
            else:
                self.attack_method = None
        else:
            self.attack_method = None

    def train(self,is_selected):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        if is_selected:
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # 修复标签越界问题
                    use_model_split = hasattr(self.model, 'base') and hasattr(self.model, 'head')
                    if use_model_split:
                        num_classes = self.model.head.out_features
                    else:
                        num_classes = list(self.model.parameters())[-1].shape[0]
                    y = torch.clamp(y, 0, num_classes - 1).long()
                    
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

    def train_malicious(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()} 
        if is_selected:
            trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                              noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
            self.model.train()   
            
            start_time = time.time()
            max_local_epochs = self.local_epochs
            # 慢客户端的迭代训练减少
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)   # 多通道
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    
                    # 修复标签越界问题
                    use_model_split = hasattr(self.model, 'base') and hasattr(self.model, 'head')
                    if use_model_split:
                        num_classes = self.model.head.out_features
                    else:
                        num_classes = list(self.model.parameters())[-1].shape[0]
                    y = torch.clamp(y, 0, num_classes - 1).long()
                    
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    
                    # 检查是否为 Bad-PFL 攻击（trigger 为 None 或 pattern 为 None）
                    if trigger is None or pattern is None:
                        # Bad-PFL 攻击：使用 Bad-PFL 的 poison_batch 方法
                        if hasattr(self, 'attack_method') and self.attack_method and hasattr(self.attack_method, 'poison_batch'):
                            try:
                                # 需要修改的部分
                                # 将poison_ratio转换为比例（如4/16=0.25）
                                poison_ratio_float = poison_ratio / len(x)
                                poisoned_x, poisoned_y = self.attack_method.poison_batch(
                                    data=x, 
                                    labels=y, 
                                    client_model=self.model, 
                                    poison_ratio=poison_ratio_float
                                )
                                x, y = poisoned_x, poisoned_y
                                # 修复投毒后的标签越界问题
                                y = torch.clamp(y, 0, num_classes - 1).long()
                            except Exception as e:
                                print(f"Bad-PFL poison_batch error: {e}")
                                # 如果出错，使用原始数据
                    
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # 此处进行梯度裁剪 限制更新幅度  
            if oneshot == 1 and clip_rate > 0:
                # 计算并应用缩放后的参数更新
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)
            
            # 更新 last_local_model 为当前状态
            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

    def train_malicious_neurotoxin(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate, grad_mask):
        """
        Neurotoxin恶意训练方法
        使用梯度掩码来控制投毒范围，只对更新幅度较小的参数进行投毒
        核心思想：将恶意梯度投影到良性梯度不常触及的参数空间
        
        训练策略：
        1. 同时进行主任务训练和投毒训练
        2. 按照poison_ratio设置投毒比例（如poison_ratio=4表示batch中4个样本投毒）
        3. 使用梯度掩码控制投毒范围
        """
        if not is_selected:
            return
        
        # 记录原始参数（用于梯度裁剪）
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()} 
        
        # 加载混合训练数据：包含主任务数据和投毒数据
        # poison_ratio=4 表示batch_size=16时，4个样本投毒，12个样本正常
        trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                          noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
        
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # 修复标签越界问题
                use_model_split = hasattr(self.model, 'base') and hasattr(self.model, 'head')
                if use_model_split:
                    num_classes = self.model.head.out_features
                else:
                    num_classes = list(self.model.parameters())[-1].shape[0]
                y = torch.clamp(y, 0, num_classes - 1).long()
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                output = self.model(x)
                loss = self.loss(output, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 应用梯度掩码 - Neurotoxin核心：只对更新幅度较小的参数进行投毒
                # 这里的关键是：梯度掩码只影响投毒相关的参数更新
                with torch.no_grad():
                    param_idx = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and param_idx < len(grad_mask):
                            # 使用梯度掩码来控制梯度更新
                            # 掩码为1的位置允许更新（良性梯度不常触及的参数）
                            # 掩码为0的位置阻止更新（良性梯度常触及的参数）
                            mask = grad_mask[param_idx]
                            if mask.shape == param.grad.shape:
                                param.grad *= mask
                            param_idx += 1
                
                # 参数更新
                self.optimizer.step()

        # 梯度裁剪：限制参数更新幅度
        if oneshot == 1 and clip_rate > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    original_param = last_local_model.get(name, param.data.clone())
                    scaled_update = (param.data - original_param) * clip_rate
                    param.data.copy_(original_param + scaled_update)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_malicious_dba(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        """
        DBA恶意训练方法
        使用分布式后门攻击，每个客户端优化其分配的触发器部分
        
        Args:
            is_selected: 是否被选中
            poison_ratio: 投毒比例
            poison_label: 投毒标签
            trigger: 触发器（全局触发器）
            pattern: 触发器模式
            oneshot: 是否一次性攻击
            clip_rate: 梯度裁剪率
        """
        if not is_selected:
            return
        
        # 记录原始参数（用于梯度裁剪）
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # 加载混合训练数据：包含主任务数据和投毒数据
        trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                          noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
        
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # 修复标签越界问题
                use_model_split = hasattr(self.model, 'base') and hasattr(self.model, 'head')
                if use_model_split:
                    num_classes = self.model.head.out_features
                else:
                    num_classes = list(self.model.parameters())[-1].shape[0]
                y = torch.clamp(y, 0, num_classes - 1).long()
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                output = self.model(x)
                loss = self.loss(output, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 参数更新
                self.optimizer.step()

        # 梯度裁剪：限制参数更新幅度
        if oneshot == 1 and clip_rate > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    original_param = last_local_model.get(name, param.data.clone())
                    scaled_update = (param.data - original_param) * clip_rate
                    param.data.copy_(original_param + scaled_update)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_attack_success_rate(self, trigger, pattern, poison_label):
        """评估攻击成功率 (ASR)"""
        if hasattr(self.base_attack, 'evaluate_asr'):
            return self.base_attack.evaluate_asr([self], trigger, pattern, poison_label)
        else:
            # 如果没有evaluate_asr方法，使用默认的poisontest方法
            asr_correct, asr_samples = self.poisontest(
                poison_label=poison_label,
                trigger=trigger,
                pattern=pattern
            )
            return asr_correct / asr_samples if asr_samples > 0 else 0.0

    def get_attack_utils(self):
        """获取攻击工具实例"""
        return {
            'base_attack': self.base_attack,
            'attack_utils': self.attack_utils,
            'trigger_utils': self.trigger_utils,
            'attack_method': self.attack_method
        }


    def load_poison_data(self, poison_ratio, poison_label, noise_trigger, pattern, batch_size):
        """加载混合训练数据：包含主任务数据和投毒数据"""
        trainloader = self.load_train_data()
        return self.attack_utils.load_poison_data(
            trainloader=trainloader,
            poison_ratio=poison_ratio,
            poison_label=poison_label,
            noise_trigger=noise_trigger,
            pattern=pattern,
            batch_size=batch_size,
            device=self.device,
            dataset=self.dataset
        )

    def poisontest(self, trigger=None, poison_label=None, pattern=None):
        """测试攻击成功率 (ASR)"""
        testloader = self.load_test_data()
        self.model.eval()
        
        # 检查攻击类型并设置相应的触发器标识
        if hasattr(self, 'attack_method') and self.attack_method == 'model_replacement':
            # 对于模型替换攻击，使用特殊的触发器标识
            trigger = "MODEL_REPLACEMENT_TRIGGER"
            pattern = "MODEL_REPLACEMENT_PATTERN"
        elif hasattr(self, 'attack_method') and self.attack_method == 'badnets':
            # 对于BadNets攻击，使用特殊的触发器标识
            trigger = "BADNETS_TRIGGER"
            pattern = "BADNETS_PATTERN"
        
        # 使用ClientAttackUtils的poisontest方法
        return self.attack_utils.poisontest(
            model=self.model,
            testloader=testloader,
            poison_label=poison_label,
            trigger=trigger,
            pattern=pattern,
            device=self.device,
            dataset=self.dataset
        )

    def evaluate(self):
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                # 修复标签越界问题
                use_model_split = hasattr(self.model, 'base') and hasattr(self.model, 'head')
                if use_model_split:
                    num_classes = self.model.head.out_features
                else:
                    num_classes = list(self.model.parameters())[-1].shape[0]
                y = torch.clamp(y, 0, num_classes - 1).long()
                
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
