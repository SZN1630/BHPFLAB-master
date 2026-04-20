import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from torch.autograd import grad
# 导入攻击相关模块
from flcore.attacks import BaseAttack, ClientAttackUtils, TriggerUtils, BadPFLAttack, NeurotoxinAttack, DBAAttack




class clientAS(Client):


    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.fim_trace_history = []
        
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
            else:
                self.attack_method = None
        else:
            self.attack_method = None

    def train(self, is_selected):
        if is_selected:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
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


            # set model to eval mode
            self.model.eval()
            # print(f'client{self.id}, start cal fim.')
            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(self.load_train_data()):
                # Forward pass
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
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

            # Evaluate on the client's test dataset
            # test_acc = self.evaluate()
            # print(f"Client {self.id}, Test Accuracy: {test_acc:.1f}, FIM-T value: {fim_trace_sum.item():.1f}")
            # print(f"Selected: {is_selected}, FIM-T value change: {(self.fim_trace_history[-1] - (self.fim_trace_history[-2] if len(self.fim_trace_history) > 1 else 0)):.1f}")

        else:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
            self.model.eval()
            # Compute FIM and its trace after training
            fim_trace_sum = 0
            for i, (x, y) in enumerate(trainloader):
                # Forward pass
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
                # Negative log likelihood as our loss
                nll = -torch.nn.functional.log_softmax(outputs, dim=1)[range(len(y)), y].mean()

                # Compute gradient of the negative log likelihood w.r.t. model parameters
                grads = grad(nll, self.model.parameters())

                # Compute and accumulate the trace of the Fisher Information Matrix
                for g in grads:
                    fim_trace_sum += torch.sum(g ** 2).detach()

            # add the fisher log
            self.fim_trace_history.append(fim_trace_sum.item())

            # Evaluate on the client's test dataset
            # test_acc = self.evaluate()
            # print(f"Client {self.id}, Test Accuracy: {test_acc:.1f}, FIM-T value: {fim_trace_sum.item():.1f}")
            # print(f"FIM-T value change: {(self.fim_trace_history[-1] - (self.fim_trace_history[-2] if len(self.fim_trace_history) > 1 else 0)):.1f}")

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

    def train_malicious_devide(self, is_selected, poison_ratio, poison_label, trigger, pattern, weight_matrix, clip_rate,
        grad_scale_eps: float = 1e-8  # 防止分母为0
    ):
        """
        FedPz 恶意客户端训练（divide on boundary params）：
        1. 正常训练：使用干净样本和中毒样本进行正常训练
        2. 额外微调：对后门边界参数进行额外的微调
        """
        if not is_selected:
            return 

        # 构建后门边界参数集合
        if isinstance(weight_matrix, dict):
            name2score = {str(k): float(v) for k, v in weight_matrix.items()}
        else:
            name2score = {str(k): float(v) for k, v in weight_matrix}
        
        selected_names = set(name2score.keys())
        
        # 如果没有边界参数，回退为常规恶意训练
        if len(selected_names) == 0:
            self.train_malicious(is_selected, poison_ratio, poison_label, trigger, pattern, 1, clip_rate)
            return

        # 计算梯度缩放因子
        scores = torch.tensor([name2score.get(n, 0.0) for n in selected_names], dtype=torch.float32)
        s_min, s_max = scores.min().item(), scores.max().item()
        if s_max - s_min < 1e-12:
            name2scale = {n: 2.0 for n in selected_names}
        else:
            name2scale = {
                n: 1.0 + 3.0 * ((name2score.get(n, 0.0) - s_min) / (s_max - s_min + grad_scale_eps))
                for n in selected_names
            }

        # 使用带触发器的数据加载器
        trainloader = self.load_poison_data(
            poison_ratio=poison_ratio,
            poison_label=poison_label,
            noise_trigger=trigger,
            pattern=pattern,
            batch_size=self.batch_size,
        )

        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs // 2 if getattr(self, 'train_slow', False) else self.local_epochs

        # 第一阶段：正常训练（干净样本 + 中毒样本）
        for _ in range(max_local_epochs):
            for x, y in trainloader:
                # 处理多通道/列表输入
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if getattr(self, 'train_slow', False):
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 正常的前向传播和反向传播
                out = self.model(x)
                loss = self.loss(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 第二阶段：对后门边界参数进行额外微调
        # 重新加载数据，只使用中毒样本进行微调
        poison_trainloader = self.load_poison_data(
            poison_ratio=1.0,  # 100%中毒样本
            poison_label=poison_label,
            noise_trigger=trigger,
            pattern=pattern,
            batch_size=self.batch_size,
        )

        # 微调轮数（比正常训练少）
        fine_tune_epochs = max(1, max_local_epochs // 4)
        
        for _ in range(fine_tune_epochs):
            for x, y in poison_trainloader:
                # 处理多通道/列表输入
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                # 前向传播
                out = self.model(x)
                loss = self.loss(out, y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 只在边界参数上进行微调，并按重要度缩放梯度
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        if name in selected_names:
                            p.grad.mul_(float(name2scale[name]))  # 重要度缩放
                        else:
                            p.grad.zero_()                         # 屏蔽非边界参数的梯度

                self.optimizer.step()

        # 学习率调度
        if getattr(self, 'learning_rate_decay', False):
            self.learning_rate_scheduler.step()

        # 统计耗时
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
    
    # def set_parameters(self, model, progress):
        # # Substitute the parameters of the base, enabling personalization
        # for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
        #     old_param.data = new_param.data.clone()

    def set_parameters(self, model, progress):

        # Get class-specific prototypes from the local model
        local_prototypes = [[] for _ in range(self.num_classes)]
        batch_size = 16  # or any other suitable value
        trainloader = self.load_train_data(batch_size=batch_size)

        # print(f'client{id}')
        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 修复标签越界问题
            y_batch = torch.clamp(y_batch, 0, self.num_classes - 1).long()

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)

            # Scatter the prototypes based on their labels
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        # print(f'client{self.id}')
        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        # Align global model's prototype with the local prototype
        alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=0.01)  # Adjust learning rate and optimizer as needed
        alignment_loss_fn = torch.nn.MSELoss()

        # print(f'client{self.id}')
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 修复标签越界问题
                y_batch = torch.clamp(y_batch, 0, self.num_classes - 1).long()
                
                global_proto_batch = model.base(x_batch)
                loss = 0
                for label in y_batch.unique():
                    label_clamped = torch.clamp(label, 0, self.num_classes - 1).long()
                    if mean_prototypes[label_clamped.item()] is not None:
                        loss += alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label_clamped.item()])
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

        # end

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

    def get_attack_utils(self):
        """获取攻击工具实例"""
        return {
            'base_attack': self.base_attack,
            'attack_utils': self.attack_utils,
            'trigger_utils': self.trigger_utils,
            'attack_method': self.attack_method
        }


