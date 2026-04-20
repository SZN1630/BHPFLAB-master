import time
import copy
import torch
import os
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.attacks import BadPFLAttack, NeurotoxinAttack, DBAAttack, ModelReplacementAttack, BadNetsAttack, TriggerUtils, ClientAttackUtils, BaseAttack
from flcore.defense import RobustAggregation, BaseDefense, GradientClippingBaseline, MedianAggregationBaseline, KrumAggregationBaseline


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        
        '''============initialize defense module============'''
        # Initialize gradient clipping baseline
        self.gradient_clipping = GradientClippingBaseline(max_norm=getattr(args, 'clip_norm', 1.0))
        # Initialize median aggregation baseline
        self.median_aggregation = MedianAggregationBaseline()
        # Initialize Krum aggregation baseline
        self.krum_aggregation = KrumAggregationBaseline(
            num_byzantine=getattr(args, 'krum_byzantine', 0),
            multi_krum=getattr(args, 'multi_krum', False)
        )
        # Defense mode: 'clip', 'median', 'krum', 'clip+median', 'clip+krum', or None
        self.defense_mode = getattr(args, 'defense_mode', None)
        print(f"Defense mode: {self.defense_mode}")
        
        '''============initialize attack module============'''
        self.base_attack = BaseAttack(args, self.device)
        
        # 根据攻击类型条件初始化具体攻击模块
        self.attack_method = getattr(args, 'attack', None)
        self.poisonlabel = getattr(args, 'poison_label', 1)
        self.poisonratio = getattr(args, 'poison_rate', 4)
        self.malicious_clients = []
        self.client_features = {}
        self.sensitivity_matrix_cache = {}
        
        if self.attack_method:
            self.malicious_clients = self.base_attack.select_malicious_clients(self.clients, num_malicious=2)
            
            # initialize attack module
            if self.attack_method == 'badpfl':
                self.badpfl_attack = BadPFLAttack(args, self.device)
            elif self.attack_method == 'neurotoxin':
                self.neurotoxin_attack = NeurotoxinAttack(args, self.device)
            elif self.attack_method == 'dba':
                self.dba_attack = DBAAttack(args, self.device)
            elif self.attack_method == 'model_replacement':
                self.model_replacement_attack = ModelReplacementAttack(args, self.device)
            elif self.attack_method == 'badnets':
                self.badnets_attack = BadNetsAttack(args, self.device)

    def aggregate_parameters(self):
        """Aggregate parameters with defense mechanisms"""
        assert (len(self.uploaded_models) > 0)
        
        # Apply gradient clipping if enabled
        if self.defense_mode and 'clip' in self.defense_mode:
            print("Applying gradient clipping...")
            self.uploaded_models = self.gradient_clipping.clip_uploaded_models(
                self.global_model, 
                self.uploaded_models
            )
        
        # Apply aggregation based on defense mode
        if self.defense_mode and 'krum' in self.defense_mode:
            print("Applying Krum aggregation...")
            self.global_model = self.krum_aggregation.aggregate_models(self.uploaded_models)
        elif self.defense_mode and 'median' in self.defense_mode:
            print("Applying median aggregation...")
            self.global_model = self.median_aggregation.aggregate_models(self.uploaded_models)
        else:
            # Standard FedAvg aggregation
            self.global_model = copy.deepcopy(self.uploaded_models[0])
            for param in self.global_model.parameters():
                param.data.zero_()
            
            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 记录正常训练数据（无攻击时）
                if not self.attack_method or self.attack_method == 'none':
                    self.base_attack.data_recorder.record_normal_training_data(i, self)

            for client in self.selected_clients:
                client.train(is_selected=True)

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # 保存实验数据（无论是否有攻击）
        if self.attack_method and self.attack_method != 'none':
            print(f"保存{self.attack_method.upper()}攻击实验数据...")
            self.base_attack.save_experiment_data()
        else:
            # 保存正常训练的实验数据
            print("保存正常训练实验数据...")
            self.base_attack.save_experiment_data()

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def evaluate_asr(self, trigger, pattern):
        return self.base_attack.evaluate_asr(
            clients=self.clients,
            trigger=trigger,
            pattern=pattern,
            poison_label=self.poisonlabel
        )
    
    def avg_generalization_metrics(self):
        """计算平均泛化指标"""
        if not self.rs_test_acc:
            return 0.0
        return sum(self.rs_test_acc) / len(self.rs_test_acc)
    
    def all_clients(self):
        return self.clients
    
    def train_with_attack(self, pattern, trigger):
        original_trigger_list = trigger
        optimized_trigger_list = copy.deepcopy(original_trigger_list)
        external_pattern = pattern 
        attack_start = 3
        oneshot = 1         
        clip_rate = 0.2     

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()
            self.send_models()

            # 根据攻击类型执行对应的攻击训练
            if self.attack_method == 'badpfl':
                badpfl_trigger, badpfl_pattern = self.badpfl_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    external_pattern=external_pattern
                )
                # 设置全局Bad-PFL攻击实例，供客户端poisontest使用
                import sys
                from flcore.attacks import client_attack_utils
                client_attack_utils._global_badpfl_attack = self.badpfl_attack
            elif self.attack_method == 'neurotoxin':
                optimized_trigger_list = self.neurotoxin_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    oneshot=oneshot,
                    clip_rate=clip_rate,
                    original_trigger_list=original_trigger_list,
                    external_pattern=external_pattern,
                    optimized_trigger_list=optimized_trigger_list
                )
                
            elif self.attack_method == 'dba':
                optimized_trigger_list = self.dba_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    oneshot=oneshot,
                    clip_rate=clip_rate,
                    original_trigger_list=original_trigger_list,
                    external_pattern=external_pattern,
                    optimized_trigger_list=optimized_trigger_list
                )
            elif self.attack_method == 'model_replacement':
                optimized_trigger_list, external_pattern = self.model_replacement_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    external_pattern=external_pattern
                )
                # 设置全局模型替换攻击实例，供客户端poisontest使用
                import sys
                from flcore.attacks import client_attack_utils
                client_attack_utils._global_model_replacement_attack = self.model_replacement_attack
            elif self.attack_method == 'badnets':
                optimized_trigger_list, external_pattern = self.badnets_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    external_pattern=external_pattern
                )
                # 设置全局BadNets攻击实例，供客户端poisontest使用
                import sys
                from flcore.attacks import client_attack_utils
                client_attack_utils._global_badnets_attack = self.badnets_attack

            # 评估和ASR计算
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                if i >= attack_start:
                    if self.attack_method == 'badpfl':
                        global_asr = self.evaluate_asr(
                            trigger=badpfl_trigger,
                            pattern=badpfl_pattern
                        )
                    elif self.attack_method == 'model_replacement':
                        # Evaluate ASR using all clients' test data
                        total_samples = 0
                        successful_attacks = 0
                        
                        for client in self.clients:
                            test_loader = client.load_test_data(batch_size=32)
                            client_successful, client_total = self.model_replacement_attack._evaluate_asr_on_loader(
                                self.global_model, test_loader, self.device
                            )
                            successful_attacks += client_successful
                            total_samples += client_total
                        
                        global_asr = successful_attacks / total_samples if total_samples > 0 else 0.0
                    elif self.attack_method == 'badnets':
                        # Evaluate BadNets ASR using all clients' test data
                        total_samples = 0
                        successful_attacks = 0
                        
                        for client in self.clients:
                            test_loader = client.load_test_data(batch_size=32)
                            client_asr = self.badnets_attack._evaluate_asr_on_loader(
                                self.global_model, test_loader, self.device
                            )
                            successful_attacks += client_asr * len(test_loader.dataset)
                            total_samples += len(test_loader.dataset)
                        
                        global_asr = successful_attacks / total_samples if total_samples > 0 else 0.0
                    else:
                        global_asr = self.evaluate_asr(
                            trigger=optimized_trigger_list, 
                            pattern=external_pattern
                        )
                    print(f"Global ASR at round {i}: {global_asr:.4f}")

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            round_time = self.Budget[-1]
            print('-'*25, 'time cost', '-'*25, round_time)


            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')

        # 保存攻击实验数据 - 统一使用base_attack保存
        if self.attack_method and self.attack_method != 'none':
            print(f"保存{self.attack_method.upper()}攻击实验数据...")
            self.base_attack.save_experiment_data()

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()