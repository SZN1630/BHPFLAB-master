import time
import copy
import torch
import os
import numpy as np
# from flcore.clients.clientavg import clientAVG
from flcore.clients.clientas import clientAS
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.attacks import BadPFLAttack, NeurotoxinAttack, DBAAttack, ModelReplacementAttack, BadNetsAttack, TriggerUtils, ClientAttackUtils, BaseAttack
from flcore.defense import GradientClippingBaseline, MedianAggregationBaseline, KrumAggregationBaseline


class FedAS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAS)

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

    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = epoch / self.global_rounds
            
            client.set_parameters(self.global_model, progress)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)    
    
    def aggregate_wrt_fisher(self):
        """Aggregate parameters with respect to Fisher Information Matrix and defense mechanisms"""
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
            # Standard FedAS FIM-weighted aggregation
            self.global_model = copy.deepcopy(self.uploaded_models[0])
            for param in self.global_model.parameters():
                param.data.zero_()

            # calculate the aggregrate weight with respect to the FIM value of model
            FIM_weight_list = []
            for id in self.uploaded_ids:
                FIM_weight_list.append(self.clients[id].fim_trace_history[-1])
            # normalization to obtain weight
            FIM_weight_list = [FIM_value/sum(FIM_weight_list) for FIM_value in FIM_weight_list]

            for w, client_model in zip(FIM_weight_list, self.uploaded_models):
                self.add_parameters(w, client_model)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            selected_ids = [client.id for client in self.selected_clients]


            # self.send_models()

            # evaluate personalized models, ie FedAvg-C
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 记录正常训练数据（无攻击时）
                if not self.attack_method or self.attack_method == 'none':
                    self.base_attack.data_recorder.record_normal_training_data(i, self)

            # self.send_models()
            self.send_selected_models(selected_ids, i)

            # print(f'send selected models done')

            # for client in self.selected_clients:
            #     client.train()
        

            for client in self.alled_clients:
                # print("===============")
                client.train(client.id in selected_ids)
            # assert 1==0


            #self.print_fim_histories()



            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)


            self.aggregate_wrt_fisher()

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

        print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')

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
            self.set_new_clients(clientAS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

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
            selected_ids = [client.id for client in self.selected_clients]

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
                
            elif self.attack_method == 'badnets':
                optimized_trigger_list, external_pattern = self.badnets_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    external_pattern=external_pattern
                )
                
            elif self.attack_method == 'model_replacement':
                optimized_trigger_list, external_pattern = self.model_replacement_attack.execute_attack_training(
                    server=self,
                    round_num=i,
                    attack_start=attack_start,
                    external_pattern=external_pattern
                )

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
                    elif self.attack_method == 'badnets':
                        global_asr = self.evaluate_asr(
                            trigger=optimized_trigger_list,
                            pattern=external_pattern
                        )
                    elif self.attack_method == 'model_replacement':
                        global_asr = self.evaluate_asr(
                            trigger=original_trigger_list,
                            pattern=external_pattern
                        )
                    else:
                        global_asr = self.evaluate_asr(
                            trigger=optimized_trigger_list, 
                            pattern=external_pattern
                        )
                    print(f"Global ASR at round {i}: {global_asr:.4f}")

            # FedAS 特有的训练流程
            self.send_selected_models(selected_ids, i)

            for client in self.alled_clients:
                client.train(client.id in selected_ids)

            #self.print_fim_histories()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_wrt_fisher()

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

        # 保存攻击实验数据
        if self.attack_method and self.attack_method != 'none':
            print(f"保存{self.attack_method.upper()}攻击实验数据...")
            self.base_attack.save_experiment_data()

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAS)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def print_fim_histories(self):
        avg_fim_histories = []

        # Print FIM trace history for each client
        # for client in self.selected_clients:
        for client in self.alled_clients:
            formatted_history = [f"{value:.1f}" for value in client.fim_trace_history]
            print(f"Client{client.id} : {formatted_history}")
            avg_fim_histories.append(client.fim_trace_history)

        # Calculate and print average FIM trace history across clients
        avg_fim_histories = np.mean(avg_fim_histories, axis=0)
        formatted_avg = [f"{value:.1f}" for value in avg_fim_histories]
        print(f"Avg Sum_T_FIM : {formatted_avg}")

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