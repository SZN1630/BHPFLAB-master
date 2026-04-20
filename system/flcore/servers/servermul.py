import time
import copy
import torch
import numpy as np
import os
from flcore.clients.clientmul import clientMUL
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.attacks import BadPFLAttack, NeurotoxinAttack, DBAAttack, ModelReplacementAttack, BadNetsAttack, TriggerUtils, ClientAttackUtils, BaseAttack
from flcore.defense import RobustAggregation, BaseDefense
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class FedMul(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # 保存args引用
        self.args = args
        self.cluster_similarity_matrices = {}
        self.join_clients = int(self.num_clients * self.join_ratio)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMUL)
        self.linkage_method = getattr(args, 'linkage_method', 'single')
        self.alpha = getattr(args, 'alpha', 0.01)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # MultiSim 框架参数
        self.num_clusters = getattr(args, 'num_clusters', 3)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.7)
        self.fusion_weight_gradient = getattr(args, 'fusion_weight_gradient', 0.5)
        self.fusion_weight_weight = getattr(args, 'fusion_weight_weight', 0.5)
        self.clustering_method = getattr(args, 'clustering_method', 'hierarchical')
        self.update_cluster_freq = getattr(args, 'update_cluster_freq', 5)
        self.fuse_method = 'attention'
        
        # 优化参数
        self.top_k_neighbors = getattr(args, 'top_k_neighbors', 3)
        self.use_pca_similarity = getattr(args, 'use_pca_similarity', True)
        self.pca_dimension = getattr(args, 'pca_dimension', 100)
        self.use_dual_similarity = getattr(args, 'use_dual_similarity', True)
        self.feature_sim_weight = getattr(args, 'feature_sim_weight', 0.7)
        
        # 模型分割支持
        self.use_model_split = hasattr(args.model, 'base') and hasattr(args.model, 'head')
        
        # 为所有客户端设置参数
        for client in self.clients:
            client.fuse_method = self.fuse_method
            client.alpha = self.alpha
        
        # 注意力机制参数
        self.temperature = getattr(args, 'temperature', 1.0)
        
        # 聚类相关
        self.client_clusters = {}  # {client_id: cluster_id}
        self.cluster_centers = {}  # {cluster_id: center_weights}
        self.similarity_matrix = None
        self.gradient_similarity_matrix = None
        self.weight_similarity_matrix = None
        
        # 个性化学习参数
        self.personalization_rounds = getattr(args, 'personalization_rounds', 5)
        self.cluster_aggregation_weight = getattr(args, 'cluster_aggregation_weight', 0.7)
        
        # 鲁棒性检测
        self.malicious_clients = []
        self.robustness_threshold = getattr(args, 'robustness_threshold', 0.3)
        
        # 攻击模块初始化
        self.base_attack = BaseAttack(args, self.device)
        
        # 根据攻击类型条件初始化具体攻击模块
        self.attack_method = getattr(args, 'attack', None)
        self.poisonlabel = getattr(args, 'poison_label', 1)
        self.poisonratio = getattr(args, 'poison_rate', 4)
        self.malicious_clients = []
        self.client_features = {}
        self.sensitivity_matrix_cache = {}
        
        if self.attack_method and self.attack_method != 'none':
            self.malicious_clients = self.base_attack.select_malicious_clients(self.clients, num_malicious=2)
            print(f"Selected {len(self.malicious_clients)} malicious clients for robustness testing")
            
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
        
        # 防御模块（可选）
        if hasattr(args, 'defense') and args.defense and args.defense != 'none':
            self.defense = BaseDefense(args, self.device)
        
        self.Budget = []
        
        print(f"FedMul initialized with {self.num_clusters} clusters")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Using attention fusion method with temperature: {self.temperature}")
    
    def evaluate_asr(self, trigger, pattern):
        """评估攻击成功率"""
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
        """返回所有客户端"""
        return self.clients

    def train(self):
        """按照MultiSim伪代码实现的训练流程"""
        
        # 阶段1: 聚类阶段 - P ← CorrelationFL(W, T0, K); C ← HierarchicalClustering(P, h)
        T0 = getattr(self.args, 'initial_rounds', 10)  # 聚类阶段轮数
        print(f"Phase 1: Clustering Stage with CorrelationFL for {T0} rounds...")
        
        # 初始化全局相似性矩阵P
        num_all_clients = len(self.clients)
        all_client_indices = list(range(num_all_clients))
        self.global_similarity_matrix = np.eye(num_all_clients)
        
        for t in range(T0):
            print(f"\nClustering Round: {t+1}/{T0}")
            
            client_weights, client_gradients = self.run_one_round(
                self.clients, 
                self.global_similarity_matrix,
                all_client_indices,
                is_clustering_phase=True,
                round_num=t,
                total_rounds=T0
            )
            
            # 计算相似性矩阵（支持双层次相似度）
            if self.use_dual_similarity and self.use_model_split:
                # 双层次相似度：特征相似度 + head相似度
                P_feat = self._compute_feature_similarity(self.clients, all_client_indices)
                P_g = self.gradient_similarity(client_gradients)
                P_w = self.weight_similarity(client_weights)
                # 融合head相似度
                P_head = self.similarity_fusion(P_g, P_w)
                # 融合特征和head相似度
                self.global_similarity_matrix = (self.feature_sim_weight * P_feat + 
                                                (1 - self.feature_sim_weight) * P_head)
            else:
                P_g = self.gradient_similarity(client_gradients)
                P_w = self.weight_similarity(client_weights)
                self.global_similarity_matrix = self.similarity_fusion(P_g, P_w)

            # 在聚类阶段，我们需要一个全局模型来同步客户端
            self.selected_clients = self.clients
            self.receive_models()
            self.aggregate_parameters() # 使用 FedAvg 进行全局聚合
            self.send_models() # 将新的全局模型发给所有客户端作为下一轮起点

            if t % self.eval_gap == 0:
                self.evaluate()
                
                # 记录正常训练数据（无攻击时）
                if not self.attack_method or self.attack_method == 'none':
                    self.base_attack.data_recorder.record_normal_training_data(t, self)

        # T0轮结束后，进行最终的层次聚类
        self.client_groups = self.hierarchical_clustering_from_similarity(self.global_similarity_matrix, all_client_indices)
        print(f"--- Clustering completed. Groups: {self.client_groups} ---")

        # 阶段2: 簇内训练阶段
        T1 = self.global_rounds - T0
        print(f"--- Phase 2: Intra-cluster Training for {T1} rounds ---")
        
        # 为每个簇初始化模型和相似性矩阵
        cluster_models = {}
        for group_id, group_client_indices in enumerate(self.client_groups):
            # 使用全局模型的深拷贝作为每个簇的初始模型
            cluster_models[group_id] = copy.deepcopy(self.global_model)
            # 初始化簇内相似性矩阵为单位矩阵
            self.cluster_similarity_matrices[group_id] = np.eye(len(group_client_indices))

        for t in range(T1):
            print(f"\nIntra-cluster Training Round: {t+1}/{T1}")
            # 并行处理每个聚类
            for group_id, group_client_indices in enumerate(self.client_groups):
                
                cluster_clients = [self.clients[i] for i in group_client_indices]
                
                # 为该簇的客户端设置簇模型
                for client in cluster_clients:
                    client.set_model(cluster_models[group_id])

                client_weights, client_gradients = self.run_one_round(
                    cluster_clients,
                    self.cluster_similarity_matrices[group_id],
                    group_client_indices,
                    is_clustering_phase=False,
                    round_num=t,
                    total_rounds=T1
                )
                
                if self.use_dual_similarity and self.use_model_split:
                    P_feat = self._compute_feature_similarity(cluster_clients, group_client_indices)
                    P_g = self.gradient_similarity(client_gradients)
                    P_w = self.weight_similarity(client_weights)
                    P_head = self.similarity_fusion(P_g, P_w)
                    self.cluster_similarity_matrices[group_id] = (self.feature_sim_weight * P_feat + 
                                                                 (1 - self.feature_sim_weight) * P_head)
                else:
                    P_g = self.gradient_similarity(client_gradients)
                    P_w = self.weight_similarity(client_weights)
                    self.cluster_similarity_matrices[group_id] = self.similarity_fusion(P_g, P_w)
                self.selected_clients = cluster_clients # 告诉服务器哪些客户端是当前活跃的
                # 保存原始值
                original_join_clients = self.join_clients
                original_current_num_join_clients = self.current_num_join_clients
                # 动态计算当前簇应该参与的数量
                cluster_join_clients = max(1, int(self.join_ratio * len(cluster_clients)))
                self.join_clients = cluster_join_clients
                self.current_num_join_clients = cluster_join_clients

                self.receive_models() # 从活跃客户端接收模型，填充 self.uploaded_models
                
                # 恢复原始值
                self.join_clients = original_join_clients
                self.current_num_join_clients = original_current_num_join_clients
                cluster_models[group_id] = self.aggregate_cluster_models(cluster_clients)

            if t % self.eval_gap == 0:
                # 评估时，需要先将最新的簇模型分发给客户端
                for group_id, group_client_indices in enumerate(self.client_groups):
                    for client_idx in group_client_indices:
                        self.clients[client_idx].set_model(cluster_models[group_id])
                self.evaluate()
                
                # 记录正常训练数据（无攻击时）
                if not self.attack_method or self.attack_method == 'none':
                    self.base_attack.data_recorder.record_normal_training_data(T0 + t, self)

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        
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
            self.set_new_clients(clientMUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def train_with_attack(self, pattern, trigger):
        """FedMul版本的攻击训练方法，适配两阶段训练流程"""
        original_trigger_list = trigger
        optimized_trigger_list = copy.deepcopy(original_trigger_list)
        external_pattern = pattern 
        attack_start = 2
        oneshot = 1         
        clip_rate = 0.2     

        # 阶段1: 聚类阶段 - P ← CorrelationFL(W, T0, K); C ← HierarchicalClustering(P, h)
        T0 = getattr(self.args, 'initial_rounds', 10)  # 聚类阶段轮数
        print(f"Phase 1: Clustering Stage with CorrelationFL for {T0} rounds (with attack)...")
        
        # 初始化全局相似性矩阵P
        num_all_clients = len(self.clients)
        all_client_indices = list(range(num_all_clients))
        self.global_similarity_matrix = np.eye(num_all_clients)
        
        for t in range(T0):
            print(f"\nClustering Round: {t+1}/{T0}")
            s_t = time.time()
            
            # 选择客户端和设置所有客户端（攻击训练需要）
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()
            self.send_models()
            
            # 执行攻击训练（在聚类阶段）
            if t >= attack_start and self.attack_method:
                self._execute_attack_training(t, attack_start, oneshot, clip_rate, 
                                           original_trigger_list, external_pattern, optimized_trigger_list)
            
            client_weights, client_gradients = self.run_one_round(
                self.clients, 
                self.global_similarity_matrix,
                all_client_indices,
                is_clustering_phase=True,
                round_num=t,
                total_rounds=T0
            )
            
            if self.use_dual_similarity and self.use_model_split:
                P_feat = self._compute_feature_similarity(self.clients, all_client_indices)
                P_g = self.gradient_similarity(client_gradients)
                P_w = self.weight_similarity(client_weights)
                P_head = self.similarity_fusion(P_g, P_w)
                self.global_similarity_matrix = (self.feature_sim_weight * P_feat + 
                                                (1 - self.feature_sim_weight) * P_head)
            else:
                P_g = self.gradient_similarity(client_gradients)
                P_w = self.weight_similarity(client_weights)
                self.global_similarity_matrix = self.similarity_fusion(P_g, P_w)

            # 在聚类阶段，我们需要一个全局模型来同步客户端
            # 注意：这里使用所有客户端进行聚合，而不是只使用selected_clients
            self.selected_clients = self.clients
            self.receive_models()
            self.aggregate_parameters() # 使用 FedAvg 进行全局聚合

            if t % self.eval_gap == 0:
                print(f"\n-------------Round number: {t}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 评估ASR
                if t >= attack_start and self.attack_method:
                    self._evaluate_attack_success(t, attack_start, original_trigger_list, 
                                                external_pattern, optimized_trigger_list)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        # T0轮结束后，进行最终的层次聚类
        self.client_groups = self.hierarchical_clustering_from_similarity(self.global_similarity_matrix, all_client_indices)
        print(f"--- Clustering completed. Groups: {self.client_groups} ---")

        # 阶段2: 簇内训练阶段
        T1 = self.global_rounds - T0
        print(f"--- Phase 2: Intra-cluster Training for {T1} rounds (with attack) ---")
        
        # 为每个簇初始化模型和相似性矩阵
        cluster_models = {}
        for group_id, group_client_indices in enumerate(self.client_groups):
            # 使用全局模型的深拷贝作为每个簇的初始模型
            cluster_models[group_id] = copy.deepcopy(self.global_model)
            # 初始化簇内相似性矩阵为单位矩阵
            self.cluster_similarity_matrices[group_id] = np.eye(len(group_client_indices))

        for t in range(T1):
            print(f"\nIntra-cluster Training Round: {t+1}/{T1}")
            s_t = time.time()
            
            # 执行攻击训练（在簇内训练阶段）
            if t >= attack_start and self.attack_method:
                self._execute_attack_training(t, attack_start, oneshot, clip_rate, 
                                           original_trigger_list, external_pattern, optimized_trigger_list)
            
            # 并行处理每个聚类
            for group_id, group_client_indices in enumerate(self.client_groups):
                
                cluster_clients = [self.clients[i] for i in group_client_indices]
                
                for client in cluster_clients:
                    client.set_model(cluster_models[group_id])

                client_weights, client_gradients = self.run_one_round(
                    cluster_clients,
                    self.cluster_similarity_matrices[group_id],
                    group_client_indices,
                    is_clustering_phase=False,
                    round_num=t,
                    total_rounds=T1
                )
                
                if self.use_dual_similarity and self.use_model_split:
                    P_feat = self._compute_feature_similarity(cluster_clients, group_client_indices)
                    P_g = self.gradient_similarity(client_gradients)
                    P_w = self.weight_similarity(client_weights)
                    P_head = self.similarity_fusion(P_g, P_w)
                    self.cluster_similarity_matrices[group_id] = (self.feature_sim_weight * P_feat + 
                                                                 (1 - self.feature_sim_weight) * P_head)
                else:
                    P_g = self.gradient_similarity(client_gradients)
                    P_w = self.weight_similarity(client_weights)
                    self.cluster_similarity_matrices[group_id] = self.similarity_fusion(P_g, P_w)
                self.selected_clients = cluster_clients # 告诉服务器哪些客户端是当前活跃的
                # 保存原始值
                original_join_clients = self.join_clients
                original_current_num_join_clients = self.current_num_join_clients
                # 动态计算当前簇应该参与的数量
                cluster_join_clients = max(1, int(self.join_ratio * len(cluster_clients)))
                self.join_clients = cluster_join_clients
                self.current_num_join_clients = cluster_join_clients

                self.receive_models() # 从活跃客户端接收模型，填充 self.uploaded_models
                
                # 恢复原始值
                self.join_clients = original_join_clients
                self.current_num_join_clients = original_current_num_join_clients
                cluster_models[group_id] = self.aggregate_cluster_models(cluster_clients)

            if t % self.eval_gap == 0:
                # 评估时，需要先将最新的簇模型分发给客户端
                for group_id, group_client_indices in enumerate(self.client_groups):
                    for client_idx in group_client_indices:
                        self.clients[client_idx].set_model(cluster_models[group_id])
                
                print(f"\n-------------Round number: {T0 + t}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 评估ASR
                if t >= attack_start and self.attack_method:
                    self._evaluate_attack_success(T0 + t, attack_start, original_trigger_list, 
                                                external_pattern, optimized_trigger_list)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

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
            self.set_new_clients(clientMUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def _execute_attack_training(self, round_num, attack_start, oneshot, clip_rate, 
                               original_trigger_list, external_pattern, optimized_trigger_list):
        """执行攻击训练的具体逻辑"""
        if round_num < attack_start:
            return optimized_trigger_list
            
        # 根据攻击类型执行对应的攻击训练
        if self.attack_method == 'badpfl':
            badpfl_trigger, badpfl_pattern = self.badpfl_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局Bad-PFL攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_badpfl_attack = self.badpfl_attack
            return badpfl_trigger, badpfl_pattern
            
        elif self.attack_method == 'neurotoxin':
            optimized_trigger_list = self.neurotoxin_attack.execute_attack_training(
                server=self,
                round_num=round_num,
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
                round_num=round_num,
                attack_start=attack_start,
                oneshot=oneshot,
                clip_rate=clip_rate,
                original_trigger_list=original_trigger_list,
                external_pattern=external_pattern,
                optimized_trigger_list=optimized_trigger_list
            )
            
        elif self.attack_method == 'model_replacement':
            optimized_trigger_list = self.model_replacement_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局模型替换攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_model_replacement_attack = self.model_replacement_attack
            
        elif self.attack_method == 'badnets':
            optimized_trigger_list = self.badnets_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局BadNets攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_badnets_attack = self.badnets_attack
        
        return optimized_trigger_list
    
    def _evaluate_attack_success(self, round_num, attack_start, original_trigger_list, 
                                external_pattern, optimized_trigger_list):
        """评估攻击成功率"""
        if round_num < attack_start:
            return
            
        if self.attack_method == 'badpfl':
            # BadPFL的ASR评估需要特殊处理
            global_asr = self.evaluate_asr(
                trigger=optimized_trigger_list,
                pattern=external_pattern
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
        
        print(f"Global ASR at round {round_num}: {global_asr:.4f}")
    
    def run_one_round(self, clients, similarity_matrix_for_training, client_indices, is_clustering_phase, round_num=None, total_rounds=None):
        """执行一轮完整的客户端训练和信息收集（优化版：Top-K + 只存储head）"""
        
        client_weights = []
        client_gradients = []
        
        # 创建client_id到client对象的映射
        client_id_to_client = {client.id: client for client in clients}
        
        # Top-K邻居模型准备（只存储head）
        neighbor_models_map = {}
        for i, client in enumerate(clients):
            my_idx = client_indices.index(client.id) if client.id in client_indices else i
            
            if similarity_matrix_for_training is not None and similarity_matrix_for_training.shape[0] > my_idx:
                similarities = [(similarity_matrix_for_training[my_idx, j], client_indices[j]) 
                               for j in range(len(client_indices)) if j != my_idx]
                top_k_neighbors = sorted(similarities, reverse=True)[:self.top_k_neighbors]
                
                if self.use_model_split:
                    neighbor_models_map[client.id] = {
                        neighbor_id: client_id_to_client[neighbor_id].model.head.state_dict() 
                        for _, neighbor_id in top_k_neighbors if neighbor_id in client_id_to_client
                    }
                else:
                    neighbor_models_map[client.id] = {
                        neighbor_id: client_id_to_client[neighbor_id].model.state_dict() 
                        for _, neighbor_id in top_k_neighbors if neighbor_id in client_id_to_client
                    }
            else:
                neighbor_models_map[client.id] = {}

        for client in clients:
            pre_params = self._get_client_params(client)
            
            client.set_neighbor_models(neighbor_models_map[client.id])
            client.train_with_similarity_constraint(
                similarity_matrix_for_training, 
                client_indices,
                round_num=round_num,
                total_rounds=total_rounds
            )
            
            post_params = self._get_client_params(client)
            gradient = [post - pre for post, pre in zip(post_params, pre_params)]
            
            client_weights.append(post_params)
            client_gradients.append(gradient)
            
        return client_weights, client_gradients
    
    def aggregate_cluster_models(self, cluster_clients):
        """对一个簇内的客户端模型进行FedAvg聚合（支持模型分割：base聚合，head个性化）"""
        
        cluster_client_ids = {client.id for client in cluster_clients}
        
        uploaded_models_in_cluster = []
        training_samples_in_cluster = []
        for i, client_id in enumerate(self.uploaded_ids):
            if client_id in cluster_client_ids:
                uploaded_models_in_cluster.append(self.uploaded_models[i])
                training_samples_in_cluster.append(self.clients[client_id].train_samples)

        if not uploaded_models_in_cluster:
            if cluster_clients:
                return copy.deepcopy(cluster_clients[0].model)
            return None

        total_samples = sum(training_samples_in_cluster)
        aggregated_model = copy.deepcopy(uploaded_models_in_cluster[0])
        
        if self.use_model_split:
            # 模型分割：只聚合base部分，head保持个性化（返回时head使用第一个客户端的head）
            for param in aggregated_model.base.parameters():
                param.data.zero_()
            
            for model, samples in zip(uploaded_models_in_cluster, training_samples_in_cluster):
                weight = samples / total_samples
                for base_param, client_param in zip(aggregated_model.base.parameters(), model.base.parameters()):
                    base_param.data += client_param.data.clone() * weight
            # head保持第一个客户端的个性化head
        else:
            # 未分割：聚合全部参数
            for param in aggregated_model.parameters():
                param.data.zero_()
            
            for model, samples in zip(uploaded_models_in_cluster, training_samples_in_cluster):
                weight = samples / total_samples
                for base_param, client_param in zip(aggregated_model.parameters(), model.parameters()):
                    base_param.data += client_param.data.clone() * weight
        
        return aggregated_model
    
    
    
    def gradient_similarity(self, client_gradients, use_pca=None):
        """计算梯度相似性矩阵 P_g（优化版：支持PCA降维）"""
        num_clients = len(client_gradients)
        similarity_matrix = np.zeros((num_clients, num_clients))
        
        if use_pca is None:
            use_pca = self.use_pca_similarity
    
        flattened_gradients = []
        for gradients in client_gradients:
            flat_grad = np.concatenate([g.flatten() if not isinstance(g, np.ndarray) else g.flatten() for g in gradients])
            flattened_gradients.append(flat_grad)
        
        # PCA降维
        if use_pca and len(flattened_gradients) > 1:
            from sklearn.decomposition import PCA
            gradients_matrix = np.array(flattened_gradients)
            pca_dim = min(self.pca_dimension, num_clients - 1, gradients_matrix.shape[1])
            if pca_dim > 0:
                pca = PCA(n_components=pca_dim)
                reduced_gradients = pca.fit_transform(gradients_matrix)
                flattened_gradients = reduced_gradients
        
        for i in range(num_clients):
            for j in range(i, num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    g1, g2 = flattened_gradients[i], flattened_gradients[j]
                    norm1, norm2 = np.linalg.norm(g1), np.linalg.norm(g2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(g1, g2) / (norm1 * norm2)
                        similarity_matrix[i, j] = max(0, sim)
                        similarity_matrix[j, i] = similarity_matrix[i, j]
                    else:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
        
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums
        return normalized_matrix
    
    def weight_similarity(self, client_weights, use_pca=None):
        """计算权重相似性矩阵 P_w（优化版：支持PCA降维）"""
        num_clients = len(client_weights)
        similarity_matrix = np.zeros((num_clients, num_clients))
        
        if use_pca is None:
            use_pca = self.use_pca_similarity

        flattened_weights = []
        for weights in client_weights:
            flat_weight = np.concatenate([w.flatten() for w in weights])
            flattened_weights.append(flat_weight)

        # PCA降维
        if use_pca and len(flattened_weights) > 1:
            from sklearn.decomposition import PCA
            weights_matrix = np.array(flattened_weights)
            pca_dim = min(self.pca_dimension, num_clients - 1, weights_matrix.shape[1])
            if pca_dim > 0:
                pca = PCA(n_components=pca_dim)
                reduced_weights = pca.fit_transform(weights_matrix)
                flattened_weights = reduced_weights

        for i in range(num_clients):
            for j in range(i, num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    w1, w2 = flattened_weights[i], flattened_weights[j]
                    norm1, norm2 = np.linalg.norm(w1), np.linalg.norm(w2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(w1, w2) / (norm1 * norm2)
                        similarity_matrix[i, j] = max(0, sim)
                        similarity_matrix[j, i] = similarity_matrix[i, j]
                    else:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
        
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums
        return normalized_matrix

    
    def similarity_fusion(self, P_g, P_w):
        """相似性融合 P^c ← SimilarityFusion(P_g^c, P_w^c) (伪代码第15行)"""
        # 使用注意力融合方法
        return self.fuse_attention([P_g, P_w])
    
    def hierarchical_clustering_from_similarity(self, similarity_matrix, indices):
        """基于相似性矩阵的层次聚类 C ← HierarchicalClustering(P, h)"""
        from scipy.cluster.hierarchy import linkage, fcluster
        
        if len(indices) <= 1:
            return [indices]
        
        try:
            # 将相似性转换为距离
            distance_matrix = 1.0 - similarity_matrix
            
            # 确保距离矩阵是对称的且对角线为0
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # 转换为压缩距离矩阵
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance_matrix)
            
            # 层次聚类
            linkage_matrix = linkage(condensed_dist, method='single')
            cluster_labels = fcluster(linkage_matrix, t=self.num_clusters, criterion='maxclust')
            
            # 分组
            groups = {}
            for i, label in enumerate(cluster_labels):
                if label not in groups:
                    groups[label] = []
                groups[label].append(indices[i])
            
            return list(groups.values())
            
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}, using random grouping")
            # 降级到随机分组
            group_size = len(indices) // self.num_clusters
            groups = []
            for i in range(self.num_clusters):
                start_idx = i * group_size
                if i == self.num_clusters - 1:
                    groups.append(indices[start_idx:])
                else:
                    groups.append(indices[start_idx:start_idx + group_size])
            return [g for g in groups if g]

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # 输出最终聚类结果和统计信息
        self.print_final_clustering_stats()

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
            self.set_new_clients(clientMUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def compute_multisim_similarity(self):
        """计算MultiSim多度量相似性矩阵"""
        print("Computing MultiSim similarity matrices...")
        
        num_clients = len(self.selected_clients)
        self.gradient_similarity_matrix = np.zeros((num_clients, num_clients))
        self.weight_similarity_matrix = np.zeros((num_clients, num_clients))
        
        # 收集所有客户端的梯度和权重签名
        client_gradients = []
        client_weights = []
        
        for client in self.selected_clients:
            gradient_sig = client.get_gradient_signature()
            weight_sig = client.get_weight_signature()
            client_gradients.append(gradient_sig)
            client_weights.append(weight_sig)
        
        # 计算梯度相似性矩阵
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    self.gradient_similarity_matrix[i][j] = 1.0
                    self.weight_similarity_matrix[i][j] = 1.0
                else:
                    # 梯度相似性（余弦相似度）
                    grad_sim = self.selected_clients[i].compute_gradient_similarity(client_gradients[j])
                    self.gradient_similarity_matrix[i][j] = grad_sim
                    
                    # 权重相似性（KL散度转换）
                    weight_sim = self.selected_clients[i].compute_weight_similarity(client_weights[j])
                    self.weight_similarity_matrix[i][j] = weight_sim
        
        # 使用注意力融合方法
        self.similarity_matrix = self.fuse_attention([
            self.gradient_similarity_matrix,
            self.weight_similarity_matrix
        ])
        
        print(f"Gradient similarity matrix shape: {self.gradient_similarity_matrix.shape}")
        print(f"Weight similarity matrix shape: {self.weight_similarity_matrix.shape}")
        print(f"Fused similarity matrix shape: {self.similarity_matrix.shape}")
        
        # 输出相似性统计信息
        avg_grad_sim = np.mean(self.gradient_similarity_matrix[np.triu_indices_from(self.gradient_similarity_matrix, k=1)])
        avg_weight_sim = np.mean(self.weight_similarity_matrix[np.triu_indices_from(self.weight_similarity_matrix, k=1)])
        avg_fused_sim = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
        
        print(f"Average gradient similarity: {avg_grad_sim:.4f}")
        print(f"Average weight similarity: {avg_weight_sim:.4f}")
        print(f"Average fused similarity: {avg_fused_sim:.4f}")

    def update_clusters(self):
        """基于相似性矩阵更新聚类"""
        if self.similarity_matrix is None:
            print("Similarity matrix not computed yet, skipping clustering")
            return
        
        print(f"Updating clusters using {self.clustering_method} method...")
        
        num_clients = len(self.selected_clients)
        
        if num_clients < self.num_clusters:
            print(f"Warning: Number of clients ({num_clients}) < number of clusters ({self.num_clusters})")
            # 每个客户端分配到不同的聚类
            for i, client in enumerate(self.selected_clients):
                self.client_clusters[client.id] = i
                client.set_cluster_id(i)
            return
        
        try:
            if self.clustering_method == 'hierarchical':
                # 使用层次聚类
                # 将相似性矩阵转换为距离矩阵
                distance_matrix = 1.0 - self.similarity_matrix
                
                # 确保距离矩阵是对称的且对角线为0
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                
                # 将距离矩阵转换为压缩格式用于层次聚类
                condensed_dist = pdist(distance_matrix, metric='precomputed')
                
                # 执行层次聚类
                linkage_matrix = linkage(condensed_dist, method='ward')
                cluster_labels = fcluster(linkage_matrix, t=self.num_clusters, criterion='maxclust')
                
            elif self.clustering_method == 'agglomerative':
                # 使用sklearn的聚合聚类
                # 将相似性转换为距离
                affinity_matrix = self.similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=self.num_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(affinity_matrix)
                cluster_labels += 1  # 使聚类标签从1开始
            
            else:
                raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
            # 更新客户端聚类分配
            for i, client in enumerate(self.selected_clients):
                cluster_id = int(cluster_labels[i])
                self.client_clusters[client.id] = cluster_id
                client.set_cluster_id(cluster_id)
            
            # 输出聚类结果
            self.print_clustering_info()
            
            # 检测异常客户端（可能的恶意客户端）
            self.detect_malicious_clients()
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            # 降级到随机分配
            for i, client in enumerate(self.selected_clients):
                cluster_id = (i % self.num_clusters) + 1
                self.client_clusters[client.id] = cluster_id
                client.set_cluster_id(cluster_id)

    def update_cluster_centers(self):
        """更新聚类中心"""
        print("Updating cluster centers...")
        
        # 按聚类分组客户端
        clusters = {}
        for client in self.selected_clients:
            cluster_id = client.get_cluster_id()
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client)
        
        # 计算每个聚类的中心
        self.cluster_centers = {}
        for cluster_id, clients_in_cluster in clusters.items():
            if not clients_in_cluster:
                continue
            
            # 获取聚类中所有客户端的模型参数
            cluster_params = []
            for client in clients_in_cluster:
                params = []
                for param in client.model.parameters():
                    params.append(param.data.flatten().cpu())
                cluster_params.append(torch.cat(params).numpy())
            
            # 计算平均值作为聚类中心
            if cluster_params:
                cluster_center = np.mean(cluster_params, axis=0)
                self.cluster_centers[cluster_id] = cluster_center
                
                # 将聚类中心发送给聚类内的客户端
                for client in clients_in_cluster:
                    client.set_cluster_weight(cluster_center)
        
        print(f"Updated {len(self.cluster_centers)} cluster centers")

    def multisim_aggregate(self):
        """MultiSim聚合：基于聚类的加权聚合 - 简化版用于baseline"""
        assert (len(self.uploaded_models) > 0)
        
        # 如果客户端聚类信息不存在，使用传统FedAvg
        if not hasattr(self, 'client_clusters') or not self.client_clusters:
            print("No clustering info, using FedAvg aggregation")
            self.aggregate_parameters()
            return
        
        # 按聚类分组上传的模型
        cluster_models = {}
        cluster_weights = {}
        
        for i, client_id in enumerate(self.uploaded_ids):
            cluster_id = self.client_clusters.get(client_id, 0)
            
            if cluster_id not in cluster_models:
                cluster_models[cluster_id] = []
                cluster_weights[cluster_id] = []
            
            cluster_models[cluster_id].append(self.uploaded_models[i])
            # 使用训练样本数作为权重
            cluster_weights[cluster_id].append(self.clients[client_id].train_samples)
        
        # 聚类内简单加权平均聚合
        cluster_aggregated = {}
        for cluster_id, models in cluster_models.items():
            if not models:
                continue
            
            weights = cluster_weights[cluster_id]
            total_weight = sum(weights)
            
            # 简单加权平均（不使用ADMM）
            aggregated_model = copy.deepcopy(models[0])
            for param in aggregated_model.parameters():
                param.data.zero_()
            
            for model, weight in zip(models, weights):
                weight_ratio = weight / total_weight
                for agg_param, model_param in zip(aggregated_model.parameters(), model.parameters()):
                    agg_param.data += model_param.data * weight_ratio
            
            cluster_aggregated[cluster_id] = (aggregated_model, total_weight)
        
        # 全局聚合：聚类间聚合
        if cluster_aggregated:
            # 计算全局聚合权重
            total_samples = sum(weight for _, weight in cluster_aggregated.values())
            
            # 初始化全局模型
            self.global_model = copy.deepcopy(list(cluster_aggregated.values())[0][0])
            for param in self.global_model.parameters():
                param.data.zero_()
            
            # 按聚类权重聚合
            for cluster_id, (model, cluster_weight) in cluster_aggregated.items():
                weight_ratio = cluster_weight / total_samples
                
                for global_param, cluster_param in zip(self.global_model.parameters(), model.parameters()):
                    global_param.data += cluster_param.data.clone() * weight_ratio
        
        else:
            # 如果没有聚类信息，使用传统FedAvg聚合
            self.aggregate_parameters()
        
        print(f"MultiSim aggregation completed with {len(cluster_aggregated)} clusters")

    def detect_malicious_clients(self):
        """基于相似性检测恶意客户端"""
        if self.similarity_matrix is None:
            return
        
        detected_malicious = []
        
        for i, client in enumerate(self.selected_clients):
            # 计算客户端与其他客户端的平均相似性
            similarities = self.similarity_matrix[i, :]
            avg_similarity = np.mean(similarities[similarities != 1.0])  # 排除自己
            
            # 如果平均相似性低于阈值，标记为可疑
            if avg_similarity < self.robustness_threshold:
                detected_malicious.append(client.id)
                print(f"Client {client.id} detected as potentially malicious (avg_sim: {avg_similarity:.4f})")
        
        # 更新恶意客户端列表
        self.malicious_clients.extend(detected_malicious)
        self.malicious_clients = list(set(self.malicious_clients))  # 去重
        
        if detected_malicious:
            print(f"Total detected malicious clients: {len(self.malicious_clients)}")
    
    
    
    

    def evaluate_personalized_models(self):
        """评估个性化模型性能"""
        print("\nEvaluating personalized models...")
        
        cluster_accuracies = {}
        total_personalized_acc = 0
        total_samples = 0
        
        for client in self.clients:
            if hasattr(client, 'evaluate_personalized'):
                acc, samples = client.evaluate_personalized()
                cluster_id = client.get_cluster_id()
                
                if cluster_id not in cluster_accuracies:
                    cluster_accuracies[cluster_id] = []
                
                cluster_accuracies[cluster_id].append(acc)
                total_personalized_acc += acc * samples
                total_samples += samples
        
        # 输出聚类级别的性能
        for cluster_id, accuracies in cluster_accuracies.items():
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"Cluster {cluster_id}: Avg Acc = {avg_acc:.2f}% ± {std_acc:.2f}%")
        
        # 输出全局个性化性能
        if total_samples > 0:
            global_personalized_acc = total_personalized_acc / total_samples
            print(f"Global Personalized Accuracy: {global_personalized_acc:.2f}%")

    def print_clustering_info(self):
        """输出聚类信息"""
        print("\nCurrent clustering assignment:")
        clusters = {}
        for client_id, cluster_id in self.client_clusters.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
        
        for cluster_id, client_ids in clusters.items():
            print(f"Cluster {cluster_id}: Clients {client_ids} ({len(client_ids)} clients)")

    def print_final_clustering_stats(self):
        """输出最终聚类统计信息"""
        print("\n" + "="*50)
        print("FINAL MULTISIM CLUSTERING STATISTICS")
        print("="*50)
        
        # 聚类分布
        cluster_sizes = {}
        for cluster_id in self.client_clusters.values():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        print(f"Number of clusters formed: {len(cluster_sizes)}")
        for cluster_id, size in cluster_sizes.items():
            print(f"Cluster {cluster_id}: {size} clients")
        
        # 相似性统计
        if self.similarity_matrix is not None:
            avg_similarity = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            max_similarity = np.max(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            min_similarity = np.min(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            
            print(f"\nSimilarity Statistics:")
            print(f"Average similarity: {avg_similarity:.4f}")
            print(f"Maximum similarity: {max_similarity:.4f}")
            print(f"Minimum similarity: {min_similarity:.4f}")
        
        # 恶意客户端检测结果
        if self.malicious_clients:
            print(f"\nDetected malicious clients: {self.malicious_clients}")
        else:
            print(f"\nNo malicious clients detected")
        
        print("="*50)

    def _get_client_params(self, client):
        """获取客户端模型参数（优化版：只获取head参数用于相似度计算）"""
        params = []
        if self.use_model_split:
            for param in client.model.head.parameters():
                params.append(param.data.cpu().numpy().copy())
        else:
            for param in client.model.parameters():
                params.append(param.data.cpu().numpy().copy())
        return params
    
    def _compute_feature_similarity(self, clients, client_indices):
        """基于特征提取器输出计算相似度"""
        num_clients = len(clients)
        similarity_matrix = np.zeros((num_clients, num_clients))
        
        if not self.use_model_split:
            # 未分割模型：使用head参数相似度
            return np.eye(num_clients)
        
        client_features = []
        for client in clients:
            trainloader = client.load_train_data()
            features_list = []
            
            with torch.no_grad():
                client.model.base.eval()
                sample_count = 0
                for x, y in trainloader:
                    if sample_count >= 100:
                        break
                    if isinstance(x, list):
                        x = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    
                    features = client.model.base(x)
                    features_list.append(features.mean(dim=0).cpu().numpy())
                    sample_count += len(x)
            
            if features_list:
                avg_feature = np.mean(features_list, axis=0)
                client_features.append(avg_feature.flatten())
            else:
                client_features.append(np.zeros(1))
        
        for i in range(num_clients):
            for j in range(i, num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    f1, f2 = client_features[i], client_features[j]
                    norm1, norm2 = np.linalg.norm(f1), np.linalg.norm(f2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(f1, f2) / (norm1 * norm2)
                        similarity_matrix[i, j] = max(0, sim)
                        similarity_matrix[j, i] = similarity_matrix[i, j]
        
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums
        return normalized_matrix


    def get_multisim_config(self):
        """获取MultiSim配置信息"""
        return {
            'num_clusters': self.num_clusters,
            'similarity_threshold': self.similarity_threshold,
            'fusion_weight_gradient': self.fusion_weight_gradient,
            'fusion_weight_weight': self.fusion_weight_weight,
            'clustering_method': self.clustering_method,
            'update_cluster_freq': self.update_cluster_freq,
            'personalization_rounds': self.personalization_rounds,
            'cluster_aggregation_weight': self.cluster_aggregation_weight,
            'robustness_threshold': self.robustness_threshold,
            'fuse_method': self.fuse_method
        }
    
    def fuse_attention(self, similarity_matrices):
            """注意力机制融合方法 - 修正版，更贴近论文"""
            print("Using attention fusion method (corrected)...")
            
            if not similarity_matrices:
                return np.array([])
                
            # 确保所有矩阵都是numpy arrays
            matrices = [np.array(m) for m in similarity_matrices]
            
            # 将矩阵堆叠: [num_matrices, num_clients, num_clients]
            stacked_matrices = np.stack(matrices)
            num_matrices, num_clients, _ = stacked_matrices.shape
            
            # 1. 设置 Q, K, V
            # K 和 V 是每个相似性矩阵
            K = V = stacked_matrices
            # Q 是所有矩阵的平均值，作为查询基准
            Q = np.mean(stacked_matrices, axis=0, keepdims=True) # Shape: [1, num_clients, num_clients]
            
            # 2. 计算注意力分数: Score(Q, K) = (Q * K^T) / sqrt(d_k)
            # 这里我们计算Q和每个K的点积相似度
            # 将每个矩阵展平进行点积计算
            Q_flat = Q.reshape(1, -1)
            K_flat = K.reshape(num_matrices, -1)
            
            # d_k 是键向量的维度
            d_k = K_flat.shape[1]
            
            # 计算 Q 和 K_flat 中每个向量的点积
            scores = np.dot(Q_flat, K_flat.T) / np.sqrt(d_k) # Shape: [1, num_matrices]
            
            # 3. 计算注意力权重 (Softmax)
            # 使用scipy的softmax避免数值问题
            from scipy.special import softmax
            attention_weights = softmax(scores.flatten() / self.temperature) # Shape: [num_matrices]
            
            print(f"Attention weights (corrected): {attention_weights}")
            
            # 4. 加权融合 V
            # Reshape weights for broadcasting: [num_matrices, 1, 1]
            reshaped_weights = attention_weights.reshape(num_matrices, 1, 1)
            fused_matrix = np.sum(reshaped_weights * V, axis=0)
            
            return fused_matrix
    
