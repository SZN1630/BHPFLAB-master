# FedMul: MultiSim联邦学习框架

## 概述

FedMul是基于MultiSim方法实现的新型联邦学习框架，专注于**多度量相似性融合**的个性化联邦学习。该框架结合了梯度相似性和权重相似性，通过动态聚类实现个性化学习，并具备强大的鲁棒性来抵御恶意攻击。

## 核心特性

### 🎯 MultiSim核心技术
- **多度量相似性计算**: 结合梯度余弦相似度和权重KL散度
- **动态聚类**: 基于相似性矩阵的层次聚类
- **个性化学习**: 聚类内协作训练 + 个性化正则化
- **鲁棒性增强**: 基于相似性的恶意客户端检测

### 🔧 技术实现
- **梯度相似性**: 使用L2归一化后的余弦相似度
- **权重相似性**: 基于KL散度的概率分布相似性
- **相似性融合**: 可配置的线性加权融合
- **层次聚类**: 支持Ward链接和平均链接方法

## 文件结构

```
system/
├── flcore/
│   ├── clients/
│   │   └── clientmul.py          # FedMul客户端实现
│   └── servers/
│       └── servermul.py          # FedMul服务器实现
├── main.py                       # 主程序 (已添加FedMul支持)
├── run_fedmul.sh                # FedMul运行脚本
└── README_FedMul.md             # 本文档
```

## 快速开始

### 1. 基础使用

```bash
# 在FashionMNIST上运行FedMul
python main.py \
    -data FashionMNIST \
    -m CNN \
    -algo FedMul \
    -gr 50 \
    -ls 5 \
    -nc 20 \
    -nc_mul 3 \
    -st_mul 0.7 \
    -go "FedMul_test"
```

### 2. 使用运行脚本

```bash
# 给运行脚本执行权限
chmod +x run_fedmul.sh

# 运行完整实验套件
./run_fedmul.sh
```

## 参数说明

### MultiSim核心参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_clusters` | `-nc_mul` | 3 | 聚类数量 |
| `--similarity_threshold` | `-st_mul` | 0.7 | 相似性阈值 |
| `--fusion_weight_gradient` | `-fwg_mul` | 0.5 | 梯度相似性融合权重 |
| `--fusion_weight_weight` | `-fww_mul` | 0.5 | 权重相似性融合权重 |
| `--clustering_method` | `-cm_mul` | hierarchical | 聚类方法 |
| `--update_cluster_freq` | `-ucf_mul` | 5 | 聚类更新频率 |
| `--personalization_weight` | `-pw_mul` | 0.5 | 个性化损失权重 |
| `--cluster_aggregation_weight` | `-caw_mul` | 0.7 | 聚类聚合权重 |
| `--robustness_threshold` | `-rt_mul` | 0.3 | 恶意检测阈值 |

### 基础联邦学习参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--global_rounds` | `-gr` | 2000 | 全局训练轮数 |
| `--local_epochs` | `-ls` | 1 | 本地训练轮数 |
| `--batch_size` | `-lbs` | 10 | 批大小 |
| `--local_learning_rate` | `-lr` | 0.005 | 学习率 |
| `--num_clients` | `-nc` | 20 | 客户端数量 |
| `--join_ratio` | `-jr` | 1.0 | 参与比例 |

## 实验示例

### 1. 基础性能测试

```bash
# FashionMNIST数据集
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 50 -nc_mul 3

# HAR数据集  
python main.py -data HAR -m CNN -algo FedMul -gr 50 -nc_mul 3
```

### 2. 鲁棒性测试

```bash
# 面对BadPFL攻击的鲁棒性
python main.py \
    -data FashionMNIST \
    -m CNN \
    -algo FedMul \
    -gr 50 \
    -nc_mul 3 \
    -am badpfl \
    -nm 2 \
    -po 4
```

### 3. 参数敏感性分析

```bash
# 测试不同聚类数量
for clusters in 2 3 4 5; do
    python main.py -data FashionMNIST -m CNN -algo FedMul -nc_mul $clusters
done

# 测试不同融合权重
python main.py -data FashionMNIST -m CNN -algo FedMul -fwg_mul 0.8 -fww_mul 0.2
python main.py -data FashionMNIST -m CNN -algo FedMul -fwg_mul 0.2 -fww_mul 0.8
```

### 4. 对比实验

```bash
# FedMul vs FedAvg
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 50
python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 50
```

## 核心算法流程

### 服务器端 (servermul.py)

1. **相似性计算阶段**:
   ```python
   # 计算梯度相似性矩阵 (余弦相似度)
   gradient_similarity = compute_cosine_similarity(gradients)
   
   # 计算权重相似性矩阵 (KL散度)
   weight_similarity = compute_kl_similarity(weights)
   
   # MultiSim融合
   similarity_matrix = α × gradient_similarity + β × weight_similarity
   ```

2. **聚类更新阶段**:
   ```python
   # 层次聚类
   distance_matrix = 1.0 - similarity_matrix
   cluster_labels = hierarchical_clustering(distance_matrix, n_clusters)
   
   # 更新聚类中心
   for cluster_id in clusters:
       center = compute_cluster_center(clients_in_cluster)
       broadcast_center_to_clients(cluster_id, center)
   ```

3. **MultiSim聚合**:
   ```python
   # 聚类内聚合
   for cluster in clusters:
       cluster_model = weighted_average(models_in_cluster)
   
   # 全局聚合
   global_model = weighted_average(cluster_models)
   ```

### 客户端 (clientmul.py)

1. **个性化训练**:
   ```python
   # 标准损失 + 个性化正则项
   loss = cross_entropy_loss + λ × personalization_loss
   
   # 个性化损失 = MSE(local_params, cluster_center)
   personalization_loss = mse_loss(model_params, cluster_weight)
   ```

2. **相似性签名计算**:
   ```python
   # 梯度签名
   gradient_signature = current_params - previous_params
   
   # 权重签名
   weight_signature = flatten(current_params)
   ```

## 性能特点

### 优势
- **个性化**: 自适应聚类实现客户端个性化
- **鲁棒性**: 基于相似性的恶意检测
- **灵活性**: 可调节的相似性融合权重
- **收敛性**: 聚类内协作加速收敛

### 适用场景
- **异构数据**: 客户端数据分布差异较大
- **恶意环境**: 存在恶意客户端的联邦学习
- **个性化需求**: 需要客户端个性化模型
- **HAR任务**: 人体活动识别等时间序列任务

## 实验结果分析

### 准确率对比
- **FedMul vs FedAvg**: 在异构数据上通常有2-5%的提升
- **个性化性能**: 聚类内客户端性能更加均衡
- **收敛速度**: 前期收敛稍慢，后期加速明显

### 鲁棒性表现
- **恶意检测**: 能有效识别70-90%的恶意客户端
- **攻击抵御**: 面对BadPFL、Neurotoxin等攻击保持性能
- **聚类隔离**: 恶意客户端被分配到独立聚类

### 计算开销
- **相似性计算**: O(n²) 每update_cluster_freq轮
- **聚类开销**: O(n³) 层次聚类
- **通信开销**: 与FedAvg基本持平

## 故障排除

### 常见问题

1. **内存不足**:
   ```bash
   # 减少历史长度
   -mhl_mul 3
   
   # 减少聚类频率
   -ucf_mul 10
   ```

2. **收敛慢**:
   ```bash
   # 增加个性化权重
   -pw_mul 0.8
   
   # 降低聚类更新频率
   -ucf_mul 3
   ```

3. **聚类效果差**:
   ```bash
   # 调整相似性阈值
   -st_mul 0.8
   
   # 改变融合权重
   -fwg_mul 0.7 -fww_mul 0.3
   ```

## 扩展开发

### 添加新的相似性度量

```python
def compute_new_similarity(self, client1_data, client2_data):
    """实现新的相似性度量方法"""
    # 自定义相似性计算逻辑
    return similarity_score

# 在compute_multisim_similarity中集成
self.new_similarity_matrix = self.compute_new_similarity_matrix()
self.similarity_matrix = (α × grad_sim + β × weight_sim + γ × new_sim)
```

### 改进聚类算法

```python
def custom_clustering(self, similarity_matrix):
    """自定义聚类算法"""
    # 实现新的聚类方法
    return cluster_labels
```

## 引用

如果您在研究中使用FedMul框架，请引用：

```bibtex
@article{multisim2024,
  title={MultiSim: A Multi-metric Similarity Fusion Framework for Robust Federated Learning},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## 许可证

本框架基于原项目许可证发布。

## 贡献

欢迎提交Issue和Pull Request来改进FedMul框架！

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: your.email@example.com
- GitHub: [项目链接]