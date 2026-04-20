import json
import numpy as np
from collections import Counter

# 读取MHEALTH配置
config_path = '../dataset/MHEALTH/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

stats = config['Size of samples for labels in clients']
num_clients = config['num_clients']
num_classes = config['num_classes']

print("="*80)
print("MHEALTH数据集分布分析")
print("="*80)
print(f"客户端数量: {num_clients}")
print(f"类别数量: {num_classes}")
print()

# 分析每个客户端的数据分布
all_label_counts = []
client_total_samples = []
client_num_labels = []

for i in range(num_clients):
    label_dist = dict(stats[i])
    total = sum([v for k, v in label_dist.items()])
    num_labels = len(label_dist)
    
    client_total_samples.append(total)
    client_num_labels.append(num_labels)
    
    labels = sorted(label_dist.keys())
    counts = [label_dist[k] for k in labels]
    
    print(f"客户端 {i:2d}: 样本数={total:4d}, 标签数={num_labels:2d}, 分布={dict(zip(labels, counts))}")
    
    # 计算每个标签的比例
    if i < 5:  # 详细显示前5个客户端
        probs = {k: v/total for k, v in label_dist.items()}
        print(f"          比例: {probs}")
    
    all_label_counts.append(label_dist)

print()
print("="*80)
print("统计摘要")
print("="*80)

# 总样本数统计
print(f"总样本数: {sum(client_total_samples)}")
print(f"平均每客户端样本数: {np.mean(client_total_samples):.1f} ± {np.std(client_total_samples):.1f}")
print(f"最小样本数: {min(client_total_samples)}")
print(f"最大样本数: {max(client_total_samples)}")

# 每个客户端的标签数量
print(f"\n每客户端标签数:")
print(f"  平均: {np.mean(client_num_labels):.1f}")
print(f"  范围: {min(client_num_labels)} - {max(client_num_labels)}")
print(f"  分布: {dict(Counter(client_num_labels))}")

# 全局标签分布
global_label_counts = {}
for client_dist in all_label_counts:
    for label, count in client_dist.items():
        global_label_counts[label] = global_label_counts.get(label, 0) + count

print(f"\n全局标签分布:")
for label in sorted(global_label_counts.keys()):
    count = global_label_counts[label]
    print(f"  标签 {label:2d}: {count:5d} 样本 ({count/sum(global_label_counts.values())*100:.1f}%)")

# 分析数据异质性
print()
print("="*80)
print("数据异质性分析")
print("="*80)

# 计算每个标签在不同客户端的分布方差
label_variance = {}
for label in range(num_classes):
    label_counts_per_client = []
    for client_dist in all_label_counts:
        label_counts_per_client.append(client_dist.get(label, 0))
    
    if sum(label_counts_per_client) > 0:
        # 计算标准差/均值 (变异系数)
        mean_count = np.mean(label_counts_per_client)
        std_count = np.std(label_counts_per_client)
        cv = std_count / mean_count if mean_count > 0 else 0
        
        # 出现在多少个客户端
        num_clients_with_label = sum(1 for c in label_counts_per_client if c > 0)
        
        print(f"标签 {label:2d}: 出现在 {num_clients_with_label:2d}/{num_clients} 个客户端, "
              f"均值={mean_count:5.1f}, 标准差={std_count:5.1f}, 变异系数={cv:.2f}")

# 估计狄利克雷分布系数
print()
print("="*80)
print("狄利克雷分布系数估计")
print("="*80)

# 方法1: 基于标签覆盖率
num_labels_per_client = [len(dist) for dist in all_label_counts]
avg_labels = np.mean(num_labels_per_client)
coverage_ratio = avg_labels / num_classes

print(f"每客户端平均标签数: {avg_labels:.1f} / {num_classes}")
print(f"标签覆盖率: {coverage_ratio:.2%}")

# 估计alpha值
# 覆盖率越高，alpha越大（越接近IID）
# 覆盖率越低，alpha越小（non-IID程度越高）
if coverage_ratio >= 0.9:
    estimated_alpha_range = "10.0 - 100.0 (接近IID)"
elif coverage_ratio >= 0.7:
    estimated_alpha_range = "1.0 - 10.0 (中等异质性)"
elif coverage_ratio >= 0.5:
    estimated_alpha_range = "0.5 - 1.0 (较高异质性)"
else:
    estimated_alpha_range = "0.1 - 0.5 (极高异质性)"

print(f"估计Dirichlet α 范围: {estimated_alpha_range}")

# 方法2: 基于分布熵
entropies = []
for client_dist in all_label_counts:
    total = sum(client_dist.values())
    probs = np.array([client_dist.get(i, 0) / total for i in range(num_classes)])
    probs = probs[probs > 0]  # 移除0概率
    entropy = -np.sum(probs * np.log(probs))
    entropies.append(entropy)

avg_entropy = np.mean(entropies)
max_entropy = np.log(num_classes)  # 均匀分布的熵

print(f"\n平均熵: {avg_entropy:.2f} / {max_entropy:.2f} (最大)")
print(f"熵比率: {avg_entropy/max_entropy:.2%}")

# 熵越接近最大值，分布越均匀，alpha越大
entropy_ratio = avg_entropy / max_entropy
if entropy_ratio >= 0.9:
    alpha_from_entropy = "> 10.0 (非常均匀)"
elif entropy_ratio >= 0.7:
    alpha_from_entropy = "1.0 - 10.0"
elif entropy_ratio >= 0.5:
    alpha_from_entropy = "0.3 - 1.0"
else:
    alpha_from_entropy = "< 0.3 (非常不均匀)"

print(f"基于熵的 α 估计: {alpha_from_entropy}")

# 数据异质性总结
print()
print("="*80)
print("总结")
print("="*80)
print("MHEALTH数据集特点:")
print(f"1. 数据分布: Non-IID (非独立同分布)")
print(f"2. 异质性程度: {'高' if coverage_ratio < 0.6 else '中等' if coverage_ratio < 0.8 else '低'}")
print(f"3. 标签覆盖: 每个客户端平均有 {avg_labels:.1f} 个标签 (共{num_classes}个)")
print(f"4. 推荐Dirichlet α: {estimated_alpha_range}")
print()
print("注意: 该数据集是基于受试者划分的，每个受试者的行为模式不同，")
print("      因此天然具有Non-IID特性，适合联邦学习个性化研究。")
print("="*80)
