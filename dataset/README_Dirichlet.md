# MHEALTH数据集Dirichlet分布生成指南

## 📊 Dirichlet分布参数说明

Dirichlet分布的α参数控制Non-IID程度：

| α值 | Non-IID程度 | 标签覆盖率 | 适用场景 |
|-----|------------|-----------|---------|
| **0.1** | 极高 | 10-30% | 极端异质性测试 |
| **0.3** | 很高 | 30-40% | 高异质性场景 |
| **0.5** | 高 | 40-50% | 典型Non-IID |
| **1.0** | 中等 | 50-70% | 中等异质性 |
| **5.0** | 低 | 80-90% | 弱Non-IID |
| **10.0** | 很低 | 90-95% | 接近IID |

## 🚀 使用方法

### 1. 修改α值

编辑 `generate_MHEALTH.py` 文件的第15行：

```python
DIRICHLET_ALPHA = 0.1  # 修改这个值
```

### 2. 生成数据集

```bash
cd dataset
python generate_MHEALTH.py
```

### 3. 验证分布

运行分析脚本查看数据分布：

```bash
cd ../system
python analyze_mhealth_distribution.py
```

## 📋 不同α值的预期效果

### α = 0.1 (极高异质性)
```
预期效果:
- 每个客户端平均 1-3 个标签
- 标签覆盖率: 10-25%
- 某些客户端可能只有1个类别
- 类别分布极不均匀
```

### α = 0.5 (高异质性)
```
预期效果:
- 每个客户端平均 4-6 个标签
- 标签覆盖率: 40-50%
- 大部分客户端有多个类别但分布不均
- 适合测试个性化FL算法
```

### α = 1.0 (中等异质性)
```
预期效果:
- 每个客户端平均 6-8 个标签
- 标签覆盖率: 50-70%
- 类别分布相对均衡但仍有差异
- 介于Non-IID和IID之间
```

### α = 10.0 (接近IID)
```
预期效果:
- 每个客户端平均 9-12 个标签
- 标签覆盖率: 90-100%
- 类别分布接近均匀
- 接近独立同分布
```

## 🔧 快速配置脚本

创建 `generate_mhealth_alpha_X.sh` (Linux/Mac) 或 `.ps1` (Windows):

### 生成 α=0.1 数据集
```bash
# 备份原数据
mv MHEALTH MHEALTH_backup

# 修改alpha值
sed -i 's/DIRICHLET_ALPHA = .*/DIRICHLET_ALPHA = 0.1/' generate_MHEALTH.py

# 生成新数据
python generate_MHEALTH.py

# 重命名
mv MHEALTH MHEALTH_alpha_0.1
```

### Windows PowerShell版本
```powershell
# 备份
Move-Item MHEALTH MHEALTH_backup -Force

# 生成 α=0.1
python generate_MHEALTH.py

# 重命名
Move-Item MHEALTH MHEALTH_alpha_0.1
```

## 📊 批量生成多个α值

创建 `generate_all_alphas.py`:

```python
import os
import shutil

alphas = [0.1, 0.3, 0.5, 1.0, 5.0, 10.0]

for alpha in alphas:
    print(f"\n{'='*60}")
    print(f"生成 α={alpha} 的数据集")
    print('='*60)
    
    # 修改generate_MHEALTH.py中的alpha值
    with open('generate_MHEALTH.py', 'r') as f:
        content = f.read()
    
    # 替换alpha值
    import re
    content = re.sub(
        r'DIRICHLET_ALPHA = [\d.]+',
        f'DIRICHLET_ALPHA = {alpha}',
        content
    )
    
    with open('generate_MHEALTH.py', 'w') as f:
        f.write(content)
    
    # 生成数据
    os.system('python generate_MHEALTH.py')
    
    # 重命名
    if os.path.exists('MHEALTH'):
        target = f'MHEALTH_alpha_{alpha}'
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move('MHEALTH', target)
        print(f"数据集保存到: {target}")

print("\n所有数据集生成完成！")
```

## 🎯 训练命令

### 使用不同α值的数据集训练

```bash
# α=0.1 (极高异质性)
python main.py -data MHEALTH_alpha_0.1 -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am none

# α=0.5 (高异质性)
python main.py -data MHEALTH_alpha_0.5 -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am none

# α=1.0 (中等异质性)
python main.py -data MHEALTH_alpha_1.0 -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am none
```

**注意**: 需要修改 `main.py` 使其支持读取不同名称的数据集文件夹。

## 💡 实验建议

### 对比实验设计

```python
实验1: 异质性影响
- 数据集: α ∈ {0.1, 0.5, 1.0, 5.0}
- 算法: FedAvg
- 观察: 准确率随α的变化

实验2: 算法鲁棒性
- 数据集: α = 0.1 (极高异质性)
- 算法: FedAvg vs FedMul vs FedAS
- 观察: 不同算法对Non-IID的抵抗能力

实验3: 攻击影响
- 数据集: α ∈ {0.1, 1.0}
- 算法: FedMul
- 攻击: BadPFL, Neurotoxin
- 观察: Non-IID程度对攻击效果的影响
```

## ⚠️ 注意事项

1. **α=0.1时的特殊处理**
   - 某些客户端可能只有1-2个类别
   - 可能需要调整本地训练参数
   - 建议增加本地训练轮数

2. **数据分布验证**
   - 生成后务必运行分析脚本验证
   - 检查是否有客户端数据过少
   - 确认标签分布符合预期

3. **随机种子**
   - 已设置 `random.seed(1)` 和 `np.random.seed(1)`
   - 相同α值应生成相同分布
   - 可修改种子获得不同分布

## 📚 参考文献

```bibtex
@article{dirichlet_fl,
  title={Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification},
  author={Li et al.},
  journal={arXiv preprint},
  year={2019}
}
```

---

**快速开始**: 修改`DIRICHLET_ALPHA = 0.1`，运行`python generate_MHEALTH.py`即可！🚀
