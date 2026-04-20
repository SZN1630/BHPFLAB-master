# BHPFLib: Backdoor-Resilient Personalized Federated Learning (FedAvg / FedAS / FedMul)

## Overview
BHPFLib is a federated learning research framework focused on personalized learning under non-IID data and robustness against backdoor attacks. It currently includes three training algorithms:

- `FedAvg`: the standard federated averaging baseline.
- `FedAS`: a personalized federated learning method based on base/head model splitting.
- `FedMul`: a two-stage clustering-based personalized framework inspired by MultiSim. See `system/flcore/servers/servermul.py` and `system/flcore/clients/clientmul.py` for the implementation.

The project also supports multiple attack and defense modules and is designed for experiments on FashionMNIST, HAR, MHEALTH, and WESAD.

## Features

### Algorithms
- `FedAvg`
- `FedAS`
- `FedMul` with clustering, similarity constraints, and personalization

### Attacks
- `badpfl`
- `neurotoxin`
- `dba`
- `model_replacement`
- `badnets`

### Defenses
- `clip` for gradient clipping
- `median` for median aggregation
- `krum` for Krum / Multi-Krum aggregation

## Repository Structure
```text
.
├─dataset/                  # Dataset generation scripts and generated data
├─system/
│  ├─main.py                # Main entry point (argument parsing and algorithm dispatch)
│  ├─run.sh                 # Common launch script
│  └─flcore/
│     ├─clients/
│     │  └─clientmul.py     # FedMul client logic
│     ├─servers/
│     │  └─servermul.py     # FedMul server logic
│     ├─attacks/            # Attack modules
│     └─defense/            # Defense modules
└─README.md
```

## Environment Setup
Python 3.9+ is recommended. Install the common dependencies first:

```bash
pip install torch torchvision numpy scipy scikit-learn h5py
```

## Data Preparation
Generate the datasets from the `dataset/` directory as needed:

```bash
cd dataset
python generate_FashionMNIST.py noniid - dir
python generate_HAR.py noniid - dir
python generate_MHEALTH.py noniid - dir
python generate_WESAD.py noniid - dir
```

## Quick Start
Run experiments from the `system/` directory:

```bash
cd system
```

### Baseline Training
```bash
python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 40 -did 0 -am none
python main.py -data HAR -m CNN -algo FedAS -gr 40 -did 0 -am none
python main.py -data MHEALTH -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am none
python main.py -data WESAD -m CNN -ncl 4 -algo FedMul -gr 40 -did 0 -am none
```

### With Attacks
```bash
python main.py -data FashionMNIST -m CNN -algo FedMul -gr 40 -did 0 -am dba
python main.py -data HAR -m CNN -algo FedMul -gr 40 -did 0 -am neurotoxin
python main.py -data MHEALTH -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am badnets
```

### With Defenses
```bash
python main.py -data FashionMNIST -m CNN -algo FedAvg -gr 40 -did 0 -am none -dm clip -cn 1.0
python main.py -data HAR -m CNN -algo FedAS -gr 40 -did 0 -am none -dm median
python main.py -data HAR -m CNN -algo FedAvg -gr 40 -did 0 -am none -dm krum -kb 2 -mk
```

## FedMul Parameters
Common FedMul options:

- `-nc_mul / --num_clusters`: number of clusters.
- `-ir_mul / --initial_rounds`: number of rounds in the first clustering stage.
- `-alpha_mul / --alpha`: weight for the similarity regularization loss.
- `-lm_mul / --linkage_method`: hierarchical clustering linkage method (`single`, `complete`, `average`, `ward`).
- `-top_k / --top_k_neighbors`: number of top-K neighbors used by the similarity constraint.
- `-sl_freq / --similarity_loss_freq`: frequency of similarity-loss computation.
- `-dual_sim / --use_dual_similarity`: enable dual similarity computation over features and heads.
- `-feat_sim_weight / --feature_sim_weight`: weight for feature similarity in the dual-similarity fusion.

Example:
```bash
python main.py -data MHEALTH -m CNN -ncl 12 -algo FedMul -gr 40 -did 0 -am none -nc_mul 6 -ir_mul 15 -alpha_mul 0.01
```

## Output Files
- Model weights: `system/models/<dataset>/<algorithm>_server.pt`
- Result files: `results/*.h5`

## Notes
- `MHEALTH` usually requires `-ncl 12`.
- `WESAD` usually requires `-ncl 4`.
- If you use `-dev cuda`, make sure CUDA is available.

---
## Temporary GitHub Description (<=350 chars)
FedMul-based federated learning benchmark for non-IID vision and sensor data. Includes FedAvg/FedAS/FedMul, clustering-aware personalization, backdoor attacks (BadPFL, DBA, Neurotoxin, BadNets, Model Replacement), and baseline defenses (clipping, median, krum) with reproducible scripts.
