# PU-GRAIL: Graph Neural Network for Protective Antigen Prediction

We propose PU-GRAIL, a graph neural network framework for protective antigen prediction under positive-unlabeled (PU) learning. By integrating protein language model embeddings with predicted 3D structures, PU-GRAIL effectively learns residue-level representations and improves antigenicity prediction performance.

This repository contains the implementation of PU-GRAIL. The pipeline consists of three main steps: data preprocessing (ESM embedding & structure prediction), graph dataset preparation, and PU-GRAIL training.

## Data Preparation

The input data requires three components:

1. **CSV file** with the following columns:
   - `id_short`: Protein identifier (e.g., 'protein_001')
   - `label`: Binary label (1 = protective antigen, 0 = unlabeled/negative)
   - `fold`: Cross-validation fold number (e.g., 0, 1, 2, ...)
   - `split`: Data split indicator ('train', 'valid', 'test')

2. **PDB files**: Predicted or experimental protein structures
   - Directory containing `{id_short}.pdb` files

3. **ESM embeddings**: Pre-computed ESM embeddings
   - Directory containing `{id_short}.npy` files
   - Shape: (L+2, D) where L is sequence length and D is embedding dimension

### Example CSV format

```csv
id_short,label,fold,split
protein_001,1,0,train
protein_002,0,0,train
protein_003,1,0,valid
protein_004,0,0,test
...
```

Place your data files in the following structure:
```
data/
└── MyDataset/
    ├── fold/
    │   └── index_all_folds.csv
    ├── pdb/
    │   ├── protein_001.pdb
    │   ├── protein_002.pdb
    │   └── ...
    └── esm/
        ├── protein_001.npy
        ├── protein_002.npy
        └── ...
```

## Pipeline

### 1. Generate ESM Embeddings

Run the `generate_esm_embeddings.py` script to generate ESM embeddings for protein sequences:

```bash
python scripts/generate_esm_embeddings.py --fasta_path data/MyDataset/sequences.fasta --output_dir data/MyDataset/esm --model_name esm2_t33_650M_UR50D --device_num 0
```

This script performs the following steps:
- Loading the ESM2 model
- Computing residue-level embeddings for each protein sequence
- Saving embeddings as `.npy` files

#### ESM Embedding Generation Script

`scripts/generate_esm_embeddings.py`: This script generates ESM embeddings for protein sequences.

**Usage**
```bash
python scripts/generate_esm_embeddings.py [--fasta_path FASTA_PATH] [--output_dir OUTPUT_DIR]
                                          [--model_name MODEL_NAME] [--device_num DEVICE_NUM]
                                          [--batch_size BATCH_SIZE]
```

**Arguments**
- `--fasta_path`: Path to FASTA file containing protein sequences (required)
- `--output_dir`: Output directory for ESM embeddings (default: "data/esm")
- `--model_name`: ESM model name (default: "esm2_t33_650M_UR50D")
- `--device_num`: CUDA device number (default: 0)
- `--batch_size`: Batch size for inference (default: 1)

**Example**
```bash
python scripts/generate_esm_embeddings.py --fasta_path data/MyDataset/sequences.fasta --output_dir data/MyDataset/esm --model_name esm2_t33_650M_UR50D --device_num 0 --batch_size 1
```

### 2. Generate Structure Predictions

Run the `generate_structures.py` script to predict protein structures using ESMFold:

```bash
python scripts/generate_structures.py --fasta_path data/MyDataset/sequences.fasta --output_dir data/MyDataset/pdb --device_num 0
```

This script performs the following steps:
- Loading the ESMFold model
- Predicting 3D structures for each protein sequence
- Saving structures as `.pdb` files

#### Structure Prediction Script

`scripts/generate_structures.py`: This script predicts protein structures using ESMFold.

**Usage**
```bash
python scripts/generate_structures.py [--fasta_path FASTA_PATH] [--output_dir OUTPUT_DIR]
                                      [--device_num DEVICE_NUM]
```

**Arguments**
- `--fasta_path`: Path to FASTA file containing protein sequences (required)
- `--output_dir`: Output directory for PDB files (default: "data/pdb")
- `--device_num`: CUDA device number (default: 0)

**Example**
```bash
python scripts/generate_structures.py --fasta_path data/MyDataset/sequences.fasta --output_dir data/MyDataset/pdb --device_num 0
```

### 3. PU-GRAIL Training

Run the `train.py` script to train the PU-GRAIL model:

```bash
python -m pugrail.train --csv_path data/MyDataset/fold/index_all_folds.csv --pdb_dir data/MyDataset/pdb --esm_dir data/MyDataset/esm --output_dir results/MyDataset --cache_dir data/MyDataset/cache --batch_size 16 --lr 1e-4 --epochs 100 --patience 5 --hidden_dim 128 --gnn_layers 3 --esm_proj_dim 256 --pos_encoding_dim 32 --pu_loss puauc --task_loss_weight 1.0 --node_loss_weight 0.3 --contrastive_loss_weight 0.1 --early_metric aupr --seed 42 --device cuda --tag pugrail_exp1
```

This script trains the PU-GRAIL model using the preprocessed data. The trained model and evaluation results will be saved in the `results/MyDataset` directory.

#### PU-GRAIL Training Script

`pugrail/train.py`: This script performs training of the PU-GRAIL model and saves the trained model.

**Usage**
```bash
python -m pugrail.train [--csv_path CSV_PATH] [--pdb_dir PDB_DIR] [--esm_dir ESM_DIR]
                        [--output_dir OUTPUT_DIR] [--cache_dir CACHE_DIR]
                        [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
                        [--patience PATIENCE] [--hidden_dim HIDDEN_DIM]
                        [--gnn_layers GNN_LAYERS] [--esm_proj_dim ESM_PROJ_DIM]
                        [--pos_encoding_dim POS_ENCODING_DIM] [--pu_loss PU_LOSS]
                        [--task_loss_weight TASK_LOSS_WEIGHT]
                        [--node_loss_weight NODE_LOSS_WEIGHT]
                        [--contrastive_loss_weight CONTRASTIVE_LOSS_WEIGHT]
                        [--early_metric EARLY_METRIC] [--seed SEED]
                        [--device DEVICE] [--tag TAG]
```

**Arguments**
- `--csv_path`: Path to CSV file with columns: id_short, label, fold, split (required)
- `--pdb_dir`: Directory containing PDB structure files (required)
- `--esm_dir`: Directory containing ESM embedding files (.npy) (required)
- `--output_dir`: Output directory for results and models (default: "./output")
- `--cache_dir`: Cache directory for processed datasets (default: "./cache")
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 1e-4)
- `--epochs`: Maximum number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: 5)
- `--hidden_dim`: Hidden dimension of GNN layers (default: 128)
- `--gnn_layers`: Number of GNN layers (default: 3)
- `--esm_proj_dim`: ESM projection dimension (default: 256)
- `--pos_encoding_dim`: Positional encoding dimension (default: 32)
- `--pu_loss`: PU loss type: "puauc", "nnpu", or "bce" (default: "puauc")
- `--task_loss_weight`: Weight for task loss (default: 1.0)
- `--node_loss_weight`: Weight for node separation loss (default: 0.3)
- `--contrastive_loss_weight`: Weight for graph contrastive loss (default: 0.1)
- `--early_metric`: Metric for early stopping: "aupr" or "auroc" (default: "aupr")
- `--seed`: Random seed (default: 42)
- `--device`: Device: "cuda" or "cpu" (default: "cuda")
- `--tag`: Experiment tag for output files (default: "pugrail")

**Example**
Run training with default parameters:
```bash
python -m pugrail.train --csv_path data/MyDataset/fold/index_all_folds.csv --pdb_dir data/MyDataset/pdb --esm_dir data/MyDataset/esm --output_dir results/MyDataset --tag exp1
```

Run training with custom parameters:
```bash
python -m pugrail.train --csv_path data/MyDataset/fold/index_all_folds.csv --pdb_dir data/MyDataset/pdb --esm_dir data/MyDataset/esm --output_dir results/MyDataset --batch_size 32 --lr 5e-5 --epochs 200 --patience 10 --hidden_dim 256 --gnn_layers 4 --pu_loss puauc --task_loss_weight 1.0 --node_loss_weight 0.5 --contrastive_loss_weight 0.2 --seed 123 --tag exp2_custom
```

### 4. Prediction on New Proteins

Run the `predict.py` script to predict antigenicity for new proteins:

```bash
python -m pugrail.predict --model_path results/MyDataset/models/pugrail_fold0_best.pt --pdb_path data/new_protein.pdb --esm_path data/new_protein.npy --output_path predictions/new_protein.json --hidden_dim 128 --gnn_layers 3 --threshold 0.5 --device cuda
```

#### Prediction Script

`pugrail/predict.py`: This script performs prediction on new proteins using a trained model.

**Usage**
```bash
# Single protein prediction
python -m pugrail.predict [--model_path MODEL_PATH] [--pdb_path PDB_PATH]
                          [--esm_path ESM_PATH] [--output_path OUTPUT_PATH]
                          [--hidden_dim HIDDEN_DIM] [--gnn_layers GNN_LAYERS]
                          [--threshold THRESHOLD] [--device DEVICE]

# Batch prediction
python -m pugrail.predict [--model_path MODEL_PATH] [--input_csv INPUT_CSV]
                          [--pdb_dir PDB_DIR] [--esm_dir ESM_DIR]
                          [--output_path OUTPUT_PATH] [--threshold THRESHOLD]
                          [--device DEVICE]
```

**Arguments**
- `--model_path`: Path to trained model checkpoint (required)
- `--pdb_path`: Path to PDB file for single protein prediction
- `--esm_path`: Path to ESM embedding file for single protein prediction
- `--input_csv`: CSV file with id_short column for batch prediction
- `--pdb_dir`: Directory containing PDB files for batch prediction
- `--esm_dir`: Directory containing ESM embeddings for batch prediction
- `--output_path`: Output path for predictions (default: "predictions.json")
- `--hidden_dim`: Hidden dimension (must match training) (default: 128)
- `--gnn_layers`: Number of GNN layers (must match training) (default: 3)
- `--threshold`: Classification threshold (default: 0.5)
- `--device`: Device: "cuda" or "cpu" (default: "cuda")

**Example**
Single protein prediction:
```bash
python -m pugrail.predict --model_path results/MyDataset/models/pugrail_fold0_best.pt --pdb_path data/query_protein.pdb --esm_path data/query_protein.npy --output_path predictions/query_result.json --threshold 0.5 --device cuda
```

Batch prediction:
```bash
python -m pugrail.predict --model_path results/MyDataset/models/pugrail_fold0_best.pt --input_csv data/query_proteins.csv --pdb_dir data/query_pdb --esm_dir data/query_esm --output_path predictions/batch_results.csv --threshold 0.5 --device cuda
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.3.0

### Install from source

```bash
git clone https://github.com/jaeminjj/PU-GRAIL.git
cd PU-GRAIL
pip install -r requirements.txt
pip install -e .
```

### Install dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy pandas scikit-learn biopython

# Optional: Install ESM for embedding generation
pip install fair-esm
```

## Model Architecture

```
Input: Protein sequence + Predicted 3D structure
       ↓
┌──────────────────────────────────────────┐
│  Node Features:                          │
│  - One-hot amino acid encoding (20-dim)  │
│  - ESM embedding (projected to 256-dim)  │
│  - Positional encoding (32-dim)          │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Graph Construction:                     │
│  - Sequential edges (i → i+1)            │
│  - Structure edges (Cα-Cα < 8Å)          │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Edge-Type Separation GNN:               │
│  - Sequence branch (GCN × 3 layers)      │
│  - Structure branch (GCN × 3 layers)     │
│  - Learnable α fusion                    │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Attention Pooling:                      │
│  - Gated attention weights               │
│  - Top-k residue aggregation             │
│  - Mean/Max pooling fusion               │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Prediction Head:                        │
│  - MLP classifier                        │
│  - Sigmoid activation                    │
└──────────────────────────────────────────┘
       ↓
Output: Protective antigen probability
```

## Benchmark Results

### Performance on PAPReC Benchmark

| Dataset | AUROC | AUPRC |
|---------|-------|-------|
| Bcipep | 0.82 | 0.96 |
| HLA | 0.75 | 0.44 |
| Protein | 0.96 | 0.96 |
| Epitope | 0.86 | 0.86 |

### Performance on ImmunoDB Benchmark

| Dataset | AUROC | AUPRC | F1 |
|---------|-------|-------|-----|
| Bacteria | 0.89 | 0.82 | 0.77 |
| Virus | 0.97 | 0.97 | 0.92 |
| Tumor | 0.87 | 0.81 | 0.77 |

## Citation

If you use PU-GRAIL in your research, please cite:

```bibtex
@article{jeon2025pugrail,
  title={PU-GRAIL: A Graph Neural Network Framework for Protective Antigen Prediction under Positive-Unlabeled Learning},
  author={Jeon, Jaemin and others},
  journal={Bioinformatics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Jaemin Jeon, jaeminjj@snu.ac.kr
