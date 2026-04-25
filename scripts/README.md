# scripts/

Training entry points for the lipophilicity prediction models. These scripts are the intended way to run model training; notebooks in `notebooks/` are reserved for evaluation and visualisation only.

## Contents

### `train_gnn.py`

Trains the GNN + ChemBERTa model defined in `src/gnn_model.py`. Accepts hyperparameters either as CLI flags or from a YAML config file; CLI flags override the file. Writes Lightning checkpoints to `--checkpoint-dir` (default `checkpoints/`) and saves a `metrics.json` summary alongside them.

**Usage**

```bash
# defaults (300-dim MPNN, 100 epochs, patience 20)
pixi run python scripts/train_gnn.py

# tune learning rate and batch size
pixi run python scripts/train_gnn.py --lr 5e-4 --batch-size 32 --max-epochs 200

# start from a YAML config, then override one key on the command line
pixi run python scripts/train_gnn.py --config configs/gnn_base.yaml --seed 1

# load a pretrained chemprop backbone checkpoint to initialise the MPNN weights
pixi run python scripts/train_gnn.py --backbone-checkpoint path/to/pretrained.ckpt
```

**Output**

- `checkpoints/gnn-best-{epoch}-{val_mae}.ckpt` — best checkpoint by validation MAE
- `checkpoints/metrics.json` — RMSE / MAE / R² for train, valid, and test splits

**YAML config format** (all keys are optional; missing keys fall back to defaults)

```yaml
d_h: 300
d_hidden: 256
depth: 3
dropout: 0.0
lr: 0.001
weight_decay: 0.0001
batch_size: 64
max_epochs: 100
patience: 20
seed: 42
```

## Dependencies on other modules

- `src/train_gnn.py` — `train_gnn()` function containing the full training pipeline
- `src/gnn_model.py` — `LipophilicityGNN` architecture
- `src/graph_data.py` — ChemBERTa encoder and chemprop dataset construction
- `src/data.py` — scaffold-split data loading
