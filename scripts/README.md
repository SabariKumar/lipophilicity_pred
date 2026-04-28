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

---

### `ensemble_gnn.py`

Trains an ensemble of independent `LipophilicityGNN` models, then aggregates their predictions into a per-molecule mean and std. Each member is an ordinary `train_gnn` run with a different random seed; after training, all checkpoints are loaded and run on the same data to produce the ensemble statistics.

**Usage**

```bash
# train 5 members (seeds 42–46) then aggregate
pixi run ensemble-gnn

# explicit seeds
pixi run ensemble-gnn --seeds 0 1 2 3 4

# skip training; aggregate from existing checkpoints
pixi run ensemble-gnn --aggregate-only --checkpoint-dir checkpoints/uq/ensemble
```

**Output** (written to `--checkpoint-dir`, default `checkpoints/uq/ensemble/`)

- `seed_<N>/` — one subdirectory per member, each containing a Lightning checkpoint and `metrics.json`
- `ensemble_predictions.csv` — test-split per-molecule DataFrame: `smiles`, `y_true`, `mean`, `std`, `pred_seed_<N>` columns
- `ensemble_metrics.json` — RMSE, MAE, R², and mean σ per split

The predictions CSV is the primary input to `evaluate_uq.py` and `notebooks/05_uq_analysis.py`.

---

### `evaluate_uq.py`

Loads a trained ensemble and a single reference checkpoint, then evaluates all three UQ methods — Deep Ensemble, Last-Layer Laplace, and Conformal Prediction — side by side on the test split. Prints a comparison table and writes figures and CSVs to `figs/`.

The conformal step uses the validation split for calibration so the test evaluation is uncontaminated. For Laplace, it fits the closed-form Gaussian posterior over the final linear layer from `src/uq.py`; no external UQ library is required.

**Usage**

```bash
# defaults: reads checkpoints/uq/ensemble/, uses seed_42 for Laplace
pixi run evaluate-uq

# explicit paths
pixi run evaluate-uq \
    --ensemble-dir checkpoints/uq/ensemble \
    --single-checkpoint checkpoints/uq/ensemble/seed_42/<run>.ckpt \
    --alpha 0.1
```

**Output**

- `figs/uq_comparison.csv` and `figs/uq_comparison.json` — RMSE, MAE, R², ECE, mean interval width, empirical coverage, Spearman ρ for each method
- `figs/uq_reliability.svg` — reliability diagrams (one panel per method)
- `figs/uq_comparison.svg` — four-panel bar chart of UQ metrics
- `checkpoints/uq/uq_test_predictions.csv` — per-molecule predictions for all three methods; consumed by `notebooks/05_uq_analysis.py`

---

### `pretrain_qm9.py`

CLI entry point for phase-1 QM9 multi-task pretraining of `SMILESTransformer`. Delegates to `src/pretrain_transformer.py:pretrain()`. Heavy imports (PyTorch, Lightning, transformers) are deferred behind the argument parse so `--help` is fast.

**Usage**

```bash
# all 12 QM9 targets (default)
pixi run pretrain-transformer

# ablation: electronic properties only
pixi run pretrain-transformer --targets homo lumo gap mu

# custom hyperparameters
pixi run pretrain-transformer --lr 1e-5 --max-epochs 20 --batch-size 64
```

**Output**

- `checkpoints/pretrain/<run>-pretrain-{epoch}-{val_loss}.ckpt` — best checkpoint by validation loss
- `checkpoints/pretrain/pretrain_metrics_history.csv` — per-epoch train/val loss

The best checkpoint path is printed at exit and is passed to `finetune_logd.py` as `--pretrained-checkpoint`.

---

### `finetune_logd.py`

CLI entry point for phase-2 logD fine-tuning of a pretrained (or fresh) `SMILESTransformer`. Delegates to `src/finetune_transformer.py:finetune()`. Can run without a pretrained checkpoint to produce a no-pretraining baseline.

**Usage**

```bash
# with pretrained backbone
pixi run finetune-transformer --pretrained-checkpoint checkpoints/pretrain/<run>.ckpt

# no-pretraining baseline (directly fine-tunes ChemBERTa on logD)
pixi run finetune-transformer

# custom hyperparameters
pixi run finetune-transformer --pretrained-checkpoint <path> --lr 5e-6 --max-epochs 30
```

**Output**

- `checkpoints/finetune/<run>-finetune-{epoch}-{val_mae}.ckpt` — best checkpoint by validation MAE
- `checkpoints/finetune/finetune_metrics.json` — RMSE, MAE, R² for train, valid, and test splits
- `checkpoints/finetune/finetune_metrics_history.csv` — per-epoch train/val loss

---

## Dependencies on other modules

- `src/train_gnn.py` — `train_gnn()`, `load_checkpoint()`
- `src/pretrain_transformer.py` — `pretrain()`, `load_pretrained_backbone()`
- `src/finetune_transformer.py` — `finetune()`, `load_finetuned_model()`
- `src/uq.py` — `fit_laplace()`, `predict_laplace()`, `conformal_calibrate()`, `conformal_predict()`, `compute_uq_metrics()`
- `src/gnn_model.py` — `LipophilicityGNN` architecture
- `src/graph_data.py` — ChemBERTa encoder and chemprop dataset construction
- `src/data.py` — scaffold-split data loading
