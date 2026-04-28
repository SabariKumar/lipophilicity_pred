# lipophilicity_pred

Lipophilicity (logD at pH 7.4) prediction on the [Therapeutics Data Commons](https://tdcommons.ai/) Lipophilicity-AstraZeneca dataset (~4,200 drug-like molecules). The project benchmarks four model families and compares three uncertainty quantification methods.

---

## Models

| Model | Description |
|-------|-------------|
| Lasso | ElasticNet with L1 regularisation on RDKit descriptors; interpretable via SHAP |
| Random Forest | Ensemble of decision trees on RDKit descriptors; tuned via 5-fold RandomizedSearchCV |
| LipophilicityGNN | Chemprop MPNN (d_h=300) + sigmoid attention pooling fused with a frozen ChemBERTa [CLS] embedding; end-to-end fine-tuning on logD |
| SMILESTransformer | ChemBERTa backbone pretrained on 12 QM9 quantum-chemical targets, then fine-tuned on logD with a two-layer MLP head |

Descriptor-based models exclude `MolLogP` (RDKit's computed logP) to prevent target leakage, since logD ≈ logP + ionisation correction.

## Uncertainty quantification

Three UQ methods are implemented for the GNN and compared on the test split:

- **Deep Ensemble** — five independently trained GNN members; mean/std of their predictions
- **Last-Layer Laplace** — closed-form Gaussian posterior over the final Linear layer; no external library
- **Conformal Prediction** — distribution-free coverage guarantee using validation-set residual quantiles

## Dataset and splits

The default split is a **stratified Murcko scaffold split**: molecules are grouped by scaffold, scaffold groups are binned by median logD, and a stratified shuffle assigns groups to train/valid/test (80/10/10) so each split sees a representative slice of the logD range.

## Directory structure

```
src/          shared library (models, data, features, training, UQ)
scripts/      CLI training entry points
notebooks/    Marimo interactive analysis notebooks
figs/         SVG figures produced by notebooks
checkpoints/  Lightning checkpoints and metrics
data/         cached TDC downloads
docs/         design documents
```

## Quick start

```bash
pixi install
pixi run setup-dgl     # install DGL wheel for your CUDA version
pixi run setup-tdc     # install PyTDC without conflicting deps

# Descriptor baselines (notebook)
pixi run notebook-eda

# GNN training
pixi run train-gnn

# Ensemble + UQ evaluation
pixi run ensemble-gnn
pixi run evaluate-uq

# Transformer pretraining + fine-tuning
pixi run pretrain-transformer
pixi run finetune-transformer --pretrained-checkpoint checkpoints/pretrain/<run>.ckpt

# Tests
pixi run test
```
