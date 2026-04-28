# src/

Shared Python library for the lipophilicity prediction pipeline. All model components, data utilities, and explainability tools live here and are imported by training scripts and evaluation notebooks.

---

## Module contents

### `data.py`

Loads and splits the Lipophilicity-AstraZeneca dataset from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

- **`load_lipophilicity()`** — returns the full dataset as a DataFrame with columns `Drug_ID`, `Drug` (SMILES), and `Y` (logD).
- **`get_splits(seed)`** — stratified Murcko scaffold split. Scaffold groups are binned by median logD into `n_strata` quantile strata, then split at the group level so each of train/valid/test sees a representative slice of the logD range. This is the default split for all training and evaluation.
- **`get_tdc_split(seed)`** — the original TDC scaffold split, without logD-distribution balancing. Used as an uncontrolled baseline and for the transformer fine-tuning task (following the original benchmark setup).
- **`get_random_split(seed)`** — purely random split; no scaffold awareness. Useful for quantifying the contribution of scaffold-aware splitting.

The three split strategies are compared side-by-side in `notebooks/01_descriptors.py`.

### `features.py`

Computes molecule-level feature matrices from SMILES strings for the descriptor-based baseline models.

- **`smiles_to_descriptors(smiles)`** — computes all RDKit descriptors (from `Descriptors._descList`) for each SMILES in a Series. Invalid SMILES produce NaN rows rather than raising or being dropped, so the output index always aligns with the input. Descriptors in `_LEAKY_DESCRIPTORS` (`MolLogP`) are dropped from the output to prevent target leakage.
- **`smiles_to_fgs(smiles)`** — binary functional group fingerprint using RDKit's built-in hierarchy (see `utils.py`). Same NaN-preservation contract as above.
- **`get_mol_descriptors(mol)`** — per-molecule helper called by `smiles_to_descriptors`; wraps each descriptor in a try/except so a single failed calculation does not abort the whole molecule.
- **`_LEAKY_DESCRIPTORS`** — frozenset of descriptor names excluded from `smiles_to_descriptors`. Currently `{"MolLogP"}` — logP is excluded because logD ≈ logP + ionisation correction, so including it would allow models to trivially recover the target.

### `preprocessing.py`

Descriptor preprocessing pipeline for the baseline models.

- **`build_preprocessor(X_train)`** — fits a two-step sklearn Pipeline on training descriptors: (1) `VarianceThreshold(threshold=0.01)` removes near-constant features (approximately 35 out of ~217), (2) `SimpleImputer(strategy="median")` fills remaining NaNs with train-split medians. Fit only on training data; applied to valid and test via `apply_preprocessor`.
- **`apply_preprocessor(pipe, X)`** — transforms a descriptor DataFrame through a fitted preprocessor and returns both the numpy array and the surviving feature names. Feature names are recovered by applying the variance filter's support mask to the original column list.

### `models.py`

Scikit-learn baseline models for descriptor-based logD prediction.

- **`fit_lasso(X, y)`** — fits a `StandardScaler → LassoCV` pipeline with 5-fold cross-validation to select the L1 regularisation strength. Returns the fitted pipeline.
- **`fit_rf(X, y)`** — fits a Random Forest regressor via `RandomizedSearchCV` (n_iter=20, 5-fold CV) over `max_depth`, `min_samples_leaf`, and `max_features`. Parallelises the search across folds rather than the trees to avoid nested parallelism. Returns the best estimator. Prints the best hyperparameters and CV RMSE.
- **`evaluate(model, X, y)`** — computes RMSE, MAE, and R² for any fitted sklearn model or Pipeline.

### `plot_utils.py`

Figure persistence for notebooks and scripts.

- **`save_fig(fig, name, figs_dir)`** — saves a matplotlib figure as an SVG file in `figs/`. Uses `svg.fonttype='none'` so text elements remain editable in Inkscape rather than being converted to paths. Creates `figs_dir` if it does not exist.

### `utils.py`

Functional group library management. The RDKit FG hierarchy is a nested tree; this module flattens it into a single dict and caches it at module level so the traversal only happens once per Python session.

- **`get_mol_fgs(mol)`** — returns a `{fg_name: 0_or_1}` dict for one molecule; called per-molecule from `smiles_to_fgs`.
- **`_get_fg_library()`** / **`_flatten_fgs(fgs)`** — private helpers that build and cache the flattened FG library.

### `graph_data.py`

Data layer for the GNN model. Bridges the HuggingFace Transformers ecosystem (ChemBERTa) with chemprop's graph data structures.

- **`ChemBertaEncoder`** — frozen `seyonec/ChemBERTa-zinc-base-v1` encoder. `encode(smiles)` returns `float32` arrays of `[CLS]` hidden states. Weights are never updated during training. Must be loaded with `use_safetensors=True` on torch < 2.6 (CVE-2025-32434 blocks `torch.load` otherwise).
- **`build_chemprop_dataset(smiles, targets, lm_embeddings)`** — creates a `chemprop.data.MoleculeDataset` where each datapoint's `x_d` field holds the precomputed ChemBERTa embedding. Embeddings are safe to precompute because the encoder is frozen. **Important:** chemprop's `TrainingBatch` exposes `x_d` as `batch.X_d`, not `batch.V_d`; `V_d` is always `None` for molecule-level extras in chemprop 2.2.x.

### `gnn_model.py`

Architecture for the GNN + ChemBERTa fusion model.

- **`ChempropBackbone`** — wraps chemprop's `BondMessagePassing` and returns per-atom hidden states of shape `(n_atoms_total, d_h)`. Default feature dims (`d_v=72`, `d_e=14`) match chemprop's `MultiHotAtomFeaturizer` and `MultiHotBondFeaturizer`. If a checkpoint path is provided, message-passing weights are loaded from a full chemprop `MPNN` checkpoint and the aggregation + FFN head are discarded. The hidden dimension is inferred from `W_o.out_features` after loading.
- **`AttentionPooling`** — sigmoid-gated atom-to-molecule pooling. Uses sigmoid (not softmax) so multiple atoms can simultaneously receive high weight; softmax would create zero-sum competition that suppresses attribution signals from co-contributing atoms.
- **`FusionMLP`** — `LayerNorm → Linear → ReLU → Dropout → Linear → scalar`. Takes the concatenation of the graph embedding and the ChemBERTa `[CLS]` vector as input.
- **`LipophilicityGNN`** — top-level module composing the three above. `forward(bmg, X_d)` accepts a `BatchMolGraph` and the batched ChemBERTa embeddings tensor; returns predicted logD of shape `(batch, 1)`.

### `train_gnn.py`

Training pipeline and utilities for `LipophilicityGNN`.

- **`evaluate_gnn(model, loader, device)`** — computes RMSE, MAE, and R² over a chemprop DataLoader.
- **`GNNLitModule`** — PyTorch Lightning wrapper. Uses AdamW + cosine annealing LR. Embeds `model_kwargs` and `split` in the checkpoint via `on_save_checkpoint` so `load_checkpoint` can reconstruct the architecture without additional arguments.
- **`train_gnn(config, checkpoint_dir)`** — end-to-end pipeline: loads data, encodes ChemBERTa embeddings once, builds chemprop datasets, trains with Lightning, loads best checkpoint, and returns the model and final metrics. Accepts a config dict; missing keys fall back to defaults.
- **`load_checkpoint(checkpoint_path, device)`** — restores a `LipophilicityGNN` from a Lightning `.ckpt` file. Strips the `"model."` prefix that Lightning adds to nested module state dict keys. Falls back to inferring `d_h`/`d_lm`/`d_hidden` from weight shapes for checkpoints that pre-date the embedded `model_kwargs` feature.

### `explain.py`

Atom-level attribution for the trained GNN using [Captum](https://captum.ai/).

- **`AtomAttributionExplainer`** — applies Captum's `IntegratedGradients` to the atom feature matrix `bmg.V` with a zero-vector baseline. Attribution is the sum over the feature dimension (not L2 norm), so the sign is preserved: positive = atom pushes logD up (hydrophobic), negative = atom pulls logD down (hydrophilic).
- **`plot_atom_contributions(smiles, scores)`** — renders a 2D structure with atoms coloured by attribution score (red = positive / hydrophobic; blue = negative / hydrophilic) using RDKit's `MolDraw2DCairo`. Scores are normalised to [-1, 1] within each molecule; the colour scale is not comparable across different molecules.

### `transformer_model.py`

ChemBERTa-based transformer architecture for the two-phase SMILES Transformer pipeline.

- **`SMILESTransformer`** — wraps the ChemBERTa RoBERTa backbone (loaded via `AutoModelForMaskedLM` to correctly resolve weight-tied embeddings, then the LM head is discarded) with a swappable regression head. Both backbone and head are unfrozen during training. `swap_head(new_head)` replaces the head in-place between pretraining and fine-tuning.
- **`QM9PretrainHead`** — single `Linear(768 → n_targets)` mapping `[CLS]` to normalised QM9 values. Stateless between pretraining and fine-tuning; discarded when the head is swapped.
- **`LogDFinetuneHead`** — `LayerNorm(768) → Linear(768→256) → ReLU → Dropout → Linear(256→1)`, matching the FusionMLP structure in `gnn_model.py` for fair comparison.
- **`tokenize(smiles, max_length)`** — tokenises a list of SMILES using the ChemBERTa tokenizer; returns `input_ids` and `attention_mask` tensors.

### `pretrain_transformer.py`

Phase-1 QM9 multi-task pretraining of `SMILESTransformer`.

- **`QM9Dataset`** — tokenises SMILES once at construction and stores normalised targets. Tokenisation is expensive and deterministic, so doing it in `__init__` rather than `__getitem__` avoids repeated work across epochs.
- **`PretrainLitModule`** — Lightning wrapper. Uses AdamW with linear warmup + cosine decay (lr=2e-5, warmup_steps=500). Logs per-target MAE for each QM9 property in wandb. Embeds the target list and normaliser state in the checkpoint for self-contained loading.
- **`pretrain(config, checkpoint_dir)`** — runs the full pretraining loop on QM9 and returns the path to the best checkpoint.
- **`load_pretrained_backbone(checkpoint_path, device)`** — restores the pretrained model, target list, and normaliser from a checkpoint. The returned model has no head attached; call `swap_head()` before fine-tuning.

### `finetune_transformer.py`

Phase-2 logD fine-tuning of a pretrained (or fresh) `SMILESTransformer`.

- **`LogDDataset`** — tokenises SMILES once at construction and stores raw logD targets. Uses the TDC scaffold split by default to match the original benchmark.
- **`FinetuneLitModule`** — Lightning wrapper. Uses AdamW with linear warmup + cosine decay (lr=1e-5, lower than pretraining to avoid catastrophic forgetting). Embeds head kwargs and the provenance of the pretrained checkpoint for traceability.
- **`finetune(config, checkpoint_dir)`** — loads an optional phase-1 checkpoint, swaps in the `LogDFinetuneHead`, and trains end-to-end. Works without a pretrained checkpoint as a no-pretraining baseline.
- **`evaluate_transformer(model, loader, device)`** — computes RMSE, MAE, R² for a fine-tuned model.
- **`load_finetuned_model(checkpoint_path, device)`** — restores a fine-tuned model from a Lightning checkpoint.

### `uq.py`

Uncertainty quantification methods for `LipophilicityGNN`. All three methods are implemented without external UQ libraries to avoid dependency conflicts with chemprop.

- **`extract_fused_features(model, loader, device)`** — runs the backbone + pooling + concatenation for all batches without the FusionMLP head, returning `(N, d_h+d_lm)` fused features. Used as the feature extractor for the Laplace approximation.
- **`fit_laplace(model, train_loader, device, prior_precision)`** — fits a closed-form Gaussian posterior over the weights of the final `Linear(d_hidden→1)` layer via the generalised Gauss-Newton approximation: `Λ = F^T F / σ² + λI`. The `prior_precision` matches the `weight_decay` used in `train_gnn` (1e-4) for consistency. Returns `{"Sigma": posterior_covariance, "noise_var": float}`.
- **`predict_laplace(la, model, loader, device)`** — returns `(mean, std)` where `std = sqrt(x^T Σ x + σ²)` decomposes into epistemic (last-layer uncertainty) and aleatoric (residual noise) components.
- **`conformal_calibrate(y_true, y_pred, alpha)`** — computes the finite-sample corrected quantile `⌈(n+1)(1-α)/n⌉` of calibration-set `|residuals|`, guaranteeing at least `1-α` empirical coverage in expectation.
- **`conformal_predict(y_pred, q)`** — applies the symmetric conformal interval `ŷ ± q`.
- **`compute_uq_metrics(y_true, y_pred, y_std, alpha)`** — computes RMSE, MAE, R², ECE (10 equal-mass bins), mean interval width, empirical coverage, and Spearman ρ between predicted uncertainty and absolute error.

---

## Data contracts

| Boundary | Shape / type | Notes |
|---|---|---|
| `get_splits()` output | `dict[str, DataFrame]`, columns `Drug_ID`, `Drug`, `Y` | `Y` is logD (float) |
| `smiles_to_descriptors()` output | `DataFrame (n, ~182)` | MolLogP excluded; NaNs present before preprocessing |
| `build_preprocessor()` threshold | variance > 0.01 | removes ~35 near-constant descriptors |
| `ChemBertaEncoder.encode()` output | `np.ndarray (n, 768)` float32 | `[CLS]` hidden state |
| `build_chemprop_dataset()` output | `MoleculeDataset` | `x_d` per point = 768-dim LM embedding |
| `TrainingBatch.X_d` | `Tensor (batch, 768)` | chemprop 2.2.x; field is `X_d`, **not** `V_d` |
| `LipophilicityGNN.forward()` output | `Tensor (batch, 1)` | predicted logD, unbounded |
| `AtomAttributionExplainer.explain()` output | `np.ndarray (n_heavy_atoms,)` float32 | signed IG scores summed over feature dim |
| `fit_laplace()` output | `dict` with `Sigma (d+1, d+1)` and `noise_var` | `d = d_hidden = 256` |

---

## Critical parameters and constraints

- `ChempropBackbone._D_V = 72` and `_D_E = 14` must match the chemprop featurizers used at graph construction time. Mismatches cause silent garbage output.
- `VarianceThreshold(threshold=0.01)` is fit on training data only; applying it to test data uses the training mask stored in `pipe.named_steps["variance"]`.
- `_LEAKY_DESCRIPTORS = {"MolLogP"}` is applied in `smiles_to_descriptors` before any train/test split, so it affects all splits consistently.
- `prior_precision=1e-4` in `fit_laplace` must match the `weight_decay` used when training the model. Mismatch will bias the Laplace posterior.
- ChemBERTa must be loaded with `use_safetensors=True`; torch 2.4 raises on `torch.load` for non-safetensors files (CVE-2025-32434).

---

## Dependencies between modules

```
data.py
  └── consumed by: features.py, graph_data.py (via train_gnn.py), finetune_transformer.py

features.py
  └── depends on: utils.py
  └── consumed by: preprocessing.py (indirectly), notebooks/02_baseline_models.py

preprocessing.py
  └── consumed by: notebooks/02_baseline_models.py

models.py
  └── consumed by: notebooks/02_baseline_models.py

plot_utils.py
  └── consumed by: all notebooks

graph_data.py
  └── depends on: transformers (ChemBERTa), chemprop.data
  └── consumed by: train_gnn.py, explain.py, uq.py

gnn_model.py
  └── depends on: chemprop.nn, chemprop.models
  └── consumed by: train_gnn.py, explain.py, uq.py

train_gnn.py
  └── depends on: data.py, graph_data.py, gnn_model.py
  └── consumed by: scripts/train_gnn.py, scripts/ensemble_gnn.py, notebooks/03_gnn_model.py

explain.py
  └── depends on: gnn_model.py, graph_data.py, captum, rdkit
  └── consumed by: notebooks/03_gnn_model.py, notebooks/05_uq_analysis.py

uq.py
  └── depends on: gnn_model.py, graph_data.py
  └── consumed by: scripts/evaluate_uq.py, notebooks/05_uq_analysis.py

transformer_model.py
  └── depends on: transformers (ChemBERTa)
  └── consumed by: pretrain_transformer.py, finetune_transformer.py

pretrain_transformer.py
  └── depends on: transformer_model.py, qm9_data.py
  └── consumed by: scripts/pretrain_qm9.py, finetune_transformer.py

finetune_transformer.py
  └── depends on: transformer_model.py, pretrain_transformer.py, data.py
  └── consumed by: scripts/finetune_logd.py, notebooks/04_transformer.py
```
