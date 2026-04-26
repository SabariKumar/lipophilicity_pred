# src/

Shared Python library for the lipophilicity prediction pipeline. All model components, data utilities, and explainability tools live here and are imported by training scripts and evaluation notebooks.

---

## Module contents

### `data.py`

Loads and splits the Lipophilicity-AstraZeneca dataset from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

- **`load_lipophilicity()`** — returns the full dataset as a DataFrame with columns `Drug_ID`, `Drug` (SMILES), and `Y` (logD).
- **`get_splits(seed)`** — returns a scaffold split as `{"train": ..., "valid": ..., "test": ...}`. Scaffold splitting groups structurally similar molecules into the same split, which prevents the model from exploiting near-duplicate structures that happen to be on different sides of a random split. All downstream training and evaluation uses these splits; never re-split independently.

### `features.py`

Computes molecule-level feature matrices from SMILES strings for the descriptor-based baseline models.

- **`smiles_to_descriptors(smiles)`** — computes all ~200 RDKit descriptors (from `Descriptors._descList`) for each SMILES in a Series. Invalid SMILES produce NaN rows rather than raising or being dropped, so the output index always aligns with the input.
- **`smiles_to_fgs(smiles)`** — binary functional group fingerprint using RDKit's built-in hierarchy (see `utils.py`). Same NaN-preservation contract as above.
- **`get_mol_descriptors(mol)`** — per-molecule helper called internally by `smiles_to_descriptors`; wraps each descriptor in a try/except so a single failed calculation does not abort the whole molecule.

### `utils.py`

Functional group library management. The RDKit FG hierarchy is a nested tree; this module flattens it into a single dict and caches it at module level so the traversal only happens once per Python session.

- **`get_mol_fgs(mol)`** — returns a `{fg_name: 0_or_1}` dict for one molecule; called per-molecule from `smiles_to_fgs`.
- **`_get_fg_library()`** / **`_flatten_fgs(fgs)`** — private helpers that build and cache the flattened FG library. Do not call these directly.

### `graph_data.py`

Data layer for the GNN model. Bridges the HuggingFace Transformers ecosystem (ChemBERTa) with chemprop's graph data structures.

- **`ChemBertaEncoder`** — frozen `seyonec/ChemBERTa-zinc-base-v1` encoder. `encode(smiles)` returns `float32` arrays of `[CLS]` hidden states. Weights are never updated during training. Must be loaded with `use_safetensors=True` on torch < 2.6 (CVE-2025-32434 blocks `torch.load` otherwise).
- **`build_chemprop_dataset(smiles, targets, lm_embeddings)`** — creates a `chemprop.data.MoleculeDataset` where each datapoint's `x_d` field holds the precomputed ChemBERTa embedding. Embeddings are safe to precompute because the encoder is frozen. **Important:** chemprop's `TrainingBatch` exposes `x_d` as `batch.X_d`, not `batch.V_d`; `V_d` is always `None` for molecule-level extras in chemprop 2.2.x.

### `gnn_model.py`

Architecture for the GNN + ChemBERTa fusion model (Option A).

- **`ChempropBackbone`** — wraps chemprop's `BondMessagePassing` and returns per-atom hidden states of shape `(n_atoms_total, d_h)`. Default feature dims (`d_v=72`, `d_e=14`) match chemprop's `MultiHotAtomFeaturizer` and `MultiHotBondFeaturizer`. If a checkpoint path is provided, the message-passing weights are loaded from a full chemprop `MPNN` checkpoint and the aggregation + FFN head are discarded. The hidden dimension is inferred from `W_o.out_features` after loading.
- **`AttentionPooling`** — sigmoid-gated atom-to-molecule pooling. Uses sigmoid (not softmax) so that multiple atoms can simultaneously receive high weight; softmax would create zero-sum competition that suppresses attribution signals from co-contributing atoms.
- **`FusionMLP`** — `LayerNorm → Linear → ReLU → Dropout → Linear → scalar`. Takes the concatenation of the graph embedding and the ChemBERTa `[CLS]` vector as input.
- **`LipophilicityGNN`** — top-level module composing the three above. `forward(bmg, X_d)` accepts a `BatchMolGraph` and the batched ChemBERTa embeddings tensor; returns predicted logD of shape `(batch, 1)`.

### `train_gnn.py`

Training pipeline and utilities.

- **`evaluate_gnn(model, loader, device)`** — computes RMSE, MAE, and R² over a chemprop DataLoader. Mirrors the `evaluate()` signature in the baseline models branch for consistent cross-model reporting.
- **`GNNLitModule`** — PyTorch Lightning module wrapping `LipophilicityGNN`. Uses AdamW with cosine annealing LR. Logs `train_loss`, `train_mae`, `val_loss`, `val_mae` each step; early stopping monitors `val_mae`.
- **`train_gnn(config, checkpoint_dir)`** — end-to-end pipeline: loads data, encodes ChemBERTa embeddings once, builds chemprop datasets, trains with Lightning, loads best checkpoint, and returns the model and final metrics. Accepts a config dict; missing keys fall back to defaults.
- **`load_checkpoint(checkpoint_path, device, **model_kwargs)`** — restores a `LipophilicityGNN` from a Lightning `.ckpt` file. Strips the `"model."` prefix that Lightning adds to nested module state dict keys. Used by the evaluation notebook to load a trained model without re-running `train_gnn`.

### `explain.py`

Atom-level attribution for the trained GNN using [Captum](https://captum.ai/).

- **`AtomAttributionExplainer`** — wraps Captum's `LayerIntegratedGradients` targeting `backbone.mp.W_o`, the final per-atom linear projection in the chemprop MPNN. IG integrates the gradient of the model output w.r.t. this layer's activations along a path from zero to actual, giving each atom a signed importance score. Attribution is aggregated to a scalar per atom by taking the L2 norm over the hidden dimension.
- **`plot_atom_contributions(smiles, scores)`** — renders a 2D structure with atoms coloured by attribution score (red = positive logD contribution / hydrophobic; blue = negative / hydrophilic) using RDKit's `MolDraw2DSVG`. Scores are normalised symmetrically per molecule; the colour scale is not comparable across different molecules.

---

## Data contracts

| Boundary | Shape / type | Notes |
|---|---|---|
| `get_splits()` output | `dict[str, DataFrame]`, columns `Drug_ID`, `Drug`, `Y` | `Y` is logD (float) |
| `ChemBertaEncoder.encode()` output | `np.ndarray (n, 768)` float32 | `[CLS]` hidden state |
| `build_chemprop_dataset()` output | `MoleculeDataset` | `x_d` per point = 768-dim LM embedding |
| `TrainingBatch.X_d` | `Tensor (batch, 768)` | chemprop 2.2.x; field is `X_d`, **not** `V_d` |
| `LipophilicityGNN.forward()` output | `Tensor (batch, 1)` | predicted logD, unbounded |
| `AtomAttributionExplainer.explain()` output | `np.ndarray (n_heavy_atoms,)` float32 | L2-normed IG scores, non-negative |

---

## Critical parameters and constraints

- `ChempropBackbone._D_V = 72` and `_D_E = 14` must match the chemprop featurizers used at graph construction time. If a custom featurizer is passed to `build_dataloader`, these defaults will be wrong and the MPNN will silently produce garbage.
- The Captum attribution target layer is `backbone.mp.W_o`. If a future chemprop version renames this layer, attribution will fail at construction time. Inspect `list(model.backbone.mp.named_modules())` to verify.
- ChemBERTa must be loaded with `use_safetensors=True`; torch 2.4 raises on `torch.load` for non-safetensors files due to CVE-2025-32434.

---

## Dependencies between modules

```
data.py
  └── consumed by: features.py, graph_data.py (via train_gnn.py)

features.py
  └── depends on: utils.py

graph_data.py
  └── depends on: transformers (ChemBERTa), chemprop.data
  └── consumed by: train_gnn.py, explain.py

gnn_model.py
  └── depends on: chemprop.nn, chemprop.models
  └── consumed by: train_gnn.py, explain.py

train_gnn.py
  └── depends on: data.py, graph_data.py, gnn_model.py
  └── consumed by: scripts/train_gnn.py, notebooks/03_gnn_model.py

explain.py
  └── depends on: gnn_model.py, graph_data.py, captum, rdkit
  └── consumed by: notebooks/03_gnn_model.py
```
