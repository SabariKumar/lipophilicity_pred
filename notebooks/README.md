# notebooks/

Interactive [Marimo](https://marimo.io) notebooks for data exploration and model evaluation. Notebooks are dashboards only — they visualise results produced by training scripts, they do not run training themselves.

Run any notebook with:

```bash
pixi run python -m marimo run notebooks/<name>.py
```

---

## Contents

### `01_descriptors.py`

Exploratory analysis of the raw dataset and RDKit descriptor space. Intended to be run once before model development to understand the data.

- Loads the full Lipophilicity-AstraZeneca dataset and displays summary statistics.
- Shows the logD target distribution and the fraction of molecules for which each descriptor could be computed.
- Plots per-descriptor variance on a log scale to illustrate how many descriptors are near-zero-variance and would be removed by preprocessing.
- Plots functional group prevalence across the dataset (via `smiles_to_fgs`).
- Shows the ClogP (MolLogP) distribution as a sanity check against the logD target.

This notebook has no outputs to persist; re-run it whenever the dataset changes.

### `02_baseline_models.py`

Training and evaluation dashboard for the Lasso and Random Forest descriptor baselines. Unlike the GNN notebooks, this one *does* run training — the models are cheap enough that fitting inside a reactive notebook is practical.

1. **Data and descriptors** — loads the stratified scaffold split and computes RDKit descriptors via `smiles_to_descriptors`. Displays a split summary table showing molecule counts and logD mean ± std per split. `MolLogP` is excluded from descriptors at source to prevent target leakage.
2. **Preprocessing** — fits a `VarianceThreshold(0.01) → SimpleImputer(median)` pipeline on the training split, reducing ~217 raw descriptors to ~182.
3. **Model training** — fits both models on the training split and displays test-split RMSE, MAE, and R² in a summary table. The RF uses `RandomizedSearchCV` (20 iterations, 5-fold CV) to tune `max_depth`, `min_samples_leaf`, and `max_features`.
4. **Parity plots** — a model selector (Lasso / RF) controls which parity plot is shown. Both are computed and saved as `figs/02_parity_lasso.svg` and `figs/02_parity_rf.svg`.
5. **SHAP beeswarms** — SHAP values computed via `shap.LinearExplainer` (Lasso) and `shap.TreeExplainer` (RF). Top-20 feature importance beeswarm plots saved as `figs/02_shap_lasso.svg` and `figs/02_shap_rf.svg`.
6. **SHAP consensus analysis** — identifies the 3 exact overlapping features across both models' top-20 SHAP rankings (`fr_COO`, `NumAromaticRings`, `fr_halogen`) and shows VSA descriptor family consensus (SMR_VSA, PEOE_VSA, SlogP_VSA appear in both models). Saved as `figs/02_shap_family_consensus.svg`.

---

### `03_gnn_model.py`

Evaluation dashboard for the trained GNN + ChemBERTa model. **Requires a trained checkpoint** — run `pixi run train-gnn` first.

Point the checkpoint path input at a `.ckpt` file (or a directory; the notebook will resolve the most recent checkpoint automatically). The notebook then:

1. **Loads** the model from the checkpoint via `load_checkpoint`.
2. **Encodes** ChemBERTa embeddings for all three splits (train/valid/test) using the frozen encoder.
3. **Metrics table** — RMSE, MAE, and R² per split.
4. **Parity plot** — predicted vs. actual logD for all splits overlaid, coloured by split.
5. **Atom attribution gallery** — six test-set molecules (two low, two mid, two high logD) drawn with per-atom Captum IntegratedGradients scores: red = hydrophobic contribution, blue = hydrophilic.
6. **ChemBERTa ablation** — validation and test metrics with the ChemBERTa embedding zeroed out, quantifying the marginal accuracy contribution of the LLM context vector.

### `04_transformer.py`

Evaluation dashboard for the two-phase SMILES Transformer pipeline. **Requires** both a pretrained and a fine-tuned checkpoint — run `pixi run pretrain-transformer` then `pixi run finetune-transformer` first.

The notebook accepts up to five checkpoint paths simultaneously, one per pretraining condition (All 12, Electronic, Structural, Thermodynamic, None / baseline). All active checkpoints are evaluated in one pass so conditions can be compared directly.

1. **Per-split metrics table** — RMSE, MAE, and R² per condition and split, displayed as a sortable `mo.ui.table`.
2. **Parity plots by pretraining condition** — one panel per loaded checkpoint. Each panel uses a marginal KDE along both axes to show the predicted vs. actual logD distribution per split. Saved as `figs/04_transformer_parity.svg`.
3. **Training curves** — reads `checkpoints/pretrain/pretrain_metrics_history.csv` and `checkpoints/finetune/finetune_metrics_history.csv` if present, plots MSE loss vs. epoch with a 5-epoch rolling mean. Saved as `figs/04_transformer_pretrain_curves.svg` and `figs/04_transformer_finetune_curves.svg`.

---

### `05_uq_analysis.py`

Analysis notebook for the three UQ methods and their chemical interpretation. **Requires** `pixi run ensemble-gnn` and `pixi run evaluate-uq` to have completed first. Reads `checkpoints/uq/uq_test_predictions.csv` and `figs/uq_comparison.json`.

The notebook is structured in six sections:

1. **UQ method comparison** — reproduces reliability diagrams (one per method: Ensemble, Laplace, Conformal) and a four-panel bar chart of ECE, mean interval width, empirical coverage, and Spearman ρ. Saved as `figs/uq_reliability.svg` and `figs/uq_comparison.svg`.

2. **Outlier analysis** — divides the test set into four quadrants by prediction error (|error| > 1.0 logD) and ensemble std (above/below the median std of inliers). The scatter plot of error vs. uncertainty colour-codes all four quadrants. Saved as `figs/uq_scatter.svg`.

3. **Chemical feature analysis** — computes 10 RDKit descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds, FractionCSP3, AromaticRings, Rings, Heteroatoms) for every test molecule and plots violin distributions per quadrant. Highlights which structural classes are over-represented among outliers. Saved as `figs/uq_descriptors.svg`.

4. **OOD analysis** — computes Morgan fingerprint (radius 2, 2048 bits) Tanimoto similarity to the nearest training-set neighbour for each test molecule. Lower similarity = more out-of-distribution; this section checks whether high-uncertainty outliers are also chemically novel. Saved as `figs/uq_tanimoto.svg`.

5. **Functional group analysis** — compares `fr_` fragment prevalence between the Outlier+High σ group (n≈10) and the Inlier+Low σ group (n≈415). Fragments present in ≥30% of outliers but ≤5% of inliers (or vice versa) are surfaced in a sortable table.

6. **Atom attribution for outliers** — applies Captum IntegratedGradients via `AtomAttributionExplainer` to all 11 outlier molecules using the seed-42 model, rendering 2D structures with per-atom signed IG scores. Saved as `figs/uq_attribution_<N>.svg` per molecule.

---

## Design note

Notebooks deliberately contain no training code. Mixing training into a reactive notebook creates two problems: the reactive execution model can re-trigger expensive operations on unrelated cell edits, and training state (random seeds, shuffled batches) is harder to reproduce than a script invocation. All training lives in `scripts/` and `src/train_gnn.py`.
