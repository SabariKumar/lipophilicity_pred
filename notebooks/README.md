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

### `03_gnn_model.py`

Evaluation dashboard for the trained GNN + ChemBERTa model. **Requires a trained checkpoint** — run `pixi run train-gnn` first.

Point the checkpoint path input at a `.ckpt` file (or a directory; the notebook will resolve the most recent checkpoint automatically). The notebook then:

1. **Loads** the model from the checkpoint via `load_checkpoint`.
2. **Encodes** ChemBERTa embeddings for all three splits (train/valid/test) using the frozen encoder.
3. **Metrics table** — RMSE, MAE, and R² per split.
4. **Parity plot** — predicted vs. actual logD for all splits overlaid, coloured by split.
5. **Atom attribution gallery** — six test-set molecules (two low, two mid, two high logD) drawn with per-atom Captum IntegratedGradients scores: red = hydrophobic contribution, blue = hydrophilic.
6. **ChemBERTa ablation** — validation and test metrics with the ChemBERTa embedding zeroed out, quantifying the marginal accuracy contribution of the LLM context vector.

---

## Design note

Notebooks deliberately contain no training code. Mixing training into a reactive notebook creates two problems: the reactive execution model can re-trigger expensive operations on unrelated cell edits, and training state (random seeds, shuffled batches) is harder to reproduce than a script invocation. All training lives in `scripts/` and `src/train_gnn.py`.
