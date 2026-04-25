import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    from src.data import get_splits
    from src.explain import AtomAttributionExplainer, plot_atom_contributions
    from src.gnn_model import LipophilicityGNN
    from src.graph_data import ChemBertaEncoder, build_chemprop_dataset
    from src.train_gnn import evaluate_gnn, load_checkpoint

    return (
        AtomAttributionExplainer,
        ChemBertaEncoder,
        LipophilicityGNN,
        build_chemprop_dataset,
        evaluate_gnn,
        get_splits,
        load_checkpoint,
        mo,
        np,
        pd,
        plot_atom_contributions,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # GNN + ChemBERTa — Evaluation Dashboard

    Train the model first:
    ```
    pixi run train-gnn
    ```
    Then point the path below at the best checkpoint to explore results.
    """
    )
    return


@app.cell
def _(mo):
    ckpt_input = mo.ui.text(
        label="Checkpoint path",
        value="checkpoints/",
        placeholder="checkpoints/gnn-best-*.ckpt",
    )
    ckpt_input
    return (ckpt_input,)


@app.cell
def _(ckpt_input, load_checkpoint, torch):
    import glob as _glob

    _path = ckpt_input.value.strip()
    # Expand a directory to the best checkpoint inside it.
    if _path.endswith("/") or not _path.endswith(".ckpt"):
        _matches = sorted(_glob.glob(f"{_path}**/*.ckpt", recursive=True))
        _path = _matches[-1] if _matches else _path  # last = highest epoch number

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(_path, device=device)
    return device, model


@app.cell
def _(mo):
    mo.md("## Data")
    return


@app.cell
def _(ChemBertaEncoder, device, get_splits):
    splits = get_splits()
    encoder = ChemBertaEncoder().to(device)
    lm_embs = {
        name: encoder.encode(split["Drug"].tolist()) for name, split in splits.items()
    }
    return encoder, lm_embs, splits


@app.cell
def _(mo, splits):
    mo.md(
        f"""
    | Split | Molecules | Target mean ± std |
    |-------|-----------|-------------------|
    | train | {len(splits['train'])} | {splits['train']['Y'].mean():.2f} ± {splits['train']['Y'].std():.2f} |
    | valid | {len(splits['valid'])} | {splits['valid']['Y'].mean():.2f} ± {splits['valid']['Y'].std():.2f} |
    | test  | {len(splits['test'])}  | {splits['test']['Y'].mean():.2f} ± {splits['test']['Y'].std():.2f}  |
    """
    )
    return


@app.cell
def _(mo):
    mo.md("## Metrics")
    return


@app.cell
def _(build_chemprop_dataset, device, evaluate_gnn, lm_embs, mo, model, pd, splits):
    from chemprop.data import build_dataloader as _bdl

    _loaders = {
        name: _bdl(
            build_chemprop_dataset(
                splits[name]["Drug"],
                splits[name]["Y"].values,
                lm_embs[name],
            ),
            batch_size=64,
            shuffle=False,
        )
        for name in ("train", "valid", "test")
    }

    _rows = [
        {
            "split": split,
            "model": "GNN+ChemBERTa",
            **evaluate_gnn(model, loader, device),
        }
        for split, loader in _loaders.items()
    ]
    results_df = pd.DataFrame(_rows).round(3)
    mo.vstack([mo.md("### Per-split metrics"), mo.ui.table(results_df)])
    return (results_df,)


@app.cell
def _(mo):
    mo.md("## Parity Plot")
    return


@app.cell
def _(build_chemprop_dataset, device, lm_embs, mo, model, np, splits, torch):
    def _():
        import matplotlib.pyplot as plt
        from chemprop.data import build_dataloader as _dl
        from sklearn.metrics import r2_score

        _colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}
        fig, ax = plt.subplots(figsize=(6, 6))
        _all_vals = []

        for _split in ("train", "valid", "test"):
            _loader = _dl(
                build_chemprop_dataset(
                    splits[_split]["Drug"], splits[_split]["Y"].values, lm_embs[_split]
                ),
                batch_size=64,
                shuffle=False,
            )
            _preds, _targets = [], []
            model.eval()
            with torch.no_grad():
                for _batch in _loader:
                    _preds.append(
                        model(
                            _batch.bmg.to(device),
                            _batch.X_d.to(device) if _batch.X_d is not None else None,
                        )
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    _targets.append(_batch.Y.squeeze(-1).numpy())
            _y_pred = np.concatenate(_preds)
            _y_true = np.concatenate(_targets)
            _r2 = r2_score(_y_true, _y_pred)
            ax.scatter(
                _y_true,
                _y_pred,
                alpha=0.5,
                s=15,
                color=_colors[_split],
                label=f"{_split} (R²={_r2:.3f})",
            )
            _all_vals += list(_y_true) + list(_y_pred)

        _lo, _hi = min(_all_vals) - 0.3, max(_all_vals) + 0.3
        ax.plot([_lo, _hi], [_lo, _hi], "k--", linewidth=1, label="ideal")
        ax.set_xlim(_lo, _hi)
        ax.set_ylim(_lo, _hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Actual logD")
        ax.set_ylabel("Predicted logD")
        ax.set_title("GNN + ChemBERTa — Parity Plot")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return mo.vstack([fig])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Atom Attribution Gallery

    Six molecules spanning the logD range.
    **Red** = hydrophobic contribution (raises logD), **blue** = hydrophilic.
    """
    )
    return


@app.cell
def _(
    AtomAttributionExplainer,
    device,
    lm_embs,
    mo,
    model,
    plot_atom_contributions,
    splits,
):
    def _():
        import matplotlib.pyplot as plt

        explainer = AtomAttributionExplainer(model, device=device)
        _test = splits["test"].copy().reset_index(drop=True)
        _sorted = _test.sort_values("Y")
        _n = len(_sorted)
        _indices = [
            _sorted.index[0],
            _sorted.index[1],
            _sorted.index[_n // 2],
            _sorted.index[_n // 2 + 1],
            _sorted.index[-2],
            _sorted.index[-1],
        ]
        _labels = ["low", "low", "mid", "mid", "high", "high"]

        figs = []
        for _idx, _lbl in zip(_indices, _labels):
            _smi = _test.loc[_idx, "Drug"]
            _y = _test.loc[_idx, "Y"]
            _lm = lm_embs["test"][_idx]
            try:
                _scores = explainer.explain(_smi, _lm)
                _fig = plot_atom_contributions(
                    _smi, _scores, title=f"{_lbl} logD  |  actual={_y:.2f}"
                )
            except Exception as e:
                _fig, _ax = plt.subplots()
                _ax.text(
                    0.5, 0.5, f"Attribution failed:\n{e}", ha="center", va="center"
                )
                _ax.axis("off")
            figs.append(_fig)

        return mo.vstack([mo.hstack(figs[:3]), mo.hstack(figs[3:])])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## ChemBERTa Ablation

    Zeroing out `h_lm` measures the marginal contribution of the LLM context.
    """
    )
    return


@app.cell
def _(build_chemprop_dataset, device, lm_embs, mo, model, np, pd, splits, torch):
    from chemprop.data import build_dataloader as _bdl2
    from sklearn.metrics import mean_absolute_error as _mae
    from sklearn.metrics import mean_squared_error as _mse
    from sklearn.metrics import r2_score as _r2

    _ablation_rows = []
    for _split in ("valid", "test"):
        for _label, _lm_use in [
            ("GNN + ChemBERTa", lm_embs[_split]),
            ("GNN only (h_lm=0)", np.zeros_like(lm_embs[_split])),
        ]:
            _loader = _bdl2(
                build_chemprop_dataset(
                    splits[_split]["Drug"], splits[_split]["Y"].values, _lm_use
                ),
                batch_size=64,
                shuffle=False,
            )
            _preds, _targets = [], []
            model.eval()
            with torch.no_grad():
                for _batch in _loader:
                    _preds.append(
                        model(
                            _batch.bmg.to(device),
                            _batch.X_d.to(device) if _batch.X_d is not None else None,
                        )
                        .squeeze(-1)
                        .cpu()
                        .numpy()
                    )
                    _targets.append(_batch.Y.squeeze(-1).numpy())
            _y_pred = np.concatenate(_preds)
            _y_true = np.concatenate(_targets)
            _ablation_rows.append(
                {
                    "split": _split,
                    "model": _label,
                    "rmse": round(float(_mse(_y_true, _y_pred) ** 0.5), 3),
                    "mae": round(float(_mae(_y_true, _y_pred)), 3),
                    "r2": round(float(_r2(_y_true, _y_pred)), 3),
                }
            )

    ablation_df = pd.DataFrame(_ablation_rows)
    mo.vstack([mo.md("### Ablation results"), mo.ui.table(ablation_df)])
    return (ablation_df,)


if __name__ == "__main__":
    app.run()
