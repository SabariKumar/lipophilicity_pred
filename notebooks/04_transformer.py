import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    from src.finetune_transformer import (
        LogDDataset,
        evaluate_transformer,
        load_finetuned_model,
    )
    from src.plot_utils import save_fig
    from src.qm9_data import QM9_TARGETS

    return (
        LogDDataset,
        QM9_TARGETS,
        evaluate_transformer,
        load_finetuned_model,
        mo,
        np,
        pd,
        save_fig,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # SMILES Transformer — Evaluation Dashboard

    Run pretraining then fine-tuning first:
    ```
    pixi run pretrain-transformer
    pixi run finetune-transformer --pretrained-checkpoint checkpoints/pretrain/<run>.ckpt
    ```
    Point the paths below at the fine-tuned checkpoints to explore results.
    """
    )
    return


@app.cell
def _(mo, QM9_TARGETS):
    from src.qm9_data import QM9_TARGETS as _ALL

    _group_options = {
        "All 12": _ALL,
        "Electronic": ["homo", "lumo", "gap", "mu"],
        "Structural": ["alpha", "r2"],
        "Thermodynamic": ["u0", "u298", "h298", "g298", "cv", "zpve"],
        "None (baseline)": [],
    }

    ablation_inputs = {
        label: mo.ui.text(
            label=f"{label} checkpoint",
            placeholder="checkpoints/finetune/<run>.ckpt",
        )
        for label in _group_options
    }
    mo.vstack(list(ablation_inputs.values()))
    return ablation_inputs, _group_options


@app.cell
def _(mo):
    mo.md("## Per-Split Metrics")
    return


@app.cell
def _(LogDDataset, ablation_inputs, evaluate_transformer, mo, pd, torch):
    def _():
        from torch.utils.data import DataLoader

        from src.data import get_tdc_split

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        splits = get_tdc_split()

        rows = []
        for label, inp in ablation_inputs.items():
            path = inp.value.strip()
            if not path or not path.endswith(".ckpt"):
                continue
            try:
                from src.finetune_transformer import load_finetuned_model as _load

                model = _load(path, device=device)
            except Exception as e:
                rows.append({"condition": label, "split": "—", "error": str(e)})
                continue
            for split_name in ("train", "valid", "test"):
                loader = DataLoader(
                    LogDDataset(splits[split_name]),
                    batch_size=64,
                    shuffle=False,
                )
                m = evaluate_transformer(model, loader, device)
                rows.append({"condition": label, "split": split_name, **m})

        if not rows:
            return mo.callout(
                mo.md("No checkpoints loaded yet. Fill in at least one path above."),
                kind="warn",
            )
        df = pd.DataFrame(rows).round(3)
        return mo.vstack([mo.md("### Ablation metrics"), mo.ui.table(df)])

    _()
    return


@app.cell
def _(mo):
    mo.md("## Parity Plots by Pretraining Condition")
    return


@app.cell
def _(LogDDataset, ablation_inputs, mo, np, save_fig, torch):
    def _():
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy.stats import gaussian_kde
        from sklearn.metrics import r2_score
        from torch.utils.data import DataLoader

        from src.data import get_tdc_split
        from src.finetune_transformer import load_finetuned_model as _load

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        splits = get_tdc_split()
        colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}

        entries = [
            (label, inp.value.strip())
            for label, inp in ablation_inputs.items()
            if inp.value.strip().endswith(".ckpt")
        ]
        if not entries:
            return mo.callout(mo.md("No checkpoints loaded."), kind="warn")

        fig, main_axes = plt.subplots(1, len(entries), figsize=(6 * len(entries), 5))
        if len(entries) == 1:
            main_axes = [main_axes]

        for ax, (label, ckpt_path) in zip(main_axes, entries):
            divider = make_axes_locatable(ax)
            ax_top = divider.append_axes("top", size="18%", pad=0.05, sharex=ax)
            ax_right = divider.append_axes("right", size="18%", pad=0.05, sharey=ax)

            try:
                model = _load(ckpt_path, device=device)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Load failed:\n{e}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax_top.set_title(label, fontsize=9)
                continue

            _split_data = {}
            _all_vals = []
            for _split in ("train", "valid", "test"):
                loader = DataLoader(
                    LogDDataset(splits[_split]),
                    batch_size=64,
                    shuffle=False,
                )
                _preds, _targets = [], []
                model.eval()
                with torch.no_grad():
                    for batch in loader:
                        out = (
                            model(
                                batch["input_ids"].to(device),
                                batch["attention_mask"].to(device),
                            )
                            .squeeze(-1)
                            .cpu()
                            .numpy()
                        )
                        _preds.append(out)
                        _targets.append(batch["label"].numpy())
                _y_pred = np.concatenate(_preds)
                _y_true = np.concatenate(_targets)
                _split_data[_split] = (_y_true, _y_pred)
                _all_vals += list(_y_true) + list(_y_pred)

            _lo, _hi = min(_all_vals) - 0.3, max(_all_vals) + 0.3
            _x_range = np.linspace(_lo, _hi, 300)

            for _split in ("train", "valid", "test"):
                _y_true, _y_pred = _split_data[_split]
                c = colors[_split]
                _r2 = r2_score(_y_true, _y_pred)
                _mu_t, _sd_t = float(_y_true.mean()), float(_y_true.std())
                _mu_p, _sd_p = float(_y_pred.mean()), float(_y_pred.std())

                ax.scatter(
                    _y_true,
                    _y_pred,
                    alpha=0.25,
                    s=7,
                    color=c,
                    label=f"{_split} R²={_r2:.3f}",
                )

                _kde_x = gaussian_kde(_y_true, bw_method=0.3)
                ax_top.plot(_x_range, _kde_x(_x_range), color=c, lw=1.5)
                ax_top.fill_between(_x_range, _kde_x(_x_range), alpha=0.2, color=c)
                ax_top.axvline(_mu_t, color=c, lw=1, linestyle=":")
                ax_top.text(
                    _mu_t,
                    0.97,
                    f"μ={_mu_t:.2f}\nσ={_sd_t:.2f}",
                    transform=ax_top.get_xaxis_transform(),
                    fontsize=5.5,
                    color=c,
                    ha="center",
                    va="top",
                    linespacing=1.3,
                )

                _kde_y = gaussian_kde(_y_pred, bw_method=0.3)
                ax_right.plot(_kde_y(_x_range), _x_range, color=c, lw=1.5)
                ax_right.fill_betweenx(_x_range, _kde_y(_x_range), alpha=0.2, color=c)
                ax_right.axhline(_mu_p, color=c, lw=1, linestyle=":")
                ax_right.text(
                    0.97,
                    _mu_p,
                    f"μ={_mu_p:.2f}\nσ={_sd_p:.2f}",
                    transform=ax_right.get_yaxis_transform(),
                    fontsize=5.5,
                    color=c,
                    ha="right",
                    va="center",
                    linespacing=1.3,
                )

            ax.plot([_lo, _hi], [_lo, _hi], "k--", linewidth=1)
            ax.set_xlim(_lo, _hi)
            ax.set_ylim(_lo, _hi)
            ax.set_aspect("equal")
            ax.set_xlabel("Actual logD")
            ax.set_ylabel("Predicted logD")
            ax.legend(fontsize=7, loc="upper left")

            ax_top.set_title(label, fontsize=9)
            ax_top.set_xlim(_lo, _hi)
            ax_top.set_yticks([])
            ax_top.tick_params(labelbottom=False)
            for _spine in ax_top.spines.values():
                _spine.set_visible(False)

            ax_right.set_ylim(_lo, _hi)
            ax_right.set_xticks([])
            ax_right.tick_params(labelleft=False)
            for _spine in ax_right.spines.values():
                _spine.set_visible(False)

        plt.suptitle("Parity plots by pretraining condition", fontsize=11, y=1.02)
        plt.tight_layout()
        save_fig(fig, "04_transformer_parity")
        return mo.vstack([fig])

    _()
    return


@app.cell
def _(mo):
    mo.md("## Training Curves")
    return


@app.cell
def _(mo, pd):
    def _():
        import glob as _glob
        import os as _os

        import matplotlib.pyplot as plt

        _dirs = {
            "Pretrain": "checkpoints/pretrain",
            "Finetune": "checkpoints/finetune",
        }
        _colors = {"train": "#4e79a7", "val": "#e15759"}
        _window = 5

        for _phase, _ckpt_dir in _dirs.items():
            _csv = _os.path.join(_ckpt_dir, f"{_phase.lower()}_metrics_history.csv")
            if not _os.path.exists(_csv):
                _candidates = sorted(
                    _glob.glob(
                        _os.path.join(_ckpt_dir, "logs", "**", "metrics.csv"),
                        recursive=True,
                    )
                )
                if _candidates:
                    _csv = _candidates[-1]
            if not _os.path.exists(_csv):
                continue

            _df = pd.read_csv(_csv)
            _metric = "loss"
            fig, ax = plt.subplots(figsize=(8, 3))
            for _split in ("train", "val"):
                _col = f"{_split}_{_metric}"
                if _col not in _df.columns:
                    continue
                _sub = _df[["epoch", _col]].dropna(subset=[_col])
                _by_epoch = _sub.groupby("epoch")[_col].mean()
                ax.plot(
                    _by_epoch.index,
                    _by_epoch.values,
                    alpha=0.3,
                    color=_colors[_split],
                    linewidth=0.8,
                )
                ax.plot(
                    _by_epoch.index,
                    _by_epoch.rolling(_window, min_periods=1).mean(),
                    color=_colors[_split],
                    linewidth=2,
                    label=f"{_split} ({_window}-ep mean)",
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title(f"{_phase} loss vs epoch")
            ax.legend(fontsize=9)
            plt.tight_layout()

            from src.plot_utils import save_fig as _sf

            _sf(fig, f"04_transformer_{_phase.lower()}_curves")
            mo.vstack([fig])

    _()
    return


if __name__ == "__main__":
    app.run()
