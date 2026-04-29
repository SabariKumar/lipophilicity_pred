import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
    from scipy.stats import norm, spearmanr

    from src.data import get_tdc_split
    from src.explain import AtomAttributionExplainer, plot_atom_contributions
    from src.graph_data import ChemBertaEncoder
    from src.plot_utils import save_fig
    from src.train_gnn import load_checkpoint
    from src.uq import conformal_calibrate

    return (
        AtomAttributionExplainer,
        Chem,
        ChemBertaEncoder,
        DataStructs,
        Descriptors,
        Fragments,
        GetMorganFingerprintAsBitVect,
        Path,
        get_tdc_split,
        json,
        load_checkpoint,
        mo,
        mpatches,
        norm,
        np,
        pd,
        plot_atom_contributions,
        plt,
        rdMolDescriptors,
        save_fig,
        torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # UQ Analysis

    Two analyses in one notebook:

    1. **UQ method comparison** — reliability diagrams and metric bar charts for
       Deep Ensemble, Last-Layer Laplace, and Conformal Prediction. Reproduces
       `figs/uq_reliability.svg` and `figs/uq_comparison.svg`.

    2. **Outlier chemical analysis** — segments the test set into four quadrants
       by prediction error and ensemble uncertainty, then compares chemical
       features (RDKit descriptors, OOD distance, functional groups) and per-atom
       attribution across quadrants.

    **Prerequisites:** run `pixi run ensemble-gnn` then `pixi run evaluate-uq`
    before opening this notebook.
    """
    )
    return


@app.cell
def _(mo):
    pred_csv_input = mo.ui.text(
        label="Per-molecule predictions CSV",
        value="checkpoints/uq/uq_test_predictions.csv",
        full_width=True,
    )
    metrics_json_input = mo.ui.text(
        label="UQ metrics JSON",
        value="figs/uq_comparison.json",
        full_width=True,
    )
    ckpt_input = mo.ui.text(
        label="Reference checkpoint (seed 42, for attribution)",
        value="checkpoints/uq/ensemble/seed_42/ensemble-seed-42-epoch=053-val_mae=0.4364.ckpt",
        full_width=True,
    )
    mo.vstack([pred_csv_input, metrics_json_input, ckpt_input])
    return ckpt_input, metrics_json_input, pred_csv_input


@app.cell
def _(Path, json, metrics_json_input, mo, pd, pred_csv_input):
    _pred_path = Path(pred_csv_input.value)
    _metrics_path = Path(metrics_json_input.value)
    mo.stop(
        not _pred_path.exists(),
        mo.callout(
            mo.md(f"**Missing:** `{_pred_path}` — run `pixi run evaluate-uq` first."),
            kind="warn",
        ),
    )
    mo.stop(
        not _metrics_path.exists(),
        mo.callout(
            mo.md(
                f"**Missing:** `{_metrics_path}` — run `pixi run evaluate-uq` first."
            ),
            kind="warn",
        ),
    )
    df_pred = pd.read_csv(_pred_path)
    with open(_metrics_path) as _f:
        metrics = json.load(_f)

    # Derived columns used throughout
    df_pred["error"] = (df_pred["y_true"] - df_pred["ensemble_mean"]).abs()

    _n = len(df_pred)
    _n_out = (df_pred["error"] > 1.0).sum()
    mo.md(f"Loaded **{_n}** test molecules — **{_n_out}** outliers (|error| > 1 logD).")
    return df_pred, metrics


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. UQ Method Comparison
    """
    )
    return


@app.cell
def _(df_pred, mo, norm, np, plt, save_fig):
    def _():
        _methods = {
            "Ensemble": ("ensemble_mean", "ensemble_std"),
            "Laplace": ("laplace_mean", "laplace_std"),
            "Conformal": ("conformal_mean", "conformal_std"),
        }
        _levels = np.linspace(0.05, 0.95, 19)
        _colors = ["#4c72b0", "#dd8452", "#55a868"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, (_method, (_mcol, _scol)), _col in zip(axes, _methods.items(), _colors):
            _y_true = df_pred["y_true"].values
            _y_pred = df_pred[_mcol].values
            _y_std = df_pred[_scol].values
            _errors = np.abs(_y_true - _y_pred)
            _expected, _observed = [], []
            for _lv in _levels:
                _z = norm.ppf(0.5 + _lv / 2)
                _expected.append(_lv)
                _observed.append(float((_errors <= _z * _y_std).mean()))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            ax.plot(_expected, _observed, "o-", ms=4, color=_col, label=_method)
            ax.set_title(_method, fontsize=11)
            ax.set_xlabel("Expected coverage")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Observed coverage")
        fig.suptitle("Reliability diagrams (test split)", fontsize=12)
        plt.tight_layout()
        save_fig(fig, "05_uq_reliability")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 1.** Reliability diagrams. Points above the diagonal = underconfident "
                    "(intervals too wide); below = overconfident. Conformal gives constant σ so "
                    "its Gaussian calibration is not meaningful — see empirical coverage instead."
                ),
            ]
        )

    _()
    return


@app.cell
def _(metrics, mo, np, plt, save_fig):
    def _():
        _methods = list(metrics.keys())
        _metric_keys = [
            "ece",
            "mean_interval_width",
            "empirical_coverage",
            "spearman_rho",
        ]
        _metric_labels = [
            "ECE ↓",
            "Mean interval width ↓",
            "Empirical coverage →",
            "Spearman ρ ↑",
        ]
        _colors = ["#4c72b0", "#dd8452", "#55a868"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, _key, _label in zip(axes, _metric_keys, _metric_labels):
            _vals = [metrics[m].get(_key, float("nan")) for m in _methods]
            _bars = ax.bar(_methods, _vals, color=_colors)
            ax.set_title(_label, fontsize=11)
            ax.set_xticks(range(len(_methods)))
            ax.set_xticklabels(_methods, rotation=15, ha="right", fontsize=9)
            for _bar, _val in zip(_bars, _vals):
                if not np.isnan(_val):
                    ax.text(
                        _bar.get_x() + _bar.get_width() / 2,
                        _bar.get_height() * 1.01,
                        f"{_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        fig.suptitle("UQ method comparison (test split)", fontsize=13)
        plt.tight_layout()
        save_fig(fig, "05_uq_comparison")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 2.** UQ metric comparison. Ensemble has the sharpest intervals "
                    "and the only meaningful Spearman ρ — the only method that can rank "
                    "molecules by confidence."
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Outlier Analysis

    Molecules are split into four quadrants using:

    - **Outlier**: |y_true − ensemble_mean| > 1.0 logD
    - **High σ**: ensemble std > median std of inliers

    Only **11 molecules** exceed the error threshold (10 high-σ, 1 low-σ),
    so statistical tests are underpowered. Results are descriptive.
    """
    )
    return


@app.cell
def _(df_pred):
    _error_thresh = 1.0
    _inlier_mask = df_pred["error"] <= _error_thresh
    _std_thresh = df_pred.loc[_inlier_mask, "ensemble_std"].median()

    _QUAD_LABELS = {
        "Outlier+High σ": (df_pred["error"] > _error_thresh)
        & (df_pred["ensemble_std"] > _std_thresh),
        "Outlier+Low σ": (df_pred["error"] > _error_thresh)
        & (df_pred["ensemble_std"] <= _std_thresh),
        "Inlier+High σ": (df_pred["error"] <= _error_thresh)
        & (df_pred["ensemble_std"] > _std_thresh),
        "Inlier+Low σ": (df_pred["error"] <= _error_thresh)
        & (df_pred["ensemble_std"] <= _std_thresh),
    }
    df_q = df_pred.copy()
    df_q["quadrant"] = "Inlier+Low σ"
    for _label, _mask in _QUAD_LABELS.items():
        df_q.loc[_mask, "quadrant"] = _label

    _counts = df_q["quadrant"].value_counts()
    print("Quadrant sizes:")
    for _q, _n in _counts.items():
        print(f"  {_q}: {_n}")

    std_thresh_val = float(_std_thresh)
    return df_q, std_thresh_val


@app.cell
def _(df_q, mo, mpatches, plt, save_fig, std_thresh_val):
    def _():
        _COLORS = {
            "Outlier+High σ": "#d62728",
            "Outlier+Low σ": "#ff7f0e",
            "Inlier+High σ": "#aec7e8",
            "Inlier+Low σ": "#c7c7c7",
        }
        fig, ax = plt.subplots(figsize=(7, 5))
        for _quad, _grp in df_q.groupby("quadrant"):
            ax.scatter(
                _grp["ensemble_std"],
                _grp["error"],
                c=_COLORS[_quad],
                s=18 if "Inlier" in _quad else 60,
                alpha=0.6 if "Inlier" in _quad else 0.9,
                label=_quad,
                zorder=3 if "Outlier" in _quad else 1,
            )
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="|error| = 1.0")
        ax.axvline(
            std_thresh_val,
            color="gray",
            lw=0.8,
            ls=":",
            label=f"σ = {std_thresh_val:.3f} (inlier median)",
        )
        ax.set_xlabel("Ensemble std (σ)")
        ax.set_ylabel("|y_true − ensemble_mean|")
        ax.set_title("Prediction error vs. uncertainty (test split)")
        _handles = [mpatches.Patch(color=c, label=l) for l, c in _COLORS.items()]
        _handles += [
            plt.Line2D([0], [0], color="black", ls="--", lw=0.8, label="|error| = 1.0"),
            plt.Line2D(
                [0],
                [0],
                color="gray",
                ls=":",
                lw=0.8,
                label=f"σ = {std_thresh_val:.3f}",
            ),
        ]
        ax.legend(handles=_handles, fontsize=8, loc="upper left")
        plt.tight_layout()
        save_fig(fig, "05_uq_scatter")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 3.** Error vs. uncertainty. Outlier+High σ (red) are correctly "
                    "flagged as uncertain. The lone Outlier+Low σ (orange) lies just below "
                    "the σ threshold — the model was confidently wrong on this molecule."
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Chemical Feature Analysis
    """
    )
    return


@app.cell
def _(Chem, Descriptors, df_q, pd, rdMolDescriptors):
    def _rdkit_descriptors(smiles_series):
        rows = []
        for smi in smiles_series:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({})
                continue
            rows.append(
                {
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "TPSA": Descriptors.TPSA(mol),
                    "HBD": rdMolDescriptors.CalcNumHBD(mol),
                    "HBA": rdMolDescriptors.CalcNumHBA(mol),
                    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
                    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "Rings": rdMolDescriptors.CalcNumRings(mol),
                    "Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
                }
            )
        return pd.DataFrame(rows, index=smiles_series.index)

    df_desc = pd.concat([df_q, _rdkit_descriptors(df_q["smiles"])], axis=1)
    desc_cols = [
        "MW",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "FractionCSP3",
        "AromaticRings",
        "Rings",
        "Heteroatoms",
    ]
    print("Descriptor means per quadrant:")
    print(df_desc.groupby("quadrant")[desc_cols].mean().round(2).T.to_string())
    return desc_cols, df_desc


@app.cell
def _(desc_cols, df_desc, mo, plt, save_fig):
    def _():
        _COLORS = {
            "Outlier+High σ": "#d62728",
            "Outlier+Low σ": "#ff7f0e",
            "Inlier+High σ": "#aec7e8",
            "Inlier+Low σ": "#c7c7c7",
        }
        _quad_order = [
            "Inlier+Low σ",
            "Inlier+High σ",
            "Outlier+Low σ",
            "Outlier+High σ",
        ]

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for ax, col in zip(axes.flat, desc_cols):
            _violin_data, _violin_pos = [], []
            for pos, q in enumerate(_quad_order):
                d = df_desc.loc[df_desc["quadrant"] == q, col].dropna().values
                if len(d) >= 2:
                    _violin_data.append(d)
                    _violin_pos.append(pos)
                elif len(d) == 1:
                    ax.scatter([pos], d, color=_COLORS[q], s=60, zorder=4)
            if _violin_data:
                _parts = ax.violinplot(
                    _violin_data,
                    positions=_violin_pos,
                    showmedians=True,
                    showextrema=False,
                )
                for _pc, pos in zip(_parts["bodies"], _violin_pos):
                    _pc.set_facecolor(_COLORS[_quad_order[pos]])
                    _pc.set_alpha(0.7)
                _parts["cmedians"].set_color("black")
            ax.set_xticks(range(len(_quad_order)))
            ax.set_xticklabels(
                [q.replace("+", "\n") for q in _quad_order],
                fontsize=6,
                rotation=0,
            )
            ax.set_title(col, fontsize=9)

        fig.suptitle("RDKit descriptor distributions per quadrant", fontsize=12)
        plt.tight_layout()
        save_fig(fig, "05_uq_descriptors")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 4.** Descriptor distributions per quadrant. Look for shifts in "
                    "MW, TPSA, and HBD/HBA between Outlier+High σ and both Inlier groups."
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. OOD Analysis: Nearest-Neighbour Tanimoto to Training Set

    For each test molecule, the Morgan fingerprint (radius 2, 2048 bits)
    Tanimoto similarity to its nearest training-set neighbour is computed.
    A low similarity indicates the molecule is chemically dissimilar from
    anything seen during training — a proxy for being out-of-distribution.
    """
    )
    return


@app.cell
def _(Chem, DataStructs, GetMorganFingerprintAsBitVect, df_q, get_tdc_split):
    _splits = get_tdc_split(seed=42)
    _train_smiles = _splits["train"]["Drug"].tolist()

    def _morgan_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    _train_fps = [_morgan_fp(s) for s in _train_smiles]
    _train_fps = [fp for fp in _train_fps if fp is not None]

    _nn_sims = []
    for _smi in df_q["smiles"]:
        _fp = _morgan_fp(_smi)
        if _fp is None:
            _nn_sims.append(float("nan"))
        else:
            _nn_sims.append(
                float(max(DataStructs.BulkTanimotoSimilarity(_fp, _train_fps)))
            )

    df_tanimoto = df_q.copy()
    df_tanimoto["nn_tanimoto"] = _nn_sims

    print("Mean NN Tanimoto per quadrant:")
    print(
        df_tanimoto.groupby("quadrant")["nn_tanimoto"]
        .agg(["mean", "median", "std"])
        .round(3)
        .to_string()
    )
    return (df_tanimoto,)


@app.cell
def _(df_tanimoto, mo, plt, save_fig):
    def _():
        _COLORS = {
            "Outlier+High σ": "#d62728",
            "Outlier+Low σ": "#ff7f0e",
            "Inlier+High σ": "#aec7e8",
            "Inlier+Low σ": "#c7c7c7",
        }
        _quad_order = [
            "Inlier+Low σ",
            "Inlier+High σ",
            "Outlier+Low σ",
            "Outlier+High σ",
        ]

        fig, ax = plt.subplots(figsize=(7, 4))
        _violin_data, _violin_pos = [], []
        for pos, q in enumerate(_quad_order):
            d = (
                df_tanimoto.loc[df_tanimoto["quadrant"] == q, "nn_tanimoto"]
                .dropna()
                .values
            )
            if len(d) >= 2:
                _violin_data.append(d)
                _violin_pos.append(pos)
            elif len(d) == 1:
                ax.scatter([pos], d, color=_COLORS[q], s=80, zorder=4)
        if _violin_data:
            _parts = ax.violinplot(
                _violin_data,
                positions=_violin_pos,
                showmedians=True,
                showextrema=False,
            )
            for _pc, pos in zip(_parts["bodies"], _violin_pos):
                _pc.set_facecolor(_COLORS[_quad_order[pos]])
                _pc.set_alpha(0.7)
            _parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(_quad_order)))
        ax.set_xticklabels([q.replace("+", "\n") for q in _quad_order], fontsize=9)
        ax.set_ylabel("NN Tanimoto to training set")
        ax.set_title("Chemical novelty per quadrant (Morgan r=2, 2048 bits)")
        plt.tight_layout()
        save_fig(fig, "05_uq_tanimoto")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 5.** Nearest-neighbour Tanimoto similarity to the training set. "
                    "Lower similarity = more OOD. If Outlier+High σ molecules have lower "
                    "similarity, the ensemble uncertainty is correctly tracking scaffold novelty."
                ),
            ]
        )

    _()
    return


@app.cell
def _(df_tanimoto, mo, np, plt, save_fig):
    def _():
        _quad_order = [
            "Inlier+Low σ",
            "Inlier+High σ",
            "Outlier+Low σ",
            "Outlier+High σ",
        ]

        def _cohens_d(a, b):
            if len(a) < 2 or len(b) < 2:
                return 0.0
            na, nb = len(a), len(b)
            pooled_sd = np.sqrt(
                ((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2)
                / (na + nb - 2)
            )
            return (
                0.0 if pooled_sd == 0 else float((np.mean(a) - np.mean(b)) / pooled_sd)
            )

        _groups = {
            q: df_tanimoto.loc[df_tanimoto["quadrant"] == q, "nn_tanimoto"]
            .dropna()
            .values
            for q in _quad_order
        }

        n = len(_quad_order)
        matrix = np.zeros((n, n))
        for i, qi in enumerate(_quad_order):
            for j, qj in enumerate(_quad_order):
                if i != j:
                    matrix[i, j] = _cohens_d(_groups[qi], _groups[qj])

        _short = [
            "Inlier\nLow σ",
            "Inlier\nHigh σ",
            "Outlier\nLow σ",
            "Outlier\nHigh σ",
        ]
        _vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(_short, fontsize=8)
        ax.set_yticklabels(_short, fontsize=8)
        ax.set_title("Pairwise Cohen's d — NN Tanimoto similarity", fontsize=10)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(matrix[i, j]) > 0.6 * _vmax else "black",
                )
        plt.colorbar(im, ax=ax, label="Cohen's d (row − col)")
        plt.tight_layout()
        save_fig(fig, "05_uq_tanimoto_cohens_d")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 6.** Pairwise Cohen's d for NN Tanimoto similarity between quadrants. "
                    "Positive (red) = row group has higher similarity than column group; "
                    "negative (blue) = lower. |d| < 0.2 small, 0.2–0.5 medium, > 0.8 large. "
                    "Empty quadrants (n < 2) are set to 0."
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Functional Group Analysis

    RDKit `fr_` fragment counts are compared between Outlier+High σ (n=10) and
    Inlier+Low σ (n≈415). With only 10 outliers, formal tests are underpowered;
    the table shows fragments present in ≥30% of outliers but ≤5% of inliers,
    and vice versa.
    """
    )
    return


@app.cell
def _(Chem, Fragments, df_q, mo, pd):
    _fr_names = [attr for attr in dir(Fragments) if attr.startswith("fr_")]

    def _fragment_counts(smiles_list):
        rows = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({f: 0 for f in _fr_names})
            else:
                rows.append({f: getattr(Fragments, f)(mol) for f in _fr_names})
        return pd.DataFrame(rows)

    _out_hi = df_q[df_q["quadrant"] == "Outlier+High σ"]
    _in_lo = df_q[df_q["quadrant"] == "Inlier+Low σ"]

    _frag_out = _fragment_counts(_out_hi["smiles"].tolist())
    _frag_in = _fragment_counts(_in_lo["smiles"].tolist())

    _freq_out = (_frag_out > 0).mean()
    _freq_in = (_frag_in > 0).mean()

    _enriched = pd.DataFrame(
        {
            "freq_outlier_hi": _freq_out,
            "freq_inlier_lo": _freq_in,
        }
    )
    _enriched["ratio"] = _enriched["freq_outlier_hi"] / (
        _enriched["freq_inlier_lo"] + 1e-6
    )
    _enriched = _enriched[
        (_enriched["freq_outlier_hi"] >= 0.3) | (_enriched["freq_inlier_lo"] >= 0.3)
    ].sort_values("ratio", ascending=False)

    mo.ui.table(
        _enriched.reset_index().rename(columns={"index": "fragment"}).round(3),
        label="Fragment frequencies: Outlier+High σ vs Inlier+Low σ",
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Atom Attribution for Outlier Molecules

    Captum IntegratedGradients applied to the atom feature matrix of the seed-42
    model. **Red** atoms push logD up (hydrophobic); **blue** atoms pull it down
    (hydrophilic). Score magnitudes indicate which atoms dominate the prediction.

    Shown for all 11 outliers, sorted by |error| descending.
    """
    )
    return


@app.cell
def _(
    AtomAttributionExplainer,
    ChemBertaEncoder,
    Path,
    ckpt_input,
    load_checkpoint,
    mo,
    torch,
):
    _ckpt_path = Path(ckpt_input.value)
    mo.stop(
        not _ckpt_path.exists(),
        mo.callout(
            mo.md(f"**Checkpoint not found:** `{_ckpt_path}`"),
            kind="warn",
        ),
    )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attr_model = load_checkpoint(_ckpt_path, device=_device)
    attr_model.eval()
    attr_encoder = ChemBertaEncoder().to(_device)
    attr_explainer = AtomAttributionExplainer(attr_model, device=_device)
    mo.md(f"Loaded checkpoint: `{_ckpt_path.name}`")
    return attr_encoder, attr_explainer


@app.cell
def _(
    attr_encoder,
    attr_explainer,
    df_q,
    mo,
    plot_atom_contributions,
    save_fig,
):
    def _():
        _outliers = (
            df_q[df_q["quadrant"].str.startswith("Outlier")]
            .sort_values("error", ascending=False)
            .reset_index(drop=True)
        )
        figs = []
        for i, row in _outliers.iterrows():
            smi = row["smiles"]
            lm_emb = attr_encoder.encode([smi])[0]
            scores = attr_explainer.explain(smi, lm_emb)
            title = (
                f"{row['quadrant']}\n"
                f"y={row['y_true']:.2f}  ŷ={row['ensemble_mean']:.2f}  "
                f"|err|={row['error']:.2f}  σ={row['ensemble_std']:.3f}"
            )
            fig = plot_atom_contributions(smi, scores, title=title)
            save_fig(fig, f"05_uq_attribution_{i:02d}")
            figs.append(fig)

        return mo.vstack(
            [mo.md(f"Attribution computed for **{len(figs)}** outlier molecules.")]
            + figs
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Outlier Chemical Motif Analysis

    For each outlier molecule, we compute **Crippen cLogP** (ionisation-naive) and
    compare it to the measured **logD** (pH 7.4). The gap `logD − cLogP` quantifies
    how much ionisation at physiological pH shifts the apparent lipophilicity away
    from the neutral-form estimate — a direct proxy for how hard the molecule is for
    a graph model trained on neutral SMILES to predict.
    """
    )
    return


@app.cell
def _(Chem, Descriptors, Fragments, df_q, pd):
    def _basic_n(mol):
        return (
            Fragments.fr_NH1(mol)
            + Fragments.fr_NH2(mol)
            + Fragments.fr_piperdine(mol)
            + Fragments.fr_piperzine(mol)
            + Fragments.fr_morpholine(mol)
            + Fragments.fr_guanido(mol)
            + Fragments.fr_amidine(mol)
        )

    def _acid_groups(mol):
        return Fragments.fr_COO(mol) + Fragments.fr_sulfonamd(mol)

    _outlier_rows = df_q[df_q["quadrant"].str.startswith("Outlier")].copy()
    _rows = []
    for _, r in _outlier_rows.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        clogp = Descriptors.MolLogP(mol)
        _rows.append(
            {
                "smiles": r["smiles"],
                "quadrant": r["quadrant"],
                "logD": round(r["y_true"], 2),
                "pred": round(r["ensemble_mean"], 2),
                "error": round(r["error"], 2),
                "cLogP": round(clogp, 2),
                "gap": round(r["y_true"] - clogp, 2),
                "MW": round(Descriptors.MolWt(mol), 0),
                "TPSA": round(Descriptors.TPSA(mol), 1),
                "basic_N": _basic_n(mol),
                "acid_grps": _acid_groups(mol),
                "has_Cl": any(a.GetSymbol() == "Cl" for a in mol.GetAtoms()),
            }
        )
    df_outlier_chem = (
        pd.DataFrame(_rows).sort_values("error", ascending=False).reset_index(drop=True)
    )
    print(
        df_outlier_chem[
            [
                "logD",
                "pred",
                "error",
                "cLogP",
                "gap",
                "MW",
                "basic_N",
                "acid_grps",
                "has_Cl",
            ]
        ].to_string()
    )
    return (df_outlier_chem,)


@app.cell
def _(df_outlier_chem, mo):
    mo.ui.table(
        df_outlier_chem[
            [
                "logD",
                "pred",
                "error",
                "cLogP",
                "gap",
                "MW",
                "TPSA",
                "basic_N",
                "acid_grps",
                "has_Cl",
            ]
        ].rename(
            columns={"basic_N": "basic N", "acid_grps": "acid grps", "has_Cl": "Cl?"}
        ),
        label="Outlier molecules — key chemical properties",
    )
    return


@app.cell
def _(df_outlier_chem, mo, plt, save_fig):
    def _():
        _df = df_outlier_chem.sort_values("gap")
        n = len(_df)
        labels = [
            f"logD={r['logD']:.2f}  Cl={'✓' if r['has_Cl'] else '–'}  "
            f"basic N={r['basic_N']}  acid={r['acid_grps']}"
            for _, r in _df.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (_, r) in enumerate(_df.iterrows()):
            color = "#d62728" if r["gap"] < 0 else "#4c72b0"
            ax.plot(
                [r["cLogP"], r["logD"]],
                [i, i],
                color=color,
                lw=2,
                alpha=0.7,
                solid_capstyle="round",
            )
            ax.scatter(r["cLogP"], i, color="#888888", s=55, zorder=4, marker="o")
            ax.scatter(r["logD"], i, color=color, s=55, zorder=4, marker="D")
            ax.text(
                max(r["cLogP"], r["logD"]) + 0.15,
                i,
                f"Δ={r['gap']:+.2f}",
                va="center",
                fontsize=7.5,
                color=color,
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.axvline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("logD / cLogP")
        ax.set_title(
            "Measured logD (◆) vs Crippen cLogP (●) for outlier molecules", fontsize=10
        )

        import matplotlib.lines as mlines

        ax.legend(
            handles=[
                mlines.Line2D(
                    [], [], color="#888888", marker="o", ls="", ms=7, label="cLogP"
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#555555",
                    marker="D",
                    ls="",
                    ms=7,
                    label="measured logD",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#d62728",
                    lw=2,
                    label="gap < 0 (ionisation suppresses logD)",
                ),
                mlines.Line2D(
                    [], [], color="#4c72b0", lw=2, label="gap > 0 (logD exceeds cLogP)"
                ),
            ],
            fontsize=8,
            loc="lower right",
        )
        plt.tight_layout()
        save_fig(fig, "05_uq_outlier_motifs")
        return mo.vstack(
            [
                fig,
                mo.md(
                    "**Figure 8.** Dumbbell chart of measured logD vs Crippen cLogP for all 11 outliers, "
                    "sorted by gap (logD − cLogP). Red = ionisation suppresses apparent lipophilicity "
                    "(logD < cLogP); blue = logD exceeds cLogP. Labels show basic-N and acid-group counts."
                ),
            ]
        )

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Three recurring motifs

    **1. Ionisable / amphoteric character** — the dominant theme in 9 of 11 outliers.
    The model encodes atom features at a fixed, neutral charge state. At pH 7.4 the
    dominant microspecies of many of these molecules is charged, pulling the measured
    logD well below cLogP (gaps as large as −6 log units). Zwitterions — molecules
    with both a basic amine and an acidic group — are the hardest cases: the two
    competing protonation equilibria interact nonlinearly, producing a net logD that
    neither atom-level nor fragment-level approaches capture well.

    **2. Multiple basic centres** — 7 of 11 molecules carry 2–4 basic nitrogen groups
    (piperidines, piperazines, morpholines, amidines). Each protonation event adds a
    full positive charge; the second protonation is harder than the first and the
    resulting dication has a very different partition coefficient. The GNN sees each
    ring individually but has no direct mechanism to model cooperative charge effects
    across the whole molecule.

    **3. Aryl chlorine paired with polar centres** — 7 of 11 outliers contain at least
    one aryl Cl. Cl itself is well-represented in training and is not the root cause.
    The pattern is Cl on a lipophilic scaffold combined with one or more ionisable
    centres elsewhere — the model anchors on the high-cLogP aromatic system and
    underweights the ionisation correction that pulls the measured logD back down.

    **Common fix:** pre-enumerate the major microspecies at pH 7.4 (e.g. via
    `Chem.MolStandardize` or Epik) before featurisation, so the model sees the
    actual charged form rather than the neutral SMILES.
    """
    )
    return


if __name__ == "__main__":
    app.run()
