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
    _methods = {
        "Ensemble": ("ensemble_mean", "ensemble_std"),
        "Laplace": ("laplace_mean", "laplace_std"),
        "Conformal": ("conformal_mean", "conformal_std"),
    }
    _levels = np.linspace(0.05, 0.95, 19)
    _colors = ["#4c72b0", "#dd8452", "#55a868"]

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for _ax, (_method, (_mcol, _scol)), _col in zip(_axes, _methods.items(), _colors):
        _y_true = df_pred["y_true"].values
        _y_pred = df_pred[_mcol].values
        _y_std = df_pred[_scol].values
        _errors = np.abs(_y_true - _y_pred)
        _expected, _observed = [], []
        for _lv in _levels:
            _z = norm.ppf(0.5 + _lv / 2)
            _expected.append(_lv)
            _observed.append(float((_errors <= _z * _y_std).mean()))
        _ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        _ax.plot(_expected, _observed, "o-", ms=4, color=_col, label=_method)
        _ax.set_title(_method, fontsize=11)
        _ax.set_xlabel("Expected coverage")
        _ax.legend(fontsize=8)
    _axes[0].set_ylabel("Observed coverage")
    _fig.suptitle("Reliability diagrams (test split)", fontsize=12)
    _fig.tight_layout()
    save_fig(_fig, "uq_reliability")
    mo.md(
        "**Figure 1.** Reliability diagrams. Points above the diagonal = underconfident "
        "(intervals too wide); below = overconfident. Conformal gives constant σ so "
        "its Gaussian calibration is not meaningful — see empirical coverage instead."
    )
    return


@app.cell
def _(metrics, mo, np, plt, save_fig):
    _methods = list(metrics.keys())
    _metric_keys = ["ece", "mean_interval_width", "empirical_coverage", "spearman_rho"]
    _metric_labels = [
        "ECE ↓",
        "Mean interval width ↓",
        "Empirical coverage →",
        "Spearman ρ ↑",
    ]
    _colors = ["#4c72b0", "#dd8452", "#55a868"]

    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))
    for _ax, _key, _label in zip(_axes, _metric_keys, _metric_labels):
        _vals = [metrics[m].get(_key, float("nan")) for m in _methods]
        _bars = _ax.bar(_methods, _vals, color=_colors)
        _ax.set_title(_label, fontsize=11)
        _ax.set_xticks(range(len(_methods)))
        _ax.set_xticklabels(_methods, rotation=15, ha="right", fontsize=9)
        for _bar, _val in zip(_bars, _vals):
            if not np.isnan(_val):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() * 1.01,
                    f"{_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    _fig.suptitle("UQ method comparison (test split)", fontsize=13)
    _fig.tight_layout()
    save_fig(_fig, "uq_comparison")
    mo.md(
        "**Figure 2.** UQ metric comparison. Ensemble has the sharpest intervals "
        "and the only meaningful Spearman ρ — the only method that can rank "
        "molecules by confidence."
    )
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
    _std_thresh = df_pred.loc[_inlier_mask, "std"].median()

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

    _std_thresh_val = float(_std_thresh)
    return (df_q,)


@app.cell
def _(df_q, mo, mpatches, plt, save_fig):
    _COLORS = {
        "Outlier+High σ": "#d62728",
        "Outlier+Low σ": "#ff7f0e",
        "Inlier+High σ": "#aec7e8",
        "Inlier+Low σ": "#c7c7c7",
    }
    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _quad, _grp in df_q.groupby("quadrant"):
        _ax.scatter(
            _grp["ensemble_std"],
            _grp["error"],
            c=_COLORS[_quad],
            s=18 if "Inlier" in _quad else 60,
            alpha=0.6 if "Inlier" in _quad else 0.9,
            label=_quad,
            zorder=3 if "Outlier" in _quad else 1,
        )
    _ax.axhline(1.0, color="black", lw=0.8, ls="--", label="|error| = 1.0")
    _ax.axvline(
        _std_thresh_val,
        color="gray",
        lw=0.8,
        ls=":",
        label=f"σ = {_std_thresh_val:.3f} (inlier median)",
    )
    _ax.set_xlabel("Ensemble std (σ)")
    _ax.set_ylabel("|y_true − ensemble_mean|")
    _ax.set_title("Prediction error vs. uncertainty (test split)")
    _handles = [mpatches.Patch(color=c, label=l) for l, c in _COLORS.items()]
    _handles += [
        plt.Line2D([0], [0], color="black", ls="--", lw=0.8, label="|error| = 1.0"),
        plt.Line2D(
            [0], [0], color="gray", ls=":", lw=0.8, label=f"σ = {_std_thresh_val:.3f}"
        ),
    ]
    _ax.legend(handles=_handles, fontsize=8, loc="upper left")
    _fig.tight_layout()
    save_fig(_fig, "uq_scatter")
    mo.md(
        "**Figure 3.** Error vs. uncertainty. Outlier+High σ (red) are correctly "
        "flagged as uncertain. The lone Outlier+Low σ (orange) lies just below "
        "the σ threshold — the model was confidently wrong on this molecule."
    )
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
    _desc_cols = [
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
    print(df_desc.groupby("quadrant")[_desc_cols].mean().round(2).T.to_string())
    return (df_desc,)


@app.cell
def _(df_desc, mo, plt, save_fig):
    _COLORS = {
        "Outlier+High σ": "#d62728",
        "Outlier+Low σ": "#ff7f0e",
        "Inlier+High σ": "#aec7e8",
        "Inlier+Low σ": "#c7c7c7",
    }
    _quad_order = ["Inlier+Low σ", "Inlier+High σ", "Outlier+Low σ", "Outlier+High σ"]

    _fig, _axes = plt.subplots(2, 5, figsize=(18, 7))
    for _ax, _col in zip(_axes.flat, _desc_cols):
        _data = [
            df_desc.loc[df_desc["quadrant"] == _q, _col].dropna().values
            for _q in _quad_order
        ]
        _parts = _ax.violinplot(
            _data,
            positions=range(len(_quad_order)),
            showmedians=True,
            showextrema=False,
        )
        for _pc, _q in zip(_parts["bodies"], _quad_order):
            _pc.set_facecolor(_COLORS[_q])
            _pc.set_alpha(0.7)
        _parts["cmedians"].set_color("black")
        _ax.set_xticks(range(len(_quad_order)))
        _ax.set_xticklabels(
            [q.replace("+", "\n") for q in _quad_order],
            fontsize=6,
            rotation=0,
        )
        _ax.set_title(_col, fontsize=9)

    _fig.suptitle("RDKit descriptor distributions per quadrant", fontsize=12)
    _fig.tight_layout()
    save_fig(_fig, "uq_descriptors")
    mo.md(
        "**Figure 4.** Descriptor distributions per quadrant. Look for shifts in "
        "MW, TPSA, and HBD/HBA between Outlier+High σ and both Inlier groups."
    )
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
    _COLORS = {
        "Outlier+High σ": "#d62728",
        "Outlier+Low σ": "#ff7f0e",
        "Inlier+High σ": "#aec7e8",
        "Inlier+Low σ": "#c7c7c7",
    }
    _quad_order = ["Inlier+Low σ", "Inlier+High σ", "Outlier+Low σ", "Outlier+High σ"]

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _data = [
        df_tanimoto.loc[df_tanimoto["quadrant"] == _q, "nn_tanimoto"].dropna().values
        for _q in _quad_order
    ]
    _parts = _ax.violinplot(
        _data, positions=range(len(_quad_order)), showmedians=True, showextrema=False
    )
    for _pc, _q in zip(_parts["bodies"], _quad_order):
        _pc.set_facecolor(_COLORS[_q])
        _pc.set_alpha(0.7)
    _parts["cmedians"].set_color("black")
    _ax.set_xticks(range(len(_quad_order)))
    _ax.set_xticklabels([q.replace("+", "\n") for q in _quad_order], fontsize=9)
    _ax.set_ylabel("NN Tanimoto to training set")
    _ax.set_title("Chemical novelty per quadrant (Morgan r=2, 2048 bits)")
    _fig.tight_layout()
    save_fig(_fig, "uq_tanimoto")
    mo.md(
        "**Figure 5.** Nearest-neighbour Tanimoto similarity to the training set. "
        "Lower similarity = more OOD. If Outlier+High σ molecules have lower "
        "similarity, the ensemble uncertainty is correctly tracking scaffold novelty."
    )
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

    # Fraction of molecules in each group with at least one occurrence
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
    _outliers = (
        df_q[df_q["quadrant"].str.startswith("Outlier")]
        .sort_values("error", ascending=False)
        .reset_index(drop=True)
    )
    _figs = []
    for _i, _row in _outliers.iterrows():
        _smi = _row["smiles"]
        _lm_emb = attr_encoder.encode([_smi])[0]
        _scores = attr_explainer.explain(_smi, _lm_emb)
        _title = (
            f"{_row['quadrant']}\n"
            f"y={_row['y_true']:.2f}  ŷ={_row['ensemble_mean']:.2f}  "
            f"|err|={_row['error']:.2f}  σ={_row['ensemble_std']:.3f}"
        )
        _fig = plot_atom_contributions(_smi, _scores, title=_title)
        save_fig(_fig, f"uq_attribution_{_i:02d}")
        _figs.append(_fig)

    mo.vstack(
        [mo.md(f"Attribution computed for **{len(_figs)}** outlier molecules.")] + _figs
    )
    return


if __name__ == "__main__":
    app.run()
