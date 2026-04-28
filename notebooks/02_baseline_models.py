import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import shap

    from src.data import get_splits
    from src.features import smiles_to_descriptors
    from src.models import evaluate, fit_lasso, fit_rf
    from src.plot_utils import save_fig
    from src.preprocessing import apply_preprocessor, build_preprocessor

    return (
        apply_preprocessor,
        build_preprocessor,
        evaluate,
        fit_lasso,
        fit_rf,
        get_splits,
        mo,
        pd,
        save_fig,
        shap,
        smiles_to_descriptors,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Baseline Models — Lasso & Random Forest
    """
    )
    return


@app.cell
def _(get_splits, smiles_to_descriptors):
    splits = get_splits()

    desc = {
        name: smiles_to_descriptors(split["Drug"]) for name, split in splits.items()
    }
    y = {name: split["Y"].values for name, split in splits.items()}
    return desc, y


@app.cell
def _(desc, mo, y):
    mo.md(
        f"""
    | Split | Molecules | Target mean ± std |
    |-------|-----------|-------------------|
    | train | {len(y['train'])} | {y['train'].mean():.2f} ± {y['train'].std():.2f} |
    | valid | {len(y['valid'])} | {y['valid'].mean():.2f} ± {y['valid'].std():.2f} |
    | test  | {len(y['test'])}  | {y['test'].mean():.2f} ± {y['test'].std():.2f}  |

    Descriptor matrix: **{desc['train'].shape[1]}** raw features per molecule.
    """
    )
    return


@app.cell
def _(apply_preprocessor, build_preprocessor, desc):
    preprocessor = build_preprocessor(desc["train"])
    X_train, feature_names = apply_preprocessor(preprocessor, desc["train"])
    X_valid, _ = apply_preprocessor(preprocessor, desc["valid"])
    X_test, _ = apply_preprocessor(preprocessor, desc["test"])
    return X_test, X_train, X_valid, feature_names


@app.cell
def _(feature_names, mo):
    mo.md(
        f"""
    After preprocessing: **{len(feature_names)}** features remain (zero-variance removed, NaN imputed with train medians).
    """
    )
    return


@app.cell
def _(X_train, fit_lasso, fit_rf, y):

    lasso_pipe = fit_lasso(X_train, y["train"])
    rf_model = fit_rf(X_train, y["train"])
    return lasso_pipe, rf_model


@app.cell
def _(X_test, X_train, X_valid, evaluate, lasso_pipe, mo, pd, rf_model, y):
    rows = []
    for label, X, y_true in [
        ("train", X_train, y["train"]),
        ("valid", X_valid, y["valid"]),
        ("test", X_test, y["test"]),
    ]:
        lasso_metrics = evaluate(lasso_pipe, X, y_true)
        rf_metrics = evaluate(rf_model, X, y_true)
        rows.append({"split": label, "model": "Lasso", **lasso_metrics})
        rows.append({"split": label, "model": "Random Forest", **rf_metrics})

    results_df = pd.DataFrame(rows).round(3)
    mo.vstack(
        [
            mo.md("### Model evaluation"),
            mo.ui.table(results_df),
        ]
    )
    return


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(
        options=["Lasso", "Random Forest"],
        value="Lasso",
        label="Model",
    )
    mo.vstack([mo.md("### Parity plot"), model_selector])
    return (model_selector,)


@app.cell
def _(X_test, X_train, X_valid, lasso_pipe, mo, model_selector, rf_model, save_fig, y):
    def _():
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score

        _colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}

        def _make_parity(model, name, fname):
            fig, ax = plt.subplots(figsize=(6, 6))
            _all_vals = []
            for _split, _X, _y in [
                ("train", X_train, y["train"]),
                ("valid", X_valid, y["valid"]),
                ("test", X_test, y["test"]),
            ]:
                _pred = model.predict(_X)
                _r2 = r2_score(_y, _pred)
                ax.scatter(
                    _y,
                    _pred,
                    alpha=0.5,
                    s=15,
                    color=_colors[_split],
                    label=f"{_split} (R²={_r2:.3f})",
                )
                _all_vals += list(_y) + list(_pred)
            _lo, _hi = min(_all_vals) - 0.3, max(_all_vals) + 0.3
            ax.plot([_lo, _hi], [_lo, _hi], "k--", linewidth=1, label="ideal")
            ax.set_xlim(_lo, _hi)
            ax.set_ylim(_lo, _hi)
            ax.set_aspect("equal")
            ax.set_xlabel("Actual logD")
            ax.set_ylabel("Predicted logD")
            ax.set_title(f"{name} — Parity Plot")
            ax.legend(fontsize=9)
            plt.tight_layout()
            save_fig(fig, fname)
            return fig

        fig_lasso = _make_parity(lasso_pipe, "Lasso", "02_parity_lasso")
        fig_rf = _make_parity(rf_model, "Random Forest", "02_parity_rf")
        selected_fig = fig_lasso if model_selector.value == "Lasso" else fig_rf
        return mo.vstack([selected_fig])

    _()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## SHAP Analysis
    """
    )
    return


@app.cell
def _(X_test, X_train, feature_names, lasso_pipe, pd, shap):

    scaler = lasso_pipe.named_steps["scaler"]
    lasso_model = lasso_pipe.named_steps["lasso"]
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    explainer_lasso = shap.LinearExplainer(lasso_model, X_train_scaled)
    shap_lasso = explainer_lasso(X_test_scaled)
    return (shap_lasso,)


@app.cell
def _(mo, save_fig, shap, shap_lasso):
    def _():
        import matplotlib.pyplot as plt

        shap.plots.beeswarm(shap_lasso, max_display=20, show=False)
        fig_lasso = plt.gcf()
        fig_lasso.suptitle("Lasso — SHAP beeswarm (top 20 features)", y=1.01)
        plt.tight_layout()
        save_fig(fig_lasso, "02_shap_lasso")
        return mo.vstack([mo.md("### Lasso SHAP"), fig_lasso])

    _()
    return


@app.cell
def _(X_test, X_train, feature_names, pd, rf_model, shap):
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    explainer_rf = shap.TreeExplainer(rf_model, X_train_df)
    shap_rf = explainer_rf(X_test_df)
    return (shap_rf,)


@app.cell
def _(mo, save_fig, shap, shap_rf):
    def _():
        import matplotlib.pyplot as plt

        shap.plots.beeswarm(shap_rf, max_display=20, show=False)
        fig_rf = plt.gcf()
        fig_rf.suptitle("Random Forest — SHAP beeswarm (top 20 features)", y=1.01)
        plt.tight_layout()
        save_fig(fig_rf, "02_shap_rf")
        return mo.vstack([mo.md("### Random Forest SHAP"), fig_rf])

    _()
    return


@app.cell
def _(feature_names, shap_lasso, shap_rf):
    import numpy as np

    _feature_names_arr = np.array(feature_names)
    _N = 20
    _lasso_imp = np.abs(shap_lasso.values).mean(axis=0)
    _rf_imp = np.abs(shap_rf.values).mean(axis=0)
    _lasso_order = np.argsort(_lasso_imp)[::-1][:_N]
    _rf_order = np.argsort(_rf_imp)[::-1][:_N]

    lasso_top20 = {
        _feature_names_arr[i]: (rank + 1, float(_lasso_imp[i]))
        for rank, i in enumerate(_lasso_order)
    }
    rf_top20 = {
        _feature_names_arr[i]: (rank + 1, float(_rf_imp[i]))
        for rank, i in enumerate(_rf_order)
    }
    consensus_20 = sorted(
        set(lasso_top20) & set(rf_top20),
        key=lambda f: lasso_top20[f][0] + rf_top20[f][0],
    )
    return consensus_20, lasso_top20, rf_top20


@app.cell
def _(consensus_20, lasso_top20, mo, rf_top20, save_fig):
    def _():
        import re

        import matplotlib.pyplot as plt
        import numpy as np

        # ── Exact consensus table ──────────────────────────────────────────────
        rows = [
            {
                "feature": f,
                "lasso_rank": lasso_top20[f][0],
                "rf_rank": rf_top20[f][0],
                "mean_rank": round((lasso_top20[f][0] + rf_top20[f][0]) / 2, 1),
            }
            for f in consensus_20
        ]

        # ── Family classification ──────────────────────────────────────────────
        def _family(name):
            if re.match(r"SMR_VSA", name):
                return "SMR_VSA"
            if re.match(r"PEOE_VSA", name):
                return "PEOE_VSA"
            if re.match(r"SlogP_VSA", name):
                return "SlogP_VSA"
            if re.match(r"(VSA_EState|EState_VSA)", name):
                return "EState_VSA"
            if name.startswith("fr_"):
                return "Frag counts (fr_)"
            if re.match(r"(Chi|Phi|HallKier|FpDensity)", name):
                return "Topological indices"
            return "Counts / global"

        all_feats = list(lasso_top20) + list(rf_top20)
        families = sorted(set(_family(f) for f in all_feats))
        lasso_fam = {
            fam: sum(1 for f in lasso_top20 if _family(f) == fam) for fam in families
        }
        rf_fam = {
            fam: sum(1 for f in rf_top20 if _family(f) == fam) for fam in families
        }

        x = np.arange(len(families))
        w = 0.35
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            x - w / 2,
            [lasso_fam[f] for f in families],
            w,
            label="Lasso",
            color="#4e79a7",
        )
        ax.bar(
            x + w / 2,
            [rf_fam[f] for f in families],
            w,
            label="Random Forest",
            color="#f28e2b",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Features in top-20")
        ax.set_title("Descriptor family representation in top-20 SHAP features")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "02_shap_family_consensus")

        return mo.vstack(
            [
                mo.md("### SHAP consensus — top-20 overlap"),
                mo.ui.table(rows),
                mo.md(
                    """
**Exact consensus (3 features):**

| Feature | Lasso rank | RF rank | Interpretation |
|---------|-----------|---------|----------------|
| `fr_COO` | 6 | 1 | Carboxylic acid count. COOH ionises at physiological pH (pKa 2–5), pulling logD well below logP — the primary mechanistic driver of the logP→logD gap. |
| `NumAromaticRings` | 13 | 3 | Number of aromatic rings. More rings → larger hydrophobic surface area, driving logD up. |
| `fr_halogen` | 3 | 19 | Total halogen count. Cl/Br/I reliably increase lipophilicity; F is context-dependent. |

**Descriptor family consensus:**

Both models load heavily on **VSA (van der Waals surface area) descriptors** — they differ on which bins matter but agree on the family:

- **SMR_VSA** — bins surface area by per-atom molar refractivity → polarisability and dispersion interactions
- **PEOE_VSA** — bins by Gasteiger partial charge → polarity of exposed surface, linked to aqueous solvation energy
- **SlogP_VSA** — bins by per-atom logP contribution → which parts of the surface are lipophilic vs hydrophilic
- **EState_VSA / VSA_EState** — bins by electrotopological state → combined electronic and topological environment

The overarching signal: **the polarity distribution across the molecular surface** is the most informative structural proxy for logD once MolLogP is removed.

Both models also pick up **ionisable group counts** beyond `fr_COO`: Lasso selects `fr_aniline` and `fr_NH2` (basic amines that protonate at pH 7.4); RF selects `fr_COO2`, `fr_Al_COO`, and `NHOHCount`. Different descriptors, same chemistry — **ionisation state at physiological pH** is the primary driver of the logP→logD gap.
            """
                ),
                fig,
            ]
        )

    _()
    return


@app.cell
def _(mo):
    top_n_slider = mo.ui.slider(
        10, 100, value=30, step=5, label="Top-N features per model"
    )
    mo.vstack([mo.md("### Top feature consensus"), top_n_slider])
    return (top_n_slider,)


@app.cell
def _(feature_names, mo, shap_lasso, shap_rf, top_n_slider):
    def _():
        import json
        from pathlib import Path

        import numpy as np

        N = top_n_slider.value
        feature_names_arr = np.array(feature_names)

        lasso_importance = np.abs(shap_lasso.values).mean(axis=0)
        rf_importance = np.abs(shap_rf.values).mean(axis=0)

        lasso_top = set(feature_names_arr[np.argsort(lasso_importance)[::-1][:N]])
        rf_top = set(feature_names_arr[np.argsort(rf_importance)[::-1][:N]])
        consensus = sorted(lasso_top & rf_top)

        out_path = Path("../data/top_features.json")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps({"top_n": N, "features": consensus}, indent=2))
        return mo.vstack(
            [
                mo.md(
                    f"**Lasso top-{N}:** {len(lasso_top)} features  \n"
                    f"**RF top-{N}:** {len(rf_top)} features  \n"
                    f"**Consensus (intersection):** {len(consensus)} features  \n"
                    f"Saved to `data/top_features.json`"
                ),
                mo.ui.table(
                    [
                        {
                            "feature": f,
                            "lasso_rank": int(
                                np.where(
                                    np.argsort(lasso_importance)[::-1]
                                    == np.where(feature_names_arr == f)[0][0]
                                )[0][0]
                            )
                            + 1,
                            "rf_rank": int(
                                np.where(
                                    np.argsort(rf_importance)[::-1]
                                    == np.where(feature_names_arr == f)[0][0]
                                )[0][0]
                            )
                            + 1,
                        }
                        for f in consensus
                    ]
                ),
            ]
        )

    _()
    return


if __name__ == "__main__":
    app.run()
