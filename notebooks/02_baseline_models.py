import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Baseline Models — Lasso & Random Forest
    """
    )
    return


@app.cell
def _():
    from src.data import get_splits
    from src.features import smiles_to_descriptors

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
def _(desc):
    from src.preprocessing import apply_preprocessor, build_preprocessor

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
def _(X_train, y):
    from src.models import fit_lasso, fit_rf

    lasso_pipe = fit_lasso(X_train, y["train"])
    rf_model = fit_rf(X_train, y["train"])
    return lasso_pipe, rf_model


@app.cell
def _(X_test, X_train, X_valid, lasso_pipe, mo, rf_model, y):
    import pandas as pd

    from src.models import evaluate

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
def _(X_test, X_train, X_valid, lasso_pipe, mo, model_selector, rf_model, y):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score

    _model = lasso_pipe if model_selector.value == "Lasso" else rf_model

    fig_parity, ax = plt.subplots(figsize=(6, 6))
    _colors = {"train": "#4e79a7", "valid": "#f28e2b", "test": "#e15759"}

    _all_vals = []
    for _split, _X, _y in [
        ("train", X_train, y["train"]),
        ("valid", X_valid, y["valid"]),
        ("test", X_test, y["test"]),
    ]:
        _pred = _model.predict(_X)
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
    ax.set_title(f"{model_selector.value} — Parity Plot")
    ax.legend(fontsize=9)
    plt.tight_layout()
    mo.vstack([fig_parity])
    return (plt,)


@app.cell
def _(mo):
    mo.md(
        """
    ## SHAP Analysis
    """
    )
    return


@app.cell
def _(X_test, X_train, feature_names, lasso_pipe):
    import shap

    scaler = lasso_pipe.named_steps["scaler"]
    lasso_model = lasso_pipe.named_steps["lasso"]
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)

    explainer_lasso = shap.LinearExplainer(lasso_model, X_train_scaled)
    shap_lasso = explainer_lasso(X_test_scaled)
    shap_lasso.feature_names = feature_names
    return shap, shap_lasso


@app.cell
def _(mo, shap, shap_lasso):
    def _():
        import matplotlib.pyplot as plt

        shap.plots.beeswarm(shap_lasso, max_display=20, show=False)
        fig_lasso = plt.gcf()
        fig_lasso.suptitle("Lasso — SHAP beeswarm (top 20 features)", y=1.01)
        plt.tight_layout()
        return mo.vstack([mo.md("### Lasso SHAP"), fig_lasso])

    _()
    return


@app.cell
def _(X_test, X_train, feature_names, rf_model, shap):
    explainer_rf = shap.TreeExplainer(rf_model, X_train)
    shap_rf = explainer_rf(X_test)
    shap_rf.feature_names = feature_names
    return (shap_rf,)


@app.cell
def _(mo, plt, shap, shap_rf):
    shap.plots.beeswarm(shap_rf, max_display=20, show=False)
    fig_rf = plt.gcf()
    fig_rf.suptitle("Random Forest — SHAP beeswarm (top 20 features)", y=1.01)
    plt.tight_layout()
    mo.vstack([mo.md("### Random Forest SHAP"), fig_rf])
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

    mo.vstack(
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
    return


if __name__ == "__main__":
    app.run()
