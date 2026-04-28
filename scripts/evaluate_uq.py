"""
Compare Deep Ensemble, Last-Layer Laplace, and Conformal Prediction on
the LipophilicityGNN model.

Loads ensemble checkpoints and a single reference checkpoint, runs all
three UQ methods, prints a comparison table, and writes reliability
diagrams and a bar-chart summary to figs/.

Usage
-----
    # defaults: reads checkpoints/uq/ensemble/, uses seed_42 for Laplace
    pixi run evaluate-uq

    # explicit paths
    pixi run evaluate-uq \\
        --ensemble-dir checkpoints/uq/ensemble \\
        --single-checkpoint checkpoints/uq/ensemble/seed_42/<name>.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the UQ evaluation script.

    Params:
        None
    Returns:
        argparse.ArgumentParser : configured parser
    """
    p = argparse.ArgumentParser(
        description="Evaluate and compare UQ methods on LipophilicityGNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("checkpoints/uq/ensemble"),
        help="directory containing seed_<N> subdirectories from ensemble_gnn.py",
    )
    p.add_argument(
        "--single-checkpoint",
        type=Path,
        default=None,
        help="checkpoint for Laplace (defaults to best ckpt in seed_42/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs"),
        help="directory for output figures and CSV",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="miscoverage level (0.1 → 90%% conformal intervals)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="tdc_scaffold",
        choices=["stratified_scaffold", "random", "tdc_scaffold"],
    )
    p.add_argument("--batch-size", type=int, default=64)
    return p


def _find_checkpoint(seed_dir: Path) -> Path:
    """
    Return the single best .ckpt file in seed_dir.

    Params:
        seed_dir: Path : directory for one ensemble member
    Returns:
        Path : path to the checkpoint file
    """
    ckpts = [p for p in seed_dir.glob("*.ckpt") if "last" not in p.name]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
    return sorted(ckpts)[0]


def _find_seed_dirs(ensemble_dir: Path) -> list[Path]:
    """
    Discover all seed_<N> subdirectories under ensemble_dir.

    Params:
        ensemble_dir: Path : root ensemble checkpoint directory
    Returns:
        list[Path] : sorted list of seed subdirectories
    """
    dirs = sorted(
        d for d in ensemble_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
    )
    if not dirs:
        raise FileNotFoundError(f"No seed_* directories found in {ensemble_dir}")
    return dirs


def _build_loaders(split: str, batch_size: int, device):
    """
    Build chemprop DataLoaders for train, valid, and test splits.

    Params:
        split: str : split strategy
        batch_size: int : loader batch size
        device: torch.device : device for ChemBertaEncoder encoding
    Returns:
        tuple[dict, dict] : (splits DataFrames, DataLoaders)
    """
    import numpy as np
    from chemprop.data import build_dataloader

    from src.data import get_random_split, get_splits, get_tdc_split
    from src.graph_data import ChemBertaEncoder, build_chemprop_dataset

    _split_fns = {
        "stratified_scaffold": get_splits,
        "random": get_random_split,
        "tdc_scaffold": get_tdc_split,
    }
    splits_df = _split_fns[split](seed=42)
    encoder = ChemBertaEncoder().to(device)
    lm_embs = {
        name: encoder.encode(df["Drug"].tolist()) for name, df in splits_df.items()
    }
    datasets = {
        name: build_chemprop_dataset(
            splits_df[name]["Drug"],
            np.asarray(splits_df[name]["Y"].values, dtype=np.float32),
            lm_embs[name],
        )
        for name in ("train", "valid", "test")
    }
    loaders = {
        name: build_dataloader(datasets[name], batch_size=batch_size, shuffle=False)
        for name in ("train", "valid", "test")
    }
    return splits_df, loaders


def _predict_gnn(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run point predictions for a single model.

    Params:
        model: LipophilicityGNN : model in eval mode
        loader: DataLoader : chemprop DataLoader
        device: torch.device : inference device
    Returns:
        tuple[ndarray, ndarray] : (predictions (N,), targets (N,))
    """
    import torch

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            out = model(batch.bmg, X_d).squeeze(-1).cpu().numpy()
            preds.append(out)
            targets.append(batch.Y.squeeze(-1).numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _plot_reliability(
    results: dict[str, dict],
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save reliability diagrams (one panel per UQ method) to output_dir.

    Each panel bins predictions by predicted std, then compares the
    expected Gaussian coverage at that std to the observed coverage.

    Params:
        results: dict[str, dict] : method name → dict with 'mean' and 'std' arrays
        y_true: ndarray : true test targets
        output_dir: Path : directory to write uq_reliability.png
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, res) in zip(axes, results.items()):
        y_pred = res["mean"]
        y_std = res["std"]
        errors = np.abs(y_true - y_pred)
        # Vary the coverage level from 0 to 1 and compare expected vs observed
        levels = np.linspace(0.05, 0.95, 19)
        from scipy.stats import norm

        expected, observed = [], []
        for level in levels:
            z = norm.ppf(0.5 + level / 2)
            expected.append(level)
            observed.append(float((errors <= z * y_std).mean()))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(expected, observed, "o-", ms=4, label=method)
        ax.set_title(method)
        ax.set_xlabel("Expected coverage")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Observed coverage")
    fig.suptitle("Reliability diagrams (test split)", fontsize=12)
    fig.tight_layout()
    out_path = output_dir / "uq_reliability.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {out_path}")


def _plot_comparison(metrics_table: dict[str, dict], output_dir: Path) -> None:
    """
    Save a four-panel bar chart comparing UQ methods across key metrics.

    Params:
        metrics_table: dict[str, dict] : method name → metrics dict from compute_uq_metrics
        output_dir: Path : directory to write uq_comparison.png
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    methods = list(metrics_table.keys())
    metric_keys = ["ece", "mean_interval_width", "empirical_coverage", "spearman_rho"]
    metric_labels = [
        "ECE ↓",
        "Mean interval width ↓",
        "Empirical coverage →",
        "Spearman ρ ↑",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, label in zip(axes, metric_keys, metric_labels):
        vals = [metrics_table[m][key] for m in methods]
        bars = ax.bar(methods, vals, color=["#4c72b0", "#dd8452", "#55a868"])
        ax.set_title(label, fontsize=11)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("UQ method comparison (test split)", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "uq_comparison.svg"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison chart saved to {out_path}")


def main() -> None:
    """
    Load all three UQ methods, evaluate, print comparison table, and save figures.

    Params:
        None
    Returns:
        None
    """
    import json

    import pandas as pd
    import torch

    from src.train_gnn import load_checkpoint
    from src.uq import (
        compute_uq_metrics,
        conformal_calibrate,
        conformal_predict,
        fit_laplace,
        predict_laplace,
    )

    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building data loaders...")
    splits_df, loaders = _build_loaders(args.split, args.batch_size, device)

    # --- Resolve checkpoints ---
    seed_dirs = _find_seed_dirs(args.ensemble_dir)
    single_ckpt = args.single_checkpoint
    if single_ckpt is None:
        seed_42_dir = args.ensemble_dir / "seed_42"
        if not seed_42_dir.exists():
            seed_42_dir = seed_dirs[0]
        single_ckpt = _find_checkpoint(seed_42_dir)
    print(f"Reference checkpoint (Laplace): {single_ckpt}")

    # Shared ground truth
    _, val_targets = zip(
        *[(None, batch.Y.squeeze(-1).numpy()) for batch in loaders["valid"]]
    )
    _, test_targets = zip(
        *[(None, batch.Y.squeeze(-1).numpy()) for batch in loaders["test"]]
    )
    y_val = np.concatenate(val_targets)
    y_test = np.concatenate(test_targets)

    results: dict[str, dict] = {}

    # ===========================================================
    # 1. Deep Ensemble
    # ===========================================================
    print(f"\n{'='*50}\nDeep Ensemble ({len(seed_dirs)} members)\n{'='*50}")
    ensemble_preds_val, ensemble_preds_test = [], []
    for seed_dir in seed_dirs:
        ckpt = _find_checkpoint(seed_dir)
        model = load_checkpoint(ckpt, device=device)
        val_pred, _ = _predict_gnn(model, loaders["valid"], device)
        test_pred, _ = _predict_gnn(model, loaders["test"], device)
        ensemble_preds_val.append(val_pred)
        ensemble_preds_test.append(test_pred)
        print(f"  loaded {ckpt.name}")

    ens_val_mean = np.stack(ensemble_preds_val).mean(0)
    ens_val_std = np.stack(ensemble_preds_val).std(0)
    ens_test_mean = np.stack(ensemble_preds_test).mean(0)
    ens_test_std = np.stack(ensemble_preds_test).std(0)

    q_ens_90 = conformal_calibrate(y_val, ens_val_mean, alpha=args.alpha)
    ens_metrics = compute_uq_metrics(
        y_test, ens_test_mean, ens_test_std, alpha=args.alpha
    )
    results["Ensemble"] = {"mean": ens_test_mean, "std": ens_test_std, **ens_metrics}
    print(f"Ensemble conformal q ({(1-args.alpha)*100:.0f}%): {q_ens_90:.4f}")

    # ===========================================================
    # 2. Last-Layer Laplace
    # ===========================================================
    print(f"\n{'='*50}\nLast-Layer Laplace\n{'='*50}")
    ref_model = load_checkpoint(single_ckpt, device=device)
    print("Fitting Laplace on training split...")
    la = fit_laplace(ref_model, loaders["train"], device)

    lap_val_mean, lap_val_std = predict_laplace(la, ref_model, loaders["valid"], device)
    lap_test_mean, lap_test_std = predict_laplace(
        la, ref_model, loaders["test"], device
    )

    q_lap_90 = conformal_calibrate(y_val, lap_val_mean, alpha=args.alpha)
    lap_metrics = compute_uq_metrics(
        y_test, lap_test_mean, lap_test_std, alpha=args.alpha
    )
    results["Laplace"] = {"mean": lap_test_mean, "std": lap_test_std, **lap_metrics}
    print(f"Laplace conformal q ({(1-args.alpha)*100:.0f}%): {q_lap_90:.4f}")

    # ===========================================================
    # 3. Conformal Prediction (on single model point predictions)
    # ===========================================================
    print(f"\n{'='*50}\nConformal Prediction (single model)\n{'='*50}")
    # Reuse the reference model for point predictions; std = constant q
    single_val_pred, _ = _predict_gnn(ref_model, loaders["valid"], device)
    single_test_pred, _ = _predict_gnn(ref_model, loaders["test"], device)

    q_conf_90 = conformal_calibrate(y_val, single_val_pred, alpha=args.alpha)
    conf_lower, conf_upper = conformal_predict(single_test_pred, q_conf_90)
    # Represent conformal as constant std = q / z_{1-α/2} for metric parity
    from scipy.stats import norm

    z = norm.ppf(1 - args.alpha / 2)
    conf_std = np.full_like(single_test_pred, q_conf_90 / z)
    conf_metrics = compute_uq_metrics(
        y_test, single_test_pred, conf_std, alpha=args.alpha
    )
    results["Conformal"] = {"mean": single_test_pred, "std": conf_std, **conf_metrics}
    print(f"Conformal q ({(1-args.alpha)*100:.0f}%): {q_conf_90:.4f}")
    empirical_cov = float(((y_test >= conf_lower) & (y_test <= conf_upper)).mean())
    print(f"Empirical coverage on test: {empirical_cov:.3f}")

    # ===========================================================
    # Print comparison table
    # ===========================================================
    scalar_keys = [
        "rmse",
        "mae",
        "r2",
        "ece",
        "mean_interval_width",
        "empirical_coverage",
        "spearman_rho",
    ]
    print(f"\n{'='*70}")
    print(f"{'Metric':<22}", end="")
    for method in results:
        print(f"  {method:>12}", end="")
    print()
    print("-" * 70)
    for key in scalar_keys:
        print(f"{key:<22}", end="")
        for method in results:
            print(f"  {results[method][key]:>12.4f}", end="")
        print()
    print("=" * 70)

    # ===========================================================
    # Save outputs
    # ===========================================================
    metrics_only = {
        m: {k: v for k, v in res.items() if k not in ("mean", "std")}
        for m, res in results.items()
    }
    csv_path = args.output_dir / "uq_comparison.csv"
    pd.DataFrame(metrics_only).T.to_csv(csv_path)
    print(f"\nMetrics CSV saved to {csv_path}")

    json_path = args.output_dir / "uq_comparison.json"
    with open(json_path, "w") as f:
        json.dump(metrics_only, f, indent=2)

    _plot_reliability(results, y_test, args.output_dir)
    _plot_comparison(metrics_only, args.output_dir)

    # Save per-molecule test predictions for notebook use
    pred_df = pd.DataFrame(
        {
            "smiles": splits_df["test"]["Drug"].tolist(),
            "y_true": y_test,
            "ensemble_mean": ens_test_mean,
            "ensemble_std": ens_test_std,
            "laplace_mean": lap_test_mean,
            "laplace_std": lap_test_std,
            "conformal_mean": single_test_pred,
            "conformal_std": conf_std,
        }
    )
    pred_csv = Path("checkpoints/uq/uq_test_predictions.csv")
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)
    print(f"Per-molecule predictions saved to {pred_csv}")


if __name__ == "__main__":
    main()
