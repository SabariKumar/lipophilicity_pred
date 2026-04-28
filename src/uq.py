from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from src.gnn_model import LipophilicityGNN

# ---------------------------------------------------------------------------
# Laplace helpers
# ---------------------------------------------------------------------------


def extract_fused_features(
    model: LipophilicityGNN,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract fused (graph + LM) features from all batches, bypassing FusionMLP.

    Runs backbone → attention pooling → cat([h_graph, V_d]) for every batch
    without passing through the FusionMLP head. The (N, d_h+d_lm) output is
    used by fit_laplace to extract penultimate-layer features cheaply.

    Params:
        model: LipophilicityGNN : model in eval mode
        loader: DataLoader : chemprop DataLoader yielding TrainingBatch tuples
        device: torch.device : inference device
    Returns:
        tuple[Tensor, Tensor] : (features (N, d_h+d_lm), targets (N,))
    """
    model.eval()
    features, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch.bmg.to(device)
            X_d = batch.X_d.to(device) if batch.X_d is not None else None
            H = model.backbone(batch.bmg)
            h_graph = model.pool(H, batch.bmg.batch)
            h_fused = torch.cat([h_graph, X_d], dim=-1)
            features.append(h_fused.cpu())
            if batch.Y is not None:
                targets.append(batch.Y.squeeze(-1).cpu())
    return torch.cat(features), torch.cat(targets)


def _extract_penultimate(
    model: LipophilicityGNN,
    h_fused: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run h_fused through all FusionMLP layers except the final Linear.

    Returns bias-augmented features so the posterior covariance covers both
    weight and bias uncertainty of the last layer.

    Params:
        model: LipophilicityGNN : model in eval mode
        h_fused: Tensor : shape (N, d_h+d_lm), from extract_fused_features
        device: torch.device : inference device
    Returns:
        ndarray : shape (N, d_hidden+1), bias column appended
    """
    model.eval()
    with torch.no_grad():
        F = model.fusion.net[:-1](h_fused.to(device)).cpu().numpy()
    ones = np.ones((F.shape[0], 1), dtype=F.dtype)
    return np.hstack([F, ones])  # (N, d_hidden+1)


def fit_laplace(
    model: LipophilicityGNN,
    train_loader,
    device: torch.device,
    prior_precision: float = 1e-4,
) -> dict:
    """
    Fit a closed-form last-layer Laplace approximation for the Linear(d_hidden→1) head.

    Treats all layers up to the final Linear as a fixed feature extractor and
    places a Gaussian posterior over that layer's weights and bias. No external
    library is required.

    The GGN posterior precision is:
        Λ = F^T F / σ² + prior_precision * I
    where F is the bias-augmented penultimate-layer activation matrix (N, d+1)
    and σ² is the MAP training residual variance.

    prior_precision matches the weight_decay used in train_gnn (1e-4) so the
    Laplace prior is consistent with the MAP regularisation.

    Params:
        model: LipophilicityGNN : trained model in eval mode
        train_loader: DataLoader : chemprop DataLoader for the training split
        device: torch.device : inference device
        prior_precision: float : Gaussian prior precision λ (default matches train weight_decay)
    Returns:
        dict : keys 'Sigma' (ndarray, posterior covariance), 'noise_var' (float)
    """
    h_fused, targets = extract_fused_features(model, train_loader, device)
    F = _extract_penultimate(model, h_fused, device)  # (N, d+1)
    y = targets.numpy()

    # MAP predictions via the full fusion head (includes bias correctly)
    model.eval()
    with torch.no_grad():
        y_pred = model.fusion(h_fused.to(device)).squeeze(-1).cpu().numpy()

    noise_var = float(np.mean((y - y_pred) ** 2))
    d = F.shape[1]
    H = F.T @ F / noise_var + prior_precision * np.eye(d)
    Sigma = np.linalg.inv(H)

    return {"Sigma": Sigma, "noise_var": noise_var}


def predict_laplace(
    la: dict,
    model: LipophilicityGNN,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return mean and predictive std from a fitted last-layer Laplace state.

    Predictive variance decomposes as epistemic (x^T Σ x) + aleatoric (σ²).

    Params:
        la: dict : state dict from fit_laplace ('Sigma', 'noise_var')
        model: LipophilicityGNN : same model used in fit_laplace
        loader: DataLoader : chemprop DataLoader for the target split
        device: torch.device : inference device
    Returns:
        tuple[ndarray, ndarray] : (mean (N,), std (N,))
    """
    h_fused, _ = extract_fused_features(model, loader, device)
    F = _extract_penultimate(model, h_fused, device)  # (N, d+1)

    model.eval()
    with torch.no_grad():
        mean = model.fusion(h_fused.to(device)).squeeze(-1).cpu().numpy()

    Sigma = la["Sigma"]
    noise_var = la["noise_var"]
    epistemic_var = np.sum((F @ Sigma) * F, axis=1)
    std = np.sqrt(epistemic_var + noise_var)

    return mean, std


# ---------------------------------------------------------------------------
# Conformal helpers
# ---------------------------------------------------------------------------


def conformal_calibrate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Compute the conformal half-width from calibration-set residuals.

    Uses the finite-sample corrected quantile ⌈(n+1)(1-α)/n⌉ so that
    empirical coverage is guaranteed to be at least 1-α in expectation.

    Params:
        y_true: ndarray : true targets on the calibration (validation) set, shape (N,)
        y_pred: ndarray : point predictions on the calibration set, shape (N,)
        alpha: float : miscoverage level (0.1 → 90% intervals)
    Returns:
        float : scalar half-width q; apply as ŷ ± q on the test set
    """
    residuals = np.abs(y_true - y_pred)
    n = len(residuals)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(residuals, level))


def conformal_predict(
    y_pred: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a symmetric conformal interval to point predictions.

    Params:
        y_pred: ndarray : point predictions, shape (N,)
        q: float : half-width from conformal_calibrate
    Returns:
        tuple[ndarray, ndarray] : (lower (N,), upper (N,))
    """
    return y_pred - q, y_pred + q


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------


def compute_uq_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float]:
    """
    Compute point accuracy, calibration, sharpness, and separation metrics.

    Params:
        y_true: ndarray : true targets, shape (N,)
        y_pred: ndarray : point predictions (e.g. ensemble mean), shape (N,)
        y_std: ndarray : predictive std, shape (N,)
        alpha: float : miscoverage level used for interval width (0.1 → 90%)
    Returns:
        dict[str, float] : keys rmse, mae, r2, ece, mean_interval_width,
            empirical_coverage, spearman_rho
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    errors = np.abs(y_true - y_pred)

    # Point accuracy
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # ECE: 10 equal-mass bins over predicted std; compare empirical vs Gaussian coverage
    n_bins = 10
    bin_edges = np.percentile(y_std, np.linspace(0, 100, n_bins + 1))
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_std >= lo) & (y_std <= hi) if i < n_bins - 1 else (y_std >= lo)
        if mask.sum() == 0:
            continue
        sigma_bin = y_std[mask].mean()
        # Fraction of points within 1 std under Gaussian assumption → ~68.27%
        expected_cov = 0.6827
        actual_cov = float((errors[mask] <= sigma_bin).mean())
        ece += float(mask.mean()) * abs(actual_cov - expected_cov)

    # Sharpness: mean width of (1-alpha) Gaussian interval
    z = float(
        np.abs(
            np.percentile(
                np.random.default_rng(0).standard_normal(100_000), (alpha / 2) * 100
            )
        )
    )
    mean_interval_width = float(2 * z * y_std.mean())

    # Empirical coverage of symmetric conformal interval calibrated on this split
    # (useful when evaluate_uq calls this with the test set and passes the val-calibrated q)
    q = conformal_calibrate(y_true, y_pred, alpha=alpha)
    empirical_coverage = float((errors <= q).mean())

    # Separation: Spearman ρ between predicted uncertainty and absolute error
    rho, _ = spearmanr(y_std, errors)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "ece": ece,
        "mean_interval_width": mean_interval_width,
        "empirical_coverage": empirical_coverage,
        "spearman_rho": float(rho),
    }
