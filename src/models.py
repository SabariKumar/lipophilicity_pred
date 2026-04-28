import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_lasso(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit a StandardScaler + LassoCV pipeline on training data.

    Params:
        X: np.ndarray : training feature matrix, shape (n_samples, n_features)
        y: np.ndarray : training targets, shape (n_samples,)
    Returns:
        Pipeline : fitted pipeline with steps 'scaler' and 'lasso'
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=5, max_iter=10000, n_jobs=-1)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def fit_rf(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Fit a random forest regressor with randomised hyperparameter search.

    Searches over max_depth, min_samples_leaf, and max_features using 5-fold
    cross-validation scored by RMSE, then returns the best estimator.

    Params:
        X: np.ndarray : training feature matrix, shape (n_samples, n_features)
        y: np.ndarray : training targets, shape (n_samples,)
    Returns:
        RandomForestRegressor : best fitted estimator from the search
    """
    param_dist = {
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.33, 0.5],
    }
    base = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    print(f"Best RF params: {search.best_params_}  CV RMSE: {-search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² for a fitted model on a given split.

    Params:
        model: fitted sklearn model or Pipeline with a predict method
        X: np.ndarray : feature matrix, shape (n_samples, n_features)
        y: np.ndarray : true targets, shape (n_samples,)
    Returns:
        dict : keys 'rmse', 'mae', 'r2' with float values
    """
    y_pred = model.predict(X)
    return {
        "rmse": float(mean_squared_error(y, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }
