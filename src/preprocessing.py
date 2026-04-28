import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """Fit a zero-variance filter and median imputer on training descriptors.

    Params:
        X_train: pd.DataFrame : descriptor DataFrame for the training split
    Returns:
        Pipeline : fitted sklearn Pipeline (VarianceThreshold -> SimpleImputer)
    """
    pipe = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.01)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    pipe.fit(X_train)
    return pipe


def apply_preprocessor(
    pipe: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Apply a fitted preprocessor to a descriptor DataFrame.

    Params:
        pipe: Pipeline : fitted preprocessor from build_preprocessor
        X: pd.DataFrame : descriptor DataFrame to transform
    Returns:
        tuple[np.ndarray, list[str]] : transformed array and surviving feature names
    """
    X_clean = pipe.transform(X)
    feature_names = X.columns[pipe.named_steps["variance"].get_support()].tolist()
    return X_clean, feature_names
