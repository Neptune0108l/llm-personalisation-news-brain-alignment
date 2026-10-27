"""Within-participant ridge decoding; accepts trial-by-voxel arrays."""
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

def fit_ridge_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RidgeCV(alphas=list(alphas))
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    coef = model.coef_.astype(float) if hasattr(model, "coef_") else np.zeros(X_train.shape[1], dtype=float)
    return np.asarray(pred, dtype=float), coef, float(model.alpha_)


def ridge_loocv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> Dict[str, np.ndarray | float]:
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)
    coefs = np.zeros((len(y), X.shape[1]), dtype=float)
    selected_alphas = np.zeros(len(y), dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        pred, coef, alpha = fit_ridge_fold(X[train_idx], y[train_idx], X[test_idx], alphas=alphas)
        preds[test_idx] = pred
        coefs[fold_idx] = coef
        selected_alphas[fold_idx] = alpha
    return {
        "predictions": preds,
        "coef_folds": coefs,
        "mean_coef": np.mean(coefs, axis=0),
        "selected_alphas": selected_alphas,
        "mean_alpha": float(np.mean(selected_alphas)),
    }
