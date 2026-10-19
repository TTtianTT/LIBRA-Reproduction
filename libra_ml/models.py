"""Quantile regressors with fixed hyperparameters (no tuning on test data)."""
import numpy as np

GBDT_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                   min_samples_leaf=200, l2_regularization=1.0)


def fit_gbdt(X, y, tau, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(loss="quantile", quantile=tau, early_stopping=False,
                                          random_state=seed, **GBDT_PARAMS)
    return model.fit(X, y)


def fit_linear(X, y, tau, max_rows=10000):
    """Linear quantile regression (exact LP via HiGHS) on an evenly spaced subsample."""
    from sklearn.linear_model import QuantileRegressor
    idx = np.linspace(0, len(y) - 1, min(len(y), max_rows)).round().astype(int)
    return QuantileRegressor(quantile=tau, alpha=0.0, solver="highs").fit(X[idx], y[idx])
