"""The challenger: LightGBM on the same application characteristics, with business-rule monotone constraints.

Banks keep a simple, explainable champion and test it against a stronger but harder-to-explain challenger.
The challenger only replaces the champion if it is clearly better, because every decline must still be explained.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from crs import binning, features, scorecard

# +1: risk rises with the value, -1: risk falls. Same business rules as the scorecard binning.
MONOTONE = {name: (1 if trend == "ascending" else -1) for name, trend in binning.TREND.items()}
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 0,
}
MAX_ROUNDS = 2000


def columns():
    return [c for c in features.ALLOWED if c not in scorecard.NOT_IN_SCORECARD]


def prepare(X):
    X = X[columns()].copy()
    for c in features.CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def align(Xp, categories):
    """Use the training categories; a category never seen in training becomes missing."""
    Xp = Xp.copy()
    for c, cats in categories.items():
        Xp[c] = pd.Categorical(Xp[c].astype(object).where(Xp[c].isin(cats)), categories=cats)
    return Xp


def _dataset(X, y, reference=None):
    return lgb.Dataset(X, y, reference=reference, free_raw_data=False)


def constraints(X):
    return [MONOTONE.get(c, 0) for c in X.columns]


def best_rounds(X_train, y_train, X_valid, y_valid):
    """Number of trees, chosen by early stopping on the next year (the last time fold)."""
    Xt, Xv = prepare(X_train), prepare(X_valid)
    Xv = align(Xv, {c: Xt[c].cat.categories for c in features.CATEGORICAL if c in Xt})
    train = _dataset(Xt, y_train)
    model = lgb.train(
        PARAMS | {"monotone_constraints": constraints(Xt)},
        train,
        MAX_ROUNDS,
        valid_sets=[_dataset(Xv, y_valid, reference=train)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model.best_iteration


def fit(X, y, rounds):
    Xp = prepare(X)
    model = lgb.train(PARAMS | {"monotone_constraints": constraints(Xp)}, _dataset(Xp, y), rounds)
    model.categories_ = {c: Xp[c].cat.categories for c in features.CATEGORICAL if c in Xp}
    return model


def predict(model, X):
    return model.predict(align(prepare(X), model.categories_))


def importance(model, X, sample=20000, seed=0):
    """Mean absolute SHAP value per characteristic (LightGBM's built-in TreeSHAP)."""
    Xp = align(prepare(X), model.categories_).sample(min(sample, len(X)), random_state=seed)
    contrib = model.predict(Xp, pred_contrib=True)[:, :-1]  # last column is the bias
    return pd.Series(np.abs(contrib).mean(axis=0), index=Xp.columns).sort_values(ascending=False)
