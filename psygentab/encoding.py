"""Shared feature encoder fit on the real *train* split only.

Numericals: median imputation + z-scoring (train statistics).
Categoricals: one-hot, scaled by 1/sqrt(2) so a single category mismatch adds
exactly 1 to the squared Euclidean distance, i.e. the same weight as a one-SD
difference in a numerical feature. Unknown categories encode as all-zeros.
The target is included by default (an attacker sees the full record).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def _scale_ohe(x):
    return x / np.sqrt(2.0)


class Encoder:
    def __init__(self, meta, include_target=True):
        self.meta = meta
        self.cats = list(meta["categorical"]) + ([meta["target"]] if include_target else [])
        self.nums = list(meta["numerical"])

    def fit(self, train):
        parts = []
        if self.nums:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                           ("sc", StandardScaler())]), self.nums))
        if self.cats:
            parts.append(("cat", Pipeline([
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("w", FunctionTransformer(_scale_ohe))]), self.cats))
        self.ct = ColumnTransformer(parts)
        self.ct.fit(train[self.nums + self.cats])
        return self

    def transform(self, df):
        X = self.ct.transform(df[self.nums + self.cats])
        return np.nan_to_num(np.asarray(X, dtype=np.float64))


def exact_keys(df, meta):
    """Row keys for exact-match tests: numerics rounded to 6 d.p., all as str."""
    cols = meta["columns"]
    out = df[cols].copy()
    for c in meta["numerical"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return pd.util.hash_pandas_object(out.astype(str), index=False).values
