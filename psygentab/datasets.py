"""Dataset registry, cleaning, and the 60/20/20 split contract.

Every experiment reads data only through `load_split`, so all generators,
the PSyGenTAB calibration step and every metric see the same partitions:

  train : fits the generator (these rows are the MIA "members")
  val   : calibration holdout used by PSyGenTAB to score U and P while
          choosing the sampling configuration (never used for reporting)
  test  : untouched; TSTR test set, DCR/NNDR holdout baseline and the MIA
          "non-members" for all reported numbers
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Raw CSVs live in <repo>/data/raw/<dataset>.csv (override with PSY_DATA).
RAW = os.environ.get("PSY_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              "data", "raw"))

# Columns treated as categorical are (a) non-numeric or (b) numeric with at
# most this many distinct values in the full table.
CAT_MAX_UNIQUE = 15

# Generator training rows are capped so every model in the zoo is trained on
# the same budget (DISCO-TAB uses the same convention).
MAX_TRAIN = 2000
MAX_VAL = 1500
MAX_TEST = 3000

DATASETS = {
    # --- clinical / health -------------------------------------------------
    "breast_cancer": dict(target="diagnosis", drop=["id"], domain="Clinical"),
    "diabetes": dict(target="Diabetes_012", domain="Survey"),
    "heart_failure": dict(target="DEATH_EVENT", domain="Clinical"),
    "hypothyroid": dict(target="binaryClass", na_values=["?"], drop=["TBG", "TBG measured"],
                domain="Clinical"),
    # `selector` is a train/test indicator, not a diagnosis (McDermott &
    # Forsyth, PRL 2016); the recommended target is drinks > 3.
    "liver_disorders": dict(target="drinks_gt3", drop=["selector"], domain="Clinical"),
    "obesity": dict(target="NObeyesdad", domain="Survey"),
    "parkinsons": dict(target="status", drop=["name"], domain="Clinical"),
    # --- generalisation ----------------------------------------------------
    "adult": dict(target="Target", na_values=["?"], domain="Demographics"),
    "pir_vision": dict(target="Target", domain="Sensor/IoT"),
    "vn_banking": dict(target="is_churned", drop=["customer_id"], domain="Financial"),
}
CLINICAL = ["breast_cancer", "diabetes", "heart_failure", "hypothyroid",
            "liver_disorders", "obesity", "parkinsons"]
ALL = list(DATASETS)


def load_raw(name):
    spec = DATASETS[name]
    df = pd.read_csv(os.path.join(RAW, f"{name}.csv"), na_values=spec.get("na_values"),
                     skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if name == "liver_disorders":
        df["drinks_gt3"] = (df["drinks"] > 3).astype(int)
        df = df.drop(columns=["drinks"])
    df = df.drop(columns=[c for c in spec.get("drop", []) if c in df.columns])
    df = df.dropna(axis=1, how="all")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip().str.rstrip(".")
    df = df.dropna(subset=[spec["target"]])
    # Targets are always strings so every generator treats them as discrete.
    df[spec["target"]] = df[spec["target"]].astype(str)
    return df.reset_index(drop=True)


def schema(df, target):
    cats, nums = [], []
    for c in df.columns:
        if c == target:
            continue
        if (not pd.api.types.is_numeric_dtype(df[c])
                or df[c].nunique(dropna=True) <= CAT_MAX_UNIQUE):
            cats.append(c)
        else:
            nums.append(c)
    return {"target": target, "categorical": cats, "numerical": nums,
            "columns": list(df.columns)}


def harmonise(df, meta):
    """Cast categoricals to str (NaN -> 'nan') and numerics to float."""
    df = df[meta["columns"]].copy()
    for c in meta["categorical"] + [meta["target"]]:
        df[c] = df[c].astype(str)
    for c in meta["numerical"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _strat(y):
    vc = y.value_counts()
    return y if vc.min() >= 2 else None


def load_split(name, split):
    """Return (train, val, test, meta) for split index `split` (0..4)."""
    df = load_raw(name)
    meta = schema(df, DATASETS[name]["target"])
    df = harmonise(df, meta)
    rs = 1000 + split
    y = df[meta["target"]]
    train, rest = train_test_split(df, test_size=0.4, random_state=rs, stratify=_strat(y))
    val, test = train_test_split(rest, test_size=0.5, random_state=rs,
                                 stratify=_strat(rest[meta["target"]]))

    def cap(d, n):
        if len(d) <= n:
            return d.reset_index(drop=True)
        d, _ = train_test_split(d, train_size=n, random_state=rs, stratify=_strat(d[meta["target"]]))
        return d.reset_index(drop=True)

    train, val, test = cap(train, MAX_TRAIN), cap(val, MAX_VAL), cap(test, MAX_TEST)
    # Missing numericals (hypothyroid lab values) -> train median; the
    # dataset's own "<lab> measured" indicators keep the missingness signal.
    for c in meta["numerical"]:
        med = train[c].median()
        for d in (train, val, test):
            d[c] = d[c].fillna(med)
    meta = dict(meta, dataset=name, split=split, n_full=len(df),
                n_train=len(train), n_val=len(val), n_test=len(test),
                classes=sorted(df[meta["target"]].unique().tolist()))
    return train, val, test, meta


def describe_all():
    rows = []
    for n in ALL:
        df = load_raw(n)
        m = schema(df, DATASETS[n]["target"])
        rows.append(dict(dataset=n, domain=DATASETS[n]["domain"], N=len(df),
                         cont=len(m["numerical"]), cat=len(m["categorical"]),
                         classes=df[m["target"]].nunique(),
                         minority=float(df[m["target"]].value_counts(normalize=True).min())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(describe_all().to_string(index=False))
    tr, va, te, meta = load_split("heart_failure", 0)
    print(meta["n_train"], meta["n_val"], meta["n_test"])
    assert not set(map(tuple, tr.values)) & set(map(tuple, te.values))
