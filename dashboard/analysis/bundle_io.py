"""Portable serialisation of the served model bundle.

v2 pickled pandas objects straight into price_model_v2.joblib. Under pandas
3, string columns default to a pyarrow-backed dtype, so loading the file
required pyarrow even though nothing else in the model needs it -- one
reviewer's environment failed exactly there. v3 stores only scikit-learn
estimators, numpy arrays and plain Python types, and rebuilds the pandas
structures after loading. The library versions used to train are recorded
so a mismatch can be reported instead of silently tolerated.
"""

import platform

import numpy as np
import pandas as pd


def versions() -> dict:
    import joblib
    import sklearn
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}


def _series_to_dict(s: pd.Series) -> dict:
    return {str(k): float(v) for k, v in s.items()}


def _frame_to_plain(df: pd.DataFrame) -> dict:
    cols, data, cats = list(df.columns), {}, {}
    for c in cols:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            cats[c] = [str(x) for x in df[c].cat.categories]
            data[c] = df[c].cat.codes.to_numpy(np.int32)
        elif pd.api.types.is_numeric_dtype(df[c]):
            data[c] = df[c].to_numpy()
        else:
            data[c] = [None if (isinstance(v, float) and np.isnan(v)) else str(v) for v in df[c]]
    return {"columns": cols, "data": data, "categoricals": cats}


def _plain_to_frame(p: dict) -> pd.DataFrame:
    out = {}
    for c in p["columns"]:
        if c in p["categoricals"]:
            out[c] = pd.Categorical.from_codes(p["data"][c], categories=p["categoricals"][c])
        else:
            out[c] = p["data"][c]
    return pd.DataFrame(out)[p["columns"]]


def to_plain(bundle: dict) -> dict:
    b = dict(bundle)
    enc = b["encoder"]
    b["encoder"] = {"town": _series_to_dict(enc["town"]), "street": _series_to_dict(enc["street"]),
                    "block": _series_to_dict(enc["block"]), "global": float(enc["global"])}
    li = b["log_index"]
    b["log_index"] = {"months": [int(m) for m in li.index], "values": [float(v) for v in li.values]}
    b["background"] = _frame_to_plain(b["background"])
    b["blocks"] = _frame_to_plain(b["blocks"].reset_index(drop=True))
    b["categories"] = {k: [str(x) for x in v] for k, v in b["categories"].items()}
    b["versions"] = versions()
    b["format"] = "plain-v3"
    return b


def from_plain(b: dict) -> dict:
    if b.get("format") != "plain-v3":
        return b  # older bundle: already pandas objects
    b = dict(b)
    b["encoder"] = {"town": pd.Series(b["encoder"]["town"], dtype=float),
                    "street": pd.Series(b["encoder"]["street"], dtype=float),
                    "block": pd.Series(b["encoder"]["block"], dtype=float),
                    "global": b["encoder"]["global"]}
    b["log_index"] = pd.Series(b["log_index"]["values"], index=b["log_index"]["months"], name="log_index")
    b["background"] = _plain_to_frame(b["background"])
    b["blocks"] = _plain_to_frame(b["blocks"])
    return b


def version_mismatches(b: dict) -> dict:
    """{library: (trained_with, running)} for every library whose version differs."""
    trained = b.get("versions", {})
    now = versions()
    return {k: (v, now.get(k)) for k, v in trained.items() if k != "python" and now.get(k) != v}
