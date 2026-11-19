"""Read-only access to the precomputed analytics tables and model bundle."""

import json
import os
from functools import lru_cache

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYTICS = os.path.join(ROOT, "processed_data", "analytics")
RAW = os.path.join(ROOT, "raw_data")
MODEL_PATH = os.path.join(ROOT, "processed_data", "price_model_v2.joblib")


def path(name):
    return os.path.join(ANALYTICS, name)


def exists(name):
    return os.path.exists(path(name))


@lru_cache(maxsize=None)
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(path(name))


@lru_cache(maxsize=1)
def meta() -> dict:
    p = path("analytics_meta.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    import joblib
    from .bundle_io import from_plain
    return from_plain(joblib.load(MODEL_PATH))


@lru_cache(maxsize=1)
def official_rpi() -> pd.DataFrame:
    p = os.path.join(RAW, "hdb_resale_price_index_official_1q2009_100.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["quarter", "index"])
    return pd.read_csv(p)
