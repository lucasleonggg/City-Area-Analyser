"""Geocode every HDB block in the resale data with OneMap (SLA).

    python pipeline/geocode_blocks.py               # fetch (resumable) + write CSV
    python pipeline/geocode_blocks.py --finalize    # only rebuild the CSV from cache

v1 placed every flat at one hand-typed town centroid, so all spatial
analysis ran on 26 points. This gives each of the ~10,000 blocks its own
coordinate, which is what makes block-level hot spots, distance-to-rail and
within-town location effects possible.

Source: OneMap Search API (https://www.onemap.gov.sg/apidocs/search),
Singapore Land Authority. Query = "<block> <street_name>". A result is
accepted when its BLK_NO equals the block; "strict" additionally requires the
road name to start with the same word as the HDB street name (HDB data uses
abbreviations such as AVE / ST / C'WEALTH, so exact road-name equality is too
strict). Every coordinate is then sanity-checked against its town: a block
more than 5 km from the median of its own town's blocks is marked
coord_ok=False and excluded from analysis rather than trusted.

Output: raw_data/hdb_block_coordinates.csv
  block, street_name, town, lat, lng, postal, match, dist_to_town_median_km, coord_ok
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_PATH, PROCESSED_DIR, ROOT  # noqa: E402

CACHE = os.path.join(PROCESSED_DIR, "geocode_cache.jsonl")
OUT = os.path.join(ROOT, "raw_data", "hdb_block_coordinates.csv")
URL = "https://www.onemap.gov.sg/api/common/elastic/search?searchVal={}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
MAX_KM_FROM_TOWN = 5.0


def fetch(workers=8, pause=0.4):
    df = pd.read_csv(DATA_PATH, usecols=["block", "street_name", "town"], dtype=str)
    df = df.drop_duplicates(["block", "street_name"])
    done = set()
    if os.path.exists(CACHE):
        for line in open(CACHE):
            r = json.loads(line)
            done.add((r["block"], r["street_name"]))
    todo = [r for r in df.itertuples(index=False) if (r.block, r.street_name) not in done]
    print(f"{len(df)} unique blocks, {len(todo)} to fetch")
    lock = threading.Lock()
    out = open(CACHE, "a")

    def work(r):
        rec = {"block": r.block, "street_name": r.street_name, "town": r.town}
        for attempt in range(4):
            try:
                q = urllib.parse.quote(f"{r.block} {r.street_name}")
                with urllib.request.urlopen(URL.format(q), timeout=20) as resp:
                    res = json.load(resp).get("results", [])
                hit = [x for x in res if x.get("BLK_NO", "").upper() == r.block.upper()]
                first = r.street_name.split()[0].upper()
                strict = [x for x in hit if x.get("ROAD_NAME", "").upper().startswith(first)]
                pick = (strict or hit or res or [None])[0]
                if pick:
                    rec.update(lat=float(pick["LATITUDE"]), lng=float(pick["LONGITUDE"]),
                               postal=pick.get("POSTAL"), road_name=pick.get("ROAD_NAME"),
                               match="strict" if strict else ("block_only" if hit else "loose"))
                else:
                    rec["match"] = "none"
                with lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                time.sleep(pause)
                return
            except Exception:  # network hiccup / rate limit: back off, retry
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))


def finalize():
    recs = [json.loads(l) for l in open(CACHE)]
    c = pd.DataFrame(recs).drop_duplicates(["block", "street_name"], keep="last")
    c = c[c["match"].isin(["strict", "block_only", "loose"])].copy()
    med = c.groupby("town")[["lat", "lng"]].median()
    lat0, lng0 = med.loc[c["town"], "lat"].to_numpy(), med.loc[c["town"], "lng"].to_numpy()
    la, lo = np.radians(c["lat"].to_numpy()), np.radians(c["lng"].to_numpy())
    la0, lo0 = np.radians(lat0), np.radians(lng0)
    a = np.sin((la - la0) / 2) ** 2 + np.cos(la) * np.cos(la0) * np.sin((lo - lo0) / 2) ** 2
    c["dist_to_town_median_km"] = (6371 * 2 * np.arcsin(np.sqrt(a))).round(3)
    # 'loose' = OneMap returned something, but not this block number: never trust it.
    c["coord_ok"] = (c["dist_to_town_median_km"] <= MAX_KM_FROM_TOWN) & (c["match"] != "loose")
    cols = ["block", "street_name", "town", "lat", "lng", "postal", "match",
            "dist_to_town_median_km", "coord_ok"]
    c[cols].sort_values(["town", "street_name", "block"]).to_csv(OUT, index=False)
    total = pd.read_csv(DATA_PATH, usecols=["block", "street_name"], dtype=str).drop_duplicates().shape[0]
    print(f"wrote {OUT}: {int(c['coord_ok'].sum())} usable of {total} blocks; "
          f"match counts {c['match'].value_counts().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if not args.finalize:
        fetch()
    finalize()
