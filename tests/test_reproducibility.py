"""Reproducibility guarantees (EquiCity 3.0).

These fail loudly when the environment differs from the one the published
results were produced in, instead of letting results drift silently.
"""

import importlib.metadata as md
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))


def test_runtime_requirements_are_exact_pins_and_installed():
    bad = []
    for line in open(os.path.join(ROOT, "dashboard", "requirements.txt")):
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([\w.]+)$", line)
        assert m, f"not an exact pin: {line!r}"
        name, want = m.groups()
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            bad.append(f"{name} not installed (want {want})")
            continue
        if have != want:
            bad.append(f"{name} {have} installed, {want} pinned")
    assert not bad, "; ".join(bad) + " -- run: pip install -r dashboard/requirements.txt"


def test_model_loads_without_pyarrow():
    """v2's model needed pyarrow just to unpickle. v3's must not."""
    # Block pyarrow with an import hook (as if it were not installed) rather
    # than a None entry in sys.modules, which libraries treat as "imported".
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pyarrow' or name.startswith('pyarrow.'):\n"
        "            raise ImportError('pyarrow blocked for this test')\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'dashboard')!r})\n"
        "from analysis import explain as E\n"
        "x = E.make_row('Tampines', '4-Room', 'Model A', 92, 8, 1988)\n"
        "p = E.predict(x)\n"
        "assert 200_000 < p['price'] < 2_000_000, p\n"
        "assert 'pyarrow' not in sys.modules, 'pyarrow was imported'\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_model_was_trained_with_installed_library_versions():
    from analysis import store
    from analysis.bundle_io import version_mismatches
    b = store.model_bundle()
    assert b is not None, "processed_data/price_model_v2.joblib missing"
    mism = version_mismatches(b)
    assert not mism, f"model trained with different versions: {mism}"


def test_inputs_match_manifest():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "manifest.py"), "--check", "--skip-missing"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_every_table_the_dashboard_reads_exists():
    from analysis import store
    required = [
        "hedonic_town_year_type.csv", "hedonic_national_month.csv", "hedonic_region_year.csv",
        "lease_curve.csv", "hedonic_fit_summary.csv", "observed_town_type_year.csv",
        "lease_band_year.csv", "recent_sales.csv", "block_premium.csv",
        "rail_access_town_year.csv", "rail_event_study.csv",
        "audit_backtest_overall.csv", "audit_backtest_by_origin.csv", "audit_subgroups.csv",
        "audit_horizons.csv", "market_forecast_baselines.csv",
        "global_shapley.csv", "shapley_sample.csv",
        "robust_moran_summary.csv", "robust_moran_ci.csv", "robust_premium_ci.csv",
    ]
    missing = [t for t in required if not store.exists(t)]
    assert not missing, f"missing analytics tables: {missing} -- run bash reproduce.sh"
