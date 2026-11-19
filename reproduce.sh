#!/usr/bin/env bash
# Rebuild every EquiCity result from the documented inputs in raw_data/.
#
#   pip install -r dashboard/requirements.txt pytest==9.1.1
#   bash reproduce.sh            # full rebuild, ~25 minutes on 2 cores
#   bash reproduce.sh --fast     # skip the slow robustness bootstrap
#
# Every step is deterministic (fixed random seeds). The only network step,
# block geocoding, is replayed from the committed cache by default; delete
# processed_data/geocode_cache.jsonl to re-query OneMap (~40 minutes).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. verify inputs against raw_data/MANIFEST.csv"
python pipeline/manifest.py --check --skip-missing

echo "== 1. raw HDB files -> processed_data/hdb-all-cleaned.csv"
python pipeline/merge.py
python pipeline/clean.py
python pipeline/aggregate.py

echo "== 2. block coordinates (from cache)"
python pipeline/geocode_blocks.py --finalize

echo "== 3. hedonic backbone, rail history, block hot spots"
python pipeline/build_analytics.py

echo "== 4. price-model audit, horizons, market baselines, served model"
python pipeline/audit_price_model.py

echo "== 5. global explanations"
python pipeline/build_explanations.py

if [[ "${1:-}" != "--fast" ]]; then
  echo "== 6. spatial robustness + bootstrap uncertainty"
  python pipeline/spatial_robustness.py
fi

echo "== 7. tests"
python -m pytest tests/ -q
echo "Done. Run: streamlit run dashboard/app.py"
