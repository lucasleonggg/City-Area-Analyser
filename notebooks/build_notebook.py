"""Builds and executes notebooks/reproduce_headline_findings.ipynb.

    python notebooks/build_notebook.py

The notebook recomputes every headline number in the research paper from
the published analytics tables, so a reader can check them without
running the full pipeline.
"""
import os

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md("# Reproducing EquiCity's headline findings\n\nEvery number in the research paper, recomputed "
       "from the aggregated tables in `processed_data/analytics/` and the income series in "
       "`raw_data/`. To rebuild those tables from the raw transactions, run `bash reproduce.sh`."),
    code("import os, sys\nimport numpy as np, pandas as pd\n"
         "ROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n"
         "sys.path.insert(0, os.path.join(ROOT, 'dashboard'))\n"
         "from analysis import affordability as A, spatial as S\n"
         "T = lambda n: pd.read_csv(os.path.join(ROOT, 'processed_data', 'analytics', n))\n"
         "pd.set_option('display.width', 140)"),
    md("## 1. The index tracks HDB's official Resale Price Index"),
    code("from analysis.temporal import national_quarterly_vs_official\n"
         "v = national_quarterly_vs_official().dropna()\n"
         "d = np.corrcoef(np.diff(np.log(v.equicity)), np.diff(np.log(v.official_rpi.astype(float))))[0, 1]\n"
         "print(f'quarterly-change correlation {d:.3f}; latest quarter {v.quarter.iloc[-1]}: '\n"
         "      f'EquiCity {v.equicity.iloc[-1]:.1f} vs official {float(v.official_rpi.iloc[-1]):.1f}')"),
    md("## 2. Affordability of the national constant-quality 4-room flat (HDB loan, 30% MSR)"),
    code("t = T('hedonic_town_year_type.csv'); t4 = t[(t.flat_type == '4-Room') & (t.n_town >= 30)]\n"
         "w = t4[t4.year.between(2015, 2025)].groupby('town').n_town.sum(); w /= w.sum()\n"
         "nat = t4.assign(w=t4.town.map(w).fillna(0)).groupby('year').apply(\n"
         "    lambda d: np.exp((d.log_price_ref * d.w).sum() / d.w.sum()), include_groups=False)\n"
         "rows = []\n"
         "for y in [2000, 2005, 2010, 2013, 2015, 2019, 2023, 2025, 2026]:\n"
         "    e50, e20 = A.evaluate(nat[y], y, 50), A.evaluate(nat[y], y, 20)\n"
         "    rows.append((y, round(nat[y]), round(e50['msr_pct'], 1), round(e20['msr_pct'], 1),\n"
         "                 round(e50['required_percentile']), round(e50['price_to_annual_income'], 2)))\n"
         "pd.DataFrame(rows, columns=['year', 'price', 'MSR at P50 %', 'MSR at P20 %', 'required pct', 'price/income'])"),
    code("af30 = A.affordability_frame(t4.rename(columns={'price_ref': 'price'}), 30)\n"
         "af30.groupby('year').within_cap.sum().loc[[2013, 2015, 2019, 2022, 2025, 2026]]"),
    md("## 3. Spatial clustering of location premia"),
    code("rs = T('robust_moran_summary.csv')\n"
         "print(f\"{(rs.change > 0).sum()} of {len(rs)} specifications rise; \"\n"
         "      f\"{((rs.mann_kendall_p < 0.05) & (rs.mann_kendall_z > 0)).sum()} significantly\")\n"
         "rs[['spec', 'I_1995_99', 'I_2022_26', 'change', 'mann_kendall_p']].round(3)"),
    code("T('robust_moran_ci.csv').query('year in [1995, 2005, 2015, 2019, 2026]').round(3)"),
    code("T('robust_block_moran.csv').pivot(index='window', columns='band_km', values='I').round(3)"),
    md("## 4. Regional share of cross-town inequality and price dispersion"),
    code("from data.loader import TOWN_COORDS\n"
         "out = []\n"
         "for y, g in t4[t4.town.isin(TOWN_COORDS)].groupby('year'):\n"
         "    if y < 1995: continue\n"
         "    ineq = S.inequality(g.price_ref.values, g.region.values)\n"
         "    out.append((y, round(ineq['cv_pct'], 1), round(ineq['between_share_pct'], 0)))\n"
         "pd.DataFrame(out, columns=['year', 'CV %', 'between-region share %']).set_index('year').loc[[1995, 2005, 2015, 2019, 2022, 2026]]"),
    md("## 5. Lease discount"),
    code("T('hedonic_fit_summary.csv').set_index('year').lease_60_vs_90.loc[[2004, 2010, 2015, 2019, 2022, 2026]]"),
    md("## 6. Rail access and the station event study (descriptive)"),
    code("a = T('rail_access_town_year.csv'); a[a.town == 'All HDB blocks'].set_index('year')[['share_within_1km']].loc[[1990, 2000, 2010, 2020, 2026]].round(1)"),
    code("T('rail_event_study.csv').round(2)"),
    md("## 7. Price model"),
    code("ov = T('audit_backtest_overall.csv'); ov[ov['mode'] == 'forecast'][['model', 'mape_pct', 'within_10pct', 'r2']]"),
    code("T('audit_horizons.csv')"),
    code("T('market_forecast_baselines.csv')"),
]
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
out = os.path.join(HERE, "reproduce_headline_findings.ipynb")
nbformat.write(nb, out)
print("wrote", out)
