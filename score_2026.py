"""Score 2026 projections for Rockies hitters against their actual 2026 lines.

Actual lines come from data/rockies_2026_actual.csv (Fox Sports team batting page,
regular season). That source has no HBP, IBB or SF columns, so:
  HBP is backed out of OBP (assuming no sacrifice bunts), and
  intentional walks are counted as ordinary walks (a slight upward bias in actual wOBA).
Rookies with no MLB plate appearances before 2026 are skipped: there's nothing to project from.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import COMPONENTS, WOBA_WEIGHTS, load_bio, load_pa  # noqa: E402
from model import COORS, Params, learn_coors_mult, park_factors, project  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path.home() / "data"
IDS = {"Connor Norby": "norbc001", "Jake McCarthy": "mccaj003", "Troy Johnston": "johnt003",
       "Willi Castro": "castw003", "Kyle Karros": "karrk001", "Tyler Freeman": "freet001",
       "Hunter Goodman": "goodh001", "Mickey Moniak": "monim001", "Brett Sullivan": "sullb001",
       "Ezequiel Tovar": "tovae001", "Brenton Doyle": "doylb001", "Edouard Julien": "julie001",
       "Jordan Beck": "beckj003"}
MIN_PA = 100

a = pd.read_csv(ROOT / "data" / "rockies_2026_actual.csv")
a = a[a.name.isin(IDS) & (a.PA >= MIN_PA)].copy()
a["batter"] = a.name.map(IDS)
a["HBP"] = (a.OBP * a.PA - a.H - a.BB).round().clip(lower=0)
a["1B"] = a.H - a["2B"] - a["3B"] - a.HR
a["actual"] = sum(a[c] * w for c, w in WOBA_WEIGHTS.items()) / a.PA

pa = load_pa(DATA, range(2012, 2026))
bio = load_bio(DATA)
p = Params(regress_pa=2000, pf_shrink_pa=5000)          # values tuned in backtest.py
p.coors_mult = learn_coors_mult(pa, bio, list(range(2015, 2020)) + list(range(2021, 2026)), p)
pf = park_factors(pa, 2026, p).loc[COORS, COMPONENTS]
mult = {s: pd.Series({c: p.coors_mult[(s, c)] for c in COMPONENTS}) for s in ("home", "road")}

raw = project(pa, bio, 2026, "raw", p)
neu_park = project(pa, bio, 2026, "park", p)    # neutral rates, park factors only
neu = project(pa, bio, 2026, "coors", p)        # neutral rates, park + Rockies multipliers


def w(s: pd.Series) -> float:
    return float(sum(float(s[c]) * wt for c, wt in WOBA_WEIGHTS.items()))
prev_team = (pa[pa.year == 2025].groupby(["batter", "team"]).PA.sum().reset_index()
               .sort_values("PA").drop_duplicates("batter", keep="last").set_index("batter").team)

rows = []
for _, r in a.iterrows():
    b = r.batter
    n, n_p = neu.loc[b, COMPONENTS], neu_park.loc[b, COMPONENTS]
    # a Rockies season = half the PA at Coors, half on the road (road parks average ~1)
    rows.append({
        "name": r["name"], "team_2025": prev_team.get(b, "—"), "pa_2026": int(r.PA),
        "plain_marcel": w(raw.loc[b]),
        "park_adjusted": (w(n_p * pf) + w(n_p)) / 2,
        "park_rockies": (w(n * pf * mult["home"]) + w(n * mult["road"])) / 2,
        "actual": r.actual})
df = pd.DataFrame(rows).sort_values("pa_2026", ascending=False)
df["new_to_col"] = df.team_2025 != "COL"

def summary(d, col):
    err = d.actual - d[col]
    return np.average(err, weights=d.pa_2026), np.sqrt(np.average(err ** 2, weights=d.pa_2026))

print(df.round(3).to_string(index=False))
for label, d in [("all", df), ("stayed", df[~df.new_to_col]), ("joined", df[df.new_to_col])]:
    print(f"\n{label} (n={len(d)})")
    for col in ["plain_marcel", "park_adjusted", "park_rockies"]:
        b, e = summary(d, col)
        print(f"  {col:14} bias {b*1000:+.1f} pts  RMSE {e*1000:.1f} pts")
df.round(4).to_csv(ROOT / "output" / "rockies_2026_scored.csv", index=False)
