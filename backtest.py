"""Backtest the three variants, tune on 2015-2019, evaluate on 2021-2025, project 2026."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import COMPONENTS, load_bio, load_pa, woba  # noqa: E402
from model import Params, learn_coors_mult, project, woba_from_rates  # noqa: E402

DATA = Path.home() / "data"
OUT = Path(__file__).resolve().parent.parent / "output"
TRAIN = [2015, 2016, 2017, 2018, 2019]
TEST = [2021, 2022, 2023, 2024, 2025]     # 2020 (60 games) is used as input only
MIN_PA = 200


def score(pa, bio, years, variant, p):
    """Per-player predictions vs actuals for the given target seasons."""
    rows = []
    for t in years:
        tgt = pa[pa.year == t]
        actual = tgt.groupby("batter")[COMPONENTS + ["PA"]].sum()
        actual = actual[actual.PA >= MIN_PA]
        proj = project(pa, bio, t, variant, p, target_rows=tgt[tgt.batter.isin(actual.index)])
        proj = proj[proj.rel_pa > 0]
        a = actual.loc[proj.index]
        # team = the club with most of the hitter's PA that season; prev = same for t-1
        team = tgt.groupby(["batter", "team"]).PA.sum().reset_index() \
                  .sort_values("PA").drop_duplicates("batter", keep="last").set_index("batter").team
        prev = pa[pa.year == t - 1].groupby(["batter", "team"]).PA.sum().reset_index() \
                  .sort_values("PA").drop_duplicates("batter", keep="last").set_index("batter").team
        df = pd.DataFrame({
            "year": t, "pa": a.PA, "woba": woba(a), "pred": woba_from_rates(proj),
            "hr": a.HR / a.PA, "pred_hr": proj.HR,
            "team": team.reindex(proj.index), "prev_team": prev.reindex(proj.index)})
        rows.append(df)
    return pd.concat(rows).rename_axis("batter").reset_index()


def rmse(df, col="woba", pred="pred"):
    w = df.pa
    return float(np.sqrt(np.average((df[col] - df[pred]) ** 2, weights=w)))


def bias(df, col="woba", pred="pred"):
    return float(np.average(df[col] - df[pred], weights=df.pa))


def segments(df):
    col = df.team == "COL"
    return {
        "all hitters": df,
        "Rockies hitters": df[col],
        "joined Rockies": df[col & df.prev_team.notna() & (df.prev_team != "COL")],
        "left Rockies": df[~col & (df.prev_team == "COL")],
    }


def main():
    OUT.mkdir(exist_ok=True)
    pa = load_pa(DATA, range(2012, 2026))
    bio = load_bio(DATA)

    # --- 1. tune the two generic hyperparameters on training seasons (park variant)
    best, tuning = None, []
    for reg_pa, shrink in itertools.product([800, 1200, 1600, 2000, 2400, 3000], [5000, 20000]):
        p = Params(regress_pa=reg_pa, pf_shrink_pa=shrink)
        e = rmse(score(pa, bio, TRAIN, "park", p))
        print(f"tune regress_pa={reg_pa:5} pf_shrink={shrink:6}  RMSE={e:.4f}")
        tuning.append({"regress_pa": reg_pa, "pf_shrink_pa": shrink, "rmse": e})
        if best is None or e < best[0]:
            best = (e, reg_pa, shrink)
    p = Params(regress_pa=best[1], pf_shrink_pa=best[2])

    # --- 2. learn Rockies multipliers on training seasons only
    p.coors_mult = learn_coors_mult(pa, bio, TRAIN, p)
    print("Coors multipliers:", {f"{k[0]}_{k[1]}": round(v, 3) for k, v in p.coors_mult.items()})

    # --- 3. evaluate all variants on held-out seasons
    results, preds = {}, {}
    for v in ["raw", "park", "coors"]:
        preds[v] = score(pa, bio, TEST, v, p)
        results[v] = {name: {"n": int(len(s)), "rmse": rmse(s), "bias": bias(s),
                             "hr_rmse": rmse(s, "hr", "pred_hr")}
                      for name, s in segments(preds[v]).items()}
    # paired bootstrap: is the improvement bigger than chance?
    rng = np.random.default_rng(7)
    for name in results["raw"]:
        segs = {v: segments(preds[v])[name].set_index(["batter", "year"]) for v in preds}
        idx = segs["raw"].index
        for a, b in [("raw", "park"), ("park", "coors")]:
            da, db = segs[a].loc[idx], segs[b].loc[idx]
            diffs = []
            for _ in range(2000):
                s = rng.integers(0, len(idx), len(idx))
                diffs.append(rmse(db.iloc[s]) - rmse(da.iloc[s]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            results[b][name][f"vs_{a}_ci"] = [float(lo), float(hi)]

    # noise floor: RMSE a perfect projection would still have, from binomial sampling alone
    d = preds["park"]
    lg = pa[pa.year.isin(TEST)][COMPONENTS + ["PA"]].sum()
    r = lg[COMPONENTS] / lg.PA
    from data import WOBA_WEIGHTS
    m1 = sum(r[c] * w for c, w in WOBA_WEIGHTS.items())
    m2 = sum(r[c] * w * w for c, w in WOBA_WEIGHTS.items())
    noise = float(np.sqrt(np.average((m2 - m1 ** 2) / d.pa, weights=d.pa)))
    print(f"\nnoise floor (sampling error only): {noise:.4f}")

    for name in results["raw"]:
        print(f"\n{name} (n={results['raw'][name]['n']})")
        for v in results:
            r = results[v][name]
            ci = "".join(f"  Δ vs {k[3:-3]} 95% CI [{x[0]:+.4f}, {x[1]:+.4f}]"
                         for k, x in r.items() if k.endswith("_ci"))
            print(f"  {v:6} wOBA RMSE {r['rmse']:.4f}  bias {r['bias']:+.4f}  HR% RMSE {r['hr_rmse']:.4f}{ci}")

    # --- 4. 2026 projections (neutral park and Coors context) for 2025 Rockies hitters
    p_final = Params(regress_pa=p.regress_pa, pf_shrink_pa=p.pf_shrink_pa)
    p_final.coors_mult = learn_coors_mult(pa, bio, TRAIN + TEST, p_final)   # refit on all data
    neutral = project(pa, bio, 2026, "coors", p_final)
    col25 = pa[(pa.year == 2025) & (pa.team == "COL")].groupby("batter").PA.sum()
    col25 = col25[col25 >= 100]
    # a Rockies season: half the PA at Coors, half spread over road parks (factor ~1)
    from model import COORS, park_factors
    pf = park_factors(pa, 2026, p_final).loc[COORS]
    proj = neutral.loc[neutral.index.intersection(col25.index)].copy()
    home = proj[COMPONENTS] * pf[COMPONENTS] * pd.Series({c: p_final.coors_mult[("home", c)] for c in COMPONENTS})
    road = proj[COMPONENTS] * pd.Series({c: p_final.coors_mult[("road", c)] for c in COMPONENTS})
    table = pd.DataFrame({
        "pa_2025": col25.reindex(proj.index),
        "age_2026": proj.age,
        "woba_neutral": woba_from_rates(proj),
        "woba_coors_home": woba_from_rates(home),
        "woba_road": woba_from_rates(road),
        "hr_pct_home": home.HR, "hr_pct_road": road.HR})
    table["woba_rockies"] = (table.woba_coors_home + table.woba_road) / 2
    table = table.join(bio.set_index("batter")["name"]).sort_values("pa_2025", ascending=False)
    table.round(4).to_csv(OUT / "rockies_2026_projections.csv")
    pf_out = park_factors(pa, 2026, p_final).round(3)
    pf_out.to_csv(OUT / "park_factors_2026.csv")

    for v, d in preds.items():
        d.to_csv(OUT / f"backtest_{v}.csv", index=False)
    json.dump({"params": {"regress_pa": p.regress_pa, "pf_shrink_pa": p.pf_shrink_pa},
               "coors_mult_train": {f"{k[0]}_{k[1]}": v for k, v in p.coors_mult.items()},
               "coors_mult_final": {f"{k[0]}_{k[1]}": v for k, v in p_final.coors_mult.items()},
               "coors_park_factor_2026": pf[COMPONENTS].to_dict(),
               "noise_floor": noise, "tuning": tuning,
               "results": results}, open(OUT / "results.json", "w"), indent=2)
    print("\n", table.head(15).to_string())


if __name__ == "__main__":
    main()
