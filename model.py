"""Marcel-style hitter projections with component park factors and a Coors Field correction.

Three variants are built so they can be compared head to head:

  raw     Plain Marcel (Tom Tango's method): 5/4/3 weighted seasons, regression to the
          league mean, age adjustment. No park handling at all.
  park    Park-aware Marcel. Each past plate appearance is neutralized by the park's
          component factor (singles, doubles, triples, HR, walks, HBP, strikeouts) and
          the projection is re-applied to the parks the hitter plays in that season.
  coors   `park` plus Rockies-specific home and road multipliers, learned only from
          training seasons. They capture what a generic park factor misses: Rockies
          hitters gain more at Coors than visitors do, and lose more on the road
          (the so-called "Coors hangover").

Everything for target season t is computed from seasons before t, so backtests are leak-free.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import COMPONENTS, WOBA_WEIGHTS

COORS = "DEN02"
MARCEL_WEIGHTS = {1: 5, 2: 4, 3: 3}


@dataclass
class Params:
    regress_pa: float = 1200.0      # league-average PA added in Marcel's regression
    pf_shrink_pa: float = 20000.0   # PA of "park factor = 1" mixed into each park factor
    pf_window: int = 3              # seasons used for park factors
    min_park_pa: int = 3000         # parks with less data get a factor of 1
    coors_mult: dict = field(default_factory=dict)  # {("home"|"road", comp): multiplier}


# ---------------------------------------------------------------- park factors

def park_factors(pa: pd.DataFrame, t: int, p: Params) -> pd.DataFrame:
    """Component park factors for target season t, from seasons t-window .. t-1.

    factor = rate in games at the park / rate in the host team's road games,
    using both teams' plate appearances, then shrunk toward 1 by sample size.
    """
    w = pa[(pa.year >= t - p.pf_window) & (pa.year < t)]
    at_park = w.groupby(["home_team", "park"])[COMPONENTS + ["PA"]].sum()
    # Road games of team X, counting both teams' plate appearances:
    # X batting as the visitor, plus X's pitchers facing the home team.
    vis_games = w[w.home_team != w.team].assign(x=lambda d: d.team)          # X batting on road
    vis_games_def = w[w.home_team == w.team].assign(x=lambda d: d.opp)       # X pitching on road
    road = pd.concat([vis_games, vis_games_def]).groupby("x")[COMPONENTS + ["PA"]].sum()
    road_rate = road[COMPONENTS].div(road["PA"], axis=0)

    rows = []
    for (host, park), r in at_park.iterrows():
        n = r["PA"]
        if host not in road_rate.index or n < p.min_park_pa:
            f = {c: 1.0 for c in COMPONENTS}
        else:
            raw = (r[COMPONENTS] / n) / road_rate.loc[host]
            k = n / (n + p.pf_shrink_pa)
            f = {c: 1 + (raw[c] - 1) * k for c in COMPONENTS}
        rows.append({"park": park, "home_team": host, **f, "park_pa": n})
    pf = pd.DataFrame(rows)
    # a park hosted by several teams (rare) -> PA-weighted average
    pf = pf.groupby("park").apply(
        lambda g: pd.Series({c: np.average(g[c], weights=g.park_pa) for c in COMPONENTS}),
        include_groups=False)
    return pf


def _context(rows: pd.DataFrame, pf: pd.DataFrame, p: Params, use_coors: bool) -> pd.DataFrame:
    """Multiplier per row and component: park factor, times Rockies multiplier if enabled."""
    m = pd.DataFrame(1.0, index=rows.index, columns=COMPONENTS)
    known = rows.park.isin(pf.index)
    m.loc[known, COMPONENTS] = pf.loc[rows.loc[known, "park"], COMPONENTS].to_numpy()
    if use_coors and p.coors_mult:
        col = rows.team == "COL"
        home = col & (rows.park == COORS)
        road = col & (rows.park != COORS)
        for c in COMPONENTS:
            m.loc[home, c] *= p.coors_mult.get(("home", c), 1.0)
            m.loc[road, c] *= p.coors_mult.get(("road", c), 1.0)
    return m


# ---------------------------------------------------------------- projections

def project(pa: pd.DataFrame, bio: pd.DataFrame, t: int, variant: str, p: Params,
            target_rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Project component rates (per wOBA PA) for season t.

    target_rows: the plate-appearance rows (park/team mix) the projection is placed into.
    When omitted, returns neutral-park rates.
    """
    hist = pa[(pa.year >= t - 3) & (pa.year < t)].copy()
    use_park = variant in ("park", "coors")
    if use_park:
        pf = park_factors(pa, t, p)
        mult = _context(hist, pf, p, use_coors=(variant == "coors"))
        hist[COMPONENTS] = hist[COMPONENTS].to_numpy() / mult.to_numpy()

    hist["w"] = (t - hist.year).map(MARCEL_WEIGHTS)
    wc = hist[COMPONENTS + ["PA"]].mul(hist["w"], axis=0)
    wc["batter"] = hist["batter"]
    player = wc.groupby("batter").sum()

    # league average rates over the same seasons, same weights
    lg = player[COMPONENTS].sum() / player["PA"].sum()

    reg = player[COMPONENTS].add(lg * p.regress_pa, axis=1).div(player["PA"] + p.regress_pa, axis=0)

    # Marcel aging: peak at 29; +0.6%/yr younger, -0.3%/yr older, applied to positive outcomes
    age = (t - bio.set_index("batter").reindex(player.index)["birth"].dt.year).fillna(29)
    age_f = np.where(age < 29, 1 + (29 - age) * 0.006, 1 + (29 - age) * 0.003)
    pos = ["1B", "2B", "3B", "HR", "BB", "HBP"]
    reg[pos] = reg[pos].mul(age_f, axis=0)
    reg["K"] = reg["K"] / age_f
    reg["age"] = age
    reg["rel_pa"] = player["PA"]

    if target_rows is None or not use_park:
        if target_rows is not None:
            reg = reg.reindex(target_rows.batter.unique()).dropna(subset=["rel_pa"])
        return reg

    # place the neutral projection into the target season's parks
    tr = target_rows[target_rows.batter.isin(reg.index)].copy()
    pf_t = park_factors(pa, t, p)
    m = _context(tr, pf_t, p, use_coors=(variant == "coors"))
    base = reg.loc[tr.batter, COMPONENTS].to_numpy()
    exp_counts = pd.DataFrame(base * m.to_numpy() * tr[["PA"]].to_numpy(),
                              columns=COMPONENTS, index=tr.index)
    exp_counts["PA"] = tr["PA"]
    exp_counts["batter"] = tr["batter"]
    out = exp_counts.groupby("batter").sum()
    out[COMPONENTS] = out[COMPONENTS].div(out["PA"], axis=0)
    out["age"] = reg["age"]
    out["rel_pa"] = reg["rel_pa"]
    return out.drop(columns="PA")


def woba_from_rates(r: pd.DataFrame) -> pd.Series:
    return sum(r[c] * w for c, w in WOBA_WEIGHTS.items())


def learn_coors_mult(pa, bio, years, p: Params) -> dict:
    """Rockies home/road multipliers from residuals of the park-aware model.

    For each component: (observed / expected) for Rockies hitters at Coors and on the
    road, divided by the same ratio for all other teams' hitters at home / on the road
    (so ordinary home-field advantage is not attributed to Coors).
    """
    acc = {}
    for t in years:
        tgt = pa[pa.year == t]
        proj = project(pa, bio, t, "park", p)            # neutral rates
        pf = park_factors(pa, t, p)
        tr = tgt[tgt.batter.isin(proj.index)].copy()
        m = _context(tr, pf, p, use_coors=False)
        exp = proj.loc[tr.batter, COMPONENTS].to_numpy() * m.to_numpy() * tr[["PA"]].to_numpy()
        is_home = tr.home_team == tr.team
        group = np.select([(tr.team == "COL") & is_home, (tr.team == "COL") & ~is_home, is_home],
                          ["col_home", "col_road", "oth_home"], "oth_road")
        for c_i, c in enumerate(COMPONENTS):
            for g in ["col_home", "col_road", "oth_home", "oth_road"]:
                sel = group == g
                o, e = acc.get((g, c), (0.0, 0.0))
                acc[(g, c)] = (o + tr.loc[sel, c].sum(), e + exp[sel, c_i].sum())
    ratio = {k: o / e for k, (o, e) in acc.items()}
    return {("home", c): ratio[("col_home", c)] / ratio[("oth_home", c)] for c in COMPONENTS} | \
           {("road", c): ratio[("col_road", c)] / ratio[("oth_road", c)] for c in COMPONENTS}
