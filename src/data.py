"""Load Retrosheet plate appearances into a tidy player/park/season table.

Input files are produced by Chadwick's `cwevent` and `cwgame` tools
(see README.md for the exact commands). Each plate appearance is
classified into the outcome components the model projects.
"""
from pathlib import Path

import pandas as pd

# Retrosheet event codes -> outcome components
EVENT_MAP = {20: "1B", 21: "2B", 22: "3B", 23: "HR", 14: "BB", 15: "IBB", 16: "HBP", 3: "K"}
COMPONENTS = ["1B", "2B", "3B", "HR", "BB", "HBP", "K"]   # projected rates (per wOBA PA)

# Fixed linear weights so wOBA is comparable across seasons. These are FanGraphs' 2025 wOBA
# weights, rounded (fangraphs.com/tools/guts?type=cn); their 2015-2025 average is slightly
# lower for 2B/3B/HR (about 1.24 / 1.57 / 2.02), which moves a hitter's wOBA by a point at most.
WOBA_WEIGHTS = {"BB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25, "3B": 1.58, "HR": 2.03}


def load_pa(data_dir: Path, years) -> pd.DataFrame:
    """One row per (year, batter, batting team, park, home flag) with outcome counts."""
    frames = []
    for y in years:
        ev = pd.read_csv(data_dir / "ev" / f"{y}.csv", header=None,
                         names=["game_id", "vis", "bat_home", "batter", "event", "pa_fl", "sh_fl"])
        gm = pd.read_csv(data_dir / "gm" / f"{y}.csv", header=None, names=["game_id", "date", "park"])
        ev = ev.merge(gm[["game_id", "park"]], on="game_id", how="left")
        ev["home_team"] = ev["game_id"].str[:3]
        ev["team"] = ev["home_team"].where(ev["bat_home"] == 1, ev["vis"])
        ev["opp"] = ev["vis"].where(ev["bat_home"] == 1, ev["home_team"])
        ev["comp"] = ev["event"].map(EVENT_MAP).fillna("OUT")
        # wOBA plate appearances exclude intentional walks and sacrifice bunts
        ev["wpa"] = ((ev["comp"] != "IBB") & (ev["sh_fl"] != "T")).astype(int)
        ev = ev[ev["wpa"] == 1]
        agg = (ev.groupby(["batter", "team", "opp", "home_team", "park", "bat_home", "comp"])
                 .size().unstack("comp", fill_value=0).reset_index())
        agg["year"] = y
        frames.append(agg)
    df = pd.concat(frames, ignore_index=True).fillna(0)
    for c in COMPONENTS + ["OUT"]:
        if c not in df:
            df[c] = 0
    df = df.drop(columns=[c for c in ["IBB"] if c in df])
    df["PA"] = df[COMPONENTS + ["OUT"]].sum(axis=1)
    return df


def load_bio(data_dir: Path) -> pd.DataFrame:
    bio = pd.read_csv(data_dir / "retrosheet" / "reference" / "biofile.csv",
                      usecols=["PLAYERID", "LAST", "NICKNAME", "BIRTHDATE"])
    bio["birth"] = pd.to_datetime(bio["BIRTHDATE"], format="%m/%d/%Y", errors="coerce")
    bio["name"] = bio["NICKNAME"].fillna("") + " " + bio["LAST"].fillna("")
    return bio.rename(columns={"PLAYERID": "batter"})[["batter", "name", "birth"]]


def woba(counts: pd.DataFrame) -> pd.Series:
    num = sum(counts[c] * w for c, w in WOBA_WEIGHTS.items())
    return num / counts["PA"]
