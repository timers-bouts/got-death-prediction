# src/data.py
"""
Data loading, wrangling, and feature engineering.

Usage:
  python -m src.data --data_dir data --out_dir reports
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import pandas as pd
from typing import Any
from src.utils import ensure_dir

# ---------- Helpers ----------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def one_hot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return df
    df_cat = pd.get_dummies(df[cols].astype("category"), dummy_na=False)
    df_num = df.drop(columns=cols)
    return pd.concat([df_num, df_cat], axis=1)

# ---------- Builders ----------
# orchestrator
def load_and_build(data_path: str):
    data = load_json(Path(data_path))           # {"episodes": [...]} or list
    episodes_df = build_episodes(data)          # episode-level only
    scenes_df   = build_scenes(data)            # one row per scene (with season/episode meta)
    return episodes_df, scenes_df

# Default drop columns drops scenes, because build_scenes takes raw JSON data
DEFAULT_DROP_COLS = ["openingSequenceLocations",
                'episodeLink',
                'episodeDescription',
                'scenes']

def build_episodes(episodes_raw: list[dict[str, Any]] | dict[str, Any],
                   drop_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Flatten the episode-level data into a DataFrame.
    Scenes are handled elsewhere (e.g., build_scenes(...)).

    Parameters
    ----------
    episodes_raw : list[dict] or dict
        Either a list of episode dicts or a dict containing an "episodes" list.

    Returns
    -------
    pd.DataFrame
        One row per episode, episode-level fields only.
    """

    # If wrapped like {"episodes": [...]}, unwrap it
    if isinstance(episodes_raw, dict):
        if "episodes" not in episodes_raw or not isinstance(episodes_raw["episodes"], list):
            raise ValueError("Expected a dict with key 'episodes' containing a list.")
        episodes_list = episodes_raw["episodes"]
    elif isinstance(episodes_raw, list):
        episodes_list = episodes_raw
    else:
        raise TypeError("episodes_raw must be a list[dict] or dict with an 'episodes' key.")

    # Normalize the episodes list into a flat table
    episodes = pd.json_normalize(episodes_list, sep=".")

    # Create a stable ID for joins/plots (only if season/episode present)
    if {"season", "episode"}.issubset(episodes.columns):
        s = pd.to_numeric(episodes["season"], errors="coerce")
        e = pd.to_numeric(episodes["episode"], errors="coerce")
        episodes["episode_id"] = "S" + s.map("{:02.0f}".format).str.zfill(2) +\
        "E" + e.map("{:02.0f}".format).str.zfill(2)

    # Drop unnecessary columns (with the possibility of dropping others in future)
    drop_cols = drop_cols or DEFAULT_DROP_COLS
    episodes = episodes.drop(columns=[c for c in drop_cols if c in episodes.columns])

    # Ensure stable ordering
    if {"season","episode"}.issubset(episodes.columns):
        episodes = episodes.sort_values(["season","episode"], kind="stable").reset_index(drop=True)

    # Uniqueness check
    if "episode_id" in episodes.columns:
        assert episodes["episode_id"].is_unique, "episode_id should be unique per episode"

    return episodes

def build_scenes(episodes_raw: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """
    If scenes are nested within episodes.json like episodes[i]['scenes'], extract them.
    """

    # If wrapped like {"episodes": [...]}, unwrap it
    if isinstance(episodes_raw, dict):
        if "episodes" not in episodes_raw or not isinstance(episodes_raw["episodes"], list):
            raise ValueError("Expected a dict with key 'scenes' containing a list.")
        episodes_list = episodes_raw["episodes"]
    elif isinstance(episodes_raw, list):
        episodes_list = episodes_raw
    else:
        raise TypeError("episodes_raw must be a list[dict] or dict with an 'episodes' key.")

    # record_path points to the nested list; meta pulls down episode fields needed for joins
    df = pd.json_normalize(
        episodes_list,
        record_path=["scenes"],
        meta=["seasonNum", "episodeNum", "episodeTitle"],
        sep="."
    )
    # Standardize column names you need later:
    # Example column assumptions (EDIT to your actual names):
    # start/end times, location, list of characters, has_death flag, etc.
    rename_map = {
        "start": "scene_start",
        "end": "scene_end",
        "location": "location",
        "sublocation": "sublocation",
        "isFlashback": "is_flashback",
        "hasDeath": "has_death",
        # Sometimes characters are a list at "characters"
        "characters": "characters",
    }
    for k in list(rename_map):
        if k not in df.columns:
            rename_map.pop(k)
    df = df.rename(columns=rename_map)

    # Derive features (safe if columns exist)
    if {"scene_start", "scene_end"}.issubset(df.columns):
        df["scene_length_sec"] = df["scene_end"].fillna(0) - df["scene_start"].fillna(0)
    if "characters" in df.columns:
        # if characters is a list, count them
        df["num_characters"] = df["characters"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if "is_flashback" in df.columns:
        df["is_flashback"] = df["is_flashback"].astype(bool).astype(int)

    # Optional: create a stable episode_id to join with episodes later
    if {"seasonNum", "episodeNum"}.issubset(df.columns):
        df["episode_id"] = df["seasonNum"].astype(str) + "x" + df["episodeNum"].astype(str)


    return df

def build_characters(characters_raw: list[dict]) -> pd.DataFrame:
    """
    Flatten characters; many fields may be lists (parents, house, killers, killed, etc.).
    Convert key categorical/list fields into usable features.
    """
    df = pd.json_normalize(characters_raw, sep=".")
    # Example: pick a minimal subset + derive
    # EDIT keys to your JSON:
    possible_cols = [c for c in [
        "name", "house", "isRoyal", "isHuman", "died", "screenTimeSec"
    ] if c in df.columns]
    df = df[possible_cols].copy()

    # Rename a few for clarity
    df = df.rename(columns={
        "isRoyal": "is_royal",
        "isHuman": "is_human",
        "screenTimeSec": "screen_time_sec",
        "died": "survives"  # we'll invert below
    })

    # Survives == 1 if NOT died (depends on your raw label—confirm your data!)
    if "survives" in df.columns:
        df["survives"] = df["survives"].apply(lambda x: 0 if bool(x) else 1)

    # One-hot house if present; convert booleans to int
    cat_cols = [c for c in ["house"] if c in df.columns]
    df = one_hot(df, cat_cols)
    for b in ["is_royal", "is_human"]:
        if b in df.columns:
            df[b] = df[b].astype(bool).astype(int)

    return df

# ---------- Main ----------
def main(data_dir: str | Path, out_dir: str | Path):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # Load raw JSON
    episodes_json = load_json(data_dir / "episodes.json")     # <-- confirm filenames
    scenes_json   = None  # if you also have a separate scenes.json, load it here
    characters_json = load_json(data_dir / "characters.json")

    # Build episodes
    df_eps = build_episodes(episodes_json)

    # Build scenes:
    # Option A: scenes nested in episodes.json (most likely in your case)
    df_scenes = build_scenes(episodes_json)
    # Option B: if you also have scenes.json separately, you could merge/align here

    # Build characters
    df_chars = build_characters(characters_json)

    # Final post-processing: minimal example of one-hot on scene categorical columns
    scene_cats = [c for c in ["location", "sublocation"] if c in df_scenes.columns]
    df_scenes_oh = one_hot(df_scenes, scene_cats)

    # Save artifacts (Parquet for compactness; switch to CSV if you prefer)
    df_eps.to_parquet(out_dir / "episodes.parquet", index=False)
    df_scenes_oh.to_parquet(out_dir / "scenes.parquet", index=False)
    df_chars.to_parquet(out_dir / "characters.parquet", index=False)

    print(f"Saved: {out_dir/'episodes.parquet'}, {out_dir/'scenes.parquet'}, {out_dir/'characters.parquet'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)
