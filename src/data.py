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
def load_and_build(episode_data_path: str, characters_data_path: str):
    episode_data = load_json(Path(episode_data_path))           # {"episodes": [...]} or list
    characters_data = load_json(Path(characters_data_path))
    episodes_df = build_episodes(episode_data)          # episode-level only
    scenes_df   = build_scenes(episode_data)            # one row per scene (with season/episode meta)
    characters_df = build_characters(characters_data)
    return episodes_df, scenes_df, characters_df

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
    if {"seasonNum", "episodeNum"}.issubset(episodes.columns):
        s = pd.to_numeric(episodes["seasonNum"], errors="coerce")
        e = pd.to_numeric(episodes["episodeNum"], errors="coerce")
        episodes["episode_id"] = "S" + s.map("{:02.0f}".format).str.zfill(2) +\
        "E" + e.map("{:02.0f}".format).str.zfill(2)

    # Drop unnecessary columns (with the possibility of dropping others in future)
    drop_cols = drop_cols or DEFAULT_DROP_COLS
    episodes = episodes.drop(columns=[c for c in drop_cols if c in episodes.columns])

    # Ensure stable ordering
    if {"seasonNum","episodeNum"}.issubset(episodes.columns):
        episodes = episodes.sort_values(["seasonNum","episodeNum"], kind="stable").reset_index(drop=True)

    # Uniqueness check
    if "episode_id" in episodes.columns:
        assert episodes["episode_id"].is_unique, "episode_id should be unique per episode"

    return episodes

## Helper method for build_scenes
def _has_death(characters):
    if not isinstance(characters, list):
        return False
    for character in characters:
        if (isinstance(character, dict)
                and character.get('alive') == False
                and ('mannerOfDeath' in character or 'killedBy' in character)):
            return True
    return False

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
        "sceneStart": "scene_start",
        "sceneEnd": "scene_end",
        "location": "location",
        "subLocation": "sub_location",
        "altLocation": "alt_location",
        "sceneDuration": "scene_duration",
        "totalCharacters": "total_characters",
        "flashback": "is_flashback",
        # Sometimes characters are a list at "characters"
        "characters": "characters",
    }
    for k in list(rename_map):
        if k not in df.columns:
            rename_map.pop(k)
    df = df.rename(columns=rename_map)

    # Derive features (safe if columns exist)
    if {"scene_start", "scene_end"}.issubset(df.columns):
        df["scene_length_sec"] = pd.to_numeric(df["scene_end"], errors="coerce").fillna(0) - \
                         pd.to_numeric(df["scene_start"], errors="coerce").fillna(0)
    if "characters" in df.columns:
        # if characters is a list, count them
        df["num_characters"] = df["characters"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if "is_flashback" in df.columns:
        df["is_flashback"] = df["is_flashback"].astype(bool).astype(int)
    if "characters" in df.columns:
        df["death_in_scene"] = df["characters"].apply(_has_death).astype(int)

    # Create a stable episode_id to join with episodes later
    if {"seasonNum", "episodeNum"}.issubset(df.columns):
        df["episode_id"] = df["seasonNum"].astype(str) + "x" + df["episodeNum"].astype(str)

    # Extract character names, then drop the raw characters column
    if "characters" in df.columns:
        df["character_names"] = df["characters"].apply(
            lambda x: [c["name"] for c in x if isinstance(c, dict) and "name" in c]
            if isinstance(x, list) else []
    )
        df = df.drop(columns=["characters"])


    return df

def build_characters(characters_raw: list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """
    Flatten characters; many fields may be lists (parents, house, killers, killed, etc.).
    Convert key categorical/list fields into usable features.
    """

    # If wrapped like {"characters": [...]}, unwrap it
    if isinstance(characters_raw, dict):
        if "characters" not in characters_raw or not isinstance(characters_raw["characters"], list):
            raise ValueError("Expected a dict with key 'characters' containing a list.")
        characters_list = characters_raw["characters"]
    elif isinstance(characters_raw, list):
        characters_list = characters_raw
    else:
        raise TypeError("characters_raw must be a list[dict] or dict with a 'characters' key.")

    df = pd.json_normalize(characters_list, sep=".")

    ####### Create derived features
    df['is_killed'] = df['killedBy'].notnull()
    df['has_killed_others'] = df['killed'].notnull()
    # Determine if the character is an animal
    animals = ['Grey Wind', 'Lady', 'Nymeria', 'Summer', 'Shaggydog',
                    'Ghost', 'Drogon', 'Rhaegal', 'Viserion']
    df['not_human'] = df['siblings'].apply(lambda x: not set(animals).isdisjoint(x) if isinstance(x, list) else False)
            
    df['is_served'] = df['servedBy'].notnull()
    df['has_siblings'] = df['siblings'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df['is_young_version'] = ['Young' in name for name in df['characterName']]
    df['has_children'] = df['parentOf'].notnull()
    df['child_of_named_character'] = df['parents'].notnull()
    df['is_married'] = df['marriedEngaged'].notnull()
    df['has_served'] = df['serves'].notnull()
    df['is_guarded'] = df['guardedBy'].notnull()
    df['is_guardian'] = df['guardianOf'].notnull()
    df['is_royal'] = df['royal'].notnull()
    df['is_kingsguard'] = df['kingsguard'].notnull()

    possible_cols = [c for c in [
        "characterName", "houseName", "is_royal", "has_siblings",
        "is_killed", "has_killed_others", "is_served", "has_served",
        "is_kingsguard", "is_guarded", "is_guardian", "not_human",
        "is_young_version", "has_children", "child_of_named_character",
        "is_married", "died"
    ] if c in df.columns]
    df = df[possible_cols].copy()

    # Rename  for clarity
    df = df.rename(columns={
        "characterName": "character_name",
        "houseName": "house_name",
        "is_killed": "survives"  # we'll invert below
    })

    # Survives == 1 if NOT died (depends on your raw label—confirm your data!)
    if "survives" in df.columns:
        df["survives"] = df["survives"].apply(lambda x: 0 if bool(x) else 1)

    # Flatten house_name from list to single value before encoding
    if "house_name" in df.columns:
        df["house_name"] = df["house_name"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    # One-hot house if present; convert booleans to int
    cat_cols = [c for c in ["house_name"] if c in df.columns]
    df = one_hot(df, cat_cols)
    if "is_royal" in df.columns:
        df["is_royal"] = df["is_royal"].astype(bool).astype(int)

    return df

# ---------- Main ----------
def main(data_dir: str | Path, out_dir: str | Path):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # Load raw JSON
    episodes_json = load_json(data_dir / "episodes.json")
    characters_json = load_json(data_dir / "characters.json")

    # Build episodes
    df_eps = build_episodes(episodes_json)

    # Build scenes:
    df_scenes = build_scenes(episodes_json)

    # Build characters
    df_chars = build_characters(characters_json)

    # Save artifacts (Parquet for compactness; switch to CSV if you prefer)
    df_eps.to_parquet(out_dir / "episodes.parquet", index=False)
    df_scenes.to_parquet(out_dir / "scenes.parquet", index=False)
    df_chars.to_parquet(out_dir / "characters.parquet", index=False)

    print(f"Saved: {out_dir/'episodes.parquet'}, {out_dir/'scenes.parquet'}, {out_dir/'characters.parquet'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)
