# src/models.py
"""
Model training and evaluation for GoT death prediction.

Trains 5 classifiers on two tasks:
  1. Character survival prediction
  2. Scene death prediction

Usage:
  python -m src.models --data_dir reports --out_dir reports --model_dir models
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir


# ---------- Data ----------

def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three Parquet artifacts produced by src.data."""
    data_dir = Path(data_dir)
    characters = pd.read_parquet(data_dir / "characters.parquet")
    scenes     = pd.read_parquet(data_dir / "scenes.parquet")
    episodes   = pd.read_parquet(data_dir / "episodes.parquet")
    return characters, scenes, episodes


# ---------- Splitting ----------

def split(df: pd.DataFrame,
          target: str,
          test_size: float = 0.3,
          val_size: float = 0.3,
          random_state: int = 42
          ) -> tuple:
    """
    Split a DataFrame into train, validation, and test sets.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    y = df[target]
    X = df.drop(columns=[target])

    # Drop any remaining non-numeric or list columns that can't be used by sklearn
    X = X.select_dtypes(include=["number", "bool"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------- Evaluation ----------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return accuracy and a full classification report for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "classification_report": metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


# ---------- Model saving ----------

def save_model(model, path: Path) -> None:
    """Serialize a fitted model to disk using joblib."""
    joblib.dump(model, path)
    print(f"Saved model: {path}")


# ---------- Character survival ----------

def train_character_model(characters: pd.DataFrame,
                          model_dir: Path
                          ) -> dict:
    """
    Train and evaluate 5 classifiers on character survival prediction.
    Saves the best model to model_dir/character_best.joblib.

    Target column: 'is_killed' (1 = character is killed, 0 = character is not killed)
    """
    target = "is_killed"

    # Drop name column — not a feature
    df = characters.drop(columns=["character_name"], errors="ignore")

    X_train, X_val, X_test, y_train, y_val, y_test = split(df, target)

    models = {
        "decision_tree": DecisionTreeClassifier(
            criterion="entropy", splitter="random", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            criterion="entropy", n_estimators=150, random_state=42
        ),
        "svc": Pipeline([
            ("scaler", StandardScaler()),
            ("svc_clf", SVC(kernel="poly", C=10, coef0=1, degree=3, random_state=42)),
        ]),
        "naive_bayes": make_pipeline(MultinomialNB()),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    best_name, best_model, best_val_f1 = None, None, -1

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            val_f1 = metrics.f1_score(y_val, model.predict(X_val), average="weighted", zero_division=0)
            results[name] = {"val_f1": val_f1, **evaluate(model, X_test, y_test)}
            print(f"[character] {name}: val_f1={val_f1:.4f}  test={results[name]['accuracy']:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_name = name
                best_model = model
        except Exception as e:
            print(f"[character] {name} failed: {e}")
            results[name] = {"error": str(e)}
    
    # Always save Random Forest for feature importance visualization
    rf = models["random_forest"]
    if hasattr(rf, 'feature_importances_'):
        save_model(rf, model_dir / "character_rf.joblib")


    if best_model is not None:
        save_model(best_model, model_dir / "character_best.joblib")
        results["best_model"] = best_name

    return results


# ---------- Scene death prediction ----------

def train_scene_model(scenes: pd.DataFrame,
                      model_dir: Path
                      ) -> dict:
    """
    Train and evaluate 5 classifiers on scene death prediction.
    Saves the best model to model_dir/scene_best.joblib.

    Target column: 'death_in_scene' (1 = death occurred, 0 = no death)
    """
    target = "death_in_scene"

    # Drop non-feature columns
    drop_cols = ["character_names", "episodeTitle", "episode_id"]
    df = scenes.drop(columns=[c for c in drop_cols if c in scenes.columns])

    X_train, X_val, X_test, y_train, y_val, y_test = split(df, target)

    models = {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini", splitter="random", class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced", random_state=42
        ),
        "svc": Pipeline([
            ("scaler", StandardScaler()),
            ("svc_clf", SVC(kernel="sigmoid", C=1, coef0=1, degree=3, random_state=42, class_weight="balanced")),
        ]),
        "naive_bayes": make_pipeline(MultinomialNB()),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    }

    results = {}
    best_name, best_model, best_val_f1 = None, None, -1

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            val_f1 = metrics.f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
            results[name] = {"val_f1": val_f1, **evaluate(model, X_test, y_test)}
            print(f"[scene] {name}: val_f1={val_f1:.4f}  test={results[name]['accuracy']:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_name = name
                best_model = model
        except Exception as e:
            print(f"[scene] {name} failed: {e}")
            results[name] = {"error": str(e)}
    
    # Always save Random Forest for feature importance visualization
    rf = models["random_forest"]
    if hasattr(rf, 'feature_importances_'):
        save_model(rf, model_dir / "scene_rf.joblib")


    if best_model is not None:
        save_model(best_model, model_dir / "scene_best.joblib")
        results["best_model"] = best_name

    return results


# ---------- Main ----------

def main(data_dir: str | Path, out_dir: str | Path, model_dir: str | Path) -> None:
    data_dir  = Path(data_dir)
    out_dir   = Path(out_dir)
    model_dir = Path(model_dir)
    ensure_dir(out_dir)
    ensure_dir(model_dir)

    characters, scenes, _ = load_data(data_dir)

    print("\n=== Character Survival ===")
    char_results = train_character_model(characters, model_dir)

    print("\n=== Scene Death Prediction ===")
    scene_results = train_scene_model(scenes, model_dir)

    # Save all metrics to a single JSON file
    all_results = {
        "character_survival": char_results,
        "scene_death": scene_results,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="reports")
    ap.add_argument("--out_dir",   default="reports")
    ap.add_argument("--model_dir", default="models")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.model_dir)
