# src/visualize.py

"""
Visualizations of model outcomes for GoT death prediction.

Usage:
  python -m src.visualize --data_dir reports --out_dir reports --model_dir models
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import joblib

from pathlib import Path
from typing import Any
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay

from src.utils import ensure_dir
from src.models import load_data, split
from src.data import load_json

def load_models(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    return {
        "character": joblib.load(model_dir / "character_best.joblib"),
        "scene": joblib.load(model_dir / "scene_best.joblib")
    }

# ---------- Visualizations ----------

def plot_class_distribution(df: pd.DataFrame, figure_prefix: str, title: str, out_dir: Path, target: str, class_names: list[str]) -> None:

    figure_name = figure_prefix + "_class_distribution.png"
    counts = df[target].value_counts()
    ax = counts.sort_index().plot(kind="bar")
    # Plot actual data value along with bar
    for bar in ax.patches:
        if isinstance(bar, Rectangle):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{int(bar.get_height())}", ha="center", va="bottom")

    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks([0, 1], class_names, rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / figure_name)
    plt.close()

def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series, figure_prefix: str, title: str, out_dir: Path) -> None:
    figure_name = figure_prefix + "_confusion_matrix.png"
    y_pred = model.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / figure_name)
    plt.close()

def plot_model_comparison(results: Any, figure_prefix: str, title: str, out_dir: Path, json_key: str):

    figure_name = figure_prefix + "_model_comparison.png"

    # Build a small DataFrame from the metrics (json)
    names = []
    scores = []
    for name, metrics in results[json_key].items():
        if "accuracy" in metrics:
            names.append(name)
            scores.append(metrics["accuracy"])

    df = pd.DataFrame({"model": names, "accuracy": scores})
    ax = df.sort_values("accuracy").plot(kind="barh", x="model", y="accuracy")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    # Plot actual data value along with bar
    for bar in ax.patches:
        if isinstance(bar, Rectangle):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}", va="center")

    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / figure_name)
    plt.close()

def plot_feature_importance(model: Any, X_train: list[str], figure_prefix: str, title: str, out_dir: Path):

    if isinstance(model, RandomForestClassifier):
        figure_name = figure_prefix + "_feature_importance.png"
        importances = model.feature_importances_  # numpy array, one value per feature
        nonzero = np.where(importances > 0)[0]  # indices of non-zero features
        indices = nonzero[np.argsort(importances[nonzero])]  # sort those by importance
        indices = indices[-15:]  # take top 15 (or fewer if less than 15 are non-zero)
        plt.figure(figsize=(12, 8))
        # Plot the actual values along with the bars
        plt.barh(range(len(indices)), importances[indices])
        ax = plt.gca()
        ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
        for bar in ax.patches:
            if isinstance(bar, Rectangle):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{bar.get_width():.4f}", va="center")


        plt.yticks(range(len(indices)), [X_train[i] for i in indices])
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.title(title)
        plt.savefig(out_dir / figure_name)
        plt.close()
    else:
        print("Best model is not a Random Forest, skipping feature importance plot")

def plot_precision_recall(model: Any, X_test: pd.DataFrame, y_test: pd.Series, figure_prefix: str, title: str, out_dir: Path) -> None:
    
    figure_name = figure_prefix + "_precision_recall.png"
    disp = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    disp.plot()

    # Plot a baseline (no-skill classifier)
    plt.axhline(y=y_test.mean(), color="r", linestyle="--", label="Baseline")
    plt.legend()

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / figure_name)
    plt.close()


# ---------- Main ----------

def main(data_dir: str | Path, out_dir: str | Path, model_dir: str | Path) -> None:
    
    # Configure and check directories
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    
    # Prepare data for plots
    characters_df, scenes_df, _ = load_data(data_dir)
    models = load_models(model_dir)
    characters_df = characters_df.drop(columns=["character_name"])
    scenes_df = scenes_df.drop(columns=["character_names", "episodeTitle", "episode_id"])
    results = load_json(data_dir / "metrics.json")
    char_rf_model = joblib.load(model_dir / "character_rf.joblib")
    scene_rf_model = joblib.load(model_dir / "scene_rf.joblib")
    X_train_char, _, X_test_char, _, _, y_test_char = split(characters_df, "is_killed")
    X_train_scene, _, X_test_scene, _, _, y_test_scene = split(scenes_df, "death_in_scene")

    plot_class_distribution(characters_df, "character", "Character Class Distribution", out_dir, "is_killed", ["Survives", "Dies"])
    plot_class_distribution(scenes_df, "scene", "Scene Class Distribution", out_dir, "death_in_scene", ["No Death in Scene", "Death in Scene"])
    plot_confusion_matrix(models["character"], X_test_char, y_test_char, "character", "Characters Confusion Matrix", out_dir)
    plot_confusion_matrix(models["scene"], X_test_scene, y_test_scene, "scene", "Scenes Confusion Matrix", out_dir)
    plot_model_comparison(results, "character", "Character Model Comparison", out_dir, "character_survival")
    plot_model_comparison(results, "scene", "Scene Model Comparison", out_dir, "scene_death")
    plot_feature_importance(char_rf_model, X_train_char.columns.tolist(), "character", "Character Feature Importances", out_dir)
    plot_feature_importance(scene_rf_model, X_train_scene.columns.tolist(), "scene", "Scene Feature Importances", out_dir)
    plot_precision_recall(models["scene"], X_test_scene, y_test_scene, "scene", "Scene Precision-Recall", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",  default="reports")
    ap.add_argument("--out_dir",   default="reports/figures")
    ap.add_argument("--model_dir", default="models")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.model_dir)