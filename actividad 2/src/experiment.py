"""Experimento reproducible de ensambles, PCA, t-SNE y Green AI."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
REPETITIONS = (42, 43, 44)


def load_data(random_state=RANDOM_STATE):
    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, stratify=target,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_configurations(random_state):
    return {
        "svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", random_state=random_state)),
        ]),
        "svm_pca": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("model", SVC(C=1.0, kernel="rbf", random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=None, random_state=random_state, n_jobs=-1),
        "random_forest_pca": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("model", RandomForestClassifier(n_estimators=120, random_state=random_state, n_jobs=-1)),
        ]),
        "boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=random_state),
        "boosting_pca": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=random_state)),
            ("model", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=random_state)),
        ]),
    }


def _parameter_count(model):
    return int(sum(value.size for value in model.__dict__.values() if isinstance(value, np.ndarray)))


def run_experiment(output_dir="reports"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    final_models = {}

    for seed in REPETITIONS:
        X_train, X_test, y_train, y_test = load_data(seed)
        for name, model in build_configurations(seed).items():
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start
            inference_start = time.perf_counter()
            predictions = model.predict(X_test)
            inference_seconds = time.perf_counter() - inference_start
            rows.append({
                "configuration": name,
                "repetition": seed,
                "f1_macro": f1_score(y_test, predictions, average="macro"),
                "accuracy": accuracy_score(y_test, predictions),
                "train_seconds": train_seconds,
                "inference_ms_per_sample": inference_seconds * 1000 / len(X_test),
                "model_size_kb": len(pickle.dumps(model)) / 1024,
                "parameters": _parameter_count(model),
            })
            if seed == REPETITIONS[0]:
                final_models[name] = model

    results = pd.DataFrame(rows)
    results.to_csv(output / "results_repetitions.csv", index=False)
    median = results.groupby("configuration", as_index=False).median(numeric_only=True)
    median.to_csv(output / "results_median.csv", index=False)

    for name, model in final_models.items():
        joblib.dump(model, output / f"model_{name}.joblib")

    _plot_reductions(output, load_data(RANDOM_STATE)[0])
    _plot_pareto(output, median)
    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (sklearn.load_breast_cancer)",
        "samples": 569,
        "features": 30,
        "repetitions": list(REPETITIONS),
        "split": "80/20 stratified per repetition",
        "best_by_f1_macro": median.loc[median["f1_macro"].idxmax()].to_dict(),
        "configurations": list(final_models),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return results, median


def _plot_reductions(output, X_train):
    scaled = StandardScaler().fit_transform(X_train)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(pca[:, 0], pca[:, 1], s=12, alpha=0.65, c=np.arange(len(pca)), cmap="viridis")
    axis.set(title="PCA del conjunto de entrenamiento", xlabel="Componente 1", ylabel="Componente 2")
    figure.tight_layout()
    figure.savefig(output / "pca_2d.png", dpi=160)
    plt.close(figure)

    for perplexity in (5, 30):
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE, init="pca", learning_rate="auto").fit_transform(scaled)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.65, c=np.arange(len(embedding)), cmap="plasma")
        axis.set(title=f"t-SNE del entrenamiento (perplexity={perplexity})", xlabel="Dimensión 1", ylabel="Dimensión 2")
        figure.tight_layout()
        figure.savefig(output / f"tsne_perplexity_{perplexity}.png", dpi=160)
        plt.close(figure)


def _plot_pareto(output, median):
    figure, axis = plt.subplots(figsize=(7, 5))
    pareto = []
    for _, candidate in median.iterrows():
        dominated = any(
            other["f1_macro"] >= candidate["f1_macro"]
            and other["train_seconds"] <= candidate["train_seconds"]
            and (other["f1_macro"] > candidate["f1_macro"] or other["train_seconds"] < candidate["train_seconds"])
            for _, other in median.iterrows()
        )
        pareto.append(not dominated)
    median = median.copy()
    median["pareto"] = pareto
    axis.scatter(median["train_seconds"], median["f1_macro"], s=70, c=median["pareto"], cmap="coolwarm")
    for _, row in median.iterrows():
        label = f"{row['configuration']}*" if row["pareto"] else row["configuration"]
        axis.annotate(label, (row["train_seconds"], row["f1_macro"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axis.text(0.02, 0.03, "* configuración no dominada", transform=axis.transAxes, fontsize=9)
    axis.set(title="Frontera coste-desempeño", xlabel="Tiempo de entrenamiento (s)", ylabel="F1 macro mediano")
    figure.tight_layout()
    figure.savefig(output / "pareto_frontier.png", dpi=160)
    plt.close(figure)
