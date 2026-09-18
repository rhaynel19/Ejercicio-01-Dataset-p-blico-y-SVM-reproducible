"""Experimento reproducible de una CNN pequeña para sklearn.digits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_data():
    digits = load_digits(return_X_y=False)
    images = (digits.images / 16.0).astype(np.float64)
    return train_test_split(
        images, digits.target, test_size=0.20, random_state=RANDOM_STATE, stratify=digits.target
    )


class TinyCNN:
    """CNN de una convolución válida, ReLU, flatten y softmax, entrenable en CPU."""

    def __init__(self, filters=8, epochs=100, learning_rate=0.08, seed=RANDOM_STATE):
        self.filters = filters
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.history = {"loss": [], "accuracy": []}
        self.weights = np.empty((filters, 3, 3))
        self.bias = np.empty(filters)
        self.classifier_weights = np.empty((filters * 36, 10))
        self.classifier_bias = np.empty(10)

    def _patches(self, images):
        if images.ndim != 3 or images.shape[1:] != (8, 8):
            raise ValueError("Las imágenes deben tener forma (n, 8x8)")
        return np.stack(
            [images[:, row:row + 3, column:column + 3] for row in range(6) for column in range(6)],
            axis=2,
        ).reshape(len(images), 6, 6, 3, 3)

    def _forward(self, images):
        patches = self._patches(images)
        convolution = np.einsum("nhwij,fij->nfhw", patches, self.weights) + self.bias[None, :, None, None]
        activation = np.maximum(convolution, 0)
        features = activation.reshape(len(images), -1)
        logits = features @ self.classifier_weights + self.classifier_bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return patches, convolution, features, probabilities

    def fit(self, images, labels):
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.12, size=(self.filters, 3, 3))
        self.bias = np.zeros(self.filters)
        self.classifier_weights = rng.normal(0, 0.12, size=(self.filters * 36, 10))
        self.classifier_bias = np.zeros(10)
        encoded = np.eye(10)[labels]

        for _ in range(self.epochs):
            patches, convolution, features, probabilities = self._forward(images)
            clipped = np.clip(probabilities, 1e-12, 1)
            loss = -np.mean(np.sum(encoded * np.log(clipped), axis=1))
            predictions = probabilities.argmax(axis=1)
            self.history["loss"].append(float(loss))
            self.history["accuracy"].append(float(np.mean(predictions == labels)))

            d_logits = (probabilities - encoded) / len(images)
            d_classifier_weights = features.T @ d_logits
            d_classifier_bias = d_logits.sum(axis=0)
            d_features = d_logits @ self.classifier_weights.T
            d_activation = d_features.reshape(len(images), self.filters, 6, 6)
            d_convolution = d_activation * (convolution > 0)
            d_weights = np.einsum("nfhw,nhwij->fij", d_convolution, patches)
            d_bias = d_convolution.sum(axis=(0, 2, 3))

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            self.classifier_weights -= self.learning_rate * d_classifier_weights
            self.classifier_bias -= self.learning_rate * d_classifier_bias
        return self

    def predict_proba(self, images):
        return self._forward(images)[-1]

    def predict(self, images):
        return self.predict_proba(images).argmax(axis=1)

    @property
    def parameter_count(self):
        return int(self.weights.size + self.bias.size + self.classifier_weights.size + self.classifier_bias.size)


def run_experiment(output_dir="reports"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train.reshape(len(X_train), -1), y_train)
    baseline_predictions = baseline.predict(X_test.reshape(len(X_test), -1))

    model = TinyCNN()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    predictions = model.predict(X_test)
    metrics = {
        "dataset": {"samples": 1797, "image_shape": [8, 8], "classes": 10},
        "split": {"train": len(y_train), "test": len(y_test), "test_size": 0.2, "random_state": RANDOM_STATE},
        "baseline": {"accuracy": accuracy_score(y_test, baseline_predictions), "f1_macro": f1_score(y_test, baseline_predictions, average="macro")},
        "cnn": {"accuracy": accuracy_score(y_test, predictions), "f1_macro": f1_score(y_test, predictions, average="macro"), "parameters": model.parameter_count, "train_seconds": train_seconds, "epochs": model.epochs},
        "errors_by_class": (y_test != predictions).astype(int).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }
    metrics["errors_by_class"] = {str(label): int(np.sum((y_test == label) & (predictions != label))) for label in range(10)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(model.history["loss"], label="Pérdida")
    axes[0].set(title="Pérdida de entrenamiento", xlabel="Época", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(model.history["accuracy"], label="Accuracy", color="darkorange")
    axes[1].set(title="Accuracy de entrenamiento", xlabel="Época", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    matrix = confusion_matrix(y_test, predictions)
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xlabel="Predicción", ylabel="Real", title="CNN · matriz de confusión")
    for row in range(10):
        for column in range(10):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(list(metrics["errors_by_class"].keys()), list(metrics["errors_by_class"].values()), color="firebrick")
    axis.set(xlabel="Clase real", ylabel="Errores", title="Errores por clase")
    figure.tight_layout()
    figure.savefig(output / "errors_by_class.png", dpi=160)
    plt.close(figure)

    return metrics
