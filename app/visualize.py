"""Visualization utilities for the perceptron classifier."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


PLOTS_DIR = Path("outputs")
PLOTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(conf_matrix: np.ndarray, class_labels: Iterable[str], filename: str = "confusion_matrix.svg") -> Path:
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_pca_scatter(features: np.ndarray, labels: np.ndarray, filename: str = "pca_scatter.svg") -> Path:
    """Plot a 2D PCA scatter plot of the features."""
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", edgecolor="k")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot")
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path
