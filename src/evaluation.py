import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay
)


# Model Evaluation
def evaluate_model(y_true, y_pred, y_proba=None, loss=None):
    metrics = {
        "Loss": loss,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Average Precision": (
            average_precision_score(y_true, y_proba)
            if y_proba is not None
            else None
        )
    }

    return metrics

# Threshold Selection
def select_threshold(y_true, y_proba, min_precision=0.5):
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_proba
    )

    # The last precision/recall values have no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    valid_thresholds = precision >= min_precision

    if not valid_thresholds.any():
        return None, None

    valid_indices = np.where(valid_thresholds)[0]

    # Maximize recall, then precision in case of a tie.
    best_idx = max(
        valid_indices,
        key=lambda i: (recall[i], precision[i])
    )

    threshold = thresholds[best_idx]

    return threshold, {
        "precision": precision[best_idx],
        "recall": recall[best_idx]
    }

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Healthy (0)", "Sick (1)"],
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()


# Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_proba, model_name):
    precision, recall, _ = precision_recall_curve(
        y_true, y_proba
    )

    average_precision = average_precision_score(
        y_true, y_proba
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"{model_name} (AP = {average_precision:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision
    }


# Model Comparison Table
def compare_models(model_metrics):
    results = pd.DataFrame(model_metrics).T
    results.index.name = "Model"

    return results


# Precision-Recall Curve Comparison
def plot_precision_recall_comparison(curves):
    plt.figure(figsize=(8, 6))

    for model_name, curve in curves.items():
        plt.plot(
            curve["recall"],
            curve["precision"],
            label=(
                f"{model_name} "
                f"(AP = {curve['average_precision']:.3f})"
            )
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()