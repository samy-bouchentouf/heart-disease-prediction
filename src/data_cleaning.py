"""
Load and preprocess the UCI Heart Disease dataset.

Handles missing values and converts the original target into a
binary classification task: healthy (0) versus heart disease (1).
"""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed.cleveland.data"
)


def load_data():
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?"
    )

    df = df.dropna()

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(float)

    return X, y