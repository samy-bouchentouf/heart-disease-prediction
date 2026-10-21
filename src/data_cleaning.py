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

    df = df.dropna().copy()

    X = df.drop(columns=["num"])
    y = df["num"]

    return X, y
