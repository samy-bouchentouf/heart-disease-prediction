import pandas as pd
from sklearn.model_selection import train_test_split

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]

df = pd.read_csv("../data/processed.cleveland.data", header=None, names=columns, na_values="?")

df = df.dropna()

X = df.drop(columns=["num"])
y = df["num"]