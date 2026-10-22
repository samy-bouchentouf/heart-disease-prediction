"""
Define machine learning models and hyperparameter search spaces.

Includes Logistic Regression with cross-validation, Random Forest
with grid search, and a Multi-Layer Perceptron implemented in PyTorch.
"""

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import torch
from torch import nn

# Logistic Regression
def create_logistic_regression(random_state):
    return LogisticRegressionCV(
        max_iter=1000,
        random_state=random_state,
        cv=5,
        scoring="recall",
        l1_ratios=(0.0,),
        use_legacy_attributes=False
    )

# Random Forest with Hyperparameter Tuning
def create_random_forest_search(random_state):
    model = RandomForestClassifier(random_state=random_state)

    params = {
        "n_estimators": [25, 30, 35, 40, 45, 50, 60],
        "max_depth": [3, 5, 6, 7, 8, 10],
        "min_samples_split": [2, 4, 6, 7, 8, 9, 10]
    }

    return GridSearchCV(
        model,
        params,
        cv=5,
        scoring="recall"
    )

# Multi-Layer Perceptron (MLP)
class MLP(nn.Module) :
    
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x