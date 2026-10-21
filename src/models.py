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
        scoring="accuracy"
    )

# Random Forest with Hyperparameter Tuning
def create_random_forest_search(random_state):
    model = RandomForestClassifier(random_state=random_state)

    params = {
        "n_estimators": [10, 25, 50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 3, 5]
    }

    return GridSearchCV(
        model,
        params,
        cv=5,
        scoring="accuracy"
    )

# Multi-Layer Perceptron (MLP)
class MLP(nn.Module) :
    
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x