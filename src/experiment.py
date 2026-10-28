
"""
Run a complete model comparison experiment for one cross-validation fold.

The experiment includes an outer train-test split provided by the caller,
an internal validation split, preprocessing, model training, validation-based
threshold selection, and evaluation on the held-out test fold.

The results can be used to compare model performance across stratified folds.
"""

import random
import copy

import numpy as np
import sklearn
import torch
from torch import nn, cuda
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data_cleaning import load_data
from src.models import (
    create_logistic_regression,
    create_random_forest_search,
    MLP
)
from src.evaluation import evaluate_model, select_threshold

RECALL_WEIGHT = 0.8

def run_experiment(random_state, train_indices, test_indices):
    """
    Run the three classification models for one cross-validation fold.

    Parameters
    ----------
    random_state : int
        Seed used for the internal validation split and model initialization.
    train_indices : array-like
        Positional indices used for training and validation in this fold.
    test_indices : array-like
        Positional indices held out for evaluation in this fold.

    Returns
    -------
    dict
        Test-set metrics for Logistic Regression, Random Forest, and MLP.
    """

    # Set random seeds
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the dataset and create the internal validation split
    X, y = load_data()

    X_train_val = X.iloc[train_indices]
    y_train_val = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train_val
    )

    # Standardize the features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ============================================================
    # Logistic Regression
    # ============================================================

    model_reg = create_logistic_regression(random_state)
    model_reg.fit(X_train, y_train)

    y_proba_val_logreg = model_reg.predict_proba(X_val)[:, 1]

    threshold_logreg, _ = select_threshold(
        y_val,
        y_proba_val_logreg,
        recall_weight=RECALL_WEIGHT
    )

    y_proba_logreg_all = model_reg.predict_proba(X_test)
    y_proba_logreg = y_proba_logreg_all[:, 1]

    y_pred_logreg = (
        y_proba_logreg >= threshold_logreg
    ).astype(int)

    loss_logreg = sklearn.metrics.log_loss(
        y_test,
        y_proba_logreg_all
    )

    logreg_metrics = evaluate_model(
        y_test,
        y_pred_logreg,
        y_proba_logreg,
        loss=loss_logreg,
        threshold=threshold_logreg
    )

    # ============================================================
    # Random Forest
    # ============================================================

    grid = create_random_forest_search(random_state)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_proba_val_rf = best_model.predict_proba(X_val)[:, 1]

    threshold_rf, _ = select_threshold(
        y_val,
        y_proba_val_rf,
        recall_weight=RECALL_WEIGHT
    )

    y_proba_rf_all = best_model.predict_proba(X_test)
    y_proba_rf = y_proba_rf_all[:, 1]

    y_pred_rf = (
        y_proba_rf >= threshold_rf
    ).astype(int)

    loss_rf = sklearn.metrics.log_loss(
        y_test,
        y_proba_rf_all
    )

    rf_metrics = evaluate_model(
        y_test,
        y_pred_rf,
        y_proba_rf,
        loss=loss_rf,
        threshold=threshold_rf
    )

    # ============================================================
    # Multi-Layer Perceptron (MLP)
    # ============================================================

    BATCH_SIZE = 16
    DROPOUT = 0.4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-2
    EPOCHS = 1500

    device = "cuda" if cuda.is_available() else "cpu"

    Xt_train = torch.tensor(
        X_train, dtype=torch.float32, device=device
    )
    yt_train = torch.tensor(
        y_train.to_numpy(), dtype=torch.long, device=device
    )

    Xt_val = torch.tensor(
        X_val, dtype=torch.float32, device=device
    )
    yt_val = torch.tensor(
        y_val.to_numpy(), dtype=torch.long, device=device
    )

    Xt_test = torch.tensor(
        X_test, dtype=torch.float32, device=device
    )
    yt_test = torch.tensor(
        y_test.to_numpy(), dtype=torch.long, device=device
    )

    in_features = Xt_train.shape[1]
    out_features = len(torch.unique(yt_train))

    train_dataset = TensorDataset(Xt_train, yt_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    mlp_model = MLP(
        in_features,
        out_features,
        DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        params=mlp_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, EPOCHS + 1):

        mlp_model.train()

        for Xt_batch, yt_batch in train_loader:
            yt_pred = mlp_model(Xt_batch)
            train_loss = criterion(yt_pred, yt_batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        mlp_model.eval()

        with torch.no_grad():
            yt_pred_val = mlp_model(Xt_val)
            val_loss = criterion(yt_pred_val, yt_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(
                    mlp_model.state_dict()
                )

    mlp_model.load_state_dict(best_weights)
    mlp_model.eval()

    # Select the MLP threshold on the validation set
    with torch.no_grad():
        yt_pred_val = mlp_model(Xt_val)

        y_proba_val_mlp = torch.softmax(
            yt_pred_val, dim=1
        )[:, 1].cpu().numpy()

    y_true_val_mlp = yt_val.cpu().numpy()

    threshold_mlp, _ = select_threshold(
        y_true_val_mlp,
        y_proba_val_mlp,
        recall_weight=RECALL_WEIGHT
    )

    # Evaluate the MLP on the test set
    with torch.no_grad():
        yt_pred_test = mlp_model(Xt_test)
        mlp_loss = criterion(yt_pred_test, yt_test).item()

        y_true_mlp = yt_test.cpu().numpy()

        y_proba_mlp = torch.softmax(
            yt_pred_test, dim=1
        )[:, 1].cpu().numpy()

        y_pred_mlp = (
            y_proba_mlp >= threshold_mlp
        ).astype(int)

    mlp_metrics = evaluate_model(
        y_true_mlp,
        y_pred_mlp,
        y_proba_mlp,
        loss=mlp_loss,
        threshold=threshold_mlp
    )

    # Return test-set metrics, true labels, and predicted probabilities
    return {
        "Logistic Regression": {
            "metrics": logreg_metrics,
            "y_true": y_test.to_numpy(),
            "y_proba": y_proba_logreg
        },
        "Random Forest": {
            "metrics": rf_metrics,
            "y_true": y_test.to_numpy(),
            "y_proba": y_proba_rf
        },
        "MLP": {
            "metrics": mlp_metrics,
            "y_true": y_true_mlp,
            "y_proba": y_proba_mlp
        }
    }