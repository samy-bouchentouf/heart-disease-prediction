# Heart Disease Prediction

## Overview

This project explores binary heart disease classification using machine learning. Three models are trained and compared: Logistic Regression, Random Forest, and a Multi-Layer Perceptron (MLP) implemented with PyTorch.

The main objective is to prioritize **recall** while maintaining a minimum precision of 0.5, reducing the number of false negatives while limiting false positives.

## Models

- **Logistic Regression** - a linear baseline for binary classification.
- **Random Forest** - an ensemble of decision trees capable of capturing nonlinear relationships.
- **Multi-Layer Perceptron (MLP)** - a neural network implemented with PyTorch.

## Methodology

### 1. Data preprocessing

- Convert the original target into a binary classification task: healthy (`0`) versus heart disease (`1`).
- Split the dataset into training (70%), validation (15%), and test (15%) sets using stratified sampling.
- Standardize the input features using `StandardScaler`, fitted exclusively on the training set.

### 2. Model training

Each model is trained on the training set. Hyperparameter selection is performed where applicable, including cross-validation for Logistic Regression and Random Forest.

The MLP is implemented in PyTorch and trained using cross-entropy loss.

### 3. Threshold selection

The default classification threshold of 0.5 is not necessarily optimal for this task.

For each model, the decision threshold is selected using the validation set to maximize recall, subject to a minimum precision of 0.5.

This approach prioritizes detecting positive cases while controlling the number of false positives.

### 4. Evaluation

The selected thresholds are applied unchanged to the test set. Models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Log loss**
- **Average Precision (AP)**

Confusion matrices and precision-recall curves are also used to analyze model performance.

## Results

The following results were obtained on the held-out test set.

| Metric | Logistic Regression | Random Forest | MLP |
|---|---:|---:|---:|
| Decision threshold | 0.1518 | 0.1379 | 0.2084 |
| Log loss | **0.3155** | 0.3702 | 0.3586 |
| Accuracy | 0.7778 | 0.7111 | **0.8000** |
| Precision | 0.6774 | 0.6176 | **0.7308** |
| Recall | **1.0000** | **1.0000** | 0.9048 |
| F1-score | 0.8077 | 0.7636 | **0.8085** |
| Average Precision | **0.9407** | 0.9285 | 0.9301 |

### Discussion

- **Logistic Regression** achieves a recall of 1.0 on the test set, with a precision of 0.6774 and the lowest log loss. It also achieves the highest Average Precision.
- **Random Forest** also achieves a recall of 1.0, but has lower precision, accuracy, and F1-score than Logistic Regression.
- **MLP** achieves the highest accuracy, precision, and F1-score, although its recall is lower at 0.9048.

The precision-recall curves provide a more complete comparison by showing how precision varies at different recall levels. The Average Precision scores are relatively close, suggesting comparable overall precision-recall performance. At the MLP's selected threshold, Logistic Regression achieves a higher recall but a lower precision, illustrating the trade-off between detecting more positive cases and limiting false positives.

These comparisons are intended to analyze model performance, not to select a final model. If a final model were to be selected, this decision should be based on the validation set, with the test set reserved for the final evaluation. Using test results to select a model would compromise the independence of that evaluation. Results should also be interpreted cautiously, as performance on a finite test set does not guarantee similar performance on unseen data.

## Limitations

- The dataset is relatively small, which limits the models' ability to generalize to unseen data.
- The results depend on data quality and the particular train/validation/test split.
- A small test set can produce substantial variation in reported metrics.
- Threshold selection on the validation set does not guarantee the same precision or recall on unseen data.
- The MLP is selected based on validation loss, which may not correspond to the best precision-recall trade-off after threshold optimization.

## Future Improvements

- Evaluate model robustness across different data splits using stratified cross-validation.
- Investigate class weighting and alternative training strategies for the MLP.
- Explore data augmentation techniques suitable for tabular data.
- Assess performance on an independent external dataset.
- Improve model interpretability and analyze false-negative cases.

## Project Structure

```text
heart-disease-prediction/
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_comparison.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── evaluation.py
│   └── models.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/samy-bouchentouf/heart-disease-prediction
cd heart-disease-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download the `processed.cleveland.data` file from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

Create a `data/` directory at the root of the project and place the downloaded file inside it.

The directory structure should look like this:

```text
heart-disease-prediction/
├── data/
│   └── processed.cleveland.data
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_comparison.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── evaluation.py
│   └── models.py
├── README.md
└── requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open data_exploration.ipynb or model_comparison.ipynb in the `notebooks/` directory and execute the cells in order.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Matplotlib
- Jupyter Notebook