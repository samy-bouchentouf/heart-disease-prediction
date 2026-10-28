# Heart Disease Prediction

## Overview

This project explores binary heart disease classification using machine learning. Three models are trained and compared: Logistic Regression, Random Forest, and a Multi-Layer Perceptron (MLP) implemented with PyTorch.

The objective is to compare the performance and trade-offs of the three model types, rather than to select a single model.

The main objective is to prioritize **recall** while balancing it against precision, reducing false negatives while limiting false positives. Decision thresholds are selected using a weighted combination of recall and precision, with greater weight assigned to recall.

## Models

- **Logistic Regression** — a linear baseline for binary classification.
- **Random Forest** — an ensemble of decision trees capable of capturing nonlinear relationships.
- **Multi-Layer Perceptron (MLP)** — a neural network implemented with PyTorch.

## Methodology

### 1. Data preprocessing

- Convert the original target into a binary classification task: healthy (`0`) versus heart disease (`1`).
- For the initial model comparison, split the dataset into training (70%), validation (15%), and test (15%) sets using stratified sampling.
- For the robustness analysis, use stratified five-fold cross-validation with `random_state=40`. In each fold, 20% of the data is held out for testing; 15% of the remaining data is used for validation, giving approximately 68% training, 12% validation, and 20% test data.
- Standardize the input features using `StandardScaler`, fitted exclusively on the training set in each experiment.

### 2. Model training

Each model is trained on the training set. Hyperparameter selection is performed where applicable, including cross-validation for Logistic Regression and Random Forest.

The MLP is implemented in PyTorch and trained using cross-entropy loss. Its training process uses validation loss for model selection.

### 3. Threshold selection

The default classification threshold of 0.5 is not necessarily optimal for this task.

For each model, the decision threshold is selected using the validation set to maximize a weighted score combining recall and precision:

`score = 0.8 × recall + 0.2 × precision`

This approach gives greater importance to recall while still accounting for precision. The selected threshold is then applied unchanged to the test set.

### 4. Evaluation

Models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Log loss**
- **Average Precision (AP)**

Confusion matrices and precision-recall curves are also used to analyze model performance.

## Results

### 1. Initial comparison: `random_state=40`

The following results were obtained on a single held-out test set using `random_state=40`.

| Metric | Logistic Regression | Random Forest | MLP |
|---|---:|---:|---:|
| Decision threshold | 0.1518 | 0.1379 | 0.2057 |
| Log loss | **0.3155** | 0.3702 | 0.3432 |
| Accuracy | 0.7778 | 0.7111 | **0.8222** |
| Precision | 0.6774 | 0.6176 | **0.7407** |
| Recall | **1.0000** | **1.0000** | 0.9524 |
| F1-score | 0.8077 | 0.7636 | **0.8333** |
| Average Precision | **0.9407** | 0.9285 | 0.9288 |

![Precision-Recall Curves](assets/precision_recall_curves.png)

#### Discussion

On this initial split, Logistic Regression and Random Forest both achieve a recall of 1.0, meaning that no positive cases are missed on this particular test set. However, Logistic Regression outperforms Random Forest in precision, accuracy, F1-score, log loss, and Average Precision.

The MLP achieves the highest accuracy, precision, and F1-score, while maintaining a recall of 0.9524. However, the precision-recall curves show that the three models have relatively similar overall performance, as reflected by their Average Precision scores. At the recall selected for the MLP (0.9524), Logistic Regression achieves lower precision, illustrating the trade-off between detecting more positive cases and limiting false positives. Overall, the MLP appears particularly competitive on this split, whereas Random Forest seems to be the weakest model.

However, these results reflect only one train-validation-test split. They may therefore be influenced by the specific samples assigned to each subset and do not necessarily represent the models' typical performance. This motivates the five-fold cross-validation analysis described below.

### 2. Robustness analysis: stratified five-fold cross-validation

The robustness analysis uses `StratifiedKFold` with five folds, shuffling enabled, and `random_state=40` to keep the fold assignments reproducible and consistent with the initial comparison. Each fold is held out once for testing, so the five test folds are disjoint and together cover the dataset.

For each fold, the remaining 80% of the data is split into training and validation sets. Logistic Regression and Random Forest hyperparameters are selected by cross-validation within the training subset. The validation set selects the MLP checkpoint using validation loss and is used to select the decision threshold for all three models by maximizing `0.8 x recall + 0.2 x precision`. Each model is evaluated on the held-out test fold using its validation-selected threshold. The resulting proportions are approximately 68% training, 12% validation, and 20% test data per fold.

The table reports the mean and sample standard deviation of each test metric across the five folds.

| Model | Threshold | Loss | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.2655 ± 0.1866 | 0.4080 ± 0.0523 | 0.7236 ± 0.0992 | 0.6668 ± 0.1471 | 0.9198 ± 0.0825 | 0.7607 ± 0.0641 | 0.8970 ± 0.0371 |
| Random Forest | 0.2706 ± 0.1564 | 0.4165 ± 0.0630 | 0.7103 ± 0.1037 | 0.6488 ± 0.1215 | 0.9127 ± 0.0822 | 0.7493 ± 0.0664 | 0.8899 ± 0.0363 |
| MLP | 0.2970 ± 0.2631 | 0.4267 ± 0.0881 | 0.7303 ± 0.0934 | 0.6996 ± 0.1769 | 0.8685 ± 0.1380 | 0.7535 ± 0.0605 | 0.8894 ± 0.0404 |

![Mean Precision-Recall Curves Across Five Folds](assets/mean_precision_recall_curves.png)

#### Discussion

The three models have broadly similar performance, and their mean precision-recall curves overlap substantially. Logistic Regression has the highest mean Average Precision (0.8970 ± 0.0371), although the difference from Random Forest (0.8899 ± 0.0363) and the MLP (0.8894 ± 0.0404) is modest.

At the thresholds selected on each fold's validation set, Logistic Regression and Random Forest achieve similar mean recall (0.9198 ± 0.0825 and 0.9127 ± 0.0822). The MLP has higher mean precision (0.6996 ± 0.1769), but lower and more variable recall (0.8685 ± 0.1380). Given the project's priority on recall, Logistic Regression has a small advantage in mean recall and Average Precision, while Random Forest achieves similar recall and the MLP has higher mean precision. Overall, the three models show different trade-offs rather than a clear overall winner.

These fold-level results are related because their training sets overlap. The reported standard deviations describe variation across this cross-validation procedure; they are not confidence intervals. The small dataset and validation-set threshold selection also limit how confidently these results can be generalized.

## Limitations

- The dataset is small, and each cross-validation test fold contains only about one fifth of the observations, limiting how precisely performance can be estimated.
- Fold-level results are related because the training sets overlap; their standard deviations are not confidence intervals.
- Threshold selection uses a small validation set and may vary substantially across folds.
- Threshold selection on the validation set does not guarantee the same precision or recall on unseen data.
- The MLP is selected based on validation loss, which may not correspond to the best precision-recall trade-off after threshold optimization.

## Future Improvements

- Compare all three models on an independent external dataset to assess whether their relative performance generalizes.
- Investigate class weighting and alternative training strategies for the MLP.
- Explore data augmentation techniques suitable for tabular data.
- Improve model interpretability and analyze false-negative cases.

## Project Structure

```text
heart-disease-prediction/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_comparison.ipynb
│   └── robustness_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── evaluation.py
│   ├── experiment.py
│   └── models.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/samy-bouchentouf/heart-disease-prediction.git
cd heart-disease-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Download `processed.cleveland.data` from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

Create a `data/` directory at the root of the project and place the downloaded file inside it.

The directory structure should include:

```text
heart-disease-prediction/
├── data/
│   └── processed.cleveland.data
├── notebooks/
├── src/
├── README.md
└── requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute the cells in order.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Matplotlib
- Jupyter Notebook