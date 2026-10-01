#
# Ensemble RF + MLP for CIC-MalMem-2022
# Combines the RF and neural network to test whether an ensemble of different
# model types performs better than either model alone.
#
# Ensembles of different model types can improve classification because each
# model produces different error patterns. RF is strong on tabular data while
# the MLP may capture complementary patterns. Two fusion strategies are compared.
#

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)

try:
    from imblearn.over_sampling import SMOTE
    HAVE_SMOTE = True
except ImportError:
    HAVE_SMOTE = False
    
    print("imbalanced-learn not installed, skipping SMOTE")
    print("Install with: pip install imbalanced-learn")

#
# Configuration
#

# Path to dataset

CSV_PATH = "/Data/CIC_MalMem_2022/Obfuscated-MalMem2022.csv"

# Where results will be saved

RESULTS_DIR = "/Results/Ensemble/CIC_MalMem"

# Where model weights will be saved


# Training hyperparameters

MLP_EPOCHS = 20
MLP_LEARNING_RATE = 0.001

# Random seed to allow reproducibility

SEED = 14

#
# Seed Setup
#

def set_seed(seed):

    # Set random seeds across all libraries for reproducible results

    import random
    random.seed(seed)
    np.random.seed(seed)

#
# Data Loading
#

def simplify_label(category):

    # Simplifies the category labels - keeps Benign as is, takes first two parts for malware

    if category == "Benign":
        return "Benign"
    parts = str(category).split("-")
    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]
    return parts[0]


def load_data(csv_path):

    # Loads the CIC-MalMem CSV and prepares features and labels

    print("Loading CIC-MalMem-2022")
    dataframe = pd.read_csv(csv_path)
    print(f"Shape: {dataframe.shape}")

    # Simplify the category labels

    new_labels = []
    for category in dataframe["Category"]:
        new_labels.append(simplify_label(category))
    dataframe["label"] = new_labels

    # Identify feature columns (exclude label columns)

    non_feature_columns = ["Category", "label"]
    feature_column_names = [col for col in dataframe.columns if col not in non_feature_columns]

    # Convert all feature columns to numeric

    for col in feature_column_names:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
    dataframe[feature_column_names] = dataframe[feature_column_names].fillna(0)

    feature_matrix = dataframe[feature_column_names].values.astype(np.float32)
    target_labels_str = dataframe["label"].values

    print(f"Classes: {sorted(set(target_labels_str))}")
    print("Samples per class:")
    labels_unique, counts = np.unique(target_labels_str, return_counts=True)
    for label, count in zip(labels_unique, counts):
        print(f"  {label}: {count}")

    return feature_matrix, target_labels_str, feature_column_names

#
# Plotting
#

def save_confusion_matrix(confusion_mat, class_family_names, title, filepath):

    # Save a confusion matrix plot as PNG

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_family_names)
    confusion_matrix_display.plot(ax=ax, colorbar=True)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


#
# Main Pipeline
#

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
# Load dataset

feature_matrix, target_labels_str, feature_column_names = load_data(CSV_PATH)

# Encode string labels to integers

label_encoder = LabelEncoder()
target_labels = label_encoder.fit_transform(target_labels_str)
class_family_names = list(label_encoder.classes_)
num_classes = len(class_family_names)
print(f"Number of classes: {num_classes}")
print(f"Number of features: {feature_matrix.shape[1]}")

# 70/15/15 stratified split

train_val_features, test_features, train_val_labels, test_labels = train_test_split(
    feature_matrix, target_labels, test_size=0.15, random_state=SEED, stratify=target_labels
)
validation_fraction_of_trainval = 0.15 / 0.85
train_features, validation_features, train_labels, validation_labels = train_test_split(
    train_val_features, train_val_labels, test_size=validation_fraction_of_trainval,
    random_state=SEED, stratify=train_val_labels
)
print(f"Train: {len(train_features)}  Val: {len(validation_features)}  Test: {len(test_features)}")

# Scale features - only fit on training data to avoid data leakage

scaler = MinMaxScaler()
train_features_scaled = scaler.fit_transform(train_features)
validation_features_scaled = scaler.transform(validation_features)
test_features_scaled = scaler.transform(test_features)

# Apply SMOTE to balance the training classes

train_features_smote = train_features_scaled.copy()
train_labels_smote = train_labels.copy()
if HAVE_SMOTE:
    print("Applying SMOTE to training data")
    try:
        smote = SMOTE(random_state=SEED)
        train_features_smote, train_labels_smote = smote.fit_resample(train_features_scaled, train_labels)
        print(f"After SMOTE - training samples: {len(train_features_smote)}")
    except Exception as exception:
        print(f"SMOTE failed: {exception}, continuing without SMOTE")

#
# Branch 1: Random Forest
#

# Grid search to find good RF parameters

print("Running grid search for RF hyperparameters")
best_rf_cv_score = 0.0
best_rf_hyperparameters = {}

param_grid_n_estimators = [100, 200, 400]
param_grid_max_depth = [None, 20]

for n_estimators in param_grid_n_estimators:
    for max_depth in param_grid_max_depth:
        print(f"  Testing n_estimators={n_estimators}, max_depth={max_depth}", end=" ")
        rf_candidate_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=SEED,
            n_jobs=2,
        )

        # 5-fold CV on training data

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_fold_scores = cross_val_score(rf_candidate_model, train_features_smote, train_labels_smote, cv=stratified_kfold, scoring="f1_macro", n_jobs=1)
        cv_mean_f1 = np.mean(cv_fold_scores)
        print(f"mean F1: {round(cv_mean_f1, 4)}")

        if cv_mean_f1 > best_rf_cv_score:
            best_rf_cv_score = cv_mean_f1
            best_rf_hyperparameters = {"n_estimators": n_estimators, "max_depth": max_depth}

print(f"\nBest RF params: {best_rf_hyperparameters} (CV F1: {round(best_rf_cv_score, 4)})")

# Train final RF with best parameters on SMOTE'd training data

print("Training final RF model")
rf_model = RandomForestClassifier(
    n_estimators=best_rf_hyperparameters["n_estimators"],
    max_depth=best_rf_hyperparameters["max_depth"],
    random_state=SEED,
    n_jobs=2,
)
rf_model.fit(train_features_smote, train_labels_smote)

# Get RF probabilities on val and test sets

rf_validation_probabilities = rf_model.predict_proba(validation_features_scaled)
rf_test_probabilities = rf_model.predict_proba(test_features_scaled)
rf_test_predictions = rf_model.predict(test_features_scaled)

rf_test_accuracy = accuracy_score(test_labels, rf_test_predictions)
rf_test_balanced_acc = balanced_accuracy_score(test_labels, rf_test_predictions)
rf_test_macro_f1 = f1_score(test_labels, rf_test_predictions, average="macro", zero_division=0)
print(f"RF Test Accuracy: {round(rf_test_accuracy * 100, 2)}%")
print(f"RF Test Balanced Acc: {round(rf_test_balanced_acc, 4)}")
print(f"RF Test Macro F1: {round(rf_test_macro_f1, 4)}")

#
# Branch 2: MLP Neural Network
#


mlp_model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    max_iter=MLP_EPOCHS,
    learning_rate_init=MLP_LEARNING_RATE,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=SEED,
    verbose=True,
)
mlp_model.fit(train_features_smote, train_labels_smote)

# Get MLP probabilities on val and test sets

mlp_validation_probabilities = mlp_model.predict_proba(validation_features_scaled)
mlp_test_probabilities = mlp_model.predict_proba(test_features_scaled)
mlp_test_predictions = mlp_model.predict(test_features_scaled)

mlp_test_accuracy = accuracy_score(test_labels, mlp_test_predictions)
mlp_test_balanced_acc = balanced_accuracy_score(test_labels, mlp_test_predictions)
mlp_test_macro_f1 = f1_score(test_labels, mlp_test_predictions, average="macro", zero_division=0)
print(f"MLP Test Accuracy: {round(mlp_test_accuracy * 100, 2)}%")
print(f"MLP Test Balanced Acc: {round(mlp_test_balanced_acc, 4)}")
print(f"MLP Test Macro F1: {round(mlp_test_macro_f1, 4)}")

#
# Fusion Strategy 1: Simple Average
#

# Average the two sets of probabilities and take the argmax

average_test_probabilities = (rf_test_probabilities + mlp_test_probabilities) / 2
average_test_predictions = np.argmax(average_test_probabilities, axis=1)

average_test_accuracy = accuracy_score(test_labels, average_test_predictions)
average_test_balanced_acc = balanced_accuracy_score(test_labels, average_test_predictions)
average_test_macro_f1 = f1_score(test_labels, average_test_predictions, average="macro", zero_division=0)
print(f"Average Ensemble Test Accuracy: {round(average_test_accuracy * 100, 2)}%")
print(f"Average Ensemble Test Balanced Acc: {round(average_test_balanced_acc, 4)}")
print(f"Average Ensemble Test Macro F1: {round(average_test_macro_f1, 4)}")

#
# Fusion Strategy 2: Logistic Regression Stacking
#

# Stacking trains a simple classifier on top of the base model predictions
# Concatenate RF and MLP probabilities as features for the stacking classifier
# Use the validation set to train the stacking model (not the training set,
# because the base models were trained on that and would be overconfident)

stacking_validation_features = np.hstack([rf_validation_probabilities, mlp_validation_probabilities])
stacking_test_features = np.hstack([rf_test_probabilities, mlp_test_probabilities])

print(f"Stacking feature shape (val): {stacking_validation_features.shape}")
print(f"Stacking feature shape (test): {stacking_test_features.shape}")

# Train logistic regression on validation set

stacking_model = LogisticRegression(
    random_state=SEED,
    max_iter=1000,
    multi_class="multinomial",
)
stacking_model.fit(stacking_validation_features, validation_labels)

stacking_test_predictions = stacking_model.predict(stacking_test_features)
stacking_test_probabilities = stacking_model.predict_proba(stacking_test_features)

stacking_test_accuracy = accuracy_score(test_labels, stacking_test_predictions)
stacking_test_balanced_acc = balanced_accuracy_score(test_labels, stacking_test_predictions)
stacking_test_macro_f1 = f1_score(test_labels, stacking_test_predictions, average="macro", zero_division=0)
print(f"Stacking Ensemble Test Accuracy: {round(stacking_test_accuracy * 100, 2)}%")
print(f"Stacking Ensemble Test Balanced Acc: {round(stacking_test_balanced_acc, 4)}")
print(f"Stacking Ensemble Test Macro F1: {round(stacking_test_macro_f1, 4)}")

#
# Comparison and Results
#

# Collect all results

all_model_results = {
    "RF":       {"acc": rf_test_accuracy,       "bal_acc": rf_test_balanced_acc,       "f1": rf_test_macro_f1,       "preds": rf_test_predictions,       "proba": rf_test_probabilities},
    "MLP":      {"acc": mlp_test_accuracy,      "bal_acc": mlp_test_balanced_acc,      "f1": mlp_test_macro_f1,      "preds": mlp_test_predictions,      "proba": mlp_test_probabilities},
    "Average":  {"acc": average_test_accuracy,   "bal_acc": average_test_balanced_acc,   "f1": average_test_macro_f1,   "preds": average_test_predictions,   "proba": average_test_probabilities},
    "Stacking": {"acc": stacking_test_accuracy, "bal_acc": stacking_test_balanced_acc, "f1": stacking_test_macro_f1, "preds": stacking_test_predictions, "proba": stacking_test_probabilities},
}

# Print comparison table

print(f"\n{'Model':<12} | {'Accuracy':>10} | {'Balanced Acc':>12} | {'Macro F1':>10}")
print("-" * 55)
for model_name, model_results in all_model_results.items():
    print(f"{model_name:<12} | {round(model_results['acc'] * 100, 2):>9}% | {round(model_results['bal_acc'], 4):>12} | {round(model_results['f1'], 4):>10}")

# Save comparison CSV

comparison_rows = []
for model_name, model_results in all_model_results.items():
    comparison_rows.append({
        "model": model_name,
        "accuracy": round(model_results["acc"] * 100, 2),
        "balanced_acc": round(model_results["bal_acc"], 4),
        "macro_f1": round(model_results["f1"], 4),
    })
comparison_df = pd.DataFrame(comparison_rows)
comparison_csv_path = os.path.join(RESULTS_DIR, "comparison.csv")
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"\nSaved: {comparison_csv_path}")

# Save comparison bar chart

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(all_model_results.keys())
accuracy_values = [all_model_results[model_name]["acc"] * 100 for model_name in model_names]
macro_f1_scores_list = [all_model_results[model_name]["f1"] for model_name in model_names]

colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

bars1 = ax1.bar(model_names, accuracy_values, color=colors)
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Test Accuracy Comparison")
ax1.set_ylim(min(accuracy_values) - 2, 100)
for bar, val in zip(bars1, accuracy_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{round(val, 2)}%", ha="center", va="bottom", fontsize=9)

bars2 = ax2.bar(model_names, macro_f1_scores_list, color=colors)
ax2.set_ylabel("Macro F1")
ax2.set_title("Test Macro F1 Comparison")
ax2.set_ylim(min(macro_f1_scores_list) - 0.02, 1.0)
for bar, val in zip(bars2, macro_f1_scores_list):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{round(val, 4)}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, "comparison_barchart.png")
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"Saved: {chart_path}")

# Save confusion matrices for all 4 approaches

for model_name, model_results in all_model_results.items():
    test_confusion_matrix = confusion_matrix(test_labels, model_results["preds"])
    save_confusion_matrix(
        test_confusion_matrix, class_family_names,
        f"CIC-MalMem {model_name} - Confusion Matrix",
        os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.lower()}.png")
    )

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Complete.")
