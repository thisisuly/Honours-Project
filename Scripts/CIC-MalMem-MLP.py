#
# CIC-MalMem-2022 MLP Classification
# This trains a simple feedforward neural network on the CIC-MalMem dataset
# to compare against the Random Forest baseline - does a neural network
# offer different performance characteristics on tabular malware features?
#
# Evaluates across static, dynamic and hybrid feature modes and both
# binary (Class) and family (FamilyTop) targets to match the RF experiments.
#

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
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

RESULTS_DIR = "/Results/MLP/CIC_MalMem"

# Training hyperparameters

EPOCHS = 20
LEARNING_RATE = 0.001

# Random seed to allow reproducibility

SEED = 14

# Feature mode prefixes (same as CIC RF script)

STATIC_PREFIXES = ("pslist.", "dlllist.", "handles.")
DYNAMIC_PREFIXES = ("ldrmodules.", "malfind.", "psxview.", "svcscan.", "callbacks.", "modules.")

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

    # Binary labels

    binary_labels = []
    for label in dataframe["label"]:
        if label == "Benign":
            binary_labels.append("Benign")
        else:
            binary_labels.append("Malware")
    dataframe["Class"] = binary_labels

    # Multiclass labels (same as RF script)

    dataframe["FamilyTop"] = dataframe["label"]

    # Identify feature columns (exclude label columns)

    non_feature_columns = ["Category", "label", "Class", "FamilyTop"]
    feature_column_names = [col for col in dataframe.columns if col not in non_feature_columns]

    # Convert all feature columns to numeric

    for col in feature_column_names:
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
    dataframe[feature_column_names] = dataframe[feature_column_names].fillna(0)

    print(f"Binary classes: {dataframe['Class'].value_counts().to_dict()}")
    print(f"FamilyTop classes: {len(dataframe['FamilyTop'].unique())} unique")

    return dataframe, feature_column_names


def get_feature_columns(dataframe, feature_mode):

    # Return list of feature column names for the given mode

    non_feature_columns = {"Category", "label", "Class", "FamilyTop"}
    all_features = [col for col in dataframe.columns if col not in non_feature_columns]

    if feature_mode == "static":
        return [col for col in all_features if col.startswith(STATIC_PREFIXES)]
    elif feature_mode == "dynamic":
        return [col for col in all_features if col.startswith(DYNAMIC_PREFIXES)]
    elif feature_mode == "hybrid":
        static = [col for col in all_features if col.startswith(STATIC_PREFIXES)]
        dynamic = [col for col in all_features if col.startswith(DYNAMIC_PREFIXES)]
        return static + dynamic
    else:
        print("ERROR: unknown mode " + feature_mode)
        return []

#
# Plotting
#

def save_confusion_matrix(confusion_mat, class_family_names, title, filepath):

    # Save a confusion matrix plot as PNG

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_classes = len(class_family_names)
    figsize = (7, 6) if num_classes <= 2 else (14, 11) if num_classes <= 20 else (18, 14)
    show_values = num_classes <= 20
    fontsize = 9 if num_classes <= 10 else 7
    rotation = 45 if num_classes <= 10 else 60

    fig, ax = plt.subplots(figsize=figsize)
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_family_names)
    confusion_matrix_display.plot(ax=ax, include_values=show_values, colorbar=True)
    plt.title(title)
    ax.set_xticklabels(class_family_names, rotation=rotation, ha="right", fontsize=fontsize)
    ax.set_yticklabels(class_family_names, fontsize=fontsize)

    if show_values:
        val_fontsize = 7 if num_classes <= 12 else 5
        for txt in ax.texts:
            txt.set_fontsize(val_fontsize)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")

#
# Run Experiment
#

def run_experiment(dataframe, target, feature_mode, feature_columns, results_dir):

    experiment_name = f"CIC-MalMem MLP ({target}, {feature_mode})"
    print(f"\n  Experiment: {experiment_name}")
    print(f"  Features: {len(feature_columns)}")

    # Prepare feature matrix and labels

    feature_matrix = dataframe[feature_columns].values.astype(np.float32)
    target_labels_str = dataframe[target].astype(str).values

    # Encode labels

    label_encoder = LabelEncoder()
    target_labels = label_encoder.fit_transform(target_labels_str)
    class_names = list(label_encoder.classes_)
    print(f"  Classes: {len(class_names)}")

    # 70/15/15 stratified split

    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        feature_matrix, target_labels, test_size=0.15, random_state=SEED, stratify=target_labels
    )
    validation_fraction_of_trainval = 0.15 / 0.85
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_val_features, train_val_labels, test_size=validation_fraction_of_trainval,
        random_state=SEED, stratify=train_val_labels
    )
    print(f"  Split: train={len(train_features)} val={len(validation_features)} test={len(test_features)}")

    # Scale features

    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(validation_features)
    test_features = scaler.transform(test_features)

    # Apply SMOTE

    if HAVE_SMOTE:
        try:
            smote = SMOTE(random_state=SEED)
            train_features, train_labels = smote.fit_resample(train_features, train_labels)
            print(f"  After SMOTE: {len(train_features)} training samples")
        except Exception as exception:
            print(f"  SMOTE failed: {exception}, continuing without SMOTE")

    # Train MLP

    start_time = time.time()

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        max_iter=EPOCHS,
        learning_rate_init=LEARNING_RATE,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=SEED,
        verbose=True,
    )
    mlp_model.fit(train_features, train_labels)

    elapsed = round(time.time() - start_time, 1)
    print(f"  Training complete in {elapsed}s")
    print(f"  Stopped at iteration: {mlp_model.n_iter_}")
    print(f"  Best validation score: {round(mlp_model.best_validation_score_, 4)}")

    # Evaluate on test set

    test_predictions = mlp_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_macro_f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)

    print(f"  Test Accuracy: {round(test_accuracy * 100, 2)}%")
    print(f"  Test Macro F1: {round(test_macro_f1, 4)}")
    print(classification_report(test_labels, test_predictions, target_names=class_names, zero_division=0))

    # Save confusion matrix

    safe_prefix = f"mlp_{target}_{feature_mode}"
    cm = confusion_matrix(test_labels, test_predictions)
    save_confusion_matrix(
        cm, class_names, f"{experiment_name} - Confusion Matrix",
        os.path.join(results_dir, f"{safe_prefix}_cm.png")
    )

    return {
        "target": target,
        "mode": feature_mode,
        "n_features": len(feature_columns),
        "n_classes": len(class_names),
        "test_accuracy": round(test_accuracy, 6),
        "test_macro_f1": round(test_macro_f1, 6),
        "iterations": mlp_model.n_iter_,
        "best_validation_score": round(mlp_model.best_validation_score_, 6),
        "training_time": elapsed,
    }

#
# Main Pipeline
#

set_seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
# Load dataset

dataframe, feature_column_names = load_data(CSV_PATH)

# Run all experiments: 2 targets x 3 feature modes

all_results = []

for target in ["Class", "FamilyTop"]:
    for feature_mode in ["static", "dynamic", "hybrid"]:

        print("\n" + "=" * 60)
        print(f"TARGET: {target} | MODE: {feature_mode}")
        print("=" * 60)

        feature_columns = get_feature_columns(dataframe, feature_mode)

        if len(feature_columns) == 0:
            print("  No features found, skipping")
            continue

        metrics = run_experiment(dataframe, target, feature_mode, feature_columns, RESULTS_DIR)
        all_results.append(metrics)

# Save experiment summary CSV

if len(all_results) > 0:
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_DIR, "mlp_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # Print final comparison

    print("\n" + "=" * 60)
    print("MLP EXPERIMENT SUMMARY")
    print("=" * 60)
    for result in all_results:
        print(f"  {result['target']:>10s} | {result['mode']:>8s} | Acc={round(result['test_accuracy'] * 100, 2):>6}% | Macro F1={result['test_macro_f1']}")

print(f"\nAll results saved to: {RESULTS_DIR}")
