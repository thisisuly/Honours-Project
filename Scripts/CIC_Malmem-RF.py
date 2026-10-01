#
# CIC-MalMem-2022 Random Forest Classification
# 70/15/15 split with 5-fold CV on the training portion.
# Classifying memory forensic samples as benign/malware and by malware family.
#
# Includes balanced accuracy and weighted F1 in addition to standard accuracy,
# as accuracy alone is not sufficient for imbalanced datasets.
#
# Compares static vs dynamic features to determine which memory forensic
# tools contribute most to detection accuracy.
#

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

#
# Configuration
#

# Path to dataset

CSV_PATH = "/Data/CIC_MalMem_2022/Obfuscated-MalMem2022.csv"

# Where the results will be saved

RESULTS_DIR = "/Results/RF/CIC_MalMem_2022"

# Random seed to allow reproducibility

SEED = 14

# Number of cross-validation folds

N_FOLDS = 5

# Feature column prefixes based on the CIC-MalMem paper
# "static" features come from Volatility plugins that inspect process/DLL/handle lists
# "dynamic" features come from plugins that detect injection, hidden processes, and similar behaviours

STATIC_PREFIXES = ("pslist.", "dlllist.", "handles.")
DYNAMIC_PREFIXES = ("ldrmodules.", "malfind.", "psxview.", "svcscan.", "callbacks.", "modules.")

# Ensures that the results directory exists

os.makedirs(RESULTS_DIR, exist_ok=True)

#
# Helper Functions
#

def make_safe_filename(text):
    
    # Remove special characters so the string can be used as a filename
    
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def coarse_family(category):
    
    # Get the coarse family name from a category string
    # e.g. "Spyware-Transponder-hash123" becomes "Spyware-Transponder"
    # Benign stays as Benign
    
    if category == "Benign":
        return "Benign"
    parts = str(category).split("-")
    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]
    return parts[0]


def get_ordered_classes(target_array):
    
    # Order classes nicely: Benign first, Other last, rest alphabetical
    
    unique_classes = sorted(pd.unique(target_array).astype(str).tolist())
    middle = []
    for class_name in unique_classes:
        if class_name != "Benign" and class_name != "Other":
            middle.append(class_name)

    ordered = []
    if "Benign" in unique_classes:
        ordered.append("Benign")
    ordered.extend(middle)
    if "Other" in unique_classes:
        ordered.append("Other")
    return np.array(ordered, dtype=str)

#
# Data Loading
#

def load_dataset(csv_path, top_n_families=25):
    
    # Load the CIC-MalMem CSV and add derived label columns
    
    print("Reading dataset from: " + csv_path)
    
    dataset_df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(dataset_df)} rows, {len(dataset_df.columns)} columns")

    # Create Family column (coarse version of Category)
    
    families = []
    
    for category in dataset_df["Category"].astype(str):
        families.append(coarse_family(category))
    
    dataset_df["Family"] = families

    # Binary classification: Benign vs Malware
    
    binary_labels = []
    
    for family in dataset_df["Family"]:
        if family == "Benign":
            binary_labels.append("Benign")
        else:
            binary_labels.append("Malware")
    dataset_df["Class"] = binary_labels

    # Multiclass: keep top N malware families, group the rest as "Other"
    
    malware_only = dataset_df[dataset_df["Family"] != "Benign"]
    family_counts = malware_only["Family"].value_counts()
    top_families = list(family_counts.head(top_n_families).index)

    family_top_labels = []
    
    for family in dataset_df["Family"]:
        if family == "Benign" or family in top_families:
            family_top_labels.append(family)
        else:
            family_top_labels.append("Other")
    
    dataset_df["FamilyTop"] = family_top_labels

    print(f"Binary classes: {dataset_df['Class'].value_counts().to_dict()}")
    print(f"FamilyTop classes: {len(dataset_df['FamilyTop'].unique())} unique")

    return dataset_df


def get_feature_columns(dataset_df, feature_mode):
   
    # Return list of feature column names for the given mode
    # These are label columns, not features
   
    label_cols = {"Category", "Family", "Class", "FamilyTop"}
    all_features = [col for col in dataset_df.columns if col not in label_cols]

    if feature_mode == "static":
        return [col for col in all_features if col.startswith(STATIC_PREFIXES)]
    elif feature_mode == "dynamic":
        return [col for col in all_features if col.startswith(DYNAMIC_PREFIXES)]
    elif feature_mode == "hybrid":

        # Combine static and dynamic features to compare against each individually

        static = [col for col in all_features if col.startswith(STATIC_PREFIXES)]
        dynamic = [col for col in all_features if col.startswith(DYNAMIC_PREFIXES)]
        return static + dynamic
    else:
        print("ERROR: unknown mode " + feature_mode)
        return []

#
# Plotting
#

def save_confusion_matrix_plot(confusion_mat, labels, title, filepath, value_format=None):
   
    # Save a confusion matrix as a PNG image
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    num_classes = len(labels)

    # Figure size depends on number of classes
    
    if num_classes <= 2:
        figsize = (7, 6)
    elif num_classes <= 10:
        figsize = (10, 8)
    elif num_classes <= 20:
        figsize = (14, 11)
    else:
        figsize = (18, 14)

    rotation = 45 if num_classes <= 10 else 60
    fontsize = 9 if num_classes <= 10 else (8 if num_classes <= 20 else 7)
    show_values = num_classes <= 20

    fig, ax = plt.subplots(figsize=figsize)
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
    confusion_matrix_display.plot(ax=ax, values_format=value_format if show_values else None,
              include_values=show_values, colorbar=True)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)

    if show_values:
        val_fontsize = 7 if num_classes <= 12 else 5
        for txt in ax.texts:
            txt.set_fontsize(val_fontsize)

    # Adjust margins based on label count
    
    if num_classes <= 10:
        fig.subplots_adjust(left=0.28, bottom=0.28, right=0.98, top=0.90)
    elif num_classes <= 20:
        fig.subplots_adjust(left=0.36, bottom=0.38, right=0.98, top=0.90)
    else:
        fig.subplots_adjust(left=0.44, bottom=0.48, right=0.98, top=0.90)

    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    
    print("  Saved confusion matrix: " + filepath)


def plot_gini_importance(rf_model, feature_column_names, title_prefix, results_dir):
    
    # Save Gini importance bar chart (top 20) and full CSV
        
    importances = rf_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]

    top_features = [feature_column_names[i] for i in top_idx]
    top_values = [importances[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features[::-1], top_values[::-1], color="#4C72B0")
    ax.set_xlabel("Gini Importance")
    ax.set_title(title_prefix + " - Top 20 Features (Gini)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "gini_top20.png"), dpi=150)
    plt.close()

    # Save all importances to CSV for further analysis
    
    rows = []
    
    for i in range(len(feature_column_names)):
        rows.append({"feature": feature_column_names[i], "gini_importance": importances[i]})
    
    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values("gini_importance", ascending=False)
    importance_df.to_csv(os.path.join(results_dir, "gini_feature_importance.csv"), index=False)
    
    print("  Saved Gini importance: " + results_dir)

#
# Main Experiment Pipeline
#

def run_experiment(dataset_df, target, feature_mode, feature_column_names, results_dir):
    
    # Run experiment with 70/15/15 split
    # 5-fold stratified CV on the 70% training portion for cross-validated estimates
    # Final model trained on full 70% training set, evaluated on 15% test set

    experiment_name = f"CIC-MalMem-2022 ({target}, {feature_mode})"
    print("\n" + "-" * 70)
    print("Experiment: " + experiment_name)
    print("-" * 70)

    # Prepare feature matrix
    # Convert everything to numeric, fill missing with 0
   
    feature_table = dataset_df[feature_column_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_matrix = feature_table.values
    target_labels = dataset_df[target].astype(str).values

    # Drop classes with too few samples stratified split needs enough per class
    
    class_counts = pd.Series(target_labels).value_counts()
    
    min_needed = 7  # need enough for stratified 70/15/15 split
    
    valid_classes = [class_name for class_name in class_counts.index if class_counts[class_name] >= min_needed]
    mask = np.isin(target_labels, valid_classes)
    
    if mask.sum() < len(target_labels):
        dropped = len(target_labels) - mask.sum()
        
        print(f"  Dropped {dropped} samples from rare classes (< {min_needed} samples)")
        
        feature_matrix = feature_matrix[mask]
        target_labels = target_labels[mask]

    print(f"  Samples: {len(target_labels)}, Features: {feature_matrix.shape[1]}, Classes: {len(np.unique(target_labels))}")

    # Show class distribution
    
    print("  Class distribution:")
    
    dist = pd.Series(target_labels).value_counts()
    
    for cls_name in dist.index[:30]:
        print("    " + str(cls_name) + ": " + str(dist[cls_name]))

    class_ids = get_ordered_classes(target_labels)
    class_names_list = class_ids.tolist()

    # 70/15/15 split
    # Hold out 15% for testing, then split remaining 85% into train (70%) and val (15%)
    
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        feature_matrix, target_labels, test_size=0.15, random_state=SEED, stratify=target_labels
    )
    validation_fraction_of_trainval = 0.15 / 0.85
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_val_features, train_val_labels, test_size=validation_fraction_of_trainval,
        random_state=SEED, stratify=train_val_labels
    )

    print(f"  Split sizes: train={len(train_labels)}, val={len(validation_labels)}, test={len(test_labels)}")
    print(f"  Split percentages: train={len(train_labels)/len(target_labels)*100:.1f}%, val={len(validation_labels)/len(target_labels)*100:.1f}%, test={len(test_labels)/len(target_labels)*100:.1f}%")

    # 5-fold CV on training set (train_val = train+val combined = 70%)
    # CV estimates generalisation performance before touching the held-out test set
    
    print("\n  Running 5-fold CV on the 70% training portion")
    
    stratified_kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_fold_accuracies = []
    cv_fold_balanced_accuracies = []
    cv_fold_macro_f1_scores = []
    cv_fold_weighted_f1_scores = []

    fold_index = 1
    
    for cv_train_indices, cv_validation_indices in tqdm(stratified_kfold.split(train_val_features, train_val_labels), total=N_FOLDS,
                                    desc="CV folds (" + target + ", " + feature_mode + ")"):
        cv_train_features = train_val_features[cv_train_indices]
        cv_validation_features = train_val_features[cv_validation_indices]
        cv_train_labels = train_val_labels[cv_train_indices]
        cv_validation_labels = train_val_labels[cv_validation_indices]

        # 400 trees, max_depth=25, balanced class weights to handle imbalance
        
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
        rf_model.fit(cv_train_features, cv_train_labels)
        cv_predictions = rf_model.predict(cv_validation_features)

        # Compute all the metrics   accuracy alone is not enough for imbalanced data
        
        cv_fold_accuracy = accuracy_score(cv_validation_labels, cv_predictions)
        cv_fold_balanced_accuracy = balanced_accuracy_score(cv_validation_labels, cv_predictions)
        cv_fold_macro_f1 = f1_score(cv_validation_labels, cv_predictions, average="macro", zero_division=0)
        cv_fold_weighted_f1 = f1_score(cv_validation_labels, cv_predictions, average="weighted", zero_division=0)

        cv_fold_accuracies.append(cv_fold_accuracy)
        cv_fold_balanced_accuracies.append(cv_fold_balanced_accuracy)
        cv_fold_macro_f1_scores.append(cv_fold_macro_f1)
        cv_fold_weighted_f1_scores.append(cv_fold_weighted_f1)

        print(f"  Fold {fold_index}: acc={cv_fold_accuracy:.3f} | bal_acc={cv_fold_balanced_accuracy:.3f} | macro_f1={cv_fold_macro_f1:.3f} | weighted_f1={cv_fold_weighted_f1:.3f}")
        
        fold_index += 1

    # Print CV summary
    
    print("\n  CV Results (mean +/- std over 5 folds on 70% training set):")
    print("    Accuracy:          " + str(round(np.mean(cv_fold_accuracies), 6)) + " +/- " + str(round(np.std(cv_fold_accuracies), 6)))
    print("    Balanced Accuracy: " + str(round(np.mean(cv_fold_balanced_accuracies), 6)) + " +/- " + str(round(np.std(cv_fold_balanced_accuracies), 6)))
    print("    Macro F1:          " + str(round(np.mean(cv_fold_macro_f1_scores), 6)) + " +/- " + str(round(np.std(cv_fold_macro_f1_scores), 6)))
    print("    Weighted F1:       " + str(round(np.mean(cv_fold_weighted_f1_scores), 6)) + " +/- " + str(round(np.std(cv_fold_weighted_f1_scores), 6)))

    # Train final model on full 70% training set (train_val_features)
    # Retrain on all available non-test data to maximise learning before final evaluation
    
    print("\n  Training final model on full 70% training set")
    final_rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )
    final_rf_model.fit(train_val_features, train_val_labels)

    # Evaluate on 15% test set
    
    test_predictions = final_rf_model.predict(test_features)

    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)
    test_macro_f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_labels, test_predictions, average="weighted", zero_division=0)
    test_per_class_f1 = f1_score(test_labels, test_predictions, labels=class_ids, average=None, zero_division=0)

    print("\n  Final Test Set Results (15% held-out):")
    print("    Accuracy:          " + str(round(test_accuracy, 6)))
    print("    Balanced Accuracy: " + str(round(test_balanced_accuracy, 6)))
    print("    Macro F1:          " + str(round(test_macro_f1, 6)))
    print("    Weighted F1:       " + str(round(test_weighted_f1, 6)))

    #
    # Save confusion matrix from test set
    #

    safe_prefix = "cic_malmem_" + target + "_" + feature_mode

    test_confusion_matrix = confusion_matrix(test_labels, test_predictions, labels=class_ids)

    counts_path = os.path.join(results_dir, safe_prefix + "_cm.png")
    save_confusion_matrix_plot(
        test_confusion_matrix, class_names_list,
        experiment_name + " Confusion Matrix (Test Set)",
        counts_path, value_format="d"
    )

    #
    # Feature importance for hybrid mode
    #

    if feature_mode == "hybrid":
        print("  Computing feature importance (final model on test data)")

        plot_gini_importance(final_rf_model, feature_column_names, experiment_name, results_dir)

        # Permutation importance on test set
        print("  Computing permutation importance on test set...")
        safe_prefix = "cic_malmem_" + target + "_" + feature_mode
        perm_result = permutation_importance(final_rf_model, test_features, test_labels,
                                             n_repeats=10, random_state=SEED, n_jobs=1,
                                             scoring="f1_macro")
        perm_rows = []
        for i in range(len(feature_column_names)):
            perm_rows.append({
                "feature": feature_column_names[i],
                "importance_mean": perm_result.importances_mean[i],
                "importance_std": perm_result.importances_std[i],
            })
        perm_df = pd.DataFrame(perm_rows).sort_values("importance_mean", ascending=False)
        perm_csv = os.path.join(results_dir, safe_prefix + "_perm_importance.csv")
        perm_df.to_csv(perm_csv, index=False)
        print("  Saved permutation importance: " + perm_csv)

    # Collect metrics for summary
    
    metrics = {
        "experiment": experiment_name,
        "target": target,
        "mode": feature_mode,
        "n_samples": len(target_labels),
        "n_train": len(train_val_labels),
        "n_test": len(test_labels),
        "n_features": feature_matrix.shape[1],
        "n_classes": len(class_ids),
        "cv_accuracy_mean": float(np.mean(cv_fold_accuracies)),
        "cv_accuracy_std": float(np.std(cv_fold_accuracies)),
        "cv_macro_f1_mean": float(np.mean(cv_fold_macro_f1_scores)),
        "cv_macro_f1_std": float(np.std(cv_fold_macro_f1_scores)),
        "cv_balanced_acc_mean": float(np.mean(cv_fold_balanced_accuracies)),
        "cv_balanced_acc_std": float(np.std(cv_fold_balanced_accuracies)),
        "cv_weighted_f1_mean": float(np.mean(cv_fold_weighted_f1_scores)),
        "cv_weighted_f1_std": float(np.std(cv_fold_weighted_f1_scores)),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_balanced_acc": float(test_balanced_accuracy),
        "test_weighted_f1": float(test_weighted_f1),
    }

    return metrics

#
# Run All Experiments
#

# Delete old summary file so we start fresh each run

summary_csv_path = os.path.join(RESULTS_DIR, "experiment_summary.csv")

if os.path.exists(summary_csv_path):
    os.remove(summary_csv_path)

# Load the dataset once

dataset_df = load_dataset(CSV_PATH, top_n_families=25)

# Run all experiments: 2 targets x 3 feature modes = 6 experiments

all_results = []

for target in ["Class", "FamilyTop"]:
    for feature_mode in ["static", "dynamic", "hybrid"]:
        print("\n" + "=" * 70)
        print("TARGET: " + target + " | MODE: " + feature_mode)
        print("=" * 70)

        feature_columns = get_feature_columns(dataset_df, feature_mode)
        
        if len(feature_columns) == 0:
            print("No features found for mode=" + feature_mode + ", skipping")
            continue

        print(f"  Using {len(feature_columns)} features")

        metrics = run_experiment(dataset_df, target, feature_mode, feature_columns, RESULTS_DIR)
        all_results.append(metrics)

# Save experiment summary CSV

if len(all_results) > 0:
    
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_csv_path, index=False)
    
    print("\nSaved experiment summary: " + summary_csv_path)

# Print final overview

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
for result in all_results:
    print(f"  {result['experiment']:50s}  cv_acc={result['cv_accuracy_mean']:.4f}  test_acc={result['test_accuracy']:.4f}  test_macro_f1={result['test_macro_f1']:.4f}")

print("\nResults saved to: " + RESULTS_DIR)
