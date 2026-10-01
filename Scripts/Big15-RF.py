#
# BIG2015 Malware Classification with Random Forest
# Using byte histograms and opcode frequency features extracted from
# raw .bytes and .asm files in the Microsoft BIG2015 dataset.
#

#
# Evaluation:
# 70/15/15 train/validation/test split with 5-fold CV on the training portion.
# Final results reported on the held-out 15% test set, ensuring reliablity and unbiased perfomance evaluation
#

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

#
# Configuration
#

# Path to dataset

DATA_FOLDER = "/Data/BIG2015"

# Where the results will be saved

RESULTS_DIR = "/Results/RF/BIG2015"

# Random seed to allow reproducability

SEED = 14

# Optional limit for debugging, None allows for the testing of the full dataset

LIMIT = None

# Mapping from class number to family name (from the BIG2015 paper)

FAMILY_MAP = {
    "1": "Ramnit",
    "2": "Lollipop",
    "3": "Kelihos_ver3",
    "4": "Vundo",
    "5": "Simda",
    "6": "Tracur",
    "7": "Kelihos_ver1",
    "8": "Obfuscator.ACY",
    "9": "Gatak"
}

# Ensures that the results Directory exists

os.makedirs(RESULTS_DIR, exist_ok=True)

#
#  FEATURE EXTRACTION
#

def byte_histogram(bytes_file_path):
    
    # Read a .bytes file and count occurrences of each byte value (0-255)
    # plus unknown bytes marked as "??".
    # Returns a dictionary with 257 features: b_0 through b_255 for each
    # possible byte value, and b_unk for unknown/unreadable bytes ("??").
    
    byte_counts = {}
    for i in range(256):
        byte_counts["b_" + str(i)] = 0
    byte_counts["b_unk"] = 0

    bytes_file = open(bytes_file_path, "r", errors="ignore")
    for line in bytes_file:
        tokens = line.strip().split()
        if len(tokens) <= 1:
            continue

        # skip address (first token), parse remaining hex bytes

        for hex_token in tokens[1:]:
            if hex_token == "??":
                byte_counts["b_unk"] += 1
            else:
                try:
                    byte_value = int(hex_token, 16)
                    if 0 <= byte_value <= 255:
                        byte_counts["b_" + str(byte_value)] += 1
                except ValueError:
                    pass
    bytes_file.close()

    return byte_counts


def opcode_counts(asm_file_path, topk=200):

    #
    # Read an .asm file and count the most common opcodes.
    # Look for lines where the first token contains ":" or ends with ":",
    # then take the next token as the opcode if it's purely alphabetic.
    # Returns a dictionary of the top-k most frequent opcodes as features,
    # keyed like "op_mov", "op_push".
    #
    
    opcode_counter = Counter()

    asm_file = open(asm_file_path, "r", errors="ignore")
    for line in asm_file:
        line = line.strip()
        
        if not line:
            continue

        parts = line.split()
        
        if len(parts) < 2:
            continue

        first = parts[0]
        opcode = None

        # Handle lines that include both a section name and a memory address
        
        if ":" in first:
            
            candidate = parts[1].lower()

            # Check if the token is a valid opcode
            
            if candidate.isalpha():
                
                opcode = candidate

            # Sometimes there are extra labels or symbols between the address and the opcode
            
            elif len(parts) >= 3:
                
                candidate = parts[2].lower()
                
                if candidate.isalpha():
                    opcode = candidate

        # Handle lines where the memory address ends with a colon
        
        if opcode is None and first.endswith(":"):
            candidate = parts[1].lower()
            
            if candidate.isalpha():
                opcode = candidate

        # If a valid opcode is found, count how many times it appears
        
        if opcode is not None:
            
            opcode_counter[opcode] += 1
    asm_file.close()

    # Keep only the most frequent opcodes
    
    opcode_features = {}
    
    for opcode_name, count in opcode_counter.most_common(topk):
        
        # Store each opcode as a feature
        
        opcode_features["op_" + opcode_name] = count

    return opcode_features

#
# Data Loading and Caching
#

def load_features(feature_mode, limit):

    #
    #Load labels from trainLabels.csv and extract features for each sample.
    # Uses pickle caching so we don't re-extract every time.
    # Modes: bytes, opcodes, combined
    # Returns (sample_feature_dicts, sample_labels) where each dict maps 
    # feature names to values for one malware sample.
    #

    labels_csv = os.path.join(DATA_FOLDER, "trainLabels.csv")
    train_dir = os.path.join(DATA_FOLDER, "train")

    if not os.path.exists(labels_csv):
        print("ERROR: labels file not found at " + labels_csv)
        return None, None
    if not os.path.exists(train_dir):
        print("ERROR: train directory not found at " + train_dir)
        return None, None

    # Check cache first to save time on reruns
    
    cache_dir = os.path.join(DATA_FOLDER, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    if limit is None:
        limit_str = "full"
    
    else:
        limit_str = "limit" + str(limit)
    
    cache_file = os.path.join(cache_dir, "big2015_" + feature_mode + "_" + limit_str + ".pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        with open(cache_file, "rb") as cache_file_handle:

            data = pickle.load(cache_file_handle)
        
        return data["feature_dicts"], data["labels"]

    # Load labels CSV
    
    labels_df = pd.read_csv(labels_csv)
    if limit is not None:
        
        labels_df = labels_df.sample(n=min(limit, len(labels_df)), random_state=SEED).reset_index(drop=True)
    
    print(f"Processing {len(labels_df)} samples in mode={feature_mode}")

    sample_feature_dicts = []
    sample_labels = []
    skipped_count = 0

    for i in tqdm(range(len(labels_df)), desc="Extracting " + feature_mode + " features"):
        
        sample_id = labels_df.iloc[i]["Id"]
        sample_label = str(labels_df.iloc[i]["Class"])

        bytes_path = os.path.join(train_dir, sample_id + ".bytes")
        asm_path = os.path.join(train_dir, sample_id + ".asm")

        # Checks files are available
        
        if feature_mode == "bytes" and not os.path.exists(bytes_path):
            skipped_count += 1
            
            continue
        
        if feature_mode == "opcodes" and not os.path.exists(asm_path):
            skipped_count += 1
            
            continue
        
        if feature_mode == "combined" and (not os.path.exists(bytes_path) or not os.path.exists(asm_path)):
            skipped_count += 1
            
            continue

        sample_features = {}

        if feature_mode == "bytes" or feature_mode == "combined":
            sample_features.update(byte_histogram(bytes_path))

        if feature_mode == "opcodes" or feature_mode == "combined":
            sample_features.update(opcode_counts(asm_path, topk=200))

        if len(sample_features) == 0:
            skipped_count += 1
            
            continue

        sample_feature_dicts.append(sample_features)
        sample_labels.append(sample_label)

    if skipped_count > 0:
        print("Skipped " + str(skipped_count) + " samples (missing files)")

    if len(sample_feature_dicts) < 10:
        print("ERROR: not enough samples extracted (" + str(len(sample_feature_dicts)) + ")")
        return None, None

    sample_labels = np.array(sample_labels)

    # Save cache to disk
    
    print("Caching features to " + cache_file)
    
    cache_handle = open(cache_file, "wb")
    pickle.dump({"feature_dicts": sample_feature_dicts, "labels": sample_labels}, cache_handle)
    cache_handle.close()

    return sample_feature_dicts, sample_labels

#
# Plotting
#

def save_confusion_matrix(confusion_mat, class_labels, title, save_path, fmt=None):

    # Save a confusion matrix plot as PNG.

    fig, ax = plt.subplots(figsize=(10, 8))

    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_labels)
    show_vals = len(class_labels) <= 12
    confusion_matrix_display.plot(ax=ax, values_format=fmt if show_vals else None,
              include_values=show_vals, colorbar=True)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_labels, fontsize=9)

    if show_vals:
        for txt in ax.texts:
            txt.set_fontsize(7)

    fig.subplots_adjust(left=0.28, bottom=0.28, right=0.98, top=0.90)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    
    print("Saved: " + save_path)


def plot_gini_importance(rf_model, feature_column_names, feature_mode, results_dir):

    # Plot top-20 features by Gini importance and save full importance into a CSV.

    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:20]

    top_names = []
    top_vals = []
    
    for idx in sorted_idx:
        top_names.append(feature_column_names[idx])
        top_vals.append(importances[idx])

    # Horizontal bar chart
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_names[::-1], top_vals[::-1], color="#4C72B0")
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"BIG2015 ({feature_mode}) - Top 20 Features (Gini)")
    plt.tight_layout()
    gini_plot_path = os.path.join(results_dir, "gini_top20.png")
    plt.savefig(gini_plot_path, dpi=150)
    plt.close()

    print("Saved importance plot: " + gini_plot_path)

    # Save all importances to CSV
    
    rows = []
    
    for i in range(len(feature_column_names)):
        rows.append({"feature": feature_column_names[i], "gini_importance": importances[i]})
    importance_df = pd.DataFrame(rows).sort_values("gini_importance", ascending=False)
    csv_out = os.path.join(results_dir, "gini_feature_importance.csv")
    importance_df.to_csv(csv_out, index=False)
    
    print("Saved importance CSV: " + csv_out)

#
# Main Experiment Pipeline
#

def run_experiment(feature_mode):

    # Run 70/15/15 train/val/test split with 5-fold stratified CV on the training portion
    # Final model gets trained on the full training set and evaluated on the held-out test set.
    # Prints results and save figures + text report.


    sample_feature_dicts, sample_labels = load_features(feature_mode=feature_mode, limit=LIMIT)
    
    if sample_feature_dicts is None:
        print("Failed to load features, skipping mode=" + feature_mode)
        return None

    # Show class distribution
    
    print("\nClass distribution:")
    
    for cls_num in sorted(set(sample_labels), key=lambda x: int(x)):
        count = sum(1 for label in sample_labels if label == cls_num)
        name = FAMILY_MAP.get(cls_num, "Unknown")
        
        print(f"  Class {cls_num} ({name}): {count} samples")

    # Convert list of dicts to a DataFrame, fill missing features with 0
    
    feature_table = pd.DataFrame(sample_feature_dicts)
    feature_table = feature_table.fillna(0)
    feature_column_names = list(feature_table.columns)
    feature_matrix = feature_table.values
    target_labels = sample_labels

    class_ids = np.array(sorted(np.unique(target_labels), key=lambda x: int(x)))
    class_family_names = [FAMILY_MAP.get(c, c) for c in class_ids]

    print(f"\nDataset: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features, {len(class_ids)} classes")

    # 70/15/15 split
    # Hold out 15% for final testing, then split remaining 85% into train (70%) and val (15%)
    
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        feature_matrix, target_labels, test_size=0.15, random_state=SEED, stratify=target_labels
    )
    validation_fraction_of_trainval = 0.15 / 0.85
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_val_features, train_val_labels, test_size=validation_fraction_of_trainval,
        random_state=SEED, stratify=train_val_labels
    )

    print(f"\nTrain: {len(train_labels)}, Val: {len(validation_labels)}, Test: {len(test_labels)}")

    # 5-Fold Stratified Cross-Validation on training set
    # CV estimates generalisation performance before touching the held-out test set
    
    print("\n  5-Fold Stratified Cross-Validation (on training set)")
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_fold_accuracies = []
    cv_fold_balanced_accuracies = []
    cv_fold_macro_f1_scores = []
    cv_fold_per_class_f1_scores = []
    cv_confusion_matrix_total = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)

    fold_index = 1
    
    for cv_train_indices, cv_validation_indices in stratified_kfold.split(train_features, train_labels):
        
        cv_train_features = train_features[cv_train_indices]
        cv_validation_features = train_features[cv_validation_indices]
        cv_train_labels = train_labels[cv_train_indices]
        cv_validation_labels = train_labels[cv_validation_indices]

        # 400 trees with balanced class weights to handle imbalance
        
        rf_model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )
        rf_model.fit(cv_train_features, cv_train_labels)
        cv_predictions = rf_model.predict(cv_validation_features)

        cv_fold_accuracy = accuracy_score(cv_validation_labels, cv_predictions)
        cv_fold_accuracies.append(cv_fold_accuracy)

        cv_fold_balanced_acc = balanced_accuracy_score(cv_validation_labels, cv_predictions)
        cv_fold_balanced_accuracies.append(cv_fold_balanced_acc)

        cv_fold_macro_f1 = f1_score(cv_validation_labels, cv_predictions, average="macro", zero_division=0)
        cv_fold_macro_f1_scores.append(cv_fold_macro_f1)

        cv_fold_per_class_f1 = f1_score(cv_validation_labels, cv_predictions, labels=class_ids, average=None, zero_division=0)
        cv_fold_per_class_f1_scores.append(cv_fold_per_class_f1)

        cv_fold_confusion_matrix = confusion_matrix(cv_validation_labels, cv_predictions, labels=class_ids)
        cv_confusion_matrix_total += cv_fold_confusion_matrix

        # Top misclassifications
        
        misclassified_true = []
        misclassified_pred = []
        for sample_index in range(len(cv_validation_labels)):
            if cv_validation_labels[sample_index] != cv_predictions[sample_index]:
                misclassified_true.append(cv_validation_labels[sample_index])
                misclassified_pred.append(cv_predictions[sample_index])

        if len(misclassified_true) > 0:
            
            misclassification_df = pd.DataFrame({"true": misclassified_true, "pred": misclassified_pred})
            mistake_counts = misclassification_df.value_counts().reset_index(name="count")
            mistake_counts["true_name"] = mistake_counts["true"].map(FAMILY_MAP)
            mistake_counts["pred_name"] = mistake_counts["pred"].map(FAMILY_MAP)
            
            print("\nFold " + str(fold_index) + " top misclassifications:")
            print(mistake_counts[["true", "true_name", "pred", "pred_name", "count"]].head(10))
        else:
            print("\nFold " + str(fold_index) + ": perfect classification!")

        print(f"Fold {fold_index}: accuracy = {cv_fold_accuracy:.4f}, macro F1 = {cv_fold_macro_f1:.4f}")
        fold_index += 1

    #
    # Aggregate CV results across folds
    #
    
    cv_fold_per_class_f1_scores = np.array(cv_fold_per_class_f1_scores)
    cv_mean_accuracy = np.mean(cv_fold_accuracies)
    cv_std_accuracy = np.std(cv_fold_accuracies)
    cv_mean_balanced_acc = np.mean(cv_fold_balanced_accuracies)
    cv_std_balanced_acc = np.std(cv_fold_balanced_accuracies)
    cv_mean_macro_f1 = np.mean(cv_fold_macro_f1_scores)
    cv_std_macro_f1 = np.std(cv_fold_macro_f1_scores)

    print(f"\n5-Fold CV on training set: accuracy = {cv_mean_accuracy:.4f} +/- {cv_std_accuracy:.4f}, balanced acc = {cv_mean_balanced_acc:.4f} +/- {cv_std_balanced_acc:.4f}, macro F1 = {cv_mean_macro_f1:.4f} +/- {cv_std_macro_f1:.4f}")

    #
    # Train final model on full training set (train + val combined)
    # Retrain on all available non-test data to maximise learning before final evaluation
    #
    
    print("\nTraining final model on full training set (train + val)")
    
    final_rf_model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=2
    )
    final_rf_model.fit(train_val_features, train_val_labels)

    #
    # Evaluate on test set (final results for the dissertation)
    #
    
    test_predictions = final_rf_model.predict(test_features)
    test_probabilities = final_rf_model.predict_proba(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    test_macro_f1 = f1_score(test_labels, test_predictions, average="macro", zero_division=0)
    test_per_class_f1 = f1_score(test_labels, test_predictions, labels=class_ids, average=None, zero_division=0)

    print(f"\nFinal test set results: accuracy = {test_accuracy:.4f}, balanced acc = {test_balanced_acc:.4f}, macro F1 = {test_macro_f1:.4f}")

    #
    # Save test set confusion matrix
    #
    
    test_confusion_matrix = confusion_matrix(test_labels, test_predictions, labels=class_ids)
    cm_path = os.path.join(RESULTS_DIR, "big2015_" + feature_mode + "_confusion_counts.png")
    save_confusion_matrix(test_confusion_matrix, class_family_names,
                          "BIG2015 (" + feature_mode + ") Test Set Confusion Matrix",
                          cm_path, fmt="d")

    #
    # Feature importance
    #
    
    if feature_mode == "combined":
        plot_gini_importance(final_rf_model, feature_column_names, feature_mode, RESULTS_DIR)

        # Permutation importance on test set
        print("Computing permutation importance on test set...")
        perm_result = permutation_importance(final_rf_model, test_features, test_labels,
                                             n_repeats=3, random_state=SEED, n_jobs=2,
                                             scoring="f1_macro")
        perm_rows = []
        for i in range(len(feature_column_names)):
            perm_rows.append({
                "feature": feature_column_names[i],
                "importance_mean": perm_result.importances_mean[i],
                "importance_std": perm_result.importances_std[i],
            })
        perm_df = pd.DataFrame(perm_rows).sort_values("importance_mean", ascending=False)
        perm_csv = os.path.join(RESULTS_DIR, "permutation_importance.csv")
        perm_df.to_csv(perm_csv, index=False)
        print("Saved permutation importance: " + perm_csv)

    return {
        "mode": feature_mode,
        "cv_accuracy_mean": cv_mean_accuracy,
        "cv_accuracy_std": cv_std_accuracy,
        "cv_balanced_acc_mean": cv_mean_balanced_acc,
        "cv_balanced_acc_std": cv_std_balanced_acc,
        "cv_macro_f1_mean": cv_mean_macro_f1,
        "cv_macro_f1_std": cv_std_macro_f1,
        "test_accuracy": test_accuracy,
        "test_balanced_acc": test_balanced_acc,
        "test_macro_f1": test_macro_f1,
        "test_f1_per_class": test_per_class_f1,
    }

#
# Run all 3 modes
# Compare bytes-only, opcodes-only, and combined features 
#

all_results = []

for feature_mode in ["bytes", "opcodes", "combined"]:
    experiment_result = run_experiment(feature_mode)
    if experiment_result is not None:
        all_results.append(experiment_result)

#
# Final comparison
#

if len(all_results) > 0:
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    comparison_rows = []
    for result in all_results:
        
        print(f"  {result['mode']:10s}  CV acc = {result['cv_accuracy_mean']:.4f} +/- {result['cv_accuracy_std']:.4f}  |  CV F1 = {result['cv_macro_f1_mean']:.4f} +/- {result['cv_macro_f1_std']:.4f}  |  Test acc = {result['test_accuracy']:.4f}  |  Test F1 = {result['test_macro_f1']:.4f}")
        
        comparison_rows.append({
            "mode": result["mode"],
            "cv_accuracy_mean": round(result["cv_accuracy_mean"], 6),
            "cv_accuracy_std": round(result["cv_accuracy_std"], 6),
            "cv_balanced_acc_mean": round(result["cv_balanced_acc_mean"], 6),
            "cv_balanced_acc_std": round(result["cv_balanced_acc_std"], 6),
            "cv_macro_f1_mean": round(result["cv_macro_f1_mean"], 6),
            "cv_macro_f1_std": round(result["cv_macro_f1_std"], 6),
            "test_accuracy": round(result["test_accuracy"], 6),
            "test_balanced_acc": round(result["test_balanced_acc"], 6),
            "test_macro_f1": round(result["test_macro_f1"], 6),
        })

    # Save comparison table as CSV
    
    comparison_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(RESULTS_DIR, "mode_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    
    print("\nSaved comparison CSV: " + comp_path)

    print("\nComplete.")
