#
# Comparison Figure Generator
# This script does not train any models. It reads saved results from
# the other scripts and produces cross-model comparison figures for the dissertation.
#

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#
# Configuration
#

BIG2015_RF_DIR = "/home/thisisuly/Documents/Honours Project/Results/RF/BIG2015"
CIC_RF_DIR = "/home/thisisuly/Documents/Honours Project/Results/RF/CIC_MalMem_2022"
CIC_MLP_DIR = "/home/thisisuly/Documents/Honours Project/Results/MLP/CIC_MalMem"
CIC_ENSEMBLE_DIR = "/home/thisisuly/Documents/Honours Project/Results/Ensemble/CIC_MalMem"
OUTPUT_DIR = "/home/thisisuly/Documents/Honours Project/Results/Comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

#
# Load BIG2015 RF Results (mode_comparison.csv)
#

big2015_rf = {}

big2015_csv_path = os.path.join(BIG2015_RF_DIR, "mode_comparison.csv")
if os.path.exists(big2015_csv_path):
    big2015_df = pd.read_csv(big2015_csv_path)
    for row_index, row in big2015_df.iterrows():
        mode = row["mode"]
        big2015_rf[mode] = {
            "test_accuracy": row["test_accuracy"] * 100.0,
            "test_macro_f1": row["test_macro_f1"],
        }
    print("BIG2015 RF results loaded:")
    for mode, metrics in big2015_rf.items():
        print("  " + mode + "  Acc=" + str(round(metrics["test_accuracy"], 2)) + "%  Macro F1=" + str(round(metrics["test_macro_f1"], 4)))
else:
    print("WARNING: BIG2015 mode_comparison.csv not found at " + big2015_csv_path)

#
# Load CIC-MalMem RF Results (experiment_summary.csv)
#

cic_rf = {}

cic_rf_csv_path = os.path.join(CIC_RF_DIR, "experiment_summary.csv")
if os.path.exists(cic_rf_csv_path):
    cic_rf_df = pd.read_csv(cic_rf_csv_path)
    cic_rf_df = cic_rf_df.drop_duplicates(subset=["experiment"])
    for row_index, row in cic_rf_df.iterrows():
        target = row["target"]
        feature_mode = row["mode"]
        cic_rf[(target, feature_mode)] = {
            "test_accuracy": row["test_accuracy"] * 100.0,
            "test_macro_f1": row["test_macro_f1"],
        }
    print("\nCIC-MalMem RF experiments loaded:")
    for key, metrics in cic_rf.items():
        print("  " + str(key) + "  Acc=" + str(round(metrics["test_accuracy"], 2)) + "%  Macro F1=" + str(round(metrics["test_macro_f1"], 4)))
else:
    print("WARNING: CIC RF experiment_summary.csv not found at " + cic_rf_csv_path)

#
# Load CIC-MalMem MLP Results (mlp_results.csv)
#

cic_mlp = {}

cic_mlp_csv_path = os.path.join(CIC_MLP_DIR, "mlp_results.csv")
if os.path.exists(cic_mlp_csv_path):
    cic_mlp_df = pd.read_csv(cic_mlp_csv_path)
    for row_index, row in cic_mlp_df.iterrows():
        target = row["target"]
        feature_mode = row["mode"]
        cic_mlp[(target, feature_mode)] = {
            "test_accuracy": row["test_accuracy"] * 100.0,
            "test_macro_f1": row["test_macro_f1"],
        }
    print("\nCIC-MalMem MLP experiments loaded:")
    for key, metrics in cic_mlp.items():
        print("  " + str(key) + "  Acc=" + str(round(metrics["test_accuracy"], 2)) + "%  Macro F1=" + str(round(metrics["test_macro_f1"], 4)))
else:
    print("WARNING: MLP mlp_results.csv not found at " + cic_mlp_csv_path)

#
# Load CIC-MalMem Ensemble Results (comparison.csv)
#

cic_ensemble = {}

ensemble_csv_path = os.path.join(CIC_ENSEMBLE_DIR, "comparison.csv")
if os.path.exists(ensemble_csv_path):
    ensemble_df = pd.read_csv(ensemble_csv_path)
    for row_index, row in ensemble_df.iterrows():
        model_name = row["model"]
        cic_ensemble[model_name] = {
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
        }
    print("\nCIC-MalMem Ensemble results loaded:")
    for model_name, metrics in cic_ensemble.items():
        print("  " + model_name + "  Acc=" + str(round(metrics["accuracy"], 2)) + "%  Macro F1=" + str(round(metrics["macro_f1"], 4)))
else:
    print("WARNING: Ensemble comparison.csv not found at " + ensemble_csv_path)

#
# Figure 1: BIG2015 Feature Mode Comparison (RF)
#

if big2015_rf:
    modes = ["bytes", "opcodes", "combined"]
    f1_values = []
    labels = []
    for mode in modes:
        if mode in big2015_rf:
            f1_values.append(big2015_rf[mode]["test_macro_f1"])
            labels.append(mode.capitalize())

    if labels:
        colours = ["#4C72B0", "#DD8452", "#55A868"]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, f1_values, color=colours[:len(labels)], width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Macro F1")
        ax.set_title("BIG2015: RF Feature Mode Comparison")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, f1_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     str(round(val, 4)), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "fig1_big2015_feature_modes.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("\nSaved Figure 1: " + fig_path)
else:
    print("Skipping Figure 1 (BIG2015 RF data missing).")

#
# Figure 2: CIC-MalMem RF vs MLP by Feature Type (side by side)
#

if cic_rf and cic_mlp:
    targets = ["Class", "FamilyTop"]
    feature_types = ["static", "dynamic", "hybrid"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, target in enumerate(targets):
        rf_f1 = []
        mlp_f1 = []
        labels = []
        for ft in feature_types:
            key = (target, ft)
            rf_f1.append(cic_rf[key]["test_macro_f1"] if key in cic_rf else 0)
            mlp_f1.append(cic_mlp[key]["test_macro_f1"] if key in cic_mlp else 0)
            labels.append(ft.capitalize())

        x = np.arange(len(labels))
        width = 0.3
        bars_rf = axes[idx].bar(x - width / 2, rf_f1, width, label="RF", color="#4C72B0")
        bars_mlp = axes[idx].bar(x + width / 2, mlp_f1, width, label="MLP", color="#DD8452")

        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels)
        axes[idx].set_ylabel("Macro F1")
        axes[idx].set_title("Target: " + target)
        axes[idx].set_ylim(0, 1.15)
        axes[idx].legend()

        for bar, val in zip(bars_rf, rf_f1):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           str(round(val, 4)), ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars_mlp, mlp_f1):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           str(round(val, 4)), ha="center", va="bottom", fontsize=7)

    fig.suptitle("CIC-MalMem-2022: RF vs MLP by Feature Type", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_cic_rf_vs_mlp.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved Figure 2: " + fig_path)
elif cic_rf:
    # Fallback: RF only (MLP results not available yet)
    targets = ["Class", "FamilyTop"]
    feature_types = ["static", "dynamic", "hybrid"]
    colours = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, target in enumerate(targets):
        f1_values = []
        labels = []
        for ft in feature_types:
            key = (target, ft)
            f1_values.append(cic_rf[key]["test_macro_f1"] if key in cic_rf else 0)
            labels.append(ft.capitalize())

        x = np.arange(len(labels))
        bars = axes[idx].bar(x, f1_values, color=colours, width=0.5)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels)
        axes[idx].set_ylabel("Macro F1")
        axes[idx].set_title("Target: " + target)
        axes[idx].set_ylim(0, 1.05)
        for bar, val in zip(bars, f1_values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           str(round(val, 4)), ha="center", va="bottom", fontsize=9)

    fig.suptitle("CIC-MalMem-2022: Feature Type Comparison (RF only)", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig2_cic_rf_vs_mlp.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print("Saved Figure 2 (RF only, MLP data missing): " + fig_path)
else:
    print("Skipping Figure 2 (CIC data missing).")

#
# Figure 3: CIC-MalMem Ensemble Comparison (RF vs MLP vs Average vs Stacking)
#

if cic_ensemble:
    model_order = ["RF", "MLP", "Average", "Stacking"]
    acc_values = []
    f1_values = []
    model_labels = []
    for name in model_order:
        if name in cic_ensemble:
            acc_values.append(cic_ensemble[name]["accuracy"])
            f1_values.append(cic_ensemble[name]["macro_f1"])
            model_labels.append(name)

    if model_labels:
        x = np.arange(len(model_labels))
        width = 0.3
        fig, ax = plt.subplots(figsize=(9, 5))
        bars1 = ax.bar(x - width / 2, acc_values, width, label="Accuracy (%)", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, [v * 100 for v in f1_values], width, label="Macro F1 (%)", color="#DD8452")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(round(bar.get_height(), 2)), ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(round(bar.get_height(), 2)), ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("Score (%)")
        ax.set_title("CIC-MalMem-2022: Ensemble Comparison (FamilyTop, Hybrid)")
        ax.legend()
        ax.set_ylim(0, 110)
        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "fig3_cic_ensemble_comparison.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("Saved Figure 3: " + fig_path)
else:
    print("Skipping Figure 3 (Ensemble data missing).")

#
# Figure 4: Ensemble Improvement Over Best Standalone
#

if cic_ensemble:
    standalone_models = {}
    ensemble_models = {}
    for model_name, metrics in cic_ensemble.items():
        if model_name in ["RF", "MLP"]:
            standalone_models[model_name] = metrics
        else:
            ensemble_models[model_name] = metrics

    if standalone_models and ensemble_models:
        best_standalone_name = max(standalone_models, key=lambda name: standalone_models[name]["macro_f1"])
        best_standalone_f1 = standalone_models[best_standalone_name]["macro_f1"]

        labels = []
        gains = []
        for ensemble_name, ensemble_metrics in ensemble_models.items():
            gain = ensemble_metrics["macro_f1"] - best_standalone_f1
            labels.append(ensemble_name)
            gains.append(gain)

        bar_colours = ["#55A868" if g >= 0 else "#C44E52" for g in gains]
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, gains, color=bar_colours, width=0.4)
        for bar, val in zip(bars, gains):
            y_pos = bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.015
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    ("{:+.4f}").format(val), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Macro F1 Gain")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Ensemble Improvement Over Best Standalone (" + best_standalone_name + ")")
        fig.subplots_adjust(bottom=0.15)
        fig_path = os.path.join(OUTPUT_DIR, "fig4_ensemble_improvement.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print("Saved Figure 4: " + fig_path)
else:
    print("Skipping Figure 4 (Ensemble data missing).")

#
# Save Results Summary CSV
#

summary_rows = []

# BIG2015 RF results

for mode, metrics in big2015_rf.items():
    summary_rows.append({
        "dataset": "BIG2015",
        "model": "RF",
        "target": "Family",
        "feature_type": mode,
        "accuracy": round(metrics["test_accuracy"], 2),
        "macro_f1": round(metrics["test_macro_f1"], 4),
    })

# CIC RF results

for (target, feature_mode), metrics in cic_rf.items():
    summary_rows.append({
        "dataset": "CIC-MalMem",
        "model": "RF",
        "target": target,
        "feature_type": feature_mode,
        "accuracy": round(metrics["test_accuracy"], 2),
        "macro_f1": round(metrics["test_macro_f1"], 4),
    })

# CIC MLP results

for (target, feature_mode), metrics in cic_mlp.items():
    summary_rows.append({
        "dataset": "CIC-MalMem",
        "model": "MLP",
        "target": target,
        "feature_type": feature_mode,
        "accuracy": round(metrics["test_accuracy"], 2),
        "macro_f1": round(metrics["test_macro_f1"], 4),
    })

# CIC Ensemble results

for model_name, metrics in cic_ensemble.items():
    summary_rows.append({
        "dataset": "CIC-MalMem",
        "model": model_name + " (Ensemble)",
        "target": "FamilyTop",
        "feature_type": "hybrid",
        "accuracy": round(metrics["accuracy"], 2),
        "macro_f1": round(metrics["macro_f1"], 4),
    })

results_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUTPUT_DIR, "full_results_summary.csv")
results_df.to_csv(csv_path, index=False)
print("\nSaved summary CSV: " + csv_path)

#
# Generate Data-Driven Conclusions
#

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("COMPARISON SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append("")

# H1: Feature type effectiveness

summary_lines.append("H1: Feature Type Effectiveness")
summary_lines.append("")

if big2015_rf:
    best_mode = max(big2015_rf, key=lambda mode_key: big2015_rf[mode_key]["test_macro_f1"])
    best_f1 = big2015_rf[best_mode]["test_macro_f1"]
    summary_lines.append("  BIG2015 (RF): best feature mode = " + best_mode +
                         " (Macro F1 = " + str(round(best_f1, 4)) + ")")
    for mode, metrics in big2015_rf.items():
        if mode != best_mode:
            summary_lines.append("    " + mode + ": Macro F1 = " + str(round(metrics["test_macro_f1"], 4)))

summary_lines.append("")

if cic_rf:
    for target in ["Class", "FamilyTop"]:
        best_ft = None
        best_f1 = -1
        for ft in ["static", "dynamic", "hybrid"]:
            key = (target, ft)
            if key in cic_rf and cic_rf[key]["test_macro_f1"] > best_f1:
                best_f1 = cic_rf[key]["test_macro_f1"]
                best_ft = ft
        if best_ft is not None:
            summary_lines.append("  CIC-MalMem (RF, " + target + "): best feature type = " + best_ft +
                                 " (Macro F1 = " + str(round(best_f1, 4)) + ")")
            for ft in ["static", "dynamic", "hybrid"]:
                key = (target, ft)
                if key in cic_rf and ft != best_ft:
                    summary_lines.append("    " + ft + ": Macro F1 = " + str(round(cic_rf[key]["test_macro_f1"], 4)))
    summary_lines.append("")

# H2: RF vs MLP (across all experiments)

summary_lines.append("H2: RF vs MLP on CIC-MalMem")
summary_lines.append("")

if cic_rf and cic_mlp:
    for target in ["Class", "FamilyTop"]:
        summary_lines.append("  Target: " + target)
        for ft in ["static", "dynamic", "hybrid"]:
            key = (target, ft)
            rf_f1 = cic_rf[key]["test_macro_f1"] if key in cic_rf else None
            mlp_f1 = cic_mlp[key]["test_macro_f1"] if key in cic_mlp else None
            if rf_f1 is not None and mlp_f1 is not None:
                diff = rf_f1 - mlp_f1
                summary_lines.append("    " + ft + ": RF=" + str(round(rf_f1, 4)) +
                                     " MLP=" + str(round(mlp_f1, 4)) +
                                     " (RF " + ("{:+.4f}").format(diff) + ")")
        summary_lines.append("")

# H3: Ensemble vs standalone

summary_lines.append("H3: Ensemble vs Standalone Models")
summary_lines.append("")

if cic_ensemble:
    standalone_names = ["RF", "MLP"]
    ensemble_names = ["Average", "Stacking"]

    best_standalone_name = None
    best_standalone_f1 = -1
    for name in standalone_names:
        if name in cic_ensemble and cic_ensemble[name]["macro_f1"] > best_standalone_f1:
            best_standalone_f1 = cic_ensemble[name]["macro_f1"]
            best_standalone_name = name

    if best_standalone_name is not None:
        summary_lines.append("  Best standalone: " + best_standalone_name +
                             " (Macro F1 = " + str(round(best_standalone_f1, 4)) + ")")
        for ens_name in ensemble_names:
            if ens_name in cic_ensemble:
                ens_f1 = cic_ensemble[ens_name]["macro_f1"]
                gain = ens_f1 - best_standalone_f1
                if gain > 0:
                    summary_lines.append("  " + ens_name + " ensemble: Macro F1 = " + str(round(ens_f1, 4)) +
                                         ", improves over " + best_standalone_name + " by " + ("{:+.4f}").format(gain))
                elif abs(gain) < 0.005:
                    summary_lines.append("  " + ens_name + " ensemble: Macro F1 = " + str(round(ens_f1, 4)) +
                                         ", similar to " + best_standalone_name + " (" + ("{:+.4f}").format(gain) + ")")
                else:
                    summary_lines.append("  " + ens_name + " ensemble: Macro F1 = " + str(round(ens_f1, 4)) +
                                         ", does not improve over " + best_standalone_name + " (" + ("{:+.4f}").format(gain) + ")")
    summary_lines.append("")

summary_lines.append("=" * 60)

# Save text summary

txt_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
with open(txt_path, "w") as summary_file:
    summary_file.write("\n".join(summary_lines))
print("Saved text summary: " + txt_path)

# Print to console

print()
for line in summary_lines:
    print(line)

print("\nAll comparison outputs saved to: " + OUTPUT_DIR)
