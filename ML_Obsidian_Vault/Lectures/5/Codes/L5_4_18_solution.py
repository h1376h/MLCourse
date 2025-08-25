import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 18: MULTI-CLASS EVALUATION METRICS")
print("=" * 80)

# Given confusion matrix
confusion_matrix = np.array([
    [85, 3, 2, 0],   # Class A
    [4, 76, 5, 5],   # Class B
    [1, 8, 82, 4],   # Class C
    [0, 3, 1, 86]    # Class D
])

classes = ['A', 'B', 'C', 'D']
n_classes = len(classes)

print("\nGiven Confusion Matrix:")
print("Actual\\Predicted | A  | B  | C  | D  |")
print("------------------|----|----|----|----|")
for i, class_name in enumerate(classes):
    row_str = f"**{class_name}**            |"
    for j in range(n_classes):
        row_str += f" {confusion_matrix[i, j]:2d} |"
    print(row_str)

print(f"\nTotal samples: {np.sum(confusion_matrix)}")

# Step 1: Calculate True Positives, False Positives, False Negatives for each class
print("\n" + "=" * 60)
print("STEP 1: CALCULATING TP, FP, FN FOR EACH CLASS")
print("=" * 60)

TP = np.zeros(n_classes)
FP = np.zeros(n_classes)
FN = np.zeros(n_classes)
TN = np.zeros(n_classes)

for i in range(n_classes):
    # True Positives: diagonal elements
    TP[i] = confusion_matrix[i, i]
    
    # False Positives: sum of column i (except diagonal)
    FP[i] = np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
    
    # False Negatives: sum of row i (except diagonal)
    FN[i] = np.sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
    
    # True Negatives: sum of all elements except row i and column i
    TN[i] = np.sum(confusion_matrix) - np.sum(confusion_matrix[i, :]) - np.sum(confusion_matrix[:, i]) + confusion_matrix[i, i]
    
    print(f"\nClass {classes[i]}:")
    print(f"  True Positives (TP) = {TP[i]:.0f}")
    print(f"  False Positives (FP) = {FP[i]:.0f}")
    print(f"  False Negatives (FN) = {FN[i]:.0f}")
    print(f"  True Negatives (TN) = {TN[i]:.0f}")

# Step 2: Calculate overall accuracy and per-class accuracy
print("\n" + "=" * 60)
print("STEP 2: CALCULATING ACCURACY METRICS")
print("=" * 60)

# Overall accuracy
overall_accuracy = np.sum(TP) / np.sum(confusion_matrix)
print(f"\nOverall Accuracy:")
print(f"A = Σ(TP_i) / N = {np.sum(TP)} / {np.sum(confusion_matrix)} = {overall_accuracy:.4f} = {overall_accuracy*100:.2f}%")

# Per-class accuracy
print(f"\nPer-class Accuracy:")
for i in range(n_classes):
    class_total = np.sum(confusion_matrix[i, :])
    class_accuracy = TP[i] / class_total
    print(f"A_{classes[i]} = TP_{classes[i]} / N_{classes[i]} = {TP[i]:.0f} / {class_total:.0f} = {class_accuracy:.4f} = {class_accuracy*100:.2f}%")

# Step 3: Calculate precision, recall, and F1-score for each class
print("\n" + "=" * 60)
print("STEP 3: CALCULATING PRECISION, RECALL, AND F1-SCORE")
print("=" * 60)

precision = np.zeros(n_classes)
recall = np.zeros(n_classes)
f1_score = np.zeros(n_classes)

for i in range(n_classes):
    # Precision = TP / (TP + FP)
    precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
    
    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    print(f"\nClass {classes[i]}:")
    print(f"  Precision = TP / (TP + FP) = {TP[i]:.0f} / ({TP[i]:.0f} + {FP[i]:.0f}) = {precision[i]:.4f} = {precision[i]*100:.2f}%")
    print(f"  Recall = TP / (TP + FN) = {TP[i]:.0f} / ({TP[i]:.0f} + {FN[i]:.0f}) = {recall[i]:.4f} = {recall[i]*100:.2f}%")
    print(f"  F1-score = 2 * (P * R) / (P + R) = 2 * ({precision[i]:.4f} * {recall[i]:.4f}) / ({precision[i]:.4f} + {recall[i]:.4f}) = {f1_score[i]:.4f} = {f1_score[i]*100:.2f}%")

# Step 4: Calculate macro-averaged and micro-averaged metrics
print("\n" + "=" * 60)
print("STEP 4: CALCULATING MACRO AND MICRO AVERAGES")
print("=" * 60)

# Macro-averaged metrics
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1_score)

print(f"\nMacro-averaged metrics:")
print(f"Macro-P = (1/4) * Σ(P_i) = (1/4) * ({precision[0]:.4f} + {precision[1]:.4f} + {precision[2]:.4f} + {precision[3]:.4f}) = {macro_precision:.4f} = {macro_precision*100:.2f}%")
print(f"Macro-R = (1/4) * Σ(R_i) = (1/4) * ({recall[0]:.4f} + {recall[1]:.4f} + {recall[2]:.4f} + {recall[3]:.4f}) = {macro_recall:.4f} = {macro_recall*100:.2f}%")
print(f"Macro-F1 = (1/4) * Σ(F1_i) = (1/4) * ({f1_score[0]:.4f} + {f1_score[1]:.4f} + {f1_score[2]:.4f} + {f1_score[3]:.4f}) = {macro_f1:.4f} = {macro_f1*100:.2f}%")

# Micro-averaged metrics
total_TP = np.sum(TP)
total_FP = np.sum(FP)
total_FN = np.sum(FN)

micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

print(f"\nMicro-averaged metrics:")
print(f"Micro-P = Σ(TP_i) / Σ(TP_i + FP_i) = {total_TP} / ({total_TP} + {total_FP}) = {micro_precision:.4f} = {micro_precision*100:.2f}%")
print(f"Micro-R = Σ(TP_i) / Σ(TP_i + FN_i) = {total_TP} / ({total_TP} + {total_FN}) = {micro_recall:.4f} = {micro_recall*100:.2f}%")
print(f"Micro-F1 = 2 * (Micro-P * Micro-R) / (Micro-P + Micro-R) = 2 * ({micro_precision:.4f} * {micro_recall:.4f}) / ({micro_precision:.4f} + {micro_recall:.4f}) = {micro_f1:.4f} = {micro_f1*100:.2f}%")

# Step 5: Calculate balanced accuracy
print("\n" + "=" * 60)
print("STEP 5: CALCULATING BALANCED ACCURACY")
print("=" * 60)

balanced_accuracy = np.mean(recall)  # Since recall is the same as per-class accuracy
print(f"Balanced Accuracy = (1/4) * Σ(TP_i / (TP_i + FN_i)) = (1/4) * ({recall[0]:.4f} + {recall[1]:.4f} + {recall[2]:.4f} + {recall[3]:.4f}) = {balanced_accuracy:.4f} = {balanced_accuracy*100:.2f}%")

# Step 6: Identify most confusing class pairs
print("\n" + "=" * 60)
print("STEP 6: IDENTIFYING MOST CONFUSING CLASS PAIRS")
print("=" * 60)

# Create a matrix of confusion between classes (excluding diagonal)
confusion_between_classes = confusion_matrix.copy()
np.fill_diagonal(confusion_between_classes, 0)

# Find the maximum confusion
max_confusion = np.max(confusion_between_classes)
max_confusion_indices = np.where(confusion_between_classes == max_confusion)

print(f"\nConfusion between classes (excluding diagonal):")
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            print(f"  {classes[i]} predicted as {classes[j]}: {confusion_matrix[i, j]}")

print(f"\nMost confusing class pairs:")
for idx in range(len(max_confusion_indices[0])):
    i, j = max_confusion_indices[0][idx], max_confusion_indices[1][idx]
    print(f"  {classes[i]} → {classes[j]}: {confusion_matrix[i, j]} samples")

# Create visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=14)
plt.ylabel('Actual Class', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix_heatmap.png'), dpi=300, bbox_inches='tight')

# 2. Metrics Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Multi-class Evaluation Metrics', fontsize=16, fontweight='bold')

# Precision
axes[0, 0].bar(classes, precision, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Precision', fontsize=12)
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(precision):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Recall
axes[0, 1].bar(classes, recall, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Recall', fontsize=12)
axes[0, 1].set_ylim(0, 1)
for i, v in enumerate(recall):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# F1-Score
axes[1, 0].bar(classes, f1_score, color='lightgreen', alpha=0.7)
axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('F1-Score', fontsize=12)
axes[1, 0].set_ylim(0, 1)
for i, v in enumerate(f1_score):
    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Accuracy
class_accuracy = TP / np.sum(confusion_matrix, axis=1)
axes[1, 1].bar(classes, class_accuracy, color='gold', alpha=0.7)
axes[1, 1].set_title('Per-class Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Accuracy', fontsize=12)
axes[1, 1].set_ylim(0, 1)
for i, v in enumerate(class_accuracy):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')

# 3. Macro vs Micro Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35

macro_metrics = [macro_precision, macro_recall, macro_f1]
micro_metrics = [micro_precision, micro_recall, micro_f1]

rects1 = ax.bar(x - width/2, macro_metrics, width, label='Macro-averaged', color='skyblue', alpha=0.7)
rects2 = ax.bar(x + width/2, micro_metrics, width, label='Micro-averaged', color='lightcoral', alpha=0.7)

ax.set_xlabel('Metric Type', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Macro vs Micro Averaged Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
ax.set_ylim(0, 1)
ax.legend()

# Add value labels on bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'macro_vs_micro.png'), dpi=300, bbox_inches='tight')

# 4. Summary Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary data
summary_data = []
for i in range(n_classes):
    summary_data.append([
        classes[i],
        f"{TP[i]:.0f}",
        f"{FP[i]:.0f}",
        f"{FN[i]:.0f}",
        f"{precision[i]:.3f}",
        f"{recall[i]:.3f}",
        f"{f1_score[i]:.3f}",
        f"{class_accuracy[i]:.3f}"
    ])

# Add overall metrics
summary_data.append([
    "OVERALL",
    f"{total_TP:.0f}",
    f"{total_FP:.0f}",
    f"{total_FN:.0f}",
    f"{micro_precision:.3f}",
    f"{micro_recall:.3f}",
    f"{micro_f1:.3f}",
    f"{overall_accuracy:.3f}"
])

# Add macro averages
summary_data.append([
    "MACRO-AVG",
    "-",
    "-",
    "-",
    f"{macro_precision:.3f}",
    f"{macro_recall:.3f}",
    f"{macro_f1:.3f}",
    f"{balanced_accuracy:.3f}"
])

table = ax.table(cellText=summary_data,
                colLabels=['Class', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score', 'Accuracy'],
                cellLoc='center',
                loc='center',
                colWidths=[0.12, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the table
for i in range(len(summary_data) + 1):
    for j in range(8):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        elif i == len(summary_data) - 1:  # Macro-avg row
            cell.set_facecolor('#FF9800')
            cell.set_text_props(weight='bold')
        elif i == len(summary_data) - 2:  # Overall row
            cell.set_facecolor('#2196F3')
            cell.set_text_props(weight='bold', color='white')

plt.title('Comprehensive Evaluation Metrics Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(os.path.join(save_dir, 'metrics_summary_table.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Print final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Balanced Accuracy: {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)")
print(f"Macro-averaged F1: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"Micro-averaged F1: {micro_f1:.4f} ({micro_f1*100:.2f}%)")
print(f"Most confusing pair: {classes[max_confusion_indices[0][0]]} → {classes[max_confusion_indices[1][0]]} ({max_confusion} samples)")
