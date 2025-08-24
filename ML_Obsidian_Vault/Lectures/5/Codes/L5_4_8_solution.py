import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 8: MULTI-CLASS EVALUATION METRICS")
print("=" * 80)

# Given 3-class confusion matrix
C = np.array([
    [85, 5, 10],   # Class 1 predictions
    [8, 78, 14],   # Class 2 predictions  
    [12, 7, 81]    # Class 3 predictions
])

print("\nGiven 3-class confusion matrix:")
print("C = [")
for i, row in enumerate(C):
    print(f"    {row}  # Class {i+1} predictions")
print("]")

print(f"\nConfusion matrix shape: {C.shape}")
print(f"Total samples: {np.sum(C)}")

# Step 1: Calculate overall accuracy
print("\n" + "="*50)
print("STEP 1: CALCULATE OVERALL ACCURACY")
print("="*50)

# Overall accuracy = sum of diagonal elements / total sum
overall_accuracy = np.trace(C) / np.sum(C)
print(f"Overall Accuracy = sum(diagonal) / total")
print(f"Overall Accuracy = {np.trace(C)} / {np.sum(C)}")
print(f"Overall Accuracy = {overall_accuracy:.4f} = {overall_accuracy*100:.2f}%")

# Step 2: Compute precision, recall, and F1-score for each class
print("\n" + "="*50)
print("STEP 2: COMPUTE PRECISION, RECALL, AND F1-SCORE FOR EACH CLASS")
print("="*50)

# Initialize arrays to store metrics
precision = np.zeros(3)
recall = np.zeros(3)
f1_score = np.zeros(3)

print("\nFor each class i:")
print("Precision_i = TP_i / (TP_i + FP_i) = C[i,i] / sum(C[:,i])")
print("Recall_i = TP_i / (TP_i + FN_i) = C[i,i] / sum(C[i,:])")
print("F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)")

for i in range(3):
    print(f"\n--- Class {i+1} ---")
    
    # True Positives
    tp = C[i, i]
    print(f"True Positives (TP_{i+1}) = C[{i},{i}] = {tp}")
    
    # False Positives (sum of column i, excluding diagonal)
    fp = np.sum(C[:, i]) - C[i, i]
    print(f"False Positives (FP_{i+1}) = sum(C[:,{i}]) - C[{i},{i}] = {np.sum(C[:, i])} - {C[i, i]} = {fp}")
    
    # False Negatives (sum of row i, excluding diagonal)
    fn = np.sum(C[i, :]) - C[i, i]
    print(f"False Negatives (FN_{i+1}) = sum(C[{i},:]) - C[{i},{i}] = {np.sum(C[i, :])} - {C[i, i]} = {fn}")
    
    # Precision
    precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"Precision_{i+1} = {tp} / ({tp} + {fp}) = {tp} / {tp + fp} = {precision[i]:.4f}")
    
    # Recall
    recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Recall_{i+1} = {tp} / ({tp} + {fn}) = {tp} / {tp + fn} = {recall[i]:.4f}")
    
    # F1-score
    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    print(f"F1_{i+1} = 2 * ({precision[i]:.4f} * {recall[i]:.4f}) / ({precision[i]:.4f} + {recall[i]:.4f}) = {f1_score[i]:.4f}")

# Create a summary table
print("\n" + "-"*60)
print("SUMMARY OF PER-CLASS METRICS")
print("-"*60)
print(f"{'Class':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-"*60)
for i in range(3):
    print(f"{'Class '+str(i+1):<6} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_score[i]:<10.4f}")

# Step 3: Calculate macro-averaged and micro-averaged metrics
print("\n" + "="*50)
print("STEP 3: CALCULATE MACRO-AVERAGED AND MICRO-AVERAGED METRICS")
print("="*50)

# Macro-averaged metrics
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1_score)

print("\nMACRO-AVERAGED METRICS:")
print("Macro-averaged metrics are the arithmetic mean of per-class metrics")
print(f"Macro Precision = mean([{precision[0]:.4f}, {precision[1]:.4f}, {precision[2]:.4f}]) = {macro_precision:.4f}")
print(f"Macro Recall = mean([{recall[0]:.4f}, {recall[1]:.4f}, {recall[2]:.4f}]) = {macro_recall:.4f}")
print(f"Macro F1 = mean([{f1_score[0]:.4f}, {f1_score[1]:.4f}, {f1_score[2]:.4f}]) = {macro_f1:.4f}")

# Micro-averaged metrics
print("\nMICRO-AVERAGED METRICS:")
print("Micro-averaged metrics aggregate TP, FP, FN across all classes")

# Calculate total TP, FP, FN across all classes
total_tp = np.trace(C)
total_fp = 0
total_fn = 0

for i in range(3):
    total_fp += np.sum(C[:, i]) - C[i, i]
    total_fn += np.sum(C[i, :]) - C[i, i]

print(f"Total TP = sum(diagonal) = {total_tp}")
print(f"Total FP = sum of all FP across classes = {total_fp}")
print(f"Total FN = sum of all FN across classes = {total_fn}")

micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

print(f"Micro Precision = {total_tp} / ({total_tp} + {total_fp}) = {micro_precision:.4f}")
print(f"Micro Recall = {total_tp} / ({total_tp} + {total_fn}) = {micro_recall:.4f}")
print(f"Micro F1 = 2 * ({micro_precision:.4f} * {micro_recall:.4f}) / ({micro_precision:.4f} + {micro_recall:.4f}) = {micro_f1:.4f}")

# Step 4: Compute balanced accuracy
print("\n" + "="*50)
print("STEP 4: COMPUTE BALANCED ACCURACY")
print("="*50)

print("Balanced Accuracy = mean of per-class recall values")
print("This gives equal weight to each class regardless of class imbalance")

balanced_accuracy = np.mean(recall)
print(f"Balanced Accuracy = mean([{recall[0]:.4f}, {recall[1]:.4f}, {recall[2]:.4f}]) = {balanced_accuracy:.4f}")

# Step 5: Design cost-sensitive evaluation metric
print("\n" + "="*50)
print("STEP 5: COST-SENSITIVE EVALUATION METRIC")
print("="*50)

print("Design a cost matrix where misclassifying class 1 as class 3 costs twice as much as other errors")
print("Cost matrix C_cost[i,j] represents the cost of predicting class i when true class is j")

# Define cost matrix
cost_matrix = np.array([
    [0, 1, 2],  # Cost of predicting class 1
    [1, 0, 1],  # Cost of predicting class 2
    [1, 1, 0]   # Cost of predicting class 3
])

print("\nCost Matrix:")
print("C_cost = [")
for i, row in enumerate(cost_matrix):
    print(f"    {row}  # Cost of predicting class {i+1}")
print("]")

print("\nNote: C_cost[0,2] = 2 means misclassifying class 1 as class 3 costs 2 units")

# Calculate total cost
total_cost = np.sum(C * cost_matrix)
print(f"\nTotal Cost = sum(C * C_cost) = {total_cost}")

# Calculate cost per sample
cost_per_sample = total_cost / np.sum(C)
print(f"Cost per sample = {total_cost} / {np.sum(C)} = {cost_per_sample:.4f}")

# Calculate cost for each type of error
print("\nBreakdown of costs by error type:")
for i in range(3):
    for j in range(3):
        if i != j:  # Only for errors
            error_cost = C[i, j] * cost_matrix[i, j]
            print(f"Predicting class {i+1} as class {j+1}: {C[i, j]} samples × {cost_matrix[i, j]} cost = {error_cost}")

# Create visualizations
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
sns.heatmap(C, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# 2. Per-class metrics bar plot
plt.subplot(2, 3, 2)
x = np.arange(3)
width = 0.25

plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
plt.bar(x, recall, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-Class Metrics')
plt.xticks(x, ['Class 1', 'Class 2', 'Class 3'])
plt.legend()
plt.ylim(0, 1)

# 3. Macro vs Micro comparison
plt.subplot(2, 3, 3)
metrics = ['Precision', 'Recall', 'F1-Score']
macro_scores = [macro_precision, macro_recall, macro_f1]
micro_scores = [micro_precision, micro_recall, micro_f1]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, macro_scores, width, label='Macro-Averaged', alpha=0.8)
plt.bar(x + width/2, micro_scores, width, label='Micro-Averaged', alpha=0.8)

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Macro vs Micro Averaging')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 1)

# 4. Overall accuracy vs balanced accuracy
plt.subplot(2, 3, 4)
accuracies = ['Overall', 'Balanced']
scores = [overall_accuracy, balanced_accuracy]
colors = ['skyblue', 'lightcoral']

plt.bar(accuracies, scores, color=colors, alpha=0.8)
plt.ylabel('Accuracy')
plt.title('Overall vs Balanced Accuracy')
plt.ylim(0, 1)

for i, score in enumerate(scores):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

# 5. Cost matrix heatmap
plt.subplot(2, 3, 5)
sns.heatmap(cost_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.title('Cost Matrix')
plt.xlabel('True Class')
plt.ylabel('Predicted Class')

# 6. Cost breakdown pie chart
plt.subplot(2, 3, 6)
cost_breakdown = []
labels = []

for i in range(3):
    for j in range(3):
        if i != j:  # Only errors
            cost = C[i, j] * cost_matrix[i, j]
            cost_breakdown.append(cost)
            labels.append(f'Class {i+1}→{j+1}')

plt.pie(cost_breakdown, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Cost Breakdown by Error Type')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'multi_class_evaluation_metrics.png'), dpi=300, bbox_inches='tight')

# Create detailed metrics table
print("\n" + "="*80)
print("COMPREHENSIVE METRICS SUMMARY")
print("="*80)

# Create a detailed summary table
summary_data = {
    'Metric': ['Overall Accuracy', 'Balanced Accuracy', 
               'Macro Precision', 'Macro Recall', 'Macro F1',
               'Micro Precision', 'Micro Recall', 'Micro F1',
               'Class 1 Precision', 'Class 1 Recall', 'Class 1 F1',
               'Class 2 Precision', 'Class 2 Recall', 'Class 2 F1',
               'Class 3 Precision', 'Class 3 Recall', 'Class 3 F1',
               'Total Cost', 'Cost per Sample'],
    'Value': [overall_accuracy, balanced_accuracy,
              macro_precision, macro_recall, macro_f1,
              micro_precision, micro_recall, micro_f1,
              precision[0], recall[0], f1_score[0],
              precision[1], recall[1], f1_score[1],
              precision[2], recall[2], f1_score[2],
              total_cost, cost_per_sample]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False, float_format='%.4f'))

# Save summary to file
summary_df.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False)

print(f"\nVisualizations and summary saved to: {save_dir}")
print("=" * 80)
