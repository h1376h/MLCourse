import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_3_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Feature subset data
subsets = {
    'A': {'features': [1, 3, 5], 'accuracy': 82, 'time': 1.2},
    'B': {'features': [2, 4, 6], 'accuracy': 85, 'time': 1.8},
    'C': {'features': [1, 2, 3, 4, 5, 6], 'accuracy': 87, 'time': 3.5}
}

print("=" * 60)
print("FEATURE SUBSET EVALUATION - QUESTION 19")
print("=" * 60)

# Task 1: Accuracy per feature
print("\n1. ACCURACY PER FEATURE")
print("-" * 40)

accuracy_per_feature = {}
for subset, data in subsets.items():
    feature_count = len(data['features'])
    acc_per_feature = data['accuracy'] / feature_count
    accuracy_per_feature[subset] = acc_per_feature
    print(f"Subset {subset}: {data['accuracy']}% / {feature_count} features = {acc_per_feature:.2f}% per feature")

best_accuracy_per_feature = max(accuracy_per_feature, key=accuracy_per_feature.get)
print(f"\nBest accuracy per feature: Subset {best_accuracy_per_feature} ({accuracy_per_feature[best_accuracy_per_feature]:.2f}%)")

# Task 2: Accuracy improvement per additional feature
print("\n2. ACCURACY IMPROVEMENT PER ADDITIONAL FEATURE")
print("-" * 50)

transitions = [
    ('A to C', 'A', 'C', len(subsets['C']['features']) - len(subsets['A']['features'])),
    ('B to C', 'B', 'C', len(subsets['C']['features']) - len(subsets['B']['features']))
]

improvements = {}
for trans_name, from_subset, to_subset, additional_features in transitions:
    from_acc = subsets[from_subset]['accuracy']
    to_acc = subsets[to_subset]['accuracy']
    improvement = to_acc - from_acc
    improvement_per_feature = improvement / additional_features
    improvements[trans_name] = improvement_per_feature
    print(f"{trans_name}: {from_acc}% → {to_acc}% (+{improvement}%) over {additional_features} features")
    print(".2f")

# Task 3: Interpretability consideration
print("\n3. INTERPRETABILITY CONSIDERATION")
print("-" * 40)
print("Subset A: 3 features - Most interpretable, fewer features to understand")
print("Subset B: 3 features - Most interpretable, fewer features to understand")
print("Subset C: 6 features - Least interpretable, more complex relationships")
print("\nRecommendation: Choose Subset A or B for interpretability")

# Task 4: Efficiency metric
print("\n4. EFFICIENCY METRIC")
print("-" * 25)
print("Efficiency = Accuracy / (Training Time × Feature Count)")

efficiencies = {}
for subset, data in subsets.items():
    feature_count = len(data['features'])
    efficiency = data['accuracy'] / (data['time'] * feature_count)
    efficiencies[subset] = efficiency
    print(".1f")

best_efficiency = max(efficiencies, key=efficiencies.get)
print(".1f")

# Task 5: Time budget constraint (2 minutes)
print("\n5. TIME BUDGET CONSTRAINT (2 minutes)")
print("-" * 40)

time_budget = 2.0
feasible_subsets = {k: v for k, v in subsets.items() if v['time'] <= time_budget}

if feasible_subsets:
    print(f"Feasible subsets within {time_budget} minutes:")
    for subset, data in feasible_subsets.items():
        print(".1f")
    best_feasible = max(feasible_subsets, key=lambda x: feasible_subsets[x]['accuracy'])
    print(f"\nBest choice within budget: Subset {best_feasible} ({feasible_subsets[best_feasible]['accuracy']}%)")
else:
    print(f"No subsets fit within {time_budget} minute budget")

# Task 6: Composite scoring function
print("\n6. COMPOSITE SCORING FUNCTION")
print("-" * 35)
print("Weights: Accuracy [0.6], Feature Count [0.3], Training Time [0.1]")

# Normalize the metrics (higher is better for accuracy, lower is better for feature count and time)
accuracy_scores = {k: v['accuracy'] for k, v in subsets.items()}
feature_scores = {k: 1/len(v['features']) for k, v in subsets.items()}  # Inverse for fewer features
time_scores = {k: 1/v['time'] for k, v in subsets.items()}  # Inverse for faster training

# Calculate normalized scores (0-1 scale)
acc_min, acc_max = min(accuracy_scores.values()), max(accuracy_scores.values())
feature_min, feature_max = min(feature_scores.values()), max(feature_scores.values())
time_min, time_max = min(time_scores.values()), max(time_scores.values())

normalized_scores = {}
for subset in subsets:
    norm_acc = (accuracy_scores[subset] - acc_min) / (acc_max - acc_min)
    norm_feature = (feature_scores[subset] - feature_min) / (feature_max - feature_min)
    norm_time = (time_scores[subset] - time_min) / (time_max - time_min)

    composite_score = 0.6 * norm_acc - 0.3 * (1 - norm_feature) - 0.1 * (1 - norm_time)
    normalized_scores[subset] = {
        'normalized_accuracy': norm_acc,
        'normalized_feature': norm_feature,
        'normalized_time': norm_time,
        'composite_score': composite_score
    }

    print(f"\nSubset {subset}:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

best_composite = max(normalized_scores, key=lambda x: normalized_scores[x]['composite_score'])
print(".3f")

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Subset Evaluation - Question 19', fontsize=16, fontweight='bold')

# Plot 1: Accuracy per feature
ax1 = axes[0, 0]
subsets_list = list(accuracy_per_feature.keys())
acc_per_feat_list = [accuracy_per_feature[s] for s in subsets_list]
bars1 = ax1.bar(subsets_list, acc_per_feat_list, color=['skyblue', 'lightgreen', 'salmon'])
ax1.set_title('Accuracy per Feature')
ax1.set_ylabel('Accuracy per Feature (%)')
ax1.set_xlabel('Subset')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, acc_per_feat_list):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             '.1f', ha='center', va='bottom')

# Plot 2: Efficiency comparison
ax2 = axes[0, 1]
efficiency_list = [efficiencies[s] for s in subsets_list]
bars2 = ax2.bar(subsets_list, efficiency_list, color=['skyblue', 'lightgreen', 'salmon'])
ax2.set_title('Efficiency Metric')
ax2.set_ylabel('Efficiency Score')
ax2.set_xlabel('Subset')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars2, efficiency_list):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             '.1f', ha='center', va='bottom')

# Plot 3: Composite scores
ax3 = axes[0, 2]
composite_list = [normalized_scores[s]['composite_score'] for s in subsets_list]
bars3 = ax3.bar(subsets_list, composite_list, color=['skyblue', 'lightgreen', 'salmon'])
ax3.set_title('Composite Score (Weighted)')
ax3.set_ylabel('Composite Score')
ax3.set_xlabel('Subset')
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars3, composite_list):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             '.3f', ha='center', va='bottom')

# Plot 4: Feature count vs Accuracy
ax4 = axes[1, 0]
feature_counts = [len(subsets[s]['features']) for s in subsets_list]
accuracies = [subsets[s]['accuracy'] for s in subsets_list]
scatter = ax4.scatter(feature_counts, accuracies, s=100, c=['blue', 'green', 'red'], alpha=0.7)

for i, subset in enumerate(subsets_list):
    ax4.annotate(f'Subset {subset}', (feature_counts[i], accuracies[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)

ax4.set_title('Feature Count vs Accuracy')
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Accuracy (%)')
ax4.grid(True, alpha=0.3)

# Plot 5: Training time vs Accuracy
ax5 = axes[1, 1]
times = [subsets[s]['time'] for s in subsets_list]
scatter2 = ax5.scatter(times, accuracies, s=100, c=['blue', 'green', 'red'], alpha=0.7)

for i, subset in enumerate(subsets_list):
    ax5.annotate(f'Subset {subset}', (times[i], accuracies[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)

ax5.set_title('Training Time vs Accuracy')
ax5.set_xlabel('Training Time (minutes)')
ax5.set_ylabel('Accuracy (%)')
ax5.grid(True, alpha=0.3)

# Plot 6: Radar chart for normalized scores
ax6 = axes[1, 2]
categories = ['Accuracy', 'Interpretability\n(1/Features)', 'Speed\n(1/Time)']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Close the plot

# Create radar chart for each subset
colors = ['blue', 'green', 'red']
for i, subset in enumerate(subsets_list):
    values = [
        normalized_scores[subset]['normalized_accuracy'],
        normalized_scores[subset]['normalized_feature'],
        normalized_scores[subset]['normalized_time']
    ]
    values += values[:1]  # Close the plot

    ax6.plot(angles, values, 'o-', linewidth=2, label=f'Subset {subset}', color=colors[i])
    ax6.fill(angles, values, alpha=0.25, color=colors[i])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories)
ax6.set_title('Normalized Performance Metrics')
ax6.grid(True, alpha=0.3)
ax6.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_subset_evaluation.png'), dpi=300, bbox_inches='tight')

# Create detailed comparison table
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_data = []
headers = ['Subset', 'Features', 'Feature Count', 'Accuracy (%)', 'Training Time (min)',
           'Accuracy/Feature (%)', 'Efficiency', 'Composite Score']

for subset in subsets_list:
    row = [
        f'Subset {subset}',
        str(subsets[subset]['features']),
        len(subsets[subset]['features']),
        subsets[subset]['accuracy'],
        subsets[subset]['time'],
        f"{accuracy_per_feature[subset]:.2f}",
        f"{efficiencies[subset]:.1f}",
        f"{normalized_scores[subset]['composite_score']:.3f}"
    ]
    table_data.append(row)

# Add summary row
summary_row = ['Summary', '-', '-', '-', '-', '-', '-', '-']
table_data.append(summary_row)

# Create table
table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2)

# Color the header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#E6E6FA')

# Color the summary row
for i in range(len(headers)):
    table[(len(table_data)-1, i)].set_facecolor('#FFF8DC')

plt.title('Feature Subset Comparison Summary', fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(save_dir, 'feature_subset_comparison_table.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}")
print("1. feature_subset_evaluation.png - Comprehensive metric visualizations")
print("2. feature_subset_comparison_table.png - Detailed comparison table")

# Final recommendations
print("\n" + "=" * 60)
print("FINAL RECOMMENDATIONS")
print("=" * 60)

print("\n1. Best accuracy per feature: Subset", best_accuracy_per_feature)
print("2. Best efficiency: Subset", best_efficiency)
print("3. Best composite score: Subset", best_composite)
print("4. For interpretability: Subset A or B")
print("5. For time budget (2 min): Subset A or B")

if feasible_subsets:
    print("6. Within 2-minute budget: Subset A (82%) or B (85%)")
else:
    print("6. No subsets within 2-minute budget")
