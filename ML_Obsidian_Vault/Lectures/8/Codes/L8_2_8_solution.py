import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import combinations

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 8: FEATURE RANKING ANALYSIS")
print("=" * 80)

# Given data
features = ['A', 'B', 'C', 'D', 'E']
correlations = [0.85, 0.72, 0.31, 0.68, 0.45]

# Create DataFrame for easier manipulation
df = pd.DataFrame({
    'Feature': features,
    'Correlation': correlations
})

print("Given Dataset:")
print(df.to_string(index=False))
print()

# Task 1: Rank features by relevance
print("TASK 1: Rank features by relevance")
print("-" * 40)

# Sort by correlation (descending order)
df_sorted = df.sort_values('Correlation', ascending=False).reset_index(drop=True)
df_sorted['Rank'] = range(1, len(df_sorted) + 1)

print("Features ranked by correlation (descending order):")
print(df_sorted[['Rank', 'Feature', 'Correlation']].to_string(index=False))
print()

# Task 2: Select top 3 features
print("TASK 2: Select top 3 features")
print("-" * 40)

top_3_features = df_sorted.head(3)
print("Top 3 features selected:")
print(top_3_features[['Feature', 'Correlation']].to_string(index=False))
print(f"Average correlation of top 3: {top_3_features['Correlation'].mean():.3f}")
print()

# Task 3: Calculate coefficient of variation
print("TASK 3: Calculate coefficient of variation")
print("-" * 40)

mean_corr = np.mean(correlations)
std_corr = np.std(correlations)
cv = std_corr / mean_corr

print(f"Mean correlation: {mean_corr:.3f}")
print(f"Standard deviation: {std_corr:.3f}")
print(f"Coefficient of variation (CV) = std/mean = {std_corr:.3f}/{mean_corr:.3f} = {cv:.3f}")
print()

# Task 4: Feature selection with constraints
print("TASK 4: Feature selection with constraints")
print("-" * 40)
print("Constraints:")
print("- Minimize combined correlation variance")
print("- Maintain average correlation > 0.6")
print()

# Calculate variance for all possible feature combinations
feature_combinations = []
for r in range(2, len(features) + 1):
    for combo in combinations(features, r):
        combo_corrs = [correlations[features.index(f)] for f in combo]
        mean_combo = np.mean(combo_corrs)
        var_combo = np.var(combo_corrs)
        
        if mean_combo > 0.6:  # Constraint: average correlation > 0.6
            feature_combinations.append({
                'Features': combo,
                'Mean_Correlation': mean_combo,
                'Variance': var_combo,
                'Num_Features': len(combo)
            })

# Sort by variance (ascending) to find minimum variance
feature_combinations.sort(key=lambda x: x['Variance'])

print("All valid feature combinations (average correlation > 0.6):")
print(f"{'Features':<15} {'Mean Corr':<12} {'Variance':<12} {'Num Features':<12}")
print("-" * 60)
for combo in feature_combinations:
    features_str = ', '.join(combo['Features'])
    print(f"{features_str:<15} {combo['Mean_Correlation']:<12.3f} {combo['Variance']:<12.3f} {combo['Num_Features']:<12}")

print()
print("Optimal solution:")
optimal = feature_combinations[0]
print(f"Features: {', '.join(optimal['Features'])}")
print(f"Mean correlation: {optimal['Mean_Correlation']:.3f}")
print(f"Variance: {optimal['Variance']:.3f}")
print(f"Number of features: {optimal['Num_Features']}")

# Verify the solution
selected_corrs = [correlations[features.index(f)] for f in optimal['Features']]
print(f"\nVerification:")
print(f"Selected correlations: {selected_corrs}")
print(f"Mean: {np.mean(selected_corrs):.3f} > 0.6 ✓")
print(f"Variance: {np.var(selected_corrs):.3f} (minimum among valid combinations)")

# Create visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Visualization 1: Feature ranking bar plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
bars = plt.bar(df_sorted['Feature'], df_sorted['Correlation'], 
                color=['gold', 'silver', 'darkgoldenrod', 'lightblue', 'lightcoral'])
plt.title('Feature Ranking by Correlation', fontsize=14, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Correlation Score')
plt.ylim(0, 1)

# Add value labels on bars
for bar, rank in zip(bars, df_sorted['Rank']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'Rank {rank}', ha='center', va='bottom', fontweight='bold')

# Add grid
plt.grid(True, alpha=0.3)

# Visualization 2: Top 3 features highlight
plt.subplot(2, 2, 2)
colors = ['gold' if i < 3 else 'lightgray' for i in range(len(features))]
bars = plt.bar(features, correlations, color=colors)
plt.title('Top 3 Features Highlighted', fontsize=14, fontweight='bold')
plt.xlabel('Feature')
plt.ylabel('Correlation Score')
plt.ylim(0, 1)

# Highlight top 3
for i, (bar, corr) in enumerate(zip(bars, correlations)):
    if i < 3:
        plt.text(bar.get_x() + bar.get_width()/2., corr + 0.02,
                f'{corr:.2f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)

# Visualization 3: Coefficient of variation explanation
plt.subplot(2, 2, 3)
plt.bar(['Mean', 'Std Dev'], [mean_corr, std_corr], 
         color=['lightblue', 'lightcoral'])
plt.title('Mean vs Standard Deviation', fontsize=14, fontweight='bold')
plt.ylabel('Value')
plt.ylim(0, 1)

# Add value labels
plt.text(0, mean_corr + 0.02, f'{mean_corr:.3f}', ha='center', va='bottom', fontweight='bold')
plt.text(1, std_corr + 0.02, f'{std_corr:.3f}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3)

# Visualization 4: Feature selection analysis
plt.subplot(2, 2, 4)
valid_combinations = [combo for combo in feature_combinations if combo['Num_Features'] <= 4]
x_pos = range(len(valid_combinations))
variances = [combo['Variance'] for combo in valid_combinations]
means = [combo['Mean_Correlation'] for combo in valid_combinations]
colors = ['red' if combo['Features'] == optimal['Features'] else 'blue' 
          for combo in valid_combinations]

plt.scatter(means, variances, c=colors, s=100, alpha=0.7)
plt.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Min Mean Threshold')

# Highlight optimal solution
plt.scatter(optimal['Mean_Correlation'], optimal['Variance'], 
           c='red', s=200, marker='*', label='Optimal Solution')

plt.title('Feature Selection Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Mean Correlation')
plt.ylabel('Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_ranking_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 5: Detailed feature selection comparison
plt.figure(figsize=(14, 10))

# Subplot 1: Variance comparison
plt.subplot(2, 3, 1)
combo_labels = [f"{len(combo['Features'])}F" for combo in feature_combinations]
variances = [combo['Variance'] for combo in feature_combinations]
colors = ['red' if combo['Features'] == optimal['Features'] else 'blue' for combo in feature_combinations]

bars = plt.bar(combo_labels, variances, color=colors, alpha=0.7)
plt.title('Variance Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Feature Combination')
plt.ylabel('Variance')
plt.xticks(rotation=45)

# Highlight optimal
for i, (bar, combo) in enumerate(zip(bars, feature_combinations)):
    if combo['Features'] == optimal['Features']:
        bar.set_color('red')
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                'OPTIMAL', ha='center', va='bottom', fontweight='bold', color='red')

plt.grid(True, alpha=0.3)

# Subplot 2: Mean correlation comparison
plt.subplot(2, 3, 2)
means = [combo['Mean_Correlation'] for combo in feature_combinations]
bars = plt.bar(combo_labels, means, color=colors, alpha=0.7)
plt.title('Mean Correlation Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Feature Combination')
plt.ylabel('Mean Correlation')
plt.xticks(rotation=45)
plt.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Threshold')

for i, (bar, combo) in enumerate(zip(bars, feature_combinations)):
    if combo['Features'] == optimal['Features']:
        bar.set_color('red')
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                'OPTIMAL', ha='center', va='bottom', fontweight='bold', color='red')

plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Feature count distribution
plt.subplot(2, 3, 3)
feature_counts = [combo['Num_Features'] for combo in feature_combinations]
unique_counts = list(set(feature_counts))
count_freq = [feature_counts.count(count) for count in unique_counts]

plt.bar(unique_counts, count_freq, color='lightgreen', alpha=0.7)
plt.title('Feature Count Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Number of Features')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Subplot 4: Pareto front analysis
plt.subplot(2, 3, 4)
plt.scatter(means, variances, c=colors, s=100, alpha=0.7)
plt.scatter(optimal['Mean_Correlation'], optimal['Variance'], 
           c='red', s=200, marker='*', label='Optimal Solution')

# Draw Pareto front
pareto_points = []
for combo in feature_combinations:
    is_pareto = True
    for other in feature_combinations:
        if (other['Mean_Correlation'] > combo['Mean_Correlation'] and 
            other['Variance'] <= combo['Variance']):
            is_pareto = False
            break
    if is_pareto:
        pareto_points.append(combo)

pareto_means = [p['Mean_Correlation'] for p in pareto_points]
pareto_vars = [p['Variance'] for p in pareto_points]
pareto_means.sort()
pareto_vars.sort(reverse=True)

plt.plot(pareto_means, pareto_vars, 'r--', linewidth=2, label='Pareto Front')

plt.title('Pareto Front Analysis', fontsize=12, fontweight='bold')
plt.xlabel('Mean Correlation')
plt.ylabel('Variance')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Feature importance heatmap
plt.subplot(2, 3, 5)
feature_importance = {}
for combo in feature_combinations:
    for feature in combo['Features']:
        if feature not in feature_importance:
            feature_importance[feature] = {'count': 0, 'total_variance': 0}
        feature_importance[feature]['count'] += 1
        feature_importance[feature]['total_variance'] += combo['Variance']

# Create heatmap data
heatmap_data = np.zeros((len(features), 2))
for i, feature in enumerate(features):
    if feature in feature_importance:
        heatmap_data[i, 0] = feature_importance[feature]['count']
        heatmap_data[i, 1] = feature_importance[feature]['total_variance']

plt.imshow(heatmap_data.T, cmap='YlOrRd', aspect='auto')
plt.xticks(range(len(features)), features)
plt.yticks([0, 1], ['Count', 'Total Variance'])
plt.title('Feature Importance Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(label='Value')

# Subplot 6: Summary statistics
plt.subplot(2, 3, 6)
plt.axis('off')
summary_text = f"""
Summary Statistics:
• Total combinations: {len(feature_combinations)}
• Optimal features: {', '.join(optimal['Features'])}
• Optimal mean corr: {optimal['Mean_Correlation']:.3f}
• Optimal variance: {optimal['Variance']:.3f}
• CV of correlations: {cv:.3f}
• Top 3 features: {', '.join(top_3_features['Feature'].tolist())}
"""
plt.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
         verticalalignment='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_feature_analysis.png'), dpi=300, bbox_inches='tight')

# Create a comprehensive summary table
print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame({
    'Metric': [
        'Total Features',
        'Top 3 Features',
        'Top 3 Average Correlation',
        'Coefficient of Variation',
        'Optimal Feature Set',
        'Optimal Mean Correlation',
        'Optimal Variance',
        'Number of Valid Combinations'
    ],
    'Value': [
        len(features),
        ', '.join(top_3_features['Feature'].tolist()),
        f"{top_3_features['Correlation'].mean():.3f}",
        f"{cv:.3f}",
        ', '.join(optimal['Features']),
        f"{optimal['Mean_Correlation']:.3f}",
        f"{optimal['Variance']:.3f}",
        len(feature_combinations)
    ]
})

print(summary_df.to_string(index=False))

print(f"\nPlots saved to: {save_dir}")
print("=" * 80)
