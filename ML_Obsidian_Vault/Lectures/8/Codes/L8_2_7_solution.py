import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Configure plotting style (avoid LaTeX for compatibility)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')

print("=" * 80)
print("Question 7: Determining the Number of Features to Select")
print("=" * 80)

# Given cross-validation results
cv_data = {
    'Features': [5, 10, 15, 20, 25],
    'CV_Accuracy': [0.82, 0.87, 0.89, 0.88, 0.86],
    'Std_Dev': [0.03, 0.02, 0.02, 0.03, 0.04]
}

df = pd.DataFrame(cv_data)
print("\n1. Given Cross-Validation Results:")
print(df.to_string(index=False))

# Part 1: Cross-validation for finding optimal k
print("\n" + "="*60)
print("PART 1: Using Cross-Validation to Find Optimal k")
print("="*60)

print("\nCross-validation process for feature selection:")
print("1. Split data into k folds")
print("2. For each number of features (k):")
print("   - Train model on k-1 folds")
print("   - Validate on remaining fold")
print("   - Record accuracy")
print("3. Repeat for all folds and calculate mean accuracy")
print("4. Select k with highest mean CV accuracy")

# Part 2: Trade-offs between too few and too many features
print("\n" + "="*60)
print("PART 2: Trade-offs Between Feature Counts")
print("="*60)

print("\nToo Few Features:")
print("- Underfitting: Model cannot capture complex patterns")
print("- High bias, low variance")
print("- Poor performance on both training and test data")
print("- Loss of important information")

print("\nToo Many Features:")
print("- Overfitting: Model learns noise in the data")
print("- Low bias, high variance")
print("- Good training performance, poor test performance")
print("- Curse of dimensionality")
print("- Increased computational cost")
print("- Risk of spurious correlations")

# Part 3: Calculate 95% confidence intervals
print("\n" + "="*60)
print("PART 3: Calculating 95% Confidence Intervals")
print("="*60)

# Assuming we have k-fold CV (let's assume 5-fold for calculation)
k_folds = 5
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, df=k_folds - 1)

print(f"\nAssumptions:")
print(f"- Using {k_folds}-fold cross-validation")
print(f"- Confidence level: {confidence_level*100}%")
print(f"- Degrees of freedom: {k_folds - 1}")
print(f"- t-critical value: {t_critical:.4f}")

# Calculate confidence intervals
df['SE'] = df['Std_Dev'] / np.sqrt(k_folds)  # Standard Error
df['Margin_Error'] = t_critical * df['SE']
df['CI_Lower'] = df['CV_Accuracy'] - df['Margin_Error']
df['CI_Upper'] = df['CV_Accuracy'] + df['Margin_Error']

print(f"\nConfidence Interval Calculations:")
print("Formula: CI = mean ± t_critical × (std_dev / √n)")
print("\nDetailed Results:")
for i, row in df.iterrows():
    print(f"\n{row['Features']} features:")
    print(f"  Mean accuracy: {row['CV_Accuracy']:.3f}")
    print(f"  Standard deviation: {row['Std_Dev']:.3f}")
    print(f"  Standard error: {row['SE']:.4f}")
    print(f"  Margin of error: {row['Margin_Error']:.4f}")
    print(f"  95% CI: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")

# Part 4: Determine optimal number of features using the given rule
print("\n" + "="*60)
print("PART 4: Determining Optimal Number of Features")
print("="*60)

print("\nRule: Select the smallest number of features where the upper confidence")
print("bound of a larger feature set doesn't exceed the lower confidence bound")
print("of the current set.")

print(f"\nApplying the rule:")
optimal_features = None

for i in range(len(df)):
    current_features = df.iloc[i]['Features']
    current_lower = df.iloc[i]['CI_Lower']
    
    # Check if any larger feature set has upper bound <= current lower bound
    larger_sets_valid = True
    for j in range(i+1, len(df)):
        larger_upper = df.iloc[j]['CI_Upper']
        if larger_upper > current_lower:
            larger_sets_valid = False
            break
    
    print(f"\n{current_features} features: CI = [{df.iloc[i]['CI_Lower']:.4f}, {df.iloc[i]['CI_Upper']:.4f}]")
    
    if larger_sets_valid and optimal_features is None:
        optimal_features = current_features
        print(f"  → This could be optimal (no larger set exceeds lower bound)")
    else:
        print(f"  → Not optimal (larger sets exceed lower bound)")

if optimal_features is None:
    # If no feature count satisfies the rule, choose the one with highest accuracy
    optimal_features = df.loc[df['CV_Accuracy'].idxmax(), 'Features']
    print(f"\nNo feature count satisfies the strict rule.")
    print(f"Selecting {optimal_features} features (highest accuracy)")
else:
    print(f"\nOptimal number of features: {optimal_features}")

# Alternative analysis: Statistical significance
print(f"\nAlternative Analysis - Statistical Significance:")
print("Comparing each feature count with the best performing one:")

best_idx = df['CV_Accuracy'].idxmax()
best_features = df.iloc[best_idx]['Features']
best_accuracy = df.iloc[best_idx]['CV_Accuracy']
best_se = df.iloc[best_idx]['SE']

print(f"\nBest performing: {best_features} features with {best_accuracy:.3f} accuracy")

for i, row in df.iterrows():
    if i == best_idx:
        continue
    
    # Two-sample t-test comparison
    diff = best_accuracy - row['CV_Accuracy']
    se_diff = np.sqrt(best_se**2 + row['SE']**2)
    t_stat = diff / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=2*(k_folds-1)))
    
    print(f"{row['Features']} features: difference = {diff:.4f}, p-value = {p_value:.4f}")
    if p_value > 0.05:
        print(f"  → Not significantly different (p > 0.05)")
    else:
        print(f"  → Significantly different (p ≤ 0.05)")

# Part 5: Calculate sample sizes for different k-fold CV
print("\n" + "="*60)
print("PART 5: Sample Sizes for Different k-fold CV")
print("="*60)

total_samples = 1200
cv_folds = [3, 5, 10]

print(f"Total dataset size: {total_samples} samples\n")

for k in cv_folds:
    train_size = int(total_samples * (k-1) / k)
    val_size = int(total_samples / k)
    
    print(f"{k}-fold Cross-Validation:")
    print(f"  Training set size: {train_size} samples ({(k-1)/k*100:.1f}%)")
    print(f"  Validation set size: {val_size} samples ({1/k*100:.1f}%)")
    print(f"  Number of iterations: {k}")
    print()

# Create visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Figure 1: CV Results with Error Bars and Confidence Intervals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: CV Accuracy with Error Bars
ax1.errorbar(df['Features'], df['CV_Accuracy'], yerr=df['Std_Dev'], 
             marker='o', markersize=8, capsize=5, capthick=2, 
             linewidth=2, color='blue', label='CV Accuracy ± Std Dev')

# Add confidence intervals
ax1.fill_between(df['Features'], df['CI_Lower'], df['CI_Upper'], 
                 alpha=0.3, color='lightblue', label='95% Confidence Interval')

# Highlight optimal features
optimal_idx = df[df['Features'] == optimal_features].index[0]
ax1.scatter(optimal_features, df.iloc[optimal_idx]['CV_Accuracy'], 
           color='red', s=100, marker='*', zorder=5, label=f'Optimal: {optimal_features} features')

ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Cross-Validation Accuracy')
ax1.set_title('Cross-Validation Results')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0.78, 0.92)

# Plot 2: Confidence Intervals Comparison
width = 0.8
x_pos = np.arange(len(df))
bars = ax2.bar(x_pos, df['CV_Accuracy'], width, yerr=df['Margin_Error'],
               capsize=5, alpha=0.7, color=['red' if f == optimal_features else 'skyblue' 
                                           for f in df['Features']])

# Add value labels on bars
for i, (bar, acc, lower, upper) in enumerate(zip(bars, df['CV_Accuracy'], 
                                                  df['CI_Lower'], df['CI_Upper'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax2.text(bar.get_x() + bar.get_width()/2., lower - 0.01,
             f'[{lower:.3f}, {upper:.3f}]', ha='center', va='top', fontsize=8)

ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Cross-Validation Accuracy')
ax2.set_title('95% Confidence Intervals')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df['Features'])
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0.78, 0.92)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cv_results_and_confidence_intervals.png'), 
            dpi=300, bbox_inches='tight')
print("Saved: cv_results_and_confidence_intervals.png")

# Figure 2: Trade-offs Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Bias-Variance Trade-off
features_range = np.linspace(1, 30, 100)
bias = 0.3 + 0.2 * np.exp(-features_range/5)  # Bias decreases with more features
variance = 0.05 + 0.001 * features_range**1.5  # Variance increases with more features
total_error = bias + variance + 0.02  # Add irreducible error

ax1.plot(features_range, bias, 'r-', linewidth=2, label='Bias')
ax1.plot(features_range, variance, 'b-', linewidth=2, label='Variance')
ax1.plot(features_range, total_error, 'g-', linewidth=3, label='Total Error')
ax1.axvline(optimal_features, color='orange', linestyle='--', linewidth=2, 
           label=f'Optimal: {optimal_features} features')
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Error')
ax1.set_title('Bias-Variance Trade-off')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Model Complexity vs Performance
complexity = np.linspace(1, 30, 100)
train_acc = 0.6 + 0.35 * (1 - np.exp(-complexity/8))  # Training accuracy improves
val_acc = 0.6 + 0.3 * (1 - np.exp(-complexity/8)) - 0.001 * (complexity-15)**2  # Validation peaks then drops

ax2.plot(complexity, train_acc, 'g-', linewidth=2, label='Training Accuracy')
ax2.plot(complexity, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
ax2.axvline(optimal_features, color='orange', linestyle='--', linewidth=2,
           label=f'Optimal: {optimal_features} features')
ax2.scatter(df['Features'], df['CV_Accuracy'], color='red', s=50, 
           zorder=5, label='CV Results')
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Complexity vs Performance')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cross-validation Fold Sizes
cv_types = ['3-fold', '5-fold', '10-fold']
train_sizes = [800, 960, 1080]
val_sizes = [400, 240, 120]

x = np.arange(len(cv_types))
width = 0.35

bars1 = ax3.bar(x - width/2, train_sizes, width, label='Training Set', alpha=0.8)
bars2 = ax3.bar(x + width/2, val_sizes, width, label='Validation Set', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom')

ax3.set_xlabel('Cross-Validation Type')
ax3.set_ylabel('Sample Size')
ax3.set_title('Sample Sizes for Different k-fold CV\n(Total: 1200 samples)')
ax3.set_xticks(x)
ax3.set_xticklabels(cv_types)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Feature Selection Decision Tree
ax4.text(0.5, 0.9, 'Feature Selection Decision Process', 
         horizontalalignment='center', fontsize=14, fontweight='bold',
         transform=ax4.transAxes)

decision_text = """
1. Start with all available features

2. Apply feature selection method:
   - Filter methods (correlation, mutual info)
   - Wrapper methods (forward/backward selection)
   - Embedded methods (L1 regularization)

3. Use cross-validation to evaluate:
   - Split data into k folds
   - Train and validate for each feature subset
   - Calculate mean CV accuracy and std dev

4. Calculate confidence intervals:
   - CI = mean +/- t_critical * (std_dev / sqrt(k))

5. Apply selection rule:
   - Choose smallest feature set where
     larger sets don't significantly improve

6. Validate on independent test set
"""

ax4.text(0.05, 0.8, decision_text, transform=ax4.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_tradeoffs.png'), 
            dpi=300, bbox_inches='tight')
print("Saved: feature_selection_tradeoffs.png")

# Figure 3: Detailed Confidence Interval Analysis
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create horizontal confidence interval plot
y_positions = np.arange(len(df))
colors = ['red' if f == optimal_features else 'blue' for f in df['Features']]

for i, (idx, row) in enumerate(df.iterrows()):
    # Plot confidence interval as horizontal line
    ax.plot([row['CI_Lower'], row['CI_Upper']], [i, i], 'o-', 
            color=colors[i], linewidth=3, markersize=8, alpha=0.7)
    
    # Plot mean as a larger marker
    ax.plot(row['CV_Accuracy'], i, 's', color=colors[i], 
            markersize=12, alpha=0.9)
    
    # Add text labels
    ax.text(row['CI_Upper'] + 0.005, i, 
            f"{row['Features']} features\nMean: {row['CV_Accuracy']:.3f}\nCI: [{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]",
            verticalalignment='center', fontsize=9)

ax.set_yticks(y_positions)
ax.set_yticklabels([f"{f} features" for f in df['Features']])
ax.set_xlabel('Cross-Validation Accuracy')
ax.set_title('95% Confidence Intervals for Different Feature Counts\n(Red indicates optimal choice)')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(0.75, 0.95)

# Add vertical line for overall mean
overall_mean = df['CV_Accuracy'].mean()
ax.axvline(overall_mean, color='gray', linestyle=':', alpha=0.7, 
          label=f'Overall Mean: {overall_mean:.3f}')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confidence_intervals_detailed.png'), 
            dpi=300, bbox_inches='tight')
print("Saved: confidence_intervals_detailed.png")

# Summary table
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)

summary_df = df[['Features', 'CV_Accuracy', 'Std_Dev', 'CI_Lower', 'CI_Upper']].copy()
summary_df['Optimal'] = summary_df['Features'] == optimal_features
print(summary_df.to_string(index=False, float_format='%.4f'))

print(f"\n" + "="*60)
print("FINAL RECOMMENDATIONS")
print("="*60)
print(f"• Optimal number of features: {optimal_features}")
print(f"• Expected CV accuracy: {df[df['Features'] == optimal_features]['CV_Accuracy'].iloc[0]:.3f}")
print(f"• 95% Confidence interval: [{df[df['Features'] == optimal_features]['CI_Lower'].iloc[0]:.4f}, {df[df['Features'] == optimal_features]['CI_Upper'].iloc[0]:.4f}]")
print(f"• Recommended CV strategy: 5-fold or 10-fold for dataset of 1200 samples")

print(f"\nImages saved to: {save_dir}")
