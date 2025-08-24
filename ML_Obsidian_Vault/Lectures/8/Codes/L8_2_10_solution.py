import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 10: Feature Selection Thresholds")
print("=" * 50)

# Given feature scores
feature_scores = np.array([0.95, 0.87, 0.76, 0.65, 0.54, 0.43, 0.32, 0.21, 0.15, 0.08])
n_features = len(feature_scores)
print(f"Given feature scores: {feature_scores}")
print(f"Number of features: {n_features}")

# Sort scores in descending order for better visualization
sorted_scores = np.sort(feature_scores)[::-1]
print(f"Sorted scores (descending): {sorted_scores}")

# Calculate basic statistics
mean_score = np.mean(feature_scores)
std_score = np.std(feature_scores)
print(f"\nBasic Statistics:")
print(f"Mean score: {mean_score:.3f}")
print(f"Standard deviation: {std_score:.3f}")
print(f"Min score: {np.min(feature_scores):.3f}")
print(f"Max score: {np.max(feature_scores):.3f}")

# Task 1: How to set thresholds for different selection criteria
print("\n" + "="*60)
print("TASK 1: How to set thresholds for different selection criteria")
print("="*60)

print("Different approaches to set feature selection thresholds:")

# 1.1 Percentage-based threshold
print("\n1.1 Percentage-based threshold:")
print("   - Select top k% of features based on their scores")
print("   - Useful when you want to control the number of features relative to total")
print("   - Example: Select top 30% of features")

# 1.2 Absolute score threshold
print("\n1.2 Absolute score threshold:")
print("   - Set a minimum score threshold (e.g., 0.5)")
print("   - All features with scores above this threshold are selected")
print("   - Useful when you have domain knowledge about acceptable score levels")

# 1.3 Statistical threshold (z-score based)
print("\n1.3 Statistical threshold (z-score based):")
print("   - Set threshold based on statistical measures (e.g., mean + k*std)")
print("   - Helps identify features that are significantly above average")
print("   - Example: Select features with scores > mean + 2*std")

# 1.4 Gap-based threshold
print("\n1.4 Gap-based threshold:")
print("   - Look for natural gaps in the score distribution")
print("   - Useful when there are clear separations between good and poor features")

# Task 2: Effects of setting threshold too high or too low
print("\n" + "="*60)
print("TASK 2: Effects of setting threshold too high or too low")
print("="*60)

print("Setting threshold too HIGH:")
print("   - Fewer features selected")
print("   - Higher quality features (but may miss useful ones)")
print("   - Risk of underfitting due to insufficient features")
print("   - May lose important but slightly lower-scoring features")

print("\nSetting threshold too LOW:")
print("   - More features selected")
print("   - May include noisy or irrelevant features")
print("   - Risk of overfitting due to too many features")
print("   - Increased computational cost and reduced interpretability")

# Task 3: Calculate threshold for exactly 30% of features
print("\n" + "="*60)
print("TASK 3: Calculate threshold for exactly 30% of features")
print("="*60)

target_percentage = 0.30
n_features_to_select = int(np.ceil(n_features * target_percentage))
print(f"Target: Select {target_percentage*100}% of {n_features} features")
print(f"Number of features to select: {n_features_to_select}")

# Find the threshold that gives exactly 30% of features
if n_features_to_select > 0:
    # Sort scores in descending order and take the nth score as threshold
    threshold_30_percent = sorted_scores[n_features_to_select - 1]
    print(f"Threshold for {target_percentage*100}%: {threshold_30_percent:.3f}")
    
    # Show which features would be selected
    selected_features_30 = feature_scores >= threshold_30_percent
    n_selected_30 = np.sum(selected_features_30)
    print(f"Features selected with this threshold: {n_selected_30}")
    print(f"Selected feature scores: {feature_scores[selected_features_30]}")
else:
    print("No features would be selected with 0% threshold")

# Task 4: Threshold for 2 standard deviations above mean
print("\n" + "="*60)
print("TASK 4: Threshold for 2 standard deviations above mean")
print("="*60)

z_threshold = 2
threshold_2std = mean_score + z_threshold * std_score
print(f"Mean score: {mean_score:.3f}")
print(f"Standard deviation: {std_score:.3f}")
print(f"Z-score threshold: {z_threshold}")
print(f"Threshold calculation: {mean_score:.3f} + {z_threshold} × {std_score:.3f}")
print(f"Threshold = {threshold_2std:.3f}")

# Find features above this threshold
selected_features_2std = feature_scores >= threshold_2std
n_selected_2std = np.sum(selected_features_2std)
print(f"\nFeatures above this threshold: {n_selected_2std}")
print(f"Selected feature scores: {feature_scores[selected_features_2std]}")

# Calculate actual z-scores for verification
z_scores = (feature_scores - mean_score) / std_score
print(f"\nZ-scores for each feature:")
for i, (score, z_score) in enumerate(zip(feature_scores, z_scores)):
    status = "✓" if z_score >= z_threshold else "✗"
    print(f"  Feature {i+1}: Score = {score:.3f}, Z-score = {z_score:.3f} {status}")

# Visualization 1: Feature scores and thresholds
plt.figure(figsize=(12, 8))

# Plot 1: Feature scores with different thresholds
plt.subplot(2, 2, 1)
x_pos = np.arange(1, n_features + 1)
plt.bar(x_pos, sorted_scores, color='skyblue', alpha=0.7, edgecolor='navy')
plt.axhline(y=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold ({threshold_30_percent:.3f})')
plt.axhline(y=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold ({threshold_2std:.3f})')
plt.axhline(y=mean_score, color='orange', linestyle='-', linewidth=2, 
            label=f'Mean ({mean_score:.3f})')

plt.xlabel('Feature Rank')
plt.ylabel('Feature Score')
plt.title('Feature Scores with Different Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Cumulative percentage of features
plt.subplot(2, 2, 2)
cumulative_percentage = np.arange(1, n_features + 1) / n_features * 100
plt.plot(sorted_scores, cumulative_percentage, 'b-o', linewidth=2, markersize=6)
plt.axvline(x=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold ({threshold_30_percent:.3f})')
plt.axvline(x=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold ({threshold_2std:.3f})')

plt.xlabel('Feature Score')
plt.ylabel('Cumulative Percentage of Features (%)')
plt.title('Cumulative Feature Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Z-scores distribution
plt.subplot(2, 2, 3)
plt.bar(x_pos, z_scores, color='lightcoral', alpha=0.7, edgecolor='darkred')
plt.axhline(y=z_threshold, color='green', linestyle='--', linewidth=2, 
            label=f'Z-score threshold ({z_threshold})')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.axhline(y=-z_threshold, color='green', linestyle='--', linewidth=2, 
            label=f'Z-score threshold (-{z_threshold})')

plt.xlabel('Feature Index')
plt.ylabel('Z-Score')
plt.title('Feature Z-Scores')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Comparison of selection methods
plt.subplot(2, 2, 4)
methods = ['30% Selection', '2$\\sigma$ Selection', 'All Features']
counts = [n_selected_30, n_selected_2std, n_features]
colors = ['red', 'green', 'blue']

bars = plt.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Number of Features Selected')
plt.title('Feature Selection Comparison')
plt.ylim(0, n_features + 1)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_thresholds.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Detailed threshold analysis
plt.figure(figsize=(14, 10))

# Plot 1: Feature scores with annotations
plt.subplot(2, 3, 1)
plt.bar(x_pos, feature_scores, color='lightblue', alpha=0.7, edgecolor='navy')
plt.axhline(y=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold ({threshold_30_percent:.3f})')
plt.axhline(y=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold ({threshold_2std:.3f})')

# Annotate each feature
for i, score in enumerate(feature_scores):
    plt.annotate(f'{score:.2f}', (i+1, score), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8)

plt.xlabel('Feature Index')
plt.ylabel('Feature Score')
plt.title('Feature Scores with Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Score distribution histogram
plt.subplot(2, 3, 2)
plt.hist(feature_scores, bins=8, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
plt.axvline(x=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold')
plt.axvline(x=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold')
plt.axvline(x=mean_score, color='orange', linestyle='-', linewidth=2, 
            label=f'Mean')

plt.xlabel('Feature Score')
plt.ylabel('Frequency')
plt.title('Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Z-scores with selection regions
plt.subplot(2, 3, 3)
colors_z = ['red' if z >= z_threshold else 'lightgray' for z in z_scores]
plt.bar(x_pos, z_scores, color=colors_z, alpha=0.7, edgecolor='black')
plt.axhline(y=z_threshold, color='green', linestyle='--', linewidth=2, 
            label=f'Selection threshold ({z_threshold})')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Annotate z-scores
for i, z_score in enumerate(z_scores):
    plt.annotate(f'{z_score:.2f}', (i+1, z_score), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8)

plt.xlabel('Feature Index')
plt.ylabel('Z-Score')
plt.title('Z-Scores with Selection')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Threshold sensitivity analysis
plt.subplot(2, 3, 4)
thresholds = np.linspace(0, 1, 100)
n_selected_vs_threshold = [np.sum(feature_scores >= t) for t in thresholds]
percentage_selected = [n/n_features * 100 for n in n_selected_vs_threshold]

plt.plot(thresholds, percentage_selected, 'b-', linewidth=2)
plt.axvline(x=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold ({threshold_30_percent:.3f})')
plt.axvline(x=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold ({threshold_2std:.3f})')
plt.axhline(y=30, color='red', linestyle=':', alpha=0.7, label='30% target')

plt.xlabel('Threshold Value')
plt.ylabel('Percentage of Features Selected (%)')
plt.title('Threshold Sensitivity')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Feature ranking
plt.subplot(2, 3, 5)
ranked_indices = np.argsort(feature_scores)[::-1]
ranked_scores = feature_scores[ranked_indices]

plt.bar(range(1, n_features + 1), ranked_scores, color='gold', alpha=0.7, edgecolor='orange')
plt.axhline(y=threshold_30_percent, color='red', linestyle='--', linewidth=2, 
            label=f'30% threshold')
plt.axhline(y=threshold_2std, color='green', linestyle='--', linewidth=2, 
            label=f'2$\\sigma$ threshold')

plt.xlabel('Feature Rank')
plt.ylabel('Feature Score')
plt.title('Feature Ranking')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Selection method comparison
plt.subplot(2, 3, 6)
selection_methods = ['30% Selection', '2$\\sigma$ Selection', 'Mean Selection', 'Median Selection']
selection_counts = [
    n_selected_30,
    n_selected_2std,
    np.sum(feature_scores >= mean_score),
    np.sum(feature_scores >= np.median(feature_scores))
]

colors_comp = ['red', 'green', 'orange', 'purple']
bars = plt.bar(selection_methods, selection_counts, color=colors_comp, alpha=0.7, edgecolor='black')

# Add value labels
for bar, count in zip(bars, selection_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{count}', ha='center', va='bottom', fontweight='bold')

plt.ylabel('Number of Features Selected')
plt.title('Selection Method Comparison')
plt.xticks(rotation=45)
plt.ylim(0, n_features + 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_threshold_analysis.png'), dpi=300, bbox_inches='tight')

# Summary table
print("\n" + "="*60)
print("SUMMARY OF FEATURE SELECTION RESULTS")
print("="*60)

print(f"{'Method':<20} {'Threshold':<12} {'Features':<10} {'Percentage':<12}")
print("-" * 60)
print(f"{'30% Selection':<20} {threshold_30_percent:<12.3f} {n_selected_30:<10} {n_selected_30/n_features*100:<12.1f}%")
print(f"{'2$\\sigma$ Selection':<20} {threshold_2std:<12.3f} {n_selected_2std:<10} {n_selected_2std/n_features*100:<12.1f}%")
print(f"{'Mean Selection':<20} {mean_score:<12.3f} {np.sum(feature_scores >= mean_score):<10} {np.sum(feature_scores >= mean_score)/n_features*100:<12.1f}%")
print(f"{'Median Selection':<20} {np.median(feature_scores):<12.3f} {np.sum(feature_scores >= np.median(feature_scores)):<10} {np.sum(feature_scores >= np.median(feature_scores))/n_features*100:<12.1f}%")

print(f"\nPlots saved to: {save_dir}")
print("\nFeature selection analysis complete!")
