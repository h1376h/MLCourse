import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 14: Random Forest Feature Importance Analysis")
print("=" * 60)

# Given feature importance data
feature_names = ['Monthly_Charges', 'Contract_Length', 'Internet_Service', 'Payment_Method', 'Gender']
importance_scores = [0.45, 0.28, 0.15, 0.08, 0.04]

# Create DataFrame for easier manipulation
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
})

# Sort by importance (descending)
feature_df = feature_df.sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nGiven Feature Importance Rankings:")
print(feature_df.to_string(index=False))

total_importance = feature_df['Importance'].sum()
print(f"\nTotal Importance: {total_importance:.2f}")

# Task 1: Remove bottom 40% of features
print("\n" + "="*60)
print("TASK 1: Remove bottom 40% of features")
print("="*60)

# Calculate how many features represent bottom 40%
total_features = len(feature_names)
print(f"Step 1.1: Calculate total number of features")
print(f"  Total features = {total_features}")

bottom_40_percent = total_features * 0.4
print(f"Step 1.2: Calculate bottom 40%")
print(f"  Bottom 40% = {total_features} × 0.4 = {bottom_40_percent}")

features_to_remove = int(bottom_40_percent)
print(f"Step 1.3: Determine features to remove")
print(f"  Features to remove = int({bottom_40_percent}) = {features_to_remove}")

# Remove bottom features
remaining_features = feature_df.iloc[:-features_to_remove]
removed_features = feature_df.iloc[-features_to_remove:]

print(f"\nStep 1.4: Identify features to remove (bottom {features_to_remove}):")
for i, (_, row) in enumerate(removed_features.iterrows()):
    print(f"  Feature {i+1}: {row['Feature']} = {row['Importance']:.2f}")

print(f"\nStep 1.5: Features remaining after removing bottom 40%:")
for i, (_, row) in enumerate(remaining_features.iterrows()):
    print(f"  Feature {i+1}: {row['Feature']} = {row['Importance']:.2f}")

remaining_importance = remaining_features['Importance'].sum()
importance_preserved_40 = (remaining_importance / total_importance) * 100
print(f"\nStep 1.6: Calculate importance preserved")
print(f"  Remaining importance = {remaining_importance:.2f}")
print(f"  Importance preserved = ({remaining_importance:.2f} / {total_importance:.2f}) × 100 = {importance_preserved_40:.1f}%")

# Task 2: Percentage of total importance for top 3 features
print("\n" + "="*60)
print("TASK 2: Percentage of total importance for top 3 features")
print("="*60)

print("Step 2.1: Identify top 3 features")
top_3_features = feature_df.head(3)
for i, (_, row) in enumerate(top_3_features.iterrows(), 1):
    print(f"  Feature {i}: {row['Feature']} = {row['Importance']:.2f}")

print(f"\nStep 2.2: Calculate total importance of top 3 features")
top_3_importance = top_3_features['Importance'].sum()
print(f"  Top 3 importance = {top_3_features.iloc[0]['Importance']:.2f} + {top_3_features.iloc[1]['Importance']:.2f} + {top_3_features.iloc[2]['Importance']:.2f}")
print(f"  Top 3 importance = {top_3_importance:.2f}")

print(f"\nStep 2.3: Calculate percentage of total importance")
percentage_top_3 = (top_3_importance / total_importance) * 100
print(f"  Percentage = ({top_3_importance:.2f} / {total_importance:.2f}) × 100")
print(f"  Percentage = {percentage_top_3:.1f}%")

# Task 3: Reduce features to 60% of original
print("\n" + "="*60)
print("TASK 3: Reduce features to 60% of original")
print("="*60)

print("Step 3.1: Calculate target number of features")
target_features = int(total_features * 0.6)
print(f"  Target features = int({total_features} × 0.6) = int({total_features * 0.6}) = {target_features}")

print(f"\nStep 3.2: Select top {target_features} features")
selected_features = feature_df.head(target_features)
for i, (_, row) in enumerate(selected_features.iterrows(), 1):
    print(f"  Feature {i}: {row['Feature']} = {row['Importance']:.2f}")

print(f"\nStep 3.3: Calculate importance of selected features")
selected_importance = selected_features['Importance'].sum()
print(f"  Selected importance = {selected_importance:.2f}")

print(f"\nStep 3.4: Calculate percentage of importance preserved")
selected_percentage = (selected_importance / total_importance) * 100
print(f"  Importance preserved = ({selected_importance:.2f} / {total_importance:.2f}) × 100 = {selected_percentage:.1f}%")

# Task 4: Design strategy to preserve 90% of importance
print("\n" + "="*60)
print("TASK 4: Design strategy to preserve 90% of importance")
print("="*60)

print("Step 4.1: Calculate target importance to preserve")
target_importance = total_importance * 0.9
print(f"  Target importance = {total_importance:.2f} × 0.9 = {target_importance:.2f}")

print(f"\nStep 4.2: Find minimum number of features needed")
print("  We need to add features one by one until we reach the target importance:")

cumulative_importance_90 = 0
features_needed = 0
selected_for_90 = []

for i, (_, row) in enumerate(feature_df.iterrows()):
    cumulative_importance_90 += row['Importance']
    selected_for_90.append(row['Feature'])
    features_needed = i + 1
    
    print(f"    Feature {features_needed}: {row['Feature']} = {row['Importance']:.2f}")
    print(f"    Cumulative importance = {cumulative_importance_90:.2f}")
    
    if cumulative_importance_90 >= target_importance:
        print(f"    ✓ Target reached! {cumulative_importance_90:.2f} ≥ {target_importance:.2f}")
        break
    else:
        print(f"    ✗ Target not reached yet. {cumulative_importance_90:.2f} < {target_importance:.2f}")

print(f"\nStep 4.3: Summary of strategy")
print(f"  Features needed: {features_needed}")
print(f"  Importance preserved: {cumulative_importance_90:.2f}")

print(f"\nStep 4.4: Calculate actual percentage preserved")
actual_percentage_90 = (cumulative_importance_90 / total_importance) * 100
print(f"  Actual percentage = ({cumulative_importance_90:.2f} / {total_importance:.2f}) × 100 = {actual_percentage_90:.1f}%")

print(f"\nStep 4.5: Selected features for 90% preservation:")
for i, feature in enumerate(selected_for_90, 1):
    importance = feature_df[feature_df['Feature'] == feature]['Importance'].iloc[0]
    print(f"  Feature {i}: {feature} = {importance:.2f}")

# Visualization 1: Feature Importance Bar Chart
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(feature_df)), feature_df['Importance'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance Score', fontsize=14)
plt.title('Random Forest Feature Importance Rankings', fontsize=16, fontweight='bold')
plt.xticks(range(len(feature_df)), feature_df['Feature'], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, importance) in enumerate(zip(bars, feature_df['Importance'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{importance:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_rankings.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Cumulative Importance Plot
plt.figure(figsize=(10, 8))
cumulative_importance = np.cumsum(feature_df['Importance'])
plt.plot(range(1, len(feature_df) + 1), cumulative_importance, 'bo-', linewidth=3, markersize=10)
plt.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% Threshold')
plt.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, label='60% Threshold')
plt.xlabel('Number of Features', fontsize=14)
plt.ylabel('Cumulative Importance', fontsize=14)
plt.title('Cumulative Importance vs Number of Features', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xticks(range(1, len(feature_df) + 1))

# Add annotations for key points
for i, (x, y) in enumerate(zip(range(1, len(feature_df) + 1), cumulative_importance)):
    plt.annotate(f'({x}, {y:.2f})', (x, y), xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cumulative_importance_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Feature Selection Strategy
plt.figure(figsize=(12, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = plt.bar(range(len(feature_df)), feature_df['Importance'], color=colors)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance Score', fontsize=14)
plt.title('Feature Selection Strategy for 90% Importance Preservation', fontsize=16, fontweight='bold')
plt.xticks(range(len(feature_df)))
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')

# Highlight selected features for 90% preservation
for i, (bar, importance) in enumerate(zip(bars, feature_df['Importance'])):
    if i < features_needed:
        bar.set_color('green')
        bar.set_alpha(0.8)
    else:
        bar.set_color('lightgray')
        bar.set_alpha(0.5)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.8, label='Selected for 90% preservation'),
                   Patch(facecolor='lightgray', alpha=0.5, label='Not selected')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_strategy.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Top 3 vs Others Pie Chart
plt.figure(figsize=(10, 8))
top_3_importance = feature_df.head(3)['Importance'].sum()
other_importance = feature_df.tail(2)['Importance'].sum()

labels = ['Top 3 Features', 'Other Features']
sizes = [top_3_importance, other_importance]
colors_pie = ['#FF6B6B', '#E0E0E0']

plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Importance: Top 3 vs Others', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'top3_vs_others_pie.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 5: Feature Reduction Strategies Comparison
plt.figure(figsize=(12, 8))
reduction_scenarios = ['Original', 'Remove Bottom 40%', '60% of Original', '90% Importance']
importance_preserved = [100, importance_preserved_40, selected_percentage, actual_percentage_90]

x_pos = np.arange(len(reduction_scenarios))
bars = plt.bar(x_pos, importance_preserved, color=['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4'])
plt.xlabel('Reduction Strategy', fontsize=14)
plt.ylabel('Importance Preserved (%)', fontsize=14)
plt.title('Feature Reduction Strategies Comparison', fontsize=16, fontweight='bold')
plt.xticks(x_pos)
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
plt.ylim(0, 105)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, importance_preserved)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'reduction_strategies_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 6: Feature Importance with Thresholds
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(feature_df)), feature_df['Importance'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])

# Add horizontal lines for different thresholds
plt.axhline(y=0.45, color='red', linestyle='--', alpha=0.7, label='Monthly_Charges (0.45)')
plt.axhline(y=0.28, color='blue', linestyle='--', alpha=0.7, label='Contract_Length (0.28)')
plt.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='Internet_Service (0.15)')

plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance Score', fontsize=14)
plt.title('Feature Importance with Individual Thresholds', fontsize=16, fontweight='bold')
plt.xticks(range(len(feature_df)))
plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Add value labels on bars
for i, (bar, importance) in enumerate(zip(bars, feature_df['Importance'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{importance:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_with_thresholds.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"1. After removing bottom 40%: {len(remaining_features)} features remain")
print(f"2. Top 3 features represent {percentage_top_3:.1f}% of total importance")
print(f"3. To reduce to 60% of original: select {target_features} features")
print(f"4. To preserve 90% importance: select {features_needed} features")

print(f"\nPlots saved to: {save_dir}")
print("Generated visualizations:")
print("  - feature_importance_rankings.png")
print("  - cumulative_importance_plot.png")
print("  - feature_selection_strategy.png")
print("  - top3_vs_others_pie.png")
print("  - reduction_strategies_comparison.png")
print("  - feature_importance_with_thresholds.png")
