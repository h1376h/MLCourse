import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Feature Relevance Analysis - Question 10")
print("="*50)

# Dataset parameters
total_samples = 500
total_features = 50
highly_relevant = 10  # features 1-10
moderately_relevant = 20  # features 11-30
irrelevant = 20  # features 31-50

print(f"Dataset: {total_samples} samples, {total_features} features")
print(f"Feature breakdown:")
print(f"  - Highly relevant: features 1-{highly_relevant} ({highly_relevant} features)")
print(f"  - Moderately relevant: features {highly_relevant+1}-{highly_relevant+moderately_relevant} ({moderately_relevant} features)")
print(f"  - Irrelevant: features {highly_relevant+moderately_relevant+1}-{total_features} ({irrelevant} features)")

# Task 1: Percentage of highly relevant features
print("\n" + "="*50)
print("TASK 1: Percentage of highly relevant features")
print("="*50)

percentage_highly_relevant = (highly_relevant / total_features) * 100
print(f"Percentage of highly relevant features = {highly_relevant}/{total_features} × 100% = {percentage_highly_relevant}%")

# Task 2: Coverage when selecting top 20 features
print("\n" + "="*50)
print("TASK 2: Coverage when selecting top 20 features")
print("="*50)

top_20_features = 20
# Assuming features are ranked by relevance, top 20 would include:
# - All 10 highly relevant features
# - 10 out of 20 moderately relevant features
# - 0 irrelevant features

highly_relevant_in_top20 = min(highly_relevant, top_20_features)
moderately_relevant_in_top20 = max(0, min(moderately_relevant, top_20_features - highly_relevant_in_top20))
irrelevant_in_top20 = max(0, top_20_features - highly_relevant_in_top20 - moderately_relevant_in_top20)

print(f"When selecting top {top_20_features} features:")
print(f"  - Highly relevant captured: {highly_relevant_in_top20}/{highly_relevant} = {(highly_relevant_in_top20/highly_relevant)*100:.1f}%")
print(f"  - Moderately relevant captured: {moderately_relevant_in_top20}/{moderately_relevant} = {(moderately_relevant_in_top20/moderately_relevant)*100:.1f}%")
print(f"  - Irrelevant captured: {irrelevant_in_top20}/{irrelevant} = {(irrelevant_in_top20/irrelevant)*100:.1f}%")

total_relevant = highly_relevant + moderately_relevant
relevant_captured = highly_relevant_in_top20 + moderately_relevant_in_top20
coverage = (relevant_captured / total_relevant) * 100
print(f"Overall relevant coverage: {relevant_captured}/{total_relevant} = {coverage:.1f}%")

# Task 3: Quality measures
print("\n" + "="*50)
print("TASK 3: Quality measures for feature selection")
print("="*50)

print("Quality measures for feature selection:")
print("1. Precision = (Relevant features selected) / (Total features selected)")
print("2. Recall = (Relevant features selected) / (Total relevant features)")
print("3. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)")
print("4. Information Retention Ratio")
print("5. Feature Efficiency = Information gained / Number of features")

# Calculate for top 20 selection
precision = relevant_captured / top_20_features
recall = relevant_captured / total_relevant
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nFor top {top_20_features} feature selection:")
print(f"  - Precision = {relevant_captured}/{top_20_features} = {precision:.3f}")
print(f"  - Recall = {relevant_captured}/{total_relevant} = {recall:.3f}")
print(f"  - F1-Score = {f1_score:.3f}")

# Task 4: Optimal number of features
print("\n" + "="*50)
print("TASK 4: Optimal number of features")
print("="*50)

print("The optimal number of features depends on the trade-off between:")
print("1. Information retention")
print("2. Model complexity")
print("3. Computational efficiency")
print("4. Overfitting prevention")

# Theoretical optimal would be all relevant features
optimal_features_theory = highly_relevant + moderately_relevant
print(f"\nTheoretical optimal: {optimal_features_theory} features (all relevant features)")
print(f"Practical considerations might suggest fewer features to avoid overfitting")

# Task 5: Information retention with different feature counts
print("\n" + "="*50)
print("TASK 5: Information retention analysis")
print("="*50)

feature_counts = [10, 20, 30, 40, 50]
information_retention = []

print("Information retention for different feature counts:")
for count in feature_counts:
    # Calculate how many of each type are selected
    hr_selected = min(highly_relevant, count)
    mr_selected = max(0, min(moderately_relevant, count - hr_selected))
    ir_selected = max(0, count - hr_selected - mr_selected)
    
    # Simple retention model: each highly relevant = 1, moderately = 0.5, irrelevant = 0.1
    retention = (hr_selected * 1.0 + mr_selected * 0.5 + ir_selected * 0.1) / (highly_relevant * 1.0 + moderately_relevant * 0.5 + irrelevant * 0.1)
    information_retention.append(retention)
    
    print(f"  {count:2d} features: HR={hr_selected:2d}, MR={mr_selected:2d}, IR={ir_selected:2d} → Retention={retention:.3f}")

# Task 6: Detailed information analysis
print("\n" + "="*50)
print("TASK 6: Detailed information contribution analysis")
print("="*50)

# Given information contributions
hr_contribution = 0.60  # 60%
mr_contribution = 0.35  # 35%
ir_contribution = 0.05  # 5%

print(f"Information contributions:")
print(f"  - Highly relevant features (1-10): {hr_contribution*100}%")
print(f"  - Moderately relevant features (11-30): {mr_contribution*100}%")
print(f"  - Irrelevant features (31-50): {ir_contribution*100}%")

# Calculate information retention for specific selections
selections = [10, 20, 30]
detailed_results = []

print(f"\nDetailed analysis for different selections:")
for sel in selections:
    hr_sel = min(highly_relevant, sel)
    mr_sel = max(0, min(moderately_relevant, sel - hr_sel))
    ir_sel = max(0, sel - hr_sel - mr_sel)
    
    # Calculate information retained
    info_retained = (hr_sel/highly_relevant) * hr_contribution + \
                   (mr_sel/moderately_relevant) * mr_contribution + \
                   (ir_sel/irrelevant) * ir_contribution
    
    # Information-to-feature ratio
    info_ratio = info_retained / sel
    
    detailed_results.append({
        'features': sel,
        'hr_selected': hr_sel,
        'mr_selected': mr_sel,
        'ir_selected': ir_sel,
        'info_retained': info_retained,
        'info_ratio': info_ratio
    })
    
    print(f"\nTop {sel} features:")
    print(f"  Selected: HR={hr_sel}, MR={mr_sel}, IR={ir_sel}")
    print(f"  Information retained: {info_retained:.4f} ({info_retained*100:.2f}%)")
    print(f"  Information-to-feature ratio: {info_ratio:.4f}")

# Find best ratio
best_ratio_idx = max(range(len(detailed_results)), key=lambda i: detailed_results[i]['info_ratio'])
best_selection = detailed_results[best_ratio_idx]

print(f"\nBest information-to-feature ratio:")
print(f"  {best_selection['features']} features with ratio {best_selection['info_ratio']:.4f}")

# Create visualizations
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Figure 1: Feature distribution and relevance
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Feature categories pie chart
categories = ['Highly Relevant', 'Moderately Relevant', 'Irrelevant']
sizes = [highly_relevant, moderately_relevant, irrelevant]
colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
explode = (0.1, 0.05, 0)

ax1.pie(sizes, explode=explode, labels=categories, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.set_title('Feature Distribution by Relevance')

# Subplot 2: Information contribution
info_contributions = [hr_contribution*100, mr_contribution*100, ir_contribution*100]
bars = ax2.bar(categories, info_contributions, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Information Contribution (%)')
ax2.set_title('Information Contribution by Feature Type')
ax2.set_ylim(0, 70)

# Add value labels on bars
for bar, value in zip(bars, info_contributions):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value:.1f}%', ha='center', va='bottom')

# Subplot 3: Information retention vs feature count
feature_range = np.arange(1, 51)
retention_curve = []

for count in feature_range:
    hr_sel = min(highly_relevant, count)
    mr_sel = max(0, min(moderately_relevant, count - hr_sel))
    ir_sel = max(0, count - hr_sel - mr_sel)
    
    info_ret = (hr_sel/highly_relevant) * hr_contribution + \
               (mr_sel/moderately_relevant) * mr_contribution + \
               (ir_sel/irrelevant) * ir_contribution
    retention_curve.append(info_ret)

ax3.plot(feature_range, retention_curve, 'b-', linewidth=2, label='Information Retention')
ax3.axhline(y=hr_contribution, color='r', linestyle='--', alpha=0.7, 
            label=f'HR Only ({hr_contribution*100:.0f}%)')
ax3.axhline(y=hr_contribution+mr_contribution, color='g', linestyle='--', alpha=0.7,
            label=f'HR+MR ({(hr_contribution+mr_contribution)*100:.0f}%)')

# Mark special points
for sel in selections:
    idx = sel - 1
    ax3.plot(sel, retention_curve[idx], 'ro', markersize=8)
    ax3.annotate(f'{sel} features\n{retention_curve[idx]*100:.1f}%', 
                xy=(sel, retention_curve[idx]), xytext=(sel+5, retention_curve[idx]),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

ax3.set_xlabel('Number of Features Selected')
ax3.set_ylabel('Information Retention')
ax3.set_title('Information Retention vs Feature Count')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_xlim(0, 50)
ax3.set_ylim(0, 1.05)

# Subplot 4: Information-to-feature ratio
ratios = [res['info_ratio'] for res in detailed_results]
feature_nums = [res['features'] for res in detailed_results]

bars = ax4.bar([str(f) for f in feature_nums], ratios, color=['#3498db', '#e74c3c', '#2ecc71'], 
               alpha=0.7, edgecolor='black')
ax4.set_ylabel('Information-to-Feature Ratio')
ax4.set_xlabel('Number of Features Selected')
ax4.set_title('Information Efficiency Analysis')

# Add value labels on bars
for bar, value in zip(bars, ratios):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.4f}', ha='center', va='bottom')

# Highlight the best ratio
best_bar = bars[best_ratio_idx]
best_bar.set_color('#f39c12')
best_bar.set_alpha(1.0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_relevance_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 2: Detailed breakdown for each selection
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

for i, result in enumerate(detailed_results):
    # Stacked bar chart showing feature composition
    hr_sel = result['hr_selected']
    mr_sel = result['mr_selected'] 
    ir_sel = result['ir_selected']
    
    ax = eval(f'ax{i+1}')
    
    # Create stacked bar
    bottom = 0
    if hr_sel > 0:
        ax.bar('Selected', hr_sel, bottom=bottom, color='#ff6b6b', 
               label='Highly Relevant', alpha=0.8)
        bottom += hr_sel
    
    if mr_sel > 0:
        ax.bar('Selected', mr_sel, bottom=bottom, color='#4ecdc4', 
               label='Moderately Relevant', alpha=0.8)
        bottom += mr_sel
    
    if ir_sel > 0:
        ax.bar('Selected', ir_sel, bottom=bottom, color='#95a5a6', 
               label='Irrelevant', alpha=0.8)
    
    # Total available features
    ax.bar('Available', highly_relevant, bottom=0, color='#ff6b6b', alpha=0.3)
    ax.bar('Available', moderately_relevant, bottom=highly_relevant, color='#4ecdc4', alpha=0.3)
    ax.bar('Available', irrelevant, bottom=highly_relevant+moderately_relevant, color='#95a5a6', alpha=0.3)
    
    ax.set_ylabel('Number of Features')
    ax.set_title(f'Top {result["features"]} Features\nInfo Retained: {result["info_retained"]*100:.1f}%\nEfficiency: {result["info_ratio"]:.4f}')
    ax.set_ylim(0, 55)
    
    # Add text annotations
    ax.text(0, hr_sel/2, f'{hr_sel}', ha='center', va='center', fontweight='bold') if hr_sel > 0 else None
    ax.text(0, hr_sel + mr_sel/2, f'{mr_sel}', ha='center', va='center', fontweight='bold') if mr_sel > 0 else None
    ax.text(0, hr_sel + mr_sel + ir_sel/2, f'{ir_sel}', ha='center', va='center', fontweight='bold') if ir_sel > 0 else None
    
    if i == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_breakdown.png'), dpi=300, bbox_inches='tight')

# Figure 3: Quality metrics comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Calculate quality metrics for different selections
metrics_data = []
feature_counts_extended = range(5, 51, 5)

for count in feature_counts_extended:
    hr_sel = min(highly_relevant, count)
    mr_sel = max(0, min(moderately_relevant, count - hr_sel))
    ir_sel = max(0, count - hr_sel - mr_sel)
    
    relevant_sel = hr_sel + mr_sel
    precision = relevant_sel / count if count > 0 else 0
    recall = relevant_sel / total_relevant
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    info_ret = (hr_sel/highly_relevant) * hr_contribution + \
               (mr_sel/moderately_relevant) * mr_contribution + \
               (ir_sel/irrelevant) * ir_contribution
    
    metrics_data.append({
        'features': count,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'info_retention': info_ret
    })

# Plot precision
precisions = [m['precision'] for m in metrics_data]
ax1.plot(feature_counts_extended, precisions, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Number of Features')
ax1.set_ylabel('Precision')
ax1.set_title('Precision vs Feature Count')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# Plot recall
recalls = [m['recall'] for m in metrics_data]
ax2.plot(feature_counts_extended, recalls, 'r-s', linewidth=2, markersize=6)
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('Recall')
ax2.set_title('Recall vs Feature Count')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

# Plot F1-score
f1_scores = [m['f1'] for m in metrics_data]
ax3.plot(feature_counts_extended, f1_scores, 'g-^', linewidth=2, markersize=6)
ax3.set_xlabel('Number of Features')
ax3.set_ylabel('F1-Score')
ax3.set_title('F1-Score vs Feature Count')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.05)

# Plot all metrics together
ax4.plot(feature_counts_extended, precisions, 'b-o', label='Precision', linewidth=2)
ax4.plot(feature_counts_extended, recalls, 'r-s', label='Recall', linewidth=2)
ax4.plot(feature_counts_extended, f1_scores, 'g-^', label='F1-Score', linewidth=2)
info_retentions = [m['info_retention'] for m in metrics_data]
ax4.plot(feature_counts_extended, info_retentions, 'm-d', label='Info Retention', linewidth=2)

ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Metric Value')
ax4.set_title('All Quality Metrics Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'quality_metrics_analysis.png'), dpi=300, bbox_inches='tight')

# Summary table
print("\n" + "="*50)
print("SUMMARY TABLE")
print("="*50)

print(f"{'Features':<10} {'HR':<4} {'MR':<4} {'IR':<4} {'Info Ret':<10} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'Efficiency':<10}")
print("-" * 80)

for result in detailed_results:
    features = result['features']
    hr_sel = result['hr_selected']
    mr_sel = result['mr_selected']
    ir_sel = result['ir_selected']
    info_ret = result['info_retained']
    
    relevant_sel = hr_sel + mr_sel
    precision = relevant_sel / features
    recall = relevant_sel / total_relevant
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    efficiency = result['info_ratio']
    
    print(f"{features:<10} {hr_sel:<4} {mr_sel:<4} {ir_sel:<4} {info_ret:<10.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f} {efficiency:<10.4f}")

print(f"\nPlots saved to: {save_dir}")
print("\nAnalysis complete!")
