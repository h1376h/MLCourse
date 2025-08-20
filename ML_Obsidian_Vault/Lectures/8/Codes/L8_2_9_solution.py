import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Given data
features = ['Age', 'Income', 'Gender', 'Education']
correlation = [0.45, 0.72, 0.15, 0.38]
mutual_info = [0.38, 0.65, 0.42, 0.35]
chi_square = [12.5, 28.3, 18.7, 15.2]

# Create DataFrame for easier manipulation
df = pd.DataFrame({
    'Feature': features,
    'Correlation': correlation,
    'Mutual_Info': mutual_info,
    'Chi_Square': chi_square
})

print("=== QUESTION 9: MULTI-CRITERIA FEATURE RANKING ===\n")
print("Given Feature-Target Relationships:")
print(df.to_string(index=False))
print()

# Task 1: Rank features by each criterion
print("=== TASK 1: RANKING FEATURES BY EACH CRITERION ===\n")

# Correlation ranking
corr_ranking = df.sort_values('Correlation', ascending=False)
print("1.1 Ranking by Correlation (higher is better):")
for i, (_, row) in enumerate(corr_ranking.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Correlation']:.2f}")
print()

# Mutual Information ranking
mi_ranking = df.sort_values('Mutual_Info', ascending=False)
print("1.2 Ranking by Mutual Information (higher is better):")
for i, (_, row) in enumerate(mi_ranking.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Mutual_Info']:.2f}")
print()

# Chi-Square ranking
chi_ranking = df.sort_values('Chi_Square', ascending=False)
print("1.3 Ranking by Chi-Square (higher is better):")
for i, (_, row) in enumerate(chi_ranking.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Chi_Square']:.1f}")
print()

# Task 2: Select exactly 2 features
print("=== TASK 2: SELECTING EXACTLY 2 FEATURES ===\n")

# Simple approach: select top 2 from each criterion and find common ones
top_corr = set(corr_ranking.head(2)['Feature'])
top_mi = set(mi_ranking.head(2)['Feature'])
top_chi = set(chi_ranking.head(2)['Feature'])

print("Top 2 features by each criterion:")
print(f"   Correlation: {', '.join(top_corr)}")
print(f"   Mutual Info: {', '.join(top_mi)}")
print(f"   Chi-Square: {', '.join(top_chi)}")

# Find features that appear in multiple top-2 lists
common_features = top_corr.intersection(top_mi).union(top_corr.intersection(top_chi)).union(top_mi.intersection(top_chi))
print(f"\nFeatures appearing in multiple top-2 lists: {', '.join(common_features) if common_features else 'None'}")

# Select 2 features based on overall performance
# Calculate average rank across all criteria
df['Avg_Rank'] = (df['Correlation'].rank(ascending=False) + 
                  df['Mutual_Info'].rank(ascending=False) + 
                  df['Chi_Square'].rank(ascending=False)) / 3

top_2_overall = df.nsmallest(2, 'Avg_Rank')
print(f"\nRecommended 2 features based on average ranking:")
for i, (_, row) in enumerate(top_2_overall.iterrows(), 1):
    print(f"   {i}. {row['Feature']} (Avg Rank: {row['Avg_Rank']:.2f})")
print()

# Task 3: Normalize metrics and calculate composite score
print("=== TASK 3: NORMALIZATION AND COMPOSITE SCORE ===\n")

# Normalize all metrics to 0-1 scale
df['Corr_Norm'] = (df['Correlation'] - df['Correlation'].min()) / (df['Correlation'].max() - df['Correlation'].min())
df['MI_Norm'] = (df['Mutual_Info'] - df['Mutual_Info'].min()) / (df['Mutual_Info'].max() - df['Mutual_Info'].min())
df['Chi_Norm'] = (df['Chi_Square'] - df['Chi_Square'].min()) / (df['Chi_Square'].max() - df['Chi_Square'].min())

print("Normalized Metrics (0-1 scale):")
print("Feature | Corr_Norm | MI_Norm | Chi_Norm")
print("-" * 40)
for _, row in df.iterrows():
    print(f"{row['Feature']:8} | {row['Corr_Norm']:9.3f} | {row['MI_Norm']:7.3f} | {row['Chi_Norm']:8.3f}")
print()

# Calculate composite score with weights: 40% correlation, 35% mutual info, 25% chi-square
weights = {'Corr_Norm': 0.40, 'MI_Norm': 0.35, 'Chi_Norm': 0.25}
df['Composite_Score'] = (df['Corr_Norm'] * weights['Corr_Norm'] + 
                         df['MI_Norm'] * weights['MI_Norm'] + 
                         df['Chi_Norm'] * weights['Chi_Norm'])

print("Composite Score Calculation:")
print("Composite = 0.40 × Corr_Norm + 0.35 × MI_Norm + 0.25 × Chi_Norm")
print()
for _, row in df.iterrows():
    comp_score = row['Composite_Score']
    print(f"{row['Feature']}: {comp_score:.3f} = 0.40×{row['Corr_Norm']:.3f} + 0.35×{row['MI_Norm']:.3f} + 0.25×{row['Chi_Norm']:.3f}")
print()

# Rank by composite score
composite_ranking = df.sort_values('Composite_Score', ascending=False)
print("Ranking by Composite Score:")
for i, (_, row) in enumerate(composite_ranking.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Composite_Score']:.3f}")
print()

# Task 4: Maximize minimum score across all criteria
print("=== TASK 4: MAXIMIZE MINIMUM SCORE ACROSS ALL CRITERIA ===\n")

# Find the minimum normalized score for each feature
df['Min_Score'] = df[['Corr_Norm', 'MI_Norm', 'Chi_Norm']].min(axis=1)

print("Minimum Score Across All Criteria:")
for _, row in df.iterrows():
    print(f"{row['Feature']}: min({row['Corr_Norm']:.3f}, {row['MI_Norm']:.3f}, {row['Chi_Norm']:.3f}) = {row['Min_Score']:.3f}")
print()

# Rank by minimum score
min_score_ranking = df.sort_values('Min_Score', ascending=False)
print("Ranking by Minimum Score (higher is better):")
for i, (_, row) in enumerate(min_score_ranking.iterrows(), 1):
    print(f"   {i}. {row['Feature']}: {row['Min_Score']:.3f}")
print()

# Create visualizations
print("=== GENERATING VISUALIZATIONS ===\n")

# 1. Bar chart of original metrics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Selection Metrics Comparison', fontsize=16)

# Original metrics
axes[0, 0].bar(features, correlation, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Correlation Coefficients')
axes[0, 0].set_ylabel('Correlation')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(features, mutual_info, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('Mutual Information')
axes[0, 1].set_ylabel('Mutual Information')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].bar(features, chi_square, color='lightgreen', alpha=0.7)
axes[1, 0].set_title('Chi-Square Statistics')
axes[1, 0].set_ylabel('Chi-Square')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Composite scores
axes[1, 1].bar(features, df['Composite_Score'], color='gold', alpha=0.7)
axes[1, 1].set_title('Composite Scores (Weighted Average)')
axes[1, 1].set_ylabel('Composite Score')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_metrics_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Heatmap of normalized metrics
plt.figure(figsize=(10, 8))
normalized_data = df[['Corr_Norm', 'MI_Norm', 'Chi_Norm']].values
sns.heatmap(normalized_data, 
            xticklabels=['Correlation', 'Mutual Info', 'Chi-Square'],
            yticklabels=features,
            annot=True, 
            fmt='.3f',
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Normalized Score (0-1)'})
plt.title('Normalized Feature Metrics Heatmap')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'normalized_metrics_heatmap.png'), dpi=300, bbox_inches='tight')

# 3. Radar chart for comprehensive comparison
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

# Prepare data for radar chart
categories = ['Correlation', 'Mutual Info', 'Chi-Square']
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Plot each feature
colors = ['red', 'blue', 'green', 'orange']
for i, feature in enumerate(features):
    values = [df.iloc[i]['Corr_Norm'], df.iloc[i]['MI_Norm'], df.iloc[i]['Chi_Norm']]
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=feature, color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Feature Performance Radar Chart (Normalized)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_radar_chart.png'), dpi=300, bbox_inches='tight')

# 4. Ranking comparison chart
fig, ax = plt.subplots(figsize=(12, 8))

# Create ranking data (lower rank = better)
ranking_data = np.array([
    [df[df['Feature'] == f]['Correlation'].rank(ascending=False).iloc[0] for f in features],
    [df[df['Feature'] == f]['Mutual_Info'].rank(ascending=False).iloc[0] for f in features],
    [df[df['Feature'] == f]['Chi_Square'].rank(ascending=False).iloc[0] for f in features]
])

x = np.arange(len(features))
width = 0.25

bars1 = ax.bar(x - width, ranking_data[0], width, label='Correlation', alpha=0.8)
bars2 = ax.bar(x, ranking_data[1], width, label='Mutual Info', alpha=0.8)
bars3 = ax.bar(x + width, ranking_data[2], width, label='Chi-Square', alpha=0.8)

ax.set_xlabel('Features')
ax.set_ylabel('Rank (Lower = Better)')
ax.set_title('Feature Rankings by Different Criteria')
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_rankings_comparison.png'), dpi=300, bbox_inches='tight')

# 5. Summary table visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary table
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Feature'],
        f"{row['Correlation']:.2f}",
        f"{row['Mutual_Info']:.2f}",
        f"{row['Chi_Square']:.1f}",
        f"{row['Composite_Score']:.3f}",
        f"{row['Min_Score']:.3f}"
    ])

table = ax.table(cellText=table_data,
                colLabels=['Feature', 'Correlation', 'Mutual Info', 'Chi-Square', 'Composite', 'Min Score'],
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the table
for i in range(len(table_data) + 1):
    for j in range(6):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#E8F5E8' if i % 2 == 0 else 'white')

plt.title('Feature Selection Summary Table', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_summary_table.png'), dpi=300, bbox_inches='tight')

print("All visualizations saved to:", save_dir)
print("\n=== SUMMARY OF RECOMMENDATIONS ===")
print("Based on the analysis:")
print(f"1. Top 2 features by composite score: {', '.join(composite_ranking.head(2)['Feature'].tolist())}")
print(f"2. Top 2 features by minimum score: {', '.join(min_score_ranking.head(2)['Feature'].tolist())}")
print(f"3. Most consistent performer: {min_score_ranking.iloc[0]['Feature']}")
print(f"4. Best overall performer: {composite_ranking.iloc[0]['Feature']}")
