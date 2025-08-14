import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 16: ALGORITHM SELECTION SCENARIOS")
print("=" * 80)

# Define the scenarios and analysis
scenarios = {
    1: {
        'description': 'Small dataset with only categorical features and no missing values',
        'characteristics': {
            'Dataset Size': 'Small',
            'Feature Types': 'Categorical only',
            'Missing Values': 'None',
            'Target Type': 'Classification',
            'Interpretability Need': 'Medium'
        },
        'best_algorithm': 'ID3',
        'reasoning': [
            'Simple and sufficient for categorical-only features',
            'No missing values to handle',
            'Small dataset means computational efficiency less critical',
            'ID3\'s simplicity makes it easy to understand and implement',
            'No need for advanced features of C4.5 or CART'
        ]
    },
    2: {
        'description': 'Large dataset with mixed feature types and 20% missing values',
        'characteristics': {
            'Dataset Size': 'Large',
            'Feature Types': 'Mixed (cat. + cont.)',
            'Missing Values': '20%',
            'Target Type': 'Classification',
            'Interpretability Need': 'Medium'
        },
        'best_algorithm': 'CART',
        'reasoning': [
            'Handles mixed feature types efficiently',
            'Surrogate splits provide robust missing value handling',
            'Scales well to large datasets',
            'Binary splits are computationally efficient',
            'Better than C4.5 for high missing value rates'
        ]
    },
    3: {
        'description': 'Medical diagnosis requiring highly interpretable rules',
        'characteristics': {
            'Dataset Size': 'Medium',
            'Feature Types': 'Mixed',
            'Missing Values': 'Low',
            'Target Type': 'Classification',
            'Interpretability Need': 'Very High'
        },
        'best_algorithm': 'C4.5',
        'reasoning': [
            'Produces highly interpretable decision rules',
            'Pruning reduces overfitting and improves generalization',
            'Handles mixed feature types naturally',
            'Gain ratio reduces bias toward multi-valued features',
            'Well-established in medical applications'
        ]
    },
    4: {
        'description': 'Predicting house prices (continuous target variable)',
        'characteristics': {
            'Dataset Size': 'Medium-Large',
            'Feature Types': 'Mixed',
            'Missing Values': 'Variable',
            'Target Type': 'Regression',
            'Interpretability Need': 'Medium'
        },
        'best_algorithm': 'CART',
        'reasoning': [
            'Only algorithm that natively supports regression',
            'Variance-based splitting optimizes for continuous targets',
            'Handles mixed feature types well',
            'Surrogate splits handle missing values',
            'Widely used in real estate prediction models'
        ]
    },
    5: {
        'description': 'Dataset with many categorical features having 10+ values each',
        'characteristics': {
            'Dataset Size': 'Medium',
            'Feature Types': 'Categorical (high cardinality)',
            'Missing Values': 'Low',
            'Target Type': 'Classification',
            'Interpretability Need': 'Medium'
        },
        'best_algorithm': 'C4.5',
        'reasoning': [
            'Gain ratio specifically addresses bias toward high-cardinality features',
            'Information gain alone (ID3) would be heavily biased',
            'C4.5 provides more balanced feature selection',
            'Better handling of features with many possible values',
            'Prevents overfitting due to high cardinality'
        ]
    }
}

print("SCENARIO ANALYSIS:")
print("=" * 50)

for scenario_id, scenario in scenarios.items():
    print(f"\nScenario {scenario_id}: {scenario['description']}")
    print(f"Best Algorithm: {scenario['best_algorithm']}")
    print("Reasoning:")
    for reason in scenario['reasoning']:
        print(f"  • {reason}")

# Create detailed comparison matrix
algorithms = ['ID3', 'C4.5', 'CART']
criteria = ['Categorical Features', 'Continuous Features', 'Missing Values', 'Regression', 
           'Large Datasets', 'High Cardinality', 'Interpretability', 'Robustness']

# Scoring matrix (1-5 scale: 1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent)
scores = {
    'ID3': [5, 1, 1, 1, 3, 2, 4, 2],      # Excellent for categorical, poor for others
    'C4.5': [5, 4, 3, 1, 4, 5, 5, 4],     # Well-rounded, excellent for high cardinality
    'CART': [4, 5, 5, 5, 5, 3, 4, 5]      # Excellent for continuous, regression, robustness
}

print("\n" + "=" * 80)
print("ALGORITHM COMPARISON MATRIX")
print("=" * 80)

# Create DataFrame for better visualization
comparison_df = pd.DataFrame(scores, index=criteria)
print(comparison_df.to_string())

# Calculate scenario-specific scores
print("\n" + "=" * 80)
print("SCENARIO-SPECIFIC ALGORITHM SCORES")
print("=" * 80)

scenario_weights = {
    1: {'Categorical Features': 0.4, 'Missing Values': 0.3, 'Interpretability': 0.2, 'Large Datasets': 0.1},
    2: {'Continuous Features': 0.3, 'Missing Values': 0.3, 'Large Datasets': 0.3, 'Robustness': 0.1},
    3: {'Interpretability': 0.4, 'Continuous Features': 0.2, 'Missing Values': 0.2, 'Robustness': 0.2},
    4: {'Regression': 0.5, 'Continuous Features': 0.3, 'Missing Values': 0.1, 'Large Datasets': 0.1},
    5: {'High Cardinality': 0.4, 'Categorical Features': 0.3, 'Interpretability': 0.2, 'Robustness': 0.1}
}

for scenario_id, weights in scenario_weights.items():
    print(f"\nScenario {scenario_id}: {scenarios[scenario_id]['description']}")
    scenario_scores = {}
    
    for alg in algorithms:
        score = 0
        for criterion, weight in weights.items():
            criterion_idx = criteria.index(criterion)
            criterion_score = scores[alg][criterion_idx]
            score += weight * criterion_score
        scenario_scores[alg] = score
    
    # Sort by score
    sorted_algs = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Algorithm rankings:")
    for i, (alg, score) in enumerate(sorted_algs):
        rank_text = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"  {rank_text} {alg}: {score:.2f}")

# Create separate focused visualizations

# Visualization 1: Algorithm comparison heatmap
fig1, ax1 = plt.subplots(figsize=(12, 6))
heatmap_data = np.array([scores[alg] for alg in algorithms])

im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
ax1.set_xticks(range(len(criteria)))
ax1.set_yticks(range(len(algorithms)))
ax1.set_xticklabels(criteria, rotation=45, ha='right')
ax1.set_yticklabels(algorithms)

# Add score annotations
for i in range(len(algorithms)):
    for j in range(len(criteria)):
        text = ax1.text(j, i, heatmap_data[i, j], ha="center", va="center", 
                       color="white" if heatmap_data[i, j] < 3 else "black", fontweight='bold')

ax1.set_title('Algorithm Capability Matrix\n(1=Poor, 2=Fair, 3=Good, 4=Very Good, 5=Excellent)', 
              fontsize=14, weight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('Capability Score', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_capability_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Scenario recommendations pie chart
fig2, ax2 = plt.subplots(figsize=(8, 8))
scenario_nums = list(scenarios.keys())
recommended_algs = [scenarios[i]['best_algorithm'] for i in scenario_nums]

# Count recommendations
alg_counts = {alg: recommended_algs.count(alg) for alg in algorithms}
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

wedges, texts, autotexts = ax2.pie(alg_counts.values(), labels=alg_counts.keys(), 
                                  colors=colors, autopct='%1.0f', startangle=90)
ax2.set_title('Algorithm Recommendations Across Scenarios', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scenario_recommendations.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Individual scenario analysis (separate images)
for scenario_id, scenario in scenarios.items():
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    best_alg = scenario['best_algorithm']
    char_names = list(scenario['characteristics'].keys())
    char_values = list(scenario['characteristics'].values())
    y_pos = np.arange(len(char_names))
    
    # Create horizontal bar chart
    colors_map = {'ID3': '#FF6B6B', 'C4.5': '#4ECDC4', 'CART': '#45B7D1'}
    bars = ax3.barh(y_pos, [1]*len(char_names), color=colors_map[best_alg], alpha=0.7)
    
    # Add text annotations
    for j, (name, value) in enumerate(scenario['characteristics'].items()):
        ax3.text(0.05, j, f"{name}: {value}", va='center', fontsize=11, weight='bold')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(char_names)
    ax3.set_xlim(0, 1)
    ax3.set_xticks([])
    ax3.set_title(f'Scenario {scenario_id}: {scenario["description"]}\nRecommended: {best_alg}', 
                 fontsize=12, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scenario_{scenario_id}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create simplified decision flowchart
fig, ax = plt.subplots(figsize=(14, 10))

ax.text(0.5, 0.95, 'Algorithm Selection Flowchart', 
        ha='center', fontsize=16, weight='bold')

# Simplified decision tree structure
# Root node
root_box = FancyBboxPatch((0.4, 0.8), 0.2, 0.08,
                         boxstyle="round,pad=0.01",
                         facecolor='lightgray', edgecolor='black', linewidth=2)
ax.add_patch(root_box)
ax.text(0.5, 0.84, 'Target Type?', ha='center', va='center', fontsize=12, weight='bold')

# Level 1: Target type
# Regression
reg_box = FancyBboxPatch((0.15, 0.65), 0.2, 0.08,
                        boxstyle="round,pad=0.01",
                        facecolor='lightcoral', edgecolor='black')
ax.add_patch(reg_box)
ax.text(0.25, 0.69, 'Regression\n(Continuous)', ha='center', va='center', fontsize=10)

# Classification
class_box = FancyBboxPatch((0.65, 0.65), 0.2, 0.08,
                          boxstyle="round,pad=0.01",
                          facecolor='lightblue', edgecolor='black')
ax.add_patch(class_box)
ax.text(0.75, 0.69, 'Classification\n(Categorical)', ha='center', va='center', fontsize=10)

# Level 2: Regression outcome
cart_reg_box = FancyBboxPatch((0.15, 0.45), 0.2, 0.08,
                             boxstyle="round,pad=0.01",
                             facecolor='#90EE90', edgecolor='black', linewidth=2)
ax.add_patch(cart_reg_box)
ax.text(0.25, 0.49, 'CART\n(Only Option)', ha='center', va='center', fontsize=11, weight='bold')

# Level 2: Classification - Missing values question
missing_box = FancyBboxPatch((0.65, 0.45), 0.2, 0.08,
                            boxstyle="round,pad=0.01",
                            facecolor='lightyellow', edgecolor='black')
ax.add_patch(missing_box)
ax.text(0.75, 0.49, 'High Missing\nValues?', ha='center', va='center', fontsize=10)

# Level 3: Missing values outcomes
# High missing values -> CART
cart_missing_box = FancyBboxPatch((0.45, 0.25), 0.15, 0.08,
                                 boxstyle="round,pad=0.01",
                                 facecolor='#90EE90', edgecolor='black', linewidth=2)
ax.add_patch(cart_missing_box)
ax.text(0.525, 0.29, 'CART\n(Robust)', ha='center', va='center', fontsize=10, weight='bold')

# Low missing values -> Feature cardinality question
cardinality_box = FancyBboxPatch((0.8, 0.25), 0.15, 0.08,
                                boxstyle="round,pad=0.01",
                                facecolor='lightcyan', edgecolor='black')
ax.add_patch(cardinality_box)
ax.text(0.875, 0.29, 'High\nCardinality?', ha='center', va='center', fontsize=9)

# Level 4: Final recommendations
# High cardinality -> C4.5
c45_box = FancyBboxPatch((0.7, 0.05), 0.15, 0.08,
                        boxstyle="round,pad=0.01",
                        facecolor='#87CEEB', edgecolor='black', linewidth=2)
ax.add_patch(c45_box)
ax.text(0.775, 0.09, 'C4.5\n(Gain Ratio)', ha='center', va='center', fontsize=10, weight='bold')

# Low cardinality -> ID3
id3_box = FancyBboxPatch((0.9, 0.05), 0.15, 0.08,
                        boxstyle="round,pad=0.01",
                        facecolor='#FFB6C1', edgecolor='black', linewidth=2)
ax.add_patch(id3_box)
ax.text(0.975, 0.09, 'ID3\n(Simple)', ha='center', va='center', fontsize=10, weight='bold')

# Draw connections with labels
connections = [
    # From root
    ((0.45, 0.8), (0.3, 0.73), 'Continuous', 'left'),
    ((0.55, 0.8), (0.7, 0.73), 'Categorical', 'right'),
    # From regression
    ((0.25, 0.61), (0.25, 0.53), '', ''),
    # From classification
    ((0.75, 0.61), (0.75, 0.53), '', ''),
    # From missing values
    ((0.7, 0.45), (0.575, 0.33), 'Yes ($>15\\%$)', 'left'),
((0.8, 0.45), (0.85, 0.33), 'No ($<15\\%$)', 'right'),
    # From cardinality
    ((0.825, 0.25), (0.8, 0.13), 'Yes', 'left'),
    ((0.925, 0.25), (0.95, 0.13), 'No', 'right'),
]

for start, end, label, side in connections:
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        offset = -0.02 if side == 'left' else 0.02
        ax.text(mid_x + offset, mid_y, label, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"))

ax.set_xlim(0, 1.2)
ax.set_ylim(-0.05, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_selection_flowchart.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create scenario summary table
fig, ax = plt.subplots(figsize=(32, 20))

# Add title ABOVE the table with proper positioning
ax.set_title('Algorithm Selection Summary', fontsize=24, weight='bold', pad=30, y=0.95)

# Prepare table data with better text handling and multi-line formatting
table_data = []
for scenario_id, scenario in scenarios.items():
    # Format description in exactly two lines for better readability
    desc = scenario['description']
    # Always split into exactly two lines for consistent formatting
    words = desc.split()
    line1 = ""
    line2 = ""
    target_length = len(desc) // 2  # Aim for roughly equal line lengths
    
    for word in words:
        if len(line1) < target_length:
            line1 += word + " "
        else:
            line2 += word + " "
    desc = line1.strip() + "\n" + line2.strip()
    
    # Format reasoning in exactly two lines for better readability
    reason = scenario['reasoning'][0]
    # Always split into exactly two lines for consistent formatting
    words = reason.split()
    line1 = ""
    line2 = ""
    target_length = len(reason) // 2  # Aim for roughly equal line lengths
    
    for word in words:
        if len(line1) < target_length:
            line1 += word + " "
        else:
            line2 += word + " "
    reason = line1.strip() + "\n" + line2.strip()
    
    row = [
        f"Scenario {scenario_id}",
        desc,
        scenario['best_algorithm'],
        reason
    ]
    table_data.append(row)

columns = ['Scenario', 'Description', 'Best Algorithm', 'Primary Reason']

# Create table with better spacing and bigger cells
table = ax.table(cellText=table_data, colLabels=columns, cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(36)  # Even bigger font for maximum readability
table.scale(1.5, 18.0)  # Half row width (1.5), maximum cell height (18.0)

# Style the table
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code by algorithm
alg_colors = {'ID3': '#FFE6E6', 'C4.5': '#E6F3FF', 'CART': '#E6FFE6'}
for i in range(1, len(table_data) + 1):
    alg = table_data[i-1][2]
    table[(i, 2)].set_facecolor(alg_colors[alg])
    table[(i, 2)].set_text_props(weight='bold')

ax.axis('off')

plt.savefig(os.path.join(save_dir, 'scenario_summary_table.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualization files saved to: {save_dir}")
print("Files created:")
print("- algorithm_capability_matrix.png")
print("- scenario_recommendations.png")
print("- scenario_1_analysis.png")
print("- scenario_2_analysis.png")
print("- scenario_3_analysis.png")
print("- scenario_4_analysis.png")
print("- scenario_5_analysis.png")
print("- algorithm_selection_flowchart.png")
print("- scenario_summary_table.png")
