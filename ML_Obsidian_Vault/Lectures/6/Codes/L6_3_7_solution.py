import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 7: ALGORITHM SELECTION STRATEGY")
print("=" * 80)

# Define algorithm characteristics
algorithms = {
    'ID3': {
        'handles_categorical': 'Excellent',
        'handles_continuous': 'No',
        'handles_missing': 'No',
        'bias_resistance': 'Poor',
        'interpretability': 'Excellent',
        'complexity': 'Low',
        'overfitting_risk': 'High',
        'best_for': 'Small educational datasets'
    },
    'C4.5': {
        'handles_categorical': 'Excellent',
        'handles_continuous': 'Good',
        'handles_missing': 'Good',
        'bias_resistance': 'Good',
        'interpretability': 'Good',
        'complexity': 'Medium',
        'overfitting_risk': 'Medium',
        'best_for': 'Mixed datasets with missing values'
    },
    'CART': {
        'handles_categorical': 'Good',
        'handles_continuous': 'Excellent',
        'handles_missing': 'Good',
        'bias_resistance': 'Excellent',
        'interpretability': 'Good',
        'complexity': 'High',
        'overfitting_risk': 'Low',
        'best_for': 'Regression and high-cardinality features'
    }
}

# Define scenarios
scenarios = [
    {
        'name': 'Small Educational Dataset',
        'description': '50 samples, 4 categorical features (2-3 values each), no missing data, interpretability crucial',
        'key_factors': ['Small size', 'Categorical only', 'No missing data', 'Interpretability needed'],
        'recommended': 'ID3',
        'reasoning': 'Perfect fit for ID3 as it handles categorical data excellently and provides maximum interpretability for educational purposes.'
    },
    {
        'name': 'Mixed-Type Dataset',
        'description': '1000 samples, 6 categorical + 4 continuous features, 15% missing values',
        'key_factors': ['Mixed data types', 'Missing values', 'Medium size'],
        'recommended': 'C4.5',
        'reasoning': 'C4.5 handles both categorical and continuous features well, and has built-in missing value handling capabilities.'
    },
    {
        'name': 'High-Cardinality Problem',
        'description': '500 samples, features include customer ID, zip code, product category with 50+ unique values',
        'key_factors': ['High-cardinality features', 'Bias risk', 'Medium size'],
        'recommended': 'CART',
        'reasoning': 'CART\'s binary splitting strategy avoids the bias toward high-cardinality features that affects ID3 and C4.5.'
    },
    {
        'name': 'Regression Task',
        'description': 'Predicting house prices using categorical (neighborhood, style) and continuous (size, age) features',
        'key_factors': ['Regression problem', 'Mixed data types', 'Continuous target'],
        'recommended': 'CART',
        'reasoning': 'Only CART can handle regression problems natively; ID3 and C4.5 are limited to classification tasks.'
    },
    {
        'name': 'Noisy Environment',
        'description': 'Dataset with many irrelevant features and measurement errors',
        'key_factors': ['Noise resistance', 'Feature selection', 'Robustness needed'],
        'recommended': 'CART',
        'reasoning': 'CART\'s Gini criterion and pruning capabilities make it more robust to noise and irrelevant features.'
    }
]

print("\nDETAILED SCENARIO ANALYSIS")
print("=" * 50)

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{i}. {scenario['name'].upper()}")
    print("-" * 40)
    print(f"Description: {scenario['description']}")
    print(f"Key factors: {', '.join(scenario['key_factors'])}")
    print(f"Recommended: {scenario['recommended']}")
    print(f"Reasoning: {scenario['reasoning']}")

# Create algorithm comparison matrix
comparison_matrix = pd.DataFrame(algorithms).T

# Create scoring matrix for visualization
score_mapping = {
    'Excellent': 5, 'Good': 4, 'Medium': 3, 'Poor': 2, 'No': 1,
    'Low': 2, 'High': 4
}

# Convert to numeric scores for heatmap
numeric_matrix = comparison_matrix.copy()
for col in numeric_matrix.columns:
    if col != 'best_for':
        numeric_matrix[col] = numeric_matrix[col].map(score_mapping)

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# Plot 1: Algorithm comparison heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(numeric_matrix.drop('best_for', axis=1).astype(float), 
            annot=True, cmap='RdYlGn', cbar_kws={'label': 'Capability Score'},
            xticklabels=True, yticklabels=True, ax=ax1)
ax1.set_title('Algorithm Capability Comparison')
ax1.set_xlabel('Characteristics')
ax1.set_ylabel('Algorithms')

# Plot 2: Scenario recommendations
ax2 = plt.subplot(2, 3, 2)
scenario_names = [s['name'] for s in scenarios]
recommendations = [s['recommended'] for s in scenarios]
colors = {'ID3': 'skyblue', 'C4.5': 'lightgreen', 'CART': 'lightcoral'}
bar_colors = [colors[rec] for rec in recommendations]

bars = ax2.barh(scenario_names, [1]*len(scenario_names), color=bar_colors, alpha=0.7)
ax2.set_xlabel('Recommendation')
ax2.set_title('Algorithm Selection by Scenario')
ax2.set_xlim(0, 1)

# Add algorithm labels on bars
for i, (bar, rec) in enumerate(zip(bars, recommendations)):
    ax2.text(0.5, i, rec, ha='center', va='center', fontweight='bold')

# Create legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=alg) 
                  for alg, color in colors.items()]
ax2.legend(handles=legend_elements, loc='upper right')

# Plot 3: Decision flowchart
ax3 = plt.subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)

# Decision tree structure
decision_boxes = [
    {'pos': (5, 9), 'text': 'Regression\nTask?', 'color': 'lightblue'},
    {'pos': (8, 7), 'text': 'CART', 'color': 'lightcoral'},
    {'pos': (2, 7), 'text': 'Missing\nValues?', 'color': 'lightblue'},
    {'pos': (3.5, 5), 'text': 'High\nCardinality?', 'color': 'lightblue'},
    {'pos': (0.5, 5), 'text': 'Educational\nPurpose?', 'color': 'lightblue'},
    {'pos': (5, 3), 'text': 'CART', 'color': 'lightcoral'},
    {'pos': (2, 3), 'text': 'C4.5', 'color': 'lightgreen'},
    {'pos': (1, 1), 'text': 'ID3', 'color': 'skyblue'},
    {'pos': (0, 1), 'text': 'C4.5', 'color': 'lightgreen'}
]

# Draw decision boxes
for box in decision_boxes:
    rect = plt.Rectangle((box['pos'][0]-0.4, box['pos'][1]-0.3), 0.8, 0.6, 
                        facecolor=box['color'], edgecolor='black', alpha=0.7)
    ax3.add_patch(rect)
    ax3.text(box['pos'][0], box['pos'][1], box['text'], ha='center', va='center', 
            fontsize=8, fontweight='bold')

# Draw arrows
arrows = [
    ((5, 8.7), (8, 7.3), 'Yes'),
    ((5, 8.7), (2, 7.3), 'No'),
    ((2, 6.7), (3.5, 5.3), 'No'),
    ((2, 6.7), (0.5, 5.3), 'Yes'),
    ((3.5, 4.7), (5, 3.3), 'Yes'),
    ((3.5, 4.7), (2, 3.3), 'No'),
    ((0.5, 4.7), (1, 1.3), 'Yes'),
    ((0.5, 4.7), (0, 1.3), 'No')
]

for start, end, label in arrows:
    ax3.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    # Add labels
    mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    ax3.text(mid_x + 0.2, mid_y + 0.1, label, fontsize=8, color='red', fontweight='bold')

ax3.set_title('Decision Tree Algorithm Selection Flowchart')
ax3.axis('off')

# Plot 4: Feature type compatibility
ax4 = plt.subplot(2, 3, 4)
feature_types = ['Categorical', 'Continuous', 'Missing Values', 'High Cardinality']
id3_scores = [5, 1, 1, 2]
c45_scores = [5, 4, 4, 3]
cart_scores = [4, 5, 4, 5]

x = np.arange(len(feature_types))
width = 0.25

ax4.bar(x - width, id3_scores, width, label='ID3', color='skyblue', alpha=0.7)
ax4.bar(x, c45_scores, width, label='C4.5', color='lightgreen', alpha=0.7)
ax4.bar(x + width, cart_scores, width, label='CART', color='lightcoral', alpha=0.7)

ax4.set_xlabel('Feature Types')
ax4.set_ylabel('Capability Score')
ax4.set_title('Algorithm Capabilities by Feature Type')
ax4.set_xticks(x)
ax4.set_xticklabels(feature_types, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Scenario complexity vs algorithm suitability
ax5 = plt.subplot(2, 3, 5)
complexity_scores = [1, 3, 4, 5, 4]  # Based on dataset complexity
scenario_short = ['Educational', 'Mixed-Type', 'High-Card', 'Regression', 'Noisy']

# Create scatter plot
for i, (complexity, scenario, rec) in enumerate(zip(complexity_scores, scenario_short, recommendations)):
    color = colors[rec]
    ax5.scatter(complexity, i, s=200, c=color, alpha=0.7, edgecolors='black')
    ax5.text(complexity + 0.1, i, rec, va='center', fontweight='bold')

ax5.set_xlabel('Problem Complexity')
ax5.set_ylabel('Scenarios')
ax5.set_title('Algorithm Selection vs Problem Complexity')
ax5.set_yticks(range(len(scenario_short)))
ax5.set_yticklabels(scenario_short)
ax5.grid(True, alpha=0.3)

# Plot 6: Summary table
ax6 = plt.subplot(2, 3, 6)
summary_data = {
    'Scenario': [s['name'] for s in scenarios],
    'Best Algorithm': [s['recommended'] for s in scenarios],
    'Key Reason': [s['reasoning'][:50] + '...' if len(s['reasoning']) > 50 else s['reasoning'] for s in scenarios]
}

df_summary = pd.DataFrame(summary_data)
ax6.axis('tight')
ax6.axis('off')

table = ax6.table(cellText=df_summary.values,
                 colLabels=df_summary.columns,
                 cellLoc='left',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Color code based on algorithm
for i, rec in enumerate(recommendations):
    table[(i+1, 1)].set_facecolor(colors[rec])
    table[(i+1, 1)].set_text_props(weight='bold')

# Header styling
for j in range(len(df_summary.columns)):
    table[(0, j)].set_facecolor('#2E8B57')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('Algorithm Selection Summary', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_selection_strategy.png'), dpi=300, bbox_inches='tight')

# Create a detailed comparison table figure
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Extended comparison table
extended_data = {
    'Characteristic': list(algorithms['ID3'].keys()),
    'ID3': list(algorithms['ID3'].values()),
    'C4.5': list(algorithms['C4.5'].values()),
    'CART': list(algorithms['CART'].values())
}

df_extended = pd.DataFrame(extended_data)
table2 = ax.table(cellText=df_extended.values,
                 colLabels=df_extended.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1, 2.5)

# Color coding
colors_map = {
    'Excellent': '#4CAF50',
    'Good': '#8BC34A', 
    'Medium': '#FFC107',
    'Poor': '#FF9800',
    'No': '#F44336',
    'Low': '#FFEB3B',
    'High': '#FF5722'
}

# Apply colors to cells
for i in range(1, len(df_extended) + 1):
    for j in range(1, len(df_extended.columns)):
        cell_value = df_extended.iloc[i-1, j]
        if cell_value in colors_map:
            table2[(i, j)].set_facecolor(colors_map[cell_value])
            if cell_value in ['Excellent', 'Good']:
                table2[(i, j)].set_text_props(color='white', weight='bold')

# Header styling
for j in range(len(df_extended.columns)):
    table2[(0, j)].set_facecolor('#1976D2')
    table2[(0, j)].set_text_props(weight='bold', color='white')

ax.set_title('Detailed Algorithm Comparison Matrix', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(save_dir, 'detailed_comparison_matrix.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print(f"ALGORITHM SELECTION DECISION RULES")
print("="*80)
print("1. REGRESSION TASK → Always choose CART")
print("2. MISSING VALUES → C4.5 or CART (avoid ID3)")
print("3. HIGH CARDINALITY → CART (best bias resistance)")
print("4. SMALL EDUCATIONAL DATASET → ID3 (maximum interpretability)")
print("5. MIXED DATA TYPES → C4.5 or CART")
print("6. NOISE/ROBUSTNESS NEEDED → CART")
print("7. PURE CATEGORICAL DATA → ID3 or C4.5")
print("8. CONTINUOUS DATA DOMINANT → CART")

print(f"\nImages saved to: {save_dir}")
