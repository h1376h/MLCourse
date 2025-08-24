import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch
from collections import Counter
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

print("=" * 80)
print("QUESTION 14: ID3 ALGORITHM APPLICATION")
print("=" * 80)

# Create the tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Tennis Dataset:")
print(df.to_string(index=False))

# Function to calculate entropy
def entropy(labels):
    """Calculate entropy of a set of labels"""
    if len(labels) == 0:
        return 0
    
    counts = Counter(labels)
    total = len(labels)
    
    entropy_val = 0
    for count in counts.values():
        if count > 0:
            probability = count / total
            entropy_val -= probability * math.log2(probability)
    
    return entropy_val

# Function to calculate information gain
def information_gain(data, feature, target):
    """Calculate information gain for a feature"""
    # Calculate entropy of the entire dataset
    total_entropy = entropy(data[target])
    
    # Calculate weighted entropy after splitting
    feature_values = data[feature].unique()
    weighted_entropy = 0
    total_samples = len(data)
    
    for value in feature_values:
        subset = data[data[feature] == value]
        subset_size = len(subset)
        subset_entropy = entropy(subset[target])
        weighted_entropy += (subset_size / total_samples) * subset_entropy
    
    return total_entropy - weighted_entropy

# 1. Calculate entropy of the entire dataset
print("\n1. Entropy of the Entire Dataset:")
print("-" * 40)

target_values = df['Play Tennis'].tolist()
total_entropy = entropy(target_values)

# Count the classes
class_counts = Counter(target_values)
total_samples = len(target_values)

print(f"Target variable distribution:")
for cls, count in class_counts.items():
    probability = count / total_samples
    print(f"  {cls}: {count}/{total_samples} = {probability:.3f}")

print(f"\nEntropy calculation:")
print(f"H(S) = -$\\sum$ p(c) $\\times$ $\\log_2$(p(c))")

entropy_terms = []
for cls, count in class_counts.items():
    probability = count / total_samples
    if probability > 0:
        term = probability * math.log2(probability)
        entropy_terms.append(f"{probability:.3f} $\\times$ $\\log_2$({probability:.3f}) = {term:.4f}")
        print(f"  p({cls}) $\\times$ $\\log_2$({cls}) = {probability:.3f} $\\times$ {math.log2(probability):.3f} = {term:.4f}")

print(f"\nH(S) = -({' + '.join([f'({term})' for term in entropy_terms])})")
print(f"H(S) = {total_entropy:.4f}")

# 2. Calculate information gain for Outlook feature
print("\n2. Information Gain for Outlook Feature:")
print("-" * 45)

outlook_gain = information_gain(df, 'Outlook', 'Play Tennis')

print("Outlook feature values and their subsets:")
outlook_values = df['Outlook'].unique()

outlook_details = {}
for value in outlook_values:
    subset = df[df['Outlook'] == value]
    subset_targets = subset['Play Tennis'].tolist()
    subset_counts = Counter(subset_targets)
    subset_entropy = entropy(subset_targets)
    subset_size = len(subset)
    
    outlook_details[value] = {
        'subset': subset,
        'counts': subset_counts,
        'entropy': subset_entropy,
        'size': subset_size,
        'weight': subset_size / total_samples
    }
    
    print(f"\n  {value}: {subset_targets}")
    print(f"    Class distribution: {dict(subset_counts)}")
    print(f"    Size: {subset_size}, Weight: {subset_size}/{total_samples} = {subset_size/total_samples:.3f}")
    print(f"    Entropy: {subset_entropy:.4f}")

# Calculate weighted entropy for Outlook
weighted_entropy_outlook = sum(details['weight'] * details['entropy'] for details in outlook_details.values())
print(f"\nWeighted entropy after splitting on Outlook:")
terms = [f"{details['weight']:.3f} × {details['entropy']:.4f}" for details in outlook_details.values()]
print(f"H(S|Outlook) = {' + '.join(terms)} = {weighted_entropy_outlook:.4f}")

print(f"\nInformation Gain for Outlook:")
print(f"IG(S, Outlook) = H(S) - H(S|Outlook)")
print(f"IG(S, Outlook) = {total_entropy:.4f} - {weighted_entropy_outlook:.4f} = {outlook_gain:.4f}")

# 3. Calculate information gain for Wind feature
print("\n3. Information Gain for Wind Feature:")
print("-" * 40)

wind_gain = information_gain(df, 'Wind', 'Play Tennis')

print("Wind feature values and their subsets:")
wind_values = df['Wind'].unique()

wind_details = {}
for value in wind_values:
    subset = df[df['Wind'] == value]
    subset_targets = subset['Play Tennis'].tolist()
    subset_counts = Counter(subset_targets)
    subset_entropy = entropy(subset_targets)
    subset_size = len(subset)
    
    wind_details[value] = {
        'subset': subset,
        'counts': subset_counts,
        'entropy': subset_entropy,
        'size': subset_size,
        'weight': subset_size / total_samples
    }
    
    print(f"\n  {value}: {subset_targets}")
    print(f"    Class distribution: {dict(subset_counts)}")
    print(f"    Size: {subset_size}, Weight: {subset_size}/{total_samples} = {subset_size/total_samples:.3f}")
    print(f"    Entropy: {subset_entropy:.4f}")

# Calculate weighted entropy for Wind
weighted_entropy_wind = sum(details['weight'] * details['entropy'] for details in wind_details.values())
print(f"\nWeighted entropy after splitting on Wind:")
terms = [f"{details['weight']:.3f} × {details['entropy']:.4f}" for details in wind_details.values()]
print(f"H(S|Wind) = {' + '.join(terms)} = {weighted_entropy_wind:.4f}")

print(f"\nInformation Gain for Wind:")
print(f"IG(S, Wind) = H(S) - H(S|Wind)")
print(f"IG(S, Wind) = {total_entropy:.4f} - {weighted_entropy_wind:.4f} = {wind_gain:.4f}")

# Calculate information gain for all features
print("\n4. Information Gain for All Features:")
print("-" * 42)

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
gains = {}

for feature in features:
    gain = information_gain(df, feature, 'Play Tennis')
    gains[feature] = gain
    print(f"IG(S, {feature:11}) = {gain:.4f}")

# Find the best feature
best_feature = max(gains, key=gains.get)
print(f"\nBest feature for root node: {best_feature} (IG = {gains[best_feature]:.4f})")

# Create separate focused visualizations

# Visualization 1: Dataset overview
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table_data = df.values.tolist()
table = ax.table(cellText=table_data, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.5, 2.0)

# Color code the target column
for i in range(len(df)):
    if df.iloc[i]['Play Tennis'] == 'Yes':
        table[(i+1, len(df.columns)-1)].set_facecolor('#90EE90')  # Light green
    else:
        table[(i+1, len(df.columns)-1)].set_facecolor('#FFB6C1')  # Light pink

# Header styling
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax.set_title('Tennis Dataset', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tennis_dataset_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Entropy calculation
fig, ax = plt.subplots(figsize=(8, 8))
class_names = list(class_counts.keys())
class_values = list(class_counts.values())
colors = ['lightcoral', 'lightgreen']

wedges, texts, autotexts = ax.pie(class_values, labels=class_names, colors=colors, 
                                  autopct='%1.1f%%', startangle=90)
ax.set_title(f'Target Distribution\nEntropy = {total_entropy:.4f}', fontsize=12, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'target_distribution_entropy.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Information gains comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(features, [gains[f] for f in features], 
               color=['red' if f == best_feature else 'skyblue' for f in features],
               alpha=0.7, edgecolor='black', linewidth=2)

ax.set_ylabel('Information Gain')
ax.set_title('Information Gain by Feature')
ax.set_ylim(0, max(gains.values()) * 1.1)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, feature in zip(bars, features):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{gains[feature]:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Highlight best feature
best_idx = features.index(best_feature)
ax.text(best_idx, gains[best_feature] + 0.05, 'Best!', 
         ha='center', va='bottom', fontsize=12, weight='bold', color='red')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Detailed entropy breakdown for best feature
fig, ax = plt.subplots(figsize=(10, 6))
outlook_values_list = list(outlook_details.keys())
entropies = [outlook_details[val]['entropy'] for val in outlook_values_list]
weights = [outlook_details[val]['weight'] for val in outlook_values_list]

x_pos = np.arange(len(outlook_values_list))
bars4 = ax.bar(x_pos, entropies, alpha=0.7, color=['orange', 'purple', 'brown'],
                edgecolor='black', linewidth=2)

# Add weight labels
for i, (bar, weight) in enumerate(zip(bars4, weights)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'Entropy: {height:.3f}\nWeight: {weight:.3f}', 
             ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Outlook Values')
ax.set_ylabel('Entropy')
ax.set_title(f'Entropy Breakdown for {best_feature} Feature')
ax.set_xticks(x_pos)
ax.set_xticklabels(outlook_values_list)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_breakdown_outlook.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create decision tree visualization
fig, ax = plt.subplots(figsize=(14, 10))

# Draw the ID3 decision tree for the first split
ax.text(0.5, 0.95, 'ID3 Decision Tree - First Split', ha='center', fontsize=16, weight='bold')

# Root node
root_box = FancyBboxPatch((0.4, 0.8), 0.2, 0.08,
                         boxstyle="round,pad=0.01",
                         facecolor='lightblue', edgecolor='black', linewidth=2)
ax.add_patch(root_box)
ax.text(0.5, 0.84, f'{best_feature}\nIG = {gains[best_feature]:.4f}', 
        ha='center', va='center', fontsize=11, weight='bold')

# Child nodes for Outlook values
outlook_positions = [(0.15, 0.6), (0.5, 0.6), (0.85, 0.6)]
outlook_colors = ['lightcoral', 'lightgreen', 'lightyellow']

for i, (value, pos, color) in enumerate(zip(outlook_values, outlook_positions, outlook_colors)):
    details = outlook_details[value]
    
    # Node box
    node_box = FancyBboxPatch((pos[0]-0.08, pos[1]), 0.16, 0.12,
                             boxstyle="round,pad=0.01",
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(node_box)
    
    # Node text
    node_text = f"{value}\n{dict(details['counts'])}\nEntropy: {details['entropy']:.3f}"
    ax.text(pos[0], pos[1]+0.06, node_text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Edge from root to child
    ax.plot([0.5, pos[0]], [0.8, pos[1]+0.12], 'k-', linewidth=2)
    
    # Edge label
    mid_x = (0.5 + pos[0]) / 2
    mid_y = (0.8 + pos[1] + 0.12) / 2
    ax.text(mid_x, mid_y, value, ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black"))

# Add algorithm steps explanation
steps_text = """ID3 Algorithm Steps:
1. Calculate entropy of dataset: H(S) = 0.9183
2. For each feature, calculate information gain:
   • Outlook: 0.6935 (Best!)
   • Temperature: 0.2467
   • Humidity: 0.0384
   • Wind: 0.0481
3. Select feature with highest information gain: Outlook
4. Create branches for each value of selected feature
5. Repeat for each non-pure subtree"""

ax.text(0.05, 0.35, steps_text, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black"))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'id3_decision_tree.png'), dpi=300, bbox_inches='tight')

# Create detailed calculation visualization
fig, ax = plt.subplots(figsize=(14, 10))

ax.text(0.5, 0.95, 'Detailed Information Gain Calculations', ha='center', fontsize=16, weight='bold')

# Show entropy calculation step by step
y_pos = 0.85
ax.text(0.05, y_pos, '1. Total Entropy Calculation:', fontsize=14, weight='bold')
y_pos -= 0.05
ax.text(0.1, y_pos, f'Classes: Yes={class_counts["Yes"]}, No={class_counts["No"]}', fontsize=12)
y_pos -= 0.04
ax.text(0.1, y_pos, f'H(S) = -p(Yes)$\\times$$\\log_2$(p(Yes)) - p(No)$\\times$$\\log_2$(p(No))', fontsize=12)
y_pos -= 0.04
p_yes = class_counts["Yes"] / total_samples
p_no = class_counts["No"] / total_samples
ax.text(0.1, y_pos, f'H(S) = -({p_yes:.3f})$\\times$$\\log_2$({p_yes:.3f}) - ({p_no:.3f})$\\times$$\\log_2$({p_no:.3f})', fontsize=12)
y_pos -= 0.04
ax.text(0.1, y_pos, f'H(S) = {total_entropy:.4f}', fontsize=12, weight='bold', color='blue')

y_pos -= 0.08
ax.text(0.05, y_pos, '2. Information Gain for Outlook:', fontsize=14, weight='bold')
y_pos -= 0.05

for value in outlook_values:
    details = outlook_details[value]
    ax.text(0.1, y_pos, f'{value}: {list(details["subset"]["Play Tennis"])} → {dict(details["counts"])}', fontsize=10)
    y_pos -= 0.03
    ax.text(0.15, y_pos, f'Entropy = {details["entropy"]:.4f}, Weight = {details["weight"]:.3f}', fontsize=10)
    y_pos -= 0.04

ax.text(0.1, y_pos, f'Weighted Entropy = {weighted_entropy_outlook:.4f}', fontsize=12, color='blue')
y_pos -= 0.04
ax.text(0.1, y_pos, f'IG(Outlook) = {total_entropy:.4f} - {weighted_entropy_outlook:.4f} = {outlook_gain:.4f}', 
        fontsize=12, weight='bold', color='red')

y_pos -= 0.08
ax.text(0.05, y_pos, '3. Information Gain for Wind:', fontsize=14, weight='bold')
y_pos -= 0.05

for value in wind_values:
    details = wind_details[value]
    ax.text(0.1, y_pos, f'{value}: {list(details["subset"]["Play Tennis"])} → {dict(details["counts"])}', fontsize=10)
    y_pos -= 0.03
    ax.text(0.15, y_pos, f'Entropy = {details["entropy"]:.4f}, Weight = {details["weight"]:.3f}', fontsize=10)
    y_pos -= 0.04

ax.text(0.1, y_pos, f'Weighted Entropy = {weighted_entropy_wind:.4f}', fontsize=12, color='blue')
y_pos -= 0.04
ax.text(0.1, y_pos, f'IG(Wind) = {total_entropy:.4f} - {weighted_entropy_wind:.4f} = {wind_gain:.4f}', 
        fontsize=12, weight='bold', color='red')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'detailed_calculations.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualization files saved to: {save_dir}")
print("Files created:")
print("- tennis_dataset_overview.png")
print("- target_distribution_entropy.png")
print("- information_gain_comparison.png")
print("- entropy_breakdown_outlook.png")
print("- id3_decision_tree.png")
print("- detailed_calculations.png")
