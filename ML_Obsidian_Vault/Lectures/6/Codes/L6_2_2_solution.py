import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Decision Trees: Information Gain Calculation")
print("=" * 50)

# Define the dataset based on the problem
data = {
    'Color': ['Red', 'Blue', 'Green'],
    'Class_A': [20, 25, 15],
    'Class_B': [10, 15, 15],
    'Total': [30, 40, 30]
}

df = pd.DataFrame(data)
print("\nDataset:")
print(df.to_string(index=False))

# Calculate total dataset size
total_samples = df['Total'].sum()
total_class_A = df['Class_A'].sum()
total_class_B = df['Class_B'].sum()

print(f"\nTotal samples: {total_samples}")
print(f"Total Class A: {total_class_A}")
print(f"Total Class B: {total_class_B}")

# Step 1: Calculate entropy function
def entropy(class_counts):
    """Calculate entropy given class counts"""
    total = sum(class_counts)
    if total == 0:
        return 0
    
    entropy_val = 0
    for count in class_counts:
        if count > 0:
            p = count / total
            entropy_val -= p * np.log2(p)
    
    return entropy_val

# Step 1: Calculate entropy for each color value
print("\n" + "="*50)
print("STEP 1: Calculate entropy for each color value")
print("="*50)

color_entropies = {}
for i, row in df.iterrows():
    color = row['Color']
    class_a = row['Class_A']
    class_b = row['Class_B']
    total = row['Total']
    
    # Calculate entropy for this color
    color_entropy = entropy([class_a, class_b])
    color_entropies[color] = color_entropy
    
    # Show detailed calculation
    if class_a > 0 and class_b > 0:
        p_a = class_a / total
        p_b = class_b / total
        
        print(f"\nColor: {color}")
        print(f"  Class A: {class_a}, Class B: {class_b}, Total: {total}")
        print(f"  P(Class A) = {class_a}/{total} = {p_a:.4f}")
        print(f"  P(Class B) = {class_b}/{total} = {p_b:.4f}")
        print(f"  Entropy({color}) = -P(A)log₂(P(A)) - P(B)log₂(P(B))")
        print(f"  Entropy({color}) = -{p_a:.4f} × log₂({p_a:.4f}) - {p_b:.4f} × log₂({p_b:.4f})")
        print(f"  Entropy({color}) = -{p_a:.4f} × {np.log2(p_a):.4f} - {p_b:.4f} × {np.log2(p_b):.4f}")
        print(f"  Entropy({color}) = {-p_a * np.log2(p_a):.4f} + {-p_b * np.log2(p_b):.4f}")
        print(f"  Entropy({color}) = {color_entropy:.4f}")
    else:
        print(f"\nColor: {color}")
        print(f"  Entropy({color}) = {color_entropy:.4f} (pure subset)")

# Step 2: Calculate weighted average entropy after splitting
print("\n" + "="*50)
print("STEP 2: Calculate weighted average entropy after splitting")
print("="*50)

weighted_entropy = 0
print("\nWeighted Average Entropy Calculation:")
print("E(S|Color) = Σ (|Sv|/|S|) × E(Sv)")

for i, row in df.iterrows():
    color = row['Color']
    total = row['Total']
    weight = total / total_samples
    entropy_val = color_entropies[color]
    contribution = weight * entropy_val
    weighted_entropy += contribution
    
    print(f"\nFor {color}:")
    print(f"  Weight = |S_{color}|/|S| = {total}/{total_samples} = {weight:.4f}")
    print(f"  Entropy = {entropy_val:.4f}")
    print(f"  Contribution = {weight:.4f} × {entropy_val:.4f} = {contribution:.4f}")

print(f"\nWeighted Average Entropy = {weighted_entropy:.4f}")

# Step 3: Calculate information gain
print("\n" + "="*50)
print("STEP 3: Calculate information gain from this split")
print("="*50)

# Calculate original entropy (before split)
original_entropy = entropy([total_class_A, total_class_B])

print("Original Entropy Calculation:")
p_a_original = total_class_A / total_samples
p_b_original = total_class_B / total_samples

print(f"P(Class A) = {total_class_A}/{total_samples} = {p_a_original:.4f}")
print(f"P(Class B) = {total_class_B}/{total_samples} = {p_b_original:.4f}")
print(f"Original Entropy = -{p_a_original:.4f} × log₂({p_a_original:.4f}) - {p_b_original:.4f} × log₂({p_b_original:.4f})")
print(f"Original Entropy = {original_entropy:.4f}")

# Calculate information gain
information_gain = original_entropy - weighted_entropy

print(f"\nInformation Gain Calculation:")
print(f"IG(S, Color) = E(S) - E(S|Color)")
print(f"IG(S, Color) = {original_entropy:.4f} - {weighted_entropy:.4f}")
print(f"IG(S, Color) = {information_gain:.4f}")

# Step 4: Evaluate if this is a good split
print("\n" + "="*50)
print("STEP 4: Evaluation of the split quality")
print("="*50)

print(f"Information Gain: {information_gain:.4f}")
print(f"Original Entropy: {original_entropy:.4f}")
print(f"Reduction in entropy: {(information_gain/original_entropy)*100:.2f}%")

if information_gain > 0.1:
    quality = "Good"
elif information_gain > 0.05:
    quality = "Moderate"
else:
    quality = "Poor"

print(f"Split quality: {quality}")

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Visualization 1: Data Distribution
colors = ['red', 'blue', 'green']
x_pos = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, df['Class_A'], width, label='Class A', alpha=0.8, color='lightblue')
bars2 = ax1.bar(x_pos + width/2, df['Class_B'], width, label='Class B', alpha=0.8, color='lightcoral')

ax1.set_xlabel('Color')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Data Distribution by Color and Class')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Color'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Visualization 2: Entropy for each color
entropy_values = [color_entropies[color] for color in df['Color']]
bars = ax2.bar(df['Color'], entropy_values, color=['red', 'blue', 'green'], alpha=0.7)
ax2.set_xlabel('Color')
ax2.set_ylabel('Entropy')
ax2.set_title('Entropy for Each Color Subset')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.1)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Visualization 3: Information Gain Calculation
components = ['Original Entropy', 'Weighted Avg Entropy', 'Information Gain']
values = [original_entropy, weighted_entropy, information_gain]
colors_ig = ['lightblue', 'lightcoral', 'lightgreen']

bars = ax3.bar(components, values, color=colors_ig, alpha=0.8)
ax3.set_ylabel('Entropy/Information Gain')
ax3.set_title('Information Gain Components')
ax3.grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Visualization 4: Weighted Contributions
contributions = []
weights = []
colors_contrib = []
labels = []

for i, row in df.iterrows():
    color = row['Color']
    total = row['Total']
    weight = total / total_samples
    entropy_val = color_entropies[color]
    contribution = weight * entropy_val
    
    contributions.append(contribution)
    weights.append(weight)
    colors_contrib.append(['red', 'blue', 'green'][i])
    labels.append(f'{color}\n(w={weight:.3f})')

bars = ax4.bar(labels, contributions, color=colors_contrib, alpha=0.7)
ax4.set_ylabel('Weighted Entropy Contribution')
ax4.set_title('Weighted Entropy Contributions')
ax4.grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_gain_analysis.png'), dpi=300, bbox_inches='tight')

# Create a decision tree visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a simple tree structure visualization
# Root node
root_x, root_y = 0.5, 0.9
ax.add_patch(plt.Rectangle((root_x-0.1, root_y-0.05), 0.2, 0.1, 
                          fill=True, facecolor='lightblue', edgecolor='black'))
ax.text(root_x, root_y, f'Root\nEntropy = {original_entropy:.3f}\nSamples = {total_samples}', 
        ha='center', va='center', fontsize=10, weight='bold')

# Child nodes
positions = [(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)]
colors_tree = ['red', 'blue', 'green']

for i, (color, pos) in enumerate(zip(df['Color'], positions)):
    x, y = pos
    samples = df.iloc[i]['Total']
    class_a = df.iloc[i]['Class_A']
    class_b = df.iloc[i]['Class_B']
    entropy_val = color_entropies[color]
    
    # Draw node
    ax.add_patch(plt.Rectangle((x-0.08, y-0.08), 0.16, 0.16, 
                              fill=True, facecolor=colors_tree[i], alpha=0.3, edgecolor='black'))
    
    # Add text
    ax.text(x, y, f'{color}\nEntropy = {entropy_val:.3f}\nSamples = {samples}\nA:{class_a}, B:{class_b}', 
            ha='center', va='center', fontsize=9)
    
    # Draw edge from root
    ax.plot([root_x, x], [root_y-0.05, y+0.08], 'k-', linewidth=2)
    
    # Add edge label
    mid_x, mid_y = (root_x + x) / 2, (root_y + y) / 2
    ax.text(mid_x, mid_y, f'{color}', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='black'),
            fontsize=8)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'Decision Tree Split by Color\nInformation Gain = {information_gain:.4f}', 
             fontsize=14, weight='bold', pad=20)

plt.savefig(os.path.join(save_dir, 'decision_tree_visualization.png'), dpi=300, bbox_inches='tight')

# Create entropy comparison visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Before and after comparison
categories = ['Before Split', 'After Split (Weighted Avg)']
entropy_values = [original_entropy, weighted_entropy]
colors_comp = ['lightcoral', 'lightblue']

bars = ax.bar(categories, entropy_values, color=colors_comp, alpha=0.8, width=0.6)
ax.set_ylabel('Entropy')
ax.set_title('Entropy Comparison: Before vs After Split')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(entropy_values) * 1.2)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, weight='bold')

# Add information gain arrow and text
ax.annotate('', xy=(1, weighted_entropy), xytext=(0, original_entropy),
            arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax.text(0.5, (original_entropy + weighted_entropy) / 2, 
        f'Information Gain\n{information_gain:.4f}', 
        ha='center', va='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', edgecolor='green'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'entropy_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print("\nSUMMARY:")
print("="*50)
print(f"1. Entropy for Red: {color_entropies['Red']:.4f}")
print(f"2. Entropy for Blue: {color_entropies['Blue']:.4f}")
print(f"3. Entropy for Green: {color_entropies['Green']:.4f}")
print(f"4. Weighted Average Entropy: {weighted_entropy:.4f}")
print(f"5. Information Gain: {information_gain:.4f}")
print(f"6. Split Quality: {quality}")
print("="*50)
