import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import itertools
import networkx as nx

print("\n=== INDEPENDENCE EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Independence in Coin Flips
print("\nExample 1: Independence in Coin Flips")
print("Consider flipping a fair coin three times in succession.")

# Define possible outcomes
outcomes = ['H', 'T']
all_combinations = list(itertools.product(outcomes, repeat=3))
print("\nAll possible outcomes:")
for combo in all_combinations:
    print(f"  {''.join(combo)}")

# Calculate probabilities
p_heads = 0.5
p_tails = 0.5
p_all_tails = p_tails ** 3
p_at_least_one_heads = 1 - p_all_tails

print("\nStep-by-step calculation:")
print(f"P(T) = {p_tails}")
print(f"P(all tails) = P(T) × P(T) × P(T) = {p_tails} × {p_tails} × {p_tails} = {p_all_tails}")
print(f"P(at least one heads) = 1 - P(all tails) = 1 - {p_all_tails} = {p_at_least_one_heads}")

# Create first visualization
plt.figure(figsize=(10, 6))
x = np.arange(0, 4)
y = [p_tails**i for i in range(4)]
plt.plot(x, y, 'bo-', label='Probability of all tails')
plt.axhline(y=p_at_least_one_heads, color='r', linestyle='--', label='Probability of at least one heads')
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Flips', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability of All Tails vs. At Least One Heads', fontsize=14)
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_flip_independence.png'), dpi=100, bbox_inches='tight')
plt.close()

# Create probability tree visualization
plt.figure(figsize=(12, 8))
G = nx.DiGraph()

# Add nodes and edges for the probability tree
nodes = ['Start']
edges = []
labels = {}
edge_labels = {}

# First level
for i, outcome in enumerate(outcomes):
    node = f'{outcome}1'
    nodes.append(node)
    edges.append(('Start', node))
    labels[node] = outcome
    edge_labels[('Start', node)] = f'P({outcome}) = {p_heads if outcome == "H" else p_tails}'

# Second level
for parent in nodes[1:]:
    for outcome in outcomes:
        node = f'{outcome}2_{parent}'
        nodes.append(node)
        edges.append((parent, node))
        labels[node] = outcome
        edge_labels[(parent, node)] = f'P({outcome}) = {p_heads if outcome == "H" else p_tails}'

# Third level
for parent in nodes[2:]:
    for outcome in outcomes:
        node = f'{outcome}3_{parent}'
        nodes.append(node)
        edges.append((parent, node))
        labels[node] = outcome
        edge_labels[(parent, node)] = f'P({outcome}) = {p_heads if outcome == "H" else p_tails}'

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Calculate positions for the tree
pos = {}
level_height = 1.0
level_spacing = 0.5
node_spacing = 0.5

# Position the root
pos['Start'] = (0, 0)

# Position first level
for i, node in enumerate(nodes[1:3]):
    pos[node] = (-node_spacing/2 + i*node_spacing, -level_height)

# Position second level
for i, node in enumerate(nodes[3:7]):
    pos[node] = (-node_spacing*1.5 + (i%2)*node_spacing*2, -level_height*2)

# Position third level
for i, node in enumerate(nodes[7:]):
    pos[node] = (-node_spacing*3.5 + (i%2)*node_spacing*7, -level_height*3)

# Draw the tree
nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', 
        font_size=8, font_weight='bold', arrowsize=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

plt.title('Probability Tree for Three Coin Flips', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_flip_tree.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Independence in Card Drawing
print("\n\nExample 2: Independence in Card Drawing")
print("Two cards are drawn from a standard deck of 52 cards.")

# With replacement
p_ace = 4/52
p_two_aces_with_replacement = p_ace * p_ace

# Without replacement
p_second_ace_given_first = 3/51
p_two_aces_without_replacement = p_ace * p_second_ace_given_first

print("\nWith replacement:")
print(f"P(1st ace) = 4/52 = {p_ace:.4f}")
print(f"P(2nd ace) = 4/52 = {p_ace:.4f}")
print(f"P(two aces) = {p_ace:.4f} × {p_ace:.4f} = {p_two_aces_with_replacement:.4f}")

print("\nWithout replacement:")
print(f"P(1st ace) = 4/52 = {p_ace:.4f}")
print(f"P(2nd ace | 1st ace) = 3/51 = {p_second_ace_given_first:.4f}")
print(f"P(two aces) = {p_ace:.4f} × {p_second_ace_given_first:.4f} = {p_two_aces_without_replacement:.4f}")

# Create visualization
plt.figure(figsize=(10, 6))
scenarios = ['With Replacement', 'Without Replacement']
probabilities = [p_two_aces_with_replacement, p_two_aces_without_replacement]
colors = ['green', 'red']

plt.bar(scenarios, probabilities, color=colors, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.ylabel('Probability', fontsize=12)
plt.title('Probability of Drawing Two Aces', fontsize=14)
plt.ylim(0, 0.01)

# Add probability values on top of each bar
for i, prob in enumerate(probabilities):
    plt.text(i, prob + 0.0005, f'{prob:.4f}', ha='center', fontsize=10)

# Add explanation
plt.annotate('Independent draws', 
            xy=(0, 0.008),
            xytext=(0, 0.008),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.annotate('Dependent draws', 
            xy=(1, 0.008),
            xytext=(1, 0.008),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'card_drawing_independence.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Independence in Machine Learning Features
print("\n\nExample 3: Independence in Machine Learning Features")
print("Spam email classification with two features:")

# Given probabilities
p_free = 0.3
p_exclam = 0.25
p_both = 0.15

# Calculate expected joint probability if independent
p_expected = p_free * p_exclam

print("\nGiven probabilities:")
print(f"P(X=1) = {p_free}")
print(f"P(Y>3) = {p_exclam}")
print(f"P(X=1, Y>3) = {p_both}")

print("\nIf independent, we would expect:")
print(f"P(X=1) × P(Y>3) = {p_free} × {p_exclam} = {p_expected}")

print("\nSince {p_both} ≠ {p_expected}, the features are not independent.")

# Create visualization
plt.figure(figsize=(10, 6))
x = np.array([0, 1])
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)
Z = np.array([[1 - p_free - p_exclam + p_both, p_exclam - p_both],
              [p_free - p_both, p_both]])

plt.imshow(Z, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks([0, 1], ['X=0', 'X=1'])
plt.yticks([0, 1], ['Y≤3', 'Y>3'])
plt.title('Joint Probability Distribution of Features', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'feature_independence.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll independence example images created successfully.") 