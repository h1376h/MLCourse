import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pandas as pd

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 25: Random Forest Configuration Battle Royale")
print("=" * 70)

# Given configurations
configs = {
    'Alpha': {'trees': 100, 'features_per_split': 4, 'max_depth': 8},
    'Beta': {'trees': 50, 'features_per_split': 8, 'max_depth': 12},
    'Gamma': {'trees': 200, 'features_per_split': 3, 'max_depth': 6}
}

print("\nGiven Random Forest Configurations:")
print("-" * 50)
for name, config in configs.items():
    print(f"Forest {name}: {config['trees']} trees, {config['features_per_split']} features per split, max_depth = {config['max_depth']}")

print("\n" + "="*70)
print("STEP 1: Calculate Tree Diversity for Each Forest")
print("="*70)

# Tree diversity is influenced by:
# 1. Number of trees (more trees = more diversity)
# 2. Features per split (fewer features = more diversity per tree)
# 3. Max depth (deeper trees = more complex, potentially more diverse)

def calculate_diversity_score(config):
    """Calculate a diversity score based on configuration parameters"""
    # More trees = higher diversity
    tree_factor = np.log(config['trees']) / np.log(200)  # Normalize to max trees
    
    # Fewer features per split = higher diversity per tree
    feature_factor = (20 - config['features_per_split']) / 20  # Normalize to max features
    
    # Moderate depth = good diversity (too shallow = simple, too deep = overfitting)
    depth_factor = 1 - abs(config['max_depth'] - 8) / 12  # Optimal around 8
    
    # Weighted combination
    diversity_score = 0.4 * tree_factor + 0.4 * feature_factor + 0.2 * depth_factor
    return diversity_score

diversity_scores = {}
print("Diversity Score Calculation:")
print("-" * 30)
for name, config in configs.items():
    diversity = calculate_diversity_score(config)
    diversity_scores[name] = diversity
    
    print(f"Forest {name}:")
    print(f"  Trees factor: {0.4 * np.log(config['trees']) / np.log(200):.3f}")
    print(f"  Feature factor: {0.4 * (20 - config['features_per_split']) / 20:.3f}")
    print(f"  Depth factor: {0.2 * (1 - abs(config['max_depth'] - 8) / 12):.3f}")
    print(f"  Total diversity score: {diversity:.3f}")
    print()

# Find forest with highest diversity
highest_diversity = max(diversity_scores.keys(), key=lambda x: diversity_scores[x])
print(f"Forest with highest diversity: {highest_diversity} (score: {diversity_scores[highest_diversity]:.3f})")

print("\n" + "="*70)
print("STEP 2: Training Time Analysis")
print("="*70)

# Each tree takes 2 seconds to train
training_times = {}
for name, config in configs.items():
    total_time = config['trees'] * 2  # seconds
    training_times[name] = total_time
    print(f"Forest {name}:")
    print(f"  Number of trees: {config['trees']}")
    print(f"  Time per tree: 2 seconds")
    print(f"  Total training time: {config['trees']} × 2 = {total_time} seconds")
    print()

# Find fastest forest
fastest_forest = min(training_times.keys(), key=lambda x: training_times[x])
print(f"Fastest training forest: {fastest_forest} ({training_times[fastest_forest]} seconds)")

print("\n" + "="*70)
print("STEP 3: Prediction Stability Analysis")
print("="*70)

# Prediction stability is influenced by:
# 1. Number of trees (more trees = more stable)
# 2. Features per split (more features = more stable per tree)
# 3. Max depth (deeper trees = potentially less stable due to overfitting)

def calculate_stability_score(config):
    """Calculate a stability score based on configuration parameters"""
    # More trees = higher stability
    tree_factor = np.log(config['trees']) / np.log(200)
    
    # More features per split = higher stability per tree
    feature_factor = config['features_per_split'] / 20
    
    # Moderate depth = good stability
    depth_factor = 1 - abs(config['max_depth'] - 8) / 12
    
    # Weighted combination
    stability_score = 0.5 * tree_factor + 0.3 * feature_factor + 0.2 * depth_factor
    return stability_score

stability_scores = {}
print("Stability Score Calculation:")
print("-" * 30)
for name, config in configs.items():
    stability = calculate_stability_score(config)
    stability_scores[name] = stability
    
    print(f"Forest {name}:")
    print(f"  Trees factor: {0.5 * np.log(config['trees']) / np.log(200):.3f}")
    print(f"  Feature factor: {0.3 * config['features_per_split'] / 20:.3f}")
    print(f"  Depth factor: {0.2 * (1 - abs(config['max_depth'] - 8) / 12):.3f}")
    print(f"  Total stability score: {stability:.3f}")
    print()

# Find most stable forest
most_stable = max(stability_scores.keys(), key=lambda x: stability_scores[x])
print(f"Most stable forest: {most_stable} (score: {stability_scores[most_stable]:.3f})")

print("\n" + "="*70)
print("STEP 4: Memory Usage Analysis")
print("="*70)

# Memory constraint: 1000 tree nodes total
# Estimate nodes per tree based on max_depth
def estimate_nodes_per_tree(max_depth):
    """Estimate number of nodes in a binary tree with given max_depth"""
    # Full binary tree: 2^(depth+1) - 1 nodes
    # But Random Forests don't always reach max_depth
    # Use a more realistic estimate: 2^depth nodes on average
    return 2 ** max_depth

memory_usage = {}
print("Memory Usage Analysis:")
print("-" * 25)
for name, config in configs.items():
    nodes_per_tree = estimate_nodes_per_tree(config['max_depth'])
    total_nodes = config['trees'] * nodes_per_tree
    memory_usage[name] = total_nodes
    
    print(f"Forest {name}:")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Estimated nodes per tree: 2^{config['max_depth']} = {nodes_per_tree}")
    print(f"  Total trees: {config['trees']}")
    print(f"  Total nodes: {config['trees']} × {nodes_per_tree} = {total_nodes}")
    print(f"  Fits in 1000 nodes: {'Yes' if total_nodes <= 1000 else 'No'}")
    print()

# Find forest that fits best in memory
best_memory_fit = min(memory_usage.keys(), key=lambda x: abs(memory_usage[x] - 1000))
print(f"Best memory fit: {best_memory_fit} ({memory_usage[best_memory_fit]} nodes)")

print("\n" + "="*70)
print("STEP 5: Feature Usage Analysis")
print("="*70)

# Calculate expected number of trees that will use a specific feature at least once
# This depends on features_per_split and number of trees

def calculate_feature_usage_probability(features_per_split, total_features=20):
    """Calculate probability that a specific feature is used in a single tree"""
    # Probability that a specific feature is NOT selected in one split
    prob_not_selected = (total_features - 1) / total_features
    
    # For a tree with multiple splits, we need to consider the probability
    # that the feature is never selected across all splits
    # This is a complex calculation, but we can approximate it
    
    # Average number of splits per tree (approximate)
    avg_splits_per_tree = 10  # Rough estimate
    
    # Probability that feature is never selected in any split
    prob_never_selected = prob_not_selected ** avg_splits_per_tree
    
    # Probability that feature is selected at least once
    prob_selected_at_least_once = 1 - prob_never_selected
    
    return prob_selected_at_least_once

feature_usage = {}
print("Feature Usage Analysis:")
print("-" * 25)
for name, config in configs.items():
    prob_per_tree = calculate_feature_usage_probability(config['features_per_split'])
    expected_trees = config['trees'] * prob_per_tree
    feature_usage[name] = expected_trees
    
    print(f"Forest {name}:")
    print(f"  Features per split: {config['features_per_split']}")
    print(f"  Probability per tree: {prob_per_tree:.3f}")
    print(f"  Number of trees: {config['trees']}")
    print(f"  Expected trees using feature: {config['trees']} × {prob_per_tree:.3f} = {expected_trees:.1f}")
    print()

# Find forest with highest feature usage
highest_feature_usage = max(feature_usage.keys(), key=lambda x: feature_usage[x])
print(f"Forest with highest feature usage: {highest_feature_usage} ({feature_usage[highest_feature_usage]:.1f} trees)")

print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON SUMMARY")
print("="*70)

# Create comparison table
comparison_data = []
for name in configs.keys():
    comparison_data.append({
        'Forest': name,
        'Diversity': diversity_scores[name],
        'Training Time (s)': training_times[name],
        'Stability': stability_scores[name],
        'Memory Usage': memory_usage[name],
        'Feature Usage': feature_usage[name]
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nComparison Table:")
print(comparison_df.to_string(index=False, float_format='%.3f'))

# Create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Radar chart for all metrics
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

# Prepare data for radar chart
categories = ['Diversity', 'Training Time', 'Stability', 'Memory Efficiency', 'Feature Usage']
N = len(categories)

# Normalize values for radar chart
diversity_norm = [diversity_scores[name] for name in configs.keys()]
training_norm = [1 - (training_times[name] / max(training_times.values())) for name in configs.keys()]  # Invert so lower is better
stability_norm = [stability_scores[name] for name in configs.keys()]
memory_norm = [1 - (memory_usage[name] / max(memory_usage.values())) for name in configs.keys()]  # Invert so lower is better
feature_norm = [feature_usage[name] / max(feature_usage.values()) for name in configs.keys()]

# Combine all metrics
values = np.array([diversity_norm, training_norm, stability_norm, memory_norm, feature_norm])

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Plot each forest
colors = ['red', 'blue', 'green']
forest_names = list(configs.keys())

for i, forest in enumerate(forest_names):
    values_forest = values[:, i]
    values_forest = np.concatenate((values_forest, [values_forest[0]]))  # Complete the circle
    ax.plot(angles, values_forest, 'o-', linewidth=2, label=f'Forest {forest}', color=colors[i])
    ax.fill(angles, values_forest, alpha=0.1, color=colors[i])

# Set labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Random Forest Configuration Comparison\n(All metrics normalized to 0-1 scale)', size=16, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'radar_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Bar chart for individual metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Random Forest Configuration Analysis', fontsize=16)

# Diversity
axes[0, 0].bar(configs.keys(), [diversity_scores[name] for name in configs.keys()], 
                color=['red', 'blue', 'green'], alpha=0.7)
axes[0, 0].set_title('Diversity Score')
axes[0, 0].set_ylabel('Diversity Score')
axes[0, 0].grid(True, alpha=0.3)

# Training Time
axes[0, 1].bar(configs.keys(), [training_times[name] for name in configs.keys()], 
                color=['red', 'blue', 'green'], alpha=0.7)
axes[0, 1].set_title('Training Time')
axes[0, 1].set_ylabel('Time (seconds)')
axes[0, 1].grid(True, alpha=0.3)

# Stability
axes[0, 2].bar(configs.keys(), [stability_scores[name] for name in configs.keys()], 
                color=['red', 'blue', 'green'], alpha=0.7)
axes[0, 2].set_title('Stability Score')
axes[0, 2].set_ylabel('Stability Score')
axes[0, 2].grid(True, alpha=0.3)

# Memory Usage
axes[1, 0].bar(configs.keys(), [memory_usage[name] for name in configs.keys()], 
                color=['red', 'blue', 'green'], alpha=0.7)
axes[1, 0].set_title('Memory Usage')
axes[1, 0].set_ylabel('Total Nodes')
axes[1, 0].axhline(y=1000, color='red', linestyle='--', label='Memory Limit')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Feature Usage
axes[1, 1].bar(configs.keys(), [feature_usage[name] for name in configs.keys()], 
                color=['red', 'blue', 'green'], alpha=0.7)
axes[1, 1].set_title('Feature Usage')
axes[1, 1].set_ylabel('Expected Trees Using Feature')
axes[1, 1].grid(True, alpha=0.3)

# Configuration Summary
axes[1, 2].axis('off')
config_text = "Configuration Summary:\n\n"
for name, config in configs.items():
    config_text += f"Forest {name}:\n"
    config_text += f"  Trees: {config['trees']}\n"
    config_text += f"  Features/split: {config['features_per_split']}\n"
    config_text += f"  Max depth: {config['max_depth']}\n\n"
axes[1, 2].text(0.1, 0.9, config_text, transform=axes[1, 2].transAxes, 
                 fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'metric_comparison.png'), dpi=300, bbox_inches='tight')

# 3. Decision boundary visualization (conceptual)
fig, ax = plt.subplots(figsize=(12, 8))

# Create sample data points
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Plot data points
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6, s=50)

# Add decision boundaries for each forest (conceptual)
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)

# Forest Alpha: More trees, moderate depth
ax.plot(x_range, -x_range + 0.2, 'r-', linewidth=2, label='Forest Alpha (100 trees, depth 8)')

# Forest Beta: Fewer trees, deeper
ax.plot(x_range, -x_range - 0.1, 'b-', linewidth=2, label='Forest Beta (50 trees, depth 12)')

# Forest Gamma: Most trees, shallow
ax.plot(x_range, -x_range + 0.5, 'g-', linewidth=2, label='Forest Gamma (200 trees, depth 6)')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Conceptual Decision Boundaries for Different Forest Configurations')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')

# 4. Training time vs performance trade-off
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each forest
for i, (name, config) in enumerate(configs.items()):
    ax.scatter(training_times[name], diversity_scores[name], 
               s=200, c=colors[i], alpha=0.7, label=f'Forest {name}')
    
    # Add configuration details
    ax.annotate(f"Trees: {config['trees']}\nDepth: {config['max_depth']}\nFeatures: {config['features_per_split']}", 
                (training_times[name], diversity_scores[name]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                fontsize=9)

ax.set_xlabel('Training Time (seconds)')
ax.set_ylabel('Diversity Score')
ax.set_title('Training Time vs Diversity Trade-off')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'time_diversity_tradeoff.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*70)
print("FINAL RECOMMENDATIONS")
print("="*70)

# Provide final recommendations based on different criteria
print("\nBest Forest by Different Criteria:")
print("-" * 40)
print(f"Highest Diversity: Forest {highest_diversity}")
print(f"Fastest Training: Forest {fastest_forest}")
print(f"Most Stable: Forest {most_stable}")
print(f"Best Memory Fit: Forest {best_memory_fit}")
print(f"Highest Feature Usage: Forest {highest_feature_usage}")

print("\nOverall Assessment:")
print("-" * 20)
print("Forest Alpha: Balanced performance, good diversity and stability")
print("Forest Beta: Fast training, deep trees, moderate stability")
print("Forest Gamma: High diversity, slow training, good feature coverage")

print(f"\nDetailed analysis complete! Check the generated images in: {save_dir}")
