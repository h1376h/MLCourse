import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import plot_tree
import seaborn as sns
import os
import time
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_7")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set matplotlib to non-interactive backend to avoid displaying plots
plt.ioff()

print("Question 7: Decision Tree Regularization Techniques")
print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF DECISION TREE REGULARIZATION METHODS")
print("=" * 80)

# ============================================================================
# PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS
# ============================================================================

print("\n" + "="*80)
print("PEN AND PAPER MATHEMATICAL SOLUTIONS WITH SYMBOLIC CALCULATIONS")
print("="*80)

# Task 1: Gini impurity threshold analysis with symbolic math
print("\n1. PEN AND PAPER: Gini Impurity Threshold Analysis (Symbolic)")
print("-" * 70)

# Define symbolic variables
G_initial = 0.5
G_final = 0.45
tau = 0.1

print("Symbolic Analysis:")
print(f"   - Initial Gini impurity: $G_{{initial}} = {G_initial}$")
print(f"   - Final Gini impurity: $G_{{final}} = {G_final}$")
print(f"   - Threshold: $\\tau = {tau}$")

# Calculate Gini reduction symbolically
Delta_G = G_initial - G_final
print(f"   - Gini impurity reduction: $\\Delta G = G_{{initial}} - G_{{final}} = {G_initial} - {G_final} = {Delta_G}$")

# Decision rule
print(f"\nDecision Rule (Symbolic):")
print(f"   - If $\\Delta G \\geq \\tau$, then: Allow split")
print(f"   - If $\\Delta G < \\tau$, then: Reject split")

# Specific case analysis
specific_case = Delta_G
print(f"\nSpecific Case Analysis:")
print(f"   - $\\Delta G = {G_initial} - {G_final} = {specific_case}$")
print(f"   - $\\tau = {tau}$")
print(f"   - Since ${specific_case} < {tau}$, the split should NOT be allowed")

# Mathematical reasoning with symbolic Gini formula
print(f"\nMathematical Reasoning:")
print(f"   - Gini impurity formula: $G = 1 - \\sum_{{i=1}}^{{c}} p_i^2$")
print(f"   - For binary classification (c=2): $G = 1 - p_1^2 - p_2^2 = 1 - p_1^2 - (1-p_1)^2 = 2p_1(1-p_1)$")
print(f"   - Split quality: $\\text{{Quality}} = \\frac{{\\Delta G}}{{\\text{{Complexity Cost}}}}$")
print(f"   - When $\\Delta G < \\tau$, the split doesn't provide sufficient improvement")

# Task 2: Feature selection with symbolic analysis
print("\n2. PEN AND PAPER: Feature Selection for Random Forests (Symbolic)")
print("-" * 70)

# Define symbolic variables
n_total = 10
n_features_sqrt = math.sqrt(n_total)
n_features_ratio = 0.3 * n_total
n_features_log = math.log2(n_total)
n_features_empirical = max(1, n_total/3)

print("Symbolic Analysis:")
print(f"   - Total features: $n_{{total}} = {n_total}$")
print(f"   - Square root rule: $n_{{features}} = \\sqrt{{{n_total}}} = {n_features_sqrt}$")
print(f"   - Fixed ratio rule: $n_{{features}} = 0.3 \\times {n_total} = {n_features_ratio}$")
print(f"   - Logarithmic rule: $n_{{features}} = \\log_2({n_total}) = {n_features_log}$")
print(f"   - Empirical rule: $n_{{features}} = \\max(1, \\frac{{{n_total}}}{{3}}) = {n_features_empirical}$")

# Optimal choice analysis
print(f"\nOptimal Choice Analysis:")
print(f"   - Recommended: $n_{{features}} = 3$ (rounded down from $\\sqrt{{{n_total}}}$)")
print(f"   - Reasoning: Balances diversity and accuracy")
print(f"   - Mathematical justification:")
print(f"     * Too few features: High variance, poor performance")
print(f"     * Too many features: Low diversity, overfitting risk")
print(f"     * Square root provides optimal balance: $\\sqrt{{n}}$")

# Task 3: max_depth vs post-pruning with symbolic depth analysis
print("\n3. PEN AND PAPER: max_depth vs Post-pruning Comparison (Symbolic)")
print("-" * 70)

# Define symbolic variables
d_max = 3
d_natural = 6

# Pre-pruning analysis
L_max_pre = 2**d_max
N_max_pre = sum(2**i for i in range(d_max + 1))

print("Symbolic Analysis:")
print(f"\nA) Pre-pruning (max_depth = {d_max}):")
print(f"   - Maximum depth: $d_{{max}} = {d_max}$")
print(f"   - Maximum leaf nodes: $L_{{max}} = 2^{{{d_max}}} = {L_max_pre}$")
print(f"   - Maximum total nodes: $N_{{max}} = \\sum_{{i=0}}^{{{d_max}}} 2^i = {N_max_pre}$")

# Post-pruning analysis
L_initial_post = 2**d_natural
N_initial_post = sum(2**i for i in range(d_natural + 1))

print(f"\nB) Post-pruning (natural depth = {d_natural}):")
print(f"   - Initial depth: $d_{{initial}} = {d_natural}$")
print(f"   - Initial leaf nodes: $L_{{initial}} = 2^{{{d_natural}}} = {L_initial_post}$")
print(f"   - Initial total nodes: $N_{{initial}} = \\sum_{{i=0}}^{{{d_natural}}} 2^i = {N_initial_post}$")

# Expected final size after post-pruning
print(f"\nPost-pruning Expected Final Size:")
print(f"   - Final depth: $d_{{final}} \\approx {d_natural-2}-{d_natural-1}$ (typically 70-80% of original)")
print(f"   - Final leaf nodes: $L_{{final}} \\approx {L_initial_post//3}-{L_initial_post//2}$ (typically 30-60% of original)")
print(f"   - Pruning ratio: $\\text{{Pruning Ratio}} = \\frac{{L_{{final}}}}{{L_{{initial}}}} \\approx 0.3-0.6$")

# Task 4: L1/L2 regularization with symbolic formulation
print("\n4. PEN AND PAPER: L1/L2 Regularization for Decision Trees (Symbolic)")
print("-" * 70)

# Define symbolic variables
lambda_1 = "λ₁"
lambda_2 = "λ₂"
R_T = "R(T)"  # Training error
T_size = "|T|"  # Number of leaf nodes
d_T = "d(T)"  # Maximum depth

# Traditional L1/L2 regularization
print("Traditional L1/L2 Regularization:")
print(f"   - L1 (Lasso): $R_{{L1}}(\\mathbf{{w}}) = {lambda_1} \\sum_{{i=1}}^n |w_i|$")
print(f"   - L2 (Ridge): $R_{{L2}}(\\mathbf{{w}}) = {lambda_2} \\sum_{{i=1}}^n w_i^2$")

# Decision tree adaptation
print(f"\nDecision Tree Adaptation:")
print(f"   - Instead of weights, regularize tree structure")
print(f"   - L1-like: Penalize number of splits")
print(f"   - L2-like: Penalize tree depth/complexity")

# Mathematical formulation
print(f"\nMathematical Formulation:")
print(f"   - L1-like: $C(T) = {R_T} + {lambda_1} {T_size}$")
print(f"     where ${T_size} = $ number of leaf nodes")
print(f"   - L2-like: $C(T) = {R_T} + {lambda_2} {d_T}^2$")
print(f"     where ${d_T} = $ maximum depth of tree")

# Implementation details
alpha = "α"
print(f"\nImplementation:")
print(f"   - L1-like: Cost-complexity pruning with $\\alpha = {lambda_1}$")
print(f"   - L2-like: Depth penalty in splitting criterion")
print(f"   - Combined: $C(T) = {R_T} + {lambda_1} {T_size} + {lambda_2} {d_T}^2$")
print(f"   - Cost-complexity pruning: $C(T) = {R_T} + {alpha} {T_size}$")

# Task 5: Probability calculation with symbolic combinatorics
print("\n5. PEN AND PAPER: Probability of Same Feature Selection (Symbolic)")
print("-" * 70)

# Define symbolic variables
n_total_sym = 10
k_sym = 3

# Probability calculations
P_feature_root = k_sym / n_total_sym
P_feature_left = k_sym / n_total_sym
P_same_feature = P_feature_root * P_feature_left

print("Symbolic Analysis:")
print(f"   - Total features: $n_{{total}} = {n_total_sym}$")
print(f"   - Features per split: $k = {k_sym}$")
print(f"   - Root selection: $P(\\text{{feature }} i \\text{{ at root}}) = \\frac{{{k_sym}}}{{{n_total_sym}}} = {P_feature_root}$")
print(f"   - Left child selection: $P(\\text{{feature }} i \\text{{ at left child}}) = \\frac{{{k_sym}}}{{{n_total_sym}}} = {P_feature_left}$")

# Independent selection
print(f"\nIndependent Selection:")
print(f"   - Assuming independent selection at each node")
print(f"   - $P(\\text{{same feature}}) = P(\\text{{feature }} i \\text{{ at root}}) \\times P(\\text{{feature }} i \\text{{ at left child}})$")
print(f"   - $P(\\text{{same feature}}) = {P_feature_root} \\times {P_feature_left} = {P_same_feature}$")

# Combinatorial analysis
print(f"\nCombinatorial Analysis:")
print(f"   - Total ways to select {k_sym} features from {n_total_sym}: $\\binom{{{n_total_sym}}}{{{k_sym}}}$")
print(f"   - Ways to select {k_sym} features excluding feature i: $\\binom{{{n_total_sym}-1}}{{{k_sym}}}$")
print(f"   - Probability feature i is NOT selected: P(not selected) = C({n_total_sym}-1,{k_sym})/C({n_total_sym},{k_sym})")

# Expected number of common features
E_common = n_total_sym * P_same_feature
print(f"   - Expected number of common features: $E[\\text{{common features}}] = {n_total_sym} \\times {P_same_feature:.3f} = {E_common:.1f}$")

# Task 6: Memory analysis with symbolic formulas
print("\n6. PEN AND PAPER: Memory-Constrained Regularization Strategy (Symbolic)")
print("-" * 70)

# Define symbolic variables
N_nodes = "N_nodes"
n_features = "n_features"
M_per_node = "M_per_node"

# Memory analysis
print("Memory Analysis:")
print(f"   - Tree memory: $M_{{tree}} \\propto {N_nodes} \\times ({n_features} + 1) \\times {M_per_node}$")
print(f"   - Simplified: $M_{{tree}} \\propto {N_nodes} \\times ({n_features} + 1)$")
print(f"   - Node storage components:")
print(f"     * Feature index: $\\log_2({n_features})$ bits")
print(f"     * Threshold: 32 bits (float)")
print(f"     * Child pointers: $2 \\times 32$ bits")
print(f"     * Total per node: $\\approx 16-24$ bytes (typical)")

# Strategy ranking analysis
print(f"\nStrategy Ranking (by memory efficiency):")
print(f"   1. Pre-pruning (max_depth): Most memory-efficient")
print(f"      - Memory: $M_{{pre}} \\propto N_{{pre}} \\times ({n_features} + 1)$")
print(f"      - Where $N_{{pre}} \\leq 2^{{d_{{max}}+1}} - 1$")
print(f"   2. Feature selection: Reduces feature storage")
print(f"      - Memory: $M_{{feat}} \\propto N_{{nodes}} \\times (\\sqrt{{{n_features}}} + 1)$")
print(f"   3. Post-pruning: Less memory-efficient")
print(f"      - Memory: $M_{{post}} \\propto N_{{post}} \\times ({n_features} + 1)$")
print(f"      - Where $N_{{post}} \\approx 0.3-0.7 \\times N_{{full}}$")
print(f"   4. Ensemble methods: Least memory-efficient")
print(f"      - Memory: $M_{{ens}} \\propto n_{{trees}} \\times M_{{tree}}$")

# Optimal strategy formulation
d_max_mem = 4
n_features_opt = math.sqrt(10)
M_optimal = (2**(d_max_mem + 1) - 1) * (n_features_opt + 1)

print(f"\nOptimal Strategy Formulation:")
print(f"   - Primary: Strict pre-pruning ($d_{{max}} \\leq {d_max_mem}$)")
print("     * Memory: $M_{optimal} = (2^{d_{max}+1} - 1) \\times (\\sqrt{10} + 1)$")
print(f"     * For $d_{{max}} = {d_max_mem}$, $n_{{features}} = 10$:")
print("       $M_{optimal} = (2^{d_{max}+1} - 1) \\times (\\sqrt{10} + 1)$")
print(f"       $M_{{optimal}} = {2**(d_max_mem+1)-1} \\times {n_features_opt + 1:.2f} \\approx {M_optimal:.0f}$ units")
print("   - Secondary: Feature selection ($n_{features} \\leq \\sqrt{10}$)")
print("   - Avoid: Post-pruning, large ensembles")

# Task 7: Expected unique features with symbolic probability
print("\n7. PEN AND PAPER: Expected Unique Features Calculation (Symbolic)")
print("-" * 70)

# Define symbolic variables
s_sym = 7  # number of splits
k_sym_unique = 3  # features per split
n_sym_unique = 10  # total features available

# Probability analysis
P_not_selected_one = math.comb(n_sym_unique - 1, k_sym_unique) / math.comb(n_sym_unique, k_sym_unique)
P_never_selected = P_not_selected_one**s_sym
P_selected_at_least_once = 1 - P_never_selected
E_unique_features = n_sym_unique * P_selected_at_least_once

print("Symbolic Analysis:")
print(f"   - Number of splits: $s = {s_sym}$")
print(f"   - Features per split: $k = {k_sym_unique}$")
print(f"   - Total features available: $n = {n_sym_unique}$")
print(f"   - Task: Calculate $E[\\text{{unique features used}}]$")

print(f"\nMathematical Analysis:")
print(f"   - Each split selects $k = {k_sym_unique}$ features randomly")
print(f"   - Total features selected: $s \\times k = {s_sym} \\times {k_sym_unique} = {s_sym * k_sym_unique}$")
print(f"   - Some features may be selected multiple times")

print(f"\nProbability Analysis:")
print(f"   - Probability feature $i$ is NOT selected in one split:")
print(f"     P(not selected) = C({n_sym_unique}-1,{k_sym_unique})/C({n_sym_unique},{k_sym_unique}) = {P_not_selected_one}")
print(f"   - Probability feature $i$ is NOT selected in any of $s$ splits:")
print(f"     $P(\\text{{never selected}}) = {P_not_selected_one}^{{s_sym}} = {P_never_selected}$")

print(f"\nExpected Unique Features:")
print(f"   - $E[\\text{{unique features}}] = n \\times P(\\text{{feature is selected at least once}})$")
print(f"   - $E[\\text{{unique features}}] = {n_sym_unique} \\times (1 - {P_never_selected}) = {E_unique_features}$")

# Substitute specific values
s_val = 7
k_val_unique = 3
n_val_unique = 10

P_not_selected_specific = float(P_not_selected_one)
P_never_selected_specific = float(P_never_selected)
E_unique_specific = float(E_unique_features)

print(f"\nSpecific Case (s = {s_val}, k = {k_val_unique}, n = {n_val_unique}):")
print(f"   - P(not selected) = C({n_val_unique-1},{k_val_unique})/C({n_val_unique},{k_val_unique}) = {P_not_selected_specific:.3f}")
print(f"   - $P(\\text{{never selected}}) = {P_not_selected_specific:.3f}^{{{s_val}}} \\approx {P_never_selected_specific:.3f}$")
print(f"   - E[unique features] = {n_val_unique} × (1 - {P_never_selected_specific:.3f}) = {n_val_unique} × {1-P_never_selected_specific:.3f} = {E_unique_specific:.2f}")

# Verification bounds
print(f"\nVerification:")
print(f"   - Minimum: $k = {k_val_unique}$ (if same features always selected)")
print(f"   - Maximum: $n = {n_val_unique}$ (if all features used)")
print(f"   - Expected: ${E_unique_specific:.2f}$ (close to maximum, indicating good diversity)")
print(f"   - Diversity ratio: $\\frac{{{E_unique_specific:.2f}}}{{{n_val_unique}}} = {E_unique_specific/n_val_unique:.3f}$")

# ============================================================================
# PRACTICAL IMPLEMENTATION AND VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PRACTICAL IMPLEMENTATION AND VISUALIZATION")
print("="*80)

# Create synthetic dataset for demonstrations
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                          n_redundant=2, n_clusters_per_class=1, random_state=42)

# Task 1: Visualize Gini impurity threshold analysis
print("\n1. PRACTICAL: Gini Impurity Threshold Analysis")
print("-" * 70)

def calculate_gini_impurity(y):
    """Calculate Gini impurity for a set of labels"""
    if len(y) == 0:
        return 0
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities**2)

def find_best_split(X, y, feature_idx, threshold):
    """Find the best split for a given feature and threshold"""
    left_mask = X[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    
    gini_left = calculate_gini_impurity(y[left_mask])
    gini_right = calculate_gini_impurity(y[right_mask])
    
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)
    n_total = len(y)
    
    gini_after = (n_left * gini_left + n_right * gini_right) / n_total
    gini_before = calculate_gini_impurity(y)
    
    return gini_before - gini_after

# Test different thresholds
thresholds = [0.01, 0.05, 0.1, 0.15, 0.2]
gini_improvements = []

for threshold in thresholds:
    # Find best split for feature 0
    best_improvement = 0
    for t in np.linspace(X[:, 0].min(), X[:, 0].max(), 100):
        improvement = find_best_split(X, y, 0, t)
        if improvement > best_improvement:
            best_improvement = improvement
    
    gini_improvements.append(best_improvement)

# Plot threshold analysis
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(thresholds, gini_improvements, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold = 0.1')
plt.xlabel('Gini Improvement Threshold')
plt.ylabel('Best Gini Improvement Found')
plt.title('Gini Improvement vs Threshold')
plt.grid(True, alpha=0.3)
plt.legend()

# Highlight the specific case from the question
plt.annotate('Question case:\n$\\Delta G = 0.05 < 0.1$\nSplit rejected', 
             xy=(0.1, 0.05), xytext=(0.15, 0.15),
             arrowprops=dict(arrowstyle='->', lw=2, color='red'),
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

# Task 2: Feature selection visualization
print("\n2. PRACTICAL: Feature Selection Analysis")
print("-" * 70)

plt.subplot(2, 2, 2)
feature_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
cv_scores = []

for n_features in feature_counts:
    if n_features <= 10:
        # Use Random Forest with different feature counts
        rf = RandomForestClassifier(n_estimators=10, max_features=n_features, 
                                  random_state=42, max_depth=5)
        scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    else:
        cv_scores.append(0)

plt.plot(feature_counts, cv_scores, 'go-', linewidth=2, markersize=8)
plt.axvline(x=3, color='r', linestyle='--', label='Optimal: $\\sqrt{10} \\approx 3$')
plt.xlabel('Number of Features per Split')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Feature Selection vs Performance')
plt.grid(True, alpha=0.3)
plt.legend()

# Task 3: max_depth vs post-pruning comparison
print("\n3. PRACTICAL: max_depth vs Post-pruning Comparison")
print("-" * 70)

plt.subplot(2, 2, 3)

# Test different max_depths
depths = [3, 4, 5, 6, 7, 8]
pre_pruning_scores = []
post_pruning_scores = []

for depth in depths:
    # Pre-pruning
    dt_pre = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores_pre = cross_val_score(dt_pre, X, y, cv=5, scoring='accuracy')
    pre_pruning_scores.append(scores_pre.mean())
    
    # Post-pruning (cost-complexity)
    dt_post = DecisionTreeClassifier(random_state=42)
    dt_post.fit(X, y)
    
    # Find optimal alpha for cost-complexity pruning
    path = dt_post.cost_complexity_pruning_path(X, y)
    optimal_alpha = path.ccp_alphas[np.argmax(path.impurities)]
    
    dt_post_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
    scores_post = cross_val_score(dt_post_pruned, X, y, cv=5, scoring='accuracy')
    post_pruning_scores.append(scores_post.mean())

plt.plot(depths, pre_pruning_scores, 'bo-', label='Pre-pruning (max_depth)', linewidth=2, markersize=8)
plt.plot(depths, post_pruning_scores, 'ro-', label='Post-pruning (cost-complexity)', linewidth=2, markersize=8)
plt.axvline(x=3, color='g', linestyle='--', label='Question case: max_depth=3')
plt.xlabel('Tree Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Pre-pruning vs Post-pruning')
plt.grid(True, alpha=0.3)
plt.legend()

# Task 4: L1/L2 regularization visualization
print("\n4. PRACTICAL: L1/L2 Regularization Effects")
print("-" * 70)

plt.subplot(2, 2, 4)

# Test different regularization strengths
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
tree_sizes = []
depths = []

for alpha in alphas:
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    
    # Apply cost-complexity pruning
    path = dt.cost_complexity_pruning_path(X, y)
    optimal_alpha = path.ccp_alphas[np.argmax(path.impurities)]
    
    dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    dt_pruned.fit(X, y)
    
    tree_sizes.append(dt_pruned.tree_.node_count)
    depths.append(dt_pruned.get_depth())

plt.plot(alphas, tree_sizes, 'mo-', label='Tree Size (L1-like)', linewidth=2, markersize=8)
plt.xlabel('Regularization Strength ($\\alpha$)')
plt.ylabel('Tree Size (nodes)')
plt.title('L1-like Regularization Effect')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_analysis.png'), dpi=300, bbox_inches='tight')

# Task 5: Probability calculation verification
print("\n5. PRACTICAL: Probability Calculation Verification")
print("-" * 70)

# Simulate feature selection process
np.random.seed(42)
n_simulations = 10000
n_features = 10
features_per_split = 3
n_splits = 2  # root and left child

same_feature_count = 0
for _ in range(n_simulations):
    # Select features for root
    root_features = np.random.choice(n_features, features_per_split, replace=False)
    # Select features for left child
    left_features = np.random.choice(n_features, features_per_split, replace=False)
    
    # Check if any feature is common
    if np.any(np.isin(root_features, left_features)):
        same_feature_count += 1

empirical_prob = same_feature_count / n_simulations
theoretical_prob = 1 - (math.comb(n_features - features_per_split, features_per_split) / 
                        math.comb(n_features, features_per_split))**2

print(f"Empirical probability: {empirical_prob:.4f}")
print(f"Theoretical probability: {theoretical_prob:.4f}")
print(f"Difference: {abs(empirical_prob - theoretical_prob):.4f}")

# Task 6: Memory usage analysis
print("\n6. PRACTICAL: Memory Usage Analysis")
print("-" * 70)

# Create trees with different regularization strategies
strategies = ['No regularization', 'max_depth=3', 'max_features=3', 'Cost-complexity pruning']
memory_usage = []
accuracy_scores = []

# No regularization
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X, y)
memory_usage.append(dt_full.tree_.node_count * 24)  # 24 bytes per node estimate
accuracy_scores.append(cross_val_score(dt_full, X, y, cv=5, scoring='accuracy').mean())

# max_depth=3
dt_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_depth.fit(X, y)
memory_usage.append(dt_depth.tree_.node_count * 24)
accuracy_scores.append(cross_val_score(dt_depth, X, y, cv=5, scoring='accuracy').mean())

# max_features=3
dt_features = DecisionTreeClassifier(max_features=3, random_state=42)
dt_features.fit(X, y)
memory_usage.append(dt_features.tree_.node_count * 24)
accuracy_scores.append(cross_val_score(dt_features, X, y, cv=5, scoring='accuracy').mean())

# Cost-complexity pruning
dt_ccp = DecisionTreeClassifier(random_state=42)
dt_ccp.fit(X, y)
path = dt_ccp.cost_complexity_pruning_path(X, y)
optimal_alpha = path.ccp_alphas[np.argmax(path.impurities)]
dt_ccp_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
dt_ccp_pruned.fit(X, y)
memory_usage.append(dt_ccp_pruned.tree_.node_count * 24)
accuracy_scores.append(cross_val_score(dt_ccp_pruned, X, y, cv=5, scoring='accuracy').mean())

# Plot memory vs accuracy trade-off
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.bar(strategies, memory_usage, color=['red', 'green', 'blue', 'orange'])
plt.ylabel('Memory Usage (bytes)')
plt.title('Memory Usage by Strategy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.bar(strategies, accuracy_scores, color=['red', 'green', 'blue', 'orange'])
plt.ylabel('Accuracy Score')
plt.title('Accuracy by Strategy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Task 7: Expected unique features verification
print("\n7. PRACTICAL: Expected Unique Features Verification")
print("-" * 70)

# Simulate the feature selection process
np.random.seed(42)
n_simulations = 10000
n_splits = 7
features_per_split = 3
n_total_features = 10

unique_features_counts = []
for _ in range(n_simulations):
    all_selected_features = []
    for _ in range(n_splits):
        split_features = np.random.choice(n_total_features, features_per_split, replace=False)
        all_selected_features.extend(split_features)
    
    unique_count = len(np.unique(all_selected_features))
    unique_features_counts.append(unique_count)

empirical_mean = np.mean(unique_features_counts)
theoretical_mean = 10 * (1 - (math.comb(9, 3) / math.comb(10, 3))**7)

print(f"Empirical mean: {empirical_mean:.2f}")
print(f"Theoretical mean: {theoretical_mean:.2f}")
print(f"Difference: {abs(empirical_mean - theoretical_mean):.2f}")

# Plot distribution of unique features
plt.subplot(2, 2, 3)
plt.hist(unique_features_counts, bins=range(min(unique_features_counts), max(unique_features_counts) + 2), 
         alpha=0.7, color='green', edgecolor='black')
plt.axvline(x=empirical_mean, color='red', linestyle='--', label=f'Mean: {empirical_mean:.2f}')
plt.axvline(x=theoretical_mean, color='blue', linestyle='--', label=f'Theoretical: {theoretical_mean:.2f}')
plt.xlabel('Number of Unique Features')
plt.ylabel('Frequency')
plt.title('Distribution of Unique Features Used')
plt.legend()
plt.grid(True, alpha=0.3)

# Create summary visualization
plt.subplot(2, 2, 4)
summary_data = {
    'Strategy': strategies,
    'Memory (KB)': [m/1024 for m in memory_usage],
    'Accuracy': accuracy_scores
}
summary_df = pd.DataFrame(summary_data)

plt.table(cellText=summary_df.values, colLabels=summary_df.columns, 
          cellLoc='center', loc='center')
plt.axis('off')
plt.title('Strategy Comparison Summary')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'memory_accuracy_analysis.png'), dpi=300, bbox_inches='tight')

# Create comprehensive summary plot
plt.figure(figsize=(16, 10))

# Summary statistics
summary_stats = {
    'Gini Threshold': f'$\\Delta G = 0.05 < 0.1 \\rightarrow$ Split rejected',
    'Feature Selection': f'$\\sqrt{{10}} \\approx 3$ features per split',
    'Pre vs Post': f'max_depth=3: $\\leq 15$ nodes, Post: $\\approx 20-40$ nodes',
    'L1/L2 Adaptation': 'Cost-complexity: $R(T) + \\alpha|T|$',
    'Same Feature Prob': f'$P = 0.09$ (theoretical)',
    'Memory Strategy': 'Pre-pruning + feature selection',
    'Unique Features': f'$E[unique] \\approx 9.18$'
}

y_pos = np.arange(len(summary_stats))
plt.barh(y_pos, [1]*len(summary_stats), color='lightblue', alpha=0.7)
plt.yticks(y_pos, list(summary_stats.keys()))
plt.xlabel('Summary')

for i, (key, value) in enumerate(summary_stats.items()):
    plt.text(0.5, i, value, ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

plt.title('Question 7: Decision Tree Regularization - Summary of Solutions')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'comprehensive_summary.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)
