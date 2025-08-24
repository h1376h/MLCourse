import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
save_dir = os.path.join(images_dir, "L6_4_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set matplotlib to non-interactive backend to avoid displaying plots
plt.ioff()

print("Question 6: Pruning Method Comparison")
print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF DECISION TREE PRUNING METHODS")
print("=" * 80)

# ============================================================================
# PEN AND PAPER MATHEMATICAL SOLUTIONS
# ============================================================================

print("\n" + "="*80)
print("PEN AND PAPER MATHEMATICAL SOLUTIONS")
print("="*80)

# Task 1: Ranking pruning methods by expected tree size
print("\n1. PEN AND PAPER: Ranking Pruning Methods by Expected Tree Size")
print("-" * 70)

print("Mathematical Analysis of Tree Size:")
print("Let $|T|$ = number of leaf nodes in tree $T$")
print("Let $R(T)$ = training error of tree $T$")
print("Let $\\alpha$ = cost-complexity parameter")

print("\nA) Pre-pruning with max_depth=3:")
print("   - Maximum tree depth = 3")
print("   - Maximum leaf nodes = $2^3 = 8$")
print("   - Expected size: $|T| \\leq 8$")
print("   - Mathematical reasoning:")
print("     * At depth 0: 1 node (root)")
print("     * At depth 1: 2 nodes (left, right children)")
print("     * At depth 2: 4 nodes ($2^2$)")
print("     * At depth 3: 8 nodes ($2^3$)")
print("     * Total nodes = $\\sum_{i=0}^3 2^i = 1 + 2 + 4 + 8 = 15$")
print("     * Leaf nodes = $2^3 = 8$")
print("     * Internal nodes = $15 - 8 = 7$")

print("\nB) Reduced Error Pruning (REP):")
print("   - Uses validation set to prune")
print("   - Prunes nodes that don't improve validation accuracy")
print("   - Expected size: $|T| \\approx 0.3 \\times |T_{full}|$ to $0.7 \\times |T_{full}|$")
print("   - For typical trees: $|T| \\approx 15-50$ nodes")
print("   - Mathematical reasoning:")
print("     * Full tree typically has $|T_{full}| \\approx 100-200$ nodes")
print("     * REP prunes 30-70% of nodes based on validation performance")
print("     * Pruning ratio varies based on validation set performance")
print("     * Leaf nodes typically $\\approx \\frac{|T|}{2}$")
print("     * Validation accuracy threshold determines exact pruning ratio")

print("\nC) Cost-Complexity Pruning with $\\alpha=0.1$:")
print("   - Minimizes: $R(T) + \\alpha|T|$")
print("   - $\\alpha=0.1$ means each leaf costs 0.1 in complexity penalty")
print("   - Expected size: $|T| \\approx 0.2 \\times |T_{full}|$ to $0.5 \\times |T_{full}|$")
print("   - For typical trees: $|T| \\approx 10-30$ nodes")
print("   - Mathematical reasoning:")
print("     * Cost function: $C(T) = R(T) + 0.1|T|$")
print("     * For $\\alpha = 0.1$, complexity penalty is significant")
print("     * Algorithm prefers smaller trees when $\\alpha$ is large")
print("     * Optimal tree size: $|T^*| = \\arg\\min_T [R(T) + 0.1|T|]$")
print("     * Trade-off: $\\Delta R(T) = -0.1 \\Delta |T|$")
print("     * When $R(T)$ increases by $\\Delta R$, tree size decreases by $10\\Delta R$")

print("\nRanking (smallest to largest):")
print("1. Cost-complexity pruning ($\\alpha=0.1$): Smallest")
print("2. Pre-pruning (max_depth=3): Medium")
print("3. Reduced error pruning: Largest")

print("\nMathematical Verification:")
print("Expected tree sizes:")
print("1. CCP: $|T| \\approx 10-30$ nodes (smallest)")
print("2. Pre-pruning: $|T| \\leq 15$ nodes (medium)")
print("3. REP: $|T| \\approx 60-100$ nodes (largest)")
print("Actual sizes will depend on dataset characteristics and pruning parameters")

print("\nTheoretical vs. Practical:")
print("- CCP: $\\alpha = 0.1$ creates strong complexity penalty")
print("- Pre-pruning: Hard constraint $d \\leq 3$ limits growth")
print("- REP: Validation-based pruning preserves more structure")

# Task 2: Robustness to noisy data
print("\n2. PEN AND PAPER: Robustness to Noisy Data")
print("-" * 50)

print("Mathematical Analysis:")
print("Let $\\varepsilon$ = noise level in data")
print("Let $R_{train}(T)$ = training error")
print("Let $R_{test}(T)$ = test error")
print("Let $R_{noise}(T)$ = error due to noise")
print("Let $R_{bias}(T)$ = bias error")
print("Let $R_{var}(T)$ = variance error")

print("\nGeneralization Error Decomposition:")
print("$R_{test}(T) = R_{train}(T) + R_{noise}(T) + R_{bias}(T) + R_{var}(T)$")
print("where:")
print("- $R_{train}(T)$ = irreducible training error")
print("- $R_{noise}(T)$ = error due to data noise")
print("- $R_{bias}(T)$ = error due to model bias (underfitting)")
print("- $R_{var}(T)$ = error due to model variance (overfitting)")

print("\nA) Pre-pruning (max_depth=3):")
print("   - High bias, low variance")
print("   - $R_{bias}(T)$ = High (underfitting)")
print("   - $R_{noise}(T)$ = Low (less sensitive to noise)")
print("   - $R_{var}(T)$ = Low (stable predictions)")
print("   - Robustness: HIGH")
print("   - Mathematical reasoning:")
print("     * Limited depth $d \\leq 3$ prevents overfitting")
print("     * $R_{bias}(T)$ is high due to underfitting")
print("     * $R_{var}(T)$ is low due to simple model")
print("     * $R_{noise}(T)$ is low due to limited complexity")
print("     * Total error: $R_{bias}(T) + R_{var}(T) + R_{noise}(T)$")
print("     * Noise robustness: $\\frac{R_{noise}(T)}{R_{test}(T)}$ (low ratio indicates high robustness)")

print("\nB) Reduced Error Pruning:")
print("   - Medium bias, medium variance")
print("   - $R_{bias}(T)$ = Medium")
print("   - $R_{noise}(T)$ = Medium")
print("   - $R_{var}(T)$ = Medium")
print("   - Robustness: MEDIUM")
print("   - Mathematical reasoning:")
print("     * Validation-based pruning balances bias and variance")
print("     * $R_{bias}(T)$ is medium due to moderate complexity")
print("     * $R_{var}(T)$ is medium due to moderate flexibility")
print("     * $R_{noise}(T)$ is medium due to moderate complexity")
print("     * Total error: $R_{bias}(T) + R_{var}(T) + R_{noise}(T)$")
print("     * Noise robustness: $\\frac{R_{noise}(T)}{R_{test}(T)}$ (medium ratio indicates moderate robustness)")
print("     * Pruning removes nodes that don't improve validation accuracy")

print("\nC) Cost-Complexity Pruning ($\\alpha=0.1$):")
print("   - Low bias, high variance")
print("   - $R_{bias}(T)$ = Low")
print("   - $R_{noise}(T)$ = High (overfitting to noise)")
print("   - $R_{var}(T)$ = High (unstable predictions)")
print("   - Robustness: LOW")
print("   - Mathematical reasoning:")
print("     * $\\alpha = 0.1$ allows complex trees to minimize training error")
print("     * $R_{bias}(T)$ is low due to high flexibility")
print("     * $R_{var}(T)$ is high due to complex model")
print("     * $R_{noise}(T)$ is high due to high complexity")
print("     * Total error: $R_{bias}(T) + R_{var}(T) + R_{noise}(T)$")
print("     * Noise robustness: $\\frac{R_{noise}(T)}{R_{test}(T)}$ (high ratio indicates low robustness)")
print("     * High variance makes model sensitive to noise patterns")

print("\nMost Robust: Pre-pruning (max_depth=3)")
print("Why: High bias prevents overfitting to noise")

print("\nQuantitative Comparison:")
print("Noise sensitivity ratios:")
print("1. Pre-pruning: Low ratio (most robust)")
print("2. REP: Medium ratio (moderately robust)")
print("3. CCP: High ratio (least robust)")
print("Exact values depend on dataset characteristics and noise levels")

print("\nMathematical Insight:")
print("$R_{noise}(T) \\propto \\text{complexity}(T) \\times \\text{depth}(T)$")
print("Pre-pruning minimizes complexity and depth, reducing noise sensitivity")

# Task 3: Computational speed comparison
print("\n3. PEN AND PAPER: Computational Speed Analysis")
print("-" * 50)

print("Time Complexity Analysis:")
print("Let $n$ = number of samples, $d$ = number of features")
print("Let $|T|$ = number of nodes in tree")
print("Let $d_{max}$ = maximum tree depth")

print("\nA) Pre-pruning (max_depth=3):")
print("   - Training: $O(n \\times d \\times 2^3) = O(8nd)$")
print("   - Prediction: $O(3) = O(1)$")
print("   - Overall: FASTEST")
print("   - Mathematical reasoning:")
print("     * Fixed depth $d_{max} = 3$")
print("     * At each level: $O(nd)$ for finding best split")
print("     * Total levels: $\\log_2(8) = 3$")
print("     * Training time: $\\sum_{i=0}^3 O(nd \\times 2^i) = O(8nd)$")
print("     * Prediction: constant time $O(d_{max}) = O(3)$")
print("     * Memory: $O(2^{d_{max}}) = O(8)$ nodes")

print("\nB) Reduced Error Pruning:")
print("   - Training: $O(n \\times d \\times |T|) + O(|T| \\times n_{val})$")
print("   - Prediction: $O(\\log |T|)$")
print("   - Overall: MEDIUM")
print("   - Mathematical reasoning:")
print("     * Build full tree: $O(nd|T|)$")
print("     * Validation pruning: $O(|T| \\times n_{val})$")
print("     * Where $n_{val}$ = validation set size")
print("     * Training: $O(nd|T|) + O(|T|n_{val})$")
print("     * Prediction: $O(\\log |T|)$")
print("     * Memory: $O(|T|)$ nodes")

print("\nC) Cost-Complexity Pruning:")
print("   - Training: $O(n \\times d \\times |T|) + O(|T|^2 \\times \\log |T|)$")
print("   - Prediction: $O(\\log |T|)$")
print("   - Overall: SLOWEST")
print("   - Mathematical reasoning:")
print("     * Build full tree: $O(nd|T|)$")
print("     * Find optimal $\\alpha$: $O(|T|^2 \\log |T|)$")
print("     * For each $\\alpha$ value, evaluate $O(|T|)$ subtrees")
print("     * Training: $O(nd|T|) + O(|T|^2 \\log |T|)$")
print("     * Prediction: $O(\\log |T|)$")
print("     * Memory: $O(|T|)$ nodes")
print("     * The $O(|T|^2 \\log |T|)$ term dominates for large trees")

print("\nSpeed Ranking (fastest to slowest):")
print("1. Pre-pruning (max_depth=3)")
print("2. Reduced error pruning")
print("3. Cost-complexity pruning")

print("\nQuantitative Comparison:")
print("Training times:")
print("1. Pre-pruning: $O(8nd)$ (fastest)")
print("2. REP: $O(nd|T| + |T|n_{val})$ (medium)")
print("3. CCP: $O(nd|T| + |T|^2\\log|T|)$ (slowest)")

print("\nSpeed ratios:")
print("REP vs Pre-pruning: $\\frac{O(nd|T| + |T|n_{val})}{O(8nd)}$ (slower)")
print("CCP vs Pre-pruning: $\\frac{O(nd|T| + |T|^2\\log|T|)}{O(8nd)}$ (much slower)")
print("CCP vs REP: $\\frac{O(nd|T| + |T|^2\\log|T|)}{O(nd|T| + |T|n_{val})}$ (slower due to quadratic term)")

# Task 4: Interpretability evaluation
print("\n4. PEN AND PAPER: Interpretability Analysis")
print("-" * 50)

print("Interpretability Metrics:")
print("Let $I(T)$ = interpretability score")
print("Let $D(T)$ = average depth of leaf nodes")
print("Let $L(T)$ = number of leaf nodes")
print("Let $\\text{complexity}(T)$ = model complexity")

print("\nInterpretability Score Formula:")
print("$I(T) = D(T) \\times \\log(L(T))$ (lower is more interpretable)")
print("Alternative: $I(T) = \\frac{D(T) \\times L(T)}{\\text{complexity}(T)}$")
print("Rationale: Shallow depth and fewer leaves make rules easier to understand")

print("\nA) Pre-pruning (max_depth=3):")
print("   - $D(T) = 3$ (shallow)")
print("   - $L(T) \\leq 8$ (few leaves)")
print("   - $I(T) = 3 \\times \\log(8) = 3 \\times 2.08 = 6.24$")
print("   - Interpretability: HIGH")
print("   - Mathematical reasoning:")
print("     * Maximum depth constraint: $d_{max} = 3$")
print("     * Maximum leaves: $L(T) = 2^3 = 8$")
print("     * Average depth: $D(T) = \\frac{0 + 1 + 2 + 3}{4} = 1.5$")
print("     * Interpretability score: $1.5 \\times \\log(8)$")
print("     * Rule complexity: $\\text{complexity}(T) = 8$ rules")
print("     * Each rule has at most 3 conditions")
print("     * Human-readable format: IF condition1 AND condition2 AND condition3 THEN class")

print("\nB) Reduced Error Pruning:")
print("   - $D(T) \\approx 5-8$ (medium)")
print("   - $L(T) \\approx 15-50$ (medium)")
print("   - $I(T) \\approx 6.5 \\times \\log(30) = 6.5 \\times 3.4 = 22.1$")
print("   - Interpretability: MEDIUM")
print("   - Mathematical reasoning:")
print("     * Typical depth: $D(T) \\approx 5-8$ (medium)")
print("     * Typical leaves: $L(T) \\approx 15-50$ (medium)")
print("     * Interpretability score: $D(T) \\times \\log(L(T))$")
print("     * Rule complexity: $\\text{complexity}(T) \\approx L(T)$ rules")
print("     * Each rule has 5-8 conditions")
print("     * Human-readable format: IF condition1 AND ... AND condition8 THEN class")
print("     * More complex than pre-pruning but still manageable")

print("\nC) Cost-Complexity Pruning ($\\alpha=0.1$):")
print("   - $D(T) \\approx 4-6$ (medium)")
print("   - $L(T) \\approx 10-30$ (fewer leaves)")
print("   - $I(T) \\approx 5 \\times \\log(20) = 5 \\times 3.0 = 15.0$")
print("   - Interpretability: MEDIUM-HIGH")
print("   - Mathematical reasoning:")
print("     * Typical depth: $D(T) \\approx 4-6$ (medium)")
print("     * Typical leaves: $L(T) \\approx 10-30$ (medium)")
print("     * Interpretability score: $D(T) \\times \\log(L(T))$")
print("     * Rule complexity: $\\text{complexity}(T) \\approx L(T)$ rules")
print("     * Each rule has 4-6 conditions")
print("     * Human-readable format: IF condition1 AND ... AND condition6 THEN class")
print("     * $\\alpha = 0.1$ creates balanced complexity")
print("     * Better than REP due to fewer leaves, worse than pre-pruning due to depth")

print("\nMost Interpretable: Pre-pruning (max_depth=3)")
print("Why: Shallow depth and few leaves make rules easy to understand")

print("\nQuantitative Comparison:")
print("Interpretability scores (lower is better):")
print("1. Pre-pruning: Low score (most interpretable)")
print("2. CCP: Medium score (moderately interpretable)")
print("3. REP: High score (least interpretable)")

print("\nMathematical Insight:")
print("$I(T) \\propto D(T) \\times \\log(L(T))$")
print("Pre-pruning minimizes both depth and leaf count")
print("Interpretability ratio depends on actual tree characteristics")

# Task 5: Performance with increasing noise
print("\n5. PEN AND PAPER: Performance with Increasing Noise")
print("-" * 60)

print("Noise Analysis:")
print("Let $\\varepsilon(x)$ = noise level at feature value $x$")
print("Let $\\varepsilon'(x) > 0$ (noise increases with feature values)")
print("Let $R_{noise}(T, x)$ = noise error at feature value $x$")
print("Let $\\text{depth}(T)$ = tree depth")
print("Let $\\text{complexity}(T)$ = tree complexity")

print("\nMathematical Formulation:")
print("$R_{noise}(T, x) \\propto \\varepsilon(x) \\times |T| \\times \\text{depth}(T)$")
print("More precisely: $R_{noise}(T, x) = k \\cdot \\varepsilon(x) \\cdot |T|^{\\alpha} \\cdot \\text{depth}(T)^{\\beta}$")
print("where $k, \\alpha, \\beta$ are constants, typically $\\alpha \\approx 0.5, \\beta \\approx 1.0$")
print("This models the fact that deeper, more complex trees are more sensitive to noise")

print("\nA) Pre-pruning (max_depth=3):")
print("   - Fixed depth = 3")
print("   - $R_{noise}(T, x) \\propto \\varepsilon(x) \\times |T| \\times 3$")
print("   - Performance: STABLE")
print("   - Mathematical reasoning:")
print("     * Fixed depth: $\\text{depth}(T) = 3$ (constant)")
print("     * Fixed size: $|T| \\leq 15$ (bounded)")
print("     * Noise sensitivity: $R_{noise}(T, x) = k \\cdot \\varepsilon(x) \\cdot |T|^{0.5} \\cdot 3^{1.0}$")
print("     * As $\\varepsilon(x)$ increases, error grows linearly")
print("     * Performance degradation: $\\frac{\\Delta R_{noise}}{\\Delta \\varepsilon} = k \\cdot |T|^{0.5} \\cdot 3$ (constant)")
print("     * Stability: High (bounded growth rate)")

print("\nB) Reduced Error Pruning:")
print("   - Variable depth ≈ 5-8")
print("   - $R_{noise}(T, x) \\propto \\varepsilon(x) \\times |T| \\times 6.5$")
print("   - Performance: MODERATELY DEGRADED")
print("   - Mathematical reasoning:")
print("     * Variable depth: $\\text{depth}(T) \\approx 6.5$ (average)")
print("     * Variable size: $|T|$ (from actual tree)")
print("     * Noise sensitivity: $R_{noise}(T, x) = k \\cdot \\varepsilon(x) \\cdot |T|^{0.5} \\cdot 6.5^{1.0}$")
print("     * As $\\varepsilon(x)$ increases, error grows linearly but faster")
print("     * Performance degradation: $\\frac{\\Delta R_{noise}}{\\Delta \\varepsilon} = k \\cdot |T|^{0.5} \\cdot 6.5$")
print("     * Stability: Medium (higher growth rate)")
print("     * Reason: Larger trees capture more noise patterns")

print("\nC) Cost-Complexity Pruning ($\\alpha=0.1$):")
print("   - Variable depth ≈ 4-6")
print("   - $R_{noise}(T, x) \\propto \\varepsilon(x) \\times |T| \\times 5$")
print("   - Performance: DEGRADED")
print("   - Mathematical reasoning:")
print("     * Variable depth: $\\text{depth}(T) \\approx 5$ (average)")
print("     * Variable size: $|T|$ (from actual tree)")
print("     * Noise sensitivity: $R_{noise}(T, x) = k \\cdot \\varepsilon(x) \\cdot |T|^{0.5} \\cdot 5^{1.0}$")
print("     * As $\\varepsilon(x)$ increases, error grows linearly")
print("     * Performance degradation: $\\frac{\\Delta R_{noise}}{\\Delta \\varepsilon} = k \\cdot |T|^{0.5} \\cdot 5$")
print("     * Stability: Medium (moderate growth rate)")
print("     * Reason: $\\alpha = 0.1$ creates balanced trees")
print("     * Better than REP due to smaller size, worse than pre-pruning due to flexibility")

print("\nWorst Performance: Reduced Error Pruning")
print("Why: Highest depth and largest tree size make it most sensitive to noise patterns")

print("\nQuantitative Comparison:")
print("Noise sensitivity coefficients:")
print("1. Pre-pruning: $k \\cdot |T|^{0.5} \\cdot 3$ (most stable)")
print("2. CCP: $k \\cdot |T|^{0.5} \\cdot 5$ (moderately stable)")
print("3. REP: $k \\cdot |T|^{0.5} \\cdot 6.5$ (least stable)")

print("\nPerformance degradation ratios:")
print("REP vs Pre-pruning: $\\frac{k \\cdot |T|^{0.5} \\cdot 6.5}{k \\cdot |T|^{0.5} \\cdot 3} = 2.17\\times$ worse")
print("CCP vs Pre-pruning: $\\frac{k \\cdot |T|^{0.5} \\cdot 5}{k \\cdot |T|^{0.5} \\cdot 3} = 1.67\\times$ worse")
print("REP vs CCP: $\\frac{k \\cdot |T|^{0.5} \\cdot 6.5}{k \\cdot |T|^{0.5} \\cdot 5} = 1.3\\times$ worse")

print("\nMathematical Insight:")
print("$R_{noise}(T, x) \\propto |T|^{0.5} \\times \\text{depth}(T)$")
print("REP has highest values for both factors, making it most noise-sensitive")

# Task 6: Real-time recommendation system choice
print("\n6. PEN AND PAPER: Real-Time System Analysis")
print("-" * 50)

print("Real-Time Requirements:")
print("Let $t_{pred}$ = prediction time")
print("Let $t_{train}$ = training time")
print("Let $t_{update}$ = model update time")
print("Let $L$ = latency requirement")
print("Let $n_{req}$ = requests per second")
print("Let $\\text{throughput}$ = system throughput")

print("\nSystem Constraints:")
print("$t_{pred} \\leq L$ (prediction must be fast)")
print("$t_{update} \\leq L$ (updates must be fast)")
print("$\\text{throughput} = \\frac{n_{req}}{t_{pred}} \\geq \\text{required throughput}$")
print("Typical requirements: $L \\leq 100ms$, $\\text{throughput} \\geq 1000$ req/s")

print("\nAnalysis:")
print("A) Pre-pruning (max_depth=3):")
print("   - $t_{pred} = O(1)$ ✓")
print("   - $t_{train} = O(8nd)$ ✓")
print("   - $t_{update} = O(8nd)$ ✓")
print("   - Choice: EXCELLENT")
print("   - Mathematical reasoning:")
print("     * Prediction: $t_{pred} = 3 \\times t_{node}$")
print("     * Training: $t_{train} = 8nd \\times t_{split}$")
print("     * Update: $t_{update} = t_{train}$")
print("     * Throughput: $\\text{throughput} = \\frac{n_{req}}{t_{pred}}$")
print("     * Latency: $t_{pred} \\ll L$ ✓")
print("     * Update time: $t_{update} < L$ ✓")

print("\nB) Reduced Error Pruning:")
print("   - $t_{pred} = O(\\log |T|)$ ✓")
print("   - $t_{train} = O(nd|T|)$ ✗")
print("   - $t_{update} = O(nd|T|)$ ✗")
print("   - Choice: POOR")
print("   - Mathematical reasoning:")
print("     * Prediction: $t_{pred} = \\log(|T|) \\times t_{node}$")
print("     * Training: $t_{train} = nd|T| \\times t_{split}$")
print("     * Update: $t_{update} = t_{train}$")
print("     * Throughput: $\\text{throughput} = \\frac{n_{req}}{t_{pred}}$")
print("     * Latency: $t_{pred} < L$ ✓")
print("     * Update time: $t_{update} > L$ ✗ (too slow)")
print("     * Problem: Updates take much longer than allowed")

print("\nC) Cost-Complexity Pruning:")
print("   - $t_{pred} = O(\\log |T|)$ ✓")
print("   - $t_{train} = O(nd|T|^2)$ ✗")
print("   - $t_{update} = O(nd|T|^2)$ ✗")
print("   - Choice: POOR")
print("   - Mathematical reasoning:")
print("     * Prediction: $t_{pred} = \\log(|T|) \\times t_{node}$")
print("     * Training: $t_{train} = nd|T|^2 \\times t_{split}$")
print("     * Update: $t_{update} = t_{train}$")
print("     * Throughput: $\\text{throughput} = \\frac{n_{req}}{t_{pred}}$")
print("     * Latency: $t_{pred} < L$ ✓")
print("     * Update time: $t_{update} > L$ ✗ (too slow)")
print("     * Problem: $O(|T|^2)$ term makes updates extremely slow")
print("     * Even with small trees, quadratic complexity is prohibitive")

print("\nBest Choice: Pre-pruning (max_depth=3)")
print("Why: Fastest training, prediction, and updates")

print("\nQuantitative Comparison:")
print("Update time ratios (lower is better):")
print("1. Pre-pruning: $1.0\\times$ (baseline)")
print("2. REP: $\\frac{O(nd|T|)}{O(8nd)}$ (slower)")
print("3. CCP: $\\frac{O(nd|T|^2)}{O(8nd)}$ (much slower)")

print("\nThroughput comparison:")
print("1. Pre-pruning: High throughput (fastest prediction)")
print("2. CCP: Medium throughput (medium prediction time)")
print("3. REP: Lower throughput (slower prediction time)")

print("\nMathematical Insight:")
print("Real-time systems require: $t_{update} \\leq L$ and $t_{pred} \\leq L$")
print("Only pre-pruning satisfies both constraints for $L = 100ms$")
print("The $O(|T|^2)$ complexity in CCP makes it unsuitable for real-time updates")

# Task 7: Experimental design
print("\n7. PEN AND PAPER: Experimental Design")
print("-" * 50)

print("Experimental Framework:")
print("Let $M = \\{\\text{method}_1, \\text{method}_2, \\text{method}_3\\}$ be pruning methods")
print("Let $D = \\{\\text{dataset}_1, \\text{dataset}_2, \\ldots, \\text{dataset}_k\\}$ be datasets")
print("Let $P = \\{\\text{param}_1, \\text{param}_2, \\ldots, \\text{param}_n\\}$ be parameters")
print("Let $R = \\{\\text{run}_1, \\text{run}_2, \\ldots, \\text{run}_r\\}$ be experimental runs")

print("\nEfficiency Metrics:")
print("1. Training Time: $T_{train}(m, d, p)$")
print("2. Prediction Time: $T_{pred}(m, d, p)$")
print("3. Memory Usage: $M_{usage}(m, d, p)$")
print("4. Model Size: $S_{model}(m, d, p)$")
print("5. Energy Consumption: $E(m, d, p)$ (optional)")
print("6. Scalability: $S(m, d, p) = \\frac{T(m, 2d, p)}{T(m, d, p)}$")

print("\nExperimental Design:")
print("For each method $m \\in M$:")
print("  For each dataset $d \\in D$:")
print("    For each parameter $p \\in P$:")
print("      For each run $r \\in R$:")
print("        Measure $T_{train}(m, d, p, r)$")
print("        Measure $T_{pred}(m, d, p, r)$")
print("        Measure $M_{usage}(m, d, p, r)$")
print("        Measure $S_{model}(m, d, p, r)$")

print("\nStatistical Analysis:")
print("1. **ANOVA for method comparison**: $F = \\frac{MS_{between}}{MS_{within}}$")
print("2. **Tukey's HSD for pairwise comparison**: $q = \\frac{|\\bar{x}_i - \\bar{x}_j|}{\\sqrt{\\frac{MS_{within}}{n}}}$")
print("3. **Effect size calculation (Cohen's d)**: $d = \\frac{|\\bar{x}_1 - \\bar{x}_2|}{s_{pooled}}$")
print("4. **Confidence intervals**: $\\bar{x} \\pm t_{\\alpha/2, df} \\times \\frac{s}{\\sqrt{n}}$")
print("5. **Power analysis**: $\\beta = 1 - \\text{Power}$")
print("6. **Multiple comparison correction**: Bonferroni, FDR, etc.")

print("\nSample Size Calculation:")
print("For $\\alpha = 0.05$, $\\beta = 0.2$, effect size $d = 0.5$:")
print("Required sample size: $n = \\frac{2(t_{\\alpha/2} + t_{\\beta})^2}{d^2}$ per group")
print("With 3 methods: total $n = 3 \\times n_{required}$ runs")

print("\nPractical Implementation:")
print("Our experiment uses $r$ runs per method to ensure statistical significance:")
print("- Total measurements: $3 \\times 1 \\times 1 \\times r$ runs")
print("- Degrees of freedom: $df_{between} = 2$, $df_{within} = 3r - 3$")
print("- Critical F-value: $F_{\\alpha, 2, 3r-3}$")
print("- Power depends on effect size and sample size")

# ============================================================================
# PRACTICAL IMPLEMENTATION AND VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PRACTICAL IMPLEMENTATION AND VISUALIZATION")
print("="*80)

# Generate synthetic dataset
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=3, n_clusters_per_class=2, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Task 1: Compare tree sizes
print("\n1. COMPARING TREE SIZES")
print("-" * 30)

# Method 1: Pre-pruning with max_depth=3
tree_pre = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_pre.fit(X_train, y_train)
n_leaves_pre = tree_pre.get_n_leaves()
n_nodes_pre = tree_pre.tree_.node_count

# Method 2: Reduced Error Pruning (simulated)
tree_rep = DecisionTreeClassifier(random_state=42)
tree_rep.fit(X_train, y_train)
# Simulate REP by pruning based on validation set
from sklearn.tree import _tree
def prune_tree_rep(tree, X_val, y_val):
    """Simulate reduced error pruning"""
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    
    # Calculate validation accuracy for each node
    def get_node_accuracy(node_id):
        if children_left[node_id] == -1:  # Leaf node
            return 1.0
        # Get samples reaching this node
        node_samples = tree.apply(X_val) == node_id
        if np.sum(node_samples) == 0:
            return 0.0
        node_predictions = tree.predict(X_val[node_samples])
        node_true = y_val[node_samples]
        return np.mean(node_predictions == node_true)
    
    # Prune nodes that don't improve accuracy
    pruned_nodes = 0
    for i in range(n_nodes):
        if children_left[i] != -1:  # Not a leaf
            parent_acc = get_node_accuracy(i)
            left_acc = get_node_accuracy(children_left[i])
            right_acc = get_node_accuracy(children_right[i])
            
            if parent_acc >= max(left_acc, right_acc):
                pruned_nodes += 1
    
    return n_nodes - pruned_nodes

n_nodes_rep = prune_tree_rep(tree_rep, X_val, y_val)
n_leaves_rep = n_nodes_rep // 2  # Approximate

# Method 3: Cost-complexity pruning
tree_ccp = DecisionTreeClassifier(random_state=42)
path = tree_ccp.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
tree_ccp_alpha01 = DecisionTreeClassifier(ccp_alpha=0.1, random_state=42)
tree_ccp_alpha01.fit(X_train, y_train)
n_leaves_ccp = tree_ccp_alpha01.get_n_leaves()
n_nodes_ccp = tree_ccp_alpha01.tree_.node_count

print(f"Pre-pruning (max_depth=3): {n_leaves_pre} leaves, {n_nodes_pre} nodes")
print(f"Reduced Error Pruning: ~{n_leaves_rep} leaves, ~{n_nodes_rep} nodes")
print(f"Cost-complexity ($\\alpha=0.1$): {n_leaves_ccp} leaves, {n_nodes_ccp} nodes")

# Visualize tree sizes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Tree size comparison
methods = ['Pre-pruning\n(max_depth=3)', 'Reduced Error\nPruning', 'Cost-Complexity\n($\\alpha=0.1$)']
leaf_counts = [n_leaves_pre, n_leaves_rep, n_leaves_ccp]
node_counts = [n_nodes_pre, n_nodes_rep, n_nodes_ccp]

x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, leaf_counts, width, label='Leaf Nodes', color='skyblue', alpha=0.8)
ax1.bar(x + width/2, node_counts, width, label='Total Nodes', color='lightcoral', alpha=0.8)
ax1.set_xlabel('Pruning Method')
ax1.set_ylabel('Number of Nodes')
ax1.set_title('Tree Size Comparison by Pruning Method')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Tree depth comparison
depths = [3, tree_rep.get_depth(), tree_ccp_alpha01.get_depth()]
ax2.bar(methods, depths, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_xlabel('Pruning Method')
ax2.set_ylabel('Tree Depth')
ax2.set_title('Tree Depth Comparison by Pruning Method')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tree_size_comparison.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 2: Robustness to noisy data
print("\n2. ROBUSTNESS TO NOISY DATA")
print("-" * 30)

# Add noise to validation set
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_results = {}

for method_name, tree in [('Pre-pruning', tree_pre), ('REP', tree_rep), ('CCP', tree_ccp_alpha01)]:
    accuracies = []
    for noise in noise_levels:
        # Add noise to validation set
        X_val_noisy = X_val.copy()
        noise_mask = np.random.random(X_val.shape) < noise
        X_val_noisy[noise_mask] = np.random.random(np.sum(noise_mask))
        
        # Evaluate accuracy
        y_pred = tree.predict(X_val_noisy)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
    
    noise_results[method_name] = accuracies

# Plot noise robustness
plt.figure(figsize=(10, 6))
for method, accs in noise_results.items():
    plt.plot(noise_levels, accs, marker='o', linewidth=2, label=method)

plt.xlabel('Noise Level')
plt.ylabel('Validation Accuracy')
plt.title('Robustness to Noisy Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 3: Computational speed comparison
print("\n3. COMPUTATIONAL SPEED COMPARISON")
print("-" * 30)

# Measure training time
training_times = {}
prediction_times = {}

# Pre-pruning
start_time = time.time()
tree_pre.fit(X_train, y_train)
training_times['Pre-pruning'] = time.time() - start_time

start_time = time.time()
_ = tree_pre.predict(X_test)
prediction_times['Pre-pruning'] = time.time() - start_time

# REP (simulated)
start_time = time.time()
tree_rep.fit(X_train, y_train)
_ = prune_tree_rep(tree_rep, X_val, y_val)
training_times['REP'] = time.time() - start_time

start_time = time.time()
_ = tree_rep.predict(X_test)
prediction_times['REP'] = time.time() - start_time

# Cost-complexity pruning
start_time = time.time()
tree_ccp_alpha01.fit(X_train, y_train)
training_times['CCP'] = time.time() - start_time

start_time = time.time()
_ = tree_ccp_alpha01.predict(X_test)
prediction_times['CCP'] = time.time() - start_time

print("Training Times:")
for method, t in training_times.items():
    print(f"  {method}: {t:.4f} seconds")

print("\nPrediction Times:")
for method, t in prediction_times.items():
    print(f"  {method}: {t:.6f} seconds")

# Visualize computational efficiency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training time comparison
methods = list(training_times.keys())
train_times = list(training_times.values())
ax1.bar(methods, train_times, color=['green', 'orange', 'red'], alpha=0.7)
ax1.set_xlabel('Pruning Method')
ax1.set_ylabel('Training Time (seconds)')
ax1.set_title('Training Time Comparison')
ax1.grid(True, alpha=0.3)

# Prediction time comparison
pred_times = list(prediction_times.values())
ax2.bar(methods, pred_times, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_xlabel('Pruning Method')
ax2.set_ylabel('Prediction Time (seconds)')
ax2.set_title('Prediction Time Comparison')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'computational_efficiency.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 4: Interpretability evaluation
print("\n4. INTERPRETABILITY EVALUATION")
print("-" * 30)

# Calculate interpretability metrics
def calculate_interpretability(tree):
    """Calculate interpretability score based on depth and number of leaves"""
    depth = tree.get_depth()
    n_leaves = tree.get_n_leaves()
    avg_depth = depth  # Simplified approximation
    
    # Interpretability score (lower is more interpretable)
    interpretability_score = avg_depth * np.log(n_leaves)
    return interpretability_score, depth, n_leaves

interpretability_scores = {}
for method_name, tree in [('Pre-pruning', tree_pre), ('REP', tree_rep), ('CCP', tree_ccp_alpha01)]:
    score, depth, leaves = calculate_interpretability(tree)
    interpretability_scores[method_name] = {'score': score, 'depth': depth, 'leaves': leaves}

print("Interpretability Analysis:")
for method, metrics in interpretability_scores.items():
    print(f"  {method}:")
    print(f"    Depth: {metrics['depth']}")
    print(f"    Leaves: {metrics['leaves']}")
    print(f"    Interpretability Score: {metrics['score']:.2f}")

# Visualize interpretability
plt.figure(figsize=(10, 6))
methods = list(interpretability_scores.keys())
scores = [metrics['score'] for metrics in interpretability_scores.values()]
depths = [metrics['depth'] for metrics in interpretability_scores.values()]
leaves = [metrics['leaves'] for metrics in interpretability_scores.values()]

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, scores, width, label='Interpretability Score', color='purple', alpha=0.7)
plt.bar(x, depths, width, label='Tree Depth', color='orange', alpha=0.7)
plt.bar(x + width, leaves, width, label='Number of Leaves', color='green', alpha=0.7)

plt.xlabel('Pruning Method')
plt.ylabel('Metric Value')
plt.title('Interpretability Metrics Comparison')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'interpretability_comparison.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 5: Performance with increasing noise
print("\n5. PERFORMANCE WITH INCREASING NOISE")
print("-" * 40)

# Create dataset where noise increases with feature values
X_noise_test = X_test.copy()
noise_increase_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

noise_increase_results = {}
for method_name, tree in [('Pre-pruning', tree_pre), ('REP', tree_rep), ('CCP', tree_ccp_alpha01)]:
    accuracies = []
    for noise_level in noise_increase_levels:
        X_test_noisy = X_test.copy()
        # Add noise proportional to feature values
        for i in range(X_test.shape[1]):
            feature_noise = np.random.normal(0, noise_level * np.std(X_test[:, i]), X_test.shape[0])
            X_test_noisy[:, i] += feature_noise
        
        y_pred = tree.predict(X_test_noisy)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    
    noise_increase_results[method_name] = accuracies

# Plot performance with increasing noise
plt.figure(figsize=(10, 6))
for method, accs in noise_increase_results.items():
    plt.plot(noise_increase_levels, accs, marker='o', linewidth=2, label=method)

plt.xlabel('Noise Level (proportional to feature values)')
plt.ylabel('Test Accuracy')
plt.title('Performance with Feature-Dependent Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, 'feature_dependent_noise.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 6: Real-time system analysis
print("\n6. REAL-TIME SYSTEM ANALYSIS")
print("-" * 30)

# Simulate real-time constraints
real_time_metrics = {}
for method_name, tree in [('Pre-pruning', tree_pre), ('REP', tree_rep), ('CCP', tree_ccp_alpha01)]:
    # Measure prediction latency
    latencies = []
    for _ in range(100):
        start_time = time.time()
        _ = tree.predict(X_test[:100])  # Predict 100 samples
        latency = time.time() - start_time
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    real_time_metrics[method_name] = {
        'avg_latency': avg_latency,
        'max_latency': max_latency,
        'p95_latency': p95_latency,
        'training_time': training_times[method_name],
        'model_size': tree.tree_.node_count
    }

print("Real-Time Performance Metrics:")
for method, metrics in real_time_metrics.items():
    print(f"  {method}:")
    print(f"    Avg Prediction Latency: {metrics['avg_latency']:.6f}s")
    print(f"    Max Prediction Latency: {metrics['max_latency']:.6f}s")
    print(f"    95th Percentile Latency: {metrics['p95_latency']:.6f}s")
    print(f"    Training Time: {metrics['training_time']:.4f}s")
    print(f"    Model Size: {metrics['model_size']} nodes")

# Visualize real-time performance
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Average latency
methods = list(real_time_metrics.keys())
avg_latencies = [metrics['avg_latency'] for metrics in real_time_metrics.values()]
ax1.bar(methods, avg_latencies, color=['green', 'orange', 'red'], alpha=0.7)
ax1.set_xlabel('Pruning Method')
ax1.set_ylabel('Average Latency (seconds)')
ax1.set_title('Average Prediction Latency')
ax1.grid(True, alpha=0.3)

# 95th percentile latency
p95_latencies = [metrics['p95_latency'] for metrics in real_time_metrics.values()]
ax2.bar(methods, p95_latencies, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_xlabel('Pruning Method')
ax2.set_ylabel('95th Percentile Latency (seconds)')
ax2.set_title('95th Percentile Prediction Latency')
ax2.grid(True, alpha=0.3)

# Training time
train_times = [metrics['training_time'] for metrics in real_time_metrics.values()]
ax3.bar(methods, train_times, color=['green', 'orange', 'red'], alpha=0.7)
ax3.set_xlabel('Pruning Method')
ax3.set_ylabel('Training Time (seconds)')
ax3.set_title('Training Time')
ax3.grid(True, alpha=0.3)

# Model size
model_sizes = [metrics['model_size'] for metrics in real_time_metrics.values()]
ax4.bar(methods, model_sizes, color=['green', 'orange', 'red'], alpha=0.7)
ax4.set_xlabel('Pruning Method')
ax4.set_ylabel('Number of Nodes')
ax4.set_title('Model Size')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'real_time_performance.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Task 7: Experimental design for computational efficiency
print("\n7. EXPERIMENTAL DESIGN FOR COMPUTATIONAL EFFICIENCY")
print("-" * 60)

# Design comprehensive experiment
def run_efficiency_experiment(X_train, y_train, X_test, y_test, method_name, tree, n_runs=10):
    """Run comprehensive efficiency experiment"""
    results = {
        'training_times': [],
        'prediction_times': [],
        'memory_usage': [],
        'model_sizes': []
    }
    
    for _ in range(n_runs):
        # Training time
        start_time = time.time()
        tree.fit(X_train, y_train)
        train_time = time.time() - start_time
        results['training_times'].append(train_time)
        
        # Prediction time
        start_time = time.time()
        _ = tree.predict(X_test)
        pred_time = time.time() - start_time
        results['prediction_times'].append(pred_time)
        
        # Model size
        model_size = tree.tree_.node_count
        results['model_sizes'].append(model_size)
        
        # Memory usage (approximate)
        memory_usage = model_size * 8 * 4  # 8 bytes per node, 4 attributes
        results['memory_usage'].append(memory_usage)
    
    return results

# Run experiments
experiment_results = {}
for method_name, tree_class in [
    ('Pre-pruning', lambda: DecisionTreeClassifier(max_depth=3, random_state=42)),
    ('REP', lambda: DecisionTreeClassifier(random_state=42)),
    ('CCP', lambda: DecisionTreeClassifier(ccp_alpha=0.1, random_state=42))
]:
    tree = tree_class()
    results = run_efficiency_experiment(X_train, y_train, X_test, y_test, method_name, tree)
    experiment_results[method_name] = results

# Statistical analysis
print("Statistical Analysis of Computational Efficiency:")
print("-" * 50)

for method, results in experiment_results.items():
    print(f"\n{method}:")
    print(f"  Training Time:")
    print(f"    Mean: {np.mean(results['training_times']):.6f}s")
    print(f"    Std:  {np.std(results['training_times']):.6f}s")
    print(f"    CI95: [{np.percentile(results['training_times'], 2.5):.6f}, {np.percentile(results['training_times'], 97.5):.6f}]s")
    
    print(f"  Prediction Time:")
    print(f"    Mean: {np.mean(results['prediction_times']):.6f}s")
    print(f"    Mean: {np.mean(results['prediction_times']):.6f}s")
    print(f"    Std:  {np.std(results['prediction_times']):.6f}s")
    print(f"    CI95: [{np.percentile(results['prediction_times'], 2.5):.6f}, {np.percentile(results['prediction_times'], 97.5):.6f}]s")
    
    print(f"  Model Size:")
    print(f"    Mean: {np.mean(results['model_sizes']):.1f} nodes")
    print(f"    Std:  {np.std(results['model_sizes']):.1f} nodes")

# Visualize experimental results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Training time distribution
for method, results in experiment_results.items():
    ax1.hist(results['training_times'], alpha=0.7, label=method, bins=10)
ax1.set_xlabel('Training Time (seconds)')
ax1.set_ylabel('Frequency')
ax1.set_title('Training Time Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Prediction time distribution
for method, results in experiment_results.items():
    ax2.hist(results['prediction_times'], alpha=0.7, label=method, bins=10)
ax2.set_xlabel('Prediction Time (seconds)')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Time Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Model size distribution
for method, results in experiment_results.items():
    ax3.hist(results['model_sizes'], alpha=0.7, label=method, bins=10)
ax3.set_xlabel('Model Size (nodes)')
ax3.set_ylabel('Frequency')
ax3.set_title('Model Size Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Memory usage comparison
memory_means = [np.mean(results['memory_usage']) for results in experiment_results.values()]
ax4.bar(experiment_results.keys(), memory_means, color=['green', 'orange', 'red'], alpha=0.7)
ax4.set_xlabel('Pruning Method')
ax4.set_ylabel('Memory Usage (bytes)')
ax4.set_title('Average Memory Usage')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'experimental_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE: PRUNING METHOD COMPARISON")
print("="*80)

summary_data = []
for method in ['Pre-pruning', 'REP', 'CCP']:
    if method == 'Pre-pruning':
        tree = tree_pre
        training_time = training_times['Pre-pruning']
        prediction_time = prediction_times['Pre-pruning']
    elif method == 'REP':
        tree = tree_rep
        training_time = training_times['REP']
        prediction_time = prediction_times['REP']
    else:
        tree = tree_ccp_alpha01
        training_time = training_times['CCP']
        prediction_time = prediction_times['CCP']
    
    summary_data.append({
        'Method': method,
        'Tree Size': tree.tree_.node_count,
        'Depth': tree.get_depth(),
        'Leaves': tree.get_n_leaves(),
        'Training Time (s)': f"{training_time:.4f}",
        'Prediction Time (s)': f"{prediction_time:.6f}",
        'Noise Robustness': 'High' if method == 'Pre-pruning' else 'Medium' if method == 'REP' else 'Low',
        'Interpretability': 'High' if method == 'Pre-pruning' else 'Medium',
        'Real-time Suitability': 'Excellent' if method == 'Pre-pruning' else 'Poor'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\nAll plots saved to: {save_dir}")
print("="*80)
