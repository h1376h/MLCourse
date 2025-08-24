import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pandas as pd
from scipy.special import comb

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 24: Random Forest Diversity Challenge Strategy")
print("=" * 70)

# Given parameters
n_patients = 500
n_features = 30
n_trees_max = 50
target_coverage = 0.80  # 80% of trees should use each feature

print(f"Medical Diagnosis Random Forest Parameters:")
print(f"- Number of patients: n_patients = {n_patients}")
print(f"- Number of medical features: n_features = {n_features}")
print(f"- Binary diagnosis: Healthy/Sick")
print(f"- Maximum trees due to computational limits: n_trees_max = {n_trees_max}")
print(f"- Target feature coverage: target_coverage = {target_coverage*100:.0f}%")

print("\n" + "="*70)
print("STEP 1: Calculate Optimal Number of Features per Split")
print("="*70)

print("Let m be the number of features per split.")
print("We need to find the optimal value of m that maximizes diversity.")

print("\nMathematical Analysis:")
print("1. For a single tree, the probability that a specific feature is selected:")
print("   P(feature selected) = m/n_features")
print(f"   P(feature selected) = m/{n_features}")

print("\n2. The probability that a feature is NOT selected in a single tree:")
print("   P(feature not selected) = 1 - m/n_features")
print(f"   P(feature not selected) = 1 - m/{n_features}")

print("\n3. The probability that a feature is never selected in any of the m splits:")
print("   P(feature never selected) = (1 - m/n_features)^m")
print(f"   P(feature never selected) = (1 - m/{n_features})^m")

print("\n4. The probability that a feature is used at least once in a tree:")
print("   P(feature used) = 1 - P(feature never selected)")
print(f"   P(feature used) = 1 - (1 - m/{n_features})^m")

print("\n5. For diversity, we want to maximize the variance in feature selection.")
print("   Variance = P(feature used) × (1 - P(feature used))")
print("   Variance = [1 - (1 - m/n_features)^m] × [(1 - m/n_features)^m]")

print("\n6. To find the optimal m, we can use common heuristics:")
print("   a) Square root rule: m = √n_features")
print(f"      m = √{n_features} = {np.sqrt(n_features):.2f} ≈ {int(np.sqrt(n_features))}")
print("   b) Logarithmic rule: m = log₂(n_features)")
print(f"      m = log₂({n_features}) = {np.log2(n_features):.2f} ≈ {int(np.log2(n_features))}")

# For medical diagnosis, we want some stability, so use sqrt rule
optimal_features = int(np.sqrt(n_features))
print(f"\n7. Recommendation: Use m = {optimal_features} (square root rule)")
print(f"   This provides good balance between diversity and stability")

# Calculate probability that a specific feature is selected for a split
p_feature_selected = optimal_features / n_features
print(f"\n8. With m = {optimal_features}:")
print(f"   P(feature selected) = {optimal_features}/{n_features} = {p_feature_selected:.3f}")

print("\n" + "="*70)
print("STEP 2: Calculate Required Trees for 80% Feature Coverage")
print("="*70)

print("We want each feature to be used in at least 80% of trees.")
print("Let T be the number of trees needed.")

print("\nMathematical Derivation:")
print("1. For a single tree, the probability a feature is used:")
print(f"   P(feature used in tree) = 1 - (1 - {optimal_features}/{n_features})^{optimal_features}")
p_used_in_tree = 1 - (1 - optimal_features/n_features) ** optimal_features
print(f"   P(feature used in tree) = 1 - (1 - {optimal_features/n_features:.3f})^{optimal_features}")
print(f"   P(feature used in tree) = 1 - {(1 - optimal_features/n_features)**optimal_features:.4f}")
print(f"   P(feature used in tree) = {p_used_in_tree:.4f}")

print("\n2. For T trees, let X be the number of trees that use a specific feature.")
print("   X follows a binomial distribution: X ~ Binomial(T, p_used_in_tree)")
print(f"   X ~ Binomial(T, {p_used_in_tree:.4f})")

print("\n3. We want P(X ≥ 0.8T) ≥ 0.80")
print("   This means: P(X ≥ 0.8T) ≥ 0.80")

print("\n4. Using the normal approximation for large T:")
print("   X ~ N(μ, σ²) where:")
print(f"   μ = T × {p_used_in_tree:.4f} = {p_used_in_tree:.4f}T")
print(f"   σ² = T × {p_used_in_tree:.4f} × (1 - {p_used_in_tree:.4f}) = T × {p_used_in_tree:.4f} × {1-p_used_in_tree:.4f}")
print(f"   σ² = T × {p_used_in_tree * (1-p_used_in_tree):.4f}")

print("\n5. Standardizing: Z = (X - μ)/σ")
print("   P(X ≥ 0.8T) = P(Z ≥ (0.8T - μ)/σ)")
print(f"   P(X ≥ 0.8T) = P(Z ≥ (0.8T - {p_used_in_tree:.4f}T)/√({p_used_in_tree * (1-p_used_in_tree):.4f}T))")
print(f"   P(X ≥ 0.8T) = P(Z ≥ (0.8 - {p_used_in_tree:.4f})T/√({p_used_in_tree * (1-p_used_in_tree):.4f}T))")
print(f"   P(X ≥ 0.8T) = P(Z ≥ ({0.8 - p_used_in_tree:.4f})T/√({p_used_in_tree * (1-p_used_in_tree):.4f}T))")

print("\n6. For 80% confidence, we need P(Z ≥ z) ≥ 0.80")
print("   This means P(Z ≤ z) ≤ 0.20, so z ≤ -0.84 (from standard normal table)")
print("   Therefore: ({0.8 - p_used_in_tree:.4f})T/√({p_used_in_tree * (1-p_used_in_tree):.4f}T) ≤ -0.84")

print("\n7. Solving for T:")
print(f"   ({0.8 - p_used_in_tree:.4f})T ≤ -0.84 × √({p_used_in_tree * (1-p_used_in_tree):.4f}T)")
print(f"   ({0.8 - p_used_in_tree:.4f})²T² ≥ (0.84)² × {p_used_in_tree * (1-p_used_in_tree):.4f}T")
print(f"   T ≥ (0.84)² × {p_used_in_tree * (1-p_used_in_tree):.4f} / ({0.8 - p_used_in_tree:.4f})²")

def calculate_required_trees_for_coverage(n_features, features_per_split, target_coverage):
    """Calculate minimum trees needed for target feature coverage"""
    
    # Probability a feature is selected for a split
    p_selected = features_per_split / n_features
    
    # For a feature to be used in a tree, it needs to be selected at least once
    # P(feature used in tree) = 1 - P(feature never selected)
    # P(feature never selected) = (1 - p_selected)^features_per_split
    
    p_used_in_tree = 1 - (1 - p_selected) ** features_per_split
    
    print(f"   P(feature used in tree) = {p_used_in_tree:.4f}")
    
    # For n trees, we want P(at least 80% of trees use the feature) >= 0.80
    # This is equivalent to P(at least 0.8*n trees use the feature) >= 0.80
    
    # Use binomial distribution approximation
    # We want P(X >= 0.8*n) >= 0.80 where X ~ Binomial(n, p_used_in_tree)
    
    # For large n, we can use normal approximation
    # X ~ N(n*p, n*p*(1-p))
    # P(X >= 0.8*n) = P(Z >= (0.8*n - n*p)/sqrt(n*p*(1-p))) >= 0.80
    
    # This gives us: (0.8*n - n*p)/sqrt(n*p*(1-p)) <= -0.84 (for 80% confidence)
    # Solving for n: n >= (0.84^2 * p*(1-p)) / (0.8 - p)^2
    
    if p_used_in_tree >= target_coverage:
        print(f"   Feature already has {p_used_in_tree*100:.1f}% chance of being used in a tree")
        print("   No additional trees needed for 80% coverage")
        return 1
    
    # Calculate required n using the formula above
    z_score = stats.norm.ppf(0.2)  # 80% confidence level
    numerator = (z_score**2) * p_used_in_tree * (1 - p_used_in_tree)
    denominator = (target_coverage - p_used_in_tree)**2
    
    required_trees = int(np.ceil(numerator / denominator))
    
    return max(required_trees, 1)

# Calculate for our optimal features per split
print("\n8. Calculating required trees:")
required_trees = calculate_required_trees_for_coverage(n_features, optimal_features, target_coverage)
print(f"\n   Required trees for {target_coverage*100:.0f}% feature coverage: T = {required_trees}")

if required_trees > n_trees_max:
    print(f"   WARNING: Required trees ({required_trees}) exceeds maximum ({n_trees_max})")
    print("   Need to adjust strategy or accept lower coverage")
else:
    print(f"   ✓ Required trees ({required_trees}) within computational limits ({n_trees_max})")

print("\n" + "="*70)
print("STEP 3: Design Feature Sampling Strategy")
print("="*70)

print("We need to design a strategy that ensures rare but important features aren't ignored.")
print("Let's analyze different approaches:")

print("\nStrategy 1: Balanced Random Sampling")
print(f"- Features per split: m = {optimal_features}")
print(f"- Random selection from all n_features = {n_features} features")
print("- Expected feature usage: uniform distribution")
print("- Mathematical expectation: E[feature usage] = T × P(feature used in tree)")
print(f"  E[feature usage] = T × {1 - (1 - optimal_features/n_features)**optimal_features:.4f}")

print("\nStrategy 2: Stratified Sampling by Feature Importance")
print("- Divide features into importance tiers:")
print("  * Critical features (10%): Always include in split")
print("  * Important features (30%): Higher probability of selection")
print("  * Standard features (60%): Standard random selection")

print("\nMathematical formulation for stratified sampling:")
critical_ratio = 0.1
important_ratio = 0.3
standard_ratio = 0.6

n_critical = int(critical_ratio * n_features)
n_important = int(important_ratio * n_features)
n_standard = n_features - n_critical - n_important

print(f"- n_critical = {critical_ratio} × {n_features} = {n_critical}")
print(f"- n_important = {important_ratio} × {n_features} = {n_features - n_critical - n_standard}")
print(f"- n_standard = {n_features} - {n_critical} - {n_features - n_critical - n_standard} = {n_standard}")

print("\nProbability calculations:")
print("- P(critical feature selected) = 1.0 (always included)")
print(f"- P(important feature selected) = min(1.0, 2 × {optimal_features}/{n_features}) = min(1.0, {2*optimal_features/n_features:.3f})")
print(f"- P(standard feature selected) = {optimal_features}/{n_features} = {optimal_features/n_features:.3f}")

print("\nStrategy 3: Adaptive Sampling")
print("- Track feature usage across trees")
print("- Boost probability of underused features")
print("- Maintain diversity while ensuring coverage")

# Calculate expected feature usage for different strategies
def calculate_expected_feature_usage(n_trees, n_features, features_per_split, strategy="uniform"):
    """Calculate expected number of trees each feature appears in"""
    
    if strategy == "uniform":
        # For uniform sampling, each feature has equal probability
        p_used_in_tree = 1 - (1 - features_per_split/n_features) ** features_per_split
        expected_usage = n_trees * p_used_in_tree
        
    elif strategy == "stratified":
        # Simplified stratified approach
        critical_features = int(0.1 * n_features)  # 10% critical
        important_features = int(0.3 * n_features)  # 30% important
        standard_features = n_features - critical_features - important_features
        
        # Critical features: always used
        # Important features: 2x probability
        # Standard features: normal probability
        
        p_critical = 1.0
        p_important = min(1.0, 2 * features_per_split / n_features)
        p_standard = features_per_split / n_features
        
        expected_usage = (critical_features * p_critical + 
                         important_features * p_important + 
                         standard_features * p_standard) * n_trees / n_features
        
    return expected_usage

# Calculate for different strategies
trees_to_test = [10, 20, 30, 40, 50]
strategies = ["uniform", "stratified"]

print(f"\nExpected Feature Usage Analysis:")
print(f"{'Trees':<8} {'Uniform':<12} {'Stratified':<12}")
print("-" * 35)

for n_trees in trees_to_test:
    uniform_usage = calculate_expected_feature_usage(n_trees, n_features, optimal_features, "uniform")
    stratified_usage = calculate_expected_feature_usage(n_trees, n_features, optimal_features, "stratified")
    print(f"{n_trees:<8} {uniform_usage:<12.1f} {stratified_usage:<12.1f}")

print("\n" + "="*70)
print("STEP 4: Trade-off Analysis")
print("="*70)

print("Trade-offs between diversity strategy and individual tree performance:")
print()

print("1. Feature Sampling vs. Tree Performance:")
print("   - Fewer features per split (smaller m) → Higher diversity, lower individual tree accuracy")
print("   - More features per split (larger m) → Lower diversity, higher individual tree accuracy")
print(f"   - Our choice (m = {optimal_features}): Balanced approach")

print("\nMathematical analysis:")
print("   - Individual tree accuracy ∝ m (more features = better splits)")
print("   - Tree diversity ∝ 1/m (fewer features = more randomness)")
print("   - Optimal m balances these competing objectives")

print("\n2. Number of Trees vs. Computational Cost:")
print("   - More trees (larger T) → Higher diversity, higher computational cost")
print("   - Fewer trees (smaller T) → Lower diversity, lower computational cost")
print(f"   - Our constraint: Maximum T = {n_trees_max} trees")

print("\nMathematical relationship:")
print("   - Computational cost ∝ T")
print("   - Diversity ∝ √T (diminishing returns)")
print("   - Optimal T depends on computational budget")

print("\n3. Feature Coverage vs. Randomness:")
print("   - Higher coverage → More stable predictions, less randomness")
print("   - Lower coverage → More random predictions, potentially higher diversity")

print("\n4. Quantitative Diversity Analysis:")
# Calculate diversity metrics
def calculate_diversity_metrics(n_trees, n_features, features_per_split):
    """Calculate various diversity metrics"""
    
    # 1. Expected number of unique features used
    p_feature_used = 1 - (1 - features_per_split/n_features) ** features_per_split
    expected_unique_features = n_features * (1 - (1 - p_feature_used)**n_trees)
    
    # 2. Feature overlap between trees
    # Probability two trees share a specific feature
    p_shared_feature = p_feature_used**2
    expected_shared_features = n_features * p_shared_feature
    
    # 3. Tree independence measure
    # Lower overlap = higher independence = higher diversity
    independence_score = 1 - (expected_shared_features / n_features)
    
    return {
        'expected_unique_features': expected_unique_features,
        'expected_shared_features': expected_shared_features,
        'independence_score': independence_score
    }

diversity_metrics = calculate_diversity_metrics(n_trees_max, n_features, optimal_features)
print(f"   - Expected unique features used: {diversity_metrics['expected_unique_features']:.1f}")
print(f"   - Independence score: {diversity_metrics['independence_score']:.3f}")

print("\nMathematical interpretation:")
print(f"   - Independence score = 1 - {diversity_metrics['expected_shared_features']:.3f}/{n_features} = {diversity_metrics['independence_score']:.3f}")
print("   - Higher independence score means more diverse trees")
print("   - Independence score ranges from 0 (identical trees) to 1 (completely independent trees)")

print("\n" + "="*70)
print("STEP 5: General Formula Derivation")
print("="*70)

print("General Formula for Expected Unique Features:")
print()

print("Let's derive the formula step by step using general variables:")
print()

print("1. For a single tree:")
print(f"   - Features per split: m = {optimal_features}")
print(f"   - Total features: n = {n_features}")
print(f"   - Probability a specific feature is selected: p = m/n = {optimal_features}/{n_features} = {optimal_features/n_features:.3f}")

print("\n2. Probability a feature is used in a tree:")
print("   - P(feature used) = 1 - P(feature never selected)")
print(f"   - P(feature never selected) = (1 - p)^m = (1 - {optimal_features/n_features:.3f})^{optimal_features} = {(1 - optimal_features/n_features)**optimal_features:.4f}")
print(f"   - P(feature used) = 1 - {(1 - optimal_features/n_features)**optimal_features:.4f} = {1 - (1 - optimal_features/n_features)**optimal_features:.4f}")

print("\n3. For T trees:")
print("   - P(feature used in at least one tree) = 1 - P(feature never used)")
print(f"   - P(feature never used) = (1 - {1 - (1 - optimal_features/n_features)**optimal_features:.4f})^T")
print(f"   - P(feature used in at least one tree) = 1 - (1 - {1 - (1 - optimal_features/n_features)**optimal_features:.4f})^T")

print("\n4. Expected number of unique features:")
print("   - E[unique features] = n × P(feature used in at least one tree)")
print(f"   - E[unique features] = {n_features} × (1 - (1 - {1 - (1 - optimal_features/n_features)**optimal_features:.4f})^T)")

print("\n5. General formula:")
print("   E[unique features] = n × (1 - (1 - p_used)^T)")
print("   where:")
print("   - n = total number of features")
print("   - p_used = probability a feature is used in a single tree")
print("   - T = number of trees")
print("   - p_used = 1 - (1 - m/n)^m")
print("   - m = features per split")

print("\n6. Mathematical properties:")
print("   - As T → ∞, E[unique features] → n (all features eventually used)")
print("   - As m → n, p_used → 1 (all features used in every tree)")
print("   - As m → 1, p_used → 1/n (minimal feature usage per tree)")

print("\n7. Rate of convergence:")
print("   - The rate at which E[unique features] approaches n depends on p_used")
print("   - Higher p_used means faster convergence")
print("   - Lower p_used means slower convergence but higher diversity")

# Now let's create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Diversity Strategy Analysis', fontsize=16, fontweight='bold')

# Plot 1: Feature usage vs number of trees
ax1 = axes[0, 0]
trees_range = np.arange(1, n_trees_max + 1)
uniform_usage = [calculate_expected_feature_usage(t, n_features, optimal_features, "uniform") for t in trees_range]
stratified_usage = [calculate_expected_feature_usage(t, n_features, optimal_features, "stratified") for t in trees_range]

ax1.plot(trees_range, uniform_usage, 'b-', linewidth=2, label='Uniform Sampling')
ax1.plot(trees_range, stratified_usage, 'r--', linewidth=2, label='Stratified Sampling')
ax1.axhline(y=target_coverage * n_trees_max, color='g', linestyle=':', linewidth=2, label=f'{target_coverage*100:.0f}% Coverage Target')
ax1.set_xlabel('Number of Trees (T)')
ax1.set_ylabel('Expected Feature Usage')
ax1.set_title('Feature Usage vs Number of Trees')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Diversity metrics vs features per split
ax2 = axes[0, 1]
features_range = np.arange(1, min(21, n_features + 1))
diversity_scores = []
independence_scores = []

for m in features_range:
    metrics = calculate_diversity_metrics(n_trees_max, n_features, m)
    diversity_scores.append(metrics['expected_unique_features'])
    independence_scores.append(metrics['independence_score'])

ax2.plot(features_range, diversity_scores, 'g-', linewidth=2, label='Expected Unique Features')
ax2.axvline(x=optimal_features, color='r', linestyle='--', linewidth=2, label=f'Optimal: m = {optimal_features}')
ax2.set_xlabel('Features per Split (m)')
ax2.set_ylabel('Expected Unique Features')
ax2.set_title('Diversity vs Features per Split')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Independence score vs features per split
ax3 = axes[1, 0]
ax3.plot(features_range, independence_scores, 'm-', linewidth=2, label='Independence Score')
ax3.axvline(x=optimal_features, color='r', linestyle='--', linewidth=2, label=f'Optimal: m = {optimal_features}')
ax3.set_xlabel('Features per Split (m)')
ax3.set_ylabel('Independence Score')
ax3.set_title('Tree Independence vs Features per Split')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Coverage probability vs number of trees
ax4 = axes[1, 1]
coverage_probabilities = []
for t in trees_range:
    p_used = 1 - (1 - optimal_features/n_features) ** optimal_features
    coverage_prob = 1 - (1 - p_used) ** t
    coverage_probabilities.append(coverage_prob)

ax4.plot(trees_range, coverage_probabilities, 'c-', linewidth=2, label='Coverage Probability')
ax4.axhline(y=target_coverage, color='g', linestyle=':', linewidth=2, label=f'{target_coverage*100:.0f}% Target')
ax4.axvline(x=required_trees, color='r', linestyle='--', linewidth=2, label=f'Required: T = {required_trees}')
ax4.set_xlabel('Number of Trees (T)')
ax4.set_ylabel('Coverage Probability')
ax4.set_title('Feature Coverage vs Number of Trees')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'random_forest_diversity_analysis.png'), dpi=300, bbox_inches='tight')

# Create additional detailed plot
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 5: Feature importance distribution simulation
ax1.set_title('Simulated Feature Importance Distribution')
feature_importance = np.random.exponential(scale=1.0, size=n_features)
feature_importance = feature_importance / np.sum(feature_importance)  # Normalize
feature_indices = np.arange(n_features)

ax1.bar(feature_indices, feature_importance, alpha=0.7, color='skyblue', edgecolor='navy')
ax1.set_xlabel('Feature Index')
ax1.set_ylabel('Importance Weight')
ax1.set_title('Simulated Medical Feature Importance')
ax1.grid(True, alpha=0.3)

# Plot 6: Tree diversity heatmap
ax2.set_title('Tree Diversity Heatmap')
# Simulate feature usage across trees
n_trees_vis = min(20, n_trees_max)  # Show first 20 trees for visualization
feature_usage_matrix = np.zeros((n_features, n_trees_vis))

for tree in range(n_trees_vis):
    # Randomly select features for this tree
    selected_features = np.random.choice(n_features, size=optimal_features, replace=False)
    feature_usage_matrix[selected_features, tree] = 1

im = ax2.imshow(feature_usage_matrix, cmap='Blues', aspect='auto')
ax2.set_xlabel('Tree Index')
ax2.set_ylabel('Feature Index')
ax2.set_title(f'Feature Usage Across {n_trees_vis} Trees\n(m = {optimal_features} features per split)')
plt.colorbar(im, ax=ax2, label='Feature Used (1) or Not (0)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance_and_usage.png'), dpi=300, bbox_inches='tight')

print(f"Visualizations saved to: {save_dir}")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"1. Optimal features per split: m = {optimal_features} (using sqrt rule)")
print(f"2. Required trees for 80% coverage: T = {required_trees}")
print(f"3. Recommended strategy: {'Stratified sampling' if required_trees <= n_trees_max else 'Adaptive sampling'}")
print(f"4. Expected unique features with {n_trees_max} trees: {calculate_diversity_metrics(n_trees_max, n_features, optimal_features)['expected_unique_features']:.1f}")
print(f"5. Tree independence score: {calculate_diversity_metrics(n_trees_max, n_features, optimal_features)['independence_score']:.3f}")

if required_trees <= n_trees_max:
    print(f"\n✓ Solution is feasible within computational constraints!")
else:
    print(f"\n⚠ Solution requires {required_trees - n_trees_max} additional trees beyond limit")
    print("Consider: reducing features per split, using stratified sampling, or accepting lower coverage")

print("\nMathematical Insights:")
print("- The square root rule (m = √n) provides optimal balance for classification")
print("- Feature coverage can be achieved with surprisingly few trees when m is well-chosen")
print("- Tree diversity is maximized when individual trees are sufficiently different")
print("- The independence score quantifies the ensemble's diversity mathematically")
