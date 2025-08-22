import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_3_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 1: FEATURE SELECTION ANALYSIS")
print("=" * 80)

# Given information
features = ['A', 'B', 'C', 'D']
n_features = len(features)

print(f"\nGiven dataset with {n_features} features: {', '.join(features)}")
print("- Features A and B: weak predictors (0.20, 0.30)")
print("- Features C and D: strong predictors (0.80, 0.75)")
print("- Performance calculations follow specific formulas based on feature types")

# 1. Calculate the number of possible feature subsets
print("\n" + "="*50)
print("1. CALCULATING POSSIBLE FEATURE SUBSETS")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)
print(f"Given: n = {n_features} features")
print("Formula: Total subsets = 2^n")
print("Reasoning: Each feature can be either included (1) or excluded (0)")
print("          This gives us 2 choices per feature")
print(f"Calculation: 2^{n_features} = {2**n_features}")

total_subsets = 2**n_features
print(f"\nTotal number of possible feature subsets = {total_subsets}")

print("\nBREAKDOWN BY SUBSET SIZE:")
print("=" * 40)
# Generate all possible subsets with detailed explanation
all_subsets = []
for r in range(n_features + 1):
    subsets = list(itertools.combinations(features, r))
    all_subsets.extend(subsets)
    
    # Calculate combinations formula
    from math import factorial
    if r == 0:
        formula_result = 1
        formula_str = "C(4,0) = 4!/(0!×4!) = 1/1 = 1"
    else:
        formula_result = factorial(n_features) // (factorial(r) * factorial(n_features - r))
        formula_str = f"C({n_features},{r}) = {n_features}!/({r}!×{n_features-r}!) = {factorial(n_features)}/({factorial(r)}×{factorial(n_features-r)}) = {formula_result}"
    
    print(f"Subsets with {r} features: {formula_str}")
    print(f"  Result: {len(subsets)} subsets")

print(f"\nVerification: 1 + 4 + 6 + 4 + 1 = {sum([1, 4, 6, 4, 1])} = 2^4 = {2**4} ✓")

print(f"\nALL {len(all_subsets)} SUBSETS:")
print("=" * 40)
for i, subset in enumerate(all_subsets):
    subset_str = "{" + ", ".join(subset) + "}" if subset else "{}"
    subset_size = len(subset)
    print(f"{i+1:2d}. {subset_str:<12} (size: {subset_size})")

# 2. Univariate selection analysis
print("\n" + "="*50)
print("2. UNIVARIATE FEATURE SELECTION")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)

# Individual correlations (given in the problem)
correlations = {
    'A': 0.2,   # Weak predictor
    'B': 0.3,   # Weak predictor  
    'C': 0.8,   # Strong predictor
    'D': 0.75   # Strong predictor
}

print("Given individual feature correlations with target:")
for feature, corr in correlations.items():
    print(f"  Feature {feature}: r = {corr:.2f}")

print("\nUnivariate Selection Process:")
print("1. Evaluate each feature independently")
print("2. Rank features by individual correlation strength") 
print("3. Select top k features (k=2 in this case)")

print("\nRanking features by correlation:")
# Sort by correlation strength with detailed explanation
feature_ranking = []
for feature, corr in correlations.items():
    feature_ranking.append((feature, corr))

# Sort in descending order
feature_ranking.sort(key=lambda x: x[1], reverse=True)

for i, (feature, corr) in enumerate(feature_ranking, 1):
    print(f"  {i}. Feature {feature}: {corr:.2f}")

print(f"\nTop 2 features selected:")
print(f"  1st choice: {feature_ranking[0][0]} (correlation = {feature_ranking[0][1]:.2f})")
print(f"  2nd choice: {feature_ranking[1][0]} (correlation = {feature_ranking[1][1]:.2f})")

print(f"\nUnivariate selection result: {{{feature_ranking[0][0]}, {feature_ranking[1][0]}}}")
print("\nReasoning: Univariate methods evaluate features independently,")
print("          selecting those with highest individual predictive power.")
print("          This approach CANNOT detect feature interactions or synergy.")

# 3. Multivariate selection analysis
print("\n" + "="*50)
print("3. MULTIVARIATE FEATURE SELECTION")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)

print("Given Formulas:")
print("1. Single Features: P = individual correlation")
print("2. Two weak features (A,B): P = r_A + r_B + (r_A × r_B × 7.5)")
print("3. Weak + Strong: P = max(r_weak, r_strong) + 0.02")
print("4. Two strong features (C,D): P = max(r_C, r_D) + 0.01")
print("5. Three+ features: P = best two-feature combination")

print(f"\nGiven correlations: A={correlations['A']}, B={correlations['B']}, C={correlations['C']}, D={correlations['D']}")

def calculate_performance_detailed(subset_str, show_calculation=True):
    """Calculate performance with detailed step-by-step explanation"""
    if subset_str == '{}':
        if show_calculation:
            print(f"\n{subset_str}: Empty set → P = 0.00")
        return 0.0
    
    # Parse subset
    features_in_subset = subset_str.replace('{', '').replace('}', '').split(', ') if subset_str != '{}' else []
    
    if len(features_in_subset) == 1:
        # Single features: performance equals individual correlation
        feature = features_in_subset[0]
        performance = correlations[feature]
        if show_calculation:
            print(f"\n{subset_str}: Single feature → P = r_{feature} = {performance:.2f}")
        return performance
    
    elif len(features_in_subset) == 2:
        f1, f2 = features_in_subset
        if (f1 == 'A' and f2 == 'B') or (f1 == 'B' and f2 == 'A'):
            # Two weak features: special synergy formula
            r_A, r_B = correlations['A'], correlations['B']
            synergy_factor = 7.5
            synergy_term = r_A * r_B * synergy_factor
            performance = r_A + r_B + synergy_term
            if show_calculation:
                print(f"\n{subset_str}: Two weak features (synergy formula)")
                print(f"  P = r_A + r_B + (r_A × r_B × synergy_factor)")
                print(f"  P = {r_A} + {r_B} + ({r_A} × {r_B} × {synergy_factor})")
                print(f"  P = {r_A} + {r_B} + {synergy_term:.3f}")
                print(f"  P = {performance:.2f}")
            return performance
        elif (f1 in ['A', 'B'] and f2 in ['C', 'D']) or (f1 in ['C', 'D'] and f2 in ['A', 'B']):
            # Weak + Strong combination
            r1, r2 = correlations[f1], correlations[f2]
            max_corr = max(r1, r2)
            performance = max_corr + 0.02
            if show_calculation:
                print(f"\n{subset_str}: Weak + Strong combination")
                print(f"  P = max(r_{f1}, r_{f2}) + 0.02")
                print(f"  P = max({r1}, {r2}) + 0.02")
                print(f"  P = {max_corr} + 0.02 = {performance:.2f}")
            return performance
        elif f1 in ['C', 'D'] and f2 in ['C', 'D']:
            # Two strong features: minimal improvement due to redundancy
            r1, r2 = correlations[f1], correlations[f2]
            max_corr = max(r1, r2)
            performance = max_corr + 0.01
            if show_calculation:
                print(f"\n{subset_str}: Two strong features (redundancy)")
                print(f"  P = max(r_{f1}, r_{f2}) + 0.01")
                print(f"  P = max({r1}, {r2}) + 0.01")
                print(f"  P = {max_corr} + 0.01 = {performance:.2f}")
            return performance
    
    else:
        # Three or more features: performance plateaus at best two-feature combination
        if show_calculation:
            print(f"\n{subset_str}: Three+ features (plateau rule)")
            print("  Finding best two-feature combination:")
        
        best_two_perf = 0
        best_pair = ""
        for i in range(len(features_in_subset)):
            for j in range(i+1, len(features_in_subset)):
                two_subset = '{' + features_in_subset[i] + ', ' + features_in_subset[j] + '}'
                two_perf = calculate_performance_detailed(two_subset, show_calculation=False)
                if show_calculation:
                    print(f"    {two_subset}: {two_perf:.2f}")
                if two_perf > best_two_perf:
                    best_two_perf = two_perf
                    best_pair = two_subset
        
        if show_calculation:
            print(f"  Best pair: {best_pair} with P = {best_two_perf:.2f}")
            print(f"  Therefore: P = {best_two_perf:.2f}")
        return best_two_perf

print("\nDETAILED CALCULATIONS FOR ALL SUBSETS:")
print("=" * 60)

# Calculate all subset performances with detailed explanations
subset_performance = {}
for subset in all_subsets:
    subset_str = "{" + ", ".join(subset) + "}" if subset else "{}"
    performance = calculate_performance_detailed(subset_str, show_calculation=True)
    subset_performance[subset_str] = performance

print("Performance scores for all feature combinations:")
print("=" * 80)
print(f"{'Feature Subset':<20} {'Performance':<15} {'Cost':<10} {'Notes'}")
print("=" * 80)

# Print formatted table
for subset, score in subset_performance.items():
    # Count actual features in the subset
    if subset == '{}':
        cost = 0
    else:
        features_in_subset = subset.replace('{', '').replace('}', '').split(', ')
        cost = len(features_in_subset) * 5
    if subset == '{}':
        notes = "Empty set"
    elif len(subset.replace('{', '').replace('}', '').replace(',', '').strip()) == 1:
        if subset in ['{A}', '{B}']:
            notes = "Single weak predictor"
        else:
            notes = "Single strong predictor"
    elif subset == '{A, B}':
        notes = "Two weak predictors combined"
    elif subset in ['{A, C}', '{A, D}', '{B, C}', '{B, D}']:
        notes = "Weak + strong predictor"
    elif subset == '{C, D}':
        notes = "Two strong predictors"
    elif subset in ['{A, B, C}', '{A, B, D}']:
        notes = "Optimal + additional feature"
    elif subset in ['{A, C, D}', '{B, C, D}']:
        notes = "Mixed combination"
    else:
        notes = "All features"
    
    print(f"{subset:<20} {score:<15.2f} ${cost:<9} {notes}")

print("=" * 80)

# Find optimal subset
optimal_subset = max(subset_performance.items(), key=lambda x: x[1])
print(f"\nOptimal subset: {optimal_subset[0]} with performance {optimal_subset[1]:.2f}")

print("Why? Features A and B together provide the highest performance (0.95) at the lowest cost ($10).")
print("Adding more features doesn't improve performance but increases cost.")

# 4. Advantage of multivariate methods
print("\n" + "="*50)
print("4. ADVANTAGE OF MULTIVARIATE METHODS")
print("="*50)

print("Main advantage: Multivariate methods can capture feature interactions and synergy.")
print("- Univariate selection would pick C and D (strong individual predictors)")
print("- But A and B combined provide much better performance (0.95 vs 0.81)")
print("- Multivariate methods can identify this synergistic relationship")

# 5. Search space comparison
print("\n" + "="*50)
print("5. SEARCH SPACE SIZE COMPARISON")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)

print("Formula: Search space size = 2^n")
print("Reasoning: Each feature can be included (1) or excluded (0)")

print("\nFor 4 features:")
search_space_4 = 2**4
print(f"  2^4 = 2 × 2 × 2 × 2 = {search_space_4} subsets")

print("\nFor 10 features:")
search_space_10 = 2**10
powers_of_2 = [2**i for i in range(1, 11)]
print(f"  2^10 = {' × '.join(['2'] * 10)}")
print(f"  2^10 = {search_space_10} subsets")

ratio = search_space_10 / search_space_4
print(f"\nComparison:")
print(f"  Ratio = 2^10 / 2^4 = 2^(10-4) = 2^6 = {int(ratio)}")
print(f"  The 10-feature problem has {int(ratio)}× larger search space")

print(f"\nExponential growth demonstration:")
for i in range(4, 11):
    size = 2**i
    print(f"  {i} features: 2^{i} = {size:,} subsets")

print(f"\nComputational implications:")
print(f"  4 features: manageable ({search_space_4} evaluations)")
print(f"  10 features: challenging ({search_space_10:,} evaluations)")
print(f"  20 features: 2^20 = {2**20:,} evaluations (impractical for exhaustive search)")

# 6. Budget constraint analysis
print("\n" + "="*50)
print("6. BUDGET CONSTRAINT ANALYSIS")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)

feature_cost = 5
budget = 15

print("Given constraints:")
print(f"  Feature cost: ${feature_cost} per feature")
print(f"  Budget limit: ${budget}")

print(f"\nMaximum affordable features:")
max_features = budget // feature_cost
print(f"  Max features = Budget ÷ Cost per feature")
print(f"  Max features = ${budget} ÷ ${feature_cost} = {max_features} features")

print(f"\nAnalyzing all {len(all_subsets)} subsets:")

# Categorize subsets by cost
cost_categories = {}
valid_subsets = []
total_cost_all = 0

for subset in all_subsets:
    subset_cost = len(subset) * feature_cost
    total_cost_all += subset_cost
    
    # Categorize by number of features
    num_features = len(subset)
    if num_features not in cost_categories:
        cost_categories[num_features] = []
    cost_categories[num_features].append(subset)
    
    if subset_cost <= budget:
        valid_subsets.append(subset)

print(f"\nCost breakdown by subset size:")
for num_features in sorted(cost_categories.keys()):
    subsets_in_category = cost_categories[num_features]
    cost = num_features * feature_cost
    valid_count = len([s for s in subsets_in_category if len(s) * feature_cost <= budget])
    total_count = len(subsets_in_category)
    
    print(f"  {num_features} features: ${cost:2d} each, {total_count} subsets, {valid_count} within budget")

print(f"\nValid subsets within budget (≤${budget}):")
affordable_count = 0
for i, subset in enumerate(all_subsets):
    subset_cost = len(subset) * feature_cost
    if subset_cost <= budget:
        affordable_count += 1
        subset_str = "{" + ", ".join(subset) + "}" if subset else "{}"
        print(f"  {affordable_count:2d}. {subset_str:<15} Cost: ${subset_cost}")

print(f"\nSummary:")
print(f"  Total possible subsets: {len(all_subsets)}")
print(f"  Affordable subsets: {len(valid_subsets)}")
print(f"  Percentage affordable: {len(valid_subsets)/len(all_subsets)*100:.1f}%")

# Calculate total cost
print(f"\nTotal cost calculation:")
for num_features in sorted(cost_categories.keys()):
    count = len(cost_categories[num_features])
    cost_per_subset = num_features * feature_cost
    total_for_category = count * cost_per_subset
    print(f"  {count} subsets × ${cost_per_subset} = ${total_for_category}")

print(f"\nTotal cost of all possible subsets: ${total_cost_all}")

# 7. Interaction strength calculation
print("\n" + "="*50)
print("7. INTERACTION STRENGTH CALCULATION")
print("="*50)

print("STEP-BY-STEP CALCULATION:")
print("=" * 40)

# Get values from our calculations
corr_A = correlations['A']
corr_B = correlations['B']
combined_corr = subset_performance['{A, B}']

print("Given formula:")
print("  Interaction = Combined_Correlation - max(Individual_Correlations) - 0.1 × min(Individual_Correlations)")

print(f"\nInput values:")
print(f"  Individual correlation A: r_A = {corr_A:.2f}")
print(f"  Individual correlation B: r_B = {corr_B:.2f}")
print(f"  Combined correlation A,B: r_AB = {combined_corr:.2f}")

print(f"\nStep 1: Identify max and min individual correlations")
max_individual = max(corr_A, corr_B)
min_individual = min(corr_A, corr_B)
print(f"  max(r_A, r_B) = max({corr_A:.2f}, {corr_B:.2f}) = {max_individual:.2f}")
print(f"  min(r_A, r_B) = min({corr_A:.2f}, {corr_B:.2f}) = {min_individual:.2f}")

print(f"\nStep 2: Calculate penalty term")
penalty_term = 0.1 * min_individual
print(f"  Penalty = 0.1 × min(Individual_Correlations)")
print(f"  Penalty = 0.1 × {min_individual:.2f} = {penalty_term:.3f}")

print(f"\nStep 3: Apply interaction formula")
print(f"  Interaction = r_AB - max(r_A, r_B) - Penalty")
print(f"  Interaction = {combined_corr:.2f} - {max_individual:.2f} - {penalty_term:.3f}")

interaction = combined_corr - max_individual - penalty_term
print(f"  Interaction = {interaction:.3f}")
print(f"  Interaction ≈ {interaction:.2f}")

print(f"\nInterpretation of result:")
print(f"  Interaction strength = {interaction:.2f}")

if interaction > 0.5:
    interpretation = "STRONG synergy"
elif interaction > 0.2:
    interpretation = "MODERATE synergy"
elif interaction > 0:
    interpretation = "WEAK synergy"
elif interaction == 0:
    interpretation = "NO interaction"
else:
    interpretation = "NEGATIVE interaction (interference)"

print(f"  Classification: {interpretation}")
print(f"\nMeaning:")
print(f"  - The positive value ({interaction:.2f}) indicates that features A and B")
print(f"    work much better together than their individual contributions suggest")
print(f"  - This demonstrates complementary predictive power")
print(f"  - The combination captures information that neither feature alone provides")
print(f"  - This synergy can only be detected through multivariate analysis")

# Create visualizations
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Visualization 1: Feature correlation comparison
plt.figure(figsize=(12, 8))

# Subplot 1: Individual vs Combined Performance
plt.subplot(2, 2, 1)
features_plot = list(correlations.keys())
corr_values = list(correlations.values())

bars = plt.bar(features_plot, corr_values, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.7)
plt.axhline(y=0.95, color='purple', linestyle='--', linewidth=2, label='A+B Combined (0.95)')
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Individual Feature Correlations vs Combined Performance')
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, corr_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Subset Performance Heatmap
plt.subplot(2, 2, 2)
subset_names = list(subset_performance.keys())
performance_values = list(subset_performance.values())

# Create a matrix for visualization
subset_matrix = np.zeros((len(subset_names), 1))
for i, perf in enumerate(performance_values):
    subset_matrix[i, 0] = perf

im = plt.imshow(subset_matrix, cmap='RdYlGn', aspect='auto')
plt.yticks(range(len(subset_names)), subset_names)
plt.xticks([])
plt.title('Feature Subset Performance')
plt.colorbar(im, label='Performance Score')

# Add performance values as text
for i, perf in enumerate(performance_values):
    plt.text(0, i, f'{perf:.2f}', ha='center', va='center', fontweight='bold')

# Subplot 3: Search Space Growth
plt.subplot(2, 2, 3)
feature_counts = [4, 5, 6, 7, 8, 9, 10]
search_spaces = [2**n for n in feature_counts]

plt.plot(feature_counts, search_spaces, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Features')
plt.ylabel('Number of Possible Subsets')
plt.title('Exponential Growth of Search Space')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Highlight our specific cases
plt.plot([4, 10], [2**4, 2**10], 'ro', markersize=10, label='Our Cases')
plt.legend()

# Subplot 4: Budget Analysis
plt.subplot(2, 2, 4)
subset_sizes = [len(subset) for subset in valid_subsets]
subset_costs = [len(subset) * feature_cost for subset in valid_subsets]

plt.scatter(subset_sizes, subset_costs, c='green', s=100, alpha=0.7)
plt.axhline(y=budget, color='red', linestyle='--', linewidth=2, label=f'Budget: {budget} dollars')
plt.xlabel('Number of Features in Subset')
plt.ylabel('Cost (dollars)')
plt.title('Feature Subset Costs vs Budget Constraint')
plt.legend()
plt.grid(True, alpha=0.3)

# Add subset labels
for i, (size, cost) in enumerate(zip(subset_sizes, subset_costs)):
    subset_str = "{" + ", ".join(valid_subsets[i]) + "}" if valid_subsets[i] else "{}"
    plt.annotate(subset_str, (size, cost), xytext=(5, 5), textcoords='offset points', 
                 fontsize=8, ha='left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Feature Interaction Analysis
plt.figure(figsize=(10, 6))

# Create a more detailed interaction analysis
plt.subplot(1, 2, 1)
# Individual vs Combined performance
individual_perf = [corr_A, corr_B]
combined_perf = [0, 0.95]  # A alone, A+B combined

x_pos = [1, 2]
width = 0.35

plt.bar([x - width/2 for x in x_pos], individual_perf, width, label='Individual Features', 
        color=['lightblue', 'lightgreen'], alpha=0.7)
plt.bar([x + width/2 for x in x_pos], combined_perf, width, label='Combined Features', 
        color=['purple', 'purple'], alpha=0.7)

plt.xlabel('Feature Configuration')
plt.ylabel('Performance Score')
plt.title('Individual vs Combined Feature Performance')
plt.xticks(x_pos, ['Feature A', 'Features A+B'])
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels
for i, (ind, comb) in enumerate(zip(individual_perf, combined_perf)):
    plt.text(x_pos[i] - width/2, ind + 0.02, f'{ind:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.text(x_pos[i] + width/2, comb + 0.02, f'{comb:.2f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Interaction strength breakdown
plt.subplot(1, 2, 2)
components = ['Combined\nCorrelation', 'Max Individual\nCorrelation', 'Min Individual\n× 0.1', 'Interaction\nStrength']
values = [combined_corr, max_individual, 0.1 * min_individual, interaction]
colors = ['green', 'red', 'orange', 'purple']

bars = plt.bar(components, values, color=colors, alpha=0.7)
plt.ylabel('Correlation Value')
plt.title('Interaction Strength Breakdown')
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_interaction_analysis.png'), dpi=300, bbox_inches='tight')

# New informative visualization: Feature Performance Comparison
print("\nGenerating Feature Performance Comparison...")
plt.figure(figsize=(12, 8))

# Create a clean performance comparison chart
feature_combinations = [
    ('A', 0.20, 'Individual'),
    ('B', 0.30, 'Individual'), 
    ('C', 0.80, 'Individual'),
    ('D', 0.75, 'Individual'),
    ('A+B', 0.95, 'Synergy'),
    ('A+C', 0.82, 'Mixed'),
    ('A+D', 0.77, 'Mixed'),
    ('B+C', 0.82, 'Mixed'),
    ('B+D', 0.77, 'Mixed'),
    ('C+D', 0.81, 'Redundancy')
]

# Extract data for plotting
names = [combo[0] for combo in feature_combinations]
performances = [combo[1] for combo in feature_combinations]
types = [combo[2] for combo in feature_combinations]

# Define colors for different types
color_map = {
    'Individual': 'lightblue',
    'Synergy': 'gold',
    'Mixed': 'lightgreen', 
    'Redundancy': 'lightcoral'
}
colors = [color_map[type_] for type_ in types]

# Create the bar chart
bars = plt.bar(names, performances, color=colors, edgecolor='black', linewidth=1)

# Highlight the optimal solution
optimal_idx = names.index('A+B')
bars[optimal_idx].set_edgecolor('red')
bars[optimal_idx].set_linewidth(3)

# Add performance values on top of bars
for bar, perf in zip(bars, performances):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{perf:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add cost information below bars
costs = [5, 5, 5, 5, 10, 10, 10, 10, 10, 10]
for i, (bar, cost) in enumerate(zip(bars, costs)):
    plt.text(bar.get_x() + bar.get_width()/2., -0.05,
             f'\\${cost}', ha='center', va='top', fontsize=9, style='italic')

# Formatting
plt.xlabel(r'$\textbf{Feature Combinations}$', fontsize=12)
plt.ylabel(r'$\textbf{Performance Score}$', fontsize=12)
plt.title(r'$\textbf{Feature Selection Performance Comparison}$', fontsize=14, pad=20)

# Add horizontal lines for comparison
plt.axhline(y=0.81, color='red', linestyle='--', alpha=0.7, label=r'$\text{Univariate Best: C+D (0.81)}$')
plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label=r'$\text{Multivariate Best: A+B (0.95)}$')

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Individual Features'),
    plt.Rectangle((0,0),1,1, facecolor='gold', label='Synergy (A+B)'),
    plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Mixed Combinations'),
    plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='Redundancy (C+D)')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Add annotations
plt.annotate(r'$\textbf{Optimal Solution}$', 
             xy=(optimal_idx, 0.95), xytext=(optimal_idx, 1.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             ha='center', fontsize=12, fontweight='bold', color='red')

plt.annotate(r'$\text{Synergy Formula: } P = r_A + r_B + (r_A \times r_B \times 7.5)$',
             xy=(0.1, 0.9), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', edgecolor='orange'),
             fontsize=10)

plt.annotate(r'$\text{Cost shown below each bar}$',
             xy=(0.1, 0.05), xycoords='axes fraction',
             fontsize=9, style='italic')

plt.grid(True, alpha=0.3, axis='y')
plt.ylim(-0.1, 1.1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_performance_comparison.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
