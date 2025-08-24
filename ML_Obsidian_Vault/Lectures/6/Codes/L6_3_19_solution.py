import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 19: COMPUTATIONAL COMPLEXITY ANALYSIS")
print("=" * 80)

# Define the three algorithms and their complexity characteristics
algorithms = {
    'ID3': {
        'categorical_complexity': 'O(n × m × v)',
        'continuous_complexity': 'Not supported',
        'reason_categorical': 'Simple counting and entropy calculation',
        'reason_continuous': 'No continuous feature handling',
        'characteristics': [
            'Basic entropy calculations',
            'Simple counting operations',
            'No feature preprocessing',
            'Linear scaling with features'
        ],
        'color': 'red'
    },
    'C4.5': {
        'categorical_complexity': 'O(n × m × v × log(v))',
        'continuous_complexity': 'O(n × m × log(n))',
        'reason_categorical': 'Gain ratio calculation and sorting',
        'reason_continuous': 'Sorting and threshold finding',
        'characteristics': [
            'Gain ratio calculations',
            'Sorting for continuous features',
            'Pessimistic error pruning',
            'Moderate computational overhead'
        ],
        'color': 'blue'
    },
    'CART': {
        'categorical_complexity': 'O(n × m × v²)',
        'continuous_complexity': 'O(n × m × log(n))',
        'reason_categorical': 'Binary splitting optimization',
        'reason_continuous': 'Efficient binary search',
        'characteristics': [
            'Binary splitting strategy',
            'Cost-complexity pruning',
            'Surrogate split handling',
            'Highest computational cost'
        ],
        'color': 'green'
    }
}

# 1. Which algorithm has the highest computational complexity for categorical features?
print("1. Which algorithm has the highest computational complexity for categorical features? Why?")
print("-" * 80)
print("Answer: CART has the highest computational complexity for categorical features")
print(f"Complexity: {algorithms['CART']['categorical_complexity']}")
print(f"Reason: {algorithms['CART']['reason_categorical']}")
print("\nDetailed explanation:")
print("  • CART uses binary splitting strategy for categorical features")
print("  • Must evaluate all possible binary partitions of categorical values")
print("  • For a feature with v values, there are 2^(v-1) - 1 possible splits")
print("  • This leads to O(v²) complexity in the worst case")

# 2. How does handling continuous features affect C4.5's time complexity?
print("\n\n2. How does handling continuous features affect C4.5's time complexity?")
print("-" * 80)
print(f"Answer: C4.5's complexity becomes {algorithms['C4.5']['continuous_complexity']}")
print(f"Reason: {algorithms['C4.5']['reason_continuous']}")
print("\nDetailed explanation:")
print("  • C4.5 must sort continuous feature values")
print("  • Sorting has O(n log n) complexity")
print("  • Must evaluate all possible split points")
print("  • Gain ratio calculation adds logarithmic factor")
print("  • Overall complexity: O(n × m × log(n))")

# 3. Rank algorithms by expected training time for the given dataset
print("\n\n3. For a dataset with 1000 samples and 10 features (5 categorical with avg 4 values, 5 continuous),")
print("   rank the algorithms by expected training time")
print("-" * 80)

# Dataset specifications
n_samples = 1000
n_categorical = 5
n_continuous = 5
avg_categorical_values = 4
n_features = n_categorical + n_continuous

print(f"Dataset specifications:")
print(f"  • Samples (n): {n_samples:,}")
print(f"  • Categorical features: {n_categorical}")
print(f"  • Continuous features: {n_continuous}")
print(f"  • Average categorical values: {avg_categorical_values}")
print(f"  • Total features: {n_features}")

# Calculate expected complexities
complexities = {}
for algo_name, algo_info in algorithms.items():
    if algo_name == 'ID3':
        # ID3 only handles categorical features
        complexity = n_samples * n_categorical * avg_categorical_values
        complexities[algo_name] = complexity
    elif algo_name == 'C4.5':
        # C4.5 handles both types
        cat_complexity = n_samples * n_categorical * avg_categorical_values * np.log2(avg_categorical_values)
        cont_complexity = n_samples * n_continuous * np.log2(n_samples)
        complexity = cat_complexity + cont_complexity
        complexities[algo_name] = complexity
    elif algo_name == 'CART':
        # CART handles both types
        cat_complexity = n_samples * n_categorical * (avg_categorical_values ** 2)
        cont_complexity = n_samples * n_continuous * np.log2(n_samples)
        complexity = cat_complexity + cont_complexity
        complexities[algo_name] = complexity

# Rank by complexity (lower = faster)
ranking = sorted(complexities.items(), key=lambda x: x[1])
print(f"\nExpected training time ranking (fastest to slowest):")
for i, (algo, complexity) in enumerate(ranking, 1):
    print(f"  {i}. {algo}: {complexity:,.0f} operations")

# 4. What makes CART more computationally expensive than ID3 for categorical features?
print("\n\n4. What makes CART more computationally expensive than ID3 for categorical features?")
print("-" * 80)
print("Answer: Binary splitting strategy and optimization requirements")
print("\nDetailed explanation:")
print("  • ID3: Simple multi-way splits (one branch per value)")
print("  • CART: Binary splits (must find optimal binary partitions)")
print("  • CART evaluates $2^{v-1} - 1$ possible splits vs. ID3's single split")
print("  • CART uses cost-complexity pruning vs. ID3's no pruning")
print("  • CART requires surrogate split handling for missing values")

# Create separate visualizations for each aspect

# Plot 1: Categorical Feature Complexity Comparison
plt.figure(figsize=(10, 6))
algo_names = list(algorithms.keys())
cat_complexities = [algorithms[algo]['categorical_complexity'] for algo in algo_names]
colors = [algorithms[algo]['color'] for algo in algo_names]

# Create complexity bars (using numerical representation for visualization)
complexity_values = [1, 2, 4]  # ID3: O(nmv), C4.5: O(nmv log v), CART: O(nmv²)
bars1 = plt.bar(algo_names, complexity_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Relative Complexity (Higher = More Complex)')
plt.ylim(0, 5)
plt.grid(True, alpha=0.3)
plt.title('Categorical Feature Complexity', fontsize=14, fontweight='bold')

# Add complexity labels on bars
for bar, complexity, formula in zip(bars1, complexity_values, cat_complexities):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             formula, ha='center', va='bottom', fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'categorical_feature_complexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Continuous Feature Complexity Comparison
plt.figure(figsize=(10, 6))
cont_complexities = []
cont_values = []

for algo_name in algo_names:
    if algo_name == 'ID3':
        cont_complexities.append('Not Supported')
        cont_values.append(0)
    else:
        cont_complexities.append(algorithms[algo_name]['continuous_complexity'])
        cont_values.append(2 if algo_name == 'C4.5' else 2)  # Both O(nm log n)

bars2 = plt.bar(algo_names, cont_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Relative Complexity (Higher = More Complex)')
plt.ylim(0, 3)
plt.grid(True, alpha=0.3)
plt.title('Continuous Feature Complexity', fontsize=14, fontweight='bold')

# Add complexity labels on bars
for bar, complexity, value in zip(bars2, cont_complexities, cont_values):
    height = bar.get_height()
    if value > 0:
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 complexity, ha='center', va='bottom', fontsize=10, rotation=0)
    else:
        plt.text(bar.get_x() + bar.get_width()/2., height/2,
                 complexity, ha='center', va='center', fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'continuous_feature_complexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Training Time Comparison for Given Dataset
plt.figure(figsize=(10, 6))
algo_names_plot = [algo for algo, _ in ranking]
complexity_values_plot = [complexity for _, complexity in ranking]
colors_plot = [algorithms[algo]['color'] for algo in algo_names_plot]

# Normalize complexity values for better visualization
max_complexity = max(complexity_values_plot)
normalized_complexities = [comp / max_complexity * 100 for comp in complexity_values_plot]

bars3 = plt.bar(algo_names_plot, normalized_complexities, color=colors_plot, alpha=0.7, edgecolor='black')
plt.ylabel('Relative Training Time (%)')
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)
plt.title('Expected Training Time for 1000×10 Dataset', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, norm_val, actual_val in zip(bars3, normalized_complexities, complexity_values_plot):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{actual_val:,.0f}', ha='center', va='bottom', fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Complexity Growth with Feature Values
plt.figure(figsize=(10, 6))
plt.title('Complexity Growth with Categorical Feature Values', fontsize=14, fontweight='bold')

# Generate sample data for complexity growth
v_values = np.arange(2, 11)  # Number of categorical values
id3_complexity = v_values  # O(v)
c45_complexity = v_values * np.log2(v_values)  # O(v log v)
cart_complexity = v_values ** 2  # O(v²)

plt.plot(v_values, id3_complexity, 'r-', linewidth=3, label='ID3: $O(v)$', marker='o')
plt.plot(v_values, c45_complexity, 'b-', linewidth=3, label='C4.5: $O(v \\log v)$', marker='s')
plt.plot(v_values, cart_complexity, 'g-', linewidth=3, label='CART: $O(v^2)$', marker='^')

plt.xlabel('Number of Categorical Values $(v)$')
plt.ylabel('Relative Complexity')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'complexity_growth.png'), dpi=300, bbox_inches='tight')
plt.close()



# Create detailed complexity table
print("\n" + "=" * 80)
print("DETAILED COMPLEXITY COMPARISON TABLE")
print("=" * 80)

complexity_data = {
    'Algorithm': [
        'ID3',
        'C4.5', 
        'CART'
    ],
    'Categorical Features': [
        '$O(n \\times m \\times v)$',
        '$O(n \\times m \\times v \\times \\log(v))$',
        '$O(n \\times m \\times v^2)$'
    ],
    'Continuous Features': [
        'Not Supported',
        '$O(n \\times m \\times \\log(n))$',
        '$O(n \\times m \\times \\log(n))$'
    ],
    'Pruning Complexity': [
        'None',
        '$O(|T| \\times \\log(|T|))$',
        '$O(|T|^2)$'
    ],
    'Memory Usage': [
        'Low',
        'Medium',
        'High'
    ],
    'Best For': [
        'Small datasets, education',
        'Medium datasets, interpretability',
        'Large datasets, production'
    ]
}

df_complexity = pd.DataFrame(complexity_data)
print(df_complexity.to_string(index=False))

# Mathematical analysis
print("\n" + "=" * 80)
print("MATHEMATICAL COMPLEXITY ANALYSIS")
print("=" * 80)

print("1. ID3 Complexity Analysis:")
print("   • Categorical features: $O(n \\times m \\times v)$")
print("     - $n$ = number of samples")
print("     - $m$ = number of features")
print("     - $v$ = average number of values per categorical feature")
print("   • No continuous feature support")
print("   • No pruning complexity")

print("\n2. C4.5 Complexity Analysis:")
print("   • Categorical features: $O(n \\times m \\times v \\times \\log(v))$")
print("     - Additional $\\log(v)$ factor due to gain ratio calculation")
print("   • Continuous features: $O(n \\times m \\times \\log(n))$")
print("     - $\\log(n)$ factor due to sorting")
print("   • Pruning: $O(|T| \\times \\log(|T|))$ where $|T|$ is tree size")

print("\n3. CART Complexity Analysis:")
print("   • Categorical features: $O(n \\times m \\times v^2)$")
print("     - $v^2$ factor due to binary splitting optimization")
print("   • Continuous features: $O(n \\times m \\times \\log(n))$")
print("     - Efficient binary search implementation")
print("   • Pruning: $O(|T|^2)$ due to cost-complexity calculations")

# Practical implications
print("\n" + "=" * 80)
print("PRACTICAL IMPLICATIONS")
print("=" * 80)

print("1. Dataset Size Considerations:")
print("   • Small datasets (< 1000 samples): ID3 is sufficient")
print("   • Medium datasets (1000-10000 samples): C4.5 provides good balance")
print("   • Large datasets (> 10000 samples): CART scales better")

print("\n2. Feature Type Considerations:")
print("   • Categorical-only: ID3 is fastest")
print("   • Mixed features: C4.5 and CART handle both efficiently")
print("   • High-cardinality categorical: CART may be slowest")

print("\n3. Production Considerations:")
print("   • ID3: Educational and prototyping")
print("   • C4.5: Interpretable models, medical applications")
print("   • CART: Production systems, real-time prediction")

print(f"\nPlots saved to: {save_dir}")
