import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("OUT-OF-BAG (OOB) ESTIMATION - COMPREHENSIVE ANALYSIS")
print("=" * 80)

# Create a synthetic dataset for demonstration
print("\n" + "="*60)
print("DATASET CREATION")
print("="*60)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=2, random_state=42)
print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
print(f"Class distribution: {np.bincount(y)}")

print("\n" + "="*60)
print("QUESTION 1: HOW DOES OUT-OF-BAG ESTIMATION WORK?")
print("="*60)

print("\nOut-of-Bag (OOB) estimation is a built-in validation method for Random Forests.")
print("Here's how it works step by step:")

# Demonstrate bootstrap sampling
print("\n" + "-"*40)
print("STEP 1: BOOTSTRAP SAMPLING")
print("-"*40)

n_samples = 1000
n_trees = 10

print(f"Dataset size: {n_samples}")
print(f"Number of trees: {n_trees}")

# Simulate bootstrap sampling for multiple trees
bootstrap_samples = []
oob_samples = []

for tree_idx in range(n_trees):
    # Bootstrap sample (with replacement)
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    bootstrap_samples.append(bootstrap_indices)
    
    # Find OOB samples (samples not in bootstrap)
    all_indices = set(range(n_samples))
    bootstrap_set = set(bootstrap_indices)
    oob_indices = list(all_indices - bootstrap_set)
    oob_samples.append(oob_indices)
    
    print(f"Tree {tree_idx + 1}:")
    print(f"  Bootstrap samples: {len(bootstrap_indices)} (with replacement)")
    print(f"  OOB samples: {len(oob_indices)}")
    print(f"  OOB percentage: {len(oob_indices)/n_samples*100:.1f}%")

# Calculate expected OOB percentage
expected_oob = (1 - 1/n_samples)**n_samples * 100
print(f"\nExpected OOB percentage: {expected_oob:.1f}%")

print("\n" + "-"*40)
print("STEP 2: OOB PREDICTION PROCESS")
print("-"*40)

print("For each sample, OOB predictions are made using only trees where that sample was NOT used in training:")
print("1. Sample is in OOB set for some trees")
print("2. Only those trees make predictions for that sample")
print("3. Final prediction is majority vote from OOB trees")

# Demonstrate OOB prediction for a specific sample
sample_idx = 0
print(f"\nExample: Sample {sample_idx}")
oob_trees_for_sample = []
for tree_idx, oob_indices in enumerate(oob_samples):
    if sample_idx in oob_indices:
        oob_trees_for_sample.append(tree_idx)

print(f"Sample {sample_idx} is OOB for {len(oob_trees_for_sample)} trees: {oob_trees_for_sample}")

print("\n" + "="*60)
print("QUESTION 2: ADVANTAGES OF OOB OVER CROSS-VALIDATION")
print("="*60)

print("\nOOB estimation offers several key advantages:")

print("\n1. COMPUTATIONAL EFFICIENCY:")
print("   - No need for separate validation splits")
print("   - Validation happens during training")
print("   - Reduces training time significantly")

print("\n2. DATA UTILIZATION:")
print("   - All data used for training")
print("   - No data lost to validation splits")
print("   - More robust for small datasets")

print("\n3. AUTOMATIC VALIDATION:")
print("   - Built into Random Forest training")
print("   - No manual cross-validation setup")
print("   - Consistent validation across runs")

# Demonstrate computational advantage
print("\n" + "-"*40)
print("COMPUTATIONAL COMPARISON")
print("-"*40)

# Time OOB vs Cross-validation
from sklearn.model_selection import cross_val_score
import time

rf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)

# Time OOB estimation
start_time = time.time()
rf.fit(X, y)
oob_score = rf.oob_score_
oob_time = time.time() - start_time

print(f"OOB estimation time: {oob_time:.4f} seconds")
print(f"OOB accuracy: {oob_score:.4f}")

# Time cross-validation
start_time = time.time()
cv_scores = cross_val_score(rf, X, y, cv=5)
cv_time = time.time() - start_time

print(f"5-fold CV time: {cv_time:.4f} seconds")
print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print(f"\nSpeedup: {cv_time/oob_time:.1f}x faster with OOB")

print("\n" + "="*60)
print("QUESTION 3: WHEN MIGHT OOB ESTIMATION NOT BE RELIABLE?")
print("="*60)

print("\nOOB estimation may not be reliable in several scenarios:")

print("\n1. SMALL DATASETS:")
print("   - Limited OOB samples per tree")
print("   - High variance in OOB estimates")
print("   - Unstable predictions")

print("\n2. IMBALANCED CLASSES:")
print("   - Some classes may have few OOB samples")
print("   - Biased OOB estimates")
print("   - Poor representation in validation")

print("\n3. OVERFITTING:")
print("   - Trees may be too complex")
print("   - OOB samples may not be representative")
print("   - Optimistic bias in estimates")

# Demonstrate with different dataset sizes
print("\n" + "-"*40)
print("RELIABILITY ANALYSIS WITH DIFFERENT DATASET SIZES")
print("-"*40)

dataset_sizes = [100, 500, 1000, 2000]
oob_variances = []

for size in dataset_sizes:
    # Create smaller dataset
    X_small = X[:size]
    y_small = y[:size]
    
    # Multiple runs to check variance
    oob_scores = []
    for run in range(10):
        rf_small = RandomForestClassifier(n_estimators=50, random_state=run, oob_score=True)
        rf_small.fit(X_small, y_small)
        oob_scores.append(rf_small.oob_score_)
    
    variance = np.var(oob_scores)
    oob_variances.append(variance)
    
    print(f"Dataset size {size}:")
    print(f"  Mean OOB score: {np.mean(oob_scores):.4f}")
    print(f"  OOB variance: {variance:.6f}")
    print(f"  Standard deviation: {np.std(oob_scores):.4f}")

# Plot reliability analysis
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(dataset_sizes, oob_variances, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Dataset Size')
plt.ylabel('OOB Score Variance')
plt.title('OOB Reliability vs Dataset Size')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 2)
# Show OOB sample distribution for different tree counts
tree_counts = [10, 25, 50, 100]
oob_percentages = [(1 - 1/n_samples)**n for n in tree_counts]
plt.plot(tree_counts, oob_percentages, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Trees')
plt.ylabel('Expected OOB Percentage')
plt.title('OOB Sample Percentage vs Tree Count')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
# Demonstrate OOB sample distribution for a single tree
tree_idx = 0
oob_indices = oob_samples[tree_idx]
bootstrap_indices = bootstrap_samples[tree_idx]

# Create a visualization of bootstrap vs OOB
all_samples = np.arange(n_samples)
bootstrap_mask = np.isin(all_samples, bootstrap_indices)
oob_mask = np.isin(all_samples, oob_indices)

plt.scatter(all_samples[bootstrap_mask], [0]*np.sum(bootstrap_mask), 
           c='blue', alpha=0.6, label='Bootstrap Samples', s=20)
plt.scatter(all_samples[oob_mask], [0]*np.sum(oob_mask), 
           c='red', alpha=0.8, label='OOB Samples', s=20)
plt.xlabel('Sample Index')
plt.ylabel('Sample Type')
plt.title(f'Bootstrap vs OOB Samples (Tree {tree_idx + 1})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# Show OOB sample count distribution across trees
oob_counts = [len(oob) for oob in oob_samples]
plt.hist(oob_counts, bins=20, alpha=0.7, color='green', edgecolor='black')
plt.axvline(np.mean(oob_counts), color='red', linestyle='--', 
            label=f'Mean: {np.mean(oob_counts):.1f}')
plt.xlabel('OOB Sample Count')
plt.ylabel('Frequency')
plt.title('Distribution of OOB Sample Counts Across Trees')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'oob_reliability_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("QUESTION 4: HOW DOES OOB HELP WITH MODEL SELECTION?")
print("="*60)

print("\nOOB estimation is valuable for model selection in several ways:")

print("\n1. HYPERPARAMETER TUNING:")
print("   - Compare OOB scores across configurations")
print("   - No need for separate validation set")
print("   - Faster iteration through parameters")

print("\n2. FEATURE SELECTION:")
print("   - OOB scores with different feature subsets")
print("   - Identify most important features")
print("   - Avoid overfitting to validation set")

print("\n3. ENSEMBLE SIZE OPTIMIZATION:")
print("   - Monitor OOB score vs number of trees")
print("   - Find optimal tree count")
print("   - Balance accuracy and computational cost")

# Demonstrate hyperparameter tuning with OOB
print("\n" + "-"*40)
print("HYPERPARAMETER TUNING WITH OOB")
print("-"*40)

# Test different hyperparameter combinations
max_depths = [5, 10, 15, 20]
max_features = [5, 10, 15, 20]
n_estimators = [50, 100, 200]

results = []

for depth in max_depths:
    for features in max_features:
        for n_trees in n_estimators:
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=depth,
                max_features=features,
                random_state=42,
                oob_score=True
            )
            rf.fit(X, y)
            
            results.append({
                'max_depth': depth,
                'max_features': features,
                'n_estimators': n_trees,
                'oob_score': rf.oob_score_,
                'training_time': 0  # Placeholder for actual timing
            })

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)
best_config = results_df.loc[results_df['oob_score'].idxmax()]

print("Best configuration based on OOB score:")
print(f"  Max depth: {best_config['max_depth']}")
print(f"  Max features: {best_config['max_features']}")
print(f"  Number of trees: {best_config['n_estimators']}")
print(f"  OOB score: {best_config['oob_score']:.4f}")

# Visualize hyperparameter tuning results
plt.figure(figsize=(15, 10))

# Plot 1: OOB score vs max_depth for different max_features
plt.subplot(2, 3, 1)
for features in max_features:
    subset = results_df[results_df['max_features'] == features]
    plt.plot(subset['max_depth'], subset['oob_score'], 
             marker='o', label=f'max_features={features}')
plt.xlabel('Max Depth')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Max Depth')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: OOB score vs max_features for different depths
plt.subplot(2, 3, 2)
for depth in max_depths:
    subset = results_df[results_df['max_depth'] == depth]
    plt.plot(subset['max_features'], subset['oob_score'], 
             marker='s', label=f'max_depth={depth}')
plt.xlabel('Max Features')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Max Features')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: OOB score vs n_estimators
plt.subplot(2, 3, 3)
for depth in max_depths:
    for features in max_features:
        subset = results_df[(results_df['max_depth'] == depth) & 
                           (results_df['max_features'] == features)]
        plt.plot(subset['n_estimators'], subset['oob_score'], 
                 marker='^', label=f'depth={depth}, features={features}')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Number of Trees')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot 4: Heatmap of OOB scores
plt.subplot(2, 3, 4)
pivot_table = results_df[results_df['n_estimators'] == 100].pivot(
    index='max_depth', columns='max_features', values='oob_score'
)
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
plt.title('OOB Score Heatmap (100 trees)')
plt.xlabel('Max Features')
plt.ylabel('Max Depth')

# Plot 5: 3D scatter plot
plt.subplot(2, 3, 5)
ax = plt.subplot(2, 3, 5, projection='3d')
scatter = ax.scatter(results_df['max_depth'], results_df['max_features'], 
                    results_df['n_estimators'], c=results_df['oob_score'], 
                    cmap='viridis', s=50)
ax.set_xlabel('Max Depth')
ax.set_ylabel('Max Features')
ax.set_zlabel('Number of Trees')
ax.set_title('3D View of Hyperparameters vs OOB Score')
plt.colorbar(scatter, ax=ax, label='OOB Score')

# Plot 6: Best configurations
plt.subplot(2, 3, 6)
top_5 = results_df.nlargest(5, 'oob_score')
plt.bar(range(len(top_5)), top_5['oob_score'],
        color=['gold', 'silver', 'brown', 'lightblue', 'lightgreen'])
plt.xticks(range(len(top_5)), [f'({r["max_depth"]},{r["max_features"]},{r["n_estimators"]})' 
                               for _, r in top_5.iterrows()], rotation=45)
plt.ylabel('OOB Score')
plt.title('Top 5 Configurations')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'oob_hyperparameter_tuning.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("QUESTION 5: CALCULATE EXPECTED OOB SAMPLES")
print("="*60)

print("\nFor a dataset with 1000 samples, we calculate the expected number of OOB samples per tree.")

print("\nFormula: OOB samples = n × (1 - 1/n)^n")
print("where n is the dataset size")

n = 1000
oob_per_tree = n * ((1 - 1/n) ** n)
oob_percentage = oob_per_tree / n * 100

print(f"\nDataset size (n): {n}")
print(f"OOB samples per tree: {oob_per_tree:.2f}")
print(f"OOB percentage: {oob_percentage:.2f}%")

# Demonstrate the mathematical foundation
print("\n" + "-"*40)
print("MATHEMATICAL FOUNDATION")
print("-"*40)

print("The formula comes from probability theory:")
print("1. Probability a sample is NOT selected in one bootstrap draw: (n-1)/n")
print("2. Probability a sample is NOT selected in n bootstrap draws: ((n-1)/n)^n")
print("3. Expected OOB samples: n × ((n-1)/n)^n")
print("4. For large n, this approaches n × (1/e) ~ n × 0.368")

# Calculate for different dataset sizes
dataset_sizes = [100, 500, 1000, 2000, 5000]
oob_calculations = []

for size in dataset_sizes:
    oob_count = size * ((1 - 1/size) ** size)
    oob_pct = oob_count / size * 100
    oob_calculations.append({
        'size': size,
        'oob_count': oob_count,
        'oob_percentage': oob_pct,
        'theoretical_limit': size * (1/np.e) / size * 100
    })

print("\n" + "-"*40)
print("OOB CALCULATIONS FOR DIFFERENT DATASET SIZES")
print("-"*40)

for calc in oob_calculations:
    print(f"Dataset size {calc['size']:4d}:")
    print(f"  OOB samples: {calc['oob_count']:6.1f}")
    print(f"  OOB percentage: {calc['oob_percentage']:5.2f}%")
    print(f"  Theoretical limit (1/e): {calc['theoretical_limit']:5.2f}%")

# Visualize the OOB calculation
plt.figure(figsize=(15, 10))

# Plot 1: OOB samples vs dataset size
plt.subplot(2, 3, 1)
sizes = [calc['size'] for calc in oob_calculations]
oob_counts = [calc['oob_count'] for calc in oob_calculations]
theoretical = [calc['size'] * (1/np.e) for calc in oob_calculations]

plt.plot(sizes, oob_counts, 'bo-', linewidth=2, markersize=8, label='Actual OOB')
plt.plot(sizes, theoretical, 'r--', linewidth=2, label='Theoretical limit (n/e)')
plt.xlabel('Dataset Size')
plt.ylabel('OOB Samples per Tree')
plt.title('OOB Samples vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: OOB percentage vs dataset size
plt.subplot(2, 3, 2)
oob_pcts = [calc['oob_percentage'] for calc in oob_calculations]
theoretical_pcts = [calc['theoretical_limit'] for calc in oob_calculations]

plt.plot(sizes, oob_pcts, 'go-', linewidth=2, markersize=8, label='Actual OOB %')
plt.plot(sizes, theoretical_pcts, 'r--', linewidth=2, label='Theoretical limit (1/e)')
plt.axhline(y=100/np.e, color='red', linestyle=':', alpha=0.7, label='36.8%')
plt.xlabel('Dataset Size')
plt.ylabel('OOB Percentage')
plt.title('OOB Percentage vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Convergence to theoretical limit
plt.subplot(2, 3, 3)
convergence = [(calc['oob_percentage'] - calc['theoretical_limit']) for calc in oob_calculations]
plt.plot(sizes, convergence, 'mo-', linewidth=2, markersize=8)
plt.xlabel('Dataset Size')
plt.ylabel('Difference from Theoretical Limit')
plt.title('Convergence to Theoretical Limit')
plt.grid(True, alpha=0.3)

# Plot 4: Bootstrap sampling visualization
plt.subplot(2, 3, 4)
# Show bootstrap sampling process for small dataset
small_n = 10
bootstrap_trials = 1000
oob_counts_small = []

for _ in range(bootstrap_trials):
    bootstrap_indices = np.random.choice(small_n, size=small_n, replace=True)
    oob_count = small_n - len(set(bootstrap_indices))
    oob_counts_small.append(oob_count)

plt.hist(oob_counts_small, bins=range(small_n+1), alpha=0.7, color='orange', edgecolor='black')
plt.axvline(np.mean(oob_counts_small), color='red', linestyle='--', 
            label=f'Mean: {np.mean(oob_counts_small):.1f}')
plt.xlabel('OOB Sample Count')
plt.ylabel('Frequency')
plt.title(f'Distribution of OOB Counts (n={small_n})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Mathematical derivation
plt.subplot(2, 3, 5)
# Show the probability calculation
n_values = np.linspace(1, 100, 1000)
prob_not_selected = ((n_values - 1) / n_values) ** n_values
theoretical_prob = 1 / np.e * np.ones_like(n_values)

plt.plot(n_values, prob_not_selected, 'b-', linewidth=2, label='P(not selected)')
plt.plot(n_values, theoretical_prob, 'r--', linewidth=2, label='1/e ~ 0.368')
plt.xlabel('Dataset Size (n)')
plt.ylabel('Probability')
plt.title('Probability of Sample Not Being Selected')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Summary statistics
plt.subplot(2, 3, 6)
# Create a summary table
summary_data = []
for calc in oob_calculations:
    summary_data.append([calc['size'], calc['oob_count'], calc['oob_percentage']])

summary_df = pd.DataFrame(summary_data, columns=['Size', 'OOB Count', 'OOB %'])
plt.axis('off')
table = plt.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
plt.title('Summary of OOB Calculations', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'oob_mathematical_analysis.png'), dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("COMPREHENSIVE SUMMARY")
print("="*60)

print("\nKey insights about Out-of-Bag estimation:")

print("\n1. MECHANISM:")
print(f"   - Each tree uses bootstrap sampling (with replacement)")
print(f"   - OOB samples are those not used in training")
print(f"   - OOB predictions provide unbiased validation")

print("\n2. ADVANTAGES:")
print(f"   - Computational efficiency: {cv_time/oob_time:.1f}x faster than CV")
print(f"   - Data utilization: 100% of data used for training")
print(f"   - Automatic validation: built into training process")

print("\n3. RELIABILITY:")
print(f"   - Dataset size 1000: OOB variance {oob_variances[2]:.6f}")
print(f"   - Expected OOB samples per tree: {oob_per_tree:.1f}")
print(f"   - Theoretical limit: {100/np.e:.1f}%")

print("\n4. MODEL SELECTION:")
print(f"   - Best configuration: depth={best_config['max_depth']}, features={best_config['max_features']}, trees={best_config['n_estimators']}")
print(f"   - Best OOB score: {best_config['oob_score']:.4f}")

print(f"\nAll plots and analysis saved to: {save_dir}")
print("=" * 80)
