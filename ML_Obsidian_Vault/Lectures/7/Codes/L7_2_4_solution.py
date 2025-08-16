import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("BAGGING ENSEMBLE PARAMETERS AND OUT-OF-BAG SAMPLES ANALYSIS")
print("=" * 80)

# Given parameters
n_trees = 100
bootstrap_sample_size = 1000
original_dataset_size = 1000
base_learner = "Decision Tree"

print(f"\nGiven Parameters:")
print(f"Number of trees: {n_trees}")
print(f"Bootstrap sample size: {bootstrap_sample_size}")
print(f"Original dataset size: {original_dataset_size}")
print(f"Base learner: {base_learner}")

# Question 1: How many different training datasets will be created?
print(f"\n{'='*60}")
print("QUESTION 1: How many different training datasets will be created?")
print(f"{'='*60}")

print(f"Answer: {n_trees} different training datasets will be created.")
print(f"\nExplanation:")
print(f"- Each tree in the bagging ensemble is trained on a different bootstrap sample")
print(f"- Since we have {n_trees} trees, we create {n_trees} different training datasets")
print(f"- Each dataset is created by sampling with replacement from the original dataset")

# Question 2: Expected number of unique samples per bootstrap sample
print(f"\n{'='*60}")
print("QUESTION 2: Expected number of unique samples per bootstrap sample")
print(f"{'='*60}")

# Theoretical calculation
expected_unique = original_dataset_size * (1 - np.exp(-bootstrap_sample_size/original_dataset_size))
print(f"Theoretical Answer: {expected_unique:.2f} unique samples")
print(f"\nMathematical Derivation:")
print(f"- For bootstrap sampling with replacement from a dataset of size N")
print(f"- The probability that a specific sample is NOT selected in one draw: (N-1)/N")
print(f"- The probability that a specific sample is NOT selected in M draws: ((N-1)/N)^M")
print(f"- The probability that a specific sample IS selected at least once: 1 - ((N-1)/N)^M")
print(f"- Expected number of unique samples: N × [1 - ((N-1)/N)^M]")
print(f"- For N = {original_dataset_size}, M = {bootstrap_sample_size}:")
print(f"  Expected unique = {original_dataset_size} × [1 - ({original_dataset_size-1}/{original_dataset_size})^{bootstrap_sample_size}]")
print(f"  Expected unique = {original_dataset_size} × [1 - {((original_dataset_size-1)/original_dataset_size)**bootstrap_sample_size:.6f}]")
print(f"  Expected unique = {original_dataset_size} × {1 - ((original_dataset_size-1)/original_dataset_size)**bootstrap_sample_size:.6f}")
print(f"  Expected unique = {expected_unique:.2f}")

# Empirical demonstration
print(f"\nEmpirical Demonstration:")
np.random.seed(42)  # For reproducibility
n_simulations = 1000
unique_counts = []

for _ in range(n_simulations):
    # Generate bootstrap sample
    bootstrap_indices = np.random.choice(original_dataset_size, size=bootstrap_sample_size, replace=True)
    unique_count = len(np.unique(bootstrap_indices))
    unique_counts.append(unique_count)

empirical_mean = np.mean(unique_counts)
empirical_std = np.std(unique_counts)

print(f"- Simulated {n_simulations} bootstrap samples")
print(f"- Empirical mean: {empirical_mean:.2f} ± {empirical_std:.2f}")
print(f"- Theoretical vs Empirical: {expected_unique:.2f} vs {empirical_mean:.2f}")
print(f"- Difference: {abs(expected_unique - empirical_mean):.2f}")

# Question 3: Expected number of out-of-bag samples per tree
print(f"\n{'='*60}")
print("QUESTION 3: Expected number of out-of-bag samples per tree")
print(f"{'='*60}")

# Theoretical calculation
expected_oob = original_dataset_size * np.exp(-bootstrap_sample_size/original_dataset_size)
print(f"Theoretical Answer: {expected_oob:.2f} out-of-bag samples")
print(f"\nMathematical Derivation:")
print(f"- Out-of-bag samples are those NOT selected in the bootstrap sample")
print(f"- Probability a sample is NOT selected: (N-1)/N")
print(f"- Probability a sample is NOT selected in M draws: ((N-1)/N)^M")
print(f"- Expected OOB samples: N × ((N-1)/N)^M")
print(f"- For N = {original_dataset_size}, M = {bootstrap_sample_size}:")
print(f"  Expected OOB = {original_dataset_size} × ({original_dataset_size-1}/{original_dataset_size})^{bootstrap_sample_size}")
print(f"  Expected OOB = {original_dataset_size} × {((original_dataset_size-1)/original_dataset_size)**bootstrap_sample_size:.6f}")
print(f"  Expected OOB = {expected_oob:.2f}")

# Empirical demonstration
oob_counts = []
for _ in range(n_simulations):
    # Generate bootstrap sample
    bootstrap_indices = np.random.choice(original_dataset_size, size=bootstrap_sample_size, replace=True)
    # Find OOB samples (not in bootstrap sample)
    oob_count = original_dataset_size - len(np.unique(bootstrap_indices))
    oob_counts.append(oob_count)

empirical_oob_mean = np.mean(oob_counts)
empirical_oob_std = np.std(oob_counts)

print(f"\nEmpirical Demonstration:")
print(f"- Simulated {n_simulations} bootstrap samples")
print(f"- Empirical OOB mean: {empirical_oob_mean:.2f} ± {empirical_oob_std:.2f}")
print(f"- Theoretical vs Empirical: {expected_oob:.2f} vs {empirical_oob_mean:.2f}")
print(f"- Difference: {abs(expected_oob - empirical_oob_mean):.2f}")

# Verification: unique + oob should equal original dataset size
print(f"\nVerification:")
print(f"- Expected unique + Expected OOB = {expected_unique:.2f} + {expected_oob:.2f} = {expected_unique + expected_oob:.2f}")
print(f"- Original dataset size = {original_dataset_size}")
print(f"- This confirms our calculations are correct!")

# Question 4: Purpose of out-of-bag samples
print(f"\n{'='*60}")
print("QUESTION 4: Purpose of out-of-bag samples")
print(f"{'='*60}")

print(f"Out-of-bag (OOB) samples serve several important purposes:")
print(f"\n1. Model Validation:")
print(f"   - OOB samples provide an unbiased estimate of model performance")
print(f"   - They act as a built-in validation set without reducing training data")
print(f"   - No need for separate cross-validation")

print(f"\n2. Feature Importance:")
print(f"   - OOB samples can be used to assess feature importance")
print(f"   - Compare performance with and without specific features")
print(f"   - More reliable than using training set performance")

print(f"\n3. Hyperparameter Tuning:")
print(f"   - OOB error can guide hyperparameter selection")
print(f"   - Helps prevent overfitting during model selection")

print(f"\n4. Ensemble Performance Estimation:")
print(f"   - OOB predictions can be aggregated for ensemble performance")
print(f"   - Provides confidence intervals for predictions")

# Visualizations
print(f"\n{'='*60}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*60}")

# 1. Bootstrap sampling visualization
plt.figure(figsize=(15, 10))

# Subplot 1: Bootstrap sampling process
plt.subplot(2, 3, 1)
plt.title('Bootstrap Sampling Process', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Frequency')

# Show first few bootstrap samples
n_show = 5
for i in range(n_show):
    bootstrap_indices = np.random.choice(original_dataset_size, size=bootstrap_sample_size, replace=True)
    counts = Counter(bootstrap_indices)
    plt.bar(range(original_dataset_size), [counts.get(j, 0) for j in range(original_dataset_size)], 
            alpha=0.3, label=f'Sample {i+1}')

plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Distribution of unique sample counts
plt.subplot(2, 3, 2)
plt.title('Distribution of Unique Sample Counts', fontsize=14, fontweight='bold')
plt.xlabel('Number of Unique Samples')
plt.ylabel('Frequency')

plt.hist(unique_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(expected_unique, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical: {expected_unique:.1f}')
plt.axvline(empirical_mean, color='green', linestyle='--', linewidth=2, 
            label=f'Empirical: {empirical_mean:.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Distribution of OOB sample counts
plt.subplot(2, 3, 3)
plt.title('Distribution of Out-of-Bag Sample Counts', fontsize=14, fontweight='bold')
plt.xlabel('Number of OOB Samples')
plt.ylabel('Frequency')

plt.hist(oob_counts, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(expected_oob, color='red', linestyle='--', linewidth=2, 
            label=f'Theoretical: {expected_oob:.1f}')
plt.axvline(empirical_oob_mean, color='green', linestyle='--', linewidth=2, 
            label=f'Empirical: {empirical_oob_mean:.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Relationship between unique and OOB samples
plt.subplot(2, 3, 4)
plt.title('Unique vs OOB Sample Relationship', fontsize=14, fontweight='bold')
plt.xlabel('Number of Unique Samples')
plt.ylabel('Number of OOB Samples')

plt.scatter(unique_counts, oob_counts, alpha=0.6, color='purple')
plt.plot([expected_unique], [expected_oob], 'ro', markersize=10, label='Theoretical')
plt.plot([empirical_mean], [empirical_oob_mean], 'go', markersize=10, label='Empirical')

# Add perfect negative correlation line
x_range = np.linspace(min(unique_counts), max(unique_counts), 100)
y_range = original_dataset_size - x_range
plt.plot(x_range, y_range, 'k--', alpha=0.5, label='Perfect negative correlation')

plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Bootstrap sample overlap analysis
plt.subplot(2, 3, 5)
plt.title('Bootstrap Sample Overlap Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Number of Trees Including Sample')

# Count how many trees include each sample
sample_inclusion_counts = np.zeros(original_dataset_size)
for _ in range(n_trees):
    bootstrap_indices = np.random.choice(original_dataset_size, size=bootstrap_sample_size, replace=True)
    unique_indices = np.unique(bootstrap_indices)
    sample_inclusion_counts[unique_indices] += 1

plt.bar(range(original_dataset_size), sample_inclusion_counts, alpha=0.7, color='orange', edgecolor='black')
plt.axhline(y=n_trees * (1 - np.exp(-bootstrap_sample_size/original_dataset_size)), 
            color='red', linestyle='--', linewidth=2, 
            label=f'Expected: {n_trees * (1 - np.exp(-bootstrap_sample_size/original_dataset_size)):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 6: OOB sample frequency across trees
plt.subplot(2, 3, 6)
plt.title('OOB Sample Frequency Across Trees', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Number of Trees NOT Including Sample')

oob_frequency = n_trees - sample_inclusion_counts
plt.bar(range(original_dataset_size), oob_frequency, alpha=0.7, color='lightgreen', edgecolor='black')
plt.axhline(y=n_trees * np.exp(-bootstrap_sample_size/original_dataset_size), 
            color='red', linestyle='--', linewidth=2, 
            label=f'Expected: {n_trees * np.exp(-bootstrap_sample_size/original_dataset_size):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bagging_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')

# 2. Detailed bootstrap sampling visualization
plt.figure(figsize=(16, 12))

# Show detailed bootstrap sampling for first few samples
n_detailed = 3
for i in range(n_detailed):
    plt.subplot(n_detailed, 2, 2*i + 1)
    plt.title(f'Bootstrap Sample {i+1}: Sample Selection', fontsize=12, fontweight='bold')
    
    # Generate bootstrap sample
    bootstrap_indices = np.random.choice(original_dataset_size, size=bootstrap_sample_size, replace=True)
    counts = Counter(bootstrap_indices)
    
    # Show sample selection frequency
    plt.bar(range(original_dataset_size), [counts.get(j, 0) for j in range(original_dataset_size)], 
            color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Original Sample Index')
    plt.ylabel('Selection Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    unique_count = len(np.unique(bootstrap_indices))
    oob_count = original_dataset_size - unique_count
    plt.text(0.02, 0.98, f'Unique: {unique_count}\nOOB: {oob_count}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Show OOB samples
    plt.subplot(n_detailed, 2, 2*i + 2)
    plt.title(f'Bootstrap Sample {i+1}: OOB Samples', fontsize=12, fontweight='bold')
    
    # Mark OOB samples
    oob_samples = [j for j in range(original_dataset_size) if j not in bootstrap_indices]
    oob_freq = [1 if j in oob_samples else 0 for j in range(original_dataset_size)]
    
    plt.bar(range(original_dataset_size), oob_freq, color='lightcoral', edgecolor='black', alpha=0.8)
    plt.xlabel('Original Sample Index')
    plt.ylabel('OOB Status (1=OOB, 0=Included)')
    plt.grid(True, alpha=0.3)
    
    # Add OOB count
    plt.text(0.02, 0.98, f'OOB Count: {len(oob_samples)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_sampling_detailed.png'), dpi=300, bbox_inches='tight')

# 3. Mathematical relationship visualization
plt.figure(figsize=(14, 10))

# Subplot 1: Probability of sample inclusion vs sample size
plt.subplot(2, 2, 1)
plt.title('Probability of Sample Inclusion vs Sample Size', fontsize=14, fontweight='bold')

sample_sizes = np.arange(100, 2001, 100)
prob_inclusion = 1 - np.exp(-bootstrap_sample_size/sample_sizes)
prob_oob = np.exp(-bootstrap_sample_size/sample_sizes)

plt.plot(sample_sizes, prob_inclusion, 'b-', linewidth=2, label='Probability of Inclusion')
plt.plot(sample_sizes, prob_oob, 'r-', linewidth=2, label='Probability of OOB')
plt.axvline(x=original_dataset_size, color='green', linestyle='--', linewidth=2, 
            label=f'Our dataset size: {original_dataset_size}')
plt.xlabel('Original Dataset Size')
plt.ylabel('Probability')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Expected unique samples vs sample size
plt.subplot(2, 2, 2)
plt.title('Expected Unique Samples vs Sample Size', fontsize=14, fontweight='bold')

expected_unique_samples = sample_sizes * (1 - np.exp(-bootstrap_sample_size/sample_sizes))
plt.plot(sample_sizes, expected_unique_samples, 'b-', linewidth=2, label='Expected Unique')
plt.axvline(x=original_dataset_size, color='green', linestyle='--', linewidth=2, 
            label=f'Our dataset size: {original_dataset_size}')
plt.axhline(y=expected_unique, color='red', linestyle='--', linewidth=2, 
            label=f'Our expected unique: {expected_unique:.1f}')
plt.xlabel('Original Dataset Size')
plt.ylabel('Expected Unique Samples')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Expected OOB samples vs sample size
plt.subplot(2, 2, 3)
plt.title('Expected OOB Samples vs Sample Size', fontsize=14, fontweight='bold')

expected_oob_samples = sample_sizes * np.exp(-bootstrap_sample_size/sample_sizes)
plt.plot(sample_sizes, expected_oob_samples, 'r-', linewidth=2, label='Expected OOB')
plt.axvline(x=original_dataset_size, color='green', linestyle='--', linewidth=2, 
            label=f'Our dataset size: {original_dataset_size}')
plt.axhline(y=expected_oob, color='red', linestyle='--', linewidth=2, 
            label=f'Our expected OOB: {expected_oob:.1f}')
plt.xlabel('Original Dataset Size')
plt.ylabel('Expected OOB Samples')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Relationship between unique and OOB
plt.subplot(2, 2, 4)
plt.title('Unique + OOB = Original Dataset Size', fontsize=14, fontweight='bold')

plt.plot(sample_sizes, expected_unique_samples + expected_oob_samples, 'g-', linewidth=2, 
         label='Unique + OOB')
plt.axhline(y=original_dataset_size, color='black', linestyle='--', linewidth=2, 
            label=f'Original dataset size: {original_dataset_size}')
plt.axvline(x=original_dataset_size, color='green', linestyle='--', linewidth=2, 
            label=f'Our dataset size: {original_dataset_size}')
plt.xlabel('Original Dataset Size')
plt.ylabel('Unique + OOB Samples')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mathematical_relationships.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print(f"Files created:")
print(f"1. bagging_analysis_comprehensive.png - Comprehensive analysis overview")
print(f"2. bootstrap_sampling_detailed.png - Detailed bootstrap sampling process")
print(f"3. mathematical_relationships.png - Mathematical relationships and trends")

print(f"\n{'='*80}")
print("SUMMARY OF ANSWERS")
print(f"{'='*80}")
print(f"1. Number of training datasets: {n_trees}")
print(f"2. Expected unique samples per bootstrap: {expected_unique:.2f}")
print(f"3. Expected OOB samples per tree: {expected_oob:.2f}")
print(f"4. OOB samples provide unbiased validation and feature importance assessment")
print(f"{'='*80}")
