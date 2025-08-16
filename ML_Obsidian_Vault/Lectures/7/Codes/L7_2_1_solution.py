import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from collections import Counter
import random

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("BOOTSTRAP SAMPLING IN BAGGING - DETAILED STEP-BY-STEP ANALYSIS")
print("=" * 80)

# 1. What is bootstrap sampling and how does it work?
print("\n" + "="*60)
print("1. WHAT IS BOOTSTRAP SAMPLING AND HOW DOES IT WORK?")
print("="*60)

print("\nBootstrap sampling is a resampling technique that:")
print("- Creates new datasets by randomly sampling with replacement from the original dataset")
print("- Each bootstrap sample has the same size as the original dataset")
print("- Some samples may appear multiple times, while others may not appear at all")
print("- This creates diversity in the training sets for ensemble methods like bagging")

# Create a small example dataset for demonstration
original_data = np.array([10, 20, 30, 40, 50])
print(f"\nOriginal dataset: {original_data}")
print(f"Dataset size: {len(original_data)}")

# Demonstrate bootstrap sampling
print("\nDemonstrating bootstrap sampling:")
for i in range(3):
    bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
    print(f"Bootstrap sample {i+1}: {bootstrap_sample}")

# 2. If you have a dataset with 1000 samples, how many samples will each bootstrap sample contain?
print("\n" + "="*60)
print("2. BOOTSTRAP SAMPLE SIZE ANALYSIS")
print("="*60)

dataset_size = 1000
print(f"\nOriginal dataset size: {dataset_size:,} samples")
print(f"Each bootstrap sample will contain: {dataset_size:,} samples")
print("This is because bootstrap sampling maintains the same size as the original dataset.")

# Demonstrate with different dataset sizes
dataset_sizes = [100, 500, 1000, 5000, 10000]
bootstrap_sizes = []

print("\nBootstrap sample sizes for different dataset sizes:")
for size in dataset_sizes:
    bootstrap_sizes.append(size)
    print(f"Dataset size: {size:,} → Bootstrap sample size: {size:,}")

# 3. What is the expected number of unique samples in each bootstrap sample?
print("\n" + "="*60)
print("3. EXPECTED NUMBER OF UNIQUE SAMPLES IN BOOTSTRAP SAMPLES")
print("="*60)

print("\nThe expected number of unique samples in a bootstrap sample is approximately:")
print("E[unique samples] ≈ n * (1 - 1/e) ≈ n * 0.632")
print("where n is the dataset size and e is Euler's number (≈ 2.718)")

# Calculate expected unique samples for different dataset sizes
print("\nExpected unique samples for different dataset sizes:")
for size in dataset_sizes:
    expected_unique = size * (1 - 1/np.e)
    print(f"Dataset size: {size:,} → Expected unique: {expected_unique:.0f} ({expected_unique/size*100:.1f}%)")

# Demonstrate with simulation
print("\nSimulation results (averaged over 1000 bootstrap samples):")
simulation_results = []
for size in [100, 500, 1000]:
    unique_counts = []
    for _ in range(1000):
        bootstrap_sample = np.random.choice(np.arange(size), size=size, replace=True)
        unique_count = len(np.unique(bootstrap_sample))
        unique_counts.append(unique_count)
    
    avg_unique = np.mean(unique_counts)
    theoretical = size * (1 - 1/np.e)
    simulation_results.append((size, avg_unique, theoretical))
    print(f"Dataset size: {size:,} → Simulated: {avg_unique:.1f}, Theoretical: {theoretical:.1f}")

# 4. Why is bootstrap sampling important for bagging?
print("\n" + "="*60)
print("4. WHY BOOTSTRAP SAMPLING IS IMPORTANT FOR BAGGING")
print("="*60)

print("\nBootstrap sampling is crucial for bagging because it:")
print("1. Creates diverse training sets - different samples in each bootstrap sample")
print("2. Introduces randomness - prevents overfitting to the same training data")
print("3. Enables ensemble learning - multiple models trained on different data")
print("4. Improves generalization - reduces variance in predictions")
print("5. Maintains dataset size - each model gets sufficient training data")

# Visual demonstration of bootstrap sampling
print("\n" + "="*60)
print("VISUAL DEMONSTRATION OF BOOTSTRAP SAMPLING")
print("="*60)

# Create a larger dataset for visualization
n_samples = 100
original_indices = np.arange(n_samples)
print(f"\nCreating visualization with {n_samples} samples...")

# Generate multiple bootstrap samples
n_bootstrap = 10
bootstrap_samples = []
unique_counts = []

for i in range(n_bootstrap):
    bootstrap_sample = np.random.choice(original_indices, size=n_samples, replace=True)
    bootstrap_samples.append(bootstrap_sample)
    unique_counts.append(len(np.unique(bootstrap_sample)))

# Create visualization 1: Bootstrap sampling process
plt.figure(figsize=(15, 10))

# Subplot 1: Original dataset
plt.subplot(2, 2, 1)
plt.scatter(original_indices, [0]*n_samples, s=50, alpha=0.7, color='blue')
plt.title('Original Dataset (100 samples)', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('Position')
plt.grid(True, alpha=0.3)
plt.xlim(-5, 105)

# Subplot 2: First bootstrap sample
plt.subplot(2, 2, 2)
bootstrap_1 = bootstrap_samples[0]
# Count occurrences of each index
counts = Counter(bootstrap_1)
for idx, count in counts.items():
    plt.scatter([idx]*count, range(count), s=50, alpha=0.7, color='red')
plt.title(f'Bootstrap Sample 1\n({len(counts)} unique samples)', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('Occurrence Count')
plt.grid(True, alpha=0.3)
plt.xlim(-5, 105)

# Subplot 3: Multiple bootstrap samples comparison
plt.subplot(2, 2, 3)
unique_counts_array = np.array(unique_counts)
plt.hist(unique_counts_array, bins=range(min(unique_counts), max(unique_counts)+2, 1), 
         alpha=0.7, color='green', edgecolor='black')
plt.axvline(np.mean(unique_counts_array), color='red', linestyle='--', 
            label=f'Mean: {np.mean(unique_counts_array):.1f}')
plt.axvline(n_samples * (1 - 1/np.e), color='blue', linestyle='--', 
            label=f'Theoretical: {n_samples * (1 - 1/np.e):.1f}')
plt.title('Distribution of Unique Samples\nAcross Bootstrap Samples', fontsize=14)
plt.xlabel('Number of Unique Samples')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Sample overlap visualization
plt.subplot(2, 2, 4)
# Show which samples appear in multiple bootstrap samples
sample_frequency = Counter()
for bootstrap_sample in bootstrap_samples:
    for sample in bootstrap_sample:
        sample_frequency[sample] += 1

frequencies = [sample_frequency[i] for i in range(n_samples)]
plt.bar(range(n_samples), frequencies, alpha=0.7, color='orange')
plt.axhline(np.mean(frequencies), color='red', linestyle='--', 
            label=f'Mean frequency: {np.mean(frequencies):.1f}')
plt.title('Sample Frequency Across Bootstrap Samples', fontsize=14)
plt.xlabel('Sample Index')
plt.ylabel('Frequency of Selection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_sampling_overview.png'), dpi=300, bbox_inches='tight')

# Create visualization 2: Detailed bootstrap sample analysis
plt.figure(figsize=(16, 12))

# Show first 5 bootstrap samples in detail
for i in range(5):
    plt.subplot(3, 2, i+1)
    bootstrap_sample = bootstrap_samples[i]
    counts = Counter(bootstrap_sample)
    
    # Create a heatmap-like visualization
    max_count = max(counts.values()) if counts else 0
    for idx in range(n_samples):
        count = counts.get(idx, 0)
        if count > 0:
            plt.bar(idx, count, alpha=0.7, color='red')
    
    plt.title(f'Bootstrap Sample {i+1}\n({len(counts)} unique, {n_samples-len(counts)} missing)', fontsize=12)
    plt.xlabel('Sample Index')
    plt.ylabel('Occurrence Count')
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 105)
    plt.ylim(0, max(5, max_count + 1))

# Add summary statistics
plt.subplot(3, 2, 6)
stats_text = f"""Bootstrap Sampling Statistics:
• Dataset size: {n_samples}
• Number of bootstrap samples: {n_bootstrap}
• Average unique samples: {np.mean(unique_counts):.1f}
• Theoretical expectation: {n_samples * (1 - 1/np.e):.1f}
• Standard deviation: {np.std(unique_counts):.1f}
• Missing samples per bootstrap: {n_samples - np.mean(unique_counts):.1f}"""
plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black"))
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_samples_detailed.png'), dpi=300, bbox_inches='tight')

# Create visualization 3: Mathematical relationship
plt.figure(figsize=(12, 8))

# Theoretical vs simulated results
dataset_sizes_plot = np.array([100, 200, 500, 1000, 2000, 5000])
theoretical_unique = dataset_sizes_plot * (1 - 1/np.e)
theoretical_percentage = theoretical_unique / dataset_sizes_plot * 100

# Simulate for these sizes
simulated_unique = []
simulated_percentage = []
for size in dataset_sizes_plot:
    unique_counts_sim = []
    for _ in range(100):  # 100 simulations per size
        bootstrap_sample = np.random.choice(np.arange(size), size=size, replace=True)
        unique_counts_sim.append(len(np.unique(bootstrap_sample)))
    avg_unique = np.mean(unique_counts_sim)
    simulated_unique.append(avg_unique)
    simulated_percentage.append(avg_unique / size * 100)

plt.subplot(2, 2, 1)
plt.plot(dataset_sizes_plot, theoretical_unique, 'b-', linewidth=2, label='Theoretical')
plt.plot(dataset_sizes_plot, simulated_unique, 'r--', linewidth=2, label='Simulated')
plt.xlabel('Dataset Size')
plt.ylabel('Expected Unique Samples')
plt.title('Unique Samples vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.subplot(2, 2, 2)
plt.plot(dataset_sizes_plot, theoretical_percentage, 'b-', linewidth=2, label='Theoretical')
plt.plot(dataset_sizes_plot, simulated_percentage, 'r--', linewidth=2, label='Simulated')
plt.xlabel('Dataset Size')
plt.ylabel('Percentage of Unique Samples (%)')
plt.title('Percentage of Unique Samples vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.axhline(y=100*(1-1/np.e), color='g', linestyle=':', alpha=0.7, label='Limit: 63.2%')

plt.subplot(2, 2, 3)
# Show the mathematical relationship
x = np.linspace(1, 100, 1000)
y = 1 - 1/x
plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(y=1-1/np.e, color='r', linestyle='--', label=f'Limit: {1-1/np.e:.3f}')
plt.xlabel('Dataset Size')
plt.ylabel('Proportion of Unique Samples')
plt.title('Mathematical Relationship: 1 - 1/n')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.subplot(2, 2, 4)
# Show convergence to the limit
plt.plot(dataset_sizes_plot, np.abs(np.array(simulated_percentage) - 100*(1-1/np.e)), 'g-', linewidth=2)
plt.xlabel('Dataset Size')
plt.ylabel('Absolute Error from Limit (%)')
plt.title('Convergence to Theoretical Limit')
plt.grid(True, alpha=0.3)
plt.xscale('log')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_mathematical_analysis.png'), dpi=300, bbox_inches='tight')

# Create visualization 4: Bagging ensemble visualization
plt.figure(figsize=(14, 10))

# Simulate a simple classification problem
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create decision boundaries for different bootstrap samples
plt.subplot(2, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', alpha=0.6, label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', alpha=0.6, label='Class 1')
plt.title('Original Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Show bootstrap samples
for i in range(3):
    plt.subplot(2, 2, i+2)
    # Create bootstrap sample
    bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
    X_bootstrap = X[bootstrap_indices]
    y_bootstrap = y[bootstrap_indices]
    
    plt.scatter(X_bootstrap[y_bootstrap==0, 0], X_bootstrap[y_bootstrap==0, 1], 
                c='red', alpha=0.6, label='Class 0')
    plt.scatter(X_bootstrap[y_bootstrap==1, 0], X_bootstrap[y_bootstrap==1, 1], 
                c='blue', alpha=0.6, label='Class 1')
    plt.title(f'Bootstrap Sample {i+1}\n({len(np.unique(bootstrap_indices))} unique samples)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bagging_ensemble_visualization.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Summary of key findings
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

print("\n1. Bootstrap Sampling Process:")
print(f"   • Original dataset size: {n_samples}")
print(f"   • Bootstrap sample size: {n_samples} (same as original)")
print(f"   • Average unique samples: {np.mean(unique_counts):.1f}")
print(f"   • Theoretical expectation: {n_samples * (1 - 1/np.e):.1f}")

print("\n2. Mathematical Relationship:")
print(f"   • Expected unique samples: n × (1 - 1/e) ≈ n × 0.632")
print(f"   • For large datasets, approximately 63.2% of samples are unique")
print(f"   • This creates diversity essential for bagging")

print("\n3. Importance for Bagging:")
print("   • Creates diverse training sets")
print("   • Introduces randomness to prevent overfitting")
print("   • Enables effective ensemble learning")
print("   • Maintains sufficient training data for each model")

print("\n4. Practical Implications:")
print("   • Each bootstrap sample contains about 63.2% unique data")
print("   • About 36.8% of data is duplicated")
print("   • This balance provides both diversity and sufficient training data")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
