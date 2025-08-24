import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter
import random

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 6: Manual Bootstrap Sampling")
print("=" * 50)

# Original dataset
D = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
print(f"Original dataset D = {D}")
print(f"Dataset size: {len(D)} samples")

# Given bootstrap indices
bootstrap_indices = [4, 2, 4, 1, 6, 2]
print(f"\nBootstrap indices (die rolls): {bootstrap_indices}")

# Step 1: Create the first bootstrap sample D1
print("\n" + "="*50)
print("Step 1: Creating Bootstrap Sample D1")
print("="*50)

D1 = [D[i-1] for i in bootstrap_indices]  # Convert to 0-based indexing
print(f"Bootstrap sample D1 = {D1}")

# Count occurrences of each sample
sample_counts = Counter(D1)
print(f"\nSample counts in D1:")
for sample, count in sorted(sample_counts.items()):
    print(f"  {sample}: {count} times")

# Step 2: Identify out-of-bag samples
print("\n" + "="*50)
print("Step 2: Identifying Out-of-Bag (OOB) Samples")
print("="*50)

# Find samples that were not selected
all_selected = set(D1)
oob_samples = set(D) - all_selected

print(f"All selected samples: {sorted(all_selected)}")
print(f"Out-of-bag samples: {sorted(oob_samples)}")
print(f"Number of OOB samples: {len(oob_samples)}")

# Step 3: Analyze possibility of single unique sample
print("\n" + "="*50)
print("Step 3: Possibility of Single Unique Sample")
print("="*50)

print("Is it possible to have only one unique sample repeated 6 times?")
print("Let's analyze this step by step:")

# Calculate probability of getting all same samples
prob_all_same = (1/6)**5  # First sample is fixed, need 5 more to match
print(f"Probability of getting all same samples: (1/6)^5 = {prob_all_same:.6f}")

# Show all possible outcomes
print(f"\nTotal possible bootstrap samples: 6^6 = {6**6:,}")
print(f"Number of samples with all same value: 6 (one for each possible sample)")
print(f"Probability = 6/6^6 = {6/(6**6):.6f}")

# Step 4: Probability of specific sample not being selected
print("\n" + "="*50)
print("Step 4: Probability of Specific Sample Not Being Selected")
print("="*50)

target_sample = 'S3'
print(f"Target sample: {target_sample}")

# Calculate probability step by step
prob_not_selected_single_draw = 5/6  # 5 out of 6 outcomes don't select S3
prob_not_selected_6_draws = (5/6)**6

print(f"Probability of not selecting {target_sample} in a single draw: 5/6 = {prob_not_selected_single_draw:.4f}")
print(f"Probability of not selecting {target_sample} in 6 draws: (5/6)^6 = {prob_not_selected_6_draws:.4f}")

# Verify with simulation
print(f"\nVerifying with simulation...")
n_simulations = 100000
not_selected_count = 0

for _ in range(n_simulations):
    bootstrap_sample = [random.randint(1, 6) for _ in range(6)]
    if 3 not in bootstrap_sample:  # S3 corresponds to index 3
        not_selected_count += 1

simulated_prob = not_selected_count / n_simulations
print(f"Simulated probability: {simulated_prob:.4f}")
print(f"Theoretical probability: {prob_not_selected_6_draws:.4f}")
print(f"Difference: {abs(simulated_prob - prob_not_selected_6_draws):.6f}")

# Create visualizations
print("\n" + "="*50)
print("Creating Visualizations")
print("="*50)

# Visualization 1: Bootstrap sample composition
plt.figure(figsize=(12, 8))

# Create subplot for bootstrap sample composition
plt.subplot(2, 2, 1)
samples = list(sample_counts.keys())
counts = list(sample_counts.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(samples)))

bars = plt.bar(samples, counts, color=colors, edgecolor='black', alpha=0.8)
plt.title('Bootstrap Sample D1 Composition', fontsize=14, fontweight='bold')
plt.xlabel('Samples')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             str(count), ha='center', va='bottom', fontweight='bold')

# Visualization 2: Original vs Bootstrap sample comparison
plt.subplot(2, 2, 2)
original_counts = [1] * len(D)
bootstrap_counts = [sample_counts.get(sample, 0) for sample in D]

x = np.arange(len(D))
width = 0.35

plt.bar(x - width/2, original_counts, width, label='Original Dataset', 
        color='lightblue', edgecolor='black', alpha=0.8)
plt.bar(x + width/2, bootstrap_counts, width, label='Bootstrap Sample D1', 
        color='lightcoral', edgecolor='black', alpha=0.8)

plt.title('Original vs Bootstrap Sample Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Samples')
plt.ylabel('Frequency')
plt.xticks(x, D)
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization 3: Out-of-bag analysis
plt.subplot(2, 2, 3)
oob_labels = ['In Sample', 'Out-of-Bag']
oob_counts = [len(all_selected), len(oob_samples)]
oob_colors = ['lightgreen', 'lightcoral']

plt.pie(oob_counts, labels=oob_labels, autopct='%1.1f%%', colors=oob_colors,
        startangle=90, explode=(0.05, 0.05))
plt.title('Sample Selection Analysis', fontsize=14, fontweight='bold')

# Visualization 4: Probability analysis
plt.subplot(2, 2, 4)
prob_labels = ['Selected', 'Not Selected']
prob_values = [1 - prob_not_selected_6_draws, prob_not_selected_6_draws]
prob_colors = ['lightblue', 'lightcoral']

plt.pie(prob_values, labels=prob_labels, autopct='%1.3f', colors=prob_colors,
        startangle=90, explode=(0.05, 0.05))
plt.title(f'Probability of {target_sample} Selection', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_analysis_overview.png'), dpi=300, bbox_inches='tight')

# Create detailed step-by-step visualization
plt.figure(figsize=(15, 10))

# Step-by-step bootstrap process
plt.subplot(2, 3, 1)
plt.title('Step 1: Die Rolls', fontsize=12, fontweight='bold')
die_faces = [1, 2, 3, 4, 5, 6]
die_counts = [bootstrap_indices.count(i) for i in die_faces]
plt.bar(die_faces, die_counts, color='gold', edgecolor='black', alpha=0.8)
plt.xlabel('Die Face')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Step 2: Index to sample mapping
plt.subplot(2, 3, 2)
plt.title('Step 2: Index to Sample Mapping', fontsize=12, fontweight='bold')
mapping_data = list(zip(bootstrap_indices, D1))
for i, (idx, sample) in enumerate(mapping_data):
    plt.annotate(f'{idx}→{sample}', (i, 0.5), ha='center', va='center', 
                 fontsize=10, fontweight='bold')
plt.xlim(-0.5, len(mapping_data)-0.5)
plt.ylim(0, 1)
plt.xticks(range(len(mapping_data)), [f'Roll {i+1}' for i in range(len(mapping_data))])
plt.yticks([])

# Step 3: Final bootstrap sample
plt.subplot(2, 3, 3)
plt.title('Step 3: Final Bootstrap Sample D1', fontsize=12, fontweight='bold')
plt.bar(range(len(D1)), [1]*len(D1), color=[colors[list(sample_counts.keys()).index(sample)] for sample in D1])
plt.xticks(range(len(D1)), D1)
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

# Step 4: OOB analysis
plt.subplot(2, 3, 4)
plt.title('Step 4: Out-of-Bag Analysis', fontsize=12, fontweight='bold')
oob_data = [(sample, 'In Sample' if sample in all_selected else 'OOB') for sample in D]
oob_colors = ['lightgreen' if status == 'In Sample' else 'lightcoral' for _, status in oob_data]
oob_labels = [sample for sample, _ in oob_data]

plt.bar(range(len(D)), [1]*len(D), color=oob_colors, edgecolor='black')
plt.xticks(range(len(D)), oob_labels)
plt.ylabel('Status')
plt.grid(True, alpha=0.3)

# Step 5: Probability calculation
plt.subplot(2, 3, 5)
plt.title('Step 5: Probability Calculation', fontsize=12, fontweight='bold')
prob_steps = ['P(not S3 in 1 draw)', 'P(not S3 in 6 draws)']
prob_values = [5/6, (5/6)**6]
prob_colors = ['lightblue', 'lightcoral']

bars = plt.bar(prob_steps, prob_values, color=prob_colors, edgecolor='black', alpha=0.8)
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, prob_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Step 6: Simulation verification
plt.subplot(2, 3, 6)
plt.title('Step 6: Simulation Verification', fontsize=12, fontweight='bold')
sim_labels = ['Theoretical', 'Simulated']
sim_values = [prob_not_selected_6_draws, simulated_prob]
sim_colors = ['lightblue', 'lightgreen']

bars = plt.bar(sim_labels, sim_values, color=sim_colors, edgecolor='black', alpha=0.8)
plt.ylabel('Probability')
plt.ylim(0, max(sim_values) * 1.1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, sim_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bootstrap_step_by_step.png'), dpi=300, bbox_inches='tight')



print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Results:")
print(f"1. Bootstrap sample D1: {D1}")
print(f"2. Out-of-bag samples: {sorted(oob_samples)}")
print(f"3. Probability of single unique sample: {prob_all_same:.6f}")
print(f"4. Probability of S3 not being selected: {prob_not_selected_6_draws:.4f}")
print(f"   (Verified with simulation: {simulated_prob:.4f})")
