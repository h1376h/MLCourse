import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def entropy(probabilities):
    """Calculate the entropy of a probability distribution."""
    return -np.sum(probabilities * np.log2(probabilities))

# Step 1: Define the given probability distribution
print_step_header(1, "Given Probability Distribution")

# Given probabilities
P = np.array([0.2, 0.3, 0.4, 0.1])
values = np.array([1, 2, 3, 4])

print("Given probability distribution:")
for value, prob in zip(values, P):
    print(f"P(X = {value}) = {prob}")

# Step 2: Calculate the entropy of the given distribution
print_step_header(2, "Calculating Entropy of Given Distribution")

H_X = entropy(P)
print(f"Entropy H(X) = -Σ p(x) log₂ p(x)")
print(f"= -({P[0]} * log₂({P[0]}) + {P[1]} * log₂({P[1]}) + {P[2]} * log₂({P[2]}) + {P[3]} * log₂({P[3]}))")
print(f"= {H_X:.4f} bits")

# Step 3: Calculate entropy of uniform distribution
print_step_header(3, "Calculating Entropy of Uniform Distribution")

P_uniform = np.array([0.25, 0.25, 0.25, 0.25])
H_uniform = entropy(P_uniform)

print("For uniform distribution over 4 values:")
print(f"P(X = 1) = P(X = 2) = P(X = 3) = P(X = 4) = 0.25")
print(f"Entropy H(X) = -4 * (0.25 * log₂(0.25))")
print(f"= {H_uniform:.4f} bits")

# Step 4: Calculate entropy of deterministic distribution
print_step_header(4, "Calculating Entropy of Deterministic Distribution")

P_deterministic = np.array([0, 0, 1, 0])  # Assuming X=3 is the certain outcome
H_deterministic = entropy(P_deterministic)

print("For deterministic distribution (X = 3 with probability 1):")
print(f"P(X = 3) = 1, P(X = other) = 0")
print(f"Entropy H(X) = -1 * log₂(1) - 0 * log₂(0) - 0 * log₂(0) - 0 * log₂(0)")
print(f"= {H_deterministic:.4f} bits")

# Step 5: Create visualizations
print_step_header(5, "Creating Visualizations")

# Create a figure with 3 subplots
plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

# Plot 1: Given Distribution
ax1 = plt.subplot(gs[0])
ax1.bar(values, P, color='blue', alpha=0.7)
ax1.set_title('Given Distribution', fontsize=12)
ax1.set_xlabel('X', fontsize=10)
ax1.set_ylabel('Probability', fontsize=10)
ax1.set_ylim(0, 0.5)
ax1.grid(True)
ax1.text(0.5, -0.15, f'H(X) = {H_X:.4f} bits', transform=ax1.transAxes, ha='center')

# Plot 2: Uniform Distribution
ax2 = plt.subplot(gs[1])
ax2.bar(values, P_uniform, color='green', alpha=0.7)
ax2.set_title('Uniform Distribution', fontsize=12)
ax2.set_xlabel('X', fontsize=10)
ax2.set_ylabel('Probability', fontsize=10)
ax2.set_ylim(0, 0.5)
ax2.grid(True)
ax2.text(0.5, -0.15, f'H(X) = {H_uniform:.4f} bits', transform=ax2.transAxes, ha='center')

# Plot 3: Deterministic Distribution
ax3 = plt.subplot(gs[2])
ax3.bar(values, P_deterministic, color='red', alpha=0.7)
ax3.set_title('Deterministic Distribution', fontsize=12)
ax3.set_xlabel('X', fontsize=10)
ax3.set_ylabel('Probability', fontsize=10)
ax3.set_ylim(0, 1.1)
ax3.grid(True)
ax3.text(0.5, -0.15, f'H(X) = {H_deterministic:.4f} bits', transform=ax3.transAxes, ha='center')

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "entropy_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Create entropy vs probability plot
print_step_header(6, "Visualizing Entropy vs Probability")

# Create a plot showing how entropy changes with probability
p_values = np.linspace(0.01, 0.99, 100)
entropy_values = -p_values * np.log2(p_values) - (1 - p_values) * np.log2(1 - p_values)

plt.figure(figsize=(10, 6))
plt.plot(p_values, entropy_values, 'b-', linewidth=2)
plt.title('Entropy of a Binary Random Variable', fontsize=14)
plt.xlabel('Probability of Outcome 1', fontsize=12)
plt.ylabel('Entropy (bits)', fontsize=12)
plt.grid(True)

# Mark the maximum entropy point
max_entropy = 1.0
plt.axhline(y=max_entropy, color='r', linestyle='--', label='Maximum Entropy')
plt.axvline(x=0.5, color='r', linestyle='--')

# Add some key points
plt.scatter([0.2, 0.5, 0.8], 
           [-0.2*np.log2(0.2) - 0.8*np.log2(0.8),
            -0.5*np.log2(0.5) - 0.5*np.log2(0.5),
            -0.8*np.log2(0.8) - 0.2*np.log2(0.2)],
           color='red', s=100)

plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "entropy_vs_probability.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion
print_step_header(7, "Conclusion")

print("Key Findings:")
print(f"1. Given Distribution Entropy: {H_X:.4f} bits")
print(f"2. Uniform Distribution Entropy: {H_uniform:.4f} bits")
print(f"3. Deterministic Distribution Entropy: {H_deterministic:.4f} bits")
print("\nThis demonstrates that:")
print("- The uniform distribution has the highest entropy (2 bits)")
print("- The deterministic distribution has the lowest entropy (0 bits)")
print("- The given distribution has entropy between these extremes")
print("\nThis confirms that the uniform distribution has maximum entropy among all distributions over the same set of values.") 