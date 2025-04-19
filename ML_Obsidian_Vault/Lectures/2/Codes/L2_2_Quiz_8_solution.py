import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("This problem requires analyzing visualizations of three key information theory concepts:")
print("1. Entropy: A measure of uncertainty or randomness in a probability distribution")
print("2. KL Divergence: A measure of how one probability distribution differs from another")
print("3. Mutual Information: A measure of the mutual dependence between two random variables\n")

print("The task is to rank these measures for different distributions shown in visualizations.")

# Step 2: Recreate the distributions from the question
print_step_header(2, "Analyzing Entropy Distributions")

# Define distributions from question
# Distribution 1: Nearly uniform (high entropy)
dist_A = np.array([0.19, 0.21, 0.18, 0.20, 0.22])

# Distribution 2: Moderately skewed (medium entropy)
dist_B = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

# Distribution 3: Highly skewed (low entropy)
dist_C = np.array([0.01, 0.04, 0.15, 0.25, 0.55])

# Distribution 4: Almost deterministic (very low entropy)
dist_D = np.array([0.02, 0.03, 0.05, 0.05, 0.85])

distributions = [dist_A, dist_B, dist_C, dist_D]
dist_labels = ['A', 'B', 'C', 'D']

# Calculate entropy values
entropy_values = [entropy(dist) for dist in distributions]

print("Entropy values for each distribution:")
for label, h in zip(dist_labels, entropy_values):
    print(f"Distribution {label}: {h:.4f}")

# Create sorted list for ranking
entropy_ranking = sorted(zip(dist_labels, entropy_values), key=lambda x: x[1], reverse=True)
print("\nRanking from highest to lowest entropy:")
print(" > ".join([f"{label} ({h:.4f})" for label, h in entropy_ranking]))

# Create visual with entropy values
plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2)

categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']

for i, (dist, label, h) in enumerate(zip(distributions, dist_labels, entropy_values)):
    ax = plt.subplot(gs[i])
    ax.bar(categories, dist, alpha=0.7, color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_title(f"Distribution {label}", fontsize=14)
    ax.set_ylabel("Probability", fontsize=12)
    
    # Add entropy value as text
    ax.text(0.05, 0.9, f"Entropy = {h:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "entropy_solution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 3: Analyze KL Divergence 
print_step_header(3, "Analyzing KL Divergence")

# Reference distribution (uniform)
p_uniform = np.ones(5) / 5

# Define the distributions from the question
alphas = [0.5, 1, 2, 5]  # Concentration parameters
kl_distributions = []

for alpha in alphas:
    # Create distribution with varying divergence
    q = np.linspace(1, 5, 5)
    q = q**alpha
    q = q / np.sum(q)
    kl_distributions.append(q)

kl_labels = ['P', 'Q', 'R', 'S']

# Calculate KL divergences
kl_values = []
for dist in kl_distributions:
    # KL(p||q)
    kl_div = np.sum(p_uniform * np.log(p_uniform / dist))
    kl_values.append(kl_div)

print("KL Divergence values from uniform distribution:")
for label, kl in zip(kl_labels, kl_values):
    print(f"KL(uniform || Distribution {label}) = {kl:.4f}")

# Create sorted list for ranking
kl_ranking = sorted(zip(kl_labels, kl_values), key=lambda x: x[1])
print("\nRanking from smallest to largest KL divergence:")
print(" < ".join([f"{label} ({kl:.4f})" for label, kl in kl_ranking]))

# Create visual with KL divergence values
plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2)

categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']

for i, (dist, label, kl) in enumerate(zip(kl_distributions, kl_labels, kl_values)):
    ax = plt.subplot(gs[i])
    
    # Plot uniform distribution
    ax.bar(categories, p_uniform, alpha=0.4, color='lightgray', label='Uniform')
    
    # Plot the distribution
    ax.bar(categories, dist, alpha=0.7, color='coral', label=f'Distribution {label}')
    
    ax.set_ylim(0, 0.6)
    ax.set_title(f"Distribution {label} vs. Uniform", fontsize=14)
    ax.set_ylabel("Probability", fontsize=12)
    ax.legend()
    
    # Add KL divergence as text
    ax.text(0.05, 0.9, f"KL(Uniform || Dist) = {kl:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "kl_divergence_solution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Analyze Mutual Information
print_step_header(4, "Analyzing Mutual Information")

# Create joint distributions from the question
# Distribution 1: Independent (MI = 0)
x_marginal = np.array([0.3, 0.7])
y_marginal = np.array([0.4, 0.6])
joint_W = np.outer(x_marginal, y_marginal)

# Distribution 2: Slight dependence
joint_X = np.array([
    [0.15, 0.20],
    [0.25, 0.40]
])

# Distribution 3: Moderate dependence
joint_Y = np.array([
    [0.25, 0.10],
    [0.15, 0.50]
])

# Distribution 4: Strong dependence
joint_Z = np.array([
    [0.40, 0.05],
    [0.05, 0.50]
])

joint_distributions = [joint_W, joint_X, joint_Y, joint_Z]
mi_labels = ['W', 'X', 'Y', 'Z']

# Calculate mutual information for each
mutual_info = []

for joint in joint_distributions:
    # Calculate marginals
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)
    
    # Calculate entropies
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    
    # Reshape joint for entropy calculation
    joint_flat = joint.flatten()
    h_xy = entropy(joint_flat)
    
    # Calculate mutual information
    mi = h_x + h_y - h_xy
    mutual_info.append(mi)

print("Mutual Information values for each joint distribution:")
for label, mi in zip(mi_labels, mutual_info):
    print(f"Joint Distribution {label}: {mi:.4f}")

# Create sorted list for ranking
mi_ranking = sorted(zip(mi_labels, mutual_info), key=lambda x: x[1])
print("\nRanking from lowest to highest mutual information:")
print(" < ".join([f"{label} ({mi:.4f})" for label, mi in mi_ranking]))

# Create visual with mutual information values
plt.figure(figsize=(12, 10))
gs = GridSpec(2, 2)

for i, (joint, label, mi) in enumerate(zip(joint_distributions, mi_labels, mutual_info)):
    ax = plt.subplot(gs[i])
    
    # Plot joint distribution as a heatmap
    im = ax.imshow(joint, cmap='YlOrRd', vmin=0, vmax=0.5)
    
    # Add text annotations
    for row in range(joint.shape[0]):
        for col in range(joint.shape[1]):
            ax.text(col, row, f"{joint[row, col]:.2f}", 
                    ha="center", va="center", color="black" if joint[row, col] < 0.3 else "white")
    
    ax.set_title(f"Joint Distribution {label}", fontsize=14)
    ax.set_xlabel("Y", fontsize=12)
    ax.set_ylabel("X", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Y=0', 'Y=1'])
    ax.set_yticklabels(['X=0', 'X=1'])
    
    # Add mutual information as text
    ax.text(0.05, 0.05, f"I(X;Y) = {mi:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mutual_information_solution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 5: Create a summary figure
print_step_header(5, "Creating Summary of Information Theory Relationships")

plt.figure(figsize=(10, 8))

# 1. Entropy vs uniformity
plt.subplot(3, 1, 1)
x = np.arange(len(dist_labels))
plt.bar(x, entropy_values, alpha=0.7, color='skyblue')
plt.xticks(x, dist_labels)
plt.ylabel("Entropy")
plt.title("Entropy Decreases as Distribution Becomes Less Uniform")

for i, v in enumerate(entropy_values):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')

# 2. KL divergence vs distance from uniform
plt.subplot(3, 1, 2)
x = np.arange(len(kl_labels))
plt.bar(x, kl_values, alpha=0.7, color='coral')
plt.xticks(x, kl_labels)
plt.ylabel("KL Divergence")
plt.title("KL Divergence Increases with Distance from Uniform Distribution")

for i, v in enumerate(kl_values):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')

# 3. Mutual information vs dependency
plt.subplot(3, 1, 3)
x = np.arange(len(mi_labels))
plt.bar(x, mutual_info, alpha=0.7, color='lightgreen')
plt.xticks(x, mi_labels)
plt.ylabel("Mutual Information")
plt.title("Mutual Information Increases with Dependency Between Variables")

for i, v in enumerate(mutual_info):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "information_theory_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll solution visualizations saved to: {save_dir}")
print("\nSolution:")
print("1. Entropy ranking (highest to lowest): A > B > C > D")
print("2. KL divergence ranking (smallest to largest): P < Q < R < S")
print("3. Mutual information ranking (lowest to highest): W < X < Y < Z")
print("4. These information-theoretic measures can be visually interpreted from the distributions:")
print("   - More uniform distributions have higher entropy")
print("   - More skewed/concentrated distributions have lower entropy")
print("   - Distributions more different from uniform have higher KL divergence")
print("   - Joint distributions with stronger dependency patterns have higher mutual information") 