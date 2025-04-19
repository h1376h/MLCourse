import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import norm, entropy

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
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Create different probability distributions with varying entropy
def create_distributions():
    # Distribution 1: Nearly uniform (high entropy)
    dist1 = np.array([0.19, 0.21, 0.18, 0.20, 0.22])
    
    # Distribution 2: Moderately skewed (medium entropy)
    dist2 = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    
    # Distribution 3: Highly skewed (low entropy)
    dist3 = np.array([0.01, 0.04, 0.15, 0.25, 0.55])
    
    # Distribution 4: Almost deterministic (very low entropy)
    dist4 = np.array([0.02, 0.03, 0.05, 0.05, 0.85])
    
    distributions = [dist1, dist2, dist3, dist4]
    labels = ['A', 'B', 'C', 'D']
    
    # Calculate actual entropy values
    entropy_values = [entropy(dist) for dist in distributions]
    
    print("Distribution entropies:")
    for i, (label, h) in enumerate(zip(labels, entropy_values)):
        print(f"Distribution {label}: {h:.4f}")
    
    return distributions, labels, entropy_values

# Create KL divergence visualizations
def create_kl_divergence_plot():
    # Reference distribution (uniform)
    p_uniform = np.ones(5) / 5
    
    # Create distributions with varying divergence from uniform
    distributions = []
    kl_divergences = []
    
    # Create 4 distributions with increasing divergence from uniform
    alphas = [0.5, 1, 2, 5]  # Concentration parameters
    
    for alpha in alphas:
        # Create distribution with varying divergence
        q = np.linspace(1, 5, 5)
        q = q**alpha
        q = q / np.sum(q)
        
        distributions.append(q)
        
        # Calculate KL divergence from uniform
        kl_div = np.sum(p_uniform * np.log(p_uniform / q))
        kl_divergences.append(kl_div)
    
    labels = ['P', 'Q', 'R', 'S']
    
    print("\nKL Divergences from uniform:")
    for i, (label, kl) in enumerate(zip(labels, kl_divergences)):
        print(f"Distribution {label}: {kl:.4f}")
    
    return distributions, labels, kl_divergences, p_uniform

# Create mutual information visualization
def create_mutual_info_visualization():
    # Create 4 joint distributions with varying levels of dependence
    # Values represent P(X=x, Y=y) for different x,y pairs
    
    # Distribution 1: Independent (MI = 0)
    x_marginal = np.array([0.3, 0.7])
    y_marginal = np.array([0.4, 0.6])
    joint1 = np.outer(x_marginal, y_marginal)
    
    # Distribution 2: Slight dependence
    joint2 = np.array([
        [0.15, 0.20],
        [0.25, 0.40]
    ])
    
    # Distribution 3: Moderate dependence
    joint3 = np.array([
        [0.25, 0.10],
        [0.15, 0.50]
    ])
    
    # Distribution 4: Strong dependence (nearly deterministic)
    joint4 = np.array([
        [0.40, 0.05],
        [0.05, 0.50]
    ])
    
    joint_distributions = [joint1, joint2, joint3, joint4]
    labels = ['W', 'X', 'Y', 'Z']
    
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
    
    print("\nMutual Information values:")
    for i, (label, mi) in enumerate(zip(labels, mutual_info)):
        print(f"Joint Distribution {label}: {mi:.4f}")
    
    return joint_distributions, labels, mutual_info

# Function to plot histograms for distributions
def plot_distributions(distributions, labels, entropy_values):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
    
    for i, (dist, label, h) in enumerate(zip(distributions, labels, entropy_values)):
        ax = axes[i]
        ax.bar(categories, dist, alpha=0.7, color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_title(f"Distribution {label}", fontsize=14)
        ax.set_ylabel("Probability", fontsize=12)
        
        # Add entropy value as text
        # Intentionally not showing the exact value in the question
        if i == 0:
            ax.text(0.05, 0.9, "Entropy = ?", transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.05, 0.9, "Entropy = ?", transform=ax.transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "entropy_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot KL divergence visualizations
def plot_kl_divergence(distributions, labels, kl_values, p_uniform):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
    
    for i, (dist, label, kl) in enumerate(zip(distributions, labels, kl_values)):
        ax = axes[i]
        
        # Plot uniform distribution
        ax.bar(categories, p_uniform, alpha=0.4, color='lightgray', label='Uniform')
        
        # Plot the distribution
        ax.bar(categories, dist, alpha=0.7, color='coral', label=f'Distribution {label}')
        
        ax.set_ylim(0, 0.6)
        ax.set_title(f"Distribution {label} vs. Uniform", fontsize=14)
        ax.set_ylabel("Probability", fontsize=12)
        ax.legend()
        
        # Add KL divergence as text
        # Intentionally not showing the exact value in the question
        ax.text(0.05, 0.9, "KL(Uniform || Dist) = ?", transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kl_divergence.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot joint distributions and mutual information
def plot_mutual_info(joint_distributions, labels, mi_values):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (joint, label, mi) in enumerate(zip(joint_distributions, labels, mi_values)):
        ax = axes[i]
        
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
        # Intentionally not showing the exact value in the question
        ax.text(0.05, 0.05, "I(X;Y) = ?", transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mutual_information.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Generate all visualizations
print("Generating visualizations for Information Theory Quiz...")

# Create and plot entropy distributions
print("\nPart 1: Entropy Distributions")
distributions, dist_labels, entropy_values = create_distributions()
plot_distributions(distributions, dist_labels, entropy_values)

# Create and plot KL divergence
print("\nPart 2: KL Divergence Visualizations")
kl_distributions, kl_labels, kl_values, p_uniform = create_kl_divergence_plot()
plot_kl_divergence(kl_distributions, kl_labels, kl_values, p_uniform)

# Create and plot mutual information
print("\nPart 3: Mutual Information Visualizations")
joint_distributions, mi_labels, mi_values = create_mutual_info_visualization()
plot_mutual_info(joint_distributions, mi_labels, mi_values)

print(f"\nAll visualizations saved to: {save_dir}") 