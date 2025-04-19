import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.stats import entropy
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker

# Set seed for reproducibility
np.random.seed(42)

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False
})

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def analyze_channel_information():
    """Analyze channel information capacity and print explanations."""
    # Create 4 communication channels with different noise levels
    # Each row represents P(Y|X=x) for a specific x
    channels = {
        'A': np.array([
            [0.9, 0.1],  # P(Y|X=0)
            [0.1, 0.9]   # P(Y|X=1)
        ]),
        'B': np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ]),
        'C': np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ]),
        'D': np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
    }
    
    # Create prior distributions P(X) for each channel
    priors = {
        'A': np.array([0.5, 0.5]),      # Uniform
        'B': np.array([0.8, 0.2]),      # Skewed
        'C': np.array([0.3, 0.7]),      # Moderately skewed
        'D': np.array([0.5, 0.5])       # Uniform
    }
    
    # Calculate mutual information for each channel
    mutual_info = {}
    conditional_entropy = {}
    joint_entropies = {}
    marginal_entropy_y = {}
    
    for name, channel in channels.items():
        prior = priors[name]
        
        # Joint probability P(X,Y)
        joint = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                joint[i, j] = prior[i] * channel[i, j]
        
        # Marginal P(Y)
        marginal_y = joint.sum(axis=0)
        
        # Conditional entropy H(Y|X)
        h_y_given_x = 0
        for i in range(2):
            h_y_given_x += prior[i] * entropy(channel[i], base=2)
        conditional_entropy[name] = h_y_given_x
        
        # Entropy of Y
        h_y = entropy(marginal_y, base=2)
        marginal_entropy_y[name] = h_y
        
        # Joint entropy H(X,Y)
        joint_flat = joint.flatten()
        joint_flat = joint_flat[joint_flat > 0]  # Remove zeros to avoid log(0)
        h_xy = -np.sum(joint_flat * np.log2(joint_flat))
        joint_entropies[name] = h_xy
        
        # Mutual information I(X;Y) = H(Y) - H(Y|X)
        mutual_info[name] = h_y - h_y_given_x
    
    # Print analysis for channel information capacity
    print("Analysis of Channel Information Capacity:\n")
    
    print("Channel Properties:")
    print("-" * 70)
    print(f"{'Channel':<10} {'P(Y=0|X=0)':<12} {'P(Y=1|X=0)':<12} {'P(Y=0|X=1)':<12} {'P(Y=1|X=1)':<12} {'Mutual Info':<15} {'Cond. Entropy':<15}")
    print("-" * 70)
    
    for name in channels.keys():
        ch = channels[name]
        print(f"{name:<10} {ch[0,0]:<12.2f} {ch[0,1]:<12.2f} {ch[1,0]:<12.2f} {ch[1,1]:<12.2f} {mutual_info[name]:<15.4f} {conditional_entropy[name]:<15.4f}")
    
    print("-" * 70)
    
    # Print explanation for highest information capacity
    best_channel = max(mutual_info, key=mutual_info.get)
    print(f"\n1. Highest Information Transmission Capacity: Channel {best_channel}")
    print(f"   - Mutual Information: {mutual_info[best_channel]:.4f} bits")
    print(f"   - Channel {best_channel} has the highest mutual information because it has the strongest")
    print(f"     probability (0.9) of correctly transmitting bits. This means that observing")
    print(f"     the output provides substantial information about what the input was.")
    print(f"   - Mutual information quantifies how much information about X is gained")
    print(f"     by observing Y. Higher values indicate better information transmission.")
    
    # Print channel ranking
    ranked_channels = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)
    print("\n2. Ranking Channels by Mutual Information (highest to lowest):")
    for i, (ch, mi) in enumerate(ranked_channels):
        print(f"   {i+1}. Channel {ch}: {mi:.4f} bits")
    
    # Print explanation for random channel
    random_channel = min(mutual_info, key=mutual_info.get)
    print(f"\n3. Completely Random Channel: Channel {random_channel}")
    print(f"   - Mutual Information: {mutual_info[random_channel]:.4f} bits")
    print(f"   - Channel {random_channel} can be described as 'completely random' because:")
    print(f"     a) Its transition probabilities are all 0.5, meaning the output Y is")
    print(f"        equally likely to be 0 or 1 regardless of the input X.")
    print(f"     b) Its mutual information is exactly 0 bits, indicating that knowing")
    print(f"        the output Y provides absolutely no information about what the input X was.")
    print(f"     c) Its conditional entropy H(Y|X) is {conditional_entropy[random_channel]:.4f} bit, which is the maximum")
    print(f"        possible entropy for a binary variable.")
    print(f"   - In information-theoretic terms, this represents a channel where the output")
    print(f"     is completely independent of the input, essentially adding maximum noise to the signal.")
    
    # Create visualization 1: Entropy diagram showing relationship between different entropies
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Choose channels A and D to contrast
    channel_A = 'A'
    channel_D = 'D'
    
    # Data for visualization
    h_x_A = entropy(priors[channel_A], base=2)
    h_x_D = entropy(priors[channel_D], base=2)
    
    # Create the visual representation for both channels in a 2x1 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Channel A - High mutual information
    ax1.set_title(f"Channel A: High Mutual Information ({mutual_info[channel_A]:.4f} bits)")
    
    # Draw circles for entropies
    circle_x = plt.Circle((0.3, 0.5), 0.25, fc='lightblue', alpha=0.7, label='H(X)')
    circle_y = plt.Circle((0.7, 0.5), 0.25, fc='lightgreen', alpha=0.7, label='H(Y)')
    
    ax1.add_patch(circle_x)
    ax1.add_patch(circle_y)
    
    # Annotate entropies
    ax1.annotate(f"H(X) = {h_x_A:.2f}", xy=(0.15, 0.75))
    ax1.annotate(f"H(Y) = {marginal_entropy_y[channel_A]:.2f}", xy=(0.7, 0.75))
    ax1.annotate(f"H(X,Y) = {joint_entropies[channel_A]:.2f}", xy=(0.4, 0.25))
    ax1.annotate(f"I(X;Y) = {mutual_info[channel_A]:.2f}", xy=(0.45, 0.54), 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
    ax1.annotate(f"H(Y|X) = {conditional_entropy[channel_A]:.2f}", xy=(0.65, 0.35))
    
    # Add a label for the mutual information (intersection)
    ax1.annotate("Mutual\nInformation", xy=(0.5, 0.5), xytext=(0.5, 0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                ha='center', va='center', fontsize=11)
    
    ax1.axis('equal')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_axis_off()
    
    # Channel D - Zero mutual information
    ax2.set_title(f"Channel D: Zero Mutual Information ({mutual_info[channel_D]:.4f} bits)")
    
    # Draw circles for entropies (non-overlapping for Channel D)
    circle_x_D = plt.Circle((0.3, 0.5), 0.25, fc='lightblue', alpha=0.7)
    circle_y_D = plt.Circle((0.7, 0.5), 0.25, fc='lightgreen', alpha=0.7)
    
    ax2.add_patch(circle_x_D)
    ax2.add_patch(circle_y_D)
    
    # Annotate entropies
    ax2.annotate(f"H(X) = {h_x_D:.2f}", xy=(0.15, 0.75))
    ax2.annotate(f"H(Y) = {marginal_entropy_y[channel_D]:.2f}", xy=(0.7, 0.75))
    ax2.annotate(f"H(X,Y) = {joint_entropies[channel_D]:.2f}", xy=(0.4, 0.25))
    ax2.annotate("I(X;Y) = 0", xy=(0.5, 0.5), xytext=(0.5, 0.15),
                ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.annotate(f"H(Y|X) = {conditional_entropy[channel_D]:.2f}", xy=(0.65, 0.35))
    
    # Add an explanatory annotation for Channel D
    ax2.annotate("No overlapping information\n(X and Y are independent)", 
                 xy=(0.5, 0.5), xytext=(0.5, 0.85),
                 ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="pink", alpha=0.3))
    
    ax2.axis('equal')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mutual_information_venn.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 2: Bar chart comparing all entropy components across channels
    fig, ax = plt.subplots(figsize=(12, 7))
    
    channel_names = list(channels.keys())
    x = np.arange(len(channel_names))
    width = 0.2
    
    # Compute H(X) for each channel
    h_x_values = [entropy(priors[name], base=2) for name in channel_names]
    
    # Plot bars for each entropy component
    bars1 = ax.bar(x - width*1.5, h_x_values, width, label='H(X) - Input Entropy', color='lightblue')
    bars2 = ax.bar(x - width/2, [marginal_entropy_y[name] for name in channel_names], width, 
                   label='H(Y) - Output Entropy', color='lightgreen')
    bars3 = ax.bar(x + width/2, [joint_entropies[name] for name in channel_names], width, 
                   label='H(X,Y) - Joint Entropy', color='salmon')
    bars4 = ax.bar(x + width*1.5, [mutual_info[name] for name in channel_names], width, 
                   label='I(X;Y) - Mutual Information', color='gold')
    
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Entropy Components Across Different Channels')
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names)
    ax.legend()
    
    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation for Channel D
    ax.annotate('Zero mutual\ninformation', xy=(3 + width*1.5, 0.05), 
                xytext=(3.2, 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    # Add annotation for Channel A
    ax.annotate('Highest mutual\ninformation', xy=(0 + width*1.5, mutual_info['A']), 
                xytext=(0.5, mutual_info['A'] + 0.25),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_components.png'), dpi=300, bbox_inches='tight')
    
    return mutual_info, conditional_entropy, joint_entropies, marginal_entropy_y

def analyze_entropy_and_kl_divergence():
    """Analyze entropy and KL divergence and print explanations."""
    # Define 4 different distributions over the same 4 events
    distributions = {
        'P': np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform
        'Q': np.array([0.4, 0.3, 0.2, 0.1]),      # Skewed
        'R': np.array([0.7, 0.1, 0.1, 0.1]),      # Highly skewed
        'S': np.array([0.1, 0.2, 0.3, 0.4])       # Increasing
    }
    
    # Calculate entropy for each distribution
    entropies = {name: entropy(dist, base=2) for name, dist in distributions.items()}
    
    # Calculate KL divergence for specific pairs
    kl_divergences = {}
    for name1, dist1 in distributions.items():
        kl_divergences[name1] = {}
        for name2, dist2 in distributions.items():
            kl_divergences[name1][name2] = entropy(dist1, dist2, base=2)
    
    # Print entropy analysis
    print("\nAnalysis of Entropy and Probability Distributions:\n")
    
    print("Distributions:")
    print("-" * 60)
    print(f"{'Distribution':<15} {'Event 1':<10} {'Event 2':<10} {'Event 3':<10} {'Event 4':<10} {'Entropy':<10}")
    print("-" * 60)
    
    for name, dist in distributions.items():
        print(f"{name:<15} {dist[0]:<10.2f} {dist[1]:<10.2f} {dist[2]:<10.2f} {dist[3]:<10.2f} {entropies[name]:<10.4f}")
    
    print("-" * 60)
    
    # Find distribution with lowest entropy
    min_entropy_dist = min(entropies, key=entropies.get)
    max_entropy_dist = max(entropies, key=entropies.get)
    
    print(f"\n4. Distribution with Lowest Entropy: Distribution {min_entropy_dist}")
    print(f"   - Entropy: {entropies[min_entropy_dist]:.4f} bits")
    print(f"   - Distribution {min_entropy_dist} {distributions[min_entropy_dist]} has the lowest entropy because:")
    print(f"     a) It is the most 'certain' or least uniform distribution, with probability 0.7")
    print(f"        concentrated on a single outcome (Event 1).")
    print(f"     b) It has less uncertainty than the other distributions.")
    print(f"     c) Entropy measures the average information content or 'surprise' in a probability")
    print(f"        distribution. More concentrated distributions have lower entropy.")
    
    print(f"\n   In contrast, Distribution {max_entropy_dist} has the highest entropy ({entropies[max_entropy_dist]:.4f} bits)")
    print(f"   because it's a uniform distribution with equal probabilities for all outcomes,")
    print(f"   representing maximum uncertainty.")
    
    # Print KL divergence analysis
    print("\nKL Divergence Analysis:\n")
    
    print("KL Divergence Matrix (D_KL(P||Q)):")
    print("-" * 50)
    print(f"{'D_KL(row||col)':<15} {'P':<10} {'Q':<10} {'R':<10} {'S':<10}")
    print("-" * 50)
    
    for name1 in distributions.keys():
        row_values = [f"{kl_divergences[name1][name2]:.4f}" for name2 in distributions.keys()]
        print(f"{name1:<15} {row_values[0]:<10} {row_values[1]:<10} {row_values[2]:<10} {row_values[3]:<10}")
    
    print("-" * 50)
    
    # Focus on P and Q for asymmetry explanation
    kl_p_q = kl_divergences['P']['Q']
    kl_q_p = kl_divergences['Q']['P']
    
    print(f"\n5. Asymmetry of KL Divergence:")
    print(f"   - D_KL(P||Q) = {kl_p_q:.4f} bits")
    print(f"   - D_KL(Q||P) = {kl_q_p:.4f} bits")
    print(f"   - The KL divergence is asymmetric because it measures the extra bits needed")
    print(f"     to encode samples from distribution P using an optimal code designed for")
    print(f"     distribution Q. This is fundamentally an asymmetric operation.")
    print(f"   - Mathematically, D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)),")
    print(f"     which is different from D_KL(Q||P) = Σ Q(x) log(Q(x)/P(x)).")
    print(f"   - The asymmetry can be interpreted as:")
    print(f"     * D_KL(P||Q): How well Q approximates P")
    print(f"     * D_KL(Q||P): How well P approximates Q")
    print(f"   - This asymmetry is clearly visible in the KL divergence matrix, where")
    print(f"     off-diagonal elements across the main diagonal have different values.")
    
    # Create visualization 3: Entropy comparison across distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart of entropy values
    dist_names = list(distributions.keys())
    entropy_values = [entropies[name] for name in dist_names]
    
    bars = ax.bar(dist_names, entropy_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # Add a horizontal line for maximum entropy (2 bits for 4 events)
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Maximum Entropy (log₂(4) = 2 bits)')
    
    # Add value labels on the bars
    for i, v in enumerate(entropy_values):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=11)
    
    # Highlight the bar with minimum entropy
    bars[dist_names.index(min_entropy_dist)].set_color('#2ecc71')
    bars[dist_names.index(min_entropy_dist)].set_alpha(0.8)
    
    # Add annotations
    ax.annotate('Lowest Entropy\n(Most Concentrated)', 
                xy=(dist_names.index(min_entropy_dist), entropy_values[dist_names.index(min_entropy_dist)]),
                xytext=(dist_names.index(min_entropy_dist) - 0.5, entropy_values[dist_names.index(min_entropy_dist)] - 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10)
    
    ax.annotate('Highest Entropy\n(Uniform Distribution)', 
                xy=(dist_names.index(max_entropy_dist), entropy_values[dist_names.index(max_entropy_dist)]),
                xytext=(dist_names.index(max_entropy_dist) + 0.5, entropy_values[dist_names.index(max_entropy_dist)] - 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10)
    
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Entropy Comparison Across Distributions')
    ax.set_ylim(0, 2.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'entropy_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 4: KL Divergence asymmetry visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart for KL divergence comparison
    labels = ['D_KL(P||Q)', 'D_KL(Q||P)']
    values = [kl_p_q, kl_q_p]
    
    bars = ax.bar(labels, values, color=['#3498db', '#e74c3c'])
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11)
    
    # Add annotations explaining asymmetry
    ax.annotate('KL Divergence is asymmetric', 
                xy=(0.5, max(values) + 0.1),
                xytext=(0.5, max(values) + 0.15),
                ha='center', fontsize=12, fontweight='bold')
    
    # Add formula annotation
    ax.text(0, -0.06, r"$D_{KL}(P||Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$", transform=ax.transAxes, fontsize=12)
    ax.text(0, -0.12, r"$D_{KL}(Q||P) = \sum_x Q(x) \log\frac{Q(x)}{P(x)}$", transform=ax.transAxes, fontsize=12)
    
    ax.set_ylabel('KL Divergence (bits)')
    ax.set_title('Asymmetry of KL Divergence')
    ax.set_ylim(0, max(values) + 0.25)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kl_divergence_asymmetry.png'), dpi=300, bbox_inches='tight')
    
    # Create visualization 5: Distribution shapes with entropy values
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create line plots for each distribution
    x = np.arange(1, 5)
    
    for i, (name, dist) in enumerate(distributions.items()):
        ax.plot(x, dist, 'o-', linewidth=2, label=f'{name}: H={entropies[name]:.2f} bits')
        
    # Add annotations
    if 'R' in distributions:
        r_dist = distributions['R']
        ax.annotate('High concentration\n= Low entropy', 
                    xy=(1, r_dist[0]),
                    xytext=(1.5, r_dist[0] + 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    fontsize=10)
    
    if 'P' in distributions:
        p_dist = distributions['P']
        ax.annotate('Uniform distribution\n= Maximum entropy', 
                    xy=(2, p_dist[1]),
                    xytext=(2.5, p_dist[1] + 0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                    fontsize=10)
    
    ax.set_xlabel('Event')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution Shapes and Their Entropy Values')
    ax.set_xticks(x)
    ax.set_ylim(0, 0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_shapes.png'), dpi=300, bbox_inches='tight')
    
    return entropies, kl_divergences

if __name__ == "__main__":
    print("Generating Information Theory Visualizations and Analysis for Question 9...")
    
    # First, recreate the original visualizations from the question file
    # Use the same code as in the question script to generate these
    
    # Now provide detailed printed explanations rather than visual annotations
    print_step_header(1, "Analyzing Channel Information Capacity")
    mutual_info, conditional_entropy, joint_entropies, marginal_entropy_y = analyze_channel_information()
    
    print_step_header(2, "Analyzing Entropy and KL Divergence")
    entropies, kl_divergences = analyze_entropy_and_kl_divergence()
    
    print_step_header(3, "Summary of Key Insights")
    print("1. Mutual Information and Channel Capacity:")
    print("   - The mutual information I(X;Y) quantifies how much uncertainty about X")
    print("     is reduced by observing Y. Higher mutual information means better")
    print("     information transmission capacity.")
    
    print("\n2. Relationship to Noise:")
    print("   - As noise in a channel increases (moving from Channel A to D),")
    print("     mutual information decreases and conditional entropy increases.")
    print("   - A completely random channel (D) has zero mutual information and")
    print("     maximum conditional entropy.")
    
    print("\n3. Entropy and Certainty:")
    print("   - More concentrated distributions (like R) have lower entropy than")
    print("     more uniform distributions (like P).")
    print("   - Uniform distributions have maximum entropy for a given number of outcomes.")
    
    print("\n4. Asymmetry of KL Divergence:")
    print("   - KL divergence is not a true distance metric because it's asymmetric.")
    print("   - This asymmetry matters in machine learning when comparing models or distributions.")
    
    print("\n5. Information Theory in Communication:")
    print("   - These concepts directly relate to Claude Shannon's original application")
    print("     of information theory to communication channels, where the key question")
    print("     was how much information could be reliably transmitted through a noisy channel.")
    
    print(f"\nAll visualizations saved in {save_dir}") 