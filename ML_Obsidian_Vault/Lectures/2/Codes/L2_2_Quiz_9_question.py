import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.stats import entropy

# Set seed for reproducibility
np.random.seed(42)

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_9")
os.makedirs(save_dir, exist_ok=True)

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# Create channel transition matrices and calculate mutual information
def channel_entropy_demo():
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
        
        # Mutual information I(X;Y) = H(Y) - H(Y|X)
        mutual_info[name] = h_y - h_y_given_x
    
    # Visualize the channels and their properties
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 4, figure=fig)
    
    # Plot channel matrices as heat maps
    for i, (name, channel) in enumerate(channels.items()):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(channel, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Channel {name} - P(Y|X)')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Y=0', 'Y=1'])
        ax.set_yticklabels(['X=0', 'X=1'])
        
        # Annotate each cell with its probability
        for di in range(2):
            for dj in range(2):
                ax.text(dj, di, f'{channel[di, dj]:.2f}', 
                        ha='center', va='center', 
                        color='black' if channel[di, dj] < 0.7 else 'white')
    
    # Plot prior distributions
    ax = fig.add_subplot(gs[1, :2])
    labels = list(priors.keys())
    x_vals = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x_vals - width/2, [priors[name][0] for name in labels], width, label='P(X=0)')
    ax.bar(x_vals + width/2, [priors[name][1] for name in labels], width, label='P(X=1)')
    
    ax.set_ylabel('Probability')
    ax.set_title('Prior Distributions P(X)')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Plot mutual information and conditional entropy
    ax = fig.add_subplot(gs[1, 2:])
    x_vals = np.arange(len(labels))
    width = 0.35
    
    mi_vals = [mutual_info[name] for name in labels]
    ce_vals = [conditional_entropy[name] for name in labels]
    
    ax.bar(x_vals - width/2, mi_vals, width, label='I(X;Y)')
    ax.bar(x_vals + width/2, ce_vals, width, label='H(Y|X)')
    
    ax.set_ylabel('Information (bits)')
    ax.set_title('Information Measures')
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Bottom row: Visualize joint distributions as 2D heatmaps
    for i, (name, channel) in enumerate(channels.items()):
        ax = fig.add_subplot(gs[2, i])
        
        prior = priors[name]
        joint = np.zeros((2, 2))
        for di in range(2):
            for dj in range(2):
                joint[di, dj] = prior[di] * channel[di, dj]
        
        im = ax.imshow(joint, cmap='Reds', vmin=0, vmax=0.5)
        ax.set_title(f'Channel {name} - P(X,Y)')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Y=0', 'Y=1'])
        ax.set_yticklabels(['X=0', 'X=1'])
        
        # Annotate each cell with its probability
        for di in range(2):
            for dj in range(2):
                ax.text(dj, di, f'{joint[di, dj]:.2f}', 
                        ha='center', va='center', 
                        color='black' if joint[di, dj] < 0.3 else 'white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'channel_entropy_demo.png'), dpi=300, bbox_inches='tight')
    
    # Display and save numerical values for reference
    print("\nChannel Properties:")
    print("-" * 50)
    print(f"{'Channel':<10} {'Mutual Info':<15} {'Cond. Entropy':<15}")
    print("-" * 50)
    for name in channels.keys():
        print(f"{name:<10} {mutual_info[name]:<15.4f} {conditional_entropy[name]:<15.4f}")
    print("-" * 50)
    
    return mutual_info, conditional_entropy

# Create entropy vs KL divergence visualization
def cross_entropy_divergence_demo():
    # Define 4 different distributions over the same 4 events
    distributions = {
        'P': np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform
        'Q': np.array([0.4, 0.3, 0.2, 0.1]),      # Skewed
        'R': np.array([0.7, 0.1, 0.1, 0.1]),      # Highly skewed
        'S': np.array([0.1, 0.2, 0.3, 0.4])       # Increasing
    }
    
    # Create synthetic data based on these distributions for visualization
    n_samples = 500
    sample_data = {}
    
    for name, dist in distributions.items():
        sample_data[name] = np.random.choice(4, size=n_samples, p=dist)
    
    # Calculate entropy, cross-entropy and KL divergence for all pairs
    metrics = {}
    for name1, dist1 in distributions.items():
        h_p = entropy(dist1, base=2)
        
        metrics[name1] = {
            'entropy': h_p,
            'cross_entropy': {},
            'kl_divergence': {}
        }
        
        for name2, dist2 in distributions.items():
            # Cross entropy H(P,Q) = -sum(P(x) * log(Q(x)))
            cross_ent = -np.sum(dist1 * np.log2(dist2 + 1e-10))
            # KL divergence
            kl_div = entropy(dist1, dist2, base=2)
            
            metrics[name1]['cross_entropy'][name2] = cross_ent
            metrics[name1]['kl_divergence'][name2] = kl_div
    
    # Visualization
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Plot the distributions
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.2
    x = np.arange(4)
    
    for i, (name, dist) in enumerate(distributions.items()):
        ax1.bar(x + i*width - 0.3, dist, width, label=name)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Event 1', 'Event 2', 'Event 3', 'Event 4'])
    ax1.set_ylabel('Probability')
    ax1.set_title('Probability Distributions')
    ax1.legend()
    
    # 2. Plot entropy of each distribution
    ax2 = fig.add_subplot(gs[0, 1])
    names = list(distributions.keys())
    entropies = [metrics[name]['entropy'] for name in names]
    
    ax2.bar(names, entropies, color='skyblue')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Entropy H(P)')
    
    for i, v in enumerate(entropies):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # 3. Plot cross-entropy matrix
    ax3 = fig.add_subplot(gs[1, 0])
    cross_entropy_matrix = np.zeros((4, 4))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            cross_entropy_matrix[i, j] = metrics[name1]['cross_entropy'][name2]
    
    im = ax3.imshow(cross_entropy_matrix, cmap='YlOrRd')
    ax3.set_xticks(np.arange(4))
    ax3.set_yticks(np.arange(4))
    ax3.set_xticklabels(names)
    ax3.set_yticklabels(names)
    ax3.set_xlabel('Q (Predicted)')
    ax3.set_ylabel('P (True)')
    ax3.set_title('Cross-Entropy H(P,Q)')
    
    # Annotate cells
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f'{cross_entropy_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='black' if cross_entropy_matrix[i, j] < 1.5 else 'white')
    
    # 4. Plot KL divergence matrix
    ax4 = fig.add_subplot(gs[1, 1])
    kl_matrix = np.zeros((4, 4))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            kl_matrix[i, j] = metrics[name1]['kl_divergence'][name2]
    
    im = ax4.imshow(kl_matrix, cmap='YlOrRd')
    ax4.set_xticks(np.arange(4))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels(names)
    ax4.set_yticklabels(names)
    ax4.set_xlabel('Q (Predicted)')
    ax4.set_ylabel('P (True)')
    ax4.set_title('KL Divergence D_KL(P||Q)')
    
    # Annotate cells
    for i in range(4):
        for j in range(4):
            text_color = 'black' if kl_matrix[i, j] < 1.0 else 'white'
            if i == j:  # Highlight diagonal which should be 0
                weight = 'bold'
            else:
                weight = 'normal'
                
            ax4.text(j, i, f'{kl_matrix[i, j]:.2f}',
                    ha='center', va='center', weight=weight,
                    color=text_color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cross_entropy_kl_demo.png'), dpi=300, bbox_inches='tight')
    
    # Print metrics for reference
    print("\nEntropy Values:")
    for name, dist in distributions.items():
        print(f"{name}: {metrics[name]['entropy']:.4f} bits")
    
    print("\nSelected Cross-Entropy Values (H(P,Q)):")
    print(f"H(P,P): {metrics['P']['cross_entropy']['P']:.4f} bits")
    print(f"H(P,Q): {metrics['P']['cross_entropy']['Q']:.4f} bits")
    print(f"H(Q,P): {metrics['Q']['cross_entropy']['P']:.4f} bits")
    
    print("\nSelected KL Divergence Values (D_KL(P||Q)):")
    print(f"D_KL(P||P): {metrics['P']['kl_divergence']['P']:.4f} bits")
    print(f"D_KL(P||Q): {metrics['P']['kl_divergence']['Q']:.4f} bits")
    print(f"D_KL(Q||P): {metrics['Q']['kl_divergence']['P']:.4f} bits")
    
    return metrics

# Main execution
if __name__ == "__main__":
    print("Generating Information Theory Visualizations for Quiz...")
    
    # Generate channel entropy visualization
    print("\nCreating Channel Entropy Visualization...")
    mutual_info, conditional_entropy = channel_entropy_demo()
    
    # Generate cross-entropy and KL divergence visualization
    print("\nCreating Cross-Entropy and KL Divergence Visualization...")
    metrics = cross_entropy_divergence_demo()
    
    print(f"\nAll visualizations saved in {save_dir}") 