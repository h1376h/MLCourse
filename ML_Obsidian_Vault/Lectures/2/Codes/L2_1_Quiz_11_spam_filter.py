import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

def calculate_spam_filter_probabilities(p_spam=0.2, sensitivity=0.95, specificity=0.98):
    """
    Calculate various probabilities for the spam filter scenario.
    
    Args:
        p_spam: Prior probability that an email is spam
        sensitivity: P(classified as spam | actually spam) - True Positive Rate
        specificity: P(classified as not spam | actually not spam) - True Negative Rate
    
    Returns:
        Dictionary with calculated probabilities
    """
    # Define variables more clearly
    # P(S) - prior probability of spam
    p_s = p_spam
    # P(not S) - prior probability of non-spam
    p_not_s = 1 - p_s
    
    # P(C_S | S) - probability of classifying as spam given it is spam (sensitivity/TPR)
    p_cs_given_s = sensitivity
    # P(C_NS | not S) - probability of classifying as not spam given it is not spam (specificity/TNR)
    p_cns_given_not_s = specificity
    
    # P(C_NS | S) - probability of classifying as not spam given it is spam (FNR)
    p_cns_given_s = 1 - p_cs_given_s
    # P(C_S | not S) - probability of classifying as spam given it is not spam (FPR)
    p_cs_given_not_s = 1 - p_cns_given_not_s
    
    # Calculate joint probabilities
    # P(S, C_S) - probability email is spam and classified as spam
    p_s_and_cs = p_s * p_cs_given_s
    # P(not S, C_S) - probability email is not spam but classified as spam
    p_not_s_and_cs = p_not_s * p_cs_given_not_s
    # P(S, C_NS) - probability email is spam but classified as not spam
    p_s_and_cns = p_s * p_cns_given_s
    # P(not S, C_NS) - probability email is not spam and classified as not spam
    p_not_s_and_cns = p_not_s * p_cns_given_not_s
    
    # Calculate marginal probabilities
    # P(C_S) - probability of classifying as spam
    p_cs = p_s_and_cs + p_not_s_and_cs
    # P(C_NS) - probability of classifying as not spam
    p_cns = p_s_and_cns + p_not_s_and_cns
    
    # Task 1: P(S | C_S) - probability email is spam given classified as spam (PPV)
    p_s_given_cs = p_s_and_cs / p_cs
    
    # Task 2: P(S | C_NS) - probability email is spam given classified as not spam (FNR)
    p_s_given_cns = p_s_and_cns / p_cns
    
    # Task 3: Overall accuracy
    accuracy = p_s_and_cs + p_not_s_and_cns
    
    # Return all calculated probabilities
    return {
        'p_s': p_s,
        'p_not_s': p_not_s,
        'p_cs_given_s': p_cs_given_s,
        'p_cns_given_not_s': p_cns_given_not_s,
        'p_cns_given_s': p_cns_given_s,
        'p_cs_given_not_s': p_cs_given_not_s,
        'p_s_and_cs': p_s_and_cs,
        'p_not_s_and_cs': p_not_s_and_cs,
        'p_s_and_cns': p_s_and_cns,
        'p_not_s_and_cns': p_not_s_and_cns,
        'p_cs': p_cs,
        'p_cns': p_cns,
        'p_s_given_cs': p_s_given_cs,
        'p_s_given_cns': p_s_given_cns,
        'accuracy': accuracy
    }

def create_confusion_matrix_visualization(probs, save_path=None):
    """
    Create a visual representation of the confusion matrix using matplotlib.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    # Create a 2x2 confusion matrix
    confusion_matrix = np.array([
        [probs['p_s_and_cs'], probs['p_s_and_cns']],  # Spam emails
        [probs['p_not_s_and_cs'], probs['p_not_s_and_cns']]  # Non-spam emails
    ])
    
    # Labels
    x_labels = ['Classified as\nSpam', 'Classified as\nNon-Spam']
    y_labels = ['Actually\nSpam', 'Actually\nNon-Spam']
    
    # Create the figure with a clean style
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a custom colormap - using a blue gradient
    cmap = plt.cm.Blues
    
    # Plot the heatmap
    ax = plt.gca()
    im = ax.imshow(confusion_matrix, cmap=cmap)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion_matrix[i, j]:.4f}', 
                   ha="center", va="center", 
                   color="white" if confusion_matrix[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticklabels(y_labels, fontsize=12)
    
    # Add title
    plt.title('Confusion Matrix for Spam Filter', fontsize=16, pad=10)
    
    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Probability', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix visualization saved to {save_path}")
    
    plt.close()

def create_bayes_theorem_visualization(probs, save_path=None):
    """
    Create a visual representation of Bayes' theorem application.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a 10x10 grid representing all emails
    grid_size = 10
    total_cells = grid_size * grid_size
    
    # Calculate number of cells for each category
    num_spam = int(probs['p_s'] * total_cells)
    num_not_spam = total_cells - num_spam
    
    # Calculate classification categories
    num_spam_classified_spam = int(probs['p_s_and_cs'] * total_cells)
    num_spam_classified_not_spam = num_spam - num_spam_classified_spam
    num_not_spam_classified_spam = int(probs['p_not_s_and_cs'] * total_cells)
    num_not_spam_classified_not_spam = num_not_spam - num_not_spam_classified_spam
    
    # Create grid
    grid = np.zeros((grid_size, grid_size))
    
    # Fill grid with values:
    # 0: Non-spam classified as non-spam (True Negative)
    # 1: Spam classified as spam (True Positive)
    # 2: Non-spam classified as spam (False Positive)
    # 3: Spam classified as non-spam (False Negative)
    
    cell_index = 0
    
    # Fill True Positives (spam classified as spam)
    for i in range(num_spam_classified_spam):
        row, col = divmod(cell_index, grid_size)
        grid[row, col] = 1
        cell_index += 1
    
    # Fill False Negatives (spam classified as non-spam)
    for i in range(num_spam_classified_not_spam):
        row, col = divmod(cell_index, grid_size)
        grid[row, col] = 3
        cell_index += 1
    
    # Fill False Positives (non-spam classified as spam)
    for i in range(num_not_spam_classified_spam):
        row, col = divmod(cell_index, grid_size)
        grid[row, col] = 2
        cell_index += 1
    
    # Fill True Negatives (non-spam classified as non-spam)
    for i in range(num_not_spam_classified_not_spam):
        row, col = divmod(cell_index, grid_size)
        grid[row, col] = 0
        cell_index += 1
    
    # Create colormap - using brighter colors
    cmap = plt.cm.colors.ListedColormap(['#8dd3c7', '#fb8072', '#fdb462', '#ffffb3'])
    
    # Plot the grid
    ax = plt.gca()
    plt.imshow(grid, cmap=cmap)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Remove regular ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#fb8072', label='True Positive (TP)'),
        Patch(facecolor='#ffffb3', label='False Negative (FN)'),
        Patch(facecolor='#fdb462', label='False Positive (FP)'),
        Patch(facecolor='#8dd3c7', label='True Negative (TN)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
              ncol=2, fontsize=12, frameon=True, title="Email Classification")
    
    # Add title only
    plt.title("Email Classification Grid", fontsize=16, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bayes' theorem visualization saved to {save_path}")
    
    plt.close()

def plot_prior_effect(save_path=None):
    """
    Plot how changing the prior probability affects the posterior.
    
    Args:
        save_path: Path to save the visualization
    """
    # Create a range of prior probabilities
    prior_probs = np.linspace(0.01, 0.99, 100)
    
    # Calculate posterior probabilities for each prior
    sensitivity = 0.95  # P(C_S|S)
    specificity = 0.98  # P(C_NS|~S)
    
    posteriors = []
    for prior in prior_probs:
        probs = calculate_spam_filter_probabilities(p_spam=prior, 
                                                   sensitivity=sensitivity, 
                                                   specificity=specificity)
        posteriors.append(probs['p_s_given_cs'])
    
    # Create the plot with a modern style
    plt.figure(figsize=(9, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot the curve with a nicer color and style
    plt.plot(prior_probs, posteriors, lw=3, color='#1f77b4')
    
    # Add markers for key values (20% and 50%)
    p_20 = calculate_spam_filter_probabilities(p_spam=0.2, 
                                              sensitivity=sensitivity, 
                                              specificity=specificity)['p_s_given_cs']
    p_50 = calculate_spam_filter_probabilities(p_spam=0.5, 
                                              sensitivity=sensitivity, 
                                              specificity=specificity)['p_s_given_cs']
    
    plt.scatter([0.2, 0.5], [p_20, p_50], color='#d62728', s=100, zorder=3)
    
    # Add simple annotations
    plt.annotate(f'(0.2, {p_20:.4f})', xy=(0.2, p_20), 
                xytext=(0.22, p_20-0.05), fontsize=12,
                arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.05))
    plt.annotate(f'(0.5, {p_50:.4f})', xy=(0.5, p_50), 
                xytext=(0.52, p_50-0.05), fontsize=12,
                arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.05))
    
    # Add reference line for prior = posterior (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Prior = Posterior')
    
    # Add axis labels and title
    plt.xlabel('Prior Probability P(S)', fontsize=14)
    plt.ylabel('Posterior Probability P(S|C_S)', fontsize=14)
    plt.title('Effect of Prior on Posterior Probability', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prior effect visualization saved to {save_path}")
    
    plt.close()

def plot_roc_curve(save_path=None):
    """
    Plot the ROC curve for the spam filter, along with different operating points.
    
    Args:
        save_path: Path to save the visualization
    """
    # For a binary classifier with fixed sensitivity/specificity, we can't generate
    # a full ROC curve. But we can show where our classifier sits on the ROC space.
    
    # Our classifier performance
    tpr = 0.95  # True Positive Rate (sensitivity)
    fpr = 0.02  # False Positive Rate (1 - specificity)
    
    # Create the plot with a modern style
    plt.figure(figsize=(8, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot the ROC space
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Random Classifier')
    
    # Plot our classifier's point
    plt.scatter(fpr, tpr, color='#d62728', s=150, zorder=3, label='Spam Filter')
    
    # Add annotation
    plt.annotate(f'({fpr:.2f}, {tpr:.2f})', xy=(fpr, tpr), 
                xytext=(fpr+0.1, tpr-0.1), fontsize=12,
                arrowprops=dict(facecolor='black', width=1, headwidth=6, shrink=0.05))
    
    # Plot some theoretical points for comparison without text
    other_points = [
        (0.01, 0.85),
        (0.05, 0.98),
        (0.10, 0.99),
    ]
    
    for x, y in other_points:
        plt.scatter(x, y, alpha=0.7, s=80, color='#1f77b4')
    
    # Perfect classifier
    plt.scatter(0, 1, marker='*', s=200, color='gold', label='Perfect Classifier')
    
    # Add grid, labels, and legend
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Space: Classifier Performance', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve visualization saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 11 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_11")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 11 of the L2.1 Probability quiz: Spam Filter Probability...")
    
    # Calculate probabilities for the original scenario
    probs = calculate_spam_filter_probabilities(p_spam=0.2, sensitivity=0.95, specificity=0.98)
    
    # Calculate probabilities for the modified scenario (50% prior)
    probs_modified = calculate_spam_filter_probabilities(p_spam=0.5, sensitivity=0.95, specificity=0.98)
    
    # Print results
    print("\nProbabilities for Spam Filter (Original Scenario):")
    print(f"Prior probability of spam P(S): {probs['p_s']:.4f}")
    print(f"True Positive Rate (sensitivity) P(C_S|S): {probs['p_cs_given_s']:.4f}")
    print(f"True Negative Rate (specificity) P(C_NS|~S): {probs['p_cns_given_not_s']:.4f}")
    print(f"Task 1: P(S|C_S) - probability email is spam given classified as spam: {probs['p_s_given_cs']:.4f}")
    print(f"Task 2: P(S|C_NS) - probability email is spam given classified as not spam: {probs['p_s_given_cns']:.4f}")
    print(f"Task 3: Overall accuracy: {probs['accuracy']:.4f}")
    print(f"Task 4: P(S|C_S) with prior=0.5: {probs_modified['p_s_given_cs']:.4f}")
    
    # Generate visualizations
    create_confusion_matrix_visualization(probs, save_path=os.path.join(save_dir, "confusion_matrix.png"))
    print("1. Confusion matrix visualization created")
    
    create_bayes_theorem_visualization(probs, save_path=os.path.join(save_dir, "bayes_theorem.png"))
    print("2. Bayes' theorem visualization created")
    
    plot_prior_effect(save_path=os.path.join(save_dir, "prior_effect.png"))
    print("3. Prior effect visualization created")
    
    plot_roc_curve(save_path=os.path.join(save_dir, "roc_curve.png"))
    print("4. ROC curve visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 