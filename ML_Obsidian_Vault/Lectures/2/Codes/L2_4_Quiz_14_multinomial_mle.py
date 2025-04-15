import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, norm
import os

def plot_multinomial_probabilities(counts, labels, save_path=None):
    """Plot the observed frequencies and MLE probabilities"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate total count and MLE probabilities
    n = np.sum(counts)
    probs = counts / n
    
    # Create the bar plot with both counts and probabilities
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, counts, width, color='steelblue', label='Observed Counts')
    
    # Add a second axis for probabilities
    ax2 = ax.twinx()
    ax2.bar(x + width/2, probs, width, color='lightcoral', label='MLE Probabilities')
    
    # Set labels and title
    ax.set_xlabel('Product Features')
    ax.set_ylabel('Count')
    ax2.set_ylabel('Probability')
    ax.set_title('Feature Preferences: Counts and MLE Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add a legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add value labels
    for i, count in enumerate(counts):
        ax.annotate(f'{count}', 
                   xy=(i - width/2, count),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center')
    
    for i, prob in enumerate(probs):
        ax2.annotate(f'{prob:.3f}', 
                   xy=(i + width/2, prob),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return probs

def calculate_confidence_intervals(counts, alpha=0.05):
    """Calculate confidence intervals for multinomial probabilities using normal approximation"""
    n = np.sum(counts)
    probs = counts / n
    
    # Calculate standard errors using normal approximation
    std_errs = np.sqrt(probs * (1 - probs) / n)
    
    # Calculate confidence intervals
    z = norm.ppf(1 - alpha/2)
    lower_bounds = probs - z * std_errs
    upper_bounds = probs + z * std_errs
    
    # Ensure bounds are within [0, 1]
    lower_bounds = np.maximum(0, lower_bounds)
    upper_bounds = np.minimum(1, upper_bounds)
    
    return lower_bounds, upper_bounds

def plot_confidence_intervals(counts, labels, alpha=0.05, save_path=None):
    """Plot the MLE probabilities with confidence intervals"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate MLE probabilities and confidence intervals
    n = np.sum(counts)
    probs = counts / n
    lower_bounds, upper_bounds = calculate_confidence_intervals(counts, alpha)
    
    # Create the bar plot
    x = np.arange(len(labels))
    width = 0.6
    
    ax.bar(x, probs, width, color='steelblue', label='MLE Probabilities')
    
    # Add error bars for confidence intervals
    yerr = np.vstack((probs - lower_bounds, upper_bounds - probs))
    ax.errorbar(x, probs, yerr=yerr, fmt='none', color='black', capsize=5)
    
    # Set labels and title
    ax.set_xlabel('Product Features')
    ax.set_ylabel('Probability')
    ax.set_title(f'Feature Preference Probabilities with {(1-alpha)*100:.0f}% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add value labels with CI
    for i, (prob, lb, ub) in enumerate(zip(probs, lower_bounds, upper_bounds)):
        ax.annotate(f'{prob:.3f}\n({lb:.3f}, {ub:.3f})', 
                   xy=(i, prob),
                   xytext=(0, 5),
                   textcoords='offset points',
                   ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return probs, lower_bounds, upper_bounds

def plot_significance_comparison(probs, lower_bounds, upper_bounds, labels, save_path=None):
    """Plot to visually check if two probabilities are significantly different"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create the line plot
    x = np.arange(len(labels))
    
    ax.errorbar(x, probs, yerr=[probs - lower_bounds, upper_bounds - probs], 
                fmt='o', color='steelblue', capsize=5, markersize=8)
    
    # Connect points with lines to show the pattern
    ax.plot(x, probs, 'b--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Product Features')
    ax.set_ylabel('Probability')
    ax.set_title('Comparison of Feature Preference Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the comparison between Feature A and Feature B
    feature_a_idx = 0  # Index of Feature A
    feature_b_idx = 1  # Index of Feature B
    
    # Shade the confidence intervals for A and B
    ax.fill_between([feature_a_idx-0.2, feature_a_idx+0.2], 
                   [lower_bounds[feature_a_idx], lower_bounds[feature_a_idx]],
                   [upper_bounds[feature_a_idx], upper_bounds[feature_a_idx]],
                   color='blue', alpha=0.3)
    
    ax.fill_between([feature_b_idx-0.2, feature_b_idx+0.2], 
                   [lower_bounds[feature_b_idx], lower_bounds[feature_b_idx]],
                   [upper_bounds[feature_b_idx], upper_bounds[feature_b_idx]],
                   color='red', alpha=0.3)
    
    # Add annotation to explain the significance
    if (lower_bounds[feature_a_idx] > upper_bounds[feature_b_idx]):
        significance_text = "Feature A is significantly more popular than Feature B"
    elif (lower_bounds[feature_b_idx] > upper_bounds[feature_a_idx]):
        significance_text = "Feature B is significantly more popular than Feature A"
    else:
        significance_text = "The difference between Feature A and B is not statistically significant"
    
    ax.annotate(significance_text, 
               xy=((feature_a_idx + feature_b_idx)/2, min(lower_bounds[feature_a_idx], lower_bounds[feature_b_idx])),
               xytext=(0, -40),
               textcoords='offset points',
               ha='center',
               va='top',
               bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 14 of the L2.4 quiz"""
    # Feature preference data from the question
    feature_counts = np.array([145, 95, 120, 85, 55])
    feature_labels = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    total_customers = np.sum(feature_counts)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_14")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 14 of the L2.4 MLE quiz...")
    print(f"Feature preference data: {feature_counts}")
    print(f"Total customers surveyed: {total_customers}")
    
    # 1. Plot observed counts and MLE probabilities
    probs = plot_multinomial_probabilities(feature_counts, feature_labels, 
                                         save_path=os.path.join(save_dir, "feature_preferences.png"))
    print("1. Feature preferences visualization created")
    print(f"MLE probability estimates: {probs}")
    
    # 2. Calculate and plot confidence intervals
    probs, lower_bounds, upper_bounds = plot_confidence_intervals(feature_counts, feature_labels, 
                                                                save_path=os.path.join(save_dir, "confidence_intervals.png"))
    print("2. Confidence intervals visualization created")
    print("95% Confidence Intervals:")
    for i, label in enumerate(feature_labels):
        print(f"{label}: ({lower_bounds[i]:.3f}, {upper_bounds[i]:.3f})")
    
    # 3. Plot significance comparison
    plot_significance_comparison(probs, lower_bounds, upper_bounds, feature_labels, 
                               save_path=os.path.join(save_dir, "significance_comparison.png"))
    print("3. Significance comparison visualization created")
    
    # Check if Feature A is significantly more popular than Feature B
    feature_a_idx = 0
    feature_b_idx = 1
    if lower_bounds[feature_a_idx] > upper_bounds[feature_b_idx]:
        print("Feature A is significantly more popular than Feature B (confidence intervals do not overlap)")
    else:
        print("Cannot conclude that Feature A is significantly more popular than Feature B (confidence intervals overlap)")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 