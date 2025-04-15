import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from scipy.special import comb

def calculate_classification_error_probabilities():
    """Calculate all probabilities for the classification error problem"""
    # Given values
    n = 5  # number of test cases
    p = 0.2  # probability of misclassification
    
    # Calculate P(X = k) for k = 0, 1, 2
    p_x_0 = stats.binom.pmf(0, n, p)
    p_x_1 = stats.binom.pmf(1, n, p)
    p_x_2 = stats.binom.pmf(2, n, p)
    
    # Calculate P(X > 0)
    p_x_gt_0 = 1 - p_x_0
    
    # Calculate P(X = 2 | X > 0)
    p_x_2_given_x_gt_0 = p_x_2 / p_x_gt_0
    
    # Calculate expected value and variance
    expected_value = n * p
    variance = n * p * (1 - p)
    std_dev = np.sqrt(variance)
    
    return {
        'P(X=0)': p_x_0,
        'P(X=1)': p_x_1,
        'P(X=2)': p_x_2,
        'P(X>0)': p_x_gt_0,
        'P(X=2|X>0)': p_x_2_given_x_gt_0,
        'E[X]': expected_value,
        'Var(X)': variance,
        'Std(X)': std_dev
    }

def plot_pmf(save_path=None):
    """Create a visualization of the binomial PMF"""
    # Parameters
    n = 5
    p = 0.2
    
    # Calculate PMF for all values of k from 0 to 5
    k_values = np.arange(0, n+1)
    pmf_values = stats.binom.pmf(k_values, n, p)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.bar(k_values, pmf_values, color='skyblue', edgecolor='black')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Highlight bars for k = 0, 1, 2
    bars[0].set_color('lightgreen')  # P(X=0)
    bars[1].set_color('lightblue')   # P(X=1)
    bars[2].set_color('lightcoral')  # P(X=2)
    
    # Add labels and title
    ax.set_xlabel('Number of Classification Errors (k)')
    ax.set_ylabel('Probability P(X = k)')
    ax.set_title('Binomial Probability Mass Function for Classification Errors\n(n=5, p=0.2)')
    ax.set_xticks(k_values)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add theoretical expected value reference line
    expected_value = n * p
    ax.axvline(x=expected_value, color='red', linestyle='--', alpha=0.7)
    ax.text(expected_value + 0.1, max(pmf_values) * 0.9, f'E[X] = {expected_value}', 
            color='red', fontsize=10)
    
    # Add annotations for key probabilities
    probs = calculate_classification_error_probabilities()
    plt.figtext(0.15, 0.02, f"P(X=0) = {probs['P(X=0)']:.4f}", ha="center", fontsize=10,
               bbox={"facecolor":"lightgreen", "alpha":0.5, "pad":5})
    plt.figtext(0.5, 0.02, f"P(X=1) = {probs['P(X=1)']:.4f}", ha="center", fontsize=10,
               bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
    plt.figtext(0.85, 0.02, f"P(X=2) = {probs['P(X=2)']:.4f}", ha="center", fontsize=10,
               bbox={"facecolor":"lightcoral", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_conditional_probability(save_path=None):
    """Visualize the conditional probability P(X=2|X>0)"""
    # Parameters
    n = 5
    p = 0.2
    
    # Calculate relevant probabilities
    probs = calculate_classification_error_probabilities()
    p_x_0 = probs['P(X=0)']
    p_x_gt_0 = probs['P(X>0)']
    p_x_2 = probs['P(X=2)']
    p_x_2_given_x_gt_0 = probs['P(X=2|X>0)']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    bar_width = 0.3
    bars1 = ax.bar([0], [p_x_0], bar_width, label='P(X=0)', color='lightgreen')
    bars2 = ax.bar([0+bar_width], [p_x_gt_0], bar_width, label='P(X>0)', color='lightpink')
    bars3 = ax.bar([1], [p_x_2], bar_width, label='P(X=2)', color='lightcoral')
    bars4 = ax.bar([1+bar_width], [p_x_2_given_x_gt_0], bar_width, label='P(X=2|X>0)', color='coral')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Probability Type')
    ax.set_ylabel('Probability Value')
    ax.set_title('Conditional Probability: P(X=2|X>0)')
    ax.set_xticks([0 + bar_width/2, 1 + bar_width/2])
    ax.set_xticklabels(['Total Probability Space', 'Conditional Probability'])
    
    # Add legend
    ax.legend()
    
    # Add formula and computation
    formula = r"$P(X=2|X>0) = \frac{P(X=2)}{P(X>0)} = \frac{P(X=2)}{1-P(X=0)}$"
    computation = f"= {p_x_2:.4f} / {p_x_gt_0:.4f} = {p_x_2_given_x_gt_0:.4f}"
    
    plt.figtext(0.5, 0.02, formula + "\n" + computation, ha="center", 
                fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_expected_variance(save_path=None):
    """Visualize the expected value and variance of the binomial distribution"""
    # Parameters
    n = 5
    p_values = np.linspace(0, 1, 100)  # Range of p values from 0 to 1
    
    # Calculate expected value and variance for each p value
    expected_values = n * p_values
    variances = n * p_values * (1 - p_values)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot expected value
    ax1.plot(p_values, expected_values, 'b-', label='E[X] = np')
    ax1.set_xlabel('Probability of Misclassification (p)')
    ax1.set_ylabel('Expected Value E[X]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    # Plot variance
    ax2.plot(p_values, variances, 'r-', label='Var(X) = np(1-p)')
    ax2.set_ylabel('Variance Var(X)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add vertical line at p = 0.2
    p_current = 0.2
    expected_current = n * p_current
    variance_current = n * p_current * (1 - p_current)
    
    ax1.axvline(x=p_current, color='green', linestyle='--', alpha=0.7)
    ax1.plot(p_current, expected_current, 'bo', markersize=8)
    ax2.plot(p_current, variance_current, 'ro', markersize=8)
    
    # Add annotations
    ax1.annotate(f'E[X] = {expected_current}',
                xy=(p_current, expected_current),
                xytext=(p_current + 0.05, expected_current + 0.2),
                arrowprops=dict(facecolor='blue', shrink=0.05),
                fontsize=10)
    
    ax2.annotate(f'Var(X) = {variance_current:.2f}',
                xy=(p_current, variance_current),
                xytext=(p_current + 0.05, variance_current + 0.2),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=10)
    
    # Add title
    plt.title('Expected Value and Variance of Binomial Distribution (n=5)')
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_combinatorial_illustration(save_path=None):
    """Visualize the combinatorial nature of the binomial distribution"""
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define parameters
    n = 5  # number of test cases
    p = 0.2  # probability of misclassification
    
    # Calculate binomial coefficient for k=2
    k = 2
    bc = comb(n, k, exact=True)
    
    # Define positions for test cases
    x_positions = np.arange(1, n+1)
    
    # Add title and labels
    ax.set_title(f"All {bc} Ways to Choose {k} Errors out of {n} Test Cases", fontsize=14)
    ax.set_xlabel("Test Case", fontsize=12)
    
    # Add annotation for the binomial coefficient formula
    formula = r"$\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{5!}{2!3!} = 10$"
    ax.text(n/2 + 0.5, 10.8, formula, ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Add annotation for the probability calculation
    prob_formula = r"$P(X=2) = \binom{5}{2} \times (0.2)^2 \times (0.8)^3 = 10 \times 0.04 \times 0.512 = 0.2048$"
    ax.text(n/2 + 0.5, 10.2, prob_formula, ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Generate all combinations of 2 errors out of 5 cases
    import itertools
    combinations = list(itertools.combinations(range(1, n+1), k))
    
    # Test labels at the top
    y_top = 9.5
    for i, x in enumerate(x_positions):
        ax.text(x, y_top, f"Test {i+1}", ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='lightgray', alpha=0.3, boxstyle='round,pad=0.3'))
    
    # Plot all combinations with better spacing
    y_spacing = 0.8
    for i, combo in enumerate(combinations):
        y = y_top - (i+1) * y_spacing
        
        # Plot dots for each test case
        for x in x_positions:
            color = 'red' if x in combo else 'green'
            marker = 'X' if x in combo else 'o'
            ax.scatter(x, y, s=300, color=color, marker=marker, alpha=0.7, edgecolor='black')
        
        # Add label for each combination
        ax.text(0.3, y, f"Combo {i+1}: {combo}", ha='right', va='center', fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, 11.5)
    ax.set_xlim(0, n+1)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Add legend
    ax.scatter([], [], s=150, color='red', marker='X', label='Error', edgecolor='black')
    ax.scatter([], [], s=150, color='green', marker='o', label='Correct', edgecolor='black')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 6"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_6")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 6 of the L2.1 Probability quiz...")
    
    # Calculate probabilities
    probs = calculate_classification_error_probabilities()
    print("\nCalculated Probabilities:")
    print(f"P(X=0): {probs['P(X=0)']:.4f}")
    print(f"P(X=1): {probs['P(X=1)']:.4f}")
    print(f"P(X=2): {probs['P(X=2)']:.4f}")
    print(f"P(X>0): {probs['P(X>0)']:.4f}")
    print(f"P(X=2|X>0): {probs['P(X=2|X>0)']:.4f}")
    print(f"E[X]: {probs['E[X]']}")
    print(f"Var(X): {probs['Var(X)']:.4f}")
    print(f"Standard Deviation: {probs['Std(X)']:.4f}")
    
    # Generate visualizations
    plot_pmf(save_path=os.path.join(save_dir, "pmf.png"))
    print("1. Probability mass function visualization created")
    
    plot_conditional_probability(save_path=os.path.join(save_dir, "conditional_probability.png"))
    print("2. Conditional probability visualization created")
    
    plot_expected_variance(save_path=os.path.join(save_dir, "expected_variance.png"))
    print("3. Expected value and variance visualization created")
    
    plot_combinatorial_illustration(save_path=os.path.join(save_dir, "combinatorial.png"))
    print("4. Combinatorial illustration created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 