import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

def calculate_geometric_probabilities(p_success=0.3):
    """
    Calculate various probabilities for the geometric distribution scenario.
    
    Args:
        p_success: Probability of success on each attempt
    
    Returns:
        Dictionary with calculated probabilities
    """
    # Task 1: Probability of success on exactly the 4th attempt
    # In geometric distribution, this is P(X = 4) = (1-p)^(4-1) * p
    p_success_on_4th = (1 - p_success) ** 3 * p_success
    
    # Task 2: Probability of success within the first 5 attempts
    # This is P(X ≤ 5) = 1 - P(X > 5) = 1 - (1-p)^5
    p_success_within_5 = 1 - (1 - p_success) ** 5
    
    # Task 3: Expected number of attempts needed
    # For geometric distribution, E[X] = 1/p
    expected_attempts = 1 / p_success
    
    # Task 4: Probability of success on next attempt given 3 failures
    # Due to memoryless property, this is just p
    p_success_after_3_failures = p_success
    
    # Additional useful calculations for visualizations
    # PMF values for 1 to 20 attempts
    k_values = np.arange(1, 21)
    pmf_values = stats.geom.pmf(k_values, p_success)
    
    # CDF values for 1 to 20 attempts
    cdf_values = stats.geom.cdf(k_values, p_success)
    
    return {
        'p_success': p_success,
        'p_failure': 1 - p_success,
        'p_success_on_4th': p_success_on_4th,
        'p_success_within_5': p_success_within_5,
        'expected_attempts': expected_attempts,
        'p_success_after_3_failures': p_success_after_3_failures,
        'k_values': k_values,
        'pmf_values': pmf_values,
        'cdf_values': cdf_values
    }

def create_pmf_visualization(probs, save_path=None):
    """
    Create a visualization of the geometric distribution PMF.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values']
    pmf_values = probs['pmf_values']
    p_success = probs['p_success']
    
    # Create bar plot
    bars = plt.bar(k_values, pmf_values, color='#1f77b4', alpha=0.7)
    
    # Highlight the 4th attempt
    bars[3].set_color('#d62728')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Attempts Until Success', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Geometric Distribution PMF with p = {p_success}', fontsize=14)
    
    # Annotate the probability of success on exactly the 4th attempt
    plt.annotate(f'P(X=4) = {probs["p_success_on_4th"]:.4f}', 
                xy=(4, probs['pmf_values'][3]), 
                xytext=(6, probs['pmf_values'][3] + 0.02),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PMF visualization saved to {save_path}")
    
    plt.close()

def create_cdf_visualization(probs, save_path=None):
    """
    Create a visualization of the geometric distribution CDF.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values']
    cdf_values = probs['cdf_values']
    p_success = probs['p_success']
    
    # Plot CDF as step function
    plt.step(k_values, cdf_values, where='post', color='#1f77b4', linewidth=2)
    plt.scatter(k_values, cdf_values, color='#1f77b4', s=50)
    
    # Highlight the CDF value at x=5 (success within 5 attempts)
    plt.plot(5, cdf_values[4], 'ro', markersize=10)
    
    # Add horizontal and vertical lines
    plt.axhline(y=cdf_values[4], color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='red', linestyle='--', alpha=0.5)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Attempts Until Success', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title(f'Geometric Distribution CDF with p = {p_success}', fontsize=14)
    
    # Annotate the probability of success within 5 attempts
    plt.annotate(f'P(X ≤ 5) = {probs["p_success_within_5"]:.4f}', 
                xy=(5, cdf_values[4]), 
                xytext=(8, cdf_values[4] - 0.08),
                fontsize=12,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CDF visualization saved to {save_path}")
    
    plt.close()

def create_expected_value_visualization(probs, save_path=None):
    """
    Create a visualization showing the expected number of attempts.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract data
    k_values = probs['k_values']
    pmf_values = probs['pmf_values']
    p_success = probs['p_success']
    expected_attempts = probs['expected_attempts']
    
    # Create bar plot
    bars = plt.bar(k_values, pmf_values, color='#1f77b4', alpha=0.7)
    
    # Add vertical line at expected value
    plt.axvline(x=expected_attempts, color='red', linestyle='--', 
               label=f'Expected Value: {expected_attempts:.2f}')
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Attempts Until Success', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Expected Number of Attempts (Geometric Distribution with p = {p_success})', fontsize=14)
    
    # Annotate the expected value
    plt.annotate(f'E[X] = 1/p = {expected_attempts:.2f}', 
                xy=(expected_attempts, 0.05), 
                xytext=(expected_attempts + 2, 0.08),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5))
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Expected value visualization saved to {save_path}")
    
    plt.close()

def create_memoryless_visualization(probs, save_path=None):
    """
    Create a visualization demonstrating the memoryless property.
    
    Args:
        probs: Dictionary with calculated probabilities
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initial state distribution on the left
    k_values = np.arange(1, 11)
    initial_pmf = stats.geom.pmf(k_values, probs['p_success'])
    
    ax1.bar(k_values, initial_pmf, color='#1f77b4', alpha=0.7)
    ax1.set_title('Original Distribution P(X = k)', fontsize=14)
    ax1.set_xlabel('Number of Attempts Until Success', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Conditional distribution after 3 failures on the right
    conditional_k = np.arange(4, 14)  # Starting from 4th attempt
    conditional_pmf = stats.geom.pmf(conditional_k - 3, probs['p_success'])  # Shifted by 3
    
    ax2.bar(conditional_k, conditional_pmf, color='#d62728', alpha=0.7)
    ax2.set_title('Conditional Distribution P(X = k | X > 3)', fontsize=14)
    ax2.set_xlabel('Number of Attempts Until Success', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Annotate the probabilities for the 4th attempt in both distributions
    ax1.annotate(f'P(X=4) = {initial_pmf[3]:.4f}', 
                xy=(4, initial_pmf[3]), 
                xytext=(6, initial_pmf[3] + 0.02),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    ax2.annotate(f'P(X=4 | X>3) = {conditional_pmf[0]:.4f}', 
                xy=(4, conditional_pmf[0]), 
                xytext=(6, conditional_pmf[0] + 0.02),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add a title for the entire figure
    plt.suptitle('Memoryless Property of Geometric Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Memoryless property visualization saved to {save_path}")
    
    plt.close()

def create_success_probability_effect(save_path=None):
    """
    Create a visualization showing how different success probabilities affect the distribution.
    
    Args:
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Range of attempts to show
    k_values = np.arange(1, 21)
    
    # Different success probabilities to compare
    probabilities = [0.1, 0.3, 0.5, 0.7]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot PMF for each probability
    for i, p in enumerate(probabilities):
        pmf_values = stats.geom.pmf(k_values, p)
        expected_value = 1/p
        
        plt.plot(k_values, pmf_values, color=colors[i], linewidth=2, 
                marker='o', markersize=4, alpha=0.7,
                label=f'p = {p} (E[X] = {expected_value:.1f})')
        
        # Add vertical line at expected value
        plt.axvline(x=expected_value, color=colors[i], linestyle='--', alpha=0.3)
    
    # Add grid and labels
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Attempts Until Success', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Geometric Distributions with Different Success Probabilities', fontsize=14)
    plt.legend(title='Success Probability', loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Success probability effect visualization saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 17 of the L2.1 quiz"""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_17")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 17 of the L2.1 Probability quiz: Geometric Distribution in RL...")
    
    # Calculate probabilities
    probs = calculate_geometric_probabilities(p_success=0.3)
    
    # Print results
    print("\nProbabilities for Geometric Distribution:")
    print(f"Success probability (p): {probs['p_success']}")
    print(f"Task 1: P(X=4) - Probability of success on exactly the 4th attempt: {probs['p_success_on_4th']:.6f}")
    print(f"Task 2: P(X≤5) - Probability of success within first 5 attempts: {probs['p_success_within_5']:.6f}")
    print(f"Task 3: E[X] - Expected number of attempts needed: {probs['expected_attempts']:.6f}")
    print(f"Task 4: P(success on next attempt | 3 failures) = {probs['p_success_after_3_failures']:.6f}")
    
    # Generate visualizations
    create_pmf_visualization(probs, save_path=os.path.join(save_dir, "geometric_pmf.png"))
    print("1. PMF visualization created")
    
    create_cdf_visualization(probs, save_path=os.path.join(save_dir, "geometric_cdf.png"))
    print("2. CDF visualization created")
    
    create_expected_value_visualization(probs, save_path=os.path.join(save_dir, "expected_value.png"))
    print("3. Expected value visualization created")
    
    create_memoryless_visualization(probs, save_path=os.path.join(save_dir, "memoryless_property.png"))
    print("4. Memoryless property visualization created")
    
    create_success_probability_effect(save_path=os.path.join(save_dir, "success_probability_effect.png"))
    print("5. Success probability effect visualization created")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 