import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.ticker import MaxNLocator

def create_images_directory():
    """Create directory for storing the generated images."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "L2_1_Quiz_21")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir

def question_1_analysis(save_dir):
    """
    Analyze Question 1: Which random variable is most likely to be modeled using a Poisson distribution?
    """
    print("\n--- Question 21.1: Poisson Distribution Applications ---")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    axes = axes.flatten()
    
    # A) Number of customers arriving at a store in one hour
    arrivals_per_hour = 20  # Average arrivals per hour
    x_poisson = np.arange(0, 40)
    y_poisson = stats.poisson.pmf(x_poisson, arrivals_per_hour)
    
    axes[0].bar(x_poisson, y_poisson, alpha=0.7, color='skyblue')
    axes[0].axvline(arrivals_per_hour, color='red', linestyle='--')
    axes[0].set_title('A) Number of Customers Arriving at a Store in One Hour')
    axes[0].set_xlabel('Number of Arrivals')
    axes[0].set_ylabel('Probability')
    
    # B) Height of adult males
    height_mean = 175  # cm
    height_std = 7     # cm
    x_normal = np.linspace(155, 195, 1000)
    y_normal = stats.norm.pdf(x_normal, height_mean, height_std)
    
    axes[1].plot(x_normal, y_normal, 'g-', linewidth=2)
    axes[1].axvline(height_mean, color='red', linestyle='--')
    axes[1].set_title('B) Height of Adult Males')
    axes[1].set_xlabel('Height (cm)')
    axes[1].set_ylabel('Probability Density')
    
    # C) Time until a radioactive particle decays
    decay_rate = 0.1  # Decay rate parameter
    x_exp = np.linspace(0, 50, 1000)
    y_exp = stats.expon.pdf(x_exp, scale=1/decay_rate)
    
    axes[2].plot(x_exp, y_exp, 'r-', linewidth=2)
    axes[2].axvline(1/decay_rate, color='blue', linestyle='--')
    axes[2].set_title('C) Time Until a Radioactive Particle Decays')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Probability Density')
    
    # D) Proportion of defective items
    n = 100     # Sample size
    p = 0.05    # Proportion defective
    x_binom = np.arange(0, 15)
    y_binom = stats.binom.pmf(x_binom, n, p)
    
    axes[3].bar(x_binom, y_binom, alpha=0.7, color='orange')
    axes[3].axvline(n*p, color='red', linestyle='--')
    axes[3].set_title('D) Number of Defective Items in a Batch of 100')
    axes[3].set_xlabel('Number of Defective Items')
    axes[3].set_ylabel('Probability')
    
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Question 21.1: Distribution Models for Different Random Variables', 
                fontsize=16, y=1.02)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_1_distributions.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comparison of Poisson and other distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lambda_vals = [5, 10, 15, 20]
    x_range = np.arange(0, 35)
    
    for lam in lambda_vals:
        pmf = stats.poisson.pmf(x_range, lam)
        ax.plot(x_range, pmf, 'o-', linewidth=2, alpha=0.7, 
               label=f'Poisson(λ={lam})')
    
    ax.set_title('Poisson Distributions with Different λ Parameters')
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_1_poisson_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Key properties of Poisson distribution
    print("\nQuestion 21.1 Explanation:")
    print("--------------------------")
    print("Key properties of the Poisson distribution:")
    print("- Models the number of events occurring in a fixed interval")
    print("- Events occur independently of each other")
    print("- Events occur at a constant average rate")
    print("- The probability of two events occurring at exactly the same time is zero")
    
    print("\nAnalysis of each option:")
    print("A) The number of customers arriving at a store in one hour:")
    print("   - Customer arrivals occur randomly and independently")
    print("   - We're counting the number of events (arrivals) in a fixed interval (one hour)")
    print("   - There's typically a constant average rate of arrivals")
    print("   - This is a classic example of the Poisson distribution")
    
    print("\nB) The height of adult males in a population:")
    print("   - Heights are continuous measurements, not counts of events")
    print("   - Heights follow a normal (Gaussian) distribution")
    print("   - The normal distribution is symmetric and bell-shaped")
    print("   - Characterized by mean μ and standard deviation σ")
    
    print("\nC) The time until a radioactive particle decays:")
    print("   - This measures waiting time until an event occurs, not count of events")
    print("   - Follows an exponential distribution, not Poisson")
    print("   - The exponential distribution models the time between events in a Poisson process")
    print("   - If events follow Poisson(λ), time between events follows Exponential(λ)")
    
    print("\nD) The proportion of defective items in a manufacturing batch:")
    print("   - This represents a fixed number of trials (batch size) with success/failure outcomes")
    print("   - Follows a binomial distribution with parameters n (batch size) and p (defect probability)")
    print("   - Not appropriate for Poisson, which models rare events over continuous intervals")
    
    print("\nThe correct answer is A) The number of customers arriving at a store in one hour.")
    
    return "A) The number of customers arriving at a store in one hour"

def question_2_analysis(save_dir):
    """
    Analyze Question 2: Property of a statistical estimator that states its expected 
    value equals the true parameter value.
    """
    print("\n--- Question 21.2: Properties of Statistical Estimators ---")
    
    # Create figure to illustrate unbiasedness vs. bias
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True parameter value
    true_param = 10
    
    # Unbiased estimator (sample mean)
    np.random.seed(42)
    sample_sizes = np.arange(1, 501)
    unbiased_estimates = []
    
    for n in sample_sizes:
        # Generate sample from normal distribution with true mean = true_param
        sample = np.random.normal(true_param, 2, n)
        unbiased_estimates.append(np.mean(sample))
    
    # Biased estimator (sample maximum as estimator of the mean)
    biased_estimates = []
    for n in sample_sizes:
        sample = np.random.normal(true_param, 2, n)
        biased_estimates.append(np.max(sample))
    
    # Plot unbiased estimator
    ax1.plot(sample_sizes, unbiased_estimates, 'b-', alpha=0.5, label='Sample Estimates')
    ax1.axhline(true_param, color='r', linestyle='-', label='True Parameter Value')
    ax1.set_title('Unbiased Estimator: Sample Mean')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Estimate Value')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Calculate and plot mean of estimates for different sample sizes
    sample_sizes_check = [10, 50, 100, 200, 500]
    expected_unbiased = []
    
    for n in sample_sizes_check:
        estimates = []
        for _ in range(1000):
            sample = np.random.normal(true_param, 2, n)
            estimates.append(np.mean(sample))
        expected_unbiased.append(np.mean(estimates))
    
    # Plot expected value of the unbiased estimator
    ax1.plot(sample_sizes_check, expected_unbiased, 'go', markersize=8, 
            label='E[estimator] ≈ true value')
    
    # Plot biased estimator
    ax2.plot(sample_sizes, biased_estimates, 'b-', alpha=0.5, label='Sample Estimates')
    ax2.axhline(true_param, color='r', linestyle='-', label='True Parameter Value')
    ax2.set_title('Biased Estimator: Sample Maximum')
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Estimate Value')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Calculate and plot mean of biased estimates
    expected_biased = []
    for n in sample_sizes_check:
        estimates = []
        for _ in range(1000):
            sample = np.random.normal(true_param, 2, n)
            estimates.append(np.max(sample))
        expected_biased.append(np.mean(estimates))
    
    # Plot expected value of the biased estimator
    ax2.plot(sample_sizes_check, expected_biased, 'mo', markersize=8, 
            label='E[estimator] > true value')
    
    # Add overall title
    fig.suptitle('Question 21.2: Unbiasedness vs. Bias in Statistical Estimators', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_2_unbiasedness.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure to compare different estimator properties
    fig, ax = plt.subplots(figsize=(10, 6))
    
    properties = ['Unbiasedness', 'Consistency', 'Efficiency', 'Sufficiency']
    descriptions = [
        'E[estimator] = true parameter value',
        'Converges to true value as n → ∞',
        'Minimum variance among unbiased estimators',
        'Captures all information about parameter'
    ]
    
    y_pos = np.arange(len(properties))
    
    # Plot horizontal bars
    ax.barh(y_pos, [0.9, 0.8, 0.7, 0.6], height=0.5, color=['green', 'blue', 'orange', 'purple'])
    
    # Add property names and descriptions
    for i, (prop, desc) in enumerate(zip(properties, descriptions)):
        ax.text(0.01, i, f"{prop}", va='center', fontsize=12)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(properties)
    ax.set_xlabel('Property Importance Scale (Illustrative)')
    ax.set_xlim(0, 1)
    ax.set_title('Statistical Estimator Properties')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_2_estimator_properties.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print explanation
    print("\nQuestion 21.2 Explanation:")
    print("--------------------------")
    print("Properties of statistical estimators:")
    
    print("\n1. Unbiasedness: E[estimator] = true parameter value")
    print("   - An estimator θ̂ is unbiased if its expected value equals the true parameter value θ")
    print("   - Mathematically: E[θ̂] = θ")
    print("   - Example: Sample mean is an unbiased estimator of population mean")
    print("   - The estimator's average value over many samples equals the parameter we're trying to estimate")
    
    print("\n2. Consistency: Estimator converges to true value as sample size increases")
    print("   - A consistent estimator gets closer to the true parameter value as sample size grows")
    print("   - Mathematically: θ̂ → θ as n → ∞")
    print("   - Example: The sample variance is a consistent estimator of population variance")
    
    print("\n3. Efficiency: Has minimum variance among all unbiased estimators")
    print("   - An efficient estimator has the smallest variance among all unbiased estimators")
    print("   - Less dispersion means more precise estimates")
    print("   - Example: Sample mean is the most efficient estimator for the population mean in normal distributions")
    
    print("\n4. Sufficiency: Captures all information about the parameter in the sample")
    print("   - A sufficient statistic contains all the information in the sample about the parameter")
    print("   - No other statistic calculated from the same sample can add information about the parameter")
    print("   - Example: Sample mean is sufficient for the population mean in normal distributions with known variance")
    
    print("\nFrom the simulation results:")
    print(f"- For unbiased sample mean: Expected values approximately {np.mean(expected_unbiased):.4f} (true param = {true_param})")
    print(f"- For biased sample maximum: Expected values approximately {np.mean(expected_biased):.4f} (true param = {true_param})")
    
    print("\nThe correct answer is C) Unbiasedness.")
    
    return "C) Unbiasedness"

def question_3_analysis(save_dir):
    """
    Analyze Question 3: What must be subtracted to avoid double-counting in
    P(A ∪ B ∪ C)?
    """
    print("\n--- Question 21.3: Inclusion-Exclusion Principle ---")
    
    # Create simple visual representation of sets instead of Venn diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First diagram: Two sets
    circles1 = plt.Circle((0.3, 0.5), 0.3, fc='red', alpha=0.5, label='A')
    circles2 = plt.Circle((0.7, 0.5), 0.3, fc='blue', alpha=0.5, label='B')
    
    ax1.add_patch(circles1)
    ax1.add_patch(circles2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Union of Two Sets')
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Second diagram: Three sets
    circles1 = plt.Circle((0.3, 0.7), 0.25, fc='red', alpha=0.5, label='A')
    circles2 = plt.Circle((0.7, 0.7), 0.25, fc='blue', alpha=0.5, label='B')
    circles3 = plt.Circle((0.5, 0.4), 0.25, fc='green', alpha=0.5, label='C')
    
    ax2.add_patch(circles1)
    ax2.add_patch(circles2)
    ax2.add_patch(circles3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Union of Three Sets')
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_3_sets.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Numeric example visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Example values for demonstration
    p_a = 0.5
    p_b = 0.4
    p_c = 0.3
    p_ab = 0.2
    p_ac = 0.15
    p_bc = 0.1
    p_abc = 0.05
    
    # Calculate union using inclusion-exclusion
    p_union = p_a + p_b + p_c - p_ab - p_ac - p_bc + p_abc
    
    # Bar chart of terms in inclusion-exclusion formula
    terms = ['P(A)', 'P(B)', 'P(C)', '-P(A∩B)', '-P(A∩C)', '-P(B∩C)', '+P(A∩B∩C)', 'P(A∪B∪C)']
    values = [p_a, p_b, p_c, -p_ab, -p_ac, -p_bc, p_abc, p_union]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'black']
    
    ax.bar(terms[:-1], values[:-1], color=colors[:-1])
    ax.axhline(p_union, color=colors[-1], linestyle='--', linewidth=2)
    ax.text(6.5, p_union+0.02, f'P(A∪B∪C) = {p_union}', fontsize=12)
    
    ax.set_title('Inclusion-Exclusion Principle for Three Sets')
    ax.set_ylabel('Probability')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'q21_3_inclusion_exclusion_calculation.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print explanation
    print("\nQuestion 21.3 Explanation:")
    print("--------------------------")
    print("The Inclusion-Exclusion Principle for Three Events:")
    print("P(A ∪ B ∪ C) = P(A) + P(B) + P(C) - P(A ∩ B) - P(A ∩ C) - P(B ∩ C) + P(A ∩ B ∩ C)")
    
    print("\nStep-by-step explanation:")
    
    print("\n1. We start by adding the individual probabilities: P(A) + P(B) + P(C)")
    print("   - This counts every element that's in at least one of the sets")
    print("   - But elements in intersections get counted multiple times")
    print("   - Elements in exactly two sets are counted twice")
    print("   - Elements in all three sets are counted three times")
    
    print("\n2. To correct for double-counting, we subtract the pairwise intersections:")
    print("   - Subtract P(A ∩ B) to remove double-counting of elements in both A and B")
    print("   - Subtract P(A ∩ C) to remove double-counting of elements in both A and C")
    print("   - Subtract P(B ∩ C) to remove double-counting of elements in both B and C")
    
    print("\n3. But now we have a new problem with the triple intersection:")
    print("   - Elements in A ∩ B ∩ C were initially counted 3 times (in steps A, B, and C)")
    print("   - Then they were subtracted 3 times (in steps A∩B, A∩C, and B∩C)")
    print("   - So they've effectively been completely removed, when they should be counted once")
    print("   - To fix this, we need to add back P(A ∩ B ∩ C)")
    
    print("\nNumerical example:")
    print(f"If P(A) = {p_a}, P(B) = {p_b}, P(C) = {p_c}")
    print(f"P(A ∩ B) = {p_ab}, P(A ∩ C) = {p_ac}, P(B ∩ C) = {p_bc}")
    print(f"P(A ∩ B ∩ C) = {p_abc}")
    print(f"Then P(A ∪ B ∪ C) = {p_a} + {p_b} + {p_c} - {p_ab} - {p_ac} - {p_bc} + {p_abc} = {p_union}")
    
    print("\nAnswer analysis:")
    print("A) Only subtracting P(A ∩ B) would account for just one pairwise intersection")
    print("B) Only subtracting P(A ∩ B ∩ C) would not account for double-counting in pairwise intersections")
    print("C) Subtracting all pairwise intersections and then adding back the triple intersection correctly accounts for all cases")
    print("D) The expected value is not relevant to the inclusion-exclusion principle")
    
    print("\nThe correct answer is C) P(A ∩ B), P(B ∩ C), P(A ∩ C), and then add P(A ∩ B ∩ C)")
    
    return "C) P(A ∩ B), P(B ∩ C), P(A ∩ C), and then add P(A ∩ B ∩ C)"

def main():
    # Create directory for saving images
    save_dir = create_images_directory()
    print(f"Images will be saved to: {save_dir}")
    
    # Solve and analyze each question
    question_1_answer = question_1_analysis(save_dir)
    question_2_answer = question_2_analysis(save_dir)
    question_3_answer = question_3_analysis(save_dir)
    
    # Print summary of answers
    print("\n=== Summary of Answers ===")
    print(f"Question 21.1: {question_1_answer}")
    print(f"Question 21.2: {question_2_answer}")
    print(f"Question 21.3: {question_3_answer}")

if __name__ == "__main__":
    main() 