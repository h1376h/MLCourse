import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom
import os

# Create Images directory if it doesn't exist
os.makedirs('../Images', exist_ok=True)

def plot_bernoulli(p):
    """Plot Bernoulli distribution for a given probability p."""
    plt.figure(figsize=(8, 6))
    x = [0, 1]
    probs = [1-p, p]
    
    bars = plt.bar(x, probs, color=['skyblue', 'lightgreen'], alpha=0.7)
    plt.xticks(x, ['Tails (0)', 'Heads (1)'])
    plt.ylabel('Probability')
    plt.title(f'Bernoulli Distribution (p = {p})')
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/bernoulli_coin_toss.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_binomial(n, p):
    """Plot Binomial distribution for given n and p."""
    plt.figure(figsize=(10, 6))
    x = np.arange(0, n+1)
    probs = binom.pmf(x, n, p)
    
    bars = plt.bar(x, probs, color='skyblue', alpha=0.7)
    plt.xticks(x)
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution (n = {n}, p = {p})')
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:  # Only show significant probabilities
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Add mean and standard deviation lines
    mean = n * p
    std = np.sqrt(n * p * (1-p))
    plt.axvline(x=mean, color='red', linestyle='--', label=f'Mean = {mean:.1f}')
    plt.axvline(x=mean - std, color='green', linestyle=':', label=f'Mean ± Std Dev')
    plt.axvline(x=mean + std, color='green', linestyle=':')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/binomial_coin_tosses.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_control(n, p):
    """Plot Binomial distribution for quality control example."""
    plt.figure(figsize=(10, 6))
    x = np.arange(0, n+1)
    probs = binom.pmf(x, n, p)
    
    bars = plt.bar(x, probs, color='skyblue', alpha=0.7)
    plt.xticks(x)
    plt.xlabel('Number of Defective Bulbs')
    plt.ylabel('Probability')
    plt.title(f'Quality Control: Binomial Distribution (n = {n}, p = {p})')
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:  # Only show significant probabilities
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Highlight specific probabilities
    plt.axvspan(1.5, 2.5, color='yellow', alpha=0.2, label='P(X=2)')
    plt.axvspan(-0.5, 2.5, color='green', alpha=0.1, label='P(X≤2)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../Images/binomial_quality_control.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Bernoulli and Binomial Distribution Examples ===\n")
    
    # Example 1: Bernoulli Distribution
    print("Example 1: Bernoulli Distribution - Coin Toss")
    p = 0.6
    print(f"Probability of success (heads): p = {p}")
    print(f"Probability of failure (tails): 1-p = {1-p}")
    print(f"Expected value: E[X] = p = {p}")
    print(f"Variance: Var(X) = p(1-p) = {p*(1-p):.2f}\n")
    plot_bernoulli(p)
    
    # Example 2: Binomial Distribution
    print("Example 2: Binomial Distribution - Multiple Coin Tosses")
    n = 10
    p = 0.6
    print(f"Number of trials: n = {n}")
    print(f"Probability of success: p = {p}")
    print(f"Expected value: E[X] = np = {n*p}")
    print(f"Variance: Var(X) = np(1-p) = {n*p*(1-p):.2f}")
    
    # Calculate specific probabilities
    print("\nSpecific probabilities:")
    for k in [0, 5, 10]:
        prob = binom.pmf(k, n, p)
        print(f"P(X={k}) = {prob:.10f}")
    print()
    plot_binomial(n, p)
    
    # Example 3: Quality Control
    print("Example 3: Binomial Distribution - Quality Control")
    n = 20
    p = 0.05
    print(f"Number of bulbs: n = {n}")
    print(f"Defect rate: p = {p}")
    
    # Calculate probabilities
    p_exactly_2 = binom.pmf(2, n, p)
    p_at_most_2 = sum(binom.pmf(k, n, p) for k in range(3))
    p_at_least_2 = 1 - binom.pmf(0, n, p) - binom.pmf(1, n, p)
    
    print(f"\nProbabilities:")
    print(f"P(X=2) = {p_exactly_2:.10f}")
    print(f"P(X≤2) = {p_at_most_2:.10f}")
    print(f"P(X≥2) = {p_at_least_2:.10f}")
    print()
    plot_quality_control(n, p)
    
    print("All visualizations have been saved to the Images directory.")

if __name__ == "__main__":
    main() 