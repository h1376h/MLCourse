import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Coin Flip Likelihood
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)
x = 7  # successes
n = 10  # trials

# Calculate likelihood
likelihoods = [stats.binom.pmf(x, n, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood L(θ|7 heads in 10 flips)', fontsize=12)
plt.title('Likelihood Function for 7 Heads in 10 Coin Flips', fontsize=14)

# Mark specific parameter values
specific_theta = [0.3, 0.5, 0.7, 0.9]
specific_likelihoods = [stats.binom.pmf(x, n, theta) for theta in specific_theta]
plt.plot(specific_theta, specific_likelihoods, 'ro', markersize=8)

# Annotate points
for theta, likelihood in zip(specific_theta, specific_likelihoods):
    plt.annotate(f'θ={theta}: L={likelihood:.3f}', 
                xy=(theta, likelihood), 
                xytext=(theta + 0.05, likelihood),
                fontsize=9,
                arrowprops=dict(facecolor='black', width=1, shrink=0.05))

# Mark MLE
mle = x / n
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.1f}')

plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_coin_flip.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Dice Roll Likelihood
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)
x = 3  # successes (number of 6's)
n = 5  # trials (dice rolls)

# Calculate likelihood
likelihoods = [stats.binom.pmf(x, n, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of Rolling a 6)', fontsize=12)
plt.ylabel('Likelihood L(θ|3 sixes in 5 rolls)', fontsize=12)
plt.title('Likelihood Function for 3 Sixes in 5 Dice Rolls', fontsize=14)

# Mark specific parameter values
specific_theta = [0.1, 0.3, 0.6, 0.9]
specific_likelihoods = [stats.binom.pmf(x, n, theta) for theta in specific_theta]
plt.plot(specific_theta, specific_likelihoods, 'ro', markersize=8)

# Annotate points
for theta, likelihood in zip(specific_theta, specific_likelihoods):
    plt.annotate(f'θ={theta}: L={likelihood:.3f}', 
                xy=(theta, likelihood), 
                xytext=(theta + 0.05, likelihood),
                fontsize=9,
                arrowprops=dict(facecolor='black', width=1, shrink=0.05))

# Mark MLE
mle = x / n
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.1f}')

plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_dice_roll.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Basketball Free Throw Likelihood
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)
x = 5  # successes (made shots)
n = 7  # trials (attempts)

# Calculate likelihood
likelihoods = [stats.binom.pmf(x, n, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of Making a Free Throw)', fontsize=12)
plt.ylabel('Likelihood L(θ|5 makes in 7 attempts)', fontsize=12)
plt.title('Likelihood Function for 5 Makes in 7 Free Throw Attempts', fontsize=14)

# Mark specific parameter values
specific_theta = [0.4, 0.6, 0.7, 0.8]
specific_likelihoods = [stats.binom.pmf(x, n, theta) for theta in specific_theta]
plt.plot(specific_theta, specific_likelihoods, 'ro', markersize=8)

# Annotate points
for theta, likelihood in zip(specific_theta, specific_likelihoods):
    plt.annotate(f'θ={theta}: L={likelihood:.3f}', 
                xy=(theta, likelihood), 
                xytext=(theta + 0.05, likelihood),
                fontsize=9,
                arrowprops=dict(facecolor='black', width=1, shrink=0.05))

# Mark MLE
mle = x / n
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.2f}')

plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_basketball.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example the 4: Normal Distribution Likelihood
plt.figure(figsize=(10, 6))
x = np.linspace(140, 190, 200)

# Sample data
heights = [158, 162, 171, 175, 164]
mean_height = np.mean(heights)
std_height = np.std(heights)

# Plot original data points
plt.plot(heights, [0.005] * len(heights), 'ko', markersize=8, label='Data points')

# Plot different normal distributions
params = [(160, 5), (165, 7), (170, 10)]
colors = ['r', 'g', 'b']
labels = ['μ=160, σ=5', 'μ=165, σ=7', 'μ=170, σ=10']

for (mu, sigma), color, label in zip(params, colors, labels):
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y, color=color, linewidth=2, label=label)
    
    # Calculate likelihood for this parameter set
    likelihood = np.prod([stats.norm.pdf(h, mu, sigma) for h in heights])
    plt.annotate(f'L={likelihood:.2e}', 
                xy=(mu, stats.norm.pdf(mu, mu, sigma)),
                xytext=(mu, stats.norm.pdf(mu, mu, sigma) + 0.003),
                fontsize=10, ha='center')

# Mark MLE
plt.axvline(x=mean_height, color='k', linestyle='--', alpha=0.7, 
           label=f'MLE: μ={mean_height:.1f}')

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Normal Distribution Likelihood for Height Data', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_normal.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Log-Likelihood Maximization
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0.01, 0.99, 100)
x = 8  # successes
n = 10  # trials

# Calculate likelihood and log-likelihood
likelihoods = [stats.binom.pmf(x, n, theta) for theta in theta_values]
log_likelihoods = [np.log(l) for l in likelihoods]

# Create two subplots
plt.subplot(2, 1, 1)
plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.ylabel('Likelihood L(θ|8 heads in 10 flips)', fontsize=12)
plt.title('Likelihood Function for 8 Heads in 10 Coin Flips', fontsize=14)

# Mark MLE on likelihood plot
mle = x / n
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.1f}')
plt.legend(fontsize=10)

plt.subplot(2, 1, 2)
plt.plot(theta_values, log_likelihoods, 'r-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Log-Likelihood log(L)', fontsize=12)
plt.title('Log-Likelihood Function', fontsize=14)

# Mark MLE on log-likelihood plot
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.1f}')

# Add derivative at MLE
plt.annotate('Derivative = 0 at MLE', 
            xy=(mle, np.log(stats.binom.pmf(x, n, mle))),
            xytext=(mle+0.2, np.log(stats.binom.pmf(x, n, mle))-1),
            fontsize=10,
            arrowprops=dict(facecolor='black', width=1, shrink=0.05))

plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'log_likelihood_maximization.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Comparing Different Likelihood Functions
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0.01, 0.99, 100)

# Define two datasets
dataset1 = [1, 1, 1, 0, 0]  # 3 heads, 2 tails
dataset2 = [1, 1, 0, 0, 0]  # 2 heads, 3 tails

# Calculate likelihoods
def coin_likelihood(data, theta):
    heads = sum(data)
    tails = len(data) - heads
    return theta**heads * (1-theta)**tails

likelihoods1 = [coin_likelihood(dataset1, theta) for theta in theta_values]
likelihoods2 = [coin_likelihood(dataset2, theta) for theta in theta_values]

# Plot likelihood functions
plt.plot(theta_values, likelihoods1, 'b-', linewidth=2, label='Data: 3H, 2T')
plt.plot(theta_values, likelihoods2, 'r-', linewidth=2, label='Data: 2H, 3T')

# Mark MLEs
mle1 = sum(dataset1) / len(dataset1)
mle2 = sum(dataset2) / len(dataset2)
plt.axvline(x=mle1, color='b', linestyle='--', alpha=0.7, label=f'MLE for 3H,2T: θ={mle1:.1f}')
plt.axvline(x=mle2, color='r', linestyle='--', alpha=0.7, label=f'MLE for 2H,3T: θ={mle2:.1f}')

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.title('Likelihood Functions for Different Coin Flip Datasets', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'different_likelihood_functions.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Likelihood Ratio Test
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0.01, 0.99, 100)
x = 7  # observed successes
n = 10  # trials

# Calculate likelihoods for two models
likelihoods_unrestricted = [stats.binom.pmf(x, n, theta) for theta in theta_values]
likelihood_restricted = stats.binom.pmf(x, n, 0.5)  # null hypothesis: θ = 0.5

# Calculate likelihood ratios
likelihood_ratios = [l / likelihood_restricted for l in likelihoods_unrestricted]

# Plot likelihood ratio
plt.plot(theta_values, likelihood_ratios, 'b-', linewidth=2)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='LR = 1')
plt.grid(True, alpha=0.3)

# Mark MLE
mle = x / n
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.1f}')

# Mark specific models
plt.plot([0.5, 0.7], [1, likelihood_ratios[np.abs(theta_values - 0.7).argmin()]], 'ro', markersize=8)

# Set proper y-axis limits to better display the ratio values
max_ratio = max(likelihood_ratios)

# Annotate points with better positioning
plt.annotate(f'Null: θ=0.5\nLR=1', 
            xy=(0.5, 1), 
            xytext=(0.3, max_ratio * 0.3),  # Position relative to max height
            fontsize=10,
            arrowprops=dict(facecolor='black', width=1, shrink=0.05))

plt.annotate(f'Alt: θ=0.7\nLR≈{likelihood_ratios[np.abs(theta_values - 0.7).argmin()]:.1f}', 
            xy=(0.7, likelihood_ratios[np.abs(theta_values - 0.7).argmin()]), 
            xytext=(0.7, max_ratio * 0.7),  # Position relative to max height
            fontsize=10,
            arrowprops=dict(facecolor='black', width=1, shrink=0.05))

plt.xlabel('θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood Ratio L(θ|data) / L(0.5|data)', fontsize=12)
plt.title('Likelihood Ratio Test for Coin Bias', fontsize=14)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_ratio_test.png'), dpi=100, bbox_inches='tight')
plt.close()

print("Likelihood example images created successfully.") 