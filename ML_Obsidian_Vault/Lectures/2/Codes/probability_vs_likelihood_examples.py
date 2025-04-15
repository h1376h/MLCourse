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

# Example 1: Visual explanation of probability vs likelihood interpretations
plt.figure(figsize=(12, 8))

# Sample data for a coin flip experiment
flip_results = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]  # 0=tails, 1=heads
n_flips = len(flip_results)
n_heads = sum(flip_results)

# Create two subplots
plt.subplot(2, 1, 1)
plt.bar(['Tails', 'Heads'], [n_flips - n_heads, n_heads], color=['skyblue', 'salmon'])
plt.title('Observed Data: 6 Heads, 4 Tails', fontsize=14)
plt.ylabel('Count', fontsize=12)

# Add annotations
for i, model in enumerate([('Fair Coin', 0.5), ('Biased Coin', 0.7)]):
    name, prob = model
    expected_heads = n_flips * prob
    expected_tails = n_flips * (1-prob)
    
    # Compute probability of observed data under this model
    prob_value = stats.binom.pmf(n_heads, n_flips, prob)
    
    plt.annotate(f'{name} Model (θ={prob}):\nExpected: {expected_heads:.1f}H, {expected_tails:.1f}T\nP(6H,4T|θ={prob}) = {prob_value:.4f}', 
                xy=(0.5, 0),
                xytext=(1.1 + i*0.8, 4),
                fontsize=10)

# Likelihood perspective
plt.subplot(2, 1, 2)
theta_values = np.linspace(0, 1, 100)
likelihoods = [stats.binom.pmf(n_heads, n_flips, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.fill_between(theta_values, likelihoods, alpha=0.2)
plt.grid(True, alpha=0.3)

# Mark specific models
plt.plot([0.5, 0.7], [stats.binom.pmf(n_heads, n_flips, 0.5), 
                     stats.binom.pmf(n_heads, n_flips, 0.7)], 
        'ro', markersize=8)

# Add annotations
plt.annotate('Fair Coin Model\nθ=0.5\nL(0.5|data) = 0.2051', 
            xy=(0.5, stats.binom.pmf(n_heads, n_flips, 0.5)),
            xytext=(0.2, 0.15),
            fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Biased Coin Model\nθ=0.7\nL(0.7|data) = 0.2001', 
            xy=(0.7, stats.binom.pmf(n_heads, n_flips, 0.7)),
            xytext=(0.7, 0.15),
            fontsize=10,
            arrowprops=dict(facecolor='black', shrink=0.05))

# Mark MLE
mle = theta_values[np.argmax(likelihoods)]
plt.axvline(x=mle, color='g', linestyle='--', alpha=0.7, 
           label=f'MLE: θ={mle:.2f}')

plt.xlabel('Parameter θ (Probability of Heads)', fontsize=12)
plt.ylabel('Likelihood L(θ|data)', fontsize=12)
plt.title('Likelihood Perspective: L(θ|6 Heads, 4 Tails)', fontsize=14)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'prob_vs_like_interpretation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Dice likelihood
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)
likelihoods = [stats.binom.pmf(3, 5, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of rolling a 6)', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.title('Likelihood Function for 3 Sixes in 5 Rolls', fontsize=14)

# Mark specific points
points = [0.1, 0.3, 0.6, 0.9]
point_likelihoods = [stats.binom.pmf(3, 5, p) for p in points]
plt.plot(points, point_likelihoods, 'ro', markersize=8)

# Annotate points
for i, (p, l) in enumerate(zip(points, point_likelihoods)):
    plt.annotate(f'θ={p}', 
                xy=(p, l), 
                xytext=(p+0.03, l+0.01 if i < 2 else l-0.02),
                fontsize=10)

plt.axvline(x=0.6, color='g', linestyle='--', alpha=0.7, label='MLE: θ=0.6')
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'dice_likelihood.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Coin flip likelihood
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)
likelihoods = [stats.binom.pmf(7, 10, theta) for theta in theta_values]

plt.plot(theta_values, likelihoods, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('θ (Probability of heads)', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.title('Likelihood Function for 7 Heads in 10 Flips', fontsize=14)

# Mark specific points
points = [0.3, 0.5, 0.7, 0.9]
point_likelihoods = [stats.binom.pmf(7, 10, p) for p in points]
plt.plot(points, point_likelihoods, 'ro', markersize=8)

# Annotate points
for i, (p, l) in enumerate(zip(points, point_likelihoods)):
    plt.annotate(f'θ={p}', 
                xy=(p, l), 
                xytext=(p+0.03, l+0.01 if i < 2 else l-0.02),
                fontsize=10)

plt.axvline(x=0.7, color='g', linestyle='--', alpha=0.7, label='MLE: θ=0.7')
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'coin_likelihood.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Normal distribution likelihood
plt.figure(figsize=(10, 6))
x = np.linspace(140, 190, 200)

# Sample data
heights = [158, 162, 171, 175, 164]
mean_height = np.mean(heights)

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

# Mark data points
for height in heights:
    plt.axvline(x=height, color='k', linestyle='--', alpha=0.3)

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Normal Distribution Likelihood for Height Data', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'normal_likelihood.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Probability distribution
plt.figure(figsize=(10, 6))
x = np.linspace(140, 190, 200)
mu, sigma = 165, 7  # Fixed parameters

# Plot probability distribution
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, y, alpha=0.2)

# Highlight specific probability regions
region1 = (x >= 160) & (x <= 170)
region2 = x >= 180
plt.fill_between(x[region1], y[region1], color='g', alpha=0.4, label='P(160≤X≤170)≈0.496')
plt.fill_between(x[region2], y[region2], color='r', alpha=0.4, label='P(X>180)≈0.016')

plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Probability Distribution: Female Heights (μ=165, σ=7)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'probability_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Likelihood function
plt.figure(figsize=(10, 6))
theta_values = np.linspace(0, 1, 100)

# Two different datasets
data1 = [1, 1, 1, 0, 0]  # 3 heads, 2 tails
data2 = [0, 0, 0, 1, 1]  # 2 heads, 3 tails

# Calculate likelihoods
def coin_likelihood(data, theta):
    heads = sum(data)
    tails = len(data) - heads
    return theta**heads * (1-theta)**tails

likelihoods1 = [coin_likelihood(data1, theta) for theta in theta_values]
likelihoods2 = [coin_likelihood(data2, theta) for theta in theta_values]

# Plot likelihood functions
plt.plot(theta_values, likelihoods1, 'b-', linewidth=2, label='Data: 3H, 2T')
plt.plot(theta_values, likelihoods2, 'r-', linewidth=2, label='Data: 2H, 3T')

# Mark MLEs
mle1 = data1.count(1) / len(data1)
mle2 = data2.count(1) / len(data2)
plt.axvline(x=mle1, color='b', linestyle='--', alpha=0.7)
plt.axvline(x=mle2, color='r', linestyle='--', alpha=0.7)

plt.xlabel('θ (Probability of heads)', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.title('Likelihood Functions for Different Coin Flip Datasets', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'likelihood_function.png'), dpi=100, bbox_inches='tight')
plt.close()

print("Probability vs Likelihood example images created successfully.")