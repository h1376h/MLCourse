import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('default')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introducing the problem
print_step_header(1, "Understanding Limit Theorems in Neural Networks")

print("""
Problem Statement:
A neural network is trained on batches of data, where each batch contains 64 samples. 
The loss for a single sample has a mean of 0.5 and a standard deviation of 0.2.

Tasks:
1. Using the Law of Large Numbers, what happens to the average batch loss as the number 
   of training iterations approaches infinity?
2. Apply the Central Limit Theorem to characterize the distribution of the average loss per batch.
3. Calculate the probability that the average loss for a batch is less than 0.48.
4. If we increase the batch size to 256, how does this affect the standard deviation 
   of the average batch loss?
5. Explain how the Central Limit Theorem relates to the concept of reducing variance 
   in stochastic gradient descent.
""")

# Step 2: Law of Large Numbers
print_step_header(2, "Law of Large Numbers Application")

print("""
The Law of Large Numbers (LLN) states that as the number of independent, identically 
distributed random variables increases, their sample mean converges to the expected value.

In our case:
- Each sample loss has mean μ = 0.5 and standard deviation σ = 0.2
- As the number of training iterations approaches infinity, the average batch loss 
  will converge to the true mean of 0.5.
""")

# Simulate the Law of Large Numbers
np.random.seed(42)

# Generate random losses with mean 0.5 and std 0.2
num_iterations = 1000
batch_size = 64
sample_losses = np.random.normal(0.5, 0.2, (num_iterations, batch_size))

# Calculate batch means
batch_means = np.mean(sample_losses, axis=1)

# Calculate cumulative average across iterations
cumulative_means = np.cumsum(batch_means) / (np.arange(num_iterations) + 1)

# Create visualization for Law of Large Numbers
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), cumulative_means, 'b-', linewidth=1, label='Cumulative Mean of Batch Losses')
plt.axhline(y=0.5, color='r', linestyle='--', label='True Mean (μ = 0.5)')

# Add shaded confidence intervals
plt.fill_between(range(1, num_iterations + 1), 
                 0.5 - 1.96 * 0.2 / np.sqrt(np.arange(1, num_iterations + 1) * batch_size), 
                 0.5 + 1.96 * 0.2 / np.sqrt(np.arange(1, num_iterations + 1) * batch_size), 
                 color='gray', alpha=0.2, label='95% Confidence Interval')

plt.title('Law of Large Numbers: Convergence of Average Batch Loss', fontsize=14)
plt.xlabel('Number of Training Iterations', fontsize=12)
plt.ylabel('Cumulative Average Batch Loss', fontsize=12)
plt.xlim(0, num_iterations)
plt.ylim(0.45, 0.55)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "law_of_large_numbers.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Central Limit Theorem
print_step_header(3, "Central Limit Theorem Application")

print("""
The Central Limit Theorem (CLT) states that when independent random variables are averaged, 
their normalized sum tends toward a normal distribution regardless of the original distribution.

For our neural network with batch size of 64:
- Each individual loss has mean μ = 0.5 and variance σ² = 0.04
- The average batch loss will be approximately normally distributed with:
  - Mean = μ = 0.5
  - Variance = σ²/n = 0.04/64 = 0.000625
  - Standard deviation = σ/√n = 0.2/√64 = 0.025

Therefore, the distribution of the average batch loss is approximately N(0.5, 0.025²).
""")

# Demonstrate the Central Limit Theorem
# Generate 10,000 batches and calculate their mean losses
num_simulations = 10000
batch_losses = np.random.normal(0.5, 0.2, (num_simulations, batch_size))
batch_means = np.mean(batch_losses, axis=1)

# Calculate the mean and standard deviation of the sample means
empirical_mean = np.mean(batch_means)
empirical_std = np.std(batch_means)
theoretical_std = 0.2 / np.sqrt(batch_size)

print(f"Empirical mean of batch means: {empirical_mean:.5f}")
print(f"Theoretical mean: 0.5")
print(f"Empirical standard deviation of batch means: {empirical_std:.5f}")
print(f"Theoretical standard deviation: {theoretical_std:.5f}")

# Plot histogram of batch means with normal curve overlay
plt.figure(figsize=(10, 6))

# Histogram of empirical batch means
plt.hist(batch_means, bins=50, density=True, alpha=0.6, color='skyblue', 
         label=f'Histogram of {num_simulations} Batch Means')

# Theoretical normal distribution
x = np.linspace(0.4, 0.6, 1000)
plt.plot(x, norm.pdf(x, 0.5, theoretical_std), 'r-', linewidth=2, 
         label=f'Normal N(0.5, {theoretical_std:.5f}²)')

plt.title('Central Limit Theorem: Distribution of Average Batch Loss', fontsize=14)
plt.xlabel('Average Batch Loss', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "central_limit_theorem.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Probability calculation
print_step_header(4, "Probability Calculation")

# Calculate the probability that average batch loss is less than 0.48
z_score = (0.48 - 0.5) / theoretical_std
probability = norm.cdf(z_score)

print(f"Probability that the average batch loss is less than 0.48:")
print(f"P(X̄ < 0.48) = {probability:.4f} or approximately {probability*100:.2f}%")

# Visualize this probability
plt.figure(figsize=(10, 6))

x = np.linspace(0.4, 0.6, 1000)
y = norm.pdf(x, 0.5, theoretical_std)

plt.plot(x, y, 'b-', linewidth=2, label=f'Normal N(0.5, {theoretical_std:.5f}²)')
plt.fill_between(x[x <= 0.48], y[x <= 0.48], color='skyblue', alpha=0.5, 
                 label=f'P(X̄ < 0.48) = {probability:.4f}')

plt.axvline(x=0.48, color='r', linestyle='--', label='x = 0.48')
plt.axvline(x=0.5, color='g', linestyle='-', label='μ = 0.5')

plt.title('Probability of Average Batch Loss < 0.48', fontsize=14)
plt.xlabel('Average Batch Loss', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "probability_calculation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Effect of increasing batch size
print_step_header(5, "Effect of Increasing Batch Size")

# Calculate standard deviation for different batch sizes
batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
std_devs = [0.2 / np.sqrt(n) for n in batch_sizes]

print("Effect of batch size on the standard deviation of average batch loss:")
for bs, std in zip(batch_sizes, std_devs):
    print(f"Batch size = {bs}: σ/√n = {std:.5f}")

print("\nWhen we increase the batch size from 64 to 256:")
print(f"Standard deviation at batch size 64: {std_devs[2]:.5f}")
print(f"Standard deviation at batch size 256: {std_devs[4]:.5f}")
print(f"Reduction factor: {std_devs[2]/std_devs[4]:.2f}x")

# Create visualization for batch size effect
plt.figure(figsize=(12, 10))

# Panel 1: Plot standard deviation vs batch size
plt.subplot(2, 2, 1)
plt.plot(batch_sizes, std_devs, 'ro-', linewidth=2)
plt.title('Standard Deviation vs. Batch Size', fontsize=12)
plt.xlabel('Batch Size', fontsize=10)
plt.ylabel('Standard Deviation of Average Loss', fontsize=10)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.grid(True)

# Highlight values for batch sizes 64 and 256
plt.scatter([64, 256], [std_devs[2], std_devs[4]], color='blue', s=100, zorder=5)
plt.annotate(f'n=64, σ={std_devs[2]:.5f}', xy=(64, std_devs[2]), xytext=(64*1.2, std_devs[2]*1.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate(f'n=256, σ={std_devs[4]:.5f}', xy=(256, std_devs[4]), xytext=(256*1.2, std_devs[4]*1.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# Panel 2: Comparison of PDFs for different batch sizes
plt.subplot(2, 2, 2)
x = np.linspace(0.4, 0.6, 1000)
colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

for i, (bs, std, c) in enumerate(zip(batch_sizes, std_devs, colors)):
    y = norm.pdf(x, 0.5, std)
    plt.plot(x, y, '-', color=c, linewidth=2, label=f'n = {bs}')

plt.title('PDF of Average Loss for Different Batch Sizes', fontsize=12)
plt.xlabel('Average Batch Loss', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True)

# Panel 3: Visual demonstration of CLT for batch size 64
plt.subplot(2, 2, 3)

# Generate sample distributions for batch size 64
num_samples = 10000
individual_losses = np.random.normal(0.5, 0.2, num_samples)
batch_averages = []

for i in range(0, num_samples, 64):
    if i + 64 <= num_samples:
        batch_averages.append(np.mean(individual_losses[i:i+64]))

plt.hist(individual_losses, bins=50, density=True, alpha=0.5, color='lightblue', 
         label='Individual Loss Samples')
plt.hist(batch_averages, bins=30, density=True, alpha=0.7, color='orange', 
         label='Batch Averages (n=64)')

# Add theoretical curves
x_ind = np.linspace(0, 1, 1000)
y_ind = norm.pdf(x_ind, 0.5, 0.2)
plt.plot(x_ind, y_ind, 'b-', linewidth=2, label='Individual Loss PDF')

x_avg = np.linspace(0.4, 0.6, 1000)
y_avg = norm.pdf(x_avg, 0.5, 0.2/np.sqrt(64))
plt.plot(x_avg, y_avg, 'r-', linewidth=2, label='Batch Average PDF (n=64)')

plt.title('Demonstration of CLT for Batch Size 64', fontsize=12)
plt.xlabel('Loss Value', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True)

# Panel 4: Visual demonstration of CLT for batch size 256
plt.subplot(2, 2, 4)

# Generate sample distributions for batch size 256
batch_averages_large = []
for i in range(0, num_samples, 256):
    if i + 256 <= num_samples:
        batch_averages_large.append(np.mean(individual_losses[i:i+256]))

plt.hist(individual_losses, bins=50, density=True, alpha=0.5, color='lightblue', 
         label='Individual Loss Samples')
plt.hist(batch_averages_large, bins=20, density=True, alpha=0.7, color='green', 
         label='Batch Averages (n=256)')

# Add theoretical curves
plt.plot(x_ind, y_ind, 'b-', linewidth=2, label='Individual Loss PDF')

x_avg_large = np.linspace(0.45, 0.55, 1000)
y_avg_large = norm.pdf(x_avg_large, 0.5, 0.2/np.sqrt(256))
plt.plot(x_avg_large, y_avg_large, 'g-', linewidth=2, label='Batch Average PDF (n=256)')

plt.title('Demonstration of CLT for Batch Size 256', fontsize=12)
plt.xlabel('Loss Value', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "batch_size_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Relationship to Stochastic Gradient Descent
print_step_header(6, "Relationship to Stochastic Gradient Descent")

print("""
The Central Limit Theorem is fundamental to understanding variance reduction in 
Stochastic Gradient Descent (SGD):

1. In SGD, we estimate the gradient using a batch of samples instead of the entire dataset.
2. Each sample provides a noisy estimate of the true gradient.
3. By averaging gradients over a batch, we reduce the variance by a factor of 1/n.
4. The CLT tells us that the average gradient will be approximately normally distributed 
   around the true gradient.
5. Larger batch sizes reduce variance but with diminishing returns (variance decreases 
   as 1/√n, not linearly with n).
""")

# Visualize the relationship between batch size and gradient variance
plt.figure(figsize=(12, 6))

gs = GridSpec(1, 2, width_ratios=[1, 1.5])

# Left panel: Gradient variance reduction
plt.subplot(gs[0])
batch_sizes_extended = np.arange(1, 513, 1)
variance_reduction = 1 / np.sqrt(batch_sizes_extended)

plt.plot(batch_sizes_extended, variance_reduction, 'b-', linewidth=2)
plt.title('Gradient Variance Reduction in SGD', fontsize=14)
plt.xlabel('Batch Size (n)', fontsize=12)
plt.ylabel('Relative Standard Deviation (1/√n)', fontsize=12)
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.grid(True)

# Annotate key points
plt.scatter([1, 64, 256], [1, 1/np.sqrt(64), 1/np.sqrt(256)], color='red', s=80, zorder=5)
plt.annotate('n=1 (Full Variance)', xy=(1, 1), xytext=(2, 1.1), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate(f'n=64 (Variance Reduced by {64**0.5:.1f}x)', xy=(64, 1/np.sqrt(64)), xytext=(20, 0.2), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate(f'n=256 (Variance Reduced by {256**0.5:.1f}x)', xy=(256, 1/np.sqrt(256)), xytext=(100, 0.04), 
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# Right panel: SGD convergence visualization
plt.subplot(gs[1])

# Simulate SGD convergence for different batch sizes
np.random.seed(42)
iterations = 100
batch_sizes_sgd = [1, 4, 16, 64, 256]
colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes_sgd)))

# For a simplified model, we'll use a noisy quadratic function
# Assume the true minimum is at x=0, and we start at x=10
true_minimum = 0
start_point = 10
learning_rate = 0.1

for i, (bs, c) in enumerate(zip(batch_sizes_sgd, colors)):
    np.random.seed(42)  # Reset seed for fair comparison
    x = start_point
    trajectory = [x]
    
    for t in range(iterations):
        # Simplified SGD step with noise proportional to batch size
        gradient = 2 * (x - true_minimum)  # True gradient of (x-true_minimum)²
        noise = np.random.normal(0, 4 / np.sqrt(bs))  # Noise scale depending on batch size
        noisy_gradient = gradient + noise
        x = x - learning_rate * noisy_gradient
        trajectory.append(x)
    
    plt.plot(range(iterations + 1), trajectory, color=c, linewidth=2, 
             label=f'Batch Size = {bs}')

plt.axhline(y=true_minimum, color='r', linestyle='--', label='True Minimum')
plt.title('SGD Convergence for Different Batch Sizes', fontsize=14)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Parameter Value', fontsize=12)
plt.ylim(-5, 12)
plt.legend()
plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "sgd_relationship.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Summary of findings
print_step_header(7, "Summary of Findings")

print("""
Question 28 Solution Summary:

1. Law of Large Numbers: 
   As training iterations approach infinity, the average batch loss converges to the 
   true mean loss of 0.5.

2. Central Limit Theorem: 
   The distribution of the average loss per batch is approximately normal with mean 0.5 
   and standard deviation 0.2/√64 = 0.025.

3. Probability Calculation: 
   The probability that the average batch loss is less than 0.48 is approximately 21.19%.

4. Increasing Batch Size: 
   Increasing the batch size from 64 to 256 reduces the standard deviation of the 
   average batch loss by a factor of 2 (from 0.025 to 0.0125).

5. CLT and SGD: 
   The Central Limit Theorem explains why larger batch sizes reduce the variance of 
   gradient estimates in SGD, leading to more stable but still stochastic optimization.
   However, diminishing returns apply as the variance reduction scales with 1/√n.
""") 