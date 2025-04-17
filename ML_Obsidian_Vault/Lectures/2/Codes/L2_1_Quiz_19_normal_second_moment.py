import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- We have a random sample X₁, X₂, ..., Xₙ from N(μ, 1)")
print("- Each Xᵢ has mean μ and variance 1")
print("- The random variables are independent")
print()
print("Question: What is the expected value of Xᵢ²?")
print()

# Step 2: Visual demonstration of a normal distribution
print_step_header(2, "Visualizing the Normal Distribution")

# Demonstrate the normal distribution for different means
mu_values = [-2, 0, 2]
x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(10, 6))
for mu in mu_values:
    pdf = norm.pdf(x, loc=mu, scale=1)
    plt.plot(x, pdf, label=f'μ = {mu}, σ² = 1')
    
plt.title('Normal Distributions with Different Means (σ² = 1)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "normal_distributions.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Recall properties of expectation
print_step_header(3, "Recalling Properties of Expectation")

print("For a random variable X with mean μₓ and variance σₓ²:")
print("1. E[X] = μₓ")
print("2. Var(X) = E[(X - μₓ)²] = E[X²] - (E[X])²")
print("3. Therefore, E[X²] = Var(X) + (E[X])² = σₓ² + μₓ²")
print()

# Step 4: Apply to our normal distribution
print_step_header(4, "Applying to Our Normal Distribution")

print("For each Xᵢ ~ N(μ, 1):")
print("- E[Xᵢ] = μ")
print("- Var(Xᵢ) = 1")
print("- Therefore, E[Xᵢ²] = Var(Xᵢ) + (E[Xᵢ])² = 1 + μ²")
print()

# Step 5: Visual demonstration with simulations
print_step_header(5, "Verifying with Simulations")

# Simulate samples from normal distributions with different means
np.random.seed(42)
mu_values = [-2, 0, 2]
n_samples = 10000
results = {}

for mu in mu_values:
    # Generate samples
    samples = np.random.normal(mu, 1, n_samples)
    
    # Calculate the mean, variance, and E[X²]
    mean_x = np.mean(samples)
    var_x = np.var(samples, ddof=0)  # Population variance
    mean_x_squared = np.mean(samples**2)
    theoretical_mean_x_squared = 1 + mu**2
    
    results[mu] = {
        'mean_x': mean_x,
        'var_x': var_x,
        'mean_x_squared': mean_x_squared,
        'theoretical_mean_x_squared': theoretical_mean_x_squared
    }
    
    print(f"For μ = {mu}:")
    print(f"- Empirical mean: {mean_x:.4f} (Theoretical: {mu})")
    print(f"- Empirical variance: {var_x:.4f} (Theoretical: 1)")
    print(f"- Empirical E[X²]: {mean_x_squared:.4f} (Theoretical: {theoretical_mean_x_squared})")
    print()

# Create a visualization for E[X²] across different means
mu_range = np.linspace(-3, 3, 100)
expected_x_squared = 1 + mu_range**2

plt.figure(figsize=(10, 6))
# Plot the theoretical curve
plt.plot(mu_range, expected_x_squared, 'r-', linewidth=2, label='Theoretical: E[X²] = 1 + μ²')

# Plot the simulation results
for mu in mu_values:
    plt.scatter(mu, results[mu]['mean_x_squared'], color='blue', s=100, 
                label=f'Simulation: μ = {mu}' if mu == mu_values[0] else "")

plt.title('E[X²] as a Function of μ for X ~ N(μ, 1)', fontsize=14)
plt.xlabel('μ (Mean)', fontsize=12)
plt.ylabel('E[X²]', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "expected_x_squared.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Visual algebraic verification
print_step_header(6, "Algebraic Verification")

print("We can verify this result by breaking down Xᵢ²:")
print("- E[Xᵢ²] = E[(Xᵢ - μ + μ)²]")
print("- = E[(Xᵢ - μ)² + 2μ(Xᵢ - μ) + μ²]")
print("- = E[(Xᵢ - μ)²] + 2μE[Xᵢ - μ] + μ²")
print("- = Var(Xᵢ) + 0 + μ²")
print("- = 1 + μ²")
print()

# Create a figure illustrating the decomposition of E[X²]
plt.figure(figsize=(12, 7))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# First subplot: show PDF and the squared deviation
ax1 = plt.subplot(gs[0])
mu = 1.5
x = np.linspace(-3 + mu, 3 + mu, 1000)
pdf = norm.pdf(x, loc=mu, scale=1)

ax1.plot(x, pdf, 'b-', label='PDF of N(μ, 1)')
ax1.fill_between(x, 0, pdf, alpha=0.2, color='blue')

# Emphasize the variance component
ax1.annotate('Var(X) = 1', xy=(mu, 0.05), xytext=(mu - 2, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10, ha='center')

# Emphasize the mean component
ax1.annotate('μ = E[X]', xy=(mu, norm.pdf(mu, mu, 1)), xytext=(mu, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10, ha='center')

ax1.set_title('Normal Distribution N(μ, 1)', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('Probability Density', fontsize=10)
ax1.grid(True)
ax1.legend()

# Second subplot: Show the relationship between μ and E[X²]
ax2 = plt.subplot(gs[1])
mu_range = np.linspace(-3, 3, 100)
ax2.plot(mu_range, 1 + mu_range**2, 'r-', linewidth=2, label='E[X²] = 1 + μ²')
ax2.plot(mu_range, mu_range**2, 'g--', linewidth=2, label='μ²')
ax2.axhline(y=1, color='b', linestyle='--', linewidth=2, label='Var(X) = 1')

# Add labels and annotations
ax2.annotate('E[X²] = Var(X) + μ²', xy=(2, 1 + 2**2), xytext=(0, 7),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10, ha='center')

ax2.set_title('Decomposition of E[X²]', fontsize=12)
ax2.set_xlabel('μ (Mean)', fontsize=10)
ax2.set_ylabel('Value', fontsize=10)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "algebraic_verification.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion and answer
print_step_header(7, "Conclusion and Answer")

print("Options:")
print("A) μ² + 1")
print("B) μ + 1")
print("C) μ²")
print("D) μ - 1")
print()
print("Correct Answer: A) μ² + 1")
print("Explanation: For a normal random variable X ~ N(μ, 1), the expected value of X² is μ² + 1")
print()
print("This result illustrates an important property of the second moment of a distribution:")
print("E[X²] = Var(X) + (E[X])² = 1 + μ²")

# Create summary figure with all options visualized
plt.figure(figsize=(10, 6))

mu_range = np.linspace(-3, 3, 100)
plt.plot(mu_range, 1 + mu_range**2, 'r-', linewidth=3, label='A) μ² + 1 (Correct)')
plt.plot(mu_range, mu_range + 1, 'g--', linewidth=2, label='B) μ + 1')
plt.plot(mu_range, mu_range**2, 'b--', linewidth=2, label='C) μ²')
plt.plot(mu_range, mu_range - 1, 'y--', linewidth=2, label='D) μ - 1')

# Highlight the simulation results
for mu in mu_values:
    plt.scatter(mu, results[mu]['mean_x_squared'], color='purple', s=100, 
                label='Simulation Results' if mu == mu_values[0] else "")

plt.title('Comparing the Different Options', fontsize=14)
plt.xlabel('μ (Mean)', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "options_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nQuestion 19 Solution Summary:")
print("1. For X ~ N(μ, 1), we want to find E[X²]")
print("2. Using the property E[X²] = Var(X) + (E[X])²")
print("3. For our case: E[X²] = 1 + μ²")
print("4. Therefore, the correct answer is A) μ² + 1") 