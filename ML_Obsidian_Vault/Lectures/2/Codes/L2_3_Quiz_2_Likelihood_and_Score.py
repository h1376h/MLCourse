import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Problem Setup")

print("Given:")
print("- We have a random sample X₁, X₂, ..., X₂₀ from a distribution with PDF:")
print("  f(x|θ) = θx^(θ-1), 0 < x < 1, θ > 0")
print("- We are told that the geometric mean of the observed data is 0.8")
print("- We need to derive the likelihood function, log-likelihood, and score function")
print("- We need to find the MLE for θ")

# Visualize the PDF for different values of theta
thetas = [0.5, 1, 2, 5]
x = np.linspace(0.01, 1, 1000)  # Avoid x=0 which causes issues for θ<1

plt.figure(figsize=(10, 6))
for theta in thetas:
    pdf = theta * x**(theta-1)
    plt.plot(x, pdf, linewidth=2, label=f'θ = {theta}')
    
plt.title('PDF: f(x|θ) = θx^(θ-1) for Different Values of θ', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density f(x|θ)', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "pdf_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 2: Likelihood Function
print_step_header(2, "Likelihood Function")

print("For a random sample X₁, X₂, ..., X_n, the likelihood function is:")
print("L(θ) = ∏ᵢ₌₁ⁿ f(xᵢ|θ) = ∏ᵢ₌₁ⁿ θxᵢ^(θ-1)")
print("     = θⁿ · ∏ᵢ₌₁ⁿ xᵢ^(θ-1)")
print("     = θⁿ · (∏ᵢ₌₁ⁿ xᵢ)^(θ-1)")

# Define a function to compute the likelihood
def likelihood(theta, x_values):
    """Compute the likelihood function for the given PDF."""
    n = len(x_values)
    product_term = np.prod(x_values**(theta-1))
    return theta**n * product_term

# For visualization, since we don't have actual data, let's assume the geometric mean is 0.8
# We can simulate data with this geometric mean
n = 20  # Sample size
geo_mean = 0.8
# For a set with geometric mean 0.8, we can use this value directly when computing the likelihood
# This is because the product term simplifies to (geometric_mean^n)^(θ-1)

# Creating a range of theta values to plot the likelihood function
theta_values = np.linspace(0.1, 10, 1000)
likelihood_values = []

for theta in theta_values:
    # For a data set with geometric mean geo_mean:
    # L(θ) = θⁿ · (geometric_mean^n)^(θ-1)
    #      = θⁿ · (geometric_mean^(θ-1))^n
    like_value = theta**n * (geo_mean**(theta-1))**n
    likelihood_values.append(like_value)

# Plot the likelihood function
plt.figure(figsize=(10, 6))
plt.plot(theta_values, likelihood_values, 'b-', linewidth=2)
plt.title('Likelihood Function L(θ) with Geometric Mean = 0.8', fontsize=14)
plt.xlabel('θ', fontsize=12)
plt.ylabel('L(θ)', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Log-Likelihood Function
print_step_header(3, "Log-Likelihood Function")

print("The log-likelihood function is:")
print("ℓ(θ) = log(L(θ)) = log(θⁿ · (∏ᵢ₌₁ⁿ xᵢ)^(θ-1))")
print("     = n·log(θ) + (θ-1)·log(∏ᵢ₌₁ⁿ xᵢ)")
print("     = n·log(θ) + (θ-1)·∑ᵢ₌₁ⁿ log(xᵢ)")
print("     = n·log(θ) + (θ-1)·n·log(geometric_mean)")

# Define a function to compute the log-likelihood
def log_likelihood(theta, n, geo_mean):
    """Compute the log-likelihood function for the given PDF."""
    return n * np.log(theta) + (theta-1) * n * np.log(geo_mean)

# Calculate log-likelihood values
log_likelihood_values = [log_likelihood(theta, n, geo_mean) for theta in theta_values]

# Plot the log-likelihood function
plt.figure(figsize=(10, 6))
plt.plot(theta_values, log_likelihood_values, 'g-', linewidth=2)
plt.title('Log-Likelihood Function ℓ(θ) with Geometric Mean = 0.8', fontsize=14)
plt.xlabel('θ', fontsize=12)
plt.ylabel('ℓ(θ)', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "log_likelihood_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Score Function
print_step_header(4, "Score Function")

print("The score function is the derivative of the log-likelihood function with respect to θ:")
print("S(θ) = dℓ(θ)/dθ = d/dθ[n·log(θ) + (θ-1)·n·log(geometric_mean)]")
print("     = n/θ + n·log(geometric_mean)")
print("     = n·[1/θ + log(geometric_mean)]")

# Define a function to compute the score function
def score_function(theta, n, geo_mean):
    """Compute the score function (derivative of log-likelihood)."""
    return n * (1/theta + np.log(geo_mean))

# Calculate score function values
score_values = [score_function(theta, n, geo_mean) for theta in theta_values]

# Plot the score function
plt.figure(figsize=(10, 6))
plt.plot(theta_values, score_values, 'r-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--')
plt.title('Score Function S(θ) with Geometric Mean = 0.8', fontsize=14)
plt.xlabel('θ', fontsize=12)
plt.ylabel('S(θ)', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "score_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Maximum Likelihood Estimation
print_step_header(5, "Maximum Likelihood Estimation")

print("To find the MLE, we set the score function equal to zero and solve for θ:")
print("S(θ) = n·[1/θ + log(geometric_mean)] = 0")
print("1/θ + log(geometric_mean) = 0")
print("1/θ = -log(geometric_mean)")
print("θ = -1/log(geometric_mean)")

# Calculate the MLE analytically
geo_mean_val = 0.8
mle_theta = -1/np.log(geo_mean_val)
print(f"With geometric mean = {geo_mean_val}:")
print(f"MLE for θ: θ̂ = -1/log({geo_mean_val}) = {mle_theta:.4f}")

# Verify using numerical optimization
def neg_log_likelihood(theta, n, geo_mean):
    """Negative log-likelihood function for minimization."""
    return -log_likelihood(theta, n, geo_mean)

# Find MLE numerically
result = minimize(neg_log_likelihood, x0=1.0, args=(n, geo_mean_val))
print(f"MLE using numerical optimization: {result.x[0]:.4f}")
print(f"This confirms our analytical result: {mle_theta:.4f}")

# Create a combined plot with log-likelihood and score function to highlight the MLE
plt.figure(figsize=(12, 8))

# Set up a 2x1 grid
gs = GridSpec(2, 1, height_ratios=[2, 1])

# Plot log-likelihood
ax1 = plt.subplot(gs[0])
ax1.plot(theta_values, log_likelihood_values, 'g-', linewidth=2)
ax1.axvline(x=mle_theta, color='r', linestyle='--', 
           label=f'MLE θ̂ = {mle_theta:.4f}')
ax1.set_title('Log-Likelihood Function ℓ(θ)', fontsize=14)
ax1.set_ylabel('ℓ(θ)', fontsize=12)
ax1.grid(True)
ax1.legend()

# Plot score function
ax2 = plt.subplot(gs[1])
ax2.plot(theta_values, score_values, 'r-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--')
ax2.axvline(x=mle_theta, color='r', linestyle='--', 
           label=f'MLE θ̂ = {mle_theta:.4f}')
ax2.set_title('Score Function S(θ)', fontsize=14)
ax2.set_xlabel('θ', fontsize=12)
ax2.set_ylabel('S(θ)', fontsize=12)
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "mle_estimation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Visual Summary and Interpretation
print_step_header(6, "Visual Summary and Interpretation")

# Create a comprehensive figure to illustrate the findings
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2)

# Plot 1: PDF with the MLE value
ax1 = plt.subplot(gs[0, 0])
for theta in [0.5, mle_theta, 5]:
    pdf = theta * x**(theta-1)
    ax1.plot(x, pdf, linewidth=2, label=f'θ = {theta:.2f}')
ax1.set_title('PDF with MLE θ', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('f(x|θ)', fontsize=10)
ax1.axvline(x=geo_mean_val, color='k', linestyle='--', label=f'Geo. Mean = {geo_mean_val}')
ax1.grid(True)
ax1.legend()

# Plot 2: Likelihood Function
ax2 = plt.subplot(gs[0, 1])
ax2.plot(theta_values, likelihood_values, 'b-', linewidth=2)
ax2.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ̂ = {mle_theta:.4f}')
ax2.set_title('Likelihood Function', fontsize=12)
ax2.set_xlabel('θ', fontsize=10)
ax2.set_ylabel('L(θ)', fontsize=10)
ax2.grid(True)
ax2.legend()

# Plot 3: Log-Likelihood Function
ax3 = plt.subplot(gs[1, 0])
ax3.plot(theta_values, log_likelihood_values, 'g-', linewidth=2)
ax3.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ̂ = {mle_theta:.4f}')
ax3.set_title('Log-Likelihood Function', fontsize=12)
ax3.set_xlabel('θ', fontsize=10)
ax3.set_ylabel('ℓ(θ)', fontsize=10)
ax3.grid(True)
ax3.legend()

# Plot 4: Score Function
ax4 = plt.subplot(gs[1, 1])
ax4.plot(theta_values, score_values, 'r-', linewidth=2)
ax4.axhline(y=0, color='k', linestyle='--')
ax4.axvline(x=mle_theta, color='r', linestyle='--', label=f'MLE θ̂ = {mle_theta:.4f}')
ax4.set_title('Score Function', fontsize=12)
ax4.set_xlabel('θ', fontsize=10)
ax4.set_ylabel('S(θ)', fontsize=10)
ax4.grid(True)
ax4.legend()

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "summary_results.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Final summary
print("\nSummary of results:")
print(f"1. The maximum likelihood estimate for θ is: θ̂ = {mle_theta:.4f}")
print(f"2. This estimation is based on a data set with geometric mean {geo_mean_val}")
print(f"3. The score function equals zero at θ = {mle_theta:.4f}, confirming it's the MLE")
print(f"4. As log({geo_mean_val}) = {np.log(geo_mean_val):.4f}, our formula θ̂ = -1/log({geo_mean_val}) gives {mle_theta:.4f}")
print("\nInterpretation:")
print(f"- For the distribution f(x|θ) = θx^(θ-1) with 0 < x < 1 and θ > 0")
print(f"- When the geometric mean of the data is {geo_mean_val}")
print(f"- The most likely value of the parameter θ is {mle_theta:.4f}")
print(f"- This value maximizes the likelihood function and makes the score function equal to zero") 