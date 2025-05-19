import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_31")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def double_exponential_pdf(x, theta):
    """Calculate the PDF of the double exponential distribution."""
    return 0.5 * np.exp(-np.abs(x - theta))

def double_exponential_cdf(x, theta):
    """Calculate the CDF of the double exponential distribution."""
    if isinstance(x, np.ndarray):
        result = np.zeros_like(x, dtype=float)
        below_theta = x < theta
        above_theta = ~below_theta
        
        result[below_theta] = 0.5 * np.exp(x[below_theta] - theta)
        result[above_theta] = 1 - 0.5 * np.exp(-(x[above_theta] - theta))
        return result
    else:
        if x < theta:
            return 0.5 * np.exp(x - theta)
        else:
            return 1 - 0.5 * np.exp(-(x - theta))

def log_likelihood(theta, data):
    """Calculate the log-likelihood for a given theta and data."""
    return np.sum(np.log(double_exponential_pdf(data, theta)))

# Step 1: Theoretical derivation of MLE for double exponential
print_step_header(1, "Theoretical Derivation of MLE")

print("For the double exponential distribution f(x|θ) = (1/2)e^(-|x-θ|), -∞ < x < ∞")
print("We want to find the MLE of θ for an i.i.d. sample of size n = 2m + 1.")
print("\nThe likelihood function is:")
print("L(θ|x₁,x₂,...,xₙ) = ∏ (1/2)e^(-|xᵢ-θ|)")
print("\nThe log-likelihood function is:")
print("log L(θ|x₁,x₂,...,xₙ) = n*log(1/2) - ∑|xᵢ-θ|")
print("\nTo maximize the log-likelihood, we need to minimize the sum ∑|xᵢ-θ|.")
print("The minimum of ∑|xᵢ-θ| occurs at the median of the sample when n is odd.")
print("This is because the median balances the number of points above and below θ.")

# Step 2: Visual demonstration with synthetic data
print_step_header(2, "Visual Demonstration with Synthetic Data")

# Generate a small synthetic dataset
np.random.seed(42)
synthetic_data = np.array([-3, -1, 0, 2, 5])
synthetic_data.sort()  # Sort the data
median_value = np.median(synthetic_data)

print(f"Synthetic data: {synthetic_data}")
print(f"Median: {median_value}")

# Calculate log-likelihood for a range of θ values
theta_range = np.linspace(-5, 7, 500)
log_likelihood_values = np.array([log_likelihood(theta, synthetic_data) for theta in theta_range])

# Plot the log-likelihood function
plt.figure(figsize=(10, 6))
plt.plot(theta_range, log_likelihood_values, 'b-', linewidth=2)
plt.axvline(x=median_value, color='r', linestyle='--', 
            label=f'Median (θ_MLE = {median_value})')

# Mark individual data points
for x in synthetic_data:
    plt.axvline(x=x, color='green', alpha=0.3)

plt.grid(True)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Log-Likelihood', fontsize=12)
plt.title('Log-Likelihood Function for Double Exponential Distribution', fontsize=14)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "log_likelihood_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Visual explanation of why median minimizes sum of absolute deviations
print_step_header(3, "Why Median Minimizes Sum of Absolute Deviations")

# Create a visualization to demonstrate why the median minimizes sum of absolute deviations
theta_values = np.linspace(-4, 6, 100)
sum_abs_deviations = np.array([np.sum(np.abs(synthetic_data - theta)) for theta in theta_values])

plt.figure(figsize=(10, 6))
plt.plot(theta_values, sum_abs_deviations, 'b-', linewidth=2)
plt.axvline(x=median_value, color='r', linestyle='--', 
            label=f'Median (θ = {median_value})')

# Mark individual data points
for x in synthetic_data:
    plt.axvline(x=x, color='green', alpha=0.3, label='Data point' if x == synthetic_data[0] else "")

plt.grid(True)
plt.xlabel('θ', fontsize=12)
plt.ylabel('Sum of Absolute Deviations', fontsize=12)
plt.title('Sum of Absolute Deviations vs. θ', fontsize=14)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sum_abs_deviations.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Demonstration with derivatives
print_step_header(4, "Illustration of Non-Differentiability")

# Illustration of the non-differentiability of the absolute value function
x_values = np.linspace(-5, 5, 1000)
abs_values = np.abs(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, abs_values, 'b-', linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', label='Non-differentiable point')
plt.grid(True)
plt.xlabel('x', fontsize=12)
plt.ylabel('|x|', fontsize=12)
plt.title('Absolute Value Function and its Non-Differentiability at x=0', fontsize=14)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "abs_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Real-world data analysis
print_step_header(5, "Real-world Data Analysis")

# Given data from the problem
data = np.array([-1.2, 0.5, 2.1, -0.7, 1.5, 0.3, -0.2, 1.8, 0.1, -1.0, 0.9])
data_sorted = np.sort(data)
median_data = np.median(data)

print(f"Given data: {data}")
print(f"Sorted data: {data_sorted}")
print(f"Median (MLE estimate θ̂): {median_data}")

# Calculate the probability density at x = 0 using the MLE estimate
pdf_at_zero = double_exponential_pdf(0, median_data)
print(f"Probability density at x = 0 using θ̂ = {median_data}: {pdf_at_zero:.6f}")

# Calculate the probability that a future observation will be greater than 2.0
prob_greater_than_2 = 1 - double_exponential_cdf(2.0, median_data)
print(f"Probability that a future observation will be greater than 2.0: {prob_greater_than_2:.6f}")

# Step 6: Visualization of the fitted distribution
print_step_header(6, "Visualization of Fitted Distribution")

# Plot the histogram of the data with the fitted double exponential distribution
x_range = np.linspace(-4, 4, 1000)
fitted_pdf = double_exponential_pdf(x_range, median_data)

plt.figure(figsize=(10, 6))

# Plot the fitted PDF
plt.plot(x_range, fitted_pdf, 'r-', linewidth=2, 
         label=f'Fitted Double Exponential (θ = {median_data:.2f})')

# Add vertical lines for the data points
for x in data:
    plt.axvline(x=x, color='blue', alpha=0.2)

# Add a specific marker for x = 0 and x = 2.0
plt.axvline(x=0, color='green', linestyle='--', 
            label=f'x = 0, f(0|θ) = {pdf_at_zero:.4f}')
plt.axvline(x=2.0, color='purple', linestyle='--', 
            label=f'x = 2.0, P(X > 2.0) = {prob_greater_than_2:.4f}')

# Add median/MLE marker
plt.axvline(x=median_data, color='orange', linestyle='-', 
            label=f'θ_MLE = {median_data:.2f}')

# Additional plot settings
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Fitted Double Exponential Distribution', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "fitted_distribution.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Visual comparison with normal distribution
print_step_header(7, "Comparison with Normal Distribution")

# Compare double exponential with normal distribution
x_range = np.linspace(-4, 4, 1000)
double_exp = double_exponential_pdf(x_range, 0)

# Calculate standard deviation for comparable normal distribution
comparable_std = np.sqrt(2)  # For standard Laplace, variance is 2
normal_pdf = stats.norm.pdf(x_range, 0, comparable_std)

plt.figure(figsize=(10, 6))
plt.plot(x_range, double_exp, 'r-', linewidth=2, label='Double Exponential')
plt.plot(x_range, normal_pdf, 'b--', linewidth=2, label='Normal (equal variance)')
plt.grid(True)
plt.xlabel('x', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Double Exponential vs. Normal Distribution', fontsize=14)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "distribution_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: CDF visualization
print_step_header(8, "CDF Visualization")

# Calculate the CDF values
cdf_values = np.array([double_exponential_cdf(x, median_data) for x in x_range])

plt.figure(figsize=(10, 6))
plt.plot(x_range, cdf_values, 'b-', linewidth=2, 
         label=f'CDF with θ = {median_data:.2f}')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='CDF = 0.5')
plt.axvline(x=median_data, color='orange', linestyle='-', 
            label=f'θ_MLE = {median_data:.2f}')
plt.axvline(x=2.0, color='purple', linestyle='--')

# Shade the area for P(X > 2.0)
idx = np.where(x_range >= 2.0)[0]
plt.fill_between(x_range[idx], cdf_values[idx], 1, alpha=0.3, color='purple', 
                 label=f'P(X > 2.0) = {prob_greater_than_2:.4f}')

plt.grid(True)
plt.xlabel('x', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('CDF of Double Exponential Distribution', fontsize=14)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "cdf_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 9: Conclusion
print_step_header(9, "Conclusion")

print("Key Findings:")
print(f"1. The MLE for θ in the double exponential distribution is the median of the sample.")
print(f"2. For the given dataset, the MLE estimate is θ̂ = {median_data:.4f}")
print(f"3. The probability density at x = 0 is {pdf_at_zero:.6f}")
print(f"4. The probability that a future observation will be greater than 2.0 is {prob_greater_than_2:.6f}")
print("\nThis analysis demonstrates that the median is the maximum likelihood estimator for the")
print("location parameter of the double exponential distribution when the sample size is odd.") 