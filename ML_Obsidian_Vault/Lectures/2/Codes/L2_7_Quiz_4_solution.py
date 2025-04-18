import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Linear regression model y = βx + ε where ε ~ N(0, σ²) with known σ² = 1")
print("- Data points (x, y):")
print("  {(1, 2.1), (2, 3.8), (3, 5.2), (4, 6.9), (5, 8.3)}")
print("- Prior for β: β ~ N(1, 0.5)")
print("\nTask:")
print("1. Derive the posterior distribution for β")
print("2. Calculate the MAP estimate for β")
print("3. Calculate the MLE for β")
print("4. Derive the posterior predictive distribution for y_new given x_new = 6")

# Step 2: Data Preparation and Visualization
print_step_header(2, "Data Preparation and Visualization")

# Define the data
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2.1, 3.8, 5.2, 6.9, 8.3])
n = len(x_data)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', s=50, label='Observed Data')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Linear Regression Data', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "data_visualization.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Calculate sufficient statistics for the linear regression
sum_x = np.sum(x_data)
sum_y = np.sum(y_data)
sum_xy = np.sum(x_data * y_data)
sum_x2 = np.sum(x_data**2)

print("\nSufficient statistics for the data:")
print(f"Sample size: n = {n}")
print(f"Sum of x: Σx = {sum_x}")
print(f"Sum of y: Σy = {sum_y}")
print(f"Sum of xy: Σxy = {sum_xy}")
print(f"Sum of x²: Σx² = {sum_x2}")

# Step 3: Deriving the Posterior Distribution
print_step_header(3, "Deriving the Posterior Distribution")

# Define the prior parameters
prior_mean = 1.0
prior_var = 0.5

# Define the likelihood variance
likelihood_var = 1.0

# Calculate posterior parameters for Bayesian linear regression
# Formula derivation follows from Bayesian linear regression with known variance
posterior_var = 1 / (1/prior_var + sum_x2/likelihood_var)
posterior_mean = posterior_var * (prior_mean/prior_var + sum_xy/likelihood_var)

print("For a Bayesian linear regression with a normal prior for β and known variance:")
print("The posterior distribution for β is also normal.")
print("\nPosterior variance calculation:")
print(f"1/σ²_posterior = 1/σ²_prior + Σx²/σ²_likelihood")
print(f"1/σ²_posterior = 1/{prior_var} + {sum_x2}/{likelihood_var}")
print(f"1/σ²_posterior = {1/prior_var} + {sum_x2/likelihood_var}")
print(f"1/σ²_posterior = {1/prior_var + sum_x2/likelihood_var}")
print(f"σ²_posterior = {posterior_var:.6f}")
print("\nPosterior mean calculation:")
print(f"β_posterior = σ²_posterior * (β_prior/σ²_prior + Σxy/σ²_likelihood)")
print(f"β_posterior = {posterior_var:.6f} * ({prior_mean}/{prior_var} + {sum_xy}/{likelihood_var})")
print(f"β_posterior = {posterior_var:.6f} * ({prior_mean/prior_var} + {sum_xy/likelihood_var})")
print(f"β_posterior = {posterior_var:.6f} * {prior_mean/prior_var + sum_xy/likelihood_var}")
print(f"β_posterior = {posterior_mean:.6f}")
print("\nPosterior distribution: β | Data ~ N({:.6f}, {:.6f})".format(posterior_mean, posterior_var))

# Step 4: Calculate the MAP Estimate
print_step_header(4, "Calculating the MAP Estimate")

# For a normal posterior distribution, the MAP equals the mean
map_estimate = posterior_mean

print("For a normal posterior distribution, the mode (MAP estimate) equals the mean:")
print(f"MAP = β_posterior = {map_estimate:.6f}")

# Step 5: Calculate the MLE Estimate
print_step_header(5, "Calculating the MLE Estimate")

# For linear regression, the MLE is given by the ordinary least squares formula
mle_estimate = sum_xy / sum_x2

print("For linear regression, the MLE is given by:")
print("MLE = Σxy / Σx²")
print(f"MLE = {sum_xy} / {sum_x2}")
print(f"MLE = {mle_estimate:.6f}")

# Step 6: Compare MAP and MLE
print_step_header(6, "Comparing MAP and MLE Estimates")

print(f"MAP estimate: {map_estimate:.6f}")
print(f"MLE estimate: {mle_estimate:.6f}")
print(f"Difference (MAP - MLE): {map_estimate - mle_estimate:.6f}")
print("\nExplanation of the difference:")
print("The MAP estimate incorporates the prior belief (β ~ N(1, 0.5)), while the MLE only considers the data.")
print(f"The prior pulls the MAP estimate toward the prior mean ({prior_mean}).")

# Visualize the comparison of MAP and MLE with the regression lines
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='blue', s=50, label='Observed Data')

# Plot the MAP regression line
x_range = np.linspace(0, 7, 100)
y_map = map_estimate * x_range
plt.plot(x_range, y_map, 'r-', label=f'MAP Regression Line: y = {map_estimate:.4f}x', linewidth=2)

# Plot the MLE regression line
y_mle = mle_estimate * x_range
plt.plot(x_range, y_mle, 'g--', label=f'MLE Regression Line: y = {mle_estimate:.4f}x', linewidth=2)

# Plot prediction range
x_new = 6.0  # For future prediction
plt.axvline(x=x_new, color='purple', linestyle=':', label=f'x_new = {x_new}', linewidth=2)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Comparison of MAP and MLE Regression Lines', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "map_vs_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 7: Derive Posterior Predictive Distribution
print_step_header(7, "Deriving Posterior Predictive Distribution")

# For a new observation at x_new = 6
x_new = 6.0

# The posterior predictive mean is the mean of the posterior times x_new
predictive_mean = posterior_mean * x_new

# The posterior predictive variance includes both parameter uncertainty and data noise
predictive_var = x_new**2 * posterior_var + likelihood_var

print(f"For a new observation at x_new = {x_new}:")
print("\nThe posterior predictive distribution is:")
print("p(y_new | x_new, Data) = ∫ p(y_new | x_new, β) p(β | Data) dβ")
print("For normal likelihood and normal posterior, this integral has a closed form solution.")
print("\nPosterior predictive mean calculation:")
print(f"μ_predictive = β_posterior × x_new")
print(f"μ_predictive = {posterior_mean:.6f} × {x_new}")
print(f"μ_predictive = {predictive_mean:.6f}")
print("\nPosterior predictive variance calculation:")
print(f"σ²_predictive = x_new² × σ²_posterior + σ²_likelihood")
print(f"σ²_predictive = {x_new}² × {posterior_var:.6f} + {likelihood_var}")
print(f"σ²_predictive = {x_new**2} × {posterior_var:.6f} + {likelihood_var}")
print(f"σ²_predictive = {x_new**2 * posterior_var:.6f} + {likelihood_var}")
print(f"σ²_predictive = {predictive_var:.6f}")
print("\nPosterior predictive distribution: y_new | x_new, Data ~ N({:.6f}, {:.6f})".format(predictive_mean, predictive_var))

# Calculate the 95% prediction interval
alpha = 0.05  # For 95% interval
z_score = norm.ppf(1 - alpha/2)  # = 1.96 for 95%
lower_bound = predictive_mean - z_score * np.sqrt(predictive_var)
upper_bound = predictive_mean + z_score * np.sqrt(predictive_var)

print("\n95% prediction interval for y_new at x_new = 6:")
print(f"[{lower_bound:.6f}, {upper_bound:.6f}]")

# Visualize the posterior predictive distribution
plt.figure(figsize=(10, 6))

# Plot the data and regression line
plt.scatter(x_data, y_data, color='blue', s=50, label='Observed Data')
plt.plot(x_range, y_map, 'r-', label=f'MAP Regression Line: y = {map_estimate:.4f}x', linewidth=2)

# Plot the prediction point
plt.scatter([x_new], [predictive_mean], color='purple', s=100, label=f'Prediction at x_new = {x_new}')

# Plot the prediction interval
plt.fill_between([x_new-0.2, x_new+0.2], [lower_bound, lower_bound], [upper_bound, upper_bound], 
                 color='purple', alpha=0.3, label=f'95% Prediction Interval')

# Plot the prediction distribution
y_new_range = np.linspace(lower_bound-2, upper_bound+2, 1000)
predictive_pdf = norm.pdf(y_new_range, predictive_mean, np.sqrt(predictive_var))
predictive_pdf_scaled = 0.4 * predictive_pdf / np.max(predictive_pdf)  # Scale for visualization
plt.plot(x_new + predictive_pdf_scaled, y_new_range, 'k-', label='Predictive Distribution')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Posterior Predictive Distribution for y_new at x_new = 6', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "posterior_predictive.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Visualize the posterior distribution for β
plt.figure(figsize=(10, 6))

# Define a range of possible β values
beta_range = np.linspace(map_estimate - 5*np.sqrt(posterior_var), 
                         map_estimate + 5*np.sqrt(posterior_var), 1000)

# Compute the prior and posterior PDFs
prior_pdf = norm.pdf(beta_range, prior_mean, np.sqrt(prior_var))
posterior_pdf = norm.pdf(beta_range, posterior_mean, np.sqrt(posterior_var))

# Plot the prior and posterior distributions
plt.plot(beta_range, prior_pdf, 'g--', label=f'Prior: N({prior_mean}, {prior_var})', linewidth=2)
plt.plot(beta_range, posterior_pdf, 'r-', label=f'Posterior: N({posterior_mean:.4f}, {posterior_var:.4f})', linewidth=2)

# Mark the MLE and MAP estimates
plt.axvline(x=mle_estimate, color='blue', linestyle=':', 
            label=f'MLE: {mle_estimate:.4f}', linewidth=2)
plt.axvline(x=map_estimate, color='red', linestyle=':', 
            label=f'MAP: {map_estimate:.4f}', linewidth=2)

plt.xlabel('β (Slope Parameter)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Prior and Posterior Distributions for β', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prior_posterior_beta.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 8: Summary
print_step_header(8, "Summary of Results")

print("1. Posterior Distribution for β:")
print(f"   β | Data ~ N({posterior_mean:.6f}, {posterior_var:.6f})")
print()
print(f"2. MAP Estimate for β: {map_estimate:.6f}")
print()
print(f"3. MLE Estimate for β: {mle_estimate:.6f}")
print()
print("4. Posterior Predictive Distribution for y_new at x_new = 6:")
print(f"   y_new | x_new = 6, Data ~ N({predictive_mean:.6f}, {predictive_var:.6f})")
print(f"   95% Prediction Interval: [{lower_bound:.6f}, {upper_bound:.6f}]")
print()
print("Conclusion:")
print("- The Bayesian approach (MAP) incorporates prior information, making it more robust when data is limited.")
print("- The posterior distribution quantifies our uncertainty about the parameter β.")
print("- The posterior predictive distribution accounts for both parameter uncertainty and inherent noise,")
print("  providing a principled way to make predictions with appropriate uncertainty quantification.") 