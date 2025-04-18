import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Six graphs related to the joint PDF of random variables X and Y on the range [0, 4]")
print("- The graphs represent various marginal and conditional distributions and expectations")
print("\nTask:")
print("1. Find the maximum likelihood (ML) estimate of Y given that X=1")
print("2. Find the maximum a posteriori (MAP) estimate of Y given that X=1")
print("3. Find the minimum mean-squared error (MMSE) estimate of Y given that X=1")
print("4. Explain the relationship between these three estimates")

# Step 2: Define the key functions based on the given graphs
print_step_header(2, "Defining Functions Based on Graphs")

# Define functions from the graphs
def f_Y_given_X_1(y):
    """PDF of Y given X=1 (likelihood function)"""
    # This is directly from graph4
    return 0.15 * np.exp(-(y-1)**2/0.15) + 0.03 * np.exp(-(y-3)**2/0.5)

def f_Y(y):
    """Marginal PDF of Y (prior distribution)"""
    # This is directly from graph2
    return np.exp(-0.75 * y)

def E_Y_given_X(x):
    """Conditional expectation of Y given X=x"""
    # This is directly from graph3
    return 3.5 - 0.5 * x

print("From the graphs, we can extract the following functions:")
print("1. f_Y|X(y|X=1) - The conditional distribution of Y given X=1 (likelihood)")
print("2. f_Y(y) - The marginal distribution of Y (prior)")
print("3. E[Y|X=x] - The conditional expectation of Y given X (MMSE estimator)")
print("\nWe'll use these to find our three estimates.")

# Step 3: Calculate ML Estimate
print_step_header(3, "Calculating Maximum Likelihood (ML) Estimate")

# Create range of y values
y_range = np.linspace(0, 4, 1000)

# Calculate likelihood values
likelihood_values = f_Y_given_X_1(y_range)

# Find ML estimate (maximizes likelihood)
ml_index = np.argmax(likelihood_values)
ml_estimate = y_range[ml_index]

print(f"The likelihood function f_Y|X(y|X=1) represents the conditional probability density of Y given X=1")
print(f"The ML estimate is the value of Y that maximizes this function")
print(f"ML Estimate = {ml_estimate:.4f}")

# Visualize the ML estimate
plt.figure(figsize=(10, 6))
plt.plot(y_range, likelihood_values, 'b-', linewidth=2, label='Likelihood f(y|X=1)')
plt.axvline(x=ml_estimate, color='red', linestyle='--', 
            label=f'ML Estimate: y = {ml_estimate:.2f}', linewidth=2)
plt.xlabel('y', fontsize=12)
plt.ylabel('Likelihood', fontsize=12)
plt.title('Maximum Likelihood Estimation for Y given X=1', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "ml_estimate.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"ML estimate visualization saved to: {file_path}")

# Step 4: Calculate MAP Estimate
print_step_header(4, "Calculating Maximum A Posteriori (MAP) Estimate")

# Calculate posterior values (proportional to likelihood * prior)
prior_values = f_Y(y_range)
posterior_values = likelihood_values * prior_values

# Normalize for better visualization
posterior_values = posterior_values / np.max(posterior_values)
likelihood_values_norm = likelihood_values / np.max(likelihood_values)
prior_values_norm = prior_values / np.max(prior_values)

# Find MAP estimate (maximizes posterior)
map_index = np.argmax(posterior_values)
map_estimate = y_range[map_index]

print(f"The posterior distribution is proportional to: likelihood × prior")
print(f"Posterior ∝ f_Y|X(y|X=1) × f_Y(y)")
print(f"The MAP estimate is the value of Y that maximizes this posterior")
print(f"MAP Estimate = {map_estimate:.4f}")

# Visualize the MAP estimate along with prior and likelihood
plt.figure(figsize=(10, 6))
plt.plot(y_range, likelihood_values_norm, 'g--', linewidth=2, label='Likelihood (normalized)')
plt.plot(y_range, prior_values_norm, 'r-.', linewidth=2, label='Prior (normalized)')
plt.plot(y_range, posterior_values, 'b-', linewidth=2, label='Posterior (normalized)')
plt.axvline(x=ml_estimate, color='green', linestyle='--', 
            label=f'ML Estimate: y = {ml_estimate:.2f}', linewidth=2)
plt.axvline(x=map_estimate, color='blue', linestyle='--', 
            label=f'MAP Estimate: y = {map_estimate:.2f}', linewidth=2)
plt.xlabel('y', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('MAP Estimation: Likelihood, Prior, and Posterior for Y given X=1', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "map_estimate.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"MAP estimate visualization saved to: {file_path}")

# Step 5: Calculate MMSE Estimate
print_step_header(5, "Calculating Minimum Mean Squared Error (MMSE) Estimate")

# The MMSE estimate is the conditional expectation
mmse_estimate = E_Y_given_X(1)

print(f"The MMSE estimate is given by the conditional expectation E[Y|X=1]")
print(f"From the graph E[Y|X=x] = 3.5 - 0.5x, we substitute x = 1")
print(f"MMSE Estimate = 3.5 - 0.5(1) = {mmse_estimate:.4f}")

# Visualize the MMSE estimate using the function
x_range = np.linspace(0, 4, 100)
expected_y_values = E_Y_given_X(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, expected_y_values, 'b-', linewidth=2, label='E[Y|X=x]')
plt.scatter([1], [mmse_estimate], color='red', s=100, 
            label=f'MMSE Estimate at X=1: y = {mmse_estimate:.2f}')
plt.xlabel('x', fontsize=12)
plt.ylabel('E[Y|X=x]', fontsize=12)
plt.title('MMSE Estimate: Conditional Expectation E[Y|X=x]', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "mmse_estimate.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"MMSE estimate visualization saved to: {file_path}")

# Step 6: Compare All Three Estimates
print_step_header(6, "Comparing All Estimates")

# Compare the estimates
print(f"ML Estimate:   {ml_estimate:.4f}")
print(f"MAP Estimate:  {map_estimate:.4f}")
print(f"MMSE Estimate: {mmse_estimate:.4f}")

# Create a visualization comparing all three estimates
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(y_range, likelihood_values_norm, 'g--', linewidth=2, label='Likelihood (normalized)')
plt.plot(y_range, prior_values_norm, 'r-.', linewidth=2, label='Prior (normalized)')
plt.plot(y_range, posterior_values, 'b-', linewidth=2, label='Posterior (normalized)')
plt.axvline(x=ml_estimate, color='green', linestyle='-', 
            label=f'ML Estimate: y = {ml_estimate:.2f}', linewidth=2)
plt.axvline(x=map_estimate, color='blue', linestyle='-', 
            label=f'MAP Estimate: y = {map_estimate:.2f}', linewidth=2)
plt.axvline(x=mmse_estimate, color='purple', linestyle='-', 
            label=f'MMSE Estimate: y = {mmse_estimate:.2f}', linewidth=2)
plt.xlabel('y', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Comparison of ML, MAP, and MMSE Estimates for Y given X=1', fontsize=14)
plt.legend()
plt.grid(True)

# Add an explanatory diagram in the lower subplot
plt.subplot(212)
plt.text(0.5, 0.9, 'Relationships Between Different Estimators:', 
         horizontalalignment='center', fontsize=14, fontweight='bold')
plt.text(0.5, 0.7, f'ML Estimate ({ml_estimate:.2f}): Maximizes the Likelihood function f(y|X=1)', 
         horizontalalignment='center', fontsize=12)
plt.text(0.5, 0.5, f'MAP Estimate ({map_estimate:.2f}): Maximizes the Posterior ∝ Likelihood × Prior', 
         horizontalalignment='center', fontsize=12)
plt.text(0.5, 0.3, f'MMSE Estimate ({mmse_estimate:.2f}): Expected value of Y given X=1 (E[Y|X=1])', 
         horizontalalignment='center', fontsize=12)
plt.text(0.5, 0.1, 'Differences: ML ignores prior; MAP accounts for prior; MMSE minimizes expected squared error',
         horizontalalignment='center', fontsize=12)
plt.axis('off')
plt.tight_layout()

file_path = os.path.join(save_dir, "all_estimates_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Comparison visualization saved to: {file_path}")

# Step 7: Analysis of Relationships between Estimators
print_step_header(7, "Analysis of Relationships Between Estimators")

print("ML Estimate vs. MAP Estimate:")
print("- The ML estimate only considers the likelihood function f(y|X=1)")
print("- The MAP estimate incorporates the prior distribution f(y)")
print("- In this case, the prior distribution decreases with y, which pulls the MAP estimate toward lower values")
print(f"- This is why the MAP estimate ({map_estimate:.2f}) is lower than the ML estimate ({ml_estimate:.2f})")
print("")
print("MAP Estimate vs. MMSE Estimate:")
print("- The MAP estimate finds the mode of the posterior distribution")
print("- The MMSE estimate is the mean of the posterior distribution (expected value)")
print("- For asymmetric distributions, the mode and mean differ")
print(f"- Here, the MMSE estimate ({mmse_estimate:.2f}) is higher than the MAP estimate ({map_estimate:.2f})")
print(f"  suggesting the posterior distribution is skewed toward higher values")
print("")
print("General Relationships:")
print("- When the prior is uniform, the MAP estimate equals the ML estimate")
print("- When the posterior is symmetric, the MAP estimate equals the MMSE estimate")
print("- In this case, neither of these conditions holds, so all three estimates are different")
print("")
print("Practical Significance:")
print("- ML: Traditional frequentist approach, only uses data likelihood")
print("- MAP: Bayesian approach, balances likelihood with prior knowledge")
print("- MMSE: Optimal for minimizing expected squared error loss")
print("- The choice among these depends on the loss function and application context") 