import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 2: Gauss-Markov Assumptions")
print("====================================")

# Step 1: Explain Homoscedasticity
print("\nStep 1: What is Homoscedasticity?")
print("-------------------------------")
print("Homoscedasticity is a key assumption in linear regression that means")
print("the variance of the errors is constant across all levels of the predictor variables.")
print()
print("In simple terms, it means that the spread or dispersion of the errors")
print("around the regression line is the same throughout the entire range of the data.")
print()
print("Mathematically, for a linear model y = Xβ + ε, homoscedasticity means:")
print("Var(ε_i) = σ² for all i = 1, 2, ..., n")
print("where σ² is a constant value (in this case, σ² = 4).")
print()

# Step 2: Explain the importance of constant variance
print("\nStep 2: Why is Constant Variance Important?")
print("----------------------------------------")
print("Constant variance is important for linear regression for several reasons:")
print()
print("1. It ensures that the Ordinary Least Squares (OLS) estimator is the Best Linear")
print("   Unbiased Estimator (BLUE) according to the Gauss-Markov theorem.")
print()
print("2. It ensures that the standard errors of the coefficient estimates are correct,")
print("   leading to valid hypothesis tests and confidence intervals.")
print()
print("3. It means that the precision of predictions is the same across all values of")
print("   the predictors, making the model reliable throughout the entire range of the data.")
print()
print("4. It simplifies statistical inference because we can use a single parameter (σ²)")
print("   to describe the error distribution rather than having different variances for")
print("   different observations.")
print()

# Step 3: Calculate variance when σ = 2
print("\nStep 3: What is the Variance When σ = 2?")
print("-------------------------------------")
print("The variance of the errors is related to the standard deviation by:")
print("Variance = σ² = Standard Deviation²")
print()
print("Given that σ = 2:")
print("Variance = σ² = 2² = 4")
print()
print("So, the variance of the errors is 4 when σ = 2.")
print("This matches the value given in the problem statement.")
print()

# Step 4: Create visualizations to understand homoscedasticity
print("\nStep 4: Visualizing Homoscedasticity and Heteroscedasticity")
print("------------------------------------------------------")
print("Creating visualizations to illustrate the concept of homoscedasticity...")
print()

np.random.seed(42)  # For reproducibility

# Generate data for visualizations
n = 100  # Number of observations
x = np.linspace(0, 10, n)

# Function to generate homoscedastic errors
def generate_homoscedastic_data(x, beta_0, beta_1, sigma):
    # True regression line
    y_true = beta_0 + beta_1 * x
    
    # Generate errors with constant variance
    epsilon = np.random.normal(0, sigma, len(x))
    
    # Observed response
    y_obs = y_true + epsilon
    
    return y_true, y_obs, epsilon

# Function to generate heteroscedastic errors
def generate_heteroscedastic_data(x, beta_0, beta_1, sigma_func):
    # True regression line
    y_true = beta_0 + beta_1 * x
    
    # Generate errors with non-constant variance
    sigmas = sigma_func(x)
    epsilon = np.array([np.random.normal(0, sigma) for sigma in sigmas])
    
    # Observed response
    y_obs = y_true + epsilon
    
    return y_true, y_obs, epsilon, sigmas

# Parameters for the model
beta_0 = 5
beta_1 = 2
sigma = 2  # Standard deviation (given in the problem)

# Generate homoscedastic data
y_true_homo, y_obs_homo, epsilon_homo = generate_homoscedastic_data(x, beta_0, beta_1, sigma)

# Function for increasing variance
def increasing_sigma(x):
    return 0.5 + 0.3 * x

# Generate heteroscedastic data
y_true_hetero, y_obs_hetero, epsilon_hetero, sigmas_hetero = generate_heteroscedastic_data(
    x, beta_0, beta_1, increasing_sigma)

# Visualization 1: Homoscedastic vs Heteroscedastic Data
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig)

# Homoscedastic data plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(x, y_obs_homo, alpha=0.7, label='Observed Data')
ax1.plot(x, y_true_homo, 'r-', linewidth=2, label='True Regression Line')

# Add error bars for a few points to show constant variance
x_sample = np.linspace(1, 9, 5)
idx_sample = [np.argmin(np.abs(x - xs)) for xs in x_sample]
for idx in idx_sample:
    ax1.plot([x[idx], x[idx]], [y_true_homo[idx], y_obs_homo[idx]], 'k--', alpha=0.5)
    
ax1.set_xlabel('Predictor (x)', fontsize=12)
ax1.set_ylabel('Response (y)', fontsize=12)
ax1.set_title('Homoscedastic Data (Constant Variance)', fontsize=14)
ax1.legend()

# Heteroscedastic data plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x, y_obs_hetero, alpha=0.7, label='Observed Data')
ax2.plot(x, y_true_hetero, 'r-', linewidth=2, label='True Regression Line')

# Add error bars for a few points to show varying variance
for idx in idx_sample:
    ax2.plot([x[idx], x[idx]], [y_true_hetero[idx], y_obs_hetero[idx]], 'k--', alpha=0.5)
    
ax2.set_xlabel('Predictor (x)', fontsize=12)
ax2.set_ylabel('Response (y)', fontsize=12)
ax2.set_title('Heteroscedastic Data (Increasing Variance)', fontsize=14)
ax2.legend()

# Residuals plot for homoscedastic data
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(x, epsilon_homo, alpha=0.7)
ax3.axhline(y=0, color='r', linestyle='-', linewidth=2)
ax3.axhline(y=2*sigma, color='g', linestyle='--', alpha=0.7, label=f'±2σ ({2*sigma})')
ax3.axhline(y=-2*sigma, color='g', linestyle='--', alpha=0.7)
ax3.set_xlabel('Predictor (x)', fontsize=12)
ax3.set_ylabel('Residuals (ε)', fontsize=12)
ax3.set_title('Residuals for Homoscedastic Data', fontsize=14)
ax3.legend()

# Residuals plot for heteroscedastic data
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(x, epsilon_hetero, alpha=0.7)
ax4.axhline(y=0, color='r', linestyle='-', linewidth=2)

# Plot the varying 2σ bounds
ax4.plot(x, 2*sigmas_hetero, 'g--', alpha=0.7, label='±2σ (varying)')
ax4.plot(x, -2*sigmas_hetero, 'g--', alpha=0.7)

ax4.set_xlabel('Predictor (x)', fontsize=12)
ax4.set_ylabel('Residuals (ε)', fontsize=12)
ax4.set_title('Residuals for Heteroscedastic Data', fontsize=14)
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "homo_vs_hetero.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Gauss-Markov Assumptions - Separate visualizations for each assumption
assumptions = [
    'Linearity: The relationship between predictors and the response is linear',
    'Random Sampling: The data is a random sample from the population',
    'No Perfect Multicollinearity: No predictor is a perfect linear combination of others',
    'Exogeneity: The expected value of errors is zero: E(ε|X) = 0',
    'Homoscedasticity: The errors have constant variance: Var(ε|X) = σ²',
    'No Autocorrelation: The errors are uncorrelated: Cov(ε_i, ε_j) = 0 for i ≠ j'
]

# Create separate visualizations for each assumption
# 1. Linearity
np.random.seed(42)
x_lin = np.linspace(0, 10, 100)
y_lin_true = 3 + 2*x_lin
y_lin_obs = y_lin_true + np.random.normal(0, 1.5, size=len(x_lin))
y_nonlin_true = 3 + 2*x_lin - 0.2*x_lin**2
y_nonlin_obs = y_nonlin_true + np.random.normal(0, 1.5, size=len(x_lin))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x_lin, y_lin_obs, alpha=0.6, label='Observed Data')
plt.plot(x_lin, y_lin_true, 'r-', linewidth=2, label='True Linear Relationship')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Linear Relationship (Assumption Met)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(x_lin, y_nonlin_obs, alpha=0.6, label='Observed Data')
plt.plot(x_lin, y_nonlin_true, 'r-', linewidth=2, label='True Nonlinear Relationship')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Nonlinear Relationship (Assumption Violated)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_linearity.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2. Random Sampling
np.random.seed(42)
pop_size = 1000
sample_size = 50
population = np.random.normal(50, 10, pop_size)

# Random sample
random_indices = np.random.choice(pop_size, sample_size, replace=False)
random_sample = population[random_indices]

# Biased sample (only from upper half)
biased_indices = np.random.choice(np.where(population > 50)[0], sample_size, replace=False)
biased_sample = population[biased_indices]

plt.figure(figsize=(12, 6))
# Random sample
plt.subplot(1, 2, 1)
plt.hist(population, bins=30, alpha=0.3, color='blue', label='Population')
plt.hist(random_sample, bins=15, alpha=0.7, color='green', label='Random Sample')
plt.axvline(np.mean(population), color='blue', linestyle='--', label=f'Population Mean: {np.mean(population):.1f}')
plt.axvline(np.mean(random_sample), color='green', linestyle='--', label=f'Sample Mean: {np.mean(random_sample):.1f}')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Random Sampling (Assumption Met)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# Biased sample
plt.subplot(1, 2, 2)
plt.hist(population, bins=30, alpha=0.3, color='blue', label='Population')
plt.hist(biased_sample, bins=15, alpha=0.7, color='red', label='Biased Sample')
plt.axvline(np.mean(population), color='blue', linestyle='--', label=f'Population Mean: {np.mean(population):.1f}')
plt.axvline(np.mean(biased_sample), color='red', linestyle='--', label=f'Sample Mean: {np.mean(biased_sample):.1f}')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Biased Sampling (Assumption Violated)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_random_sampling.png"), dpi=300, bbox_inches='tight')
plt.close()

# 3. No Perfect Multicollinearity
np.random.seed(42)
n_samples = 100
x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(0, 1, n_samples)
x3 = 2*x1 + 3*x2  # Perfect multicollinearity: x3 = 2*x1 + 3*x2
x4 = x1 + np.random.normal(0, 0.1, n_samples)  # Near multicollinearity

plt.figure(figsize=(12, 6))
# No multicollinearity
plt.subplot(1, 2, 1)
plt.scatter(x1, x2, alpha=0.7)
plt.xlabel('X1', fontsize=12)
plt.ylabel('X2', fontsize=12)
plt.title(f'No Multicollinearity (Assumption Met)\nCorrelation: {np.corrcoef(x1, x2)[0, 1]:.2f}', fontsize=14)
plt.grid(alpha=0.3)

# Perfect multicollinearity
plt.subplot(1, 2, 2)
plt.scatter(x1, x4, alpha=0.7)
plt.xlabel('X1', fontsize=12)
plt.ylabel('X4 (≈ X1)', fontsize=12)
plt.title(f'Near Perfect Multicollinearity (Assumption Violated)\nCorrelation: {np.corrcoef(x1, x4)[0, 1]:.2f}', fontsize=14)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_no_multicollinearity.png"), dpi=300, bbox_inches='tight')
plt.close()

# 4. Exogeneity (Zero Mean Error)
np.random.seed(42)
x_exog = np.linspace(0, 10, 100)
y_exog_true = 3 + 2*x_exog
errors_zero_mean = np.random.normal(0, 1.5, size=len(x_exog))
errors_nonzero_mean = np.random.normal(2, 1.5, size=len(x_exog))  # Mean is 2, not 0
y_exog_obs = y_exog_true + errors_zero_mean
y_nonexog_obs = y_exog_true + errors_nonzero_mean

plt.figure(figsize=(12, 6))
# Zero mean errors
plt.subplot(1, 2, 1)
plt.scatter(x_exog, errors_zero_mean, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2, label=f'Error Mean: {np.mean(errors_zero_mean):.2f}')
plt.xlabel('X', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Zero Mean Errors (Assumption Met)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# Non-zero mean errors
plt.subplot(1, 2, 2)
plt.scatter(x_exog, errors_nonzero_mean, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero')
plt.axhline(y=np.mean(errors_nonzero_mean), color='b', linestyle='-', linewidth=2, 
            label=f'Error Mean: {np.mean(errors_nonzero_mean):.2f}')
plt.xlabel('X', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Non-Zero Mean Errors (Assumption Violated)', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_exogeneity.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5. Homoscedasticity
# Already demonstrated in the homo_vs_hetero.png visualization
# Create a simplified version for consistency
np.random.seed(42)
x_homo = np.linspace(0, 10, 100)
const_error = np.random.normal(0, 1, size=len(x_homo))
varying_error = np.random.normal(0, 0.3 + 0.25*x_homo, size=len(x_homo))

plt.figure(figsize=(12, 6))
# Homoscedastic errors
plt.subplot(1, 2, 1)
plt.scatter(x_homo, const_error, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
plt.axhline(y=2, color='g', linestyle='--', alpha=0.7, label='±2σ Bounds')
plt.axhline(y=-2, color='g', linestyle='--', alpha=0.7)
plt.xlabel('X', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Homoscedastic Errors (Assumption Met)\nVar(ε|X) = σ² = 4', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

# Heteroscedastic errors
plt.subplot(1, 2, 2)
plt.scatter(x_homo, varying_error, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
plt.plot(x_homo, 2*(0.3 + 0.25*x_homo), 'g--', alpha=0.7, label='±2σ Bounds')
plt.plot(x_homo, -2*(0.3 + 0.25*x_homo), 'g--', alpha=0.7)
plt.xlabel('X', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Heteroscedastic Errors (Assumption Violated)\nVar(ε|X) increases with X', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_homoscedasticity.png"), dpi=300, bbox_inches='tight')
plt.close()

# 6. No Autocorrelation
np.random.seed(42)
n = 100
x_auto = np.arange(n)

# Uncorrelated errors
uncorr_errors = np.random.normal(0, 1, size=n)

# Autocorrelated errors (AR(1) process)
autocorr_errors = np.zeros(n)
rho = 0.8  # Autocorrelation coefficient
autocorr_errors[0] = np.random.normal(0, 1)
for i in range(1, n):
    autocorr_errors[i] = rho * autocorr_errors[i-1] + np.random.normal(0, np.sqrt(1 - rho**2))

plt.figure(figsize=(12, 10))
# Uncorrelated errors
plt.subplot(2, 2, 1)
plt.scatter(x_auto, uncorr_errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
plt.xlabel('Observation Number', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Uncorrelated Errors (Assumption Met)', fontsize=14)
plt.grid(alpha=0.3)

# Autocorrelated errors
plt.subplot(2, 2, 2)
plt.scatter(x_auto, autocorr_errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
plt.xlabel('Observation Number', fontsize=12)
plt.ylabel('Error (ε)', fontsize=12)
plt.title('Autocorrelated Errors (Assumption Violated)', fontsize=14)
plt.grid(alpha=0.3)

# Autocorrelation plot for uncorrelated errors
plt.subplot(2, 2, 3)
lags = np.arange(1, 11)
autocorr_values = [np.corrcoef(uncorr_errors[:-lag], uncorr_errors[lag:])[0, 1] for lag in lags]
plt.bar(lags, autocorr_values)
plt.axhline(y=0, color='r', linestyle='-', linewidth=1)
plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.title('Autocorrelation Function for Uncorrelated Errors', fontsize=14)
plt.xticks(lags)
plt.grid(alpha=0.3)

# Autocorrelation plot for autocorrelated errors
plt.subplot(2, 2, 4)
autocorr_values = [np.corrcoef(autocorr_errors[:-lag], autocorr_errors[lag:])[0, 1] for lag in lags]
plt.bar(lags, autocorr_values)
plt.axhline(y=0, color='r', linestyle='-', linewidth=1)
plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7)
plt.axhline(y=-0.2, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Lag', fontsize=12)
plt.ylabel('Autocorrelation', fontsize=12)
plt.title('Autocorrelation Function for Autocorrelated Errors', fontsize=14)
plt.xticks(lags)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "assumption_no_autocorrelation.png"), dpi=300, bbox_inches='tight')
plt.close()

# Create a summary visualization for BLUE property
plt.figure(figsize=(12, 6))
plt.axis('off')
plt.text(0.5, 0.9, 'Gauss-Markov Theorem', fontsize=20, ha='center', weight='bold')
plt.text(0.5, 0.8, 'When all six assumptions are met, OLS is the BLUE:', fontsize=16, ha='center')
plt.text(0.5, 0.7, 'Best Linear Unbiased Estimator', fontsize=18, ha='center', weight='bold', color='blue')

plt.text(0.5, 0.6, 'BLUE means:', fontsize=16, ha='center')
plt.text(0.05, 0.5, '• Best: Has minimum variance among all linear unbiased estimators', fontsize=14)
plt.text(0.05, 0.45, '• Linear: Is a linear function of the dependent variable', fontsize=14)
plt.text(0.05, 0.4, '• Unbiased: Expected value of β equals the true parameter value', fontsize=14)
plt.text(0.05, 0.35, '• Estimator: Procedure for estimating the parameters', fontsize=14)

plt.text(0.5, 0.2, 'For more details on each assumption, refer to their individual visualizations', 
         fontsize=14, ha='center', bbox=dict(facecolor='lightgray', alpha=0.2))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "gauss_markov_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Impact of Homoscedasticity on Inference
# Generate data for confidence intervals under both scenarios
np.random.seed(123)
n_sim = 50  # Number of simulations
x_test = np.array([2, 5, 8])  # Three test points: low, middle, high value

# Initialize arrays to store predictions for each simulation
y_pred_homo = np.zeros((n_sim, len(x_test)))
y_pred_hetero = np.zeros((n_sim, len(x_test)))

# Run multiple simulations to get distribution of predictions
for i in range(n_sim):
    # Generate new data
    _, y_homo, _ = generate_homoscedastic_data(x, beta_0, beta_1, sigma)
    _, y_hetero, _, _ = generate_heteroscedastic_data(x, beta_0, beta_1, increasing_sigma)
    
    # Fit linear models to the data
    beta_1_homo = np.sum((x - np.mean(x)) * (y_homo - np.mean(y_homo))) / np.sum((x - np.mean(x))**2)
    beta_0_homo = np.mean(y_homo) - beta_1_homo * np.mean(x)
    
    beta_1_hetero = np.sum((x - np.mean(x)) * (y_hetero - np.mean(y_hetero))) / np.sum((x - np.mean(x))**2)
    beta_0_hetero = np.mean(y_hetero) - beta_1_hetero * np.mean(x)
    
    # Predict at test points
    y_pred_homo[i, :] = beta_0_homo + beta_1_homo * x_test
    y_pred_hetero[i, :] = beta_0_hetero + beta_1_hetero * x_test

# Calculate prediction means and standard deviations
mean_homo = np.mean(y_pred_homo, axis=0)
std_homo = np.std(y_pred_homo, axis=0)

mean_hetero = np.mean(y_pred_hetero, axis=0)
std_hetero = np.std(y_pred_hetero, axis=0)

# Create the visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the true regression line
x_line = np.linspace(0, 10, 100)
y_line = beta_0 + beta_1 * x_line
ax.plot(x_line, y_line, 'k-', linewidth=2, label='True Regression Line')

# Plot confidence bands for homoscedastic case
ax.fill_between(x_test, mean_homo - 1.96*std_homo, mean_homo + 1.96*std_homo, 
                color='blue', alpha=0.2, label='95% CI (Homoscedastic)')

# Plot confidence bands for heteroscedastic case
ax.fill_between(x_test, mean_hetero - 1.96*std_hetero, mean_hetero + 1.96*std_hetero, 
                color='red', alpha=0.2, label='95% CI (Heteroscedastic)')

# Add markers for the means
ax.scatter(x_test, mean_homo, color='blue', s=80, marker='o', label='Mean Prediction (Homoscedastic)')
ax.scatter(x_test, mean_hetero, color='red', s=80, marker='x', label='Mean Prediction (Heteroscedastic)')

# Add annotations showing the widths
for i, x_val in enumerate(x_test):
    # Width of homoscedastic CI
    width_homo = 2 * 1.96 * std_homo[i]
    ax.annotate(f"Width: {width_homo:.2f}", 
                xy=(x_val, mean_homo[i] + 1.96*std_homo[i]), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='blue')
    
    # Width of heteroscedastic CI
    width_hetero = 2 * 1.96 * std_hetero[i]
    ax.annotate(f"Width: {width_hetero:.2f}", 
                xy=(x_val, mean_hetero[i] + 1.96*std_hetero[i]), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=10, color='red')

# Add the standard deviations to the legend
legend_entries = [
    plt.Line2D([0], [0], color='white', label=f'σ = 2 (given in problem)'),
    plt.Line2D([0], [0], color='white', label=f'Var(ε) = σ² = 4')
]
ax.legend(handles=ax.get_legend_handles_labels()[0] + legend_entries, 
          loc='upper left', fontsize=12)

ax.set_xlabel('Predictor (x)', fontsize=14)
ax.set_ylabel('Response (y)', fontsize=14)
ax.set_title('Effect of Homoscedasticity on Prediction Intervals', fontsize=16)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "inference_impact.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Normal Distribution with σ = 2
x_norm = np.linspace(-8, 8, 1000)
y_norm = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm/sigma)**2))

plt.figure(figsize=(10, 6))
plt.plot(x_norm, y_norm, 'b-', linewidth=3)

# Shade the area within 1σ, 2σ, and 3σ
plt.fill_between(x_norm, 0, y_norm, where=np.abs(x_norm) <= sigma, 
                 color='blue', alpha=0.4, label='±1σ (68.3%)')
plt.fill_between(x_norm, 0, y_norm, where=(np.abs(x_norm) > sigma) & (np.abs(x_norm) <= 2*sigma), 
                 color='green', alpha=0.3, label='±2σ (95.4%)')
plt.fill_between(x_norm, 0, y_norm, where=(np.abs(x_norm) > 2*sigma) & (np.abs(x_norm) <= 3*sigma), 
                 color='red', alpha=0.2, label='±3σ (99.7%)')

# Add vertical lines at σ, 2σ, and 3σ
plt.axvline(x=sigma, color='blue', linestyle='--', alpha=0.7)
plt.axvline(x=-sigma, color='blue', linestyle='--', alpha=0.7)
plt.axvline(x=2*sigma, color='green', linestyle='--', alpha=0.7)
plt.axvline(x=-2*sigma, color='green', linestyle='--', alpha=0.7)
plt.axvline(x=3*sigma, color='red', linestyle='--', alpha=0.7)
plt.axvline(x=-3*sigma, color='red', linestyle='--', alpha=0.7)

# Annotate the values
plt.annotate(f'σ = {sigma}', xy=(sigma, 0.03), xytext=(sigma+0.5, 0.03), 
            arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=12)
plt.annotate(f'2σ = {2*sigma}', xy=(2*sigma, 0.01), xytext=(2*sigma+0.5, 0.01), 
            arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
plt.annotate(f'Var = σ² = {sigma**2}', xy=(0, 0.18), xytext=(3, 0.18), 
            fontsize=14, bbox=dict(facecolor='yellow', alpha=0.2))

plt.xlabel('Error Value (ε)', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.title('Normal Distribution of Errors with σ = 2 (Var = 4)', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "error_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to directory: {save_dir}")
print("\nQuestion 2 Answers:")
print("1. Homoscedasticity means that the errors in a regression model have constant")
print("   variance across all levels of the predictor variables. It implies that the")
print("   spread or dispersion of the errors is the same throughout the entire range of data.")
print()
print("2. Constant variance is important for linear regression because:")
print("   - It ensures the OLS estimator is the Best Linear Unbiased Estimator (BLUE)")
print("   - It leads to correct standard errors and valid statistical inference")
print("   - It provides consistent prediction accuracy across all predictor values")
print("   - It simplifies statistical theory by using a single parameter (σ²) for error variance")
print()
print("3. Given σ = 2, the variance of the errors is σ² = 2² = 4") 