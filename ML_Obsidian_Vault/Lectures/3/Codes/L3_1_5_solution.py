import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_1_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"{title.center(80)}")
    print("="*80 + "\n")

# Introduction
print_section_header("QUESTION 5: GAUSS-MARKOV THEOREM AND BLUE")

print("""Problem Statement:
In the context of the Gauss-Markov theorem, consider the Best Linear Unbiased Estimator (BLUE).

Tasks:
1. List three key assumptions of the Gauss-Markov theorem
2. Explain what "unbiased" means in the context of BLUE
3. Why is the OLS estimator considered "best" among all linear unbiased estimators?
""")

# Step 1: Gauss-Markov Assumptions
print_section_header("STEP 1: GAUSS-MARKOV ASSUMPTIONS")

print("""The Gauss-Markov theorem relies on several assumptions about the linear regression model:
y = Xβ + ε

1. Linearity: The relationship between X and y is linear in parameters.
   - This means that y can be written as a linear combination of the parameters (β).
   - Mathematically: E[y|X] = Xβ

2. Random Sampling: The data is sampled randomly from the population.
   - This implies that observations are independently and identically distributed (i.i.d.).

3. No Perfect Multicollinearity: No exact linear relationships among predictors.
   - The matrix X has full column rank, meaning (X'X) is invertible.
   - This ensures that the OLS estimator β̂ = (X'X)⁻¹X'y is unique.

4. Exogeneity / Zero Conditional Mean: The errors have zero mean conditional on X.
   - E[ε|X] = 0
   - This means predictors are not correlated with the error term.

5. Homoscedasticity: The errors have constant variance.
   - Var(ε|X) = σ²I
   - This means the error variance is the same for all observations.

Note: For the BLUE property, we don't need to assume normality of errors, though normality
is often assumed for hypothesis testing and confidence intervals.
""")

# Step 2: What "Unbiased" Means
print_section_header("STEP 2: WHAT 'UNBIASED' MEANS")

print("""An estimator β̂ is unbiased if its expected value equals the true parameter value:
E[β̂] = β

This means that if we could repeatedly sample data and compute the estimator each time,
the average of these estimates would converge to the true parameter value.

For the OLS estimator β̂ = (X'X)⁻¹X'y, we can prove unbiasedness as follows:

1. Substitute y = Xβ + ε:
   β̂ = (X'X)⁻¹X'(Xβ + ε) = (X'X)⁻¹X'Xβ + (X'X)⁻¹X'ε

2. Simplify: (X'X)⁻¹X'X = I (identity matrix)
   β̂ = β + (X'X)⁻¹X'ε

3. Take expectations:
   E[β̂] = E[β + (X'X)⁻¹X'ε] = β + (X'X)⁻¹X'E[ε]

4. Given our assumption that E[ε|X] = 0, we have E[ε] = 0, so:
   E[β̂] = β

Therefore, the OLS estimator is unbiased.
""")

# Step 3: Why OLS is "Best"
print_section_header("STEP 3: WHY OLS IS 'BEST'")

print("""In the context of the Gauss-Markov theorem, "best" means having minimum variance
among all linear unbiased estimators. This is why we call it the Best Linear Unbiased Estimator (BLUE).

To understand this:

1. Consider any linear estimator: β̃ = Cy, where C is any p×n matrix.

2. For this estimator to be unbiased, we need E[β̃] = β for all β, which implies:
   C must equal (X'X)⁻¹X' or CX must equal I.

3. The Gauss-Markov theorem proves that among all such unbiased linear estimators,
   the OLS estimator β̂ = (X'X)⁻¹X'y has the smallest variance.

4. Specifically, for any other unbiased linear estimator β̃, we have:
   Var(β̂) ≤ Var(β̃) in the positive semi-definite sense.

5. This means that Var(β̃) - Var(β̂) is a positive semi-definite matrix.

This minimum variance property is crucial because it means OLS gives us the most precise
estimates possible (smallest standard errors) among all linear unbiased estimators.
""")

# Visualizations
print_section_header("VISUALIZATIONS")

# Visualization 1: Biased vs Unbiased Estimators
np.random.seed(42)
true_beta = 2.5  # True parameter value
num_samples = 10  # Samples for each estimator

# Generate data for multiple sampling instances
sample_betas_ols = []  # OLS estimator (unbiased)
sample_betas_biased1 = []  # Biased estimator 1 (deliberately underestimates)
sample_betas_biased2 = []  # Biased estimator 2 (deliberately overestimates)

for _ in range(num_samples):
    # Generate a simple random sample
    n = 30
    X = np.random.normal(0, 1, n)
    X = np.column_stack([np.ones(n), X])  # Add intercept
    epsilon = np.random.normal(0, 1, n)
    y = X @ np.array([1, true_beta]) + epsilon
    
    # Compute OLS estimator - focus on slope coefficient
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    sample_betas_ols.append(beta_ols[1])
    
    # Compute biased estimator 1 (shrinkage toward 0)
    beta_biased1 = 0.7 * beta_ols
    sample_betas_biased1.append(beta_biased1[1])
    
    # Compute biased estimator 2 (adds a constant)
    beta_biased2 = beta_ols + np.array([0, 0.8])
    sample_betas_biased2.append(beta_biased2[1])

# Plot the samples and their means
plt.figure(figsize=(10, 6))
plt.axhline(y=true_beta, color='r', linestyle='-', label='True β', linewidth=2)

plt.scatter(np.ones(num_samples)*1, sample_betas_ols, alpha=0.7, s=60, color='blue')
plt.scatter(np.ones(num_samples)*2, sample_betas_biased1, alpha=0.7, s=60, color='green')
plt.scatter(np.ones(num_samples)*3, sample_betas_biased2, alpha=0.7, s=60, color='purple')

# Add mean values
plt.scatter(1, np.mean(sample_betas_ols), color='blue', s=150, marker='X', 
            label=f'OLS: Mean={np.mean(sample_betas_ols):.3f}')
plt.scatter(2, np.mean(sample_betas_biased1), color='green', s=150, marker='X', 
            label=f'Biased 1: Mean={np.mean(sample_betas_biased1):.3f}')
plt.scatter(3, np.mean(sample_betas_biased2), color='purple', s=150, marker='X', 
            label=f'Biased 2: Mean={np.mean(sample_betas_biased2):.3f}')

plt.xticks([1, 2, 3], ['OLS (Unbiased)', 'Shrinkage (Biased)', 'Shifted (Biased)'])
plt.ylabel('Estimated β', fontsize=12)
plt.title('Unbiased vs. Biased Estimators', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add explanatory text
textbox = "An unbiased estimator has an expected value\nequal to the true parameter value.\nThe OLS estimator (blue) is centered around\nthe true value (red line)."
plt.text(0.02, 0.98, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
unbiased_plot_path = os.path.join(save_dir, "unbiased_estimators.png")
plt.savefig(unbiased_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Minimum Variance Property (BLUE)
np.random.seed(42)
n_sims = 1000  # Number of simulations
n_obs = 30  # Observations per simulation
beta = np.array([1, 2])  # True parameters: intercept and slope

# Arrays to store estimates
ols_estimates = np.zeros((n_sims, 2))  # OLS estimator
other_estimates = np.zeros((n_sims, 2))  # Another unbiased linear estimator

# Run simulations
for i in range(n_sims):
    # Generate data
    X = np.random.normal(0, 1, n_obs)
    X = np.column_stack([np.ones(n_obs), X])  # Add intercept
    epsilon = np.random.normal(0, 1, n_obs)
    y = X @ beta + epsilon
    
    # OLS estimator
    ols_estimates[i] = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Another unbiased linear estimator with higher variance
    # This creates a valid unbiased estimator by adding noise to OLS
    # in a way that preserves unbiasedness
    noise = np.random.normal(0, 0.5, 2)  # Random noise
    other_estimates[i] = ols_estimates[i] + np.array([noise[0] - np.mean(noise), noise[1] - np.mean(noise)])

# Calculate means and variances
ols_mean = np.mean(ols_estimates, axis=0)
other_mean = np.mean(other_estimates, axis=0)
ols_var = np.var(ols_estimates, axis=0)
other_var = np.var(other_estimates, axis=0)

print(f"OLS Estimator - Mean: {ols_mean}, Variance: {ols_var}")
print(f"Other Estimator - Mean: {other_mean}, Variance: {other_var}")
print(f"Variance Ratio (Other/OLS): {other_var/ols_var}")

# Plot the estimates (focusing on slope coefficient - beta[1])
plt.figure(figsize=(10, 6))

plt.hist(ols_estimates[:, 1], bins=30, alpha=0.5, color='blue', label=f'OLS: Var={ols_var[1]:.3f}')
plt.hist(other_estimates[:, 1], bins=30, alpha=0.5, color='red', label=f'Other: Var={other_var[1]:.3f}')

plt.axvline(x=beta[1], color='black', linestyle='--', linewidth=2, label='True β')
plt.axvline(x=ols_mean[1], color='blue', linestyle='-', linewidth=2)
plt.axvline(x=other_mean[1], color='red', linestyle='-', linewidth=2)

plt.xlabel('Estimated Slope (β₁)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Estimators: The BLUE Property', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add explanatory text
textbox = "The OLS estimator (blue) has lower variance\nthan the alternative estimator (red),\ndemonstrating the 'Best' property of BLUE.\nBoth estimators are unbiased (centered on true β)."
plt.text(0.02, 0.98, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
blue_plot_path = os.path.join(save_dir, "blue_property.png")
plt.savefig(blue_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Gauss-Markov Assumptions
plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=plt.gcf())

# 1. Linearity and Zero Mean Errors
ax1 = plt.subplot(gs[0, 0])
np.random.seed(123)
x = np.linspace(-3, 3, 100)
y_true = 1 + 2*x
y = y_true + np.random.normal(0, 0.5, 100)

ax1.scatter(x, y, alpha=0.6, color='blue', label='Data')
ax1.plot(x, y_true, 'r-', label='True Line: y = 1 + 2x')
ax1.set_title('Linearity & Zero Mean Errors', fontsize=12)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Add zero mean error annotation
residuals = y - y_true
ax1.text(0.05, 0.95, f'Mean of Errors: {np.mean(residuals):.3f} ≈ 0', 
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Homoscedasticity vs. Heteroscedasticity
ax2 = plt.subplot(gs[0, 1])
x_homo = np.linspace(-3, 3, 100)
np.random.seed(456)
error_homo = np.random.normal(0, 0.5, 100)  # Constant variance
error_hetero = np.random.normal(0, 0.1 + 0.2*np.abs(x_homo), 100)  # Variance increases with |x|
y_homo = 1 + 2*x_homo + error_homo
y_hetero = 1 + 2*x_homo + error_hetero

ax2.scatter(x_homo, y_homo, alpha=0.6, color='green', label='Homoscedastic')
ax2.scatter(x_homo, y_hetero, alpha=0.6, color='red', label='Heteroscedastic')
ax2.plot(x_homo, 1 + 2*x_homo, 'k-', alpha=0.7, label='True Line')
ax2.set_title('Homoscedasticity vs. Heteroscedasticity', fontsize=12)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. No Multicollinearity
ax3 = plt.subplot(gs[1, 0])
np.random.seed(789)
n = 100
x1 = np.random.normal(0, 1, n)
# Create x2 with different levels of correlation with x1
x2_indep = np.random.normal(0, 1, n)  # Independent of x1
x2_corr = 0.8*x1 + 0.2*np.random.normal(0, 1, n)  # Correlated with x1
x2_perf = x1  # Perfect multicollinearity

ax3.scatter(x1, x2_indep, alpha=0.6, color='blue', label='Low correlation')
ax3.scatter(x1, x2_corr, alpha=0.6, color='orange', label='High correlation')
ax3.scatter(x1, x2_perf, alpha=0.6, color='red', label='Perfect multicollinearity')
ax3.set_title('Multicollinearity', fontsize=12)
ax3.set_xlabel('x₁', fontsize=10)
ax3.set_ylabel('x₂', fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Calculate and display correlations
corr_indep = np.corrcoef(x1, x2_indep)[0, 1]
corr_high = np.corrcoef(x1, x2_corr)[0, 1]
corr_perf = np.corrcoef(x1, x2_perf)[0, 1]

ax3.text(0.05, 0.95, f'Correlations:\nLow: {corr_indep:.3f}\nHigh: {corr_high:.3f}\nPerfect: {corr_perf:.3f}', 
         transform=ax3.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 4. OLS vs Other Estimators
ax4 = plt.subplot(gs[1, 1])
estimator_names = ['OLS (BLUE)', 'Ridge', 'Lasso', 'Random']
bias = [0, 0.2, 0.3, 0.5]  # Relative bias
variance = [1, 0.7, 0.6, 1.2]  # Relative variance

# Calculate MSE = bias² + variance
mse = [b**2 + v for b, v in zip(bias, variance)]

x_pos = np.arange(len(estimator_names))
width = 0.3

ax4.bar(x_pos - width/2, bias, width, alpha=0.7, color='red', label='Bias²')
ax4.bar(x_pos + width/2, variance, width, alpha=0.7, color='blue', label='Variance')
ax4.bar(x_pos, mse, width/8, alpha=1.0, color='black', label='MSE')

ax4.set_xticks(x_pos)
ax4.set_xticklabels(estimator_names, rotation=45, ha='right')
ax4.set_title('Trade-off: Bias vs. Variance', fontsize=12)
ax4.set_ylabel('Relative Value', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Add text explaining OLS is BLUE
ax4.text(0.05, 0.95, 'OLS has zero bias but may not\nhave lowest MSE due to variance.', 
         transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Gauss-Markov Assumptions and BLUE', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

gm_assumptions_path = os.path.join(save_dir, "gm_assumptions.png")
plt.savefig(gm_assumptions_path, dpi=300, bbox_inches='tight')
plt.close()

# Visualization 4: Unbiasedness vs. Efficiency
plt.figure(figsize=(10, 6))

# Parameters for normal distributions
mu_true = 2.5  # True parameter value
se_ols = 0.3   # Standard error of OLS (lower variance)
se_other = 0.6  # Standard error of another unbiased estimator (higher variance)
se_biased = 0.2  # Standard error of a biased estimator (very efficient but biased)
bias = -0.5     # Bias amount

# Create x-axis points
x = np.linspace(0.5, 4.5, 1000)

# Plot the distributions
y_ols = stats.norm.pdf(x, mu_true, se_ols)
y_other = stats.norm.pdf(x, mu_true, se_other)
y_biased = stats.norm.pdf(x, mu_true + bias, se_biased)

plt.plot(x, y_ols, 'b-', linewidth=2, label=f'OLS (BLUE): Unbiased, Var={se_ols**2:.2f}')
plt.plot(x, y_other, 'g-', linewidth=2, label=f'Other Unbiased: Var={se_other**2:.2f}')
plt.plot(x, y_biased, 'r-', linewidth=2, label=f'Biased: Bias={bias}, Var={se_biased**2:.2f}')

plt.axvline(x=mu_true, color='black', linestyle='--', label='True Parameter')
plt.axvline(x=mu_true + bias, color='red', linestyle='--')

plt.xlabel('Parameter Estimate', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Unbiasedness vs. Efficiency in Estimators', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations
plt.annotate('Bias', xy=(mu_true - 0.02, 0.05), xytext=(mu_true - 0.25, 0.3),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))

# Add explanatory text
textbox = "• BLUE estimators (blue) are unbiased and have minimum variance\n  among all linear unbiased estimators.\n• Other unbiased estimators (green) have higher variance.\n• Some biased estimators (red) can have lower MSE due to variance reduction."
plt.text(0.02, 0.98, textbox, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
efficiency_path = os.path.join(save_dir, "efficiency_plot.png")
plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
plt.close()

# Summary
print_section_header("SUMMARY")

print("""Key Points about the Gauss-Markov Theorem and BLUE:

1. Gauss-Markov Assumptions:
   - Linearity: The model is linear in parameters
   - Random Sampling: Observations are i.i.d.
   - No Perfect Multicollinearity: X has full column rank
   - Exogeneity: E[ε|X] = 0 (zero conditional mean of errors)
   - Homoscedasticity: Var(ε|X) = σ²I (constant error variance)

2. Unbiasedness:
   - An estimator β̂ is unbiased if E[β̂] = β
   - OLS is unbiased: E[(X'X)⁻¹X'y] = β

3. BLUE (Best Linear Unbiased Estimator):
   - "Best" means minimum variance among all linear unbiased estimators
   - For any other unbiased linear estimator β̃, we have Var(β̂) ≤ Var(β̃)
   - This means OLS gives the most precise estimates possible

4. Practical Implications:
   - When Gauss-Markov assumptions hold, OLS is optimal among unbiased estimators
   - If some assumptions are violated, other estimators might be more appropriate
   - Biased estimators (like Ridge or Lasso) may have lower MSE in some cases
""")

print(f"\nVisualizations saved in: {save_dir}")
print(f"1. Unbiased Estimators: {unbiased_plot_path}")
print(f"2. BLUE Property: {blue_plot_path}")
print(f"3. Gauss-Markov Assumptions: {gm_assumptions_path}")
print(f"4. Efficiency Plot: {efficiency_path}") 