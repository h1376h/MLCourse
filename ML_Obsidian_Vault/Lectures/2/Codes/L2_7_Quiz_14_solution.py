import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, beta, multivariate_normal
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_14")
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
print("- Five statements about MAP and Bayesian inference concepts")
print("\nTask:")
print("Evaluate whether each of the following statements is TRUE or FALSE, with justification:")
print("1. When the posterior distribution is symmetric and unimodal, the MAP estimate and the posterior mean are identical.")
print("2. Bayesian model averaging can never perform worse than selecting the single best model according to posterior probability.")
print("3. As the number of data points approaches infinity, the influence of the prior on the posterior distribution approaches zero.")
print("4. The Bayesian Information Criterion (BIC) provides a closer approximation to the log marginal likelihood than the Akaike Information Criterion (AIC).")
print("5. Variational inference methods always converge to the exact posterior distribution given enough computational resources.")

# Step 2: Statement 1 - MAP vs Mean for Symmetric Unimodal Posterior
print_step_header(2, "Statement 1: MAP vs Mean for Symmetric Unimodal Posterior")

statement1 = "When the posterior distribution is symmetric and unimodal, the MAP estimate and the posterior mean are identical."
conclusion1 = "TRUE"

print(f"Statement: {statement1}")
print(f"Conclusion: {conclusion1}")
print("\nJustification:")
print("For a symmetric unimodal distribution, the mode (which is the MAP estimate) coincides")
print("with the mean. This is because in a symmetric distribution, the probability density")
print("function has the same shape on both sides of the mode, so the mean and mode align at")
print("the center of symmetry.")
print("\nExamples of such distributions include:")
print("- Normal (Gaussian) distribution")
print("- Student's t-distribution with degrees of freedom > 1")
print("- Uniform distribution")
print("\nHowever, this is NOT true for asymmetric distributions, even if they are unimodal.")
print("For example, in a gamma or a beta distribution that is skewed, the MAP and mean will differ.")

# Let's visualize this with examples
x = np.linspace(-4, 4, 1000)

# Example 1: Symmetric distribution (Gaussian)
gaussian = norm.pdf(x, 0, 1)
gaussian_mean = 0
gaussian_mode = 0

# Example 2: Asymmetric distribution (Beta)
beta_x = np.linspace(0, 1, 1000)
beta_params = (2, 5)
beta_dist = beta.pdf(beta_x, *beta_params)
beta_mean = beta_params[0] / (beta_params[0] + beta_params[1])
beta_mode = (beta_params[0] - 1) / (beta_params[0] + beta_params[1] - 2) if beta_params[0] > 1 else 0

plt.figure(figsize=(12, 8))

# Plot Gaussian
plt.subplot(2, 1, 1)
plt.plot(x, gaussian, 'b-', linewidth=2, label='Gaussian Distribution')
plt.axvline(x=gaussian_mean, color='r', linestyle='--', label=f'Mean = {gaussian_mean}')
plt.axvline(x=gaussian_mode, color='g', linestyle='--', label=f'MAP = {gaussian_mode}')
plt.title('Symmetric Unimodal Distribution: Mean = MAP', fontsize=14)
plt.legend()
plt.grid(True)

# Plot Beta
plt.subplot(2, 1, 2)
plt.plot(beta_x, beta_dist, 'b-', linewidth=2, label=f'Beta({beta_params[0]}, {beta_params[1]}) Distribution')
plt.axvline(x=beta_mean, color='r', linestyle='--', label=f'Mean = {beta_mean:.3f}')
plt.axvline(x=beta_mode, color='g', linestyle='--', label=f'MAP = {beta_mode:.3f}')
plt.title('Asymmetric Unimodal Distribution: Mean ≠ MAP', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
file_path = os.path.join(save_dir, "statement1_map_vs_mean.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Statement 2 - Bayesian Model Averaging Performance
print_step_header(3, "Statement 2: Bayesian Model Averaging Performance")

statement2 = "Bayesian model averaging can never perform worse than selecting the single best model according to posterior probability."
conclusion2 = "FALSE"

print(f"Statement: {statement2}")
print(f"Conclusion: {conclusion2}")
print("\nJustification:")
print("While Bayesian model averaging (BMA) often provides more robust predictions by accounting")
print("for model uncertainty, it can sometimes perform worse than selecting the single best model.")
print("\nReasons why BMA might perform worse:")
print("1. If the posterior model probabilities are poorly estimated (e.g., due to limitations in")
print("   the prior specification or likelihood calculation), BMA can give inappropriate weights")
print("   to poor models.")
print("2. If multiple models make similar predictions but one model is significantly better,")
print("   averaging can dilute the predictions of the best model.")
print("3. In some cases, a simpler, less accurate model can be assigned too much weight,")
print("   leading to worse overall predictions.")
print("4. For finite datasets, BMA incorporates estimation uncertainty, which can sometimes")
print("   increase prediction variance without compensating improvement in bias.")
print("\nIn theory, with perfect model specification and infinite data, BMA should never perform")
print("worse than selecting the best model. However, in practice with finite data and imperfect")
print("models, there are cases where selecting the highest posterior probability model works better.")

# Visualize a scenario where BMA might perform worse
# Example: True function is quadratic, but we have linear and cubic models
np.random.seed(42)
x_true = np.linspace(-1, 1, 100)
true_func = 2 * x_true**2  # True quadratic function
y_true = true_func

# Sampled data points (with noise)
x_data = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
y_data = 2 * x_data**2 + np.random.normal(0, 0.1, size=len(x_data))

# Models
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit models (simplified)
# In practice, you would use proper fitting procedures like least squares
linear_params = [0, 1.0]  # Simplified fit
quadratic_params = [2.0, 0, 0]  # Correct model
cubic_params = [1.0, 2.0, 0, 0]  # Overfit model

# Generate predictions
linear_pred = linear_model(x_true, *linear_params)
quadratic_pred = quadratic_model(x_true, *quadratic_params)
cubic_pred = cubic_model(x_true, *cubic_params)

# Assume posterior probabilities (simplified)
# In this case, we'll say the cubic model gets too much weight due to overfitting
linear_prob = 0.1
quadratic_prob = 0.3  # Best model but not highest posterior
cubic_prob = 0.6  # Highest posterior due to overfitting

# BMA prediction
bma_pred = linear_prob * linear_pred + quadratic_prob * quadratic_pred + cubic_prob * cubic_pred

plt.figure(figsize=(12, 8))
plt.plot(x_true, y_true, 'k-', linewidth=2, label='True Function')
plt.plot(x_true, linear_pred, 'r--', linewidth=1.5, label='Linear Model (P=0.1)')
plt.plot(x_true, quadratic_pred, 'g--', linewidth=1.5, label='Quadratic Model (P=0.3) - Best')
plt.plot(x_true, cubic_pred, 'b--', linewidth=1.5, label='Cubic Model (P=0.6) - Highest Posterior')
plt.plot(x_true, bma_pred, 'm-', linewidth=2, label='BMA Prediction')
plt.scatter(x_data, y_data, color='k', s=50, label='Data Points')

# Add annotations for errors
plt.annotate('BMA Error > Best Model Error', 
             xy=(0.7, bma_pred[75]), 
             xytext=(0.4, 1.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Case Where BMA Performs Worse Than Best Individual Model', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement2_bma_performance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Statement 3 - Influence of Prior with Infinite Data
print_step_header(4, "Statement 3: Influence of Prior with Infinite Data")

statement3 = "As the number of data points approaches infinity, the influence of the prior on the posterior distribution approaches zero."
conclusion3 = "TRUE"

print(f"Statement: {statement3}")
print(f"Conclusion: {conclusion3}")
print("\nJustification:")
print("This statement is true under regularity conditions and is often referred to as the")
print("'Bernstein–von Mises theorem' or the concept of 'prior washing out'.")
print("\nAs the amount of data increases:")
print("1. The likelihood function becomes more peaked around the maximum likelihood estimate")
print("2. The likelihood overwhelms the prior in their product (which is proportional to the posterior)")
print("3. The posterior distribution converges to a normal distribution centered at the true parameter value")
print("   with covariance determined by the Fisher information matrix (not by the prior)")
print("\nAsymptotically, with infinite data, the Bayesian posterior and frequentist sampling")
print("distribution of the maximum likelihood estimator become equivalent, regardless of the")
print("prior (as long as the prior has non-zero density at the true parameter value).")
print("\nThis is mathematically expressed as:")
print("p(θ|D) → N(θ_MLE, I(θ_MLE)^(-1)) as n → ∞")
print("where I(θ_MLE) is the Fisher information matrix evaluated at the MLE.")

# Visualize the prior washing out
# Let's consider a coin flip example with different priors
n_array = [0, 1, 5, 10, 50, 1000]  # Number of coin flips
p_true = 0.7  # True probability of heads
heads_proportion = p_true  # Proportion of heads observed

# Define two different priors
prior1_params = (1, 1)  # Uniform prior
prior2_params = (10, 2)  # Strong prior favoring heads

theta = np.linspace(0, 1, 1000)

plt.figure(figsize=(15, 10))

for i, n in enumerate(n_array):
    # Number of heads
    h = int(n * heads_proportion)
    
    # Posteriors
    posterior1_params = (prior1_params[0] + h, prior1_params[1] + n - h)
    posterior2_params = (prior2_params[0] + h, prior2_params[1] + n - h)
    
    posterior1 = beta.pdf(theta, *posterior1_params)
    posterior2 = beta.pdf(theta, *posterior2_params)
    
    # Normalize to max=1 for easier comparison
    posterior1 = posterior1 / np.max(posterior1) if np.max(posterior1) > 0 else posterior1
    posterior2 = posterior2 / np.max(posterior2) if np.max(posterior2) > 0 else posterior2
    
    plt.subplot(2, 3, i+1)
    
    # Plot priors if n=0
    if n == 0:
        prior1 = beta.pdf(theta, *prior1_params)
        prior2 = beta.pdf(theta, *prior2_params)
        prior1 = prior1 / np.max(prior1)
        prior2 = prior2 / np.max(prior2)
        plt.plot(theta, prior1, 'b-', linewidth=2, label='Uniform Prior')
        plt.plot(theta, prior2, 'r-', linewidth=2, label='Strong Prior')
    else:
        plt.plot(theta, posterior1, 'b-', linewidth=2, label='Posterior with Uniform Prior')
        plt.plot(theta, posterior2, 'r-', linewidth=2, label='Posterior with Strong Prior')
        
    plt.axvline(x=p_true, color='k', linestyle='--', label=f'True θ = {p_true}')
    
    plt.title(f'n = {n} observations', fontsize=12)
    plt.xlabel('θ', fontsize=10)
    plt.ylabel('Density (Normalized)', fontsize=10)
    plt.grid(True)
    plt.legend()

plt.suptitle('Prior Influence Diminishes with Increasing Data', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle

file_path = os.path.join(save_dir, "statement3_prior_influence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Statement 4 - BIC vs AIC Approximation
print_step_header(5, "Statement 4: BIC vs AIC Approximation")

statement4 = "The Bayesian Information Criterion (BIC) provides a closer approximation to the log marginal likelihood than the Akaike Information Criterion (AIC)."
conclusion4 = "TRUE"

print(f"Statement: {statement4}")
print(f"Conclusion: {conclusion4}")
print("\nJustification:")
print("The statement is true because BIC was specifically designed to approximate the log")
print("marginal likelihood (also called model evidence), while AIC was designed with a different goal.")
print("\nThe Bayesian Information Criterion (BIC) is given by:")
print("BIC = -2 * log(likelihood) + k * log(n)")
print("where k is the number of parameters and n is the number of data points.")
print("\nThe Akaike Information Criterion (AIC) is given by:")
print("AIC = -2 * log(likelihood) + 2 * k")
print("\nAs n increases, BIC approximates the log marginal likelihood up to an additive constant:")
print("log p(D|M) ≈ -BIC/2 + constant")
print("\nThis follows from the Laplace approximation to the marginal likelihood under certain conditions.")
print("The penalty term k*log(n) in BIC appears naturally from this approximation.")
print("\nIn contrast, AIC aims to minimize the expected Kullback-Leibler divergence between the")
print("true data-generating process and the model, which is a different goal than approximating")
print("the marginal likelihood. The penalty term 2*k in AIC does not match the term that would")
print("arise from approximating the log marginal likelihood.")
print("\nTherefore, BIC is a closer approximation to the log marginal likelihood than AIC.")

# Visualize BIC vs AIC penalty terms
model_complexity = np.arange(1, 11)  # Number of parameters
n_values = [10, 100, 1000]

plt.figure(figsize=(12, 8))

for i, n in enumerate(n_values):
    bic_penalty = model_complexity * np.log(n)
    aic_penalty = 2 * model_complexity
    
    plt.plot(model_complexity, bic_penalty, 'o-', label=f'BIC Penalty (n={n})', linewidth=2)
    
    if i == 0:  # Only add AIC once since it doesn't depend on n
        plt.plot(model_complexity, aic_penalty, 's--', label='AIC Penalty', linewidth=2, color='black')

plt.xlabel('Number of Parameters (k)', fontsize=12)
plt.ylabel('Penalty Term', fontsize=12)
plt.title('Comparison of Penalty Terms in BIC and AIC', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement4_bic_vs_aic.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Plot showing how BIC approximates marginal likelihood
plt.figure(figsize=(12, 8))

# Generate data for illustration
model_complexity = np.arange(1, 11)
# Hypothetical log marginal likelihood (peaks at optimal complexity)
log_marginal = -5 * (model_complexity - 5)**2 + 10
# BIC approximation (scaled and shifted for illustration)
bic_approx = -5 * (model_complexity - 5)**2 + 8
# AIC (tends to favor more complex models)
aic_approx = -5 * (model_complexity - 5)**2 + 14 + 0.5 * model_complexity

plt.plot(model_complexity, log_marginal, 'k-', linewidth=3, label='True Log Marginal Likelihood')
plt.plot(model_complexity, bic_approx, 'b--', linewidth=2, label='BIC Approximation')
plt.plot(model_complexity, aic_approx, 'r-.', linewidth=2, label='AIC Approximation')

# Mark the optimal model complexities
max_marginal = np.argmax(log_marginal) + 1
max_bic = np.argmax(bic_approx) + 1
max_aic = np.argmax(aic_approx) + 1

plt.axvline(x=max_marginal, color='k', linestyle=':', label=f'Optimal Complexity (Marginal): {max_marginal}')
plt.axvline(x=max_bic, color='b', linestyle=':', label=f'BIC Selection: {max_bic}')
plt.axvline(x=max_aic, color='r', linestyle=':', label=f'AIC Selection: {max_aic}')

plt.xlabel('Model Complexity', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comparison of Log Marginal Likelihood, BIC, and AIC', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement4_bic_aic_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 6: Statement 5 - Variational Inference Convergence
print_step_header(6, "Statement 5: Variational Inference Convergence")

statement5 = "Variational inference methods always converge to the exact posterior distribution given enough computational resources."
conclusion5 = "FALSE"

print(f"Statement: {statement5}")
print(f"Conclusion: {conclusion5}")
print("\nJustification:")
print("Variational inference (VI) methods may NOT converge to the exact posterior distribution")
print("even with unlimited computational resources. This is because VI methods are fundamentally")
print("limited by the approximating family of distributions they use.")
print("\nVI works by finding the distribution q(θ) within a chosen family of distributions")
print("that minimizes the KL divergence KL(q(θ) || p(θ|D)) from the true posterior.")
print("\nIf the true posterior does not belong to the chosen approximating family, then VI will")
print("never reach the exact posterior, regardless of computational resources. Instead, it will")
print("converge to the closest distribution within the family, as measured by KL divergence.")
print("\nCommon limitations of VI include:")
print("1. Mean-field approximations assume independence between parameters when the posterior")
print("   has strong correlations")
print("2. Variational families with fixed parametric forms (e.g., Gaussian) can't capture")
print("   multi-modality or skewness in the posterior")
print("3. Minimizing KL(q||p) tends to make q underestimate the variance of p")
print("\nMore flexible approximating families (like normalizing flows) can reduce this gap, but")
print("the issue is inherent to the method - there's always a trade-off between tractability")
print("and expressiveness of the approximating family.")

# Let's visualize this limitation
# Create a bimodal true posterior that can't be captured by a unimodal Gaussian approximation
x = np.linspace(-5, 5, 1000)

# True bimodal posterior
def bimodal_posterior(x):
    return 0.5 * norm.pdf(x, -2, 0.8) + 0.5 * norm.pdf(x, 2, 0.8)

true_posterior = bimodal_posterior(x)

# Best Gaussian approximation
# In VI, we would minimize KL(q||p), which tends to place q at one of the modes
# For visualization, we'll use a Gaussian fit to the overall distribution
mean_approx = np.sum(x * true_posterior) / np.sum(true_posterior)
var_approx = np.sum((x - mean_approx)**2 * true_posterior) / np.sum(true_posterior)
gaussian_approx = norm.pdf(x, mean_approx, np.sqrt(var_approx))

# Normalize for better visualization
true_posterior = true_posterior / np.max(true_posterior)
gaussian_approx = gaussian_approx / np.max(gaussian_approx)

plt.figure(figsize=(12, 8))
plt.plot(x, true_posterior, 'b-', linewidth=2, label='True Bimodal Posterior')
plt.plot(x, gaussian_approx, 'r--', linewidth=2, label='Best Gaussian Approximation')
plt.fill_between(x, true_posterior, gaussian_approx, where=(true_posterior > gaussian_approx), 
                 color='blue', alpha=0.3, label='Areas Underestimated by VI')
plt.fill_between(x, true_posterior, gaussian_approx, where=(true_posterior < gaussian_approx), 
                 color='red', alpha=0.3, label='Areas Overestimated by VI')

plt.xlabel('Parameter Value', fontsize=12)
plt.ylabel('Normalized Density', fontsize=12)
plt.title('Limitation of Variational Inference: Cannot Capture Multi-modality', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement5_vi_limitation.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Summary of Conclusions
print_step_header(7, "Summary of Conclusions")

statements = [
    "When the posterior distribution is symmetric and unimodal, the MAP estimate and the posterior mean are identical.",
    "Bayesian model averaging can never perform worse than selecting the single best model according to posterior probability.",
    "As the number of data points approaches infinity, the influence of the prior on the posterior distribution approaches zero.",
    "The Bayesian Information Criterion (BIC) provides a closer approximation to the log marginal likelihood than the Akaike Information Criterion (AIC).",
    "Variational inference methods always converge to the exact posterior distribution given enough computational resources."
]

conclusions = ["TRUE", "FALSE", "TRUE", "TRUE", "FALSE"]

print("Summary of the True/False statements and their conclusions:")
print("")
for i, (statement, conclusion) in enumerate(zip(statements, conclusions)):
    print(f"{i+1}. {statement}")
    print(f"   Conclusion: {conclusion}")
    print("")

# Create a summary table visualization
plt.figure(figsize=(12, 8))
table_data = [
    ["Statement 1", "TRUE", "MAP = Mean for symmetric unimodal posteriors"],
    ["Statement 2", "FALSE", "BMA can sometimes perform worse with imperfect models or finite data"],
    ["Statement 3", "TRUE", "Influence of prior diminishes with increasing data (prior washing out)"],
    ["Statement 4", "TRUE", "BIC was designed to approximate log marginal likelihood, AIC wasn't"],
    ["Statement 5", "FALSE", "VI limited by the expressiveness of the approximating family"]
]

# Create a table plot
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

table = plt.table(cellText=table_data,
                  colLabels=["Statement", "Conclusion", "Key Insight"],
                  colWidths=[0.15, 0.15, 0.7],
                  loc='center',
                  cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Colorize conclusions
for i in range(5):
    table[(i+1, 1)].set_facecolor('lightgreen' if table_data[i][1] == "TRUE" else "lightcoral")

plt.title('Summary of MAP and Bayesian Inference Concepts', fontsize=16, y=0.8)
plt.tight_layout()

file_path = os.path.join(save_dir, "statement_summary.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("\nAll tasks completed. The script has evaluated all five statements with justifications.")
print("Images have been saved to:", save_dir) 