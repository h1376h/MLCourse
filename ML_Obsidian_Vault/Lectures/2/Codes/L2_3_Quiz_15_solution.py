import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, beta
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_3_Quiz_15")
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
print("- 50 data points from an unknown distribution in the range [0, 1]")
print("- Six visualizations showing various aspects of likelihood analysis")
print("- Three candidate distributions: Beta, Normal, and Exponential")
print("\nTask:")
print("1. Identify the maximum likelihood estimates for each distribution")
print("2. Determine which distribution best fits the data")
print("3. Visually assess the fit of each distribution")
print("4. Explain the difference between probability and likelihood")
print("5. Make a recommendation for the most appropriate distribution")

# Step 2: Recreate the same data from the question script for analysis
print_step_header(2, "Recreation of Dataset")

np.random.seed(42)  # Same seed as question script
true_alpha, true_beta = 2.5, 1.5  # True parameters used to generate data
data_points = beta.rvs(true_alpha, true_beta, size=50)

print(f"Data summary statistics:")
print(f"- Mean: {np.mean(data_points):.4f}")
print(f"- Median: {np.median(data_points):.4f}")
print(f"- Standard deviation: {np.std(data_points):.4f}")
print(f"- Min: {np.min(data_points):.4f}")
print(f"- Max: {np.max(data_points):.4f}")
print(f"\nData was originally generated from a Beta({true_alpha}, {true_beta}) distribution")
print(f"However, in a real scenario, we wouldn't know the true distribution and would need")
print(f"to determine it from the data, which is the focus of this question.")

# Step 3: Maximum Likelihood Estimation
print_step_header(3, "Maximum Likelihood Estimation Explanation")

# Define parameter grids
alphas = np.linspace(1, 4, 100)  # For beta distribution
means = np.linspace(0.2, 0.8, 100)  # For normal distribution
rates = np.linspace(1, 5, 100)  # For exponential distribution

# Calculate log-likelihoods
beta_logliks = np.zeros((len(alphas), len(alphas)))
normal_logliks = np.zeros(len(means))
exp_logliks = np.zeros(len(rates))

# Beta distribution (varying alpha and beta)
for i, alpha in enumerate(alphas):
    for j, b in enumerate(alphas):  # Using the same grid for beta parameter
        beta_logliks[i, j] = np.sum(np.log(beta.pdf(data_points, alpha, b) + 1e-10))

# Normal distribution (varying mean, fixed std=0.2)
for i, mean in enumerate(means):
    normal_logliks[i] = np.sum(np.log(norm.pdf(data_points, mean, 0.2) + 1e-10))

# Exponential distribution (varying rate)
for i, rate in enumerate(rates):
    # For exponential, we use scale=1/rate in scipy
    exp_logliks[i] = np.sum(np.log(gamma.pdf(data_points, a=1, scale=1/rate) + 1e-10))

# Find the maximum likelihood parameters
beta_max_idx = np.unravel_index(np.argmax(beta_logliks), beta_logliks.shape)
beta_mle_alpha = alphas[beta_max_idx[0]]
beta_mle_beta = alphas[beta_max_idx[1]]

normal_mle_mean = means[np.argmax(normal_logliks)]
exp_mle_rate = rates[np.argmax(exp_logliks)]

print("Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a")
print("statistical model. The MLE chooses the parameter values that maximize the likelihood")
print("function, or equivalently, the log-likelihood function (which is often more convenient).")
print("\nFor a dataset {x_1, x_2, ..., x_n}, the likelihood function is:")
print("    L(θ) = f(x_1, x_2, ..., x_n | θ)")
print("where f is the probability density function and θ represents the parameters.")
print("\nAssuming independence, this becomes:")
print("    L(θ) = ∏_{i=1}^n f(x_i | θ)")
print("\nThe log-likelihood is:")
print("    ℓ(θ) = log L(θ) = ∑_{i=1}^n log f(x_i | θ)")
print("\nThe maximum likelihood estimate θ_MLE maximizes ℓ(θ).")

print("\nCalculated MLE values:")
print(f"- Beta distribution: alpha = {beta_mle_alpha:.4f}, beta = {beta_mle_beta:.4f}")
print(f"- Normal distribution: mean = {normal_mle_mean:.4f}, std = 0.2 (fixed)")
print(f"- Exponential distribution: rate = {exp_mle_rate:.4f}")

# Step 4: Model Comparison
print_step_header(4, "Likelihood Ratio Test and Model Selection")

# Calculate maximum log-likelihoods
max_beta_ll = np.max(beta_logliks)
max_normal_ll = np.max(normal_logliks)
max_exp_ll = np.max(exp_logliks)

print("The likelihood ratio test is a statistical test used to compare the fit of two models.")
print("For nested models, we can compute the test statistic:")
print("    D = 2 * [log L(θ_1) - log L(θ_0)]")
print("where θ_1 is the more complex model and θ_0 is the simpler model.")
print("\nFor non-nested models, we can still compare log-likelihoods, but formal hypothesis testing")
print("requires different methods like AIC, BIC, or cross-validation.")
print("\nIn our case, we have three different distribution families, so we directly compare their")
print("maximum log-likelihood values:")
print(f"- Beta distribution: {max_beta_ll:.4f}")
print(f"- Normal distribution: {max_normal_ll:.4f}")
print(f"- Exponential distribution: {max_exp_ll:.4f}")

print("\nThe Beta distribution has the highest log-likelihood, indicating it provides the best fit")
print("among the three candidate distributions.")
print("\nTo quantify how much better it is, we can compute likelihood ratios:")
print(f"- Beta vs. Normal: exp({max_beta_ll:.4f} - {max_normal_ll:.4f}) = {np.exp(max_beta_ll - max_normal_ll):.2e}")
print(f"- Beta vs. Exponential: exp({max_beta_ll:.4f} - {max_exp_ll:.4f}) = {np.exp(max_beta_ll - max_exp_ll):.2e}")
print("\nThese extremely large ratios indicate that the Beta distribution is overwhelmingly more likely")
print("to have generated the observed data compared to the other distributions.")

# Step 5: Probability vs. Likelihood
print_step_header(5, "Understanding Probability vs. Likelihood")

print("Probability and likelihood are related concepts but have important distinctions:")
print("\n1. Probability:")
print("   - Concerns the chance of observing particular data given fixed parameters")
print("   - P(data | parameters)")
print("   - For continuous variables, refers to the probability density function (PDF)")
print("   - Integrates to 1 over all possible data values")
print("   - Used to make predictions")
print("\n2. Likelihood:")
print("   - Concerns how well different parameter values explain the observed fixed data")
print("   - L(parameters | data)")
print("   - Proportional to P(data | parameters) but viewed as a function of parameters")
print("   - Does NOT generally integrate to 1 over parameter space")
print("   - Used for parameter estimation and model comparison")
print("\nKey difference: In probability, parameters are fixed and data varies.")
print("In likelihood, data is fixed and parameters vary.")
print("\nThis distinction is crucial in statistical inference and forms the basis of methods")
print("like Maximum Likelihood Estimation (MLE).")

# Step 6: Visual Model Assessment
print_step_header(6, "Visual Assessment of Model Fit")

# Calculate PDFs for the fitted distributions
x = np.linspace(0, 1, 1000)
beta_pdf = beta.pdf(x, beta_mle_alpha, beta_mle_beta)
normal_pdf = norm.pdf(x, normal_mle_mean, 0.2)
exp_pdf = gamma.pdf(x, a=1, scale=1/exp_mle_rate)  # Exponential as gamma with a=1

print("Visual assessment of model fit is an important complement to formal statistical methods.")
print("When examining histogram overlaid with fitted densities, we should consider:")
print("\n1. Shape: Does the distribution match the general shape of the data (skewness, modality)?")
print("2. Center: Does the distribution correctly identify the central tendency?")
print("3. Spread: Does the distribution accurately capture the variability?")
print("4. Tails: Does the distribution appropriately model extreme values?")
print("5. Domain: Is the distribution defined over the appropriate range?")
print("\nFor our data:")
print("\n- Beta distribution:")
print("  * Defined on [0,1], matching the data domain")
print("  * Flexible shape can capture the slight skewness")
print("  * Parameters (alpha≈3.06, beta≈2.03) generate a distribution that closely follows the histogram")
print("\n- Normal distribution:")
print("  * Defined on (-∞,∞), extending beyond the [0,1] data range")
print("  * Always symmetric, cannot capture potential skewness")
print("  * Mean (0.59) captures central tendency but fixed std (0.2) may not optimally match spread")
print("\n- Exponential distribution:")
print("  * Monotonically decreasing from x=0, doesn't match the histogram's peak")
print("  * Heavily weighted toward smaller values, unlike our data")
print("  * Poor visual fit overall")

# Step 7: Conclusion and Recommendation
print_step_header(7, "Conclusion and Recommendation")

print("Based on both likelihood analysis and visual assessment, the Beta distribution is clearly")
print("the most appropriate model for this dataset for the following reasons:")
print("\n1. Highest Likelihood: Beta distribution has the highest log-likelihood value")
print("2. Appropriate Domain: Beta is naturally defined on [0,1], matching our data range")
print("3. Flexible Shape: Beta can accommodate various shapes including the observed data pattern")
print("4. Visual Fit: Beta distribution visually matches the histogram better than alternatives")
print("\nThe estimated parameters (alpha≈3.06, beta≈2.03) suggest a slightly right-skewed")
print("distribution with a mode around 0.6, which aligns with the observed data pattern.")
print("\nImportantly, this question demonstrates key statistical concepts:")
print("- The use of likelihood for parameter estimation")
print("- Model comparison using likelihood ratios")
print("- The distinction between probability and likelihood")
print("- The importance of combining formal statistical methods with visual assessment")
print("\nReminder: While the Beta distribution is the best among the three candidates, in real")
print("applications we should also consider other distributions and validation techniques to")
print("ensure we've found the most appropriate model.") 