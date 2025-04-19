import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy import stats

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "MAP_Special_Cases")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("\n=== MAP SPECIAL CASES EXAMPLES ===\n")
print("This script demonstrates the special cases of Maximum A Posteriori (MAP) estimation\n"
      "and how MAP behaves under different limiting conditions.")

# Define a function to compute MAP estimate for Gaussian prior and likelihood
def gaussian_map_estimate(prior_mean, prior_var, sample_mean, sample_var, n):
    """
    Compute the MAP estimate for Gaussian prior and likelihood.
    
    Parameters:
    - prior_mean: Mean of the prior distribution
    - prior_var: Variance of the prior distribution
    - sample_mean: Mean of the observed data
    - sample_var: Variance of the observed data
    - n: Number of observations
    
    Returns:
    - MAP estimate
    """
    # Step 1: Calculate the weights for prior mean and sample mean
    prior_weight = 1/prior_var
    data_weight = n/sample_var
    
    # Step 2: Calculate the weighted sum of prior mean and sample mean
    weighted_sum = prior_mean * prior_weight + sample_mean * data_weight
    
    # Step 3: Calculate the sum of weights
    sum_of_weights = prior_weight + data_weight
    
    # Step 4: Divide the weighted sum by the sum of weights to get the MAP estimate
    map_estimate = weighted_sum / sum_of_weights
    
    # Print the detailed steps
    print("\n--- Detailed MAP Estimation Steps ---")
    print(f"Step 1: Calculate weights")
    print(f"  Prior weight = 1/prior_var = 1/{prior_var:.10f} = {prior_weight:.10f}")
    print(f"  Data weight = n/sample_var = {n}/{sample_var:.10f} = {data_weight:.10f}")
    
    print(f"\nStep 2: Calculate weighted sum")
    print(f"  Weighted sum = prior_mean × prior_weight + sample_mean × data_weight")
    print(f"  Weighted sum = {prior_mean:.10f} × {prior_weight:.10f} + {sample_mean:.10f} × {data_weight:.10f}")
    print(f"  Weighted sum = {prior_mean * prior_weight:.10f} + {sample_mean * data_weight:.10f}")
    print(f"  Weighted sum = {weighted_sum:.10f}")
    
    print(f"\nStep 3: Calculate sum of weights")
    print(f"  Sum of weights = prior_weight + data_weight")
    print(f"  Sum of weights = {prior_weight:.10f} + {data_weight:.10f}")
    print(f"  Sum of weights = {sum_of_weights:.10f}")
    
    print(f"\nStep 4: Calculate MAP estimate")
    print(f"  MAP estimate = weighted_sum / sum_of_weights")
    print(f"  MAP estimate = {weighted_sum:.10f} / {sum_of_weights:.10f}")
    print(f"  MAP estimate = {map_estimate:.10f}")
    
    # Explain interpretation
    print("\nInterpretation:")
    if prior_weight > data_weight:
        print(f"  The prior weight ({prior_weight:.4f}) is greater than the data weight ({data_weight:.4f}),")
        print(f"  so the MAP estimate ({map_estimate:.4f}) is closer to the prior mean ({prior_mean:.4f}).")
    elif data_weight > prior_weight:
        print(f"  The data weight ({data_weight:.4f}) is greater than the prior weight ({prior_weight:.4f}),")
        print(f"  so the MAP estimate ({map_estimate:.4f}) is closer to the sample mean ({sample_mean:.4f}).")
    else:
        print(f"  The prior weight ({prior_weight:.4f}) equals the data weight ({data_weight:.4f}),")
        print(f"  so the MAP estimate ({map_estimate:.4f}) is exactly halfway between")
        print(f"  the prior mean ({prior_mean:.4f}) and the sample mean ({sample_mean:.4f}).")
        
    print("-----------------------------------\n")
    
    return map_estimate

# Function to plot prior, likelihood, and posterior
def plot_distributions(prior_mean, prior_var, sample_mean, sample_var, n, map_estimate, title, filename, special_case=None):
    """
    Plot the prior, likelihood, and posterior distributions.
    
    Parameters:
    - prior_mean, prior_var: Prior distribution parameters
    - sample_mean, sample_var: Likelihood parameters
    - n: Number of observations
    - map_estimate: Calculated MAP estimate
    - title: Plot title
    - filename: Filename to save the plot
    - special_case: Special condition to handle (e.g., 'no_prior', 'perfect_prior')
    """
    plt.figure(figsize=(12, 8))
    
    # Define the range for x-axis
    if special_case == 'no_prior':
        # For flat prior
        x = np.linspace(sample_mean - 4*np.sqrt(sample_var/n), 
                        sample_mean + 4*np.sqrt(sample_var/n), 1000)
    elif special_case == 'perfect_prior':
        # For perfect prior (very narrow)
        x = np.linspace(prior_mean - 0.5, prior_mean + 0.5, 1000)
    else:
        # Standard case
        x_min = min(prior_mean - 3*np.sqrt(prior_var), sample_mean - 3*np.sqrt(sample_var/n))
        x_max = max(prior_mean + 3*np.sqrt(prior_var), sample_mean + 3*np.sqrt(sample_var/n))
        x = np.linspace(x_min, x_max, 1000)
    
    # Calculate distributions
    if special_case == 'no_prior':
        # Flat prior (improper)
        prior = np.ones_like(x) / (max(x) - min(x))  # Uniform over the range
        likelihood = stats.norm.pdf(x, sample_mean, np.sqrt(sample_var/n))
        posterior = likelihood.copy()  # Posterior proportional to likelihood with flat prior
        
        # Print detailed calculations
        print(f"\n--- Detailed Calculation for {special_case} case ---")
        print("For a flat (uninformative) prior with infinite variance:")
        print("  p(θ) ∝ constant (flat line)")
        print(f"  Likelihood: p(x|θ) = Normal(θ; μ={sample_mean:.4f}, σ²={sample_var/n:.4f})")
        print("  Posterior: p(θ|x) ∝ p(x|θ) × p(θ) ∝ p(x|θ) ∝ Likelihood")
        print("  Thus, posterior is proportional to the likelihood")
        print(f"  MAP estimate = MLE = {sample_mean:.4f}")
        print("--------------------------------------------------\n")
    
    elif special_case == 'perfect_prior':
        # Almost perfect prior (very narrow)
        prior = stats.norm.pdf(x, prior_mean, np.sqrt(0.001))  # Very small variance
        likelihood = stats.norm.pdf(x, sample_mean, np.sqrt(sample_var/n))
        # Posterior will be almost identical to the prior due to the very small prior variance
        posterior_var = 1 / (1/0.001 + n/sample_var)
        posterior_mean = (prior_mean/0.001 + n*sample_mean/sample_var) / (1/0.001 + n/sample_var)
        posterior = stats.norm.pdf(x, posterior_mean, np.sqrt(posterior_var))
        
        # Print detailed calculations
        print(f"\n--- Detailed Calculation for {special_case} case ---")
        print("For a nearly perfect prior with extremely small variance:")
        print(f"  Prior: p(θ) = Normal(θ; μ={prior_mean:.4f}, σ²=0.001)")
        print(f"  Likelihood: p(x|θ) = Normal(θ; μ={sample_mean:.4f}, σ²={sample_var/n:.4f})")
        print("  Posterior variance = 1 / (1/prior_var + n/sample_var)")
        print(f"  Posterior variance = 1 / (1/0.001 + {n}/{sample_var:.4f})")
        print(f"  Posterior variance = {posterior_var:.10f}")
        print("  As prior_var → 0, posterior mean → prior mean")
        print(f"  MAP estimate ≈ prior mean = {prior_mean:.4f}")
        print("--------------------------------------------------\n")
        
    else:
        # Standard case
        prior = stats.norm.pdf(x, prior_mean, np.sqrt(prior_var))
        likelihood = stats.norm.pdf(x, sample_mean, np.sqrt(sample_var/n))
        # Posterior variance
        posterior_var = 1 / (1/prior_var + n/sample_var)
        # Posterior mean
        posterior_mean = (prior_mean/prior_var + n*sample_mean/sample_var) / (1/prior_var + n/sample_var)
        posterior = stats.norm.pdf(x, posterior_mean, np.sqrt(posterior_var))
        
        # Print detailed calculations if not already done
        if special_case:
            print(f"\n--- Detailed Calculation for {special_case} case ---")
            print(f"  Prior: p(θ) = Normal(θ; μ={prior_mean:.4f}, σ²={prior_var:.4f})")
            print(f"  Likelihood: p(x|θ) = Normal(θ; μ={sample_mean:.4f}, σ²={sample_var/n:.4f})")
            print("  Posterior variance = 1 / (1/prior_var + n/sample_var)")
            print(f"  Posterior variance = 1 / (1/{prior_var:.4f} + {n}/{sample_var:.4f})")
            print(f"  Posterior variance = {posterior_var:.10f}")
            print("  Posterior mean = (prior_mean/prior_var + n*sample_mean/sample_var) / (1/prior_var + n/sample_var)")
            print(f"  Posterior mean = ({prior_mean:.4f}/{prior_var:.4f} + {n}*{sample_mean:.4f}/{sample_var:.4f}) / (1/{prior_var:.4f} + {n}/{sample_var:.4f})")
            print(f"  Posterior mean = {posterior_mean:.10f}")
            print(f"  MAP estimate = {map_estimate:.10f}")
            print("--------------------------------------------------\n")
    
    # Scale distributions for better visualization
    prior = prior / np.max(prior)
    likelihood = likelihood / np.max(likelihood)
    posterior = posterior / np.max(posterior)
    
    # Plot the distributions
    plt.plot(x, prior, 'b-', label='Prior')
    plt.plot(x, likelihood, 'r-', label='Likelihood')
    plt.plot(x, posterior, 'g-', label='Posterior')
    
    # Add vertical lines for key values
    plt.axvline(x=prior_mean, color='b', linestyle='--', alpha=0.5, label='Prior Mean')
    plt.axvline(x=sample_mean, color='r', linestyle='--', alpha=0.5, label='Sample Mean (MLE)')
    plt.axvline(x=map_estimate, color='g', linestyle='-', linewidth=2, label='MAP Estimate')
    
    plt.title(title)
    plt.xlabel('θ')
    plt.ylabel('Normalized Density')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add annotations
    plt.annotate(f'MAP Estimate = {map_estimate:.4f}', 
                xy=(map_estimate, 0.2), 
                xytext=(map_estimate + 0.5, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), dpi=100)
    plt.close()

# Special Case 1: No Prior Knowledge
print("\n=== Special Case 1: No Prior Knowledge ===")
print("When we have no meaningful prior knowledge, we can use an uninformative or 'flat' prior.")
print("As the prior variance approaches infinity, the MAP estimate approaches the MLE.")

# Example setup
example1_measurements = [23.2, 22.8, 23.5, 23.1, 23.4]
n1 = len(example1_measurements)
sample_mean1 = np.mean(example1_measurements)
sample_var1 = np.var(example1_measurements)

print("\nProblem Statement:")
print(f"We have temperature measurements: {example1_measurements}")
print("We want to estimate the true temperature with no prior knowledge.")

print("\nStep 1: Calculate the sample statistics")
print(f"  Sample size: n = {n1}")
print(f"  Sample mean (MLE): x̄ = ({' + '.join(map(str, example1_measurements))}) / {n1} = {sample_mean1:.2f}°C")
print(f"  Sample variance: s² = {sample_var1:.4f}°C²")

# Prior parameters (uninformative prior)
prior_mean1 = 0  # Arbitrary value as prior is flat
prior_var1 = 1e10  # Very large variance to represent flat prior

print("\nStep 2: Define an uninformative prior")
print(f"  Prior mean: μ₀ = {prior_mean1} (arbitrary since the prior is flat)")
print(f"  Prior variance: σ₀² = {prior_var1} (very large to represent no prior knowledge)")

# Calculate MAP estimate
map_estimate1 = gaussian_map_estimate(prior_mean1, prior_var1, sample_mean1, sample_var1, n1)

print("\nStep 3: Interpret the result")
print(f"  As the prior variance approaches infinity, the MAP estimate converges to the MLE")
print(f"  MAP estimate: θ̂_MAP = {map_estimate1:.4f}°C")
print(f"  MLE: θ̂_MLE = {sample_mean1:.4f}°C")
print(f"  The difference is: {abs(map_estimate1 - sample_mean1):.10f}°C (effectively zero)")

# Plot the distributions
plot_distributions(prior_mean1, prior_var1, sample_mean1, sample_var1, n1, 
                  map_estimate1, 
                  "Special Case 1: No Prior Knowledge (Flat Prior)",
                  "map_special_case_no_prior.png",
                  special_case="no_prior")

# Special Case 2: Perfect Prior Knowledge
print("\n=== Special Case 2: Perfect Prior Knowledge ===")
print("When we have complete certainty in our prior knowledge, the prior variance approaches zero.")
print("In this case, the MAP estimate ignores the data completely and equals the prior mean.")

# Example setup
example2_measurements = [0.02, -0.01, 0.03, -0.02, 0.01]  # Measurements from calibrated device
n2 = len(example2_measurements)
sample_mean2 = np.mean(example2_measurements)
sample_var2 = np.var(example2_measurements)

print("\nProblem Statement:")
print(f"A calibrated device should have zero measurement bias, but shows these readings: {example2_measurements}")
print("We are certain the true bias is zero, and want to estimate it using MAP.")

print("\nStep 1: Calculate the sample statistics")
print(f"  Sample size: n = {n2}")
print(f"  Sample mean (MLE): x̄ = ({' + '.join(map(str, example2_measurements))}) / {n2} = {sample_mean2:.4f}")
print(f"  Sample variance: s² = {sample_var2:.6f}")

# Prior parameters (certain prior)
prior_mean2 = 0  # Known bias is exactly 0
prior_var2 = 1e-10  # Very small variance to represent certainty

print("\nStep 2: Define a precise prior based on calibration knowledge")
print(f"  Prior mean: μ₀ = {prior_mean2} (we're certain the true bias is zero)")
print(f"  Prior variance: σ₀² = {prior_var2} (extremely small to represent certainty)")

# Calculate MAP estimate
map_estimate2 = gaussian_map_estimate(prior_mean2, prior_var2, sample_mean2, sample_var2, n2)

print("\nStep 3: Interpret the result")
print(f"  As the prior variance approaches zero, the MAP estimate converges to the prior mean")
print(f"  MAP estimate: θ̂_MAP = {map_estimate2:.10f}")
print(f"  Prior mean: μ₀ = {prior_mean2}")
print(f"  The difference is: {abs(map_estimate2 - prior_mean2):.10f} (effectively zero)")
print(f"  Despite the sample mean being {sample_mean2}, our extreme certainty in the prior")
print(f"  causes the MAP estimate to effectively ignore the data.")

# Plot the distributions
plot_distributions(prior_mean2, prior_var2, sample_mean2, sample_var2, n2, 
                  map_estimate2, 
                  "Special Case 2: Perfect Prior Knowledge",
                  "map_special_case_perfect_prior.png",
                  special_case="perfect_prior")

# Special Case 3: Equal Confidence
print("\n=== Special Case 3: Equal Confidence ===")
print("When we have equal confidence in our prior knowledge and each data point,")
print("the MAP estimate becomes a weighted average of the prior mean and sample mean.")

# Example setup
student_scores = [85, 82, 88]
n3 = len(student_scores)
sample_mean3 = np.mean(student_scores)
sample_var3 = 25  # Assume known variance for simplicity

print("\nProblem Statement:")
print(f"A student has test scores {student_scores}, and the school average is 75%.")
print("Assuming equal confidence in the prior and each data point, what is our best estimate")
print("of the student's true ability?")

print("\nStep 1: Calculate the sample statistics")
print(f"  Sample size: n = {n3}")
print(f"  Sample mean (MLE): x̄ = ({' + '.join(map(str, student_scores))}) / {n3} = {sample_mean3:.2f}%")
print(f"  Sample variance: s² = {sample_var3} (assumed known)")

# Prior parameters (equal confidence)
prior_mean3 = 75  # School-wide average
prior_var3 = sample_var3  # Equal confidence: prior_var = sample_var

print("\nStep 2: Define a prior representing the school average")
print(f"  Prior mean: μ₀ = {prior_mean3}% (school average)")
print(f"  Prior variance: σ₀² = {prior_var3} (equal to sample variance for equal confidence)")

# Calculate MAP estimate
map_estimate3 = gaussian_map_estimate(prior_mean3, prior_var3, sample_mean3, sample_var3, n3)

print("\nStep 3: Verify using the special formula for equal confidence")
equal_confidence_map = (prior_mean3 + n3*sample_mean3) / (1 + n3)
print(f"  For equal confidence, we can use: θ̂_MAP = (μ₀ + n⋅x̄)/(1 + n)")
print(f"  θ̂_MAP = ({prior_mean3} + {n3}⋅{sample_mean3:.2f})/(1 + {n3})")
print(f"  θ̂_MAP = ({prior_mean3} + {n3*sample_mean3:.2f})/{1+n3}")
print(f"  θ̂_MAP = {prior_mean3 + n3*sample_mean3:.2f}/{1+n3}")
print(f"  θ̂_MAP = {equal_confidence_map:.2f}%")
print(f"  This matches our general MAP formula result: {map_estimate3:.2f}%")

print("\nStep 4: Interpret the result")
print(f"  With equal confidence, the MAP estimate gives weight 1 to the prior")
print(f"  and weight 1 to each data point, resulting in:")
print(f"  θ̂_MAP = (75 + 85 + 82 + 88)/4 = {(75 + 85 + 82 + 88)/4:.2f}%")
print(f"  This is equivalent to treating the prior as one extra data point.")

# Plot the distributions
plot_distributions(prior_mean3, prior_var3, sample_mean3, sample_var3, n3, 
                  map_estimate3, 
                  "Special Case 3: Equal Confidence",
                  "map_special_case_equal_confidence.png",
                  special_case="equal_confidence")

# Special Case 4: Large Sample Size
print("\n=== Special Case 4: Large Sample Size ===")
print("As the sample size becomes very large, the influence of the prior diminishes,")
print("and the MAP estimate approaches the MLE.")

# Example setup
n4 = 1000
defects = 30
sample_mean4 = defects / n4  # 3% defect rate
sample_var4 = sample_mean4 * (1 - sample_mean4)  # Variance for binomial proportion

print("\nProblem Statement:")
print(f"A manufacturing process historically had a 5% defect rate. In a recent")
print(f"quality check of {n4} products, {defects} defects were found.")
print("What is our updated estimate of the defect rate?")

print("\nStep 1: Calculate the sample statistics")
print(f"  Sample size: n = {n4} (large sample)")
print(f"  Defects observed: {defects}")
print(f"  Sample defect rate (MLE): p̂ = {defects}/{n4} = {sample_mean4:.4f} ({sample_mean4*100:.1f}%)")
print(f"  Sample variance: s² = p̂(1-p̂) = {sample_mean4:.4f}×{1-sample_mean4:.4f} = {sample_var4:.6f}")

# Prior parameters
prior_mean4 = 0.05  # Historical 5% defect rate
prior_var4 = 0.001  # Some uncertainty in the prior

print("\nStep 2: Define a prior based on historical data")
print(f"  Prior mean: μ₀ = {prior_mean4:.2f} (historical 5% defect rate)")
print(f"  Prior variance: σ₀² = {prior_var4:.3f} (some uncertainty in the prior)")

# Calculate MAP estimate
map_estimate4 = gaussian_map_estimate(prior_mean4, prior_var4, sample_mean4, sample_var4, n4)

print("\nStep 3: Analyze the impact of large sample size")
# Calculate the relative weights
prior_weight = 1/prior_var4
data_weight = n4/sample_var4
total_weight = prior_weight + data_weight
prior_relative_weight = prior_weight / total_weight
data_relative_weight = data_weight / total_weight

print(f"  Prior weight: 1/σ₀² = 1/{prior_var4:.3f} = {prior_weight:.1f}")
print(f"  Data weight: n/σ² = {n4}/{sample_var4:.6f} = {data_weight:.1f}")
print(f"  Relative prior influence: {prior_weight:.1f}/({prior_weight:.1f}+{data_weight:.1f}) = {prior_relative_weight:.4f} ({prior_relative_weight*100:.1f}%)")
print(f"  Relative data influence: {data_weight:.1f}/({prior_weight:.1f}+{data_weight:.1f}) = {data_relative_weight:.4f} ({data_relative_weight*100:.1f}%)")

print("\nStep 4: Interpret the result")
print(f"  MAP estimate: θ̂_MAP = {map_estimate4:.4f} ({map_estimate4*100:.1f}%)")
print(f"  MLE (sample defect rate): p̂ = {sample_mean4:.4f} ({sample_mean4*100:.1f}%)")
print(f"  Prior mean: μ₀ = {prior_mean4:.4f} ({prior_mean4*100:.1f}%)")
print(f"  With {n4} samples, the data has {data_relative_weight*100:.1f}% of the influence,")
print(f"  causing the MAP estimate to be much closer to the MLE than to the prior mean.")
print(f"  As n → ∞, the MAP estimate would converge exactly to the MLE.")

# Plot the distributions
plot_distributions(prior_mean4, prior_var4, sample_mean4, sample_var4, n4, 
                  map_estimate4, 
                  "Special Case 4: Large Sample Size",
                  "map_special_case_large_sample.png",
                  special_case="large_sample")

# Special Case 5: Conflicting Information
print("\n=== Special Case 5: Conflicting Information ===")
print("When prior knowledge strongly conflicts with observed data,")
print("the MAP estimate balances the two based on their relative variances.")

# Example setup - Medical Diagnostic Test
# Prior: disease prevalence is 1%
prior_mean5 = 0.01
prior_var5 = 0.0001  # Small variance to indicate confidence in prevalence

# Likelihood: test says positive with 95% accuracy
# Simplifying by using a direct calculation for this special case
test_accuracy = 0.95
false_positive_rate = 0.05

print("\nProblem Statement:")
print(f"A disease has a known prevalence of {prior_mean5*100:.1f}% in the population.")
print(f"A diagnostic test with {test_accuracy*100:.1f}% accuracy shows a positive result.")
print("What is the probability that the person actually has the disease?")

print("\nStep 1: Define the prior probability and test characteristics")
print(f"  Prior probability (prevalence): p(disease) = {prior_mean5:.4f} ({prior_mean5*100:.1f}%)")
print(f"  Test sensitivity (true positive rate): p(positive|disease) = {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"  Test specificity: p(negative|no disease) = {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"  False positive rate: p(positive|no disease) = {false_positive_rate:.4f} ({false_positive_rate*100:.1f}%)")

# Calculating the probability using Bayes' theorem
p_disease_given_positive = (prior_mean5 * test_accuracy) / (prior_mean5 * test_accuracy + (1 - prior_mean5) * false_positive_rate)

print("\nStep 2: Apply Bayes' theorem to find the posterior probability")
print("  p(disease|positive) = p(positive|disease) × p(disease) / p(positive)")
print("  p(positive) = p(positive|disease) × p(disease) + p(positive|no disease) × p(no disease)")
print(f"  p(positive) = {test_accuracy:.4f} × {prior_mean5:.4f} + {false_positive_rate:.4f} × {1-prior_mean5:.4f}")
print(f"  p(positive) = {test_accuracy * prior_mean5:.6f} + {false_positive_rate * (1-prior_mean5):.6f}")
print(f"  p(positive) = {test_accuracy * prior_mean5 + false_positive_rate * (1-prior_mean5):.6f}")

print(f"\n  p(disease|positive) = ({test_accuracy:.4f} × {prior_mean5:.4f}) / {test_accuracy * prior_mean5 + false_positive_rate * (1-prior_mean5):.6f}")
print(f"  p(disease|positive) = {test_accuracy * prior_mean5:.6f} / {test_accuracy * prior_mean5 + false_positive_rate * (1-prior_mean5):.6f}")
print(f"  p(disease|positive) = {p_disease_given_positive:.4f} ({p_disease_given_positive*100:.1f}%)")

print("\nStep 3: Interpret the result")
print(f"  Despite the test being {test_accuracy*100:.1f}% accurate, the probability of disease given")
print(f"  a positive result is only {p_disease_given_positive*100:.1f}% due to the low prevalence ({prior_mean5*100:.1f}%).")
print("  This is a classic example of Bayes' theorem resolving apparently conflicting information:")
print("  - The prevalence suggests the disease is very rare")
print("  - The positive test suggests the disease is present")
print("  - The MAP estimate (posterior probability) balances these, accounting for false positives")

# For visualization purposes, we'll create a simplified model with Gaussian approximation
sample_mean5 = 0.95  # Test says 95% chance
sample_var5 = 0.01   # Some uncertainty in the test
n5 = 1  # Just one test result

# Calculate an approximate MAP estimate for visualization
map_estimate5 = gaussian_map_estimate(prior_mean5, prior_var5, sample_mean5, sample_var5, n5)

# Plot the distributions - note this is a visualization approximation
plot_distributions(prior_mean5, prior_var5, sample_mean5, sample_var5, n5, 
                  p_disease_given_positive,  # Using the exact Bayes result for the marker 
                  "Special Case 5: Conflicting Information",
                  "map_special_case_conflicting.png",
                  special_case="conflicting")

# Additional Example: Beta-Bernoulli MAP with different priors
print("\n=== Beta-Bernoulli MAP Estimation ===")
print("Demonstrating MAP estimation with Beta priors for a Bernoulli likelihood.")

# Function to compute Beta-Bernoulli MAP estimate
def beta_bernoulli_map(alpha, beta, heads, flips):
    """
    Compute the MAP estimate for Beta prior and Bernoulli likelihood.
    
    Parameters:
    - alpha, beta: Parameters of the Beta prior
    - heads: Number of successes (heads)
    - flips: Total number of trials (coin flips)
    
    Returns:
    - MAP estimate
    """
    print(f"\n--- Beta-Bernoulli MAP Calculation (α={alpha}, β={beta}) ---")
    print("Step 1: Identify the conjugate relationship")
    print("  Prior: Beta(α, β)")
    print("  Likelihood: Bernoulli(θ) for each coin flip")
    print("  Posterior: Beta(α + heads, β + tails)")
    
    print("\nStep 2: Calculate the posterior parameters")
    posterior_alpha = alpha + heads
    posterior_beta = beta + flips - heads
    print(f"  Posterior parameters: Beta({posterior_alpha}, {posterior_beta})")
    
    print("\nStep 3: Find the mode of the posterior (MAP estimate)")
    # Handle special cases for improper priors
    if alpha <= 1 and beta <= 1:
        if heads == 0:
            map_estimate = 0
            print("  Special case: With α≤1, β≤1 and heads=0, MAP = 0")
        elif heads == flips:
            map_estimate = 1
            print("  Special case: With α≤1, β≤1 and heads=flips, MAP = 1")
        else:
            map_estimate = heads / flips
            print(f"  Special case: With α≤1, β≤1, MAP = MLE = heads/flips = {heads}/{flips} = {map_estimate:.4f}")
    else:
        # Formula for the mode of a Beta distribution
        map_estimate = (heads + alpha - 1) / (flips + alpha + beta - 2)
        print("  For Beta(a,b) distribution, the mode is (a-1)/(a+b-2) for a,b > 1")
        print(f"  MAP = (heads + α - 1)/(flips + α + β - 2)")
        print(f"  MAP = ({heads} + {alpha} - 1)/({flips} + {alpha} + {beta} - 2)")
        print(f"  MAP = ({heads + alpha - 1:.1f})/({flips + alpha + beta - 2:.1f})")
        print(f"  MAP = {map_estimate:.4f}")
    
    print("-------------------------------------------------------")
    return map_estimate

# Example: 8 heads in 10 flips
heads = 8
flips = 10
print("\nProblem Statement:")
print(f"A coin is flipped {flips} times, resulting in {heads} heads.")
print("We want to estimate the probability of heads using MAP with different priors.")

print("\nStep 1: Calculate the MLE (Maximum Likelihood Estimate)")
mle = heads / flips
print(f"  MLE = heads/flips = {heads}/{flips} = {mle:.4f}")

# Case 1: Jeffreys prior (Beta(0.5, 0.5))
print("\nStep 2: Apply Jeffreys prior - Beta(0.5, 0.5)")
print("  Jeffreys prior is a non-informative prior that is invariant to reparameterization")
print("  For the Bernoulli model, Jeffreys prior is Beta(0.5, 0.5)")
alpha1, beta1 = 0.5, 0.5
map_jeffrey = beta_bernoulli_map(alpha1, beta1, heads, flips)

# Case 2: Uniform prior (Beta(1, 1))
print("\nStep 3: Apply Uniform prior - Beta(1, 1)")
print("  Uniform prior represents complete uncertainty about θ - all values equally likely")
print("  For the Bernoulli model, Uniform prior is Beta(1, 1)")
alpha2, beta2 = 1, 1
map_uniform = beta_bernoulli_map(alpha2, beta2, heads, flips)

# Case 3: Reference prior (Beta(0, 0))
print("\nStep 4: Apply Reference prior - Beta(0, 0)")
print("  Reference prior is an improper prior sometimes used in Bayesian analysis")
print("  For the Bernoulli model, Reference prior can be represented as Beta(0, 0)")
print("  Note: This is an improper prior as the Beta distribution requires α,β > 0")
alpha3, beta3 = 0, 0
map_reference = beta_bernoulli_map(alpha3, beta3, heads, flips)

print("\nStep 5: Compare the different MAP estimates")
print(f"  MLE: {mle:.4f}")
print(f"  MAP with Jeffreys prior: {map_jeffrey:.4f}")
print(f"  MAP with Uniform prior: {map_uniform:.4f}")
print(f"  MAP with Reference prior: {map_reference:.4f}")

print("\nObservation:")
print(f"  In this particular case ({heads} heads in {flips} flips), all methods give similar results")
print("  because there is enough data to overwhelm the priors.")
print("  With smaller samples or more extreme proportions, the differences would be more noticeable.")

# Plot Beta-Bernoulli MAP with different priors
plt.figure(figsize=(14, 8))

# Values for theta
theta = np.linspace(0, 1, 1000)

# Calculate the likelihood (proportional to theta^s * (1-theta)^(n-s))
likelihood = stats.beta.pdf(theta, heads + 1, flips - heads + 1)
likelihood = likelihood / np.max(likelihood)

# Calculate posteriors for different priors
posterior_jeffrey = stats.beta.pdf(theta, heads + alpha1, flips - heads + beta1)
posterior_jeffrey = posterior_jeffrey / np.max(posterior_jeffrey)

posterior_uniform = stats.beta.pdf(theta, heads + alpha2, flips - heads + beta2)
posterior_uniform = posterior_uniform / np.max(posterior_uniform)

# Calculate the priors
prior_jeffrey = stats.beta.pdf(theta, alpha1, beta1)
prior_jeffrey = prior_jeffrey / np.max(prior_jeffrey)

prior_uniform = stats.beta.pdf(theta, alpha2, beta2)
prior_uniform = prior_uniform / np.max(prior_uniform)

# Plot everything
plt.plot(theta, likelihood, 'r-', label='Likelihood')
plt.plot(theta, prior_jeffrey, 'b--', label='Jeffreys Prior (Beta(0.5, 0.5))')
plt.plot(theta, prior_uniform, 'g--', label='Uniform Prior (Beta(1, 1))')
plt.plot(theta, posterior_jeffrey, 'b-', label='Posterior with Jeffreys Prior')
plt.plot(theta, posterior_uniform, 'g-', label='Posterior with Uniform Prior')

# Add vertical lines for MAP estimates
plt.axvline(x=map_jeffrey, color='b', linestyle='-', linewidth=2)
plt.axvline(x=map_uniform, color='g', linestyle='-', linewidth=2)
plt.axvline(x=heads/flips, color='r', linestyle='--', label='MLE')

plt.title('Beta-Bernoulli MAP Estimation with Different Priors')
plt.xlabel('θ (Probability of Heads)')
plt.ylabel('Normalized Density')
plt.legend()
plt.grid(alpha=0.3)

# Add annotations
plt.annotate(f'MAP with Jeffreys Prior = {map_jeffrey:.4f}', 
            xy=(map_jeffrey, 0.5), 
            xytext=(map_jeffrey + 0.1, 0.6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10)

plt.annotate(f'MAP with Uniform Prior = {map_uniform:.4f}\n(Same as MLE)', 
            xy=(map_uniform, 0.4), 
            xytext=(map_uniform - 0.4, 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'beta_bernoulli_map.png'), dpi=100)
plt.close()

print("\n=== KEY INSIGHTS ===")
print("\nTheoretical Insights:")
print("- MAP estimation bridges the gap between purely data-driven estimation (MLE) and purely prior-based estimation")
print("- The relative influence of prior and data depends on their respective variances")
print("- As sample size grows, MAP converges to MLE regardless of prior")
print("- With proper conjugate priors, MAP estimation often has closed-form solutions")
print("- The MAP estimate can be viewed as a regularized version of the MLE, where the prior acts as a regularizer")

print("\nPractical Applications:")
print("- Understanding when to rely more on prior knowledge versus observed data")
print("- Explaining why initial MAP estimates may differ significantly from MLEs with small samples")
print("- Recognizing when the choice of prior becomes irrelevant")
print("- Using MAP to reduce estimation variance in small sample scenarios")
print("- Incorporating domain knowledge systematically into statistical estimation")

print("\nCommon Pitfalls:")
print("- Using MAP point estimates without considering the full posterior distribution")
print("- Forgetting that non-informative priors can still influence MAP estimates")
print("- Assuming convergence to MLE without checking sample size adequacy")
print("- Overconfidence in priors leading to rejection of surprising but valid data")
print("- Using improper priors without understanding their implications")

print("\nAll example visualizations have been saved in the Images directory.")
print("This completes the demonstrations of MAP special cases.") 