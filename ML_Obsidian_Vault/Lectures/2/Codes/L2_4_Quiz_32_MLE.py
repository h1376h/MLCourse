import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sp
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def print_latex(latex_str):
    """Print a LaTeX formatted equation as a string."""
    print(f"LaTeX: {latex_str}")

# Define symbolic variables for analytical derivations
theta, x = sp.symbols('theta x', real=True)

# Step 1: Introduction to the problem
print_step_header(1, "Problem Introduction")
print("We need to find the Maximum Likelihood Estimator (MLE) for three probability density functions:")
print_latex(r"(a) f(x;\theta) = \frac{1}{\theta^2}x e^{-x/\theta}, 0 < x < \infty, 0 < \theta < \infty")
print_latex(r"(b) f(x;\theta) = \frac{1}{2\theta^3}x^2 e^{-x/\theta}, 0 < x < \infty, 0 < \theta < \infty")
print_latex(r"(c) f(x;\theta) = \frac{1}{2}e^{-|x-\theta|}, -\infty < x < \infty, -\infty < \theta < \infty")

# Step 2: Solving case (a)
print_step_header(2, "MLE for case (a): f(x;θ) = (1/θ²)x e^(-x/θ)")

# Define the PDF for case (a)
def pdf_a(x, theta):
    return (1/theta**2) * x * np.exp(-x/theta)

# Define log-likelihood function for case (a)
def log_likelihood_a(data, theta):
    if theta <= 0:
        return -np.inf
    return np.sum(np.log(pdf_a(data, theta)))

# Analytical derivation
print("Analytical Derivation:")
print_latex(r"1. \text{The PDF is } f(x;\theta) = \frac{1}{\theta^2}x e^{-x/\theta}")
print_latex(r"2. \text{The likelihood function for } n \text{ samples is:}")
print_latex(r"   L(\theta) = \prod_{i=1}^n \frac{1}{\theta^2}x_i e^{-x_i/\theta}")
print_latex(r"   L(\theta) = \frac{1}{\theta^{2n}}\left(\prod_{i=1}^n x_i\right) e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}")
print_latex(r"3. \text{The log-likelihood function is:}")
print_latex(r"   \ell(\theta) = \ln L(\theta) = \ln\left[\frac{1}{\theta^{2n}}\left(\prod_{i=1}^n x_i\right) e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}\right]")
print_latex(r"   \ell(\theta) = \ln\left[\frac{1}{\theta^{2n}}\right] + \ln\left[\prod_{i=1}^n x_i\right] + \ln\left[e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}\right]")
print_latex(r"   \ell(\theta) = -2n\ln(\theta) + \sum_{i=1}^n \ln(x_i) - \frac{1}{\theta}\sum_{i=1}^n x_i")
print_latex(r"4. \text{Taking the derivative with respect to } \theta \text{:}")
print_latex(r"   \frac{d\ell(\theta)}{d\theta} = -\frac{2n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^n x_i")
print_latex(r"5. \text{Setting the derivative to zero:}")
print_latex(r"   -\frac{2n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^n x_i = 0")
print_latex(r"   \frac{1}{\theta^2}\sum_{i=1}^n x_i = \frac{2n}{\theta}")
print_latex(r"   \sum_{i=1}^n x_i = 2n\theta")
print_latex(r"6. \text{Solving for } \theta \text{:}")
print_latex(r"   \hat{\theta} = \frac{1}{2n}\sum_{i=1}^n x_i = \frac{\bar{x}}{2}  \text{ (where } \bar{x} \text{ is the sample mean)}")
print_latex(r"7. \text{Second derivative to confirm maximum:}")
print_latex(r"   \frac{d^2\ell(\theta)}{d\theta^2} = \frac{2n}{\theta^2} - \frac{2}{\theta^3}\sum_{i=1}^n x_i")
print_latex(r"   \text{At the critical point: } \sum_{i=1}^n x_i = 2n\theta \Rightarrow \frac{d^2\ell(\theta)}{d\theta^2} = \frac{2n}{\theta^2} - \frac{2}{\theta^3}(2n\theta) = \frac{2n}{\theta^2} - \frac{4n}{\theta^2} = -\frac{2n}{\theta^2} < 0")
print_latex(r"   \text{Since the second derivative is negative, this confirms that } \hat{\theta} = \frac{\bar{x}}{2} \text{ is a maximum}")

# Numerical verification for case (a)
np.random.seed(42)
sample_a = np.random.gamma(2, 5, 1000)  # shape=2, scale=5 gives mean=10
true_param_a = 5
sample_mean_a = np.mean(sample_a)
mle_a = sample_mean_a / 2

print("\nNumerical Verification:")
print(f"Generated sample with true θ = {true_param_a}")
print(f"Sample mean = {sample_mean_a:.4f}")
print(f"MLE estimator θ̂ = x̄/2 = {mle_a:.4f}")
print(f"Theoretical expected value of distribution = {2*true_param_a:.4f}")
print(f"Theoretical MLE = {(2*true_param_a)/2:.4f} = {true_param_a:.4f}")

# Detailed numeric calculation explanation
print("\nDetailed Calculation Steps:")
print(f"1. Sample size n = {len(sample_a)}")
print(f"2. Sum of all observations: Σx_i = {np.sum(sample_a):.4f}")
print(f"3. Sample mean: x̄ = (Σx_i)/n = {np.sum(sample_a):.4f}/{len(sample_a)} = {sample_mean_a:.4f}")
print(f"4. MLE: θ̂ = x̄/2 = {sample_mean_a:.4f}/2 = {mle_a:.4f}")

# Calculate log-likelihood at various points to demonstrate maximum
theta_check_points = [mle_a*0.5, mle_a*0.75, mle_a, mle_a*1.25, mle_a*1.5]
print("\nLog-likelihood values at different θ points:")
for theta_val in theta_check_points:
    ll_val = log_likelihood_a(sample_a, theta_val)
    print(f"Log-likelihood at θ = {theta_val:.4f}: {ll_val:.4f}")

# Plotting log-likelihood for case (a)
theta_range_a = np.linspace(2, 8, 100)
ll_values_a = [log_likelihood_a(sample_a, t) for t in theta_range_a]

plt.figure(figsize=(10, 6))
plt.plot(theta_range_a, ll_values_a)
plt.axvline(x=mle_a, color='r', linestyle='--', label=f'MLE: θ̂ = {mle_a:.4f}')
plt.axvline(x=true_param_a, color='g', linestyle=':', label=f'True θ = {true_param_a}')
plt.xlabel('θ')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood function for case (a): Gamma(2,θ)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path_a = os.path.join(save_dir, "case_a_log_likelihood.png")
plt.savefig(file_path_a, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_a}")

# Step 3: Solving case (b)
print_step_header(3, "MLE for case (b): f(x;θ) = (1/2θ³)x² e^(-x/θ)")

# Define the PDF for case (b)
def pdf_b(x, theta):
    return (1/(2*theta**3)) * x**2 * np.exp(-x/theta)

# Define log-likelihood function for case (b)
def log_likelihood_b(data, theta):
    if theta <= 0:
        return -np.inf
    return np.sum(np.log(pdf_b(data, theta)))

# Analytical derivation
print("Analytical Derivation:")
print_latex(r"1. \text{The PDF is } f(x;\theta) = \frac{1}{2\theta^3}x^2 e^{-x/\theta}")
print_latex(r"2. \text{The likelihood function for } n \text{ samples is:}")
print_latex(r"   L(\theta) = \prod_{i=1}^n \frac{1}{2\theta^3}x_i^2 e^{-x_i/\theta}")
print_latex(r"   L(\theta) = \frac{1}{2^n\theta^{3n}}\left(\prod_{i=1}^n x_i^2\right) e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}")
print_latex(r"3. \text{The log-likelihood function is:}")
print_latex(r"   \ell(\theta) = \ln L(\theta) = \ln\left[\frac{1}{2^n\theta^{3n}}\left(\prod_{i=1}^n x_i^2\right) e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}\right]")
print_latex(r"   \ell(\theta) = \ln\left[\frac{1}{2^n\theta^{3n}}\right] + \ln\left[\prod_{i=1}^n x_i^2\right] + \ln\left[e^{-\frac{1}{\theta}\sum_{i=1}^n x_i}\right]")
print_latex(r"   \ell(\theta) = -n\ln(2) - 3n\ln(\theta) + \sum_{i=1}^n \ln(x_i^2) - \frac{1}{\theta}\sum_{i=1}^n x_i")
print_latex(r"   \ell(\theta) = -n\ln(2) - 3n\ln(\theta) + 2\sum_{i=1}^n \ln(x_i) - \frac{1}{\theta}\sum_{i=1}^n x_i")
print_latex(r"4. \text{Taking the derivative with respect to } \theta \text{:}")
print_latex(r"   \frac{d\ell(\theta)}{d\theta} = -\frac{3n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^n x_i")
print_latex(r"5. \text{Setting the derivative to zero:}")
print_latex(r"   -\frac{3n}{\theta} + \frac{1}{\theta^2}\sum_{i=1}^n x_i = 0")
print_latex(r"   \frac{1}{\theta^2}\sum_{i=1}^n x_i = \frac{3n}{\theta}")
print_latex(r"   \sum_{i=1}^n x_i = 3n\theta")
print_latex(r"6. \text{Solving for } \theta \text{:}")
print_latex(r"   \hat{\theta} = \frac{1}{3n}\sum_{i=1}^n x_i = \frac{\bar{x}}{3} \text{ (where } \bar{x} \text{ is the sample mean)}")
print_latex(r"7. \text{Second derivative to confirm maximum:}")
print_latex(r"   \frac{d^2\ell(\theta)}{d\theta^2} = \frac{3n}{\theta^2} - \frac{2}{\theta^3}\sum_{i=1}^n x_i")
print_latex(r"   \text{At the critical point: } \sum_{i=1}^n x_i = 3n\theta \Rightarrow \frac{d^2\ell(\theta)}{d\theta^2} = \frac{3n}{\theta^2} - \frac{2}{\theta^3}(3n\theta) = \frac{3n}{\theta^2} - \frac{6n}{\theta^2} = -\frac{3n}{\theta^2} < 0")
print_latex(r"   \text{Since the second derivative is negative, this confirms that } \hat{\theta} = \frac{\bar{x}}{3} \text{ is a maximum}")

# Numerical verification for case (b)
np.random.seed(43)
sample_b = np.random.gamma(3, 4, 1000)  # shape=3, scale=4 gives mean=12
true_param_b = 4
sample_mean_b = np.mean(sample_b)
mle_b = sample_mean_b / 3

print("\nNumerical Verification:")
print(f"Generated sample with true θ = {true_param_b}")
print(f"Sample mean = {sample_mean_b:.4f}")
print(f"MLE estimator θ̂ = x̄/3 = {mle_b:.4f}")
print(f"Theoretical expected value of distribution = {3*true_param_b:.4f}")
print(f"Theoretical MLE = {(3*true_param_b)/3:.4f} = {true_param_b:.4f}")

# Detailed numeric calculation explanation
print("\nDetailed Calculation Steps:")
print(f"1. Sample size n = {len(sample_b)}")
print(f"2. Sum of all observations: Σx_i = {np.sum(sample_b):.4f}")
print(f"3. Sample mean: x̄ = (Σx_i)/n = {np.sum(sample_b):.4f}/{len(sample_b)} = {sample_mean_b:.4f}")
print(f"4. MLE: θ̂ = x̄/3 = {sample_mean_b:.4f}/3 = {mle_b:.4f}")

# Calculate log-likelihood at various points to demonstrate maximum
theta_check_points_b = [mle_b*0.5, mle_b*0.75, mle_b, mle_b*1.25, mle_b*1.5]
print("\nLog-likelihood values at different θ points:")
for theta_val in theta_check_points_b:
    ll_val = log_likelihood_b(sample_b, theta_val)
    print(f"Log-likelihood at θ = {theta_val:.4f}: {ll_val:.4f}")

# Plotting log-likelihood for case (b)
theta_range_b = np.linspace(2, 6, 100)
ll_values_b = [log_likelihood_b(sample_b, t) for t in theta_range_b]

plt.figure(figsize=(10, 6))
plt.plot(theta_range_b, ll_values_b)
plt.axvline(x=mle_b, color='r', linestyle='--', label=f'MLE: θ̂ = {mle_b:.4f}')
plt.axvline(x=true_param_b, color='g', linestyle=':', label=f'True θ = {true_param_b}')
plt.xlabel('θ')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood function for case (b): Gamma(3,θ)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path_b = os.path.join(save_dir, "case_b_log_likelihood.png")
plt.savefig(file_path_b, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_b}")

# Step 4: Solving case (c)
print_step_header(4, "MLE for case (c): f(x;θ) = (1/2)e^(-|x-θ|)")

# Define the PDF for case (c)
def pdf_c(x, theta):
    return 0.5 * np.exp(-np.abs(x - theta))

# Define negative log-likelihood function for case (c) for numerical optimization
def neg_log_likelihood_c(theta, data):
    return -np.sum(np.log(pdf_c(data, theta)))

# Define the sum of absolute deviations function
def sum_abs_dev(theta, data):
    return np.sum(np.abs(data - theta))

# Analytical reasoning
print("Analytical Reasoning:")
print_latex(r"1. \text{The PDF is } f(x;\theta) = \frac{1}{2}e^{-|x-\theta|}")
print_latex(r"2. \text{The likelihood function for } n \text{ samples is:}")
print_latex(r"   L(\theta) = \prod_{i=1}^n \frac{1}{2}e^{-|x_i-\theta|} = \left(\frac{1}{2}\right)^n e^{-\sum_{i=1}^n|x_i-\theta|}")
print_latex(r"3. \text{The log-likelihood function is:}")
print_latex(r"   \ell(\theta) = \ln L(\theta) = \ln\left[\left(\frac{1}{2}\right)^n e^{-\sum_{i=1}^n|x_i-\theta|}\right]")
print_latex(r"   \ell(\theta) = n\ln\left(\frac{1}{2}\right) - \sum_{i=1}^n|x_i-\theta|")
print_latex(r"   \ell(\theta) = -n\ln(2) - \sum_{i=1}^n|x_i-\theta|")
print_latex(r"4. \text{Maximizing the log-likelihood is equivalent to minimizing } \sum_{i=1}^n |x_i-\theta|")
print_latex(r"5. \text{The derivative of } |x_i-\theta| \text{ with respect to } \theta \text{ is:}")
print_latex(r"   \frac{d|x_i-\theta|}{d\theta} = \begin{cases} -1 & \text{if } x_i > \theta \\ 1 & \text{if } x_i < \theta \end{cases}")
print_latex(r"6. \text{Setting the derivative of the log-likelihood to zero:}")
print_latex(r"   \frac{d\ell(\theta)}{d\theta} = -\sum_{i=1}^n\frac{d|x_i-\theta|}{d\theta} = 0")
print_latex(r"   \sum_{i: x_i > \theta} (-1) + \sum_{i: x_i < \theta} 1 = 0")
print_latex(r"   \sum_{i: x_i > \theta} 1 = \sum_{i: x_i < \theta} 1")
print_latex(r"7. \text{This equation is satisfied when } \theta \text{ is the median of the data}")
print_latex(r"   \text{The number of points above the median equals the number of points below the median}")
print_latex(r"   \text{Therefore, } \hat{\theta} = \text{median of } \{x_1, x_2, \ldots, x_n\}")

# Example with given data points
data_c = np.array([6.1, -1.1, 3.2, 0.7, 1.7])
median_c = np.median(data_c)

print("\nExample with n=5 data points:")
print(f"Data: {data_c}")
print(f"Median = {median_c}")

# Sort the data to work through the example step by step
sorted_data = np.sort(data_c)
print(f"Sorted data: {sorted_data}")

# Detailed calculation for Laplace distribution
print("\nDetailed Calculation for Laplace Distribution MLE:")
print(f"1. Sort the data: {sorted_data}")
print(f"2. Since n = {len(data_c)} is odd ({len(data_c)}), the median is at position {(len(data_c)+1)//2}: {sorted_data[(len(data_c)-1)//2]}")
print(f"3. For even n, the median would be the average of the middle two values")
print(f"4. MLE: θ̂ = median = {median_c}")

print("\nCalculating the sum of absolute deviations for different values of θ:")
for i, value in enumerate(sorted_data):
    sad = sum_abs_dev(value, data_c)
    print(f"θ = {value}, Sum of |x_i - θ| = {sad:.2f}")
    # Show detailed calculation
    deviations = [abs(x_i - value) for x_i in data_c]
    dev_str = " + ".join([f"|{x_i} - {value}| = {abs(x_i - value):.2f}" for x_i in data_c])
    print(f"   Detailed: {dev_str} = {sad:.2f}")

# Additional calculation for points around the median
print("\nChecking points around the median to verify minimum:")
test_points = np.linspace(median_c-0.5, median_c+0.5, 5)
for point in test_points:
    sad = sum_abs_dev(point, data_c)
    print(f"θ = {point:.2f}, Sum of |x_i - θ| = {sad:.2f}")

# Create a function that computes sums for many theta values
def compute_abs_dev_for_range(data, theta_range):
    return [sum_abs_dev(t, data) for t in theta_range]

# Plotting sum of absolute deviations for case (c)
theta_range_c = np.linspace(-2, 7, 100)
sad_values = compute_abs_dev_for_range(data_c, theta_range_c)

plt.figure(figsize=(10, 6))
plt.plot(theta_range_c, sad_values)
plt.axvline(x=median_c, color='r', linestyle='--', label=f'Median (MLE): θ̂ = {median_c}')
for i, x_i in enumerate(sorted_data):
    plt.axvline(x=x_i, color='g', linestyle=':', alpha=0.5, label=f'Data point: {x_i}' if i == 0 else None)
plt.xlabel('θ')
plt.ylabel('Sum of |x_i - θ|')
plt.title('Sum of Absolute Deviations for case (c): Laplace Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path_c1 = os.path.join(save_dir, "case_c_sum_abs_dev.png")
plt.savefig(file_path_c1, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_c1}")

# Plotting the negative log-likelihood for case (c)
ll_values_c = [-neg_log_likelihood_c(t, data_c) for t in theta_range_c]

plt.figure(figsize=(10, 6))
plt.plot(theta_range_c, ll_values_c)
plt.axvline(x=median_c, color='r', linestyle='--', label=f'Median (MLE): θ̂ = {median_c}')
for i, x_i in enumerate(sorted_data):
    plt.axvline(x=x_i, color='g', linestyle=':', alpha=0.5, label=f'Data point: {x_i}' if i == 0 else None)
plt.xlabel('θ')
plt.ylabel('Log-likelihood')
plt.title('Log-likelihood function for case (c): Laplace Distribution')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path_c2 = os.path.join(save_dir, "case_c_log_likelihood.png")
plt.savefig(file_path_c2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_c2}")

# Visualize the PDFs with different parameter values
plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

# Plot for case (a)
x_range_a = np.linspace(0, 30, 1000)
ax1 = plt.subplot(gs[0])
for theta in [3, 5, 7]:
    y = pdf_a(x_range_a, theta)
    ax1.plot(x_range_a, y, label=f'θ = {theta}')
ax1.set_title('PDF for case (a): Gamma(2,θ)')
ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True)

# Plot for case (b)
x_range_b = np.linspace(0, 30, 1000)
ax2 = plt.subplot(gs[1])
for theta in [3, 4, 5]:
    y = pdf_b(x_range_b, theta)
    ax2.plot(x_range_b, y, label=f'θ = {theta}')
ax2.set_title('PDF for case (b): Gamma(3,θ)')
ax2.set_xlabel('x')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True)

# Plot for case (c)
x_range_c = np.linspace(-10, 10, 1000)
ax3 = plt.subplot(gs[2])
for theta in [-2, 0, 2]:
    y = pdf_c(x_range_c, theta)
    ax3.plot(x_range_c, y, label=f'θ = {theta}')
ax3.set_title('PDF for case (c): Laplace Distribution')
ax3.set_xlabel('x')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True)

plt.tight_layout()

# Save the figure
file_path_pdfs = os.path.join(save_dir, "all_pdfs.png")
plt.savefig(file_path_pdfs, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_pdfs}")

# Add new visualization 1: Comparing the three PDFs with the same θ
plt.figure(figsize=(10, 6))
x_range = np.linspace(0, 20, 1000)
x_range_c_full = np.linspace(-10, 30, 2000)

# Use the same parameter value for all distributions
common_theta = 5
pdf_a_values = pdf_a(x_range, common_theta)
pdf_b_values = pdf_b(x_range, common_theta)
pdf_c_values = pdf_c(x_range_c_full, common_theta)

plt.plot(x_range, pdf_a_values, label=f'Case (a): Gamma(2,{common_theta})')
plt.plot(x_range, pdf_b_values, label=f'Case (b): Gamma(3,{common_theta})')
plt.plot(x_range_c_full, pdf_c_values, label=f'Case (c): Laplace({common_theta})')
plt.axvline(x=common_theta, color='black', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Comparison of all three PDFs with θ = 5')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path_pdf_compare = os.path.join(save_dir, "pdf_comparison.png")
plt.savefig(file_path_pdf_compare, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_pdf_compare}")

# Add new visualization 2: Sampling distribution of MLE estimators
plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

# Generate multiple samples and compute MLEs for case (a)
np.random.seed(123)
n_samples = 500
sample_size = 50
mle_samples_a = []
for _ in range(n_samples):
    sample = np.random.gamma(2, true_param_a, sample_size)
    mle_samples_a.append(np.mean(sample) / 2)

# Generate multiple samples and compute MLEs for case (b)
mle_samples_b = []
for _ in range(n_samples):
    sample = np.random.gamma(3, true_param_b, sample_size)
    mle_samples_b.append(np.mean(sample) / 3)

# Generate multiple samples and compute MLEs for case (c)
mle_samples_c = []
for _ in range(n_samples):
    sample = np.random.laplace(0, 1, sample_size)  # Generate from standard Laplace
    mle_samples_c.append(np.median(sample))  # MLE is median

# Plot histograms of MLE distributions
ax1 = plt.subplot(gs[0])
ax1.hist(mle_samples_a, bins=30, alpha=0.7, color='blue')
ax1.axvline(x=true_param_a, color='r', linestyle='--', label=f'True θ = {true_param_a}')
ax1.set_title('Sampling Distribution of MLE for case (a)')
ax1.set_xlabel('θ̂ = x̄/2')
ax1.set_ylabel('Frequency')
ax1.legend()

ax2 = plt.subplot(gs[1])
ax2.hist(mle_samples_b, bins=30, alpha=0.7, color='green')
ax2.axvline(x=true_param_b, color='r', linestyle='--', label=f'True θ = {true_param_b}')
ax2.set_title('Sampling Distribution of MLE for case (b)')
ax2.set_xlabel('θ̂ = x̄/3')
ax2.set_ylabel('Frequency')
ax2.legend()

ax3 = plt.subplot(gs[2])
ax3.hist(mle_samples_c, bins=30, alpha=0.7, color='purple')
ax3.axvline(x=0, color='r', linestyle='--', label='True θ = 0')
ax3.set_title('Sampling Distribution of MLE for case (c)')
ax3.set_xlabel('θ̂ = median')
ax3.set_ylabel('Frequency')
ax3.legend()

plt.tight_layout()

# Save the figure
file_path_mle_dist = os.path.join(save_dir, "mle_sampling_distributions.png")
plt.savefig(file_path_mle_dist, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path_mle_dist}")

# Step 5: Summary
print_step_header(5, "Summary of Results")

print("Maximum Likelihood Estimators:")
print_latex(r"(a) \text{For } f(x;\theta) = \frac{1}{\theta^2}x e^{-x/\theta}:")
print_latex(r"    \hat{\theta} = \frac{\bar{x}}{2} = " + f"{mle_a:.4f}")
print_latex(r"(b) \text{For } f(x;\theta) = \frac{1}{2\theta^3}x^2 e^{-x/\theta}:")
print_latex(r"    \hat{\theta} = \frac{\bar{x}}{3} = " + f"{mle_b:.4f}")
print_latex(r"(c) \text{For } f(x;\theta) = \frac{1}{2}e^{-|x-\theta|}:")
print_latex(r"    \hat{\theta} = \text{median} = " + f"{median_c}")

print("\nKey Insights:")
print_latex(r"1. \text{PDFs (a) and (b) have MLEs related to the sample mean by factors of } \frac{1}{2} \text{ and } \frac{1}{3} \text{ respectively.}")
print_latex(r"2. \text{PDF (c) has the median as its MLE, which is a robust estimator less affected by outliers.}")
print_latex(r"3. \text{The MLE minimizes the sum of absolute deviations in case (c), consistent with L1 loss in statistics.}")
print_latex(r"4. \text{The log-likelihood functions all have unique global maxima, confirming our analytical solutions.}")
print_latex(r"5. \text{The sampling distributions of the MLEs are approximately normally distributed, as expected by asymptotic theory.}") 