import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

def create_directory(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def bernoulli_mle_proof(save_dir):
    """Generate visualizations and proof details for Bernoulli MLE."""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # First plot: Bernoulli distributions for different probability values
    ax1 = plt.subplot(gs[0, 0])
    p_values = [0.2, 0.4, 0.6, 0.8]
    x = np.array([0, 1])
    width = 0.2
    
    for i, p in enumerate(p_values):
        pmf = np.array([1-p, p])
        ax1.bar(x + (i-len(p_values)/2+0.5)*width, pmf, width=width, alpha=0.7, 
               label=f'p = {p}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability')
    ax1.set_title('Bernoulli Distribution PMF for Different p Values')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['0', '1'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Second plot: Sample data and MLE visualization
    ax2 = plt.subplot(gs[0, 1])
    # Generate sample data
    np.random.seed(42)
    N = 20
    true_p = 0.7
    samples = np.random.binomial(1, true_p, size=N)
    
    # Calculate MLE
    p_mle = np.mean(samples)
    
    # Plot samples
    x_vals = np.arange(1, N+1)
    ax2.stem(x_vals, samples, linefmt='b-', markerfmt='bo', basefmt='gray', label='Samples')
    ax2.axhline(y=p_mle, color='red', linestyle='--', 
               label=f'MLE $\\hat{{p}} = {p_mle:.2f}$')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Value (0 or 1)')
    ax2.set_title(f'Bernoulli Samples and MLE Estimate')
    ax2.set_yticks([0, 1])
    ax2.set_ylim(-0.1, 1.2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Third plot: Likelihood function for different p values given sample data
    ax3 = plt.subplot(gs[1, 0])
    p_range = np.linspace(0.01, 0.99, 100)
    
    # Calculate number of successes and failures
    successes = np.sum(samples)
    failures = N - successes
    
    # Compute likelihood for each p value
    likelihood = [p**successes * (1-p)**failures for p in p_range]
    log_likelihood = [successes * np.log(p) + failures * np.log(1-p) for p in p_range]
    
    ax3.plot(p_range, likelihood, 'b-', linewidth=2)
    ax3.axvline(x=p_mle, color='red', linestyle='--', 
               label=f'MLE $\\hat{{p}} = {p_mle:.2f}$')
    ax3.set_xlabel('p')
    ax3.set_ylabel('Likelihood L(p)')
    ax3.set_title('Likelihood Function for Bernoulli MLE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Fourth plot: Log-likelihood function
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(p_range, log_likelihood, 'g-', linewidth=2)
    ax4.axvline(x=p_mle, color='red', linestyle='--', 
               label=f'MLE $\\hat{{p}} = {p_mle:.2f}$')
    ax4.set_xlabel('p')
    ax4.set_ylabel('Log-Likelihood ℓ(p)')
    ax4.set_title('Log-Likelihood Function for Bernoulli MLE')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, "bernoulli_mle.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print step-by-step derivation
    print("\n=== Bernoulli MLE Proof ===")
    print(f"Sample data: {samples}")
    print(f"N = {N}, Number of successes = {successes}, Number of failures = {failures}")
    print(f"MLE estimate: p_hat = {p_mle:.4f}")
    
    print("\nStep 1: Write down the Bernoulli PMF")
    print("P(X=x|p) = p^x × (1-p)^(1-x) for x ∈ {0, 1}")
    
    print("\nStep 2: Form the likelihood function")
    print(f"L(p) = Product from i=1 to {N} of p^x_i × (1-p)^(1-x_i)")
    print(f"L(p) = p^{successes} × (1-p)^{failures}")
    
    print("\nStep 3: Take the logarithm to get the log-likelihood")
    print(f"ℓ(p) = log(L(p)) = {successes} × log(p) + {failures} × log(1-p)")
    
    print("\nStep 4: Find the critical points by taking the derivative")
    print(f"dℓ/dp = {successes}/p - {failures}/(1-p) = 0")
    
    print("\nStep 5: Solve for p")
    print(f"{successes}/p = {failures}/(1-p)")
    print(f"{successes}(1-p) = {failures}p")
    print(f"{successes} - {successes}p = {failures}p")
    print(f"{successes} = ({successes} + {failures})p")
    print(f"{successes} = {N}p")
    print(f"p = {successes}/{N} = {p_mle:.4f}")
    
    print("\nStep 6: Verify this is a maximum (not a minimum)")
    print(f"d²ℓ/dp² = -{successes}/p² - {failures}/(1-p)²")
    second_deriv = -successes/(p_mle**2) - failures/((1-p_mle)**2)
    print(f"At p = {p_mle:.4f}, d²ℓ/dp² = {second_deriv:.4f} < 0")
    print("Since the second derivative is negative, our critical point is indeed a maximum.")
    
    results = {
        "samples": samples,
        "N": N,
        "successes": successes,
        "failures": failures,
        "p_mle": p_mle
    }
    
    return results

def multinomial_mle_proof(save_dir):
    """Generate visualizations and proof details for Multinomial MLE."""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # First plot: Multinomial probabilities visualization
    ax1 = plt.subplot(gs[0, 0])
    # Example with 4 categories
    K = 4
    true_probs = np.array([0.3, 0.4, 0.2, 0.1])
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    
    ax1.bar(categories, true_probs, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Probability')
    ax1.set_title('True Multinomial Probabilities')
    ax1.grid(True, alpha=0.3)
    
    # Second plot: Sample data and MLE
    ax2 = plt.subplot(gs[0, 1])
    
    # Generate sample data
    np.random.seed(42)
    N = 100
    samples = np.random.multinomial(1, true_probs, size=N)
    # Count occurrences of each category
    counts = np.sum(samples, axis=0)
    # Calculate MLE
    p_mle = counts / N
    
    # Create a grouped bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, true_probs, width, alpha=0.7, label='True Probabilities')
    ax2.bar(x + width/2, p_mle, width, alpha=0.7, label='MLE Estimates')
    
    ax2.set_ylabel('Probability')
    ax2.set_title('True vs. MLE Probabilities')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Third plot: Convergence of MLE with increasing sample size
    ax3 = plt.subplot(gs[1, 0])
    
    # Simulate different sample sizes
    sample_sizes = [10, 50, 100, 500, 1000, 5000]
    mse_values = []
    
    for size in sample_sizes:
        mse_sum = 0
        for _ in range(50):  # Multiple trials for each sample size
            samples = np.random.multinomial(1, true_probs, size=size)
            counts = np.sum(samples, axis=0)
            p_est = counts / size
            mse = np.mean((true_probs - p_est) ** 2)
            mse_sum += mse
        mse_values.append(mse_sum / 50)
    
    ax3.plot(sample_sizes, mse_values, 'bo-', linewidth=2)
    ax3.set_xlabel('Sample Size (N)')
    ax3.set_ylabel('Mean Squared Error')
    ax3.set_title('Convergence of Multinomial MLE Estimator')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Fourth plot: 3D surface for multinomial log-likelihood with 3 categories
    ax4 = plt.subplot(gs[1, 1], projection='3d')
    
    # For simplicity, focus on 3 categories (since we can visualize in 3D)
    # The third probability is determined by the constraint that they sum to 1
    p1_range = np.linspace(0.01, 0.98, 30)
    p2_range = np.linspace(0.01, 0.98, 30)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    # Filter points where p1 + p2 <= 1 (valid probability constraint)
    mask = P1 + P2 <= 0.99
    
    # Example counts for 3 categories
    N1, N2, N3 = 30, 40, 30
    N_total = N1 + N2 + N3
    
    # Calculate log-likelihood for valid (p1, p2) combinations
    log_L = np.zeros_like(P1)
    log_L.fill(np.nan)  # Set all to NaN initially
    
    for i in range(len(p1_range)):
        for j in range(len(p2_range)):
            if P1[i, j] + P2[i, j] <= 0.99:
                p3 = 1 - P1[i, j] - P2[i, j]
                log_L[i, j] = N1 * np.log(P1[i, j]) + N2 * np.log(P2[i, j]) + N3 * np.log(p3)
    
    # Plot the valid surface
    surf = ax4.plot_surface(P1, P2, log_L, cmap=cm.viridis, alpha=0.8)
    
    # Mark the MLE point
    p1_mle, p2_mle, p3_mle = N1/N_total, N2/N_total, N3/N_total
    log_L_mle = N1 * np.log(p1_mle) + N2 * np.log(p2_mle) + N3 * np.log(p3_mle)
    ax4.scatter([p1_mle], [p2_mle], [log_L_mle], color='red', s=100, label='MLE')
    
    ax4.set_xlabel('p₁')
    ax4.set_ylabel('p₂')
    ax4.set_zlabel('Log-Likelihood')
    ax4.set_title('Multinomial Log-Likelihood Surface')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, "multinomial_mle.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print step-by-step derivation
    print("\n=== Multinomial MLE Proof ===")
    print(f"Using a multinomial distribution with {K} categories")
    print(f"True probabilities: {true_probs}")
    print(f"Sample size: N = {N}")
    print(f"Observed counts: {counts}")
    print(f"MLE estimates: {p_mle}")
    
    print("\nStep 1: Write down the Multinomial PMF")
    print("P(X₁=n₁, X₂=n₂, ..., X_K=n_K) = N! / (n₁! × n₂! × ... × n_K!) × p₁^n₁ × p₂^n₂ × ... × p_K^n_K")
    
    print("\nStep 2: Form the likelihood function")
    print("L(p₁, p₂, ..., p_K) = N! / (n₁! × n₂! × ... × n_K!) × p₁^n₁ × p₂^n₂ × ... × p_K^n_K")
    
    print("\nStep 3: Take the logarithm to get the log-likelihood")
    print("ℓ(p₁, p₂, ..., p_K) = log[N! / (n₁! × n₂! × ... × n_K!)] + n₁log(p₁) + n₂log(p₂) + ... + n_Klog(p_K)")
    print("The first term is constant with respect to the parameters, so we can focus on:")
    print("ℓ(p₁, p₂, ..., p_K) ∝ n₁log(p₁) + n₂log(p₂) + ... + n_Klog(p_K)")
    
    print("\nStep 4: Maximize subject to the constraint p₁ + p₂ + ... + p_K = 1")
    print("Using Lagrange multipliers, form the Lagrangian:")
    print("L(p₁, p₂, ..., p_K, λ) = n₁log(p₁) + n₂log(p₂) + ... + n_Klog(p_K) - λ(p₁ + p₂ + ... + p_K - 1)")
    
    print("\nStep 5: Take partial derivatives and set to zero")
    print("∂L/∂p₁ = n₁/p₁ - λ = 0 ⟹ p₁ = n₁/λ")
    print("∂L/∂p₂ = n₂/p₂ - λ = 0 ⟹ p₂ = n₂/λ")
    print("...")
    print("∂L/∂p_K = n_K/p_K - λ = 0 ⟹ p_K = n_K/λ")
    
    print("\nStep 6: Use the constraint to find λ")
    print("p₁ + p₂ + ... + p_K = 1")
    print("n₁/λ + n₂/λ + ... + n_K/λ = 1")
    print("(n₁ + n₂ + ... + n_K)/λ = 1")
    print("N/λ = 1")
    print("λ = N")
    
    print("\nStep 7: Substitute back to find the MLEs")
    print("p₁ = n₁/λ = n₁/N")
    print("p₂ = n₂/λ = n₂/N")
    print("...")
    print("p_K = n_K/λ = n_K/N")
    
    print("\nTherefore, the maximum likelihood estimator for category k is:")
    print("p̂_k = n_k/N")
    print("That is, the proportion of observations falling in category k.")
    
    results = {
        "K": K,
        "true_probs": true_probs,
        "N": N,
        "counts": counts,
        "p_mle": p_mle
    }
    
    return results

def gaussian_mean_mle_proof(save_dir):
    """Generate visualizations and proof details for Gaussian MLE (known variance)."""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # First plot: Gaussian distributions for different mean values
    ax1 = plt.subplot(gs[0, 0])
    x = np.linspace(-5, 10, 1000)
    mean_values = [0, 2, 4, 6]
    sigma = 1.5  # Fixed variance
    
    for mu in mean_values:
        pdf = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, pdf, label=f'μ = {mu}, σ² = {sigma**2}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Gaussian PDF for Different Mean Values (Fixed Variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Second plot: Sample data and MLE visualization
    ax2 = plt.subplot(gs[0, 1])
    
    # Generate sample data
    np.random.seed(42)
    N = 50
    true_mu = 3.5
    true_sigma = 1.5
    samples = np.random.normal(true_mu, true_sigma, size=N)
    
    # Calculate MLE for mean
    mu_mle = np.mean(samples)
    
    # Plot histogram and density
    ax2.hist(samples, bins=15, density=True, alpha=0.7, color='skyblue', 
            label='Sample Histogram')
    
    # Plot true and estimated densities
    x_range = np.linspace(min(samples)-2, max(samples)+2, 1000)
    true_pdf = stats.norm.pdf(x_range, true_mu, true_sigma)
    mle_pdf = stats.norm.pdf(x_range, mu_mle, true_sigma)
    
    ax2.plot(x_range, true_pdf, 'g-', linewidth=2, 
            label=f'True: μ = {true_mu}, σ = {true_sigma}')
    ax2.plot(x_range, mle_pdf, 'r--', linewidth=2,
            label=f'MLE: μ̂ = {mu_mle:.2f}, σ = {true_sigma}')
    
    # Add vertical lines for the means
    ax2.axvline(x=true_mu, color='green', linestyle='-')
    ax2.axvline(x=mu_mle, color='red', linestyle='--')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Gaussian Sample Data and MLE Fit (N={N})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Third plot: Likelihood function for different mu values
    ax3 = plt.subplot(gs[1, 0])
    mu_range = np.linspace(mu_mle-2, mu_mle+2, 1000)
    
    # Compute likelihood for each mu value
    def compute_likelihood(mu, samples, sigma):
        return np.prod(stats.norm.pdf(samples, mu, sigma))
    
    def compute_log_likelihood(mu, samples, sigma):
        return np.sum(stats.norm.logpdf(samples, mu, sigma))
    
    likelihoods = [compute_likelihood(mu, samples, true_sigma) for mu in mu_range]
    log_likelihoods = [compute_log_likelihood(mu, samples, true_sigma) for mu in mu_range]
    
    ax3.plot(mu_range, likelihoods, 'b-', linewidth=2)
    ax3.axvline(x=mu_mle, color='red', linestyle='--', 
               label=f'MLE μ̂ = {mu_mle:.2f}')
    ax3.axvline(x=true_mu, color='green', linestyle='-', 
               label=f'True μ = {true_mu}')
    
    ax3.set_xlabel('μ')
    ax3.set_ylabel('Likelihood L(μ)')
    ax3.set_title('Likelihood Function for Gaussian Mean')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Fourth plot: Log-likelihood function
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(mu_range, log_likelihoods, 'g-', linewidth=2)
    ax4.axvline(x=mu_mle, color='red', linestyle='--', 
               label=f'MLE μ̂ = {mu_mle:.2f}')
    ax4.axvline(x=true_mu, color='green', linestyle='-', 
               label=f'True μ = {true_mu}')
    
    ax4.set_xlabel('μ')
    ax4.set_ylabel('Log-Likelihood ℓ(μ)')
    ax4.set_title('Log-Likelihood Function for Gaussian Mean')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, "gaussian_mean_mle.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print step-by-step derivation
    print("\n=== Gaussian Mean MLE Proof (Known Variance) ===")
    print(f"True parameters: μ = {true_mu}, σ² = {true_sigma**2}")
    print(f"Sample size: N = {N}")
    print(f"MLE estimate for mean: μ̂ = {mu_mle:.4f}")
    
    print("\nStep 1: Write down the Gaussian PDF")
    print("f(x|μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))")
    
    print("\nStep 2: Form the likelihood function")
    print(f"L(μ) = Product from i=1 to {N} of (1/√(2πσ²)) × exp(-(x_i-μ)²/(2σ²))")
    print("L(μ) = (1/√(2πσ²))^N × exp(-sum(x_i-μ)²/(2σ²))")
    
    print("\nStep 3: Take the logarithm to get the log-likelihood")
    print("ℓ(μ) = -N/2 × log(2πσ²) - sum(x_i-μ)²/(2σ²)")
    print("Since the first term is constant with respect to μ, we can focus on:")
    print("ℓ(μ) ∝ -sum(x_i-μ)²/(2σ²)")
    print("Maximizing this is equivalent to minimizing sum(x_i-μ)²")
    
    print("\nStep 4: Find the critical points by taking the derivative")
    print("dℓ/dμ = sum(x_i-μ)/σ² = 0")
    print("sum(x_i-μ) = 0")
    print("sum(x_i) - Nμ = 0")
    
    print("\nStep 5: Solve for μ")
    print("Nμ = sum(x_i)")
    print("μ = (1/N)sum(x_i)")
    print(f"μ = {mu_mle:.4f}")
    
    print("\nStep 6: Verify this is a maximum (not a minimum)")
    print("d²ℓ/dμ² = -N/σ² < 0")
    print("Since the second derivative is negative, our critical point is indeed a maximum.")
    
    print("\nTherefore, the maximum likelihood estimator for μ is:")
    print("μ̂ = (1/N)sum(x_i) = sample mean")
    
    results = {
        "true_mu": true_mu,
        "true_sigma": true_sigma,
        "N": N,
        "samples": samples,
        "mu_mle": mu_mle
    }
    
    return results

def gaussian_variance_mle_proof(save_dir):
    """Generate visualizations and proof details for Gaussian MLE (unknown mean and variance)."""
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # First plot: Gaussian distributions for different variance values
    ax1 = plt.subplot(gs[0, 0])
    x = np.linspace(-8, 8, 1000)
    sigma_values = [0.5, 1.0, 1.5, 2.0]
    mu = 0  # Fixed mean
    
    for sigma in sigma_values:
        pdf = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, pdf, label=f'μ = {mu}, σ = {sigma}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Gaussian PDF for Different Variance Values (Fixed Mean)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Second plot: Sample data and MLE visualization
    ax2 = plt.subplot(gs[0, 1])
    
    # Generate sample data
    np.random.seed(42)
    N = 100
    true_mu = 0
    true_sigma = 1.5
    samples = np.random.normal(true_mu, true_sigma, size=N)
    
    # Calculate MLE for mean and variance
    mu_mle = np.mean(samples)
    sigma2_mle = np.mean((samples - mu_mle)**2)  # MLE for variance
    sigma_mle = np.sqrt(sigma2_mle)
    
    # Plot histogram and density
    ax2.hist(samples, bins=20, density=True, alpha=0.7, color='skyblue', 
            label='Sample Histogram')
    
    # Plot true and estimated densities
    x_range = np.linspace(min(samples)-2, max(samples)+2, 1000)
    true_pdf = stats.norm.pdf(x_range, true_mu, true_sigma)
    mle_pdf = stats.norm.pdf(x_range, mu_mle, sigma_mle)
    
    ax2.plot(x_range, true_pdf, 'g-', linewidth=2, 
            label=f'True: μ = {true_mu}, σ = {true_sigma}')
    ax2.plot(x_range, mle_pdf, 'r--', linewidth=2,
            label=f'MLE: μ̂ = {mu_mle:.2f}, σ̂ = {sigma_mle:.2f}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Gaussian Sample Data and MLE Fit (N={N})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Third plot: 3D surface for log-likelihood as function of mean and variance
    ax3 = plt.subplot(gs[1, 0], projection='3d')
    
    mu_range = np.linspace(mu_mle-0.5, mu_mle+0.5, 25)
    sigma2_range = np.linspace(sigma2_mle*0.7, sigma2_mle*1.3, 25)
    Mu, Sigma2 = np.meshgrid(mu_range, sigma2_range)
    
    # Compute log-likelihood for each (mu, sigma2) combination
    log_L = np.zeros_like(Mu)
    for i in range(len(mu_range)):
        for j in range(len(sigma2_range)):
            mu_val = Mu[j, i]
            sigma_val = np.sqrt(Sigma2[j, i])
            log_L[j, i] = np.sum(stats.norm.logpdf(samples, mu_val, sigma_val))
    
    # Plot the surface
    surf = ax3.plot_surface(Mu, Sigma2, log_L, cmap=cm.viridis, alpha=0.8)
    
    # Mark the MLE point
    log_L_mle = np.sum(stats.norm.logpdf(samples, mu_mle, sigma_mle))
    ax3.scatter([mu_mle], [sigma2_mle], [log_L_mle], color='red', s=100, label='MLE')
    
    ax3.set_xlabel('μ')
    ax3.set_ylabel('σ²')
    ax3.set_zlabel('Log-Likelihood')
    ax3.set_title('Gaussian Log-Likelihood Surface')
    
    # Fourth plot: Convergence of variance MLE with increasing sample size
    ax4 = plt.subplot(gs[1, 1])
    
    # Simulate different sample sizes and track MSE of variance estimator
    sample_sizes = [10, 20, 50, 100, 200, 500, 1000]
    var_mle_values = []  # MLE estimator values
    var_unbiased_values = []  # Unbiased estimator values
    
    num_trials = 100
    for size in sample_sizes:
        mle_sum = 0
        unbiased_sum = 0
        for _ in range(num_trials):
            sample = np.random.normal(true_mu, true_sigma, size=size)
            sample_mean = np.mean(sample)
            
            # MLE for variance
            var_mle = np.mean((sample - sample_mean)**2)
            mle_sum += var_mle
            
            # Unbiased estimator for variance
            var_unbiased = np.sum((sample - sample_mean)**2) / (size - 1)
            unbiased_sum += var_unbiased
            
        var_mle_values.append(mle_sum / num_trials)
        var_unbiased_values.append(unbiased_sum / num_trials)
    
    # True variance line
    ax4.axhline(y=true_sigma**2, color='green', linestyle='-', label='True σ²')
    
    # Plot MLE and unbiased estimator convergence
    ax4.plot(sample_sizes, var_mle_values, 'ro-', linewidth=2, label='MLE σ²')
    ax4.plot(sample_sizes, var_unbiased_values, 'bo--', linewidth=2, label='Unbiased s²')
    
    ax4.set_xlabel('Sample Size (N)')
    ax4.set_ylabel('Variance Estimate')
    ax4.set_title('Convergence of Variance Estimators')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, "gaussian_variance_mle.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print step-by-step derivation
    print("\n=== Gaussian Variance MLE Proof (Unknown Mean and Variance) ===")
    print(f"True parameters: μ = {true_mu}, σ² = {true_sigma**2}")
    print(f"Sample size: N = {N}")
    print(f"MLE estimates: μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_mle:.4f}")
    
    print("\nStep 1: Write down the Gaussian PDF")
    print("f(x|μ,σ²) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))")
    
    print("\nStep 2: Form the likelihood function")
    print(f"L(μ,σ²) = Product from i=1 to {N} of (1/√(2πσ²)) × exp(-(x_i-μ)²/(2σ²))")
    print("L(μ,σ²) = (1/√(2πσ²))^N × exp(-sum(x_i-μ)²/(2σ²))")
    
    print("\nStep 3: Take the logarithm to get the log-likelihood")
    print("ℓ(μ,σ²) = -N/2 × log(2πσ²) - sum(x_i-μ)²/(2σ²)")
    print("ℓ(μ,σ²) = -N/2 × log(2π) - N/2 × log(σ²) - sum(x_i-μ)²/(2σ²)")
    
    print("\nStep 4: Find the critical points by taking partial derivatives")
    print("∂ℓ/∂μ = sum(x_i-μ)/σ² = 0")
    print("∂ℓ/∂(σ²) = -N/(2σ²) + sum(x_i-μ)²/(2(σ²)²) = 0")
    
    print("\nStep 5: Solve for μ from the first equation")
    print("sum(x_i-μ) = 0")
    print("sum(x_i) - Nμ = 0")
    print("μ = (1/N)sum(x_i)")
    print(f"μ̂ = {mu_mle:.4f}")
    
    print("\nStep 6: Solve for σ² from the second equation")
    print("N/(2σ²) = sum(x_i-μ)²/(2(σ²)²)")
    print("N(σ²) = sum(x_i-μ)²")
    print("σ² = (1/N)sum(x_i-μ)²")
    print(f"σ̂² = {sigma2_mle:.4f}")
    
    print("\nStep 7: Verify these are maximum values (not minimum)")
    print("The Hessian matrix of second partial derivatives is negative definite at the critical point,")
    print("confirming that our solution is indeed a maximum.")
    
    print("\nStep 8: Note on bias of the variance estimator")
    unbiased_var = np.sum((samples - mu_mle)**2) / (N - 1)
    print(f"While σ̂² = (1/N)sum(x_i-μ̂)² = {sigma2_mle:.4f} is the MLE,")
    print(f"it is biased. The unbiased estimator is s² = (1/(N-1))sum(x_i-μ̂)² = {unbiased_var:.4f}")
    print(f"The bias of the MLE is E[σ̂²] - σ² = -σ²/N")
    
    results = {
        "true_mu": true_mu,
        "true_sigma": true_sigma,
        "N": N,
        "samples": samples,
        "mu_mle": mu_mle,
        "sigma2_mle": sigma2_mle,
        "unbiased_var": unbiased_var
    }
    
    return results

def main():
    """Main function to execute all MLE proof demonstrations."""
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_26")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating MLE proofs and visualizations for Question 26...")
    
    # Run all proofs
    bernoulli_results = bernoulli_mle_proof(save_dir)
    print("\n" + "="*80)
    
    multinomial_results = multinomial_mle_proof(save_dir)
    print("\n" + "="*80)
    
    gaussian_mean_results = gaussian_mean_mle_proof(save_dir)
    print("\n" + "="*80)
    
    gaussian_variance_results = gaussian_variance_mle_proof(save_dir)
    print("\n" + "="*80)
    
    print("\nAll visualizations have been saved to:")
    print(save_dir)
    
    # Print summary of all results
    print("\n=== Summary of MLE Proofs ===")
    print("1. Bernoulli MLE: p_hat = k/N (proportion of successes)")
    print("2. Multinomial MLE: p_hat_k = n_k/N (proportion in each category)")
    print("3. Gaussian Mean MLE (known variance): mu_hat = (1/N)sum(x_i) (sample mean)")
    print("4. Gaussian Variance MLE: sigma^2_hat = (1/N)sum(x_i-mu_hat)^2 (biased estimator)")
    
    print("\nAll these estimators can be derived through the maximum likelihood method,")
    print("which involves finding parameter values that maximize the likelihood of")
    print("observing the given sample data.")

if __name__ == "__main__":
    main()