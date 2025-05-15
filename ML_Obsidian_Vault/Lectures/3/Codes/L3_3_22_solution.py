import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import os
import pandas as pd
from scipy.integrate import quad

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Step 1: Demonstrate how to compute marginal likelihood
def explain_marginal_likelihood():
    """Explain how to compute the marginal likelihood for model comparison."""
    print("Step 1: Computing the Marginal Likelihood (Model Evidence)")
    print("="*80)
    print("The marginal likelihood for a model M_i is given by:")
    print("p(y|M_i) = ∫ p(y|θ,M_i) p(θ|M_i) dθ")
    print("where:")
    print("  - p(y|θ,M_i) is the likelihood of the data given model parameters θ")
    print("  - p(θ|M_i) is the prior distribution of parameters under model M_i")
    print()
    print("For linear regression models, this integral can be computed in several ways:")
    print()
    print("1. Analytical solution (for conjugate priors):")
    print("   For linear regression with Gaussian likelihood and Gaussian prior on weights,")
    print("   the marginal likelihood has a closed-form solution:")
    print("   p(y|X,M_i) = N(y|0, σ²I + XΣX^T)")
    print("   where Σ is the prior covariance matrix of weights.")
    print()
    print("2. Laplace approximation:")
    print("   Approximate the posterior as a Gaussian centered at the MAP estimate θ_MAP:")
    print("   p(y|M_i) ≈ p(y|θ_MAP,M_i)p(θ_MAP|M_i) × (2π)^{d/2}|Σ_MAP|^{1/2}")
    print("   where d is the dimensionality of θ and Σ_MAP is the inverse Hessian at θ_MAP.")
    print()
    print("3. Numerical integration:")
    print("   For low-dimensional models, direct numerical integration can be used.")
    print()
    print("4. Monte Carlo estimation:")
    print("   Draw samples θ^(j) from the prior p(θ|M_i) and estimate:")
    print("   p(y|M_i) ≈ (1/J) ∑_j p(y|θ^(j),M_i)")
    print()
    print("5. Importance sampling:")
    print("   Use a proposal distribution q(θ) and estimate:")
    print("   p(y|M_i) ≈ ∑_j [p(y|θ^(j),M_i)p(θ^(j)|M_i)/q(θ^(j))]")
    print("   where θ^(j) are samples from q(θ).")
    print()

# Step 2: Explain and visualize Occam's razor
def explain_occams_razor():
    """Explain and visualize Occam's razor in Bayesian model selection."""
    print("\nStep 2: Occam's Razor in Bayesian Model Selection")
    print("="*80)
    print("Occam's razor is the principle that, all else being equal, simpler explanations")
    print("are generally better than more complex ones. In Bayesian model selection, this")
    print("principle is naturally incorporated through the marginal likelihood.")
    print()
    print("The marginal likelihood can be written as:")
    print("p(y|M_i) = ∫ p(y|θ,M_i) p(θ|M_i) dθ")
    print()
    print("This integral has a natural interpretation as the average fit of the model over")
    print("the prior distribution of parameters. More complex models can achieve higher")
    print("likelihood values but spread this prior over a larger parameter space.")
    print()
    print("The marginal likelihood implements a trade-off between:")
    print("1. Goodness of fit: How well the model fits the observed data")
    print("2. Model complexity: How much parameter space the model uses")
    print()
    print("More complex models are penalized because they spread their prior probability")
    print("over a larger parameter space, which means they assign lower prior probability")
    print("to any specific setting of parameters. This automatic penalty for complexity")
    print("is known as the 'Bayesian Occam's Razor'.")
    print()
    
    # Create visualizations for Occam's razor
    # 1. Simple model vs complex model visualization
    plt.figure(figsize=(12, 7))
    
    # Parameters
    x = np.linspace(-5, 5, 1000)
    
    # Simple model (narrow but tall distribution)
    y_simple = stats.norm.pdf(x, 0, 1) * 2  # Higher peak
    
    # Complex model (wider but shorter distribution)
    y_complex = stats.norm.pdf(x, 0, 3) * 0.7  # Lower peak, wider spread
    
    # Plot distributions
    plt.plot(x, y_simple, 'b-', linewidth=2, label='Simple Model (M₁)')
    plt.plot(x, y_complex, 'r-', linewidth=2, label='Complex Model (M₃)')
    
    # Highlight "true parameter" region
    true_param = 1.5
    true_region = np.abs(x - true_param) < 0.5
    plt.fill_between(x[true_region], 0, y_simple[true_region], color='blue', alpha=0.2)
    plt.fill_between(x[true_region], 0, y_complex[true_region], color='red', alpha=0.2)
    
    # Add vertical line for "true parameter"
    plt.axvline(x=true_param, color='k', linestyle='--', alpha=0.7, 
                label='True Parameter Region')
    
    plt.title("Bayesian Occam's Razor: Parameter Space and Prior Distribution", fontsize=14)
    plt.xlabel('Parameter Value', fontsize=12)
    plt.ylabel('Prior Probability Density', fontsize=12)
    plt.legend(fontsize=11)
    
    # Add annotations
    plt.annotate('Simple model concentrates\nprior probability in\nsmaller region of\nparameter space',
                xy=(0, 0.6), xytext=(-4, 0.4),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
                fontsize=10, color='blue')
    
    plt.annotate('Complex model spreads\nprior probability over\nlarger parameter space',
                xy=(3, 0.1), xytext=(3, 0.3),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                fontsize=10, color='red')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    occams_razor_file = os.path.join(save_dir, "occams_razor_prior.png")
    plt.savefig(occams_razor_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Visualization of the likelihood and marginal likelihood tradeoff
    plt.figure(figsize=(12, 7))
    
    # Create some example data
    np.random.seed(42)
    x_data = np.linspace(0, 10, 20)
    y_true = 2 + 0.5 * x_data
    y_data = y_true + np.random.normal(0, 1, len(x_data))
    
    # Function to generate predictions for different model complexities
    x_fine = np.linspace(0, 10, 100)
    
    # Model 1: Linear (complexity = 2 parameters)
    def model1(x, theta):
        return theta[0] + theta[1] * x
    
    # Model 2: Quadratic (complexity = 3 parameters)
    def model2(x, theta):
        return theta[0] + theta[1] * x + theta[2] * x**2
    
    # Model 3: Cubic (complexity = 4 parameters)
    def model3(x, theta):
        return theta[0] + theta[1] * x + theta[2] * x**2 + theta[3] * x**3
    
    # Fit models using least squares
    from scipy.optimize import curve_fit
    
    def fit_model1(x, a, b):
        return a + b * x
    
    def fit_model2(x, a, b, c):
        return a + b * x + c * x**2
    
    def fit_model3(x, a, b, c, d):
        return a + b * x + c * x**2 + d * x**3
    
    # Fit the models
    popt1, _ = curve_fit(fit_model1, x_data, y_data)
    popt2, _ = curve_fit(fit_model2, x_data, y_data)
    popt3, _ = curve_fit(fit_model3, x_data, y_data)
    
    # Calculate predictions
    y_pred1 = fit_model1(x_fine, *popt1)
    y_pred2 = fit_model2(x_fine, *popt2)
    y_pred3 = fit_model3(x_fine, *popt3)
    
    # Calculate log-likelihoods for each model
    def calc_log_likelihood(y_data, y_pred_at_data):
        # Assuming constant variance of 1
        sigma = 1.0
        return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y_data - y_pred_at_data) / sigma)**2)
    
    y_pred1_at_data = fit_model1(x_data, *popt1)
    y_pred2_at_data = fit_model2(x_data, *popt2)
    y_pred3_at_data = fit_model3(x_data, *popt3)
    
    log_lik1 = calc_log_likelihood(y_data, y_pred1_at_data)
    log_lik2 = calc_log_likelihood(y_data, y_pred2_at_data)
    log_lik3 = calc_log_likelihood(y_data, y_pred3_at_data)
    
    # Calculate BIC for each model
    n = len(y_data)
    
    def calc_bic(log_lik, n, k):
        return -2 * log_lik + k * np.log(n)
    
    bic1 = calc_bic(log_lik1, n, 2)
    bic2 = calc_bic(log_lik2, n, 3)
    bic3 = calc_bic(log_lik3, n, 4)
    
    # Plot the data and model fits
    plt.scatter(x_data, y_data, color='k', alpha=0.7, label='Data')
    plt.plot(x_fine, y_pred1, 'b-', linewidth=2, label=f'Model 1 (Linear): BIC={bic1:.2f}')
    plt.plot(x_fine, y_pred2, 'r-', linewidth=2, label=f'Model 2 (Quadratic): BIC={bic2:.2f}')
    plt.plot(x_fine, y_pred3, 'g-', linewidth=2, label=f'Model 3 (Cubic): BIC={bic3:.2f}')
    
    plt.title("Model Complexity vs. Goodness of Fit", fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=11)
    
    # Add annotations
    plt.annotate('As complexity increases, fit improves\nbut risk of overfitting increases',
                xy=(8, 9), xytext=(6, 11),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10)
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    model_complexity_file = os.path.join(save_dir, "model_complexity_fit.png")
    plt.savefig(model_complexity_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Conceptual visualization of Bayesian Occam's Razor
    plt.figure(figsize=(12, 7))
    
    # Define complexity (x) and goodness of fit (y)
    complexity = np.linspace(1, 10, 100)
    
    # Goodness of fit increases with complexity but with diminishing returns
    fit_quality = 10 * (1 - np.exp(-0.4 * complexity))
    
    # Complexity penalty increases with complexity
    complexity_penalty = 0.5 * complexity
    
    # Evidence is goodness of fit minus complexity penalty
    evidence = fit_quality - complexity_penalty
    
    # Calculate optimal complexity
    optimal_idx = np.argmax(evidence)
    optimal_complexity = complexity[optimal_idx]
    
    # Plot
    plt.plot(complexity, fit_quality, 'b-', linewidth=2, label='Goodness of Fit')
    plt.plot(complexity, complexity_penalty, 'r-', linewidth=2, label='Complexity Penalty')
    plt.plot(complexity, evidence, 'g-', linewidth=3, label='Model Evidence')
    
    # Mark optimal point
    plt.scatter([optimal_complexity], [evidence[optimal_idx]], c='k', s=100, zorder=5,
                label=f'Optimal Complexity = {optimal_complexity:.2f}')
    
    plt.axvline(x=optimal_complexity, color='k', linestyle='--', alpha=0.5)
    
    plt.title("Bayesian Occam's Razor: Balancing Fit and Complexity", fontsize=14)
    plt.xlabel('Model Complexity', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    
    # Add annotations
    plt.annotate('Higher complexity improves fit\nbut with diminishing returns',
                xy=(8, 8), xytext=(6, 10),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='blue')
    
    plt.annotate('Complexity penalty\nincreases linearly',
                xy=(8, 4), xytext=(8, 6),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='red')
    
    plt.annotate('Marginal likelihood\nbalances fit vs. complexity',
                xy=(5, 4.5), xytext=(2, 2),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save plot
    bayes_balance_file = os.path.join(save_dir, "bayesian_occam_balance.png")
    plt.savefig(bayes_balance_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [occams_razor_file, model_complexity_file, bayes_balance_file]

# Step 3: Compute posterior probabilities from log model evidences
def compute_posterior_probabilities():
    """Compute posterior probabilities from the given log model evidences."""
    print("\nStep 3: Computing Posterior Probabilities from Log Model Evidences")
    print("="*80)
    
    # Given log model evidences
    log_evidence_1 = -45.2
    log_evidence_2 = -42.8
    log_evidence_3 = -43.1
    
    print(f"Given log model evidences:")
    print(f"log p(y|M₁) = {log_evidence_1}")
    print(f"log p(y|M₂) = {log_evidence_2}")
    print(f"log p(y|M₃) = {log_evidence_3}")
    print()
    
    # Convert to linear scale (need to be careful about numerical precision)
    # First, find the maximum log evidence to prevent numerical overflow
    max_log_evidence = max(log_evidence_1, log_evidence_2, log_evidence_3)
    
    # Subtract the maximum value for numerical stability before exponentiation
    evidence_1 = np.exp(log_evidence_1 - max_log_evidence)
    evidence_2 = np.exp(log_evidence_2 - max_log_evidence)
    evidence_3 = np.exp(log_evidence_3 - max_log_evidence)
    
    print("Converting log evidences to linear scale (with normalization):")
    print(f"p(y|M₁) ∝ exp({log_evidence_1} - {max_log_evidence}) = {evidence_1:.6f}")
    print(f"p(y|M₂) ∝ exp({log_evidence_2} - {max_log_evidence}) = {evidence_2:.6f}")
    print(f"p(y|M₃) ∝ exp({log_evidence_3} - {max_log_evidence}) = {evidence_3:.6f}")
    print()
    
    # Calculate the sum for normalization
    evidence_sum = evidence_1 + evidence_2 + evidence_3
    
    # Calculate posterior probabilities (assuming equal prior probabilities)
    posterior_1 = evidence_1 / evidence_sum
    posterior_2 = evidence_2 / evidence_sum
    posterior_3 = evidence_3 / evidence_sum
    
    print("Computing posterior probabilities (assuming equal prior probabilities):")
    print(f"p(M₁|y) = p(y|M₁) / [p(y|M₁) + p(y|M₂) + p(y|M₃)] = {posterior_1:.6f} = {posterior_1*100:.2f}%")
    print(f"p(M₂|y) = p(y|M₂) / [p(y|M₁) + p(y|M₂) + p(y|M₃)] = {posterior_2:.6f} = {posterior_2*100:.2f}%")
    print(f"p(M₃|y) = p(y|M₃) / [p(y|M₁) + p(y|M₂) + p(y|M₃)] = {posterior_3:.6f} = {posterior_3*100:.2f}%")
    print()
    
    print("Interpretation:")
    print(f"Based on the given log evidences and assuming equal prior probabilities, ")
    print(f"Model 2 has the highest posterior probability ({posterior_2*100:.2f}%), ")
    print(f"followed by Model 3 ({posterior_3*100:.2f}%) and then Model 1 ({posterior_1*100:.2f}%).")
    print(f"This suggests that Model 2, which includes house size and number of bedrooms, ")
    print(f"is most supported by the data when accounting for both fit and complexity.")
    print()
    
    # Create a visualization of the posterior probabilities
    plt.figure(figsize=(10, 6))
    
    models = ['Model 1\n(Size Only)', 'Model 2\n(Size + Bedrooms)', 'Model 3\n(Size + Bedrooms + Age)']
    posteriors = [posterior_1, posterior_2, posterior_3]
    evidences = [evidence_1, evidence_2, evidence_3]
    log_evidences = [log_evidence_1, log_evidence_2, log_evidence_3]
    
    bars = plt.bar(models, posteriors, color=['blue', 'green', 'red'])
    
    # Add text labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2., 
                 bar.get_height() + 0.01, 
                 f'{posteriors[i]*100:.1f}%', 
                 ha='center', va='bottom', fontsize=12)
    
    plt.title('Posterior Probabilities of Housing Price Models', fontsize=14)
    plt.ylabel('Posterior Probability', fontsize=12)
    plt.ylim(0, max(posteriors) * 1.2)  # Add some space for the labels
    
    # Add log evidences as a second line of text
    for i, model in enumerate(models):
        plt.text(i, 0.02, f'log p(y|M) = {log_evidences[i]}', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    posterior_prob_file = os.path.join(save_dir, "posterior_probabilities.png")
    plt.savefig(posterior_prob_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a pie chart for the posterior probabilities
    plt.figure(figsize=(8, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    explode = (0.05, 0.1, 0.05)  # explode the 2nd slice (Model 2) as it has highest probability
    
    plt.pie(posteriors, explode=explode, labels=models, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 12})
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Model Posterior Probability Distribution', fontsize=14)
    
    # Save plot
    posterior_pie_file = os.path.join(save_dir, "posterior_probabilities_pie.png")
    plt.savefig(posterior_pie_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [posterior_prob_file, posterior_pie_file], posteriors

# Step 4: Explain relationship between Bayesian model selection and information criteria
def explain_bayesian_info_criteria():
    """Explain the relationship between Bayesian model selection and information criteria like BIC."""
    print("\nStep 4: Relationship Between Bayesian Model Selection and Information Criteria")
    print("="*80)
    print("The Bayesian Information Criterion (BIC) is an approximation to the log marginal likelihood")
    print("that is derived from the Laplace approximation of the marginal likelihood integral.")
    print()
    print("The BIC is defined as:")
    print("BIC = -2 * ln(L) + k * ln(n)")
    print("where:")
    print("  - ln(L) is the maximized log-likelihood of the model")
    print("  - k is the number of parameters in the model")
    print("  - n is the number of data points")
    print()
    print("The relationship to Bayesian model selection is as follows:")
    print()
    print("1. Approximation to log marginal likelihood:")
    print("   log p(y|M_i) ≈ log p(y|θ_MAP,M_i) - (k/2) * log(n) + constant terms")
    print("   where θ_MAP is the maximum a posteriori parameter estimate.")
    print()
    print("2. BIC as an approximation:")
    print("   BIC = -2 * log p(y|θ_MLE,M_i) + k * log(n)")
    print("   = -2 * [log p(y|M_i) + (k/2) * log(n) - constant terms]")
    print()
    print("3. Model selection:")
    print("   - Lower BIC indicates higher marginal likelihood and thus higher posterior probability")
    print("   - The k*log(n) term acts as a complexity penalty, implementing Occam's razor")
    print("   - As sample size n increases, the penalty for complexity becomes stronger")
    print()
    print("4. Comparison with other criteria:")
    print("   - AIC (Akaike Information Criterion): -2*ln(L) + 2k")
    print("   - BIC penalizes complexity more strongly than AIC when n > 7")
    print("   - BIC is consistent (selects the true model as n→∞ if it's in the candidate set)")
    print("   - AIC is efficient (minimizes prediction error)")
    print()
    print("5. Advantages of full Bayesian approach over BIC:")
    print("   - Incorporates prior information about parameters")
    print("   - Provides full posterior distribution, not just a point selection")
    print("   - Can handle model uncertainty through Bayesian Model Averaging")
    print("   - More accurate for small sample sizes")
    print()
    
    # Create visualizations for BIC and Bayesian model selection
    # 1. Comparison of BIC and AIC penalties
    plt.figure(figsize=(10, 6))
    
    # Parameters
    k_values = np.arange(1, 11)  # Number of parameters from 1 to 10
    sample_sizes = [10, 20, 50, 100, 1000]
    
    # Create plot for BIC penalty
    for n in sample_sizes:
        bic_penalty = k_values * np.log(n)
        plt.plot(k_values, bic_penalty, 'o-', linewidth=2, 
                 label=f'BIC penalty (n={n})', alpha=0.7)
    
    # Add AIC penalty for comparison
    aic_penalty = 2 * k_values
    plt.plot(k_values, aic_penalty, 's-', linewidth=2, color='black', 
             label='AIC penalty', alpha=0.9)
    
    plt.title('Complexity Penalties in Information Criteria', fontsize=14)
    plt.xlabel('Number of Parameters (k)', fontsize=12)
    plt.ylabel('Penalty Term', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Add annotations
    plt.annotate('BIC penalty increases with sample size',
                xy=(8, 40), xytext=(5, 50),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10)
    
    plt.annotate('AIC penalty is constant\nregardless of sample size',
                xy=(8, 16), xytext=(3, 20),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    info_criteria_file = os.path.join(save_dir, "information_criteria_comparison.png")
    plt.savefig(info_criteria_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. BIC approximation to marginal likelihood
    plt.figure(figsize=(12, 6))
    
    # Create some simulated data to compare BIC to marginal likelihood
    np.random.seed(42)
    n_samples = 30
    true_order = 2  # True model is quadratic
    
    # Generate x values
    x = np.linspace(-1, 1, n_samples)
    
    # Generate y values from a quadratic function plus noise
    y_true = 1 + 2*x - 1.5*x**2
    noise_level = 0.5
    y = y_true + np.random.normal(0, noise_level, n_samples)
    
    # Function to calculate the marginal likelihood for polynomial regression
    # Assuming conjugate priors: w ~ N(0, alpha^-1 I) and likelihood with precision beta
    def calculate_log_marginal_likelihood(x, y, degree, alpha=0.1, beta=1/noise_level**2):
        # Create design matrix
        X = np.vstack([x**i for i in range(degree + 1)]).T
        
        # Calculate posterior precision matrix
        posterior_precision = alpha * np.eye(degree + 1) + beta * X.T @ X
        
        # Calculate posterior mean
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = beta * posterior_covariance @ X.T @ y
        
        # Calculate log marginal likelihood
        n = len(y)
        log_ml = -0.5 * n * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(alpha * np.eye(degree + 1))) \
              - 0.5 * np.log(np.linalg.det(posterior_precision)) - 0.5 * beta * np.sum((y - X @ posterior_mean)**2) \
              - 0.5 * alpha * np.sum(posterior_mean**2)
              
        return log_ml, posterior_mean, posterior_covariance
    
    # Function to calculate BIC
    def calculate_bic(x, y, degree):
        # Create design matrix
        X = np.vstack([x**i for i in range(degree + 1)]).T
        
        # Fit the model using least squares
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate the residual sum of squares
        rss = np.sum((y - X @ w)**2)
        
        # Calculate the log-likelihood (assuming Gaussian errors)
        n = len(y)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * rss / n) - 0.5 * rss
        
        # Calculate BIC
        k = degree + 1  # Number of parameters
        bic = -2 * log_likelihood + k * np.log(n)
        
        return bic, w
    
    # Compare log marginal likelihood and BIC for different model orders
    max_degree = 6
    log_marginal_likelihoods = []
    bic_values = []
    posterior_means = []
    
    for degree in range(1, max_degree + 1):
        log_ml, post_mean, _ = calculate_log_marginal_likelihood(x, y, degree)
        bic, mle_weights = calculate_bic(x, y, degree)
        
        log_marginal_likelihoods.append(log_ml)
        bic_values.append(bic)
        posterior_means.append(post_mean)
    
    # Plot the comparison
    degrees = np.arange(1, max_degree + 1)
    
    # Create a two-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Negative BIC (higher is better) and log marginal likelihood
    ax1.plot(degrees, log_marginal_likelihoods, 'bo-', linewidth=2, 
             label='Log Marginal Likelihood')
    
    # Normalize BIC for easier comparison (negate and shift)
    normalized_bic = -np.array(bic_values)
    normalized_bic = normalized_bic - np.min(normalized_bic) + np.min(log_marginal_likelihoods)
    
    ax1.plot(degrees, normalized_bic, 'ro-', linewidth=2, 
             label='Normalized -BIC')
    
    ax1.axvline(x=true_order, color='g', linestyle='--', alpha=0.7, 
                label=f'True Model Order = {true_order}')
    
    ax1.set_title('Log Marginal Likelihood vs. BIC', fontsize=14)
    ax1.set_xlabel('Polynomial Degree', fontsize=12)
    ax1.set_ylabel('Log Probability / -BIC (scaled)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    
    # Plot 2: Model fits
    ax2.scatter(x, y, color='black', alpha=0.7, label='Data')
    ax2.plot(x, y_true, 'g-', linewidth=2, label='True Function')
    
    x_fine = np.linspace(-1, 1, 100)
    
    # Plot model fits
    selected_degrees = [1, 2, 5]  # Linear, quadratic, and higher-order
    line_styles = ['-', '--', '-.']
    colors = ['blue', 'red', 'purple']
    
    for i, degree in enumerate(selected_degrees):
        X_fine = np.vstack([x_fine**j for j in range(degree + 1)]).T
        y_pred = X_fine @ posterior_means[degree-1]
        
        ax2.plot(x_fine, y_pred, color=colors[i], linestyle=line_styles[i], linewidth=2,
                 label=f'Degree {degree} Fit')
    
    ax2.set_title('Polynomial Fits of Different Degrees', fontsize=14)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    bic_ml_file = os.path.join(save_dir, "bic_marginal_likelihood.png")
    plt.savefig(bic_ml_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return [info_criteria_file, bic_ml_file]

# Main program
if __name__ == "__main__":
    print("Solution to Question 22: Bayesian Model Selection")
    print("="*80)
    
    explain_marginal_likelihood()
    occams_razor_files = explain_occams_razor()
    posterior_files, posteriors = compute_posterior_probabilities()
    info_criteria_files = explain_bayesian_info_criteria()
    
    print("\nSummary of Results:")
    print("="*80)
    print(f"1. The marginal likelihood p(y|M_i) is calculated by integrating over the parameter space: ")
    print(f"   p(y|M_i) = ∫ p(y|θ,M_i) p(θ|M_i) dθ")
    print()
    print(f"2. Occam's razor is naturally incorporated in Bayesian model selection because the")
    print(f"   marginal likelihood penalizes complexity by spreading prior probability over larger parameter spaces.")
    print()
    print(f"3. Posterior model probabilities (assuming equal priors):")
    print(f"   Model 1: {posteriors[0]*100:.2f}%")
    print(f"   Model 2: {posteriors[1]*100:.2f}%")
    print(f"   Model 3: {posteriors[2]*100:.2f}%")
    print()
    print(f"4. BIC is an approximation of -2 times the log marginal likelihood plus a constant.")
    print(f"   Both implement Occam's razor by penalizing model complexity.")
    print()
    print(f"Generated visualizations have been saved to: {save_dir}")
    for file_path in occams_razor_files + posterior_files + info_criteria_files:
        print(f"- {os.path.basename(file_path)}") 