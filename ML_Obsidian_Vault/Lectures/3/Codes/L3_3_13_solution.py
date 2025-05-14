import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_13")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
np.random.seed(42)  # For reproducibility

def normal_pdf(x, mu, sigma_squared):
    """
    Calculate the probability density function (PDF) of a normal distribution.
    
    Parameters:
    x (float or array): The point(s) at which to evaluate the PDF
    mu (float): The mean of the normal distribution
    sigma_squared (float): The variance of the normal distribution
    
    Returns:
    float or array: The PDF evaluated at x
    """
    return (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-(x - mu)**2 / (2 * sigma_squared))

def step1_pdf_derivation():
    """
    Step 1: Write down the probability density function for observing y^(i) given x^(i), w, and σ^2
    """
    print("\n==== Step 1: PDF for a single observation ====")
    print("For a linear regression model: y = w^T x + ε where ε ~ N(0, σ^2)")
    print("The probability density function for observing y^(i) given x^(i), w, and σ^2 is:")
    print("p(y^(i) | x^(i), w, σ^2) = (1 / √(2πσ^2)) * exp(-(y^(i) - w^T x^(i))^2 / (2σ^2))")
    print("\nThis is the normal distribution PDF with:")
    print("- Mean: μ = w^T x^(i)")
    print("- Variance: σ^2")
    print("\nIntuitively, this means y^(i) is normally distributed around the predicted value w^T x^(i)")
    print("with variance σ^2, reflecting the noise in our observations.")
    
    # Create a visualization of the PDF
    plt.figure(figsize=(10, 6))
    
    # Generate some example data
    x = 2.5  # example feature value
    w = 1.2  # example weight
    sigma_squared_values = [0.2, 1.0, 3.0]  # different noise levels
    
    # Calculate the mean (predicted value)
    mu = w * x
    
    # Range of y values to plot
    y_range = np.linspace(mu - 4, mu + 4, 1000)
    
    # Plot PDF for different noise levels
    colors = ['blue', 'green', 'red']
    for i, sigma_squared in enumerate(sigma_squared_values):
        pdf_values = normal_pdf(y_range, mu, sigma_squared)
        plt.plot(y_range, pdf_values, color=colors[i], 
                 label=f'σ² = {sigma_squared}')
        
        # Mark the mean (w^T x)
        plt.axvline(x=mu, color=colors[i], linestyle='--', alpha=0.5)
    
    # Add vertical line at the mean
    plt.axvline(x=mu, color='black', linestyle='-', alpha=0.8, label='Mean (w^T x)')
    
    # Add annotations
    plt.annotate(f'Mean (w^T x = {mu})', xy=(mu, 0.05), 
                 xytext=(mu+0.5, 0.3), 
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.title('Probability Density Function p(y|x,w,σ²)')
    plt.xlabel('Target value (y)')
    plt.ylabel('Probability density')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step1_pdf.png'), dpi=300, bbox_inches='tight')
    
    return mu, sigma_squared_values

def step2_likelihood_function(mu, sigma_squared_values):
    """
    Step 2: Construct the likelihood function for a dataset with n observations
    """
    print("\n==== Step 2: Likelihood Function for n Observations ====")
    print("The likelihood function is the joint probability of all observations, assuming independence:")
    print("L(w, σ^2 | X, y) = p(y | X, w, σ^2) = ∏(i=1 to n) p(y^(i) | x^(i), w, σ^2)")
    print("\nSubstituting the PDF from Step 1:")
    print("L(w, σ^2 | X, y) = ∏(i=1 to n) (1 / √(2πσ^2)) * exp(-(y^(i) - w^T x^(i))^2 / (2σ^2))")
    print("\nThe likelihood represents how probable our observed data is given our model parameters.")
    print("Higher likelihood suggests better parameter values.")
    
    # Create a visualization of the likelihood function
    plt.figure(figsize=(10, 6))
    
    # Generate some synthetic data
    n_samples = 5  # number of samples
    x_true = np.linspace(1, 5, n_samples)  # example feature values
    w_true = 1.2  # true weight
    sigma_squared = 0.5  # true noise variance
    
    # Generate target values with noise
    np.random.seed(42)
    y_true = w_true * x_true + np.random.normal(0, np.sqrt(sigma_squared), n_samples)
    
    # Calculate likelihood for different weight values
    w_range = np.linspace(0, 2.5, 100)
    
    likelihood_values = []
    for w in w_range:
        # Calculate individual likelihoods for each observation
        individual_likelihoods = []
        for i in range(n_samples):
            mu_i = w * x_true[i]
            p_i = normal_pdf(y_true[i], mu_i, sigma_squared)
            individual_likelihoods.append(p_i)
        
        # Multiply all individual likelihoods to get the joint likelihood
        joint_likelihood = np.prod(individual_likelihoods)
        likelihood_values.append(joint_likelihood)
    
    # Plot likelihood function
    plt.plot(w_range, likelihood_values, color='blue', linewidth=2)
    
    # Mark the true weight value
    plt.axvline(x=w_true, color='red', linestyle='--', 
                label=f'True w = {w_true}')
    
    # Find the maximum likelihood estimator (MLE)
    w_mle = w_range[np.argmax(likelihood_values)]
    plt.axvline(x=w_mle, color='green', linestyle='--', 
                label=f'MLE w = {w_mle:.4f}')
    
    plt.title('Likelihood Function L(w, σ²|X, y)')
    plt.xlabel('Weight parameter (w)')
    plt.ylabel('Likelihood')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step2_likelihood.png'), dpi=300, bbox_inches='tight')
    
    # Create a small figure showing the data points and regression lines
    plt.figure(figsize=(6, 4))
    plt.scatter(x_true, y_true, color='black', label='Data points')
    
    # Plot the true line
    x_line = np.linspace(0, 6, 100)
    plt.plot(x_line, w_true * x_line, color='red', label=f'True (w={w_true})')
    
    # Plot the MLE line
    plt.plot(x_line, w_mle * x_line, color='green', label=f'MLE (w={w_mle:.4f})')
    
    plt.title('Data and Regression Lines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step2_data_regression.png'), dpi=300, bbox_inches='tight')
    
    return x_true, y_true, w_true, sigma_squared

def step3_log_likelihood(x_true, y_true, w_true, sigma_squared):
    """
    Step 3: Derive the log-likelihood function and simplify it
    """
    print("\n==== Step 3: Log-Likelihood Function ====")
    print("Taking the natural logarithm of the likelihood function (which is monotonically increasing,")
    print("so maximizing log-likelihood is equivalent to maximizing likelihood):")
    print("\nln L(w, σ^2 | X, y) = ln[∏(i=1 to n) (1 / √(2πσ^2)) * exp(-(y^(i) - w^T x^(i))^2 / (2σ^2))]")
    print("\nUsing the properties of logarithms:")
    print("ln L(w, σ^2 | X, y) = ∑(i=1 to n) ln[(1 / √(2πσ^2)) * exp(-(y^(i) - w^T x^(i))^2 / (2σ^2))]")
    print("\nSimplifying:")
    print("ln L(w, σ^2 | X, y) = ∑(i=1 to n) [ln(1 / √(2πσ^2)) + ln(exp(-(y^(i) - w^T x^(i))^2 / (2σ^2)))]")
    print("\nFurther simplification:")
    print("ln L(w, σ^2 | X, y) = ∑(i=1 to n) [-(1/2)ln(2πσ^2) - (y^(i) - w^T x^(i))^2 / (2σ^2)]")
    print("\nCollecting terms:")
    print("ln L(w, σ^2 | X, y) = -(n/2)ln(2πσ^2) - (1/(2σ^2)) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nThis is our final log-likelihood function. The second term contains the sum of squared errors.")
    
    # Create a visualization of the log-likelihood function
    plt.figure(figsize=(10, 6))
    
    # Calculate log-likelihood for different weight values
    w_range = np.linspace(0, 2.5, 100)
    
    log_likelihood_values = []
    for w in w_range:
        # Calculate predicted values
        y_pred = w * x_true
        
        # Calculate sum of squared errors
        sse = np.sum((y_true - y_pred)**2)
        
        # Calculate log-likelihood
        n = len(x_true)
        log_likelihood = -(n/2) * np.log(2 * np.pi * sigma_squared) - (1/(2*sigma_squared)) * sse
        
        log_likelihood_values.append(log_likelihood)
    
    # Plot log-likelihood function
    plt.plot(w_range, log_likelihood_values, color='blue', linewidth=2)
    
    # Mark the true weight value
    plt.axvline(x=w_true, color='red', linestyle='--', 
                label=f'True w = {w_true}')
    
    # Find the maximum log-likelihood estimator
    w_mle = w_range[np.argmax(log_likelihood_values)]
    plt.axvline(x=w_mle, color='green', linestyle='--', 
                label=f'MLE w = {w_mle:.4f}')
    
    plt.title('Log-Likelihood Function ln L(w, σ²|X, y)')
    plt.xlabel('Weight parameter (w)')
    plt.ylabel('Log-likelihood')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step3_log_likelihood.png'), dpi=300, bbox_inches='tight')
    
    return w_range, log_likelihood_values

def step4_likelihood_vs_sse(x_true, y_true, w_range, log_likelihood_values, sigma_squared):
    """
    Step 4: Show that maximizing log-likelihood is equivalent to minimizing sum of squared errors
    """
    print("\n==== Step 4: Maximizing Log-Likelihood vs. Minimizing SSE ====")
    print("From the log-likelihood function in Step 3:")
    print("ln L(w, σ^2 | X, y) = -(n/2)ln(2πσ^2) - (1/(2σ^2)) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nTo maximize this expression, we need to minimize the negative terms.")
    print("The first term -(n/2)ln(2πσ^2) is constant with respect to w.")
    print("Therefore, maximizing the log-likelihood is equivalent to minimizing:")
    print("(1/(2σ^2)) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nSince (1/(2σ^2)) is a positive constant, this is equivalent to minimizing:")
    print("∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nWhich is precisely the sum of squared errors (SSE).")
    print("Therefore, maximizing the log-likelihood is equivalent to minimizing SSE.")
    
    # Create a visualization comparing log-likelihood and SSE
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Calculate SSE for different weight values
    sse_values = []
    for w in w_range:
        # Calculate predicted values
        y_pred = w * x_true
        
        # Calculate sum of squared errors
        sse = np.sum((y_true - y_pred)**2)
        sse_values.append(sse)
    
    # Plot log-likelihood
    ax1.set_xlabel('Weight parameter (w)')
    ax1.set_ylabel('Log-likelihood', color='blue')
    ax1.plot(w_range, log_likelihood_values, color='blue', linewidth=2, label='Log-likelihood')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second y-axis for SSE
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sum of Squared Errors (SSE)', color='red')
    ax2.plot(w_range, sse_values, color='red', linewidth=2, label='SSE')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Find the maximum log-likelihood and minimum SSE
    w_max_ll = w_range[np.argmax(log_likelihood_values)]
    w_min_sse = w_range[np.argmin(sse_values)]
    
    # Add vertical lines
    plt.axvline(x=w_max_ll, color='green', linestyle='--', 
                label=f'Max LL / Min SSE (w = {w_max_ll:.4f})')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Comparison of Log-Likelihood and Sum of Squared Errors')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step4_ll_vs_sse.png'), dpi=300, bbox_inches='tight')
    
    # Create a 3D visualization to show the relationship
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a grid of w and sigma^2 values
    w_grid = np.linspace(0.5, 2.0, 20)
    sigma_squared_grid = np.linspace(0.1, 1.0, 20)
    W, SIGMA = np.meshgrid(w_grid, sigma_squared_grid)
    
    # Calculate log-likelihood for each combination
    LL = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w = W[i, j]
            sigma_sq = SIGMA[i, j]
            
            # Calculate predicted values
            y_pred = w * x_true
            
            # Calculate sum of squared errors
            sse = np.sum((y_true - y_pred)**2)
            
            # Calculate log-likelihood
            n = len(x_true)
            ll = -(n/2) * np.log(2 * np.pi * sigma_sq) - (1/(2*sigma_sq)) * sse
            
            LL[i, j] = ll
    
    # Create the surface plot
    surf = ax.plot_surface(W, SIGMA, LL, cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Log-Likelihood')
    
    # Mark the maximum log-likelihood point
    max_idx = np.unravel_index(np.argmax(LL), LL.shape)
    max_w = W[max_idx]
    max_sigma = SIGMA[max_idx]
    max_ll = LL[max_idx]
    
    ax.scatter([max_w], [max_sigma], [max_ll], color='red', s=100, label='Maximum Log-Likelihood')
    
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Variance (σ²)')
    ax.set_zlabel('Log-Likelihood')
    ax.set_title('Log-Likelihood Surface for Different w and σ² Values')
    ax.legend()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step4_ll_surface.png'), dpi=300, bbox_inches='tight')
    
    return sse_values, w_max_ll

def step5_mle_for_sigma_squared(x_true, y_true, w_true, w_mle, sigma_squared):
    """
    Step 5: Derive the maximum likelihood estimator for the noise variance sigma^2
    """
    print("\n==== Step 5: MLE for Noise Variance σ² ====")
    print("To find the MLE for σ², we take the partial derivative of the log-likelihood with respect to σ²")
    print("and set it to zero.")
    print("\nRecall the log-likelihood function:")
    print("ln L(w, σ^2 | X, y) = -(n/2)ln(2πσ^2) - (1/(2σ^2)) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nTaking the partial derivative with respect to σ²:")
    print("∂/∂σ² ln L(w, σ^2 | X, y) = -(n/2) * (1/σ^2) + (1/2) * (1/σ^4) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nSetting this equal to zero:")
    print("-(n/2) * (1/σ^2) + (1/2) * (1/σ^4) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2 = 0")
    print("\nSolving for σ²:")
    print("(n/2) * (1/σ^2) = (1/2) * (1/σ^4) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("n * σ^2 = (1/σ^2) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("n * (σ^2)^2 = ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("(σ^2)^2 = (1/n) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("σ² = (1/n) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nTherefore, the MLE for σ² is the average of the squared residuals:")
    print("σ²_MLE = (1/n) * ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("This is the mean squared error (MSE) between the observed and predicted values.")
    
    # Calculate the MLE for sigma^2 using the optimal w
    y_pred_mle = w_mle * x_true
    residuals_mle = y_true - y_pred_mle
    sse_mle = np.sum(residuals_mle**2)
    n = len(x_true)
    sigma_squared_mle = sse_mle / n
    
    print(f"\nFor our example dataset with w_MLE = {w_mle:.4f}:")
    print(f"σ²_MLE = (1/{n}) * {sse_mle:.6f} = {sigma_squared_mle:.6f}")
    print(f"True σ² = {sigma_squared}")
    
    # Create a visualization to demonstrate the MLE for sigma^2
    plt.figure(figsize=(10, 6))
    
    # Calculate sigma^2 MLE for different w values
    sigma_squared_mle_values = []
    for w in w_range:
        # Calculate predicted values
        y_pred = w * x_true
        
        # Calculate residuals and SSE
        residuals = y_true - y_pred
        sse = np.sum(residuals**2)
        
        # Calculate sigma^2 MLE
        sigma_squared_mle_w = sse / n
        
        sigma_squared_mle_values.append(sigma_squared_mle_w)
    
    # Plot sigma^2 MLE vs w
    plt.plot(w_range, sigma_squared_mle_values, color='purple', linewidth=2)
    
    # Mark the MLE estimates
    plt.axvline(x=w_mle, color='green', linestyle='--', 
                label=f'w_MLE = {w_mle:.4f}')
    plt.axhline(y=sigma_squared_mle, color='blue', linestyle='--',
                label=f'σ²_MLE = {sigma_squared_mle:.4f}')
    
    # Mark true sigma^2
    plt.axhline(y=sigma_squared, color='red', linestyle='-',
                label=f'True σ² = {sigma_squared}')
    
    # Mark the minimum point
    plt.plot(w_mle, sigma_squared_mle, 'ro', markersize=8)
    
    plt.title('MLE for σ² as a Function of w')
    plt.xlabel('Weight parameter (w)')
    plt.ylabel('σ² MLE')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step5_sigma_squared_mle.png'), dpi=300, bbox_inches='tight')
    
    # Create an additional figure showing the effect of sigma^2 on the likelihood
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig)
    
    # Top left: Original data with error bars
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(x_true, y_true, yerr=np.sqrt(sigma_squared), fmt='o', 
                 color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    
    # Plot the true and MLE models
    x_line = np.linspace(0, 6, 100)
    ax1.plot(x_line, w_true * x_line, color='red', label=f'True (w={w_true})')
    ax1.plot(x_line, w_mle * x_line, color='green', label=f'MLE (w={w_mle:.4f})')
    
    ax1.set_title('Data with Error Bars (True σ)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    
    # Top right: Data with MLE sigma error bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(x_true, y_true, yerr=np.sqrt(sigma_squared_mle), fmt='o', 
                 color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    
    # Plot the true and MLE models
    ax2.plot(x_line, w_true * x_line, color='red', label=f'True (w={w_true})')
    ax2.plot(x_line, w_mle * x_line, color='green', label=f'MLE (w={w_mle:.4f})')
    
    ax2.set_title('Data with Error Bars (MLE σ)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)
    
    # Bottom: Distribution of residuals
    ax3 = fig.add_subplot(gs[1, :])
    
    # Calculate residuals for MLE
    residuals = y_true - w_mle * x_true
    
    # Plot histogram of residuals
    ax3.hist(residuals, bins=10, density=True, alpha=0.6, color='skyblue',
             label='Residual distribution')
    
    # Plot normal distribution with MLE sigma
    x_range = np.linspace(min(residuals) - 1, max(residuals) + 1, 1000)
    normal_dist = normal_pdf(x_range, 0, sigma_squared_mle)
    ax3.plot(x_range, normal_dist, color='green', linewidth=2,
             label=f'N(0, σ²_MLE = {sigma_squared_mle:.4f})')
    
    # Plot normal distribution with true sigma
    normal_dist_true = normal_pdf(x_range, 0, sigma_squared)
    ax3.plot(x_range, normal_dist_true, color='red', linewidth=2, linestyle='--',
             label=f'N(0, σ²_true = {sigma_squared})')
    
    ax3.set_title('Distribution of Residuals')
    ax3.set_xlabel('Residual value (y - w_MLE^T x)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'step5_residuals_analysis.png'), dpi=300, bbox_inches='tight')
    
    return sigma_squared_mle

def summarize_results(w_mle, sigma_squared_mle, sigma_squared, saved_files):
    """
    Summarize the results of all steps
    """
    print("\n==== Summary of Results ====")
    print("1. We derived the probability density function for observing a target value y^(i)")
    print("   given input x^(i) and parameters w and σ^2.")
    print("\n2. We constructed the likelihood function for a dataset with n observations,")
    print("   which is the product of individual PDFs.")
    print("\n3. We derived the log-likelihood function and simplified it to:")
    print("   ln L(w, σ^2 | X, y) = -(n/2)ln(2πσ^2) - (1/(2σ^2)) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\n4. We showed that maximizing the log-likelihood is equivalent to minimizing")
    print("   the sum of squared errors (SSE).")
    print("\n5. We derived the maximum likelihood estimator for σ^2:")
    print("   σ²_MLE = (1/n) ∑(i=1 to n) (y^(i) - w^T x^(i))^2")
    print("\nFor our example dataset:")
    print(f"- MLE weight estimate: w_MLE = {w_mle:.4f}")
    print(f"- MLE variance estimate: σ²_MLE = {sigma_squared_mle:.6f}")
    print(f"- True variance: σ²_true = {sigma_squared}")
    print(f"\nVisualizations saved to: {', '.join(saved_files)}")

# Run all steps
mu, sigma_squared_values = step1_pdf_derivation()
x_true, y_true, w_true, sigma_squared = step2_likelihood_function(mu, sigma_squared_values)
w_range, log_likelihood_values = step3_log_likelihood(x_true, y_true, w_true, sigma_squared)
sse_values, w_mle = step4_likelihood_vs_sse(x_true, y_true, w_range, log_likelihood_values, sigma_squared)
sigma_squared_mle = step5_mle_for_sigma_squared(x_true, y_true, w_true, w_mle, sigma_squared)

# List saved files
saved_files = [
    os.path.join(save_dir, 'step1_pdf.png'),
    os.path.join(save_dir, 'step2_likelihood.png'),
    os.path.join(save_dir, 'step2_data_regression.png'),
    os.path.join(save_dir, 'step3_log_likelihood.png'),
    os.path.join(save_dir, 'step4_ll_vs_sse.png'),
    os.path.join(save_dir, 'step4_ll_surface.png'),
    os.path.join(save_dir, 'step5_sigma_squared_mle.png'),
    os.path.join(save_dir, 'step5_residuals_analysis.png')
]

# Summarize results
summarize_results(w_mle, sigma_squared_mle, sigma_squared, saved_files) 