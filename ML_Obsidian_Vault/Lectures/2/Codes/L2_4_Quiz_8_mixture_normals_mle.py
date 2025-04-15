import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import os

def mixture_pdf(x, mu1, mu2, sigma=1.0):
    """PDF for a mixture of two normals with equal mixing proportions"""
    return 0.5 * (norm.pdf(x, mu1, sigma) + norm.pdf(x, mu2, sigma))

def log_likelihood(params, data, sigma=1.0):
    """Log-likelihood function for mixture of two normals with equal mixing proportions"""
    mu1, mu2 = params
    n = len(data)
    
    # Calculate likelihood for each data point
    likelihoods = np.zeros(n)
    for i in range(n):
        likelihoods[i] = mixture_pdf(data[i], mu1, mu2, sigma)
    
    # Return negative log-likelihood (for minimization)
    return -np.sum(np.log(likelihoods))

def score_function(params, data, sigma=1.0):
    """Score function (gradient of log-likelihood) for mixture of two normals"""
    mu1, mu2 = params
    n = len(data)
    
    # Calculate gradients
    grad_mu1 = np.zeros(n)
    grad_mu2 = np.zeros(n)
    
    for i in range(n):
        x = data[i]
        p1 = norm.pdf(x, mu1, sigma)
        p2 = norm.pdf(x, mu2, sigma)
        mixture = p1 + p2
        
        grad_mu1[i] = (p1 * (x - mu1) / sigma**2) / mixture
        grad_mu2[i] = (p2 * (x - mu2) / sigma**2) / mixture
    
    return np.array([np.sum(grad_mu1), np.sum(grad_mu2)])

def plot_mixture_pdfs(mu1_values, mu2_values, sigma=1.0, save_path=None):
    """Plot mixture PDFs for different mean parameter values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-6, 6, 1000)
    
    for mu1, mu2 in zip(mu1_values, mu2_values):
        y = [mixture_pdf(xi, mu1, mu2, sigma) for xi in x]
        ax.plot(x, y, linewidth=2, label=f'μ₁={mu1}, μ₂={mu2}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density f(x|μ₁,μ₂,σ)')
    ax.set_title('Mixture of Two Normal Distributions (Equal Mixing)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_likelihood_surface(data, sigma=1.0, save_path=None):
    """Plot the likelihood function surface for a grid of parameter values"""
    # Create a grid of parameter values
    mu1_range = np.linspace(-3, 3, 100)
    mu2_range = np.linspace(-3, 3, 100)
    mu1_grid, mu2_grid = np.meshgrid(mu1_range, mu2_range)
    
    # Calculate negative log-likelihood for each parameter combination
    nll_grid = np.zeros_like(mu1_grid)
    for i in range(len(mu1_range)):
        for j in range(len(mu2_range)):
            nll_grid[j, i] = log_likelihood([mu1_grid[j, i], mu2_grid[j, i]], data, sigma)
    
    # Find MLE parameters - we need to try multiple initializations
    # as the likelihood surface is multimodal
    best_nll = float('inf')
    best_params = None
    
    # Try different initializations
    initializations = [
        [-2, 2],  # Try starting with well-separated means
        [2, -2],  # Try the opposite order
        [-1, 1],  # Try closer values
        [0, 0]    # Try starting at the center
    ]
    
    for init in initializations:
        result = minimize(log_likelihood, init, args=(data, sigma), method='BFGS')
        if result.fun < best_nll:
            best_nll = result.fun
            best_params = result.x
    
    mle_mu1, mle_mu2 = best_params
    
    # Ensure mu1 <= mu2 for consistency (arbitrary labeling)
    if mle_mu1 > mle_mu2:
        mle_mu1, mle_mu2 = mle_mu2, mle_mu1
    
    # Create the surface plot
    fig = plt.figure(figsize=(12, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(mu1_grid, mu2_grid, nll_grid, cmap='viridis', alpha=0.8)
    ax1.scatter([mle_mu1], [mle_mu2], [log_likelihood([mle_mu1, mle_mu2], data, sigma)], 
                color='r', s=50, label='MLE')
    ax1.set_xlabel('μ₁')
    ax1.set_ylabel('μ₂')
    ax1.set_zlabel('Negative Log-Likelihood')
    ax1.set_title('3D Negative Log-Likelihood Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(2, 2, 2)
    levels = np.linspace(np.min(nll_grid), np.min(nll_grid) + 20, 30)
    contour = ax2.contourf(mu1_grid, mu2_grid, nll_grid, levels=levels, cmap='viridis')
    ax2.scatter(mle_mu1, mle_mu2, color='r', s=50, label='MLE')
    ax2.set_xlabel('μ₁')
    ax2.set_ylabel('μ₂')
    ax2.set_title('Contour Plot of Negative Log-Likelihood')
    fig.colorbar(contour, ax=ax2)
    
    # Slice along μ₁ (at fixed MLE μ₂)
    ax3 = fig.add_subplot(2, 2, 3)
    idx = np.argmin(np.abs(mu2_range - mle_mu2))
    ax3.plot(mu1_range, nll_grid[idx, :])
    ax3.axvline(x=mle_mu1, color='r', linestyle='--')
    ax3.set_xlabel('μ₁')
    ax3.set_ylabel('Negative Log-Likelihood')
    ax3.set_title(f'Slice at μ₂ = {mle_mu2:.2f}')
    ax3.grid(True, alpha=0.3)
    
    # Slice along μ₂ (at fixed MLE μ₁)
    ax4 = fig.add_subplot(2, 2, 4)
    idx = np.argmin(np.abs(mu1_range - mle_mu1))
    ax4.plot(mu2_range, nll_grid[:, idx])
    ax4.axvline(x=mle_mu2, color='r', linestyle='--')
    ax4.set_xlabel('μ₂')
    ax4.set_ylabel('Negative Log-Likelihood')
    ax4.set_title(f'Slice at μ₁ = {mle_mu1:.2f}')
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    
    return mle_mu1, mle_mu2

def plot_mle_fit(data, mle_mu1, mle_mu2, sigma=1.0, save_path=None):
    """Plot the fitted MLE distribution against the data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(min(data) - 3, max(data) + 3, 1000)
    y_mle = [mixture_pdf(xi, mle_mu1, mle_mu2, sigma) for xi in x]
    
    # Plot histogram of the data
    ax.hist(data, bins=min(30, len(data)//5), density=True, alpha=0.5, 
            color='blue', label='Observed Data')
    
    # Plot the fitted PDF based on MLE
    ax.plot(x, y_mle, 'r-', linewidth=2, 
            label=f'MLE Fit (μ₁={mle_mu1:.2f}, μ₂={mle_mu2:.2f})')
    
    # Plot the component normal distributions
    ax.plot(x, [0.5 * norm.pdf(xi, mle_mu1, sigma) for xi in x], 'g--', linewidth=1.5, 
            label=f'Component 1 (μ₁={mle_mu1:.2f})')
    ax.plot(x, [0.5 * norm.pdf(xi, mle_mu2, sigma) for xi in x], 'm--', linewidth=1.5, 
            label=f'Component 2 (μ₂={mle_mu2:.2f})')
    
    # Mark the MLE parameters
    ax.axvline(x=mle_mu1, color='g', linestyle=':', alpha=0.7)
    ax.axvline(x=mle_mu2, color='m', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('Maximum Likelihood Estimation for Mixture of Two Normals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_label_switching_problem(save_path=None):
    """Illustrate the label switching problem in mixture models"""
    # Create a grid of parameter values
    mu1_range = np.linspace(-2, 2, 100)
    mu2_range = np.linspace(-2, 2, 100)
    mu1_grid, mu2_grid = np.meshgrid(mu1_range, mu2_range)
    
    # Generate a fixed dataset
    np.random.seed(42)
    true_mu1, true_mu2 = -1.5, 1.5
    sigma = 1.0
    n = 200
    
    # Generate mixture data
    component = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    data = np.zeros(n)
    for i in range(n):
        if component[i] == 0:
            data[i] = np.random.normal(true_mu1, sigma)
        else:
            data[i] = np.random.normal(true_mu2, sigma)
    
    # Calculate negative log-likelihood for each parameter combination
    nll_grid = np.zeros_like(mu1_grid)
    for i in range(len(mu1_range)):
        for j in range(len(mu2_range)):
            nll_grid[j, i] = log_likelihood([mu1_grid[j, i], mu2_grid[j, i]], data, sigma)
    
    # Create the contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    levels = np.linspace(np.min(nll_grid), np.min(nll_grid) + 20, 30)
    contour = ax.contourf(mu1_grid, mu2_grid, nll_grid, levels=levels, cmap='viridis')
    
    # Mark the true parameters
    ax.scatter(true_mu1, true_mu2, color='r', s=100, marker='*', label='True (μ₁,μ₂)')
    ax.scatter(true_mu2, true_mu1, color='g', s=100, marker='*', label='True (μ₂,μ₁)')
    
    # Mark the diagonal line
    ax.plot([-2, 2], [-2, 2], 'r--', linewidth=1.5, alpha=0.7, label='Symmetry Line (μ₁=μ₂)')
    
    # Add arrows to illustrate label switching
    ax.annotate('', xy=(true_mu2, true_mu1), xytext=(true_mu1, true_mu2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    
    ax.set_xlabel('μ₁')
    ax.set_ylabel('μ₂')
    ax.set_title('Label Switching Problem in Mixture Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.colorbar(contour, ax=ax, label='Negative Log-Likelihood')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def plot_em_algorithm_illustration(save_path=None):
    """Illustrate the EM algorithm for mixture models using synthetic data"""
    # Generate synthetic data
    np.random.seed(42)
    true_mu1, true_mu2 = -2, 2
    sigma = 1.0
    n = 200
    
    # Generate mixture data
    component = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    data = np.zeros(n)
    for i in range(n):
        if component[i] == 0:
            data[i] = np.random.normal(true_mu1, sigma)
        else:
            data[i] = np.random.normal(true_mu2, sigma)
    
    # Initial guess for parameters
    mu1_init, mu2_init = -1, 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values for plotting
    x = np.linspace(min(data) - 3, max(data) + 3, 1000)
    
    # Plot histogram of the data
    ax.hist(data, bins=30, density=True, alpha=0.5, 
            color='blue', label='Observed Data')
    
    # Plot the true PDF
    y_true = [mixture_pdf(xi, true_mu1, true_mu2, sigma) for xi in x]
    ax.plot(x, y_true, 'k-', linewidth=2, 
            label=f'True PDF (μ₁={true_mu1}, μ₂={true_mu2})')
    
    # Plot the initial guess
    y_init = [mixture_pdf(xi, mu1_init, mu2_init, sigma) for xi in x]
    ax.plot(x, y_init, 'r--', linewidth=1.5, 
            label=f'Initial Guess (μ₁={mu1_init}, μ₂={mu2_init})')
    
    # Store current parameter values
    mu1_current, mu2_current = mu1_init, mu2_init
    
    # Run a few iterations of EM and plot
    colors = ['g', 'm', 'c', 'y']
    for i in range(4):
        # E-step: Calculate responsibilities
        gamma1 = np.zeros(n)
        gamma2 = np.zeros(n)
        
        for j in range(n):
            p1 = norm.pdf(data[j], mu1_current, sigma)
            p2 = norm.pdf(data[j], mu2_current, sigma)
            denominator = p1 + p2
            gamma1[j] = p1 / denominator
            gamma2[j] = p2 / denominator
        
        # M-step: Update parameters
        mu1_new = np.sum(gamma1 * data) / np.sum(gamma1)
        mu2_new = np.sum(gamma2 * data) / np.sum(gamma2)
        
        # Plot the updated PDF
        y_em = [mixture_pdf(xi, mu1_new, mu2_new, sigma) for xi in x]
        ax.plot(x, y_em, f'{colors[i]}-', linewidth=1.5, 
                label=f'EM Iteration {i+1} (μ₁={mu1_new:.2f}, μ₂={mu2_new:.2f})')
        
        # Update for next iteration
        mu1_current, mu2_current = mu1_new, mu2_new
    
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.set_title('EM Algorithm Iterations for Mixture of Two Normals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()

def main():
    """Generate all visualizations for Question 8 of the L2.4 quiz"""
    # Create synthetic data
    np.random.seed(42)
    true_mu1, true_mu2 = -2, 2
    sigma = 1.0
    n = 200
    
    # Generate mixture data
    component = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    data = np.zeros(n)
    for i in range(n):
        if component[i] == 0:
            data[i] = np.random.normal(true_mu1, sigma)
        else:
            data[i] = np.random.normal(true_mu2, sigma)
    
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_4_Quiz_8")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating visualizations for Question 8 of the L2.4 MLE quiz...")
    
    # 1. Plot PDFs for different mean parameter values
    plot_mixture_pdfs([-2, -1, -1.5], [2, 1, 0], sigma=1.0, 
                     save_path=os.path.join(save_dir, "mixture_pdfs.png"))
    print("1. PDF visualization created")
    
    # 2. Plot likelihood surface
    mle_mu1, mle_mu2 = plot_likelihood_surface(data, sigma=1.0, 
                                             save_path=os.path.join(save_dir, "likelihood_surface.png"))
    print(f"2. Likelihood surface visualization created, MLE μ₁={mle_mu1:.4f}, μ₂={mle_mu2:.4f}")
    
    # 3. Plot MLE fit to data
    plot_mle_fit(data, mle_mu1, mle_mu2, sigma=1.0, 
                save_path=os.path.join(save_dir, "mle_fit.png"))
    print("3. MLE fit visualization created")
    
    # 4. Illustrate label switching problem
    plot_label_switching_problem(save_path=os.path.join(save_dir, "label_switching.png"))
    print("4. Label switching problem visualization created")
    
    # 5. Illustrate EM algorithm
    plot_em_algorithm_illustration(save_path=os.path.join(save_dir, "em_algorithm.png"))
    print("5. EM algorithm illustration created")
    
    print(f"\nSummary of findings:")
    print(f"True parameters: μ₁={true_mu1}, μ₂={true_mu2}")
    print(f"MLE parameter estimates: μ₁={mle_mu1:.4f}, μ₂={mle_mu2:.4f}")
    print(f"\nChallenges in mixture model estimation:")
    print(f"1. The likelihood surface has multiple modes (label switching)")
    print(f"2. The score equations do not have a closed-form solution")
    print(f"3. EM algorithm is typically used for mixture models instead of direct optimization")
    
    print(f"\nAll visualizations have been saved to: {save_dir}")

if __name__ == "__main__":
    main() 