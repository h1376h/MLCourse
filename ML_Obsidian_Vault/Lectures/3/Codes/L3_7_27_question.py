import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

def create_directories():
    """Create necessary directories for saving plots"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    question_dir = os.path.join(images_dir, "L3_7_Quiz_27")
    
    os.makedirs(question_dir, exist_ok=True)
    
    return question_dir

def regularization_paths_visualization():
    """Generate regularization path plots comparing Ridge and Lasso regression"""
    save_dir = create_directories()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic data for a linear model
    n_features = 10
    true_weights = np.zeros(n_features)
    true_weights[0:3] = [2.5, -1.8, 3.0]  # Only 3 features are relevant
    
    # Function to simulate regularization effect
    def simulate_regularization(method='ridge', lambda_values=None):
        if lambda_values is None:
            lambda_values = np.logspace(-2, 2, 20)
        
        coefficients = np.zeros((len(lambda_values), n_features))
        
        for i, lam in enumerate(lambda_values):
            # Simulate weights after regularization
            if method == 'ridge':
                # For ridge, all coefficients shrink proportionally
                shrinkage_factor = 1 / (1 + lam)
                coefficients[i] = true_weights * shrinkage_factor
            elif method == 'lasso':
                # For lasso, small coefficients get set to zero
                # This is a simplified simulation
                for j in range(n_features):
                    if abs(true_weights[j]) > lam / 2:
                        coefficients[i, j] = true_weights[j] - np.sign(true_weights[j]) * lam / 2
                    else:
                        coefficients[i, j] = 0
        
        return coefficients
    
    # Generate data
    lambda_values = np.logspace(-2, 2, 20)
    ridge_coefficients = simulate_regularization('ridge', lambda_values)
    lasso_coefficients = simulate_regularization('lasso', lambda_values)
    
    # Create figures
    plt.figure(figsize=(10, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1])
    
    # Plot Ridge coefficients
    ax1 = plt.subplot(gs[0])
    for i in range(n_features):
        if i < 3:  # Relevant features
            ax1.plot(np.log10(lambda_values), ridge_coefficients[:, i], 
                    linewidth=2, label=f'Feature {i+1} (Relevant)')
        else:  # Irrelevant features
            ax1.plot(np.log10(lambda_values), ridge_coefficients[:, i], 
                    'k--', alpha=0.3, linewidth=1)
    
    ax1.set_title('Ridge Regression (L2): Coefficient Paths', fontsize=14)
    ax1.set_xlabel(r'log($\lambda$)', fontsize=12)
    ax1.set_ylabel('Coefficient Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Plot Lasso coefficients
    ax2 = plt.subplot(gs[1])
    for i in range(n_features):
        if i < 3:  # Relevant features
            ax2.plot(np.log10(lambda_values), lasso_coefficients[:, i], 
                    linewidth=2, label=f'Feature {i+1} (Relevant)')
        else:  # Irrelevant features
            ax2.plot(np.log10(lambda_values), lasso_coefficients[:, i], 
                    'k--', alpha=0.3, linewidth=1)
    
    ax2.set_title('Lasso Regression (L1): Coefficient Paths', fontsize=14)
    ax2.set_xlabel(r'log($\lambda$)', fontsize=12)
    ax2.set_ylabel('Coefficient Value', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regularization_paths.png'), dpi=300)
    plt.close()
    
    print(f"Regularization path plots saved to {save_dir}")

if __name__ == "__main__":
    regularization_paths_visualization() 