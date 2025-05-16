import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
np.random.seed(42)  # For reproducibility

def statement2_aic_selection():
    """
    Statement 2: When comparing models using information criteria, 
    the model with the highest AIC value should be selected.
    """
    print("\n==== Statement 2: AIC Model Selection ====")
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    true_func = lambda x: 3 + 2*x + 0.5*x**2  # True quadratic function
    y = true_func(X.squeeze()) + np.random.normal(0, 5, n_samples)
    
    # Define sample size (n) for AIC calculation
    n = len(y)
    
    # Print explanation of AIC
    print("\nAkaike Information Criterion (AIC) Explanation:")
    print("AIC is a measure used for model selection that balances:")
    print("  1. Goodness of fit (how well the model explains the data)")
    print("  2. Model complexity (to prevent overfitting)")
    print("\nThe AIC formula is: AIC = n·ln(MSE) + 2k")
    print("  Where:")
    print("  - n is the number of samples")
    print("  - MSE is the mean squared error")
    print("  - k is the number of parameters in the model")
    print("  - The first term (n·ln(MSE)) measures model fit")
    print("  - The second term (2k) penalizes model complexity")
    print("\nLOWER AIC values indicate better models.")
    
    # Define function to calculate AIC
    def calculate_aic(y_true, y_pred, num_params):
        n = len(y_true)
        mse = np.mean((y_true - y_pred)**2)
        aic = n * np.log(mse) + 2 * num_params
        return aic, mse
    
    # Create models of different complexity
    degrees = [1, 2, 3, 5, 10]
    models = []
    predictions = []
    aic_values = []
    mse_values = []
    penalty_values = []
    
    for degree in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        model.fit(X, y)
        models.append(model)
        
        y_pred = model.predict(X)
        predictions.append(y_pred)
        
        # Calculate number of parameters
        num_params = degree + 1  # degree + 1 for polynomial coefficients
        
        # Calculate penalty term (2k)
        penalty = 2 * num_params
        penalty_values.append(penalty)
        
        # Calculate AIC
        aic, mse = calculate_aic(y, y_pred, num_params)
        aic_values.append(aic)
        mse_values.append(mse)
    
    # Find best model according to AIC
    best_idx = np.argmin(aic_values)
    best_degree = degrees[best_idx]
    
    # Print results
    print("\nModel comparison results:")
    print("Degree\tParams\tMSE\t\tPenalty\t\tAIC")
    for i, degree in enumerate(degrees):
        num_params = degree + 1
        print(f"{degree}\t{num_params}\t{mse_values[i]:.2f}\t\t{penalty_values[i]:.2f}\t\t{aic_values[i]:.2f}")
    
    print(f"\nBest model according to AIC: Polynomial degree {best_degree}")
    print(f"This model has the LOWEST AIC value: {aic_values[best_idx]:.2f}")
    
    # Plot 1: Model comparison
    plt.figure(figsize=(10, 6))
    
    # Plot the true function and data points
    x_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    plt.scatter(X, y, color='black', alpha=0.5, label='Data points')
    plt.plot(x_plot, true_func(x_plot.squeeze()), color='blue', linewidth=2, label='True function')
    
    # Plot the model predictions
    for i, degree in enumerate(degrees):
        y_plot = models[i].predict(x_plot)
        plt.plot(x_plot, y_plot, linestyle='--', label=f'Degree {degree}, AIC: {aic_values[i]:.1f}')
    
    # Highlight the best model
    best_y_plot = models[best_idx].predict(x_plot)
    plt.plot(x_plot, best_y_plot, linewidth=3, color='green', 
             label=f'Best model (degree {best_degree})')
    
    plt.title('Model Comparison using AIC', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'statement2_aic_selection.png'), dpi=300, bbox_inches='tight')
    
    # Print explanation of the model selection
    print("\nModel Selection Explanation:")
    print(f"- The degree 1 model (linear) has the lowest complexity but highest error (MSE: {mse_values[0]:.2f})")
    print(f"- The degree 10 model has the lowest error (MSE: {mse_values[-1]:.2f}) but highest complexity")
    print(f"- The degree {best_degree} model provides the best balance between fit and complexity")
    print("- This demonstrates that AIC helps prevent overfitting by penalizing unnecessarily complex models")
    
    # Plot 2: AIC components as a line plot (simpler visualization)
    plt.figure(figsize=(10, 6))
    
    # Calculate components for visualization
    fit_term = [n * np.log(mse) for mse in mse_values]
    
    # Create the plot showing the different components
    plt.plot(degrees, fit_term, 'o-', label='Fit term: n·ln(MSE)', color='blue')
    plt.plot(degrees, penalty_values, 'o-', label='Penalty term: 2k', color='red')
    plt.plot(degrees, aic_values, 'o-', label='Total AIC', color='purple', linewidth=2)
    
    # Highlight the minimum AIC
    plt.axvline(x=best_degree, color='green', linestyle='--', label=f'Best model (degree {best_degree})')
    plt.scatter([best_degree], [aic_values[best_idx]], s=150, color='green', zorder=10)
    
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('AIC Components by Model Complexity', fontsize=14)
    plt.xticks(degrees)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'statement2_aic_components.png'), dpi=300, bbox_inches='tight')
    
    # Plot 3: New visualization - BIC vs AIC
    # Calculate BIC (Bayesian Information Criterion) for comparison
    bic_values = [n * np.log(mse) + np.log(n) * (degree + 1) for degree, mse in zip(degrees, mse_values)]
    best_bic_idx = np.argmin(bic_values)
    
    # Create a comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, aic_values, 'o-', label='AIC', color='purple', linewidth=2)
    plt.plot(degrees, bic_values, 'o-', label='BIC', color='orange', linewidth=2)
    
    # Highlight minima
    plt.axvline(x=degrees[best_idx], color='purple', linestyle='--', 
                label=f'Best by AIC (degree {degrees[best_idx]})')
    plt.axvline(x=degrees[best_bic_idx], color='orange', linestyle='--', 
                label=f'Best by BIC (degree {degrees[best_bic_idx]})')
    
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Criterion Value', fontsize=12)
    plt.title('AIC vs BIC for Model Selection', fontsize=14)
    plt.xticks(degrees)
    plt.grid(True)
    plt.legend()
    
    # Add LaTeX formula annotation
    plt.annotate(r'$\mathrm{AIC} = n \ln(\mathrm{MSE}) + 2k$', 
                xy=(0.05, 0.92), xycoords='axes fraction', fontsize=12)
    plt.annotate(r'$\mathrm{BIC} = n \ln(\mathrm{MSE}) + k \ln(n)$', 
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
    
    plt.savefig(os.path.join(save_dir, 'statement2_aic_vs_bic.png'), dpi=300, bbox_inches='tight')
    
    # Print AIC vs BIC explanation
    print("\nAIC vs BIC Comparison:")
    print("AIC and BIC are both information criteria used for model selection.")
    print("- AIC: Akaike Information Criterion - Penalty term: 2k")
    print("- BIC: Bayesian Information Criterion - Penalty term: k·ln(n)")
    print("- BIC penalizes model complexity more heavily than AIC")
    print(f"- AIC selects degree {degrees[best_idx]} as optimal")
    print(f"- BIC selects degree {degrees[best_bic_idx]} as optimal")
    if best_bic_idx < best_idx:
        print("- This demonstrates that BIC tends to favor simpler models than AIC")
    else:
        print("- In this case, both criteria selected similar models")
    
    result = {
        'statement': "When comparing models using information criteria, the model with the highest AIC value should be selected.",
        'is_true': False,
        'explanation': "When using information criteria like AIC (Akaike Information Criterion), we should select the model with the LOWEST AIC value, not the highest. AIC balances model fit and complexity by penalizing models with more parameters. Lower AIC values indicate better models with a good trade-off between fit and complexity.",
        'image_path': ['statement2_aic_selection.png', 'statement2_aic_components.png', 'statement2_aic_vs_bic.png']
    }
    
    return result

if __name__ == "__main__":
    result = statement2_aic_selection()
    print(f"\nStatement: {result['statement']}")
    print(f"True or False: {'True' if result['is_true'] else 'False'}")
    print(f"Explanation: {result['explanation']}")
    print(f"Images saved: {', '.join(result['image_path'])}") 