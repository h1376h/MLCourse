import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Arrow
from matplotlib.path import Path
import matplotlib.patches as patches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_2_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Define the data from Question 15
ages = np.array([43, 21, 25, 42, 57, 59])
glucose_levels = np.array([99, 65, 79, 75, 87, 81])

# Quick solution calculations
def quick_solution():
    """Calculate regression parameters using simplified formulas for quick solution."""
    n = len(ages)
    
    # Calculate means
    x_mean = np.mean(ages)
    y_mean = np.mean(glucose_levels)
    
    # Calculate sums
    sum_x = np.sum(ages)
    sum_y = np.sum(glucose_levels)
    sum_xy = np.sum(ages * glucose_levels)
    sum_x2 = np.sum(ages**2)
    sum_y2 = np.sum(glucose_levels**2)
    
    # Calculate slope using simplified formula
    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    
    # Calculate intercept using means
    w0 = y_mean - w1 * x_mean
    
    # Predicted values
    pred_age = 55
    pred_glucose = w0 + w1 * pred_age
    
    # Calculate correlation coefficient
    r_numerator = n * sum_xy - sum_x * sum_y
    r_denominator = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
    r = r_numerator / r_denominator
    
    # Calculate R-squared as r squared
    r_squared = r**2
    
    return {
        'w0': w0,
        'w1': w1,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'sum_x': sum_x,
        'sum_y': sum_y,
        'sum_xy': sum_xy,
        'sum_x2': sum_x2,
        'n': n,
        'r': r,
        'r_squared': r_squared,
        'pred_age': pred_age,
        'pred_glucose': pred_glucose
    }

# Run calculations
results = quick_solution()

# Print summary of results
print("Quick Solution Results:")
print(f"Mean of ages (x̄): {results['x_mean']:.2f}")
print(f"Mean of glucose levels (ȳ): {results['y_mean']:.2f}")
print(f"Sum of x: {results['sum_x']}")
print(f"Sum of y: {results['sum_y']}")
print(f"Sum of xy: {results['sum_xy']}")
print(f"Sum of x²: {results['sum_x2']}")
print(f"Slope (w₁): {results['w1']:.4f}")
print(f"Intercept (w₀): {results['w0']:.4f}")
print(f"Correlation coefficient (r): {results['r']:.4f}")
print(f"Coefficient of determination (R²): {results['r_squared']:.4f}")
print(f"Prediction for age {results['pred_age']}: {results['pred_glucose']:.2f}")

# Print workflow steps instead of creating a diagram
def print_workflow_steps():
    """Print the step-by-step workflow for the quick solution approach."""
    steps = [
        "Step 1: Calculate means first\nx̄ = 41.17, ȳ = 81.0",
        "Step 2: Calculate sums\nΣx = 247, Σy = 486\nΣxy = 20,485, Σx² = 11,409",
        "Step 3: Calculate slope (w₁)\nw₁ = (6·20,485 - 247·486) / (6·11,409 - 247²)\nw₁ = 0.3852",
        "Step 4: Calculate intercept (w₀)\nw₀ = ȳ - w₁·x̄ = 81 - 0.3852·41.17\nw₀ = 65.14",
        "Step 5: Make predictions\nFor age = 55:\nGlucose = 65.14 + 0.3852·55 = 86.33",
        "Step 6: Calculate R²\nR² = r² = 0.2807 (28.07%)"
    ]
    
    print("\nQuick Solution Workflow:")
    for step in steps:
        print(f"\n{step}")

# Create formula comparison visualization
def create_formula_comparison():
    """Create a comparison between traditional and simplified formulas."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    
    # Left side: Traditional approach
    traditional_steps = [
        r"1. Calculate means: $\bar{x}, \bar{y}$",
        r"2. Calculate deviations: $(x_i - \bar{x}), (y_i - \bar{y})$",
        r"3. Calculate products: $(x_i - \bar{x})(y_i - \bar{y})$",
        r"4. Calculate squared deviations: $(x_i - \bar{x})^2$",
        r"5. Sum up all products and squares",
        r"6. Calculate slope: $\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$",
        r"7. Calculate intercept: $\beta_0 = \bar{y} - \beta_1 \bar{x}$"
    ]
    
    # Right side: Simplified approach
    simplified_steps = [
        r"1. Calculate sums directly: $\sum x, \sum y, \sum xy, \sum x^2$",
        r"2. Calculate means: $\bar{x}, \bar{y}$",
        r"3. Use simplified slope formula: $w_1 = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}$",
        r"4. Use means for intercept: $w_0 = \bar{y} - w_1\bar{x}$"
    ]
    
    # Plot traditional approach
    ax[0].set_title('Traditional Approach', fontsize=16)
    for i, step in enumerate(traditional_steps):
        ax[0].text(0.05, 0.9 - i*0.12, step, transform=ax[0].transAxes, fontsize=12,
                  bbox={'facecolor':'lightcoral', 'alpha':0.2, 'pad':10})
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].axis('off')
    
    # Plot simplified approach
    ax[1].set_title('Simplified Approach', fontsize=16)
    for i, step in enumerate(simplified_steps):
        ax[1].text(0.05, 0.9 - i*0.18, step, transform=ax[1].transAxes, fontsize=12,
                  bbox={'facecolor':'lightgreen', 'alpha':0.2, 'pad':10})
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].axis('off')
    
    # Add comparison notes
    plt.figtext(0.5, 0.05, 'Simplified approach reduces computation steps from 7 to 4,\nwhile producing identical results: w₀ = 65.14, w₁ = 0.3852', 
                ha='center', fontsize=14, bbox={'facecolor':'#f0f8ff', 'alpha':0.5, 'pad':10})
    
    # Add title
    plt.suptitle('Formula Comparison: Traditional vs. Simplified Approach', fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "formula_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Print the workflow steps instead of generating the image
print_workflow_steps()

# Generate the formula comparison visualization only
create_formula_comparison()

print(f"\nVisualizations saved to: {save_dir}")
print("1. formula_comparison.png - Comparison between traditional and simplified approaches")

# Remove quick_workflow.png if it exists
workflow_image_path = os.path.join(save_dir, "quick_workflow.png")
if os.path.exists(workflow_image_path):
    try:
        os.remove(workflow_image_path)
        print("2. quick_workflow.png has been removed as requested")
    except Exception as e:
        print(f"Failed to remove quick_workflow.png: {e}") 