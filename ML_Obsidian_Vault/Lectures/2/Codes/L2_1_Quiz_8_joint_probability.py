import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle

def calculate_marginal_distributions(joint_pmf):
    """
    Calculate the marginal distributions P(X) and P(Y) from a joint PMF.
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    
    Returns:
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    """
    px = {}
    py = {}
    
    # Get unique x and y values
    x_values = sorted(set(k[0] for k in joint_pmf.keys()))
    y_values = sorted(set(k[1] for k in joint_pmf.keys()))
    
    # Calculate P(X=x)
    for x in x_values:
        px[x] = sum(joint_pmf.get((x, y), 0) for y in y_values)
    
    # Calculate P(Y=y)
    for y in y_values:
        py[y] = sum(joint_pmf.get((x, y), 0) for x in x_values)
    
    return px, py

def calculate_conditional_distributions(joint_pmf, px, py):
    """
    Calculate the conditional distributions P(Y|X) and P(X|Y).
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    
    Returns:
    pyx -- Dictionary with (x,y) tuples as keys and P(Y=y|X=x) as values
    pxy -- Dictionary with (x,y) tuples as keys and P(X=x|Y=y) as values
    """
    pyx = {}
    pxy = {}
    
    for (x, y), p in joint_pmf.items():
        # P(Y=y|X=x) = P(X=x,Y=y) / P(X=x)
        pyx[(x, y)] = p / px[x]
        
        # P(X=x|Y=y) = P(X=x,Y=y) / P(Y=y)
        pxy[(x, y)] = p / py[y]
    
    return pyx, pxy

def check_independence(joint_pmf, px, py):
    """
    Check if X and Y are independent.
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    
    Returns:
    is_independent -- Boolean indicating whether X and Y are independent
    max_diff -- Maximum absolute difference between P(X,Y) and P(X)P(Y)
    """
    max_diff = 0
    
    for (x, y), p_xy in joint_pmf.items():
        p_x = px[x]
        p_y = py[y]
        product = p_x * p_y
        diff = abs(p_xy - product)
        max_diff = max(max_diff, diff)
    
    # Allow for small numerical errors
    is_independent = max_diff < 1e-10
    
    return is_independent, max_diff

def calculate_covariance_correlation(joint_pmf, px, py):
    """
    Calculate the covariance and correlation coefficient of X and Y.
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    
    Returns:
    ex -- Expected value of X
    ey -- Expected value of Y
    varx -- Variance of X
    vary -- Variance of Y
    cov -- Covariance between X and Y
    corr -- Correlation coefficient between X and Y
    """
    # Calculate E[X] and E[Y]
    ex = sum(x * p for x, p in px.items())
    ey = sum(y * p for y, p in py.items())
    
    # Calculate E[X²] and E[Y²]
    ex2 = sum(x**2 * p for x, p in px.items())
    ey2 = sum(y**2 * p for y, p in py.items())
    
    # Calculate Var(X) and Var(Y)
    varx = ex2 - ex**2
    vary = ey2 - ey**2
    
    # Calculate E[XY]
    exy = sum(x * y * p for (x, y), p in joint_pmf.items())
    
    # Calculate Cov(X,Y)
    cov = exy - ex * ey
    
    # Calculate correlation coefficient
    corr = cov / (np.sqrt(varx) * np.sqrt(vary))
    
    return ex, ey, varx, vary, cov, corr

def plot_joint_pmf(joint_pmf, save_path=None):
    """
    Create a heatmap visualization of the joint PMF.
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    save_path -- Path to save the figure
    """
    # Convert joint PMF to a matrix form
    x_values = sorted(set(k[0] for k in joint_pmf.keys()))
    y_values = sorted(set(k[1] for k in joint_pmf.keys()))
    
    joint_matrix = np.zeros((len(x_values), len(y_values)))
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            joint_matrix[i, j] = joint_pmf.get((x, y), 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(joint_matrix, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            text = ax.text(j, i, f"{joint_matrix[i, j]:.2f}",
                           ha="center", va="center", color="white" if joint_matrix[i, j] > 0.15 else "black")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(y_values)))
    ax.set_yticks(np.arange(len(x_values)))
    ax.set_xticklabels(y_values)
    ax.set_yticklabels(x_values)
    
    # Add titles and labels
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Joint Probability Mass Function P(X,Y)')
    
    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Probability')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Joint PMF plot saved to {save_path}")
    
    plt.close()

def plot_marginal_distributions(px, py, save_path=None):
    """
    Create visualizations of the marginal distributions.
    
    Parameters:
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    save_path -- Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot P(X)
    x_values = sorted(px.keys())
    ax1.bar(x_values, [px[x] for x in x_values], width=0.4, color='skyblue')
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(x_values)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('P(X=x)')
    ax1.set_title('Marginal Distribution of X')
    
    # Add text annotations for P(X)
    for x in x_values:
        ax1.text(x, px[x] + 0.02, f"{px[x]:.2f}", ha='center')
    
    # Plot P(Y)
    y_values = sorted(py.keys())
    ax2.bar(y_values, [py[y] for y in y_values], width=0.4, color='lightgreen')
    ax2.set_xticks(y_values)
    ax2.set_xticklabels(y_values)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('y')
    ax2.set_ylabel('P(Y=y)')
    ax2.set_title('Marginal Distribution of Y')
    
    # Add text annotations for P(Y)
    for y in y_values:
        ax2.text(y, py[y] + 0.02, f"{py[y]:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Marginal distributions plot saved to {save_path}")
    
    plt.close()

def plot_conditional_distributions(pyx, pxy, save_path=None):
    """
    Create visualizations of the conditional distributions.
    
    Parameters:
    pyx -- Dictionary with (x,y) tuples as keys and P(Y=y|X=x) as values
    pxy -- Dictionary with (x,y) tuples as keys and P(X=x|Y=y) as values
    save_path -- Path to save the figure
    """
    # Get unique x and y values
    x_values = sorted(set(k[0] for k in pyx.keys()))
    y_values = sorted(set(k[1] for k in pyx.keys()))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot P(Y|X=1)
    width = 0.4
    for x in x_values:
        y_vals = [pyx.get((x, y), 0) for y in y_values]
        ax1.bar(
            [y + (x-1.5)*width for y in y_values], 
            y_vals, 
            width=width, 
            label=f'X={x}'
        )
    
    ax1.set_xticks(y_values)
    ax1.set_xlabel('y')
    ax1.set_ylabel('P(Y=y|X=x)')
    ax1.set_title('Conditional Distribution P(Y|X)')
    ax1.legend()
    
    # Plot P(X|Y=2)
    for y in y_values:
        x_vals = [pxy.get((x, y), 0) for x in x_values]
        ax2.bar(
            [x + (y-2)*width for x in x_values], 
            x_vals, 
            width=width, 
            label=f'Y={y}'
        )
    
    ax2.set_xticks(x_values)
    ax2.set_xlabel('x')
    ax2.set_ylabel('P(X=x|Y=y)')
    ax2.set_title('Conditional Distribution P(X|Y)')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Conditional distributions plot saved to {save_path}")
    
    plt.close()

def plot_independence_check(joint_pmf, px, py, save_path=None):
    """
    Create a visualization to check for independence between X and Y.
    
    Parameters:
    joint_pmf -- Dictionary with (x,y) tuples as keys and probabilities as values
    px -- Dictionary with x values as keys and P(X=x) as values
    py -- Dictionary with y values as keys and P(Y=y) as values
    save_path -- Path to save the figure
    """
    # Get unique x and y values
    x_values = sorted(set(k[0] for k in joint_pmf.keys()))
    y_values = sorted(set(k[1] for k in joint_pmf.keys()))
    
    # Calculate the product distribution P(X)P(Y)
    product_pmf = {}
    for x in x_values:
        for y in y_values:
            product_pmf[(x, y)] = px[x] * py[y]
    
    # Create a side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create matrices for both distributions
    joint_matrix = np.zeros((len(x_values), len(y_values)))
    product_matrix = np.zeros((len(x_values), len(y_values)))
    
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            joint_matrix[i, j] = joint_pmf.get((x, y), 0)
            product_matrix[i, j] = product_pmf.get((x, y), 0)
    
    # Plot P(X,Y)
    im1 = axes[0].imshow(joint_matrix, cmap='viridis', aspect='auto')
    # Add text annotations
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            axes[0].text(j, i, f"{joint_matrix[i, j]:.2f}",
                        ha="center", va="center", color="white" if joint_matrix[i, j] > 0.15 else "black")
    
    # Set ticks and labels
    axes[0].set_xticks(np.arange(len(y_values)))
    axes[0].set_yticks(np.arange(len(x_values)))
    axes[0].set_xticklabels(y_values)
    axes[0].set_yticklabels(x_values)
    axes[0].set_title('Joint Distribution P(X,Y)')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    
    # Plot P(X)P(Y)
    im2 = axes[1].imshow(product_matrix, cmap='viridis', aspect='auto')
    # Add text annotations
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            axes[1].text(j, i, f"{product_matrix[i, j]:.2f}",
                        ha="center", va="center", color="white" if product_matrix[i, j] > 0.15 else "black")
    
    # Set ticks and labels
    axes[1].set_xticks(np.arange(len(y_values)))
    axes[1].set_yticks(np.arange(len(x_values)))
    axes[1].set_xticklabels(y_values)
    axes[1].set_yticklabels(x_values)
    axes[1].set_title('Product Distribution P(X)P(Y)')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('X')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Independence check plot saved to {save_path}")
    
    plt.close()

def plot_covariance_explanation(ex, ey, cov, corr, save_path=None):
    """
    Create a visual explanation of covariance and correlation.
    
    Parameters:
    ex -- Expected value of X
    ey -- Expected value of Y
    cov -- Covariance between X and Y
    corr -- Correlation coefficient between X and Y
    save_path -- Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid of points
    x_values = [1, 2]
    y_values = [1, 2, 3]
    
    # Create joint PMF data
    joint_pmf = {
        (1, 1): 0.10,
        (1, 2): 0.05,
        (1, 3): 0.15,
        (2, 1): 0.20,
        (2, 2): 0.30,
        (2, 3): 0.20
    }
    
    # Plot the joint distribution points
    for (x, y), p in joint_pmf.items():
        size = p * 1000  # Scale by probability
        ax.scatter(x, y, s=size, alpha=0.6, edgecolor='black', linewidth=1)
        ax.text(x, y, f"({x},{y}): {p}", ha='center', va='bottom', fontsize=10)
    
    # Add lines for expected values
    ax.axvline(x=ex, color='blue', linestyle='--', label=f'E[X]={ex:.2f}')
    ax.axhline(y=ey, color='red', linestyle='--', label=f'E[Y]={ey:.2f}')
    
    # Plot the expected value point
    ax.scatter(ex, ey, color='purple', s=100, marker='X', label='(E[X],E[Y])')
    
    # Set axis limits and labels
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add covariance and correlation information
    plt.title(f'Joint Distribution with Covariance and Correlation')
    plt.figtext(0.5, 0.01, f"Cov(X,Y) = {cov:.4f}, Correlation = {corr:.4f}", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covariance explanation plot saved to {save_path}")
    
    plt.close()

def main():
    """
    Solve the joint probability problem and create visualizations.
    """
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_8")
    os.makedirs(save_dir, exist_ok=True)
    
    # Problem statement - Joint PMF
    joint_pmf = {
        (1, 1): 0.10,
        (1, 2): 0.05,
        (1, 3): 0.15,
        (2, 1): 0.20,
        (2, 2): 0.30,
        (2, 3): 0.20
    }
    
    print("\n=== Question 8: Joint Probability and Correlation ===")
    print("Joint PMF P(X,Y):")
    for (x, y), p in sorted(joint_pmf.items()):
        print(f"P(X={x}, Y={y}) = {p}")
    
    # Task 1: Find marginal distributions P(X) and P(Y)
    px, py = calculate_marginal_distributions(joint_pmf)
    
    print("\n1. Marginal Distributions:")
    print("\nP(X):")
    for x, p in sorted(px.items()):
        print(f"P(X={x}) = {p}")
    
    print("\nP(Y):")
    for y, p in sorted(py.items()):
        print(f"P(Y={y}) = {p}")
    
    # Task 2: Calculate conditional distributions
    pyx, pxy = calculate_conditional_distributions(joint_pmf, px, py)
    
    print("\n2. Conditional Distributions:")
    print("\nP(Y|X=1):")
    for y in sorted(set(k[1] for k in joint_pmf.keys())):
        print(f"P(Y={y}|X=1) = {pyx.get((1, y), 0)}")
    
    print("\nP(X|Y=2):")
    for x in sorted(set(k[0] for k in joint_pmf.keys())):
        print(f"P(X={x}|Y=2) = {pxy.get((x, 2), 0)}")
    
    # Task 3: Check independence
    is_independent, max_diff = check_independence(joint_pmf, px, py)
    
    print("\n3. Independence Check:")
    print(f"Are X and Y independent? {is_independent}")
    print(f"Maximum difference between P(X,Y) and P(X)P(Y): {max_diff:.6f}")
    
    if not is_independent:
        print("Since this difference is not zero, X and Y are not independent.")
        print("For independent variables, P(X,Y) = P(X)P(Y) for all values of X and Y.")
    
    # Task 4: Calculate covariance and correlation
    ex, ey, varx, vary, cov, corr = calculate_covariance_correlation(joint_pmf, px, py)
    
    print("\n4. Covariance and Correlation:")
    print(f"E[X] = {ex}")
    print(f"E[Y] = {ey}")
    print(f"Var(X) = {varx}")
    print(f"Var(Y) = {vary}")
    print(f"Cov(X,Y) = {cov}")
    print(f"Correlation coefficient ρ(X,Y) = {corr}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_joint_pmf(joint_pmf, save_path=os.path.join(save_dir, "joint_pmf.png"))
    plot_marginal_distributions(px, py, save_path=os.path.join(save_dir, "marginal_distributions.png"))
    plot_conditional_distributions(pyx, pxy, save_path=os.path.join(save_dir, "conditional_distributions.png"))
    plot_independence_check(joint_pmf, px, py, save_path=os.path.join(save_dir, "independence_check.png"))
    plot_covariance_explanation(ex, ey, cov, corr, save_path=os.path.join(save_dir, "covariance_correlation.png"))
    
    print(f"\nAll calculations and visualizations for Question 8 have been completed.")
    print(f"Visualization files have been saved to: {save_dir}")
    
    # Summary of results
    print("\n=== Summary of Results ===")
    print("1. Marginal Distributions:")
    print(f"   P(X=1) = {px[1]}, P(X=2) = {px[2]}")
    print(f"   P(Y=1) = {py[1]}, P(Y=2) = {py[2]}, P(Y=3) = {py[3]}")
    
    print("\n2. Conditional Distributions:")
    print(f"   P(Y=1|X=1) = {pyx[(1,1)]}, P(Y=2|X=1) = {pyx[(1,2)]}, P(Y=3|X=1) = {pyx[(1,3)]}")
    print(f"   P(X=1|Y=2) = {pxy[(1,2)]}, P(X=2|Y=2) = {pxy[(2,2)]}")
    
    print("\n3. Independence: X and Y are not independent")
    
    print("\n4. Covariance and Correlation:")
    print(f"   Cov(X,Y) = {cov}")
    print(f"   Correlation coefficient ρ(X,Y) = {corr}")

if __name__ == "__main__":
    main() 