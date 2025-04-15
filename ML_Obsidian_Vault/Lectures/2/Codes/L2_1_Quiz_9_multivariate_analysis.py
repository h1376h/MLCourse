import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

def calculate_correlation(cov_matrix):
    """
    Calculate the correlation coefficient from the covariance matrix.
    
    Parameters:
    cov_matrix -- 2x2 covariance matrix
    
    Returns:
    correlation -- Correlation coefficient
    """
    var_x = cov_matrix[0, 0]
    var_y = cov_matrix[1, 1]
    cov_xy = cov_matrix[0, 1]
    
    correlation = cov_xy / np.sqrt(var_x * var_y)
    
    return correlation

def calculate_linear_transformation(mean_vector, cov_matrix, a, b, c):
    """
    Calculate the expected value and variance of a linear transformation.
    
    Y = aX₁ + bX₂ + c
    
    Parameters:
    mean_vector -- Mean vector [μ₁, μ₂] of X = [X₁, X₂]
    cov_matrix -- Covariance matrix of X
    a, b, c -- Coefficients of the linear transformation
    
    Returns:
    mean_y -- E[Y]
    var_y -- Var(Y)
    """
    # For Y = aX₁ + bX₂ + c:
    # E[Y] = aE[X₁] + bE[X₂] + c
    mean_y = a * mean_vector[0] + b * mean_vector[1] + c
    
    # For Y = aX₁ + bX₂ + c:
    # Var(Y) = a²Var(X₁) + b²Var(X₂) + 2ab·Cov(X₁,X₂)
    var_y = (a**2 * cov_matrix[0, 0] + 
             b**2 * cov_matrix[1, 1] + 
             2 * a * b * cov_matrix[0, 1])
    
    return mean_y, var_y

def check_independence(cov_matrix):
    """
    Check if X₁ and X₂ are independent.
    
    Parameters:
    cov_matrix -- Covariance matrix of X
    
    Returns:
    is_independent -- Boolean indicating whether X₁ and X₂ are independent
    """
    # X₁ and X₂ are independent if and only if their covariance is 0
    cov_xy = cov_matrix[0, 1]
    is_independent = abs(cov_xy) < 1e-10
    
    return is_independent

def plot_bivariate_normal(mean_vector, cov_matrix, save_path=None):
    """
    Visualize the bivariate normal distribution.
    
    Parameters:
    mean_vector -- Mean vector [μ₁, μ₂] of X = [X₁, X₂]
    cov_matrix -- Covariance matrix of X
    save_path -- Path to save the figure
    """
    # Generate data from the multivariate normal distribution
    np.random.seed(42)  # For reproducibility
    data = np.random.multivariate_normal(mean_vector, cov_matrix, 1000)
    
    # Extract X₁ and X₂ values
    x1 = data[:, 0]
    x2 = data[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the data points
    ax.scatter(x1, x2, s=10, alpha=0.5, label='Simulated Data')
    
    # Plot the mean vector
    ax.scatter(mean_vector[0], mean_vector[1], color='red', s=100, marker='X', 
              label=f'Mean: ({mean_vector[0]}, {mean_vector[1]})')
    
    # Add confidence ellipses (1, 2, and 3 standard deviations)
    for n_std in [1, 2, 3]:
        ellipse = Ellipse(xy=mean_vector, 
                          width=2 * n_std * np.sqrt(cov_matrix[0, 0]), 
                          height=2 * n_std * np.sqrt(cov_matrix[1, 1]),
                          angle=np.degrees(0.5 * np.arctan2(2 * cov_matrix[0, 1],
                                                          cov_matrix[0, 0] - cov_matrix[1, 1])))
        ellipse.set_alpha(0.3)
        ellipse.set_facecolor('green')
        ax.add_artist(ellipse)
        ax.text(mean_vector[0] + n_std * np.sqrt(cov_matrix[0, 0]) * 0.7,
               mean_vector[1] + n_std * np.sqrt(cov_matrix[1, 1]) * 0.7,
               f"{n_std}σ", fontsize=12)
    
    # Show variances
    ax.axvline(x=mean_vector[0], color='blue', linestyle='--',
              label=f'Var(X₁) = {cov_matrix[0, 0]}')
    ax.axhline(y=mean_vector[1], color='green', linestyle='--',
              label=f'Var(X₂) = {cov_matrix[1, 1]}')
    
    # Add title and labels
    correlation = calculate_correlation(cov_matrix)
    is_independent = check_independence(cov_matrix)
    
    ax.set_title(f'Bivariate Normal Distribution\nCorrelation: {correlation:.4f}, Independent: {is_independent}')
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.legend(loc='upper left')
    
    # Set reasonable axis limits
    ax.set_xlim(mean_vector[0] - 3 * np.sqrt(cov_matrix[0, 0]),
               mean_vector[0] + 3 * np.sqrt(cov_matrix[0, 0]))
    ax.set_ylim(mean_vector[1] - 3 * np.sqrt(cov_matrix[1, 1]),
               mean_vector[1] + 3 * np.sqrt(cov_matrix[1, 1]))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bivariate normal plot saved to {save_path}")
    
    plt.close()

def plot_linear_transformation(mean_vector, cov_matrix, a, b, c, save_path=None):
    """
    Visualize the linear transformation Y = aX₁ + bX₂ + c.
    
    Parameters:
    mean_vector -- Mean vector [μ₁, μ₂] of X = [X₁, X₂]
    cov_matrix -- Covariance matrix of X
    a, b, c -- Coefficients of the linear transformation
    save_path -- Path to save the figure
    """
    # Calculate E[Y] and Var(Y)
    mean_y, var_y = calculate_linear_transformation(mean_vector, cov_matrix, a, b, c)
    
    # Generate data from the multivariate normal distribution
    np.random.seed(42)  # For reproducibility
    data = np.random.multivariate_normal(mean_vector, cov_matrix, 1000)
    
    # Calculate the transformed values Y = aX₁ + bX₂ + c
    y = a * data[:, 0] + b * data[:, 1] + c
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot the original data points
    ax1.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5)
    
    # Plot the direction of the linear transformation as a vector
    vector_length = np.sqrt(a**2 + b**2)
    normalized_a = a / vector_length * 3
    normalized_b = b / vector_length * 3
    
    ax1.arrow(mean_vector[0], mean_vector[1], normalized_a, normalized_b, 
             head_width=0.2, head_length=0.3, fc='red', ec='red',
             label=f'Direction: [{a}, {b}]')
    
    # Plot the mean vector
    ax1.scatter(mean_vector[0], mean_vector[1], color='red', s=100, marker='X', 
               label=f'Mean: ({mean_vector[0]}, {mean_vector[1]})')
    
    # Add title and labels
    ax1.set_title(f'Original Bivariate Distribution\nY = {a}X₁ + {b}X₂ + {c}')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.legend(loc='upper left')
    
    # Set reasonable axis limits
    ax1.set_xlim(mean_vector[0] - 3 * np.sqrt(cov_matrix[0, 0]),
                mean_vector[0] + 3 * np.sqrt(cov_matrix[0, 0]))
    ax1.set_ylim(mean_vector[1] - 3 * np.sqrt(cov_matrix[1, 1]),
                mean_vector[1] + 3 * np.sqrt(cov_matrix[1, 1]))
    
    # Plot the histogram of the transformed values
    ax2.hist(y, bins=30, density=True, alpha=0.6, color='skyblue')
    
    # Plot the normal PDF with mean E[Y] and variance Var(Y)
    x_range = np.linspace(mean_y - 4 * np.sqrt(var_y), mean_y + 4 * np.sqrt(var_y), 1000)
    pdf = 1 / np.sqrt(2 * np.pi * var_y) * np.exp(-0.5 * ((x_range - mean_y) / np.sqrt(var_y))**2)
    ax2.plot(x_range, pdf, 'r-', lw=2, label=f'Normal PDF\nμ = {mean_y:.2f}, σ² = {var_y:.2f}')
    
    # Add a vertical line for the mean
    ax2.axvline(x=mean_y, color='green', linestyle='--', 
                label=f'E[Y] = {mean_y:.2f}')
    
    # Add shaded area for +/- 1 standard deviation
    ax2.axvspan(mean_y - np.sqrt(var_y), mean_y + np.sqrt(var_y), 
                alpha=0.2, color='yellow', label=f'±1 Std Dev = {np.sqrt(var_y):.2f}')
    
    # Add title and labels
    ax2.set_title(f'Transformed Distribution: Y = {a}X₁ + {b}X₂ + {c}')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Probability Density')
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Linear transformation plot saved to {save_path}")
    
    plt.close()

def main():
    """
    Solve the multivariate analysis problem and create visualizations.
    """
    # Create directory for saving images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images")
    save_dir = os.path.join(images_dir, "L2_1_Quiz_9")
    os.makedirs(save_dir, exist_ok=True)
    
    # Problem parameters
    mean_vector = np.array([2, 3])
    cov_matrix = np.array([
        [4, 2],
        [2, 5]
    ])
    
    # Linear transformation parameters for Y = 3X₁ + 2X₂ - 1
    a, b, c = 3, 2, -1
    
    print("\n=== Question 9: Multivariate Analysis ===")
    
    # Task 1: Variance of X₁ and X₂
    var_x1 = cov_matrix[0, 0]
    var_x2 = cov_matrix[1, 1]
    
    print("\n1. Variances:")
    print(f"   Var(X₁) = {var_x1}")
    print(f"   Var(X₂) = {var_x2}")
    
    # Task 2: Correlation coefficient
    correlation = calculate_correlation(cov_matrix)
    
    print("\n2. Correlation Coefficient:")
    print(f"   ρ(X₁, X₂) = {correlation}")
    
    # Task 3: Linear transformation Y = 3X₁ + 2X₂ - 1
    mean_y, var_y = calculate_linear_transformation(mean_vector, cov_matrix, a, b, c)
    
    print("\n3. Linear Transformation Y = 3X₁ + 2X₂ - 1:")
    print(f"   E[Y] = 3·E[X₁] + 2·E[X₂] - 1")
    print(f"        = 3·{mean_vector[0]} + 2·{mean_vector[1]} - 1")
    print(f"        = {3*mean_vector[0]} + {2*mean_vector[1]} - 1")
    print(f"        = {mean_y}")
    
    print(f"\n   Var(Y) = 3²·Var(X₁) + 2²·Var(X₂) + 2·3·2·Cov(X₁,X₂)")
    print(f"          = 9·{var_x1} + 4·{var_x2} + 12·{cov_matrix[0, 1]}")
    print(f"          = {9*var_x1} + {4*var_x2} + {12*cov_matrix[0, 1]}")
    print(f"          = {var_y}")
    
    # Task 4: Independence check
    is_independent = check_independence(cov_matrix)
    
    print("\n4. Independence Check:")
    print(f"   Are X₁ and X₂ independent? {is_independent}")
    
    if not is_independent:
        print(f"   X₁ and X₂ are not independent because their covariance ({cov_matrix[0, 1]}) is not zero.")
        print(f"   For independent random variables, the covariance would be 0.")
        print(f"   Additionally, the correlation coefficient ({correlation}) is not 0, which also indicates non-independence.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_bivariate_normal(mean_vector, cov_matrix, 
                         save_path=os.path.join(save_dir, "bivariate_normal.png"))
    
    plot_linear_transformation(mean_vector, cov_matrix, a, b, c, 
                              save_path=os.path.join(save_dir, "linear_transformation.png"))
    
    print(f"\nAll calculations and visualizations for Question 9 have been completed.")
    print(f"Visualization files have been saved to: {save_dir}")
    
    # Summary of results
    print("\n=== Summary of Results ===")
    print(f"1. Var(X₁) = {var_x1}, Var(X₂) = {var_x2}")
    print(f"2. Correlation coefficient ρ(X₁, X₂) = {correlation}")
    print(f"3. For Y = 3X₁ + 2X₂ - 1: E[Y] = {mean_y}, Var(Y) = {var_y}")
    print(f"4. X₁ and X₂ are {'independent' if is_independent else 'not independent'}")

if __name__ == "__main__":
    main() 