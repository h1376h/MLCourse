import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

# Use a clean style
plt.style.use('default')

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
images_dir = os.path.join(parent_dir, "Images", "Mahalanobis_Examples")

# Create images directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

def print_step(step_num, title):
    """Print a formatted step header."""
    print(f"\nStep {step_num}: {title}")
    print("-" * 50)

def print_matrix(name, matrix):
    """Print matrix in a readable format with box drawing characters."""
    if isinstance(matrix, (np.ndarray, list)):
        matrix = np.array(matrix)
        if len(matrix.shape) == 1:
            print(f"{name} = [{' '.join(f'{x:8.4f}' for x in matrix)}]")
        else:
            print(f"{name} =")
            print("┌" + " "*max(len(str(x)) for x in matrix.flatten())*2 + "┐")
            for row in matrix:
                print(f"│ {' '.join(f'{x:8.4f}' for x in row)} │")
            print("└" + " "*max(len(str(x)) for x in matrix.flatten())*2 + "┘")
    else:
        print(f"{name} = {matrix:8.4f}")
    print()

def mahalanobis_distance(x, mu, sigma, show_steps=False):
    """Calculate the Mahalanobis distance with detailed step-by-step output."""
    if show_steps:
        print_step(1, "Calculate difference vector (x - μ)")
        print_matrix("x", x)
        print_matrix("μ", mu)
        diff = x - mu
        print_matrix("x - μ", diff)
        
        print_step(2, "Calculate inverse of covariance matrix (Σ⁻¹)")
        print_matrix("Σ", sigma)
        inv_sigma = np.linalg.inv(sigma)
        print_matrix("Σ⁻¹", inv_sigma)
        
        print_step(3, "Calculate quadratic form (x - μ)ᵀ Σ⁻¹ (x - μ)")
        temp = inv_sigma @ diff
        print("First multiply Σ⁻¹(x - μ):")
        print_matrix("Σ⁻¹(x - μ)", temp)
        
        print("Then multiply by (x - μ)ᵀ:")
        d_squared = diff.T @ temp
        print(f"(x - μ)ᵀ Σ⁻¹ (x - μ) = {d_squared:.4f}")
        
        print_step(4, "Take square root to get Mahalanobis distance")
        d = np.sqrt(d_squared)
        print(f"d = √{d_squared:.4f} = {d:.4f}")
        
        return d
    else:
        diff = x - mu
        inv_sigma = np.linalg.inv(sigma)
        return np.sqrt(diff.T @ inv_sigma @ diff)

def plot_contours_and_points(means, sigma, points=None, title="", filename="", show_decision_boundary=False):
    """Enhanced plot with improved aesthetics."""
    plt.figure(figsize=(12, 10))

    # Set clean white background
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # Create a grid of points with higher resolution
    x = np.linspace(-2, 8, 300)
    y = np.linspace(0, 12, 300)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distances
    Z = np.zeros((len(y), len(x), len(means)))
    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[j,i], Y[j,i]])
            for k, mu in enumerate(means):
                Z[j,i,k] = mahalanobis_distance(point, mu, sigma)
    
    # Plot decision regions with subtle colors
    if show_decision_boundary and len(means) > 1:
        class_regions = np.argmin(Z, axis=2)
        colors = ['#E6F3FF', '#FFE6E6', '#E6FFE6']  # Very light colors
        plt.contourf(X, Y, class_regions, alpha=0.1, colors=colors)
    
    # Plot contours for each class
    colors = ['#0066CC', '#CC0000', '#006600']  # Strong colors for contours
    for k, (mu, color) in enumerate(zip(means, colors)):
        levels = np.array([1, 2, 3])
        CS = plt.contour(X, Y, Z[:,:,k], levels=levels, 
                        colors=color, alpha=0.7, linewidths=1.5,
                        linestyles=['--', '-', ':'])
        plt.clabel(CS, inline=True, fontsize=9, fmt='%.1f')
        
        # Add confidence ellipses
        eigenvals, eigenvecs = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(eigenvecs[1,0], eigenvecs[0,0]))
        for nstd in [1, 2, 3]:
            ell = Ellipse(xy=mu, width=2*nstd*np.sqrt(eigenvals[0]), 
                         height=2*nstd*np.sqrt(eigenvals[1]), angle=angle,
                         color=color, alpha=0.1)
            plt.gca().add_patch(ell)
    
    # Plot means with distinct markers
    for i, (mu, color) in enumerate(zip(means, colors)):
        plt.plot(mu[0], mu[1], marker='*', color=color, markersize=15, 
                markeredgewidth=2, markeredgecolor='black',
                label=f'Class {chr(65+i)} Mean')
    
    # Plot test points
    if points is not None:
        for i, point in enumerate(points):
            plt.plot(point[0], point[1], 'ko', markersize=10, 
                    markeredgewidth=2, markeredgecolor='black',
                    label=f'Test Point {i+1}')
    
    # Customize grid and axes
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_axisbelow(True)
    
    # Customize labels and title
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title, pad=20, size=14)
    plt.xlabel('X₁', size=12)
    plt.ylabel('X₂', size=12)
    
    # Save with high quality
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, filename), 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def example1():
    """Example 1: Binary Classification with detailed steps"""
    print("\n" + "="*80)
    print("Example 1: Binary Classification".center(80))
    print("="*80)
    
    print("\nProblem Statement:")
    print("-"*50)
    print("Given two classes with multivariate normal distributions having the same")
    print("covariance matrix but different means, classify a new observation.")
    
    # Parameters
    mu1 = np.array([2, 4])
    mu2 = np.array([5, 6])
    sigma = np.array([[5, 1], [1, 3]])
    x = np.array([3, 5])
    
    print("\nGiven Parameters:")
    print("-"*50)
    print_matrix("Class 1 mean (μ₁)", mu1)
    print_matrix("Class 2 mean (μ₂)", mu2)
    print_matrix("Shared covariance matrix (Σ)", sigma)
    print_matrix("Test point (x)", x)
    
    print("\nSolution Method:")
    print("-"*50)
    print("1. Calculate Mahalanobis distances to both class means")
    print("2. Assign the point to the class with minimum distance")
    print("\nFormula: d²(x,μ) = (x-μ)ᵀ Σ⁻¹ (x-μ)")
    
    print("\nDetailed Calculations:")
    print("="*50)
    
    print("\nPart 1: Distance to Class 1")
    d1 = mahalanobis_distance(x, mu1, sigma, show_steps=True)
    
    print("\nPart 2: Distance to Class 2")
    d2 = mahalanobis_distance(x, mu2, sigma, show_steps=True)
    
    print("\nFinal Classification Decision:")
    print("-"*50)
    print(f"Distance to Class 1 (d₁) = {d1:.4f}")
    print(f"Distance to Class 2 (d₂) = {d2:.4f}")
    print(f"\nSince d₁ {'<' if d1 < d2 else '>'} d₂, the point is classified as Class {1 if d1 < d2 else 2}")
    
    # Visualize
    plot_contours_and_points(
        [mu1, mu2], 
        sigma, 
        [x],
        "Binary Classification Example",
        "example1_binary_classification.png",
        show_decision_boundary=True
    )

def example2():
    """Example 2: Outlier Detection with detailed steps"""
    print("\n" + "="*80)
    print("Example 2: Outlier Detection".center(80))
    print("="*80)
    
    print("\nProblem Statement:")
    print("-"*50)
    print("Determine if a measurement is an outlier using Mahalanobis distance")
    print("and the chi-square distribution at 5% significance level.")
    
    # Parameters
    mu = np.array([120, 5.5])
    sigma = np.array([[25, 2.5], [2.5, 0.64]])
    x = np.array([130, 6.0])
    
    print("\nGiven Parameters:")
    print("-"*50)
    print_matrix("Population mean (μ)", mu)
    print_matrix("Covariance matrix (Σ)", sigma)
    print_matrix("Test point (x)", x)
    
    print("\nSolution Method:")
    print("-"*50)
    print("1. Calculate the Mahalanobis distance")
    print("2. Find critical value from chi-square distribution")
    print("3. Compare distance with critical value")
    
    print("\nTheoretical Background:")
    print("-"*50)
    print("For multivariate normal data:")
    print("1. d²(x,μ) follows χ²(p) distribution")
    print("2. p = dimension of data (p=2 in this case)")
    print("3. Reject H₀ if d²(x,μ) > χ²(1-α,p)")
    
    print("\nDetailed Calculations:")
    print("="*50)
    
    print("\nPart 1: Mahalanobis Distance")
    d = mahalanobis_distance(x, mu, sigma, show_steps=True)
    
    print("\nPart 2: Critical Value Determination")
    print("-"*50)
    alpha = 0.05
    df = 2
    chi2_val = chi2.ppf(1-alpha, df)
    threshold = np.sqrt(chi2_val)
    
    print(f"1. Significance level (α) = {alpha}")
    print(f"2. Degrees of freedom (p) = {df}")
    print(f"3. χ²(1-α,p) = χ²({1-alpha}, {df}) = {chi2_val:.4f}")
    print(f"4. Critical value = √{chi2_val:.4f} = {threshold:.4f}")
    
    print("\nFinal Decision:")
    print("-"*50)
    print(f"Mahalanobis distance (d) = {d:.4f}")
    print(f"Critical value = {threshold:.4f}")
    print(f"Since d {'>' if d > threshold else '<'} critical value,")
    print(f"the point is {'an outlier' if d > threshold else 'not an outlier'}")
    
    # Visualize
    plot_contours_and_points(
        [mu], 
        sigma, 
        [x],
        "Outlier Detection Example",
        "example2_outlier_detection.png"
    )

def example3():
    """Example 3: Three-Class Classification with detailed steps"""
    print("\n" + "="*80)
    print("Example 3: Three-Class Classification with Prior Probabilities".center(80))
    print("="*80)
    
    print("\nProblem Statement:")
    print("-"*50)
    print("Classify a point among three classes considering prior probabilities.")
    
    # Parameters
    mu_a = np.array([3, 8])
    mu_b = np.array([5, 6])
    mu_c = np.array([4, 10])
    sigma = np.array([[1.2, 0.4], [0.4, 2.0]])
    x = np.array([4, 9])
    priors = [0.5, 0.3, 0.2]
    
    print("\nGiven Parameters:")
    print("-"*50)
    print_matrix("Class A mean (μₐ)", mu_a)
    print_matrix("Class B mean (μᵦ)", mu_b)
    print_matrix("Class C mean (μc)", mu_c)
    print_matrix("Shared covariance matrix (Σ)", sigma)
    print_matrix("Test point (x)", x)
    print("Prior probabilities:")
    print(f"P(A) = {priors[0]:.2f}")
    print(f"P(B) = {priors[1]:.2f}")
    print(f"P(C) = {priors[2]:.2f}\n")
    
    print("\nTheoretical Background:")
    print("-"*50)
    print("1. Without priors: Classify using minimum Mahalanobis distance")
    print("2. With priors: Use discriminant function")
    print("   g_i(x) = ln(P(C_i)) - (1/2)d_i²(x)")
    print("   where d_i(x) is Mahalanobis distance to class i")
    
    print("\nDetailed Calculations:")
    print("="*50)
    
    print("\nPart 1: Mahalanobis Distances")
    print("\nTo Class A:")
    d_a = mahalanobis_distance(x, mu_a, sigma, show_steps=True)
    print("\nTo Class B:")
    d_b = mahalanobis_distance(x, mu_b, sigma, show_steps=True)
    print("\nTo Class C:")
    d_c = mahalanobis_distance(x, mu_c, sigma, show_steps=True)
    
    print("\nPart 2: Classification without Priors")
    print("-"*50)
    distances = [d_a, d_b, d_c]
    min_dist = min(distances)
    class_label = ['A', 'B', 'C'][distances.index(min_dist)]
    print(f"Distance to Class A (dₐ) = {d_a:.4f}")
    print(f"Distance to Class B (dᵦ) = {d_b:.4f}")
    print(f"Distance to Class C (dc) = {d_c:.4f}")
    print(f"\nUsing minimum distance, classify as Class {class_label}")
    
    print("\nPart 3: Classification with Priors")
    print("-"*50)
    print("Calculate discriminant scores:")
    print("g_i(x) = ln(P(C_i)) - (1/2)d_i²(x)")
    
    scores = [
        np.log(priors[0]) - 0.5 * d_a**2,
        np.log(priors[1]) - 0.5 * d_b**2,
        np.log(priors[2]) - 0.5 * d_c**2
    ]
    
    print(f"\nClass A:")
    print(f"gₐ(x) = ln({priors[0]:.2f}) - (1/2)({d_a:.4f})²")
    print(f"     = {np.log(priors[0]):.4f} - {0.5 * d_a**2:.4f}")
    print(f"     = {scores[0]:.4f}")
    
    print(f"\nClass B:")
    print(f"gᵦ(x) = ln({priors[1]:.2f}) - (1/2)({d_b:.4f})²")
    print(f"     = {np.log(priors[1]):.4f} - {0.5 * d_b**2:.4f}")
    print(f"     = {scores[1]:.4f}")
    
    print(f"\nClass C:")
    print(f"gc(x) = ln({priors[2]:.2f}) - (1/2)({d_c:.4f})²")
    print(f"     = {np.log(priors[2]):.4f} - {0.5 * d_c**2:.4f}")
    print(f"     = {scores[2]:.4f}")
    
    class_label = ['A', 'B', 'C'][np.argmax(scores)]
    print(f"\nFinal Classification:")
    print("-"*50)
    print(f"Since g_{class_label.lower()}(x) is maximum,")
    print(f"classify as Class {class_label} when considering prior probabilities")
    
    # Visualize
    plot_contours_and_points(
        [mu_a, mu_b, mu_c], 
        sigma, 
        [x],
        "Three-Class Classification Example",
        "example3_three_class_classification.png",
        show_decision_boundary=True
    )

def main():
    print("\n" + "="*80)
    print("Mahalanobis Distance and Classification Examples".center(80))
    print("="*80 + "\n")
    
    example1()
    example2()
    example3()
    
    print("\n" + "="*80)
    print("All examples completed. Check the Images/Mahalanobis_Examples directory for visualizations.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 