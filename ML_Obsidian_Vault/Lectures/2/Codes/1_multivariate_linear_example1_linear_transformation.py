import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import matplotlib.patches as patches
import os
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def example1_linear_transformation():
    """
    Linear Transformation Example 1
    
    Problem Statement:
    Let X = [X₁, X₂, X₃]ᵀ follow a multivariate normal distribution with 
    mean vector μ = [1, 2, 3]ᵀ and covariance matrix 
    Σ = [[4, 1, 0], [1, 9, 2], [0, 2, 16]].
    
    Define Y = AX + b where A = [[2, 1, 0], [0, 3, 1]] and b = [5, -2]ᵀ.
    
    a) Find the distribution of Y.
    b) Calculate Cov(Y₁, Y₂).
    c) Are Y₁ and Y₂ independent? Why or why not?
    """
    print("\n" + "="*80)
    print("Example 1: Linear Transformations of Multivariate Normal Distributions")
    print("="*80)
    
    # Original parameters
    mu_X = np.array([1, 2, 3])
    Sigma_X = np.array([
        [4, 1, 0],
        [1, 9, 2],
        [0, 2, 16]
    ])
    
    # Transformation parameters
    A = np.array([
        [2, 1, 0],
        [0, 3, 1]
    ])
    b = np.array([5, -2])
    
    print("\nGiven:")
    print(f"Mean vector μ_X = {mu_X}")
    print(f"Covariance matrix Σ_X = \n{Sigma_X}")
    print(f"Transformation matrix A = \n{A}")
    print(f"Shift vector b = {b}")
    print(f"We need to find the distribution of Y = AX + b")
    
    # (a) Find the distribution of Y
    print("\n" + "-"*60)
    print("(a) Finding the distribution of Y:")
    print("-"*60)
    
    # Calculate mean of Y
    print("\nSTEP 1: Calculate the mean vector of Y using μ_Y = A·μ_X + b")
    
    # Detailed calculation of mean vector
    print("\nDetailed calculation:")
    print("For a linear transformation Y = AX + b, the mean vector is given by μ_Y = A·μ_X + b")
    
    # First row calculation - more detailed
    print("\nCalculating the first element of μ_Y (row 1):")
    row1_step1 = f"μ_Y[1] = {A[0,0]}×{mu_X[0]} + {A[0,1]}×{mu_X[1]} + {A[0,2]}×{mu_X[2]} + {b[0]}"
    row1_step2 = f"μ_Y[1] = {A[0,0]*mu_X[0]} + {A[0,1]*mu_X[1]} + {A[0,2]*mu_X[2]} + {b[0]}"
    row1_step3 = f"μ_Y[1] = {A[0,0]*mu_X[0] + A[0,1]*mu_X[1] + A[0,2]*mu_X[2]} + {b[0]}"
    row1_step4 = f"μ_Y[1] = {A[0,0]*mu_X[0] + A[0,1]*mu_X[1] + A[0,2]*mu_X[2] + b[0]}"
    print(row1_step1)
    print(row1_step2)
    print(row1_step3)
    print(row1_step4)
    
    # Second row calculation - more detailed
    print("\nCalculating the second element of μ_Y (row 2):")
    row2_step1 = f"μ_Y[2] = {A[1,0]}×{mu_X[0]} + {A[1,1]}×{mu_X[1]} + {A[1,2]}×{mu_X[2]} + {b[1]}"
    row2_step2 = f"μ_Y[2] = {A[1,0]*mu_X[0]} + {A[1,1]*mu_X[1]} + {A[1,2]*mu_X[2]} + {b[1]}"
    row2_step3 = f"μ_Y[2] = {A[1,0]*mu_X[0] + A[1,1]*mu_X[1] + A[1,2]*mu_X[2]} + {b[1]}"
    row2_step4 = f"μ_Y[2] = {A[1,0]*mu_X[0] + A[1,1]*mu_X[1] + A[1,2]*mu_X[2] + b[1]}"
    print(row2_step1)
    print(row2_step2)
    print(row2_step3)
    print(row2_step4)
    
    # Calculate the full mean vector
    mu_Y = np.dot(A, mu_X) + b
    print(f"\nFinal result: μ_Y = A·μ_X + b = {mu_Y}")
    
    # Calculate covariance of Y
    print("\n" + "-"*60)
    print("STEP 2: Calculate the covariance matrix of Y using Σ_Y = A·Σ_X·A^T")
    print("-"*60)
    print("\nFor a linear transformation Y = AX + b, the covariance matrix is given by Σ_Y = A·Σ_X·A^T")
    print("This formula does not involve the vector b because shifting does not affect covariance.")
    
    # Detailed calculation of A·Σ_X
    print("\nSTEP 2.1: Calculate A·Σ_X:")
    
    # First row of A·Σ_X
    print("\nFirst row of A·Σ_X calculation:")
    ASigma_row1_step1 = f"[{A[0,0]}, {A[0,1]}, {A[0,2]}] · {Sigma_X.tolist()}"
    ASigma_row1_step2 = f"[{A[0,0]}×{Sigma_X[0,0]} + {A[0,1]}×{Sigma_X[1,0]} + {A[0,2]}×{Sigma_X[2,0]}, {A[0,0]}×{Sigma_X[0,1]} + {A[0,1]}×{Sigma_X[1,1]} + {A[0,2]}×{Sigma_X[2,1]}, {A[0,0]}×{Sigma_X[0,2]} + {A[0,1]}×{Sigma_X[1,2]} + {A[0,2]}×{Sigma_X[2,2]}]"
    ASigma_row1_step3 = f"[{A[0,0]*Sigma_X[0,0]} + {A[0,1]*Sigma_X[1,0]} + {A[0,2]*Sigma_X[2,0]}, {A[0,0]*Sigma_X[0,1]} + {A[0,1]*Sigma_X[1,1]} + {A[0,2]*Sigma_X[2,1]}, {A[0,0]*Sigma_X[0,2]} + {A[0,1]*Sigma_X[1,2]} + {A[0,2]*Sigma_X[2,2]}]"
    ASigma_row1_result = f"[{A[0,0]*Sigma_X[0,0] + A[0,1]*Sigma_X[1,0] + A[0,2]*Sigma_X[2,0]}, {A[0,0]*Sigma_X[0,1] + A[0,1]*Sigma_X[1,1] + A[0,2]*Sigma_X[2,1]}, {A[0,0]*Sigma_X[0,2] + A[0,1]*Sigma_X[1,2] + A[0,2]*Sigma_X[2,2]}]"
    print(ASigma_row1_step1)
    print(ASigma_row1_step2)
    print(ASigma_row1_step3)
    print(ASigma_row1_result)
    
    # Second row of A·Σ_X
    print("\nSecond row of A·Σ_X calculation:")
    ASigma_row2_step1 = f"[{A[1,0]}, {A[1,1]}, {A[1,2]}] · {Sigma_X.tolist()}"
    ASigma_row2_step2 = f"[{A[1,0]}×{Sigma_X[0,0]} + {A[1,1]}×{Sigma_X[1,0]} + {A[1,2]}×{Sigma_X[2,0]}, {A[1,0]}×{Sigma_X[0,1]} + {A[1,1]}×{Sigma_X[1,1]} + {A[1,2]}×{Sigma_X[2,1]}, {A[1,0]}×{Sigma_X[0,2]} + {A[1,1]}×{Sigma_X[1,2]} + {A[1,2]}×{Sigma_X[2,2]}]"
    ASigma_row2_step3 = f"[{A[1,0]*Sigma_X[0,0]} + {A[1,1]*Sigma_X[1,0]} + {A[1,2]*Sigma_X[2,0]}, {A[1,0]*Sigma_X[0,1]} + {A[1,1]*Sigma_X[1,1]} + {A[1,2]*Sigma_X[2,1]}, {A[1,0]*Sigma_X[0,2]} + {A[1,1]*Sigma_X[1,2]} + {A[1,2]*Sigma_X[2,2]}]"
    ASigma_row2_result = f"[{A[1,0]*Sigma_X[0,0] + A[1,1]*Sigma_X[1,0] + A[1,2]*Sigma_X[2,0]}, {A[1,0]*Sigma_X[0,1] + A[1,1]*Sigma_X[1,1] + A[1,2]*Sigma_X[2,1]}, {A[1,0]*Sigma_X[0,2] + A[1,1]*Sigma_X[1,2] + A[1,2]*Sigma_X[2,2]}]"
    print(ASigma_row2_step1)
    print(ASigma_row2_step2)
    print(ASigma_row2_step3)
    print(ASigma_row2_result)
    
    # Compute A·Σ_X
    ASigma = np.dot(A, Sigma_X)
    print(f"\nResult of A·Σ_X = \n{ASigma}")
    
    # Detailed calculation of (A·Σ_X)·A^T
    print("\nSTEP 2.2: Calculate (A·Σ_X)·A^T:")
    print(f"A^T = \n{A.T}")
    print("Where A^T is the transpose of A, with dimensions 3×2.")
    
    # Calculate elements of Σ_Y
    print("\nElement-by-element calculation of Σ_Y:")
    
    # Upper-left element (0,0) - should be the dot product of first row of ASigma with first column of A^T
    ASigma_00 = ASigma[0, 0] * A.T[0, 0] + ASigma[0, 1] * A.T[1, 0] + ASigma[0, 2] * A.T[2, 0]
    print(f"Σ_Y[0,0] = {ASigma[0, 0]}×{A.T[0, 0]} + {ASigma[0, 1]}×{A.T[1, 0]} + {ASigma[0, 2]}×{A.T[2, 0]} = {ASigma_00}")
    
    # Upper-right element (0,1) - should be the dot product of first row of ASigma with second column of A^T
    ASigma_01 = ASigma[0, 0] * A.T[0, 1] + ASigma[0, 1] * A.T[1, 1] + ASigma[0, 2] * A.T[2, 1]
    print(f"Σ_Y[0,1] = {ASigma[0, 0]}×{A.T[0, 1]} + {ASigma[0, 1]}×{A.T[1, 1]} + {ASigma[0, 2]}×{A.T[2, 1]} = {ASigma_01}")
    
    # Lower-left element (1,0) - should be the dot product of second row of ASigma with first column of A^T
    ASigma_10 = ASigma[1, 0] * A.T[0, 0] + ASigma[1, 1] * A.T[1, 0] + ASigma[1, 2] * A.T[2, 0]
    print(f"Σ_Y[1,0] = {ASigma[1, 0]}×{A.T[0, 0]} + {ASigma[1, 1]}×{A.T[1, 0]} + {ASigma[1, 2]}×{A.T[2, 0]} = {ASigma_10}")
    
    # Lower-right element (1,1) - should be the dot product of second row of ASigma with second column of A^T
    ASigma_11 = ASigma[1, 0] * A.T[0, 1] + ASigma[1, 1] * A.T[1, 1] + ASigma[1, 2] * A.T[2, 1]
    print(f"Σ_Y[1,1] = {ASigma[1, 0]}×{A.T[0, 1]} + {ASigma[1, 1]}×{A.T[1, 1]} + {ASigma[1, 2]}×{A.T[2, 1]} = {ASigma_11}")
    
    # Verify our manual calculations match the matrix multiplication
    Sigma_Y_manual = np.array([
        [ASigma_00, ASigma_01],
        [ASigma_10, ASigma_11]
    ])
    
    # Compute the full covariance matrix using numpy
    Sigma_Y = np.dot(ASigma, A.T)
    print(f"\nFinal result: Σ_Y = A·Σ_X·A^T = \n{Sigma_Y}")
    
    # Verify that manual calculation matches numpy calculation
    print(f"\nVerification - are manual and numpy calculations equal? {np.allclose(Sigma_Y_manual, Sigma_Y)}")
    
    print("\nThus, Y follows the distribution:")
    print(f"Y ~ N({mu_Y}, \n{Sigma_Y})")
    
    # (b) Calculate Cov(Y₁, Y₂)
    print("\n" + "-"*60)
    print("(b) Calculating Cov(Y₁, Y₂):")
    print("-"*60)
    
    cov_Y1_Y2 = Sigma_Y[0, 1]
    print(f"The covariance between Y₁ and Y₂ is given by the off-diagonal element of Σ_Y:")
    print(f"Cov(Y₁, Y₂) = Σ_Y[0,1] = {cov_Y1_Y2}")
    
    # (c) Are Y₁ and Y₂ independent?
    print("\n" + "-"*60)
    print("(c) Determining if Y₁ and Y₂ are independent:")
    print("-"*60)
    
    if abs(cov_Y1_Y2) < 1e-10:  # Check for numerical precision issues
        print("Since Cov(Y₁, Y₂) = 0, Y₁ and Y₂ are independent.")
        print("For multivariate normal distributions, zero covariance is equivalent to independence.")
    else:
        print(f"Since Cov(Y₁, Y₂) = {cov_Y1_Y2} ≠ 0, Y₁ and Y₂ are NOT independent.")
        print("For multivariate normal distributions, zero covariance is equivalent to independence.")
        print("The non-zero covariance indicates that knowledge of one variable provides information about the other.")
    
    # Calculate correlation coefficient
    corr = cov_Y1_Y2 / np.sqrt(Sigma_Y[0, 0] * Sigma_Y[1, 1])
    print(f"\nFurthermore, the correlation coefficient is:")
    print(f"ρ = Cov(Y₁, Y₂) / (σ_Y₁ · σ_Y₂) = {cov_Y1_Y2} / (√{Sigma_Y[0, 0]} · √{Sigma_Y[1, 1]}) = {corr:.6f}")
    
    # Create visualizations of the resulting distribution
    print("\n" + "-"*60)
    print("Creating visualizations:")
    print("-"*60)
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the parent directory
    parent_dir = os.path.dirname(current_dir)
    # Use Images/Linear_Transformations directory
    images_dir = os.path.join(parent_dir, "Images", "Linear_Transformations")
    
    # Create directory if it doesn't exist
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate grid for plotting
    x1 = np.linspace(mu_Y[0] - 3*np.sqrt(Sigma_Y[0, 0]), mu_Y[0] + 3*np.sqrt(Sigma_Y[0, 0]), 100)
    x2 = np.linspace(mu_Y[1] - 3*np.sqrt(Sigma_Y[1, 1]), mu_Y[1] + 3*np.sqrt(Sigma_Y[1, 1]), 100)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))
    
    # Create multivariate normal distribution
    rv = multivariate_normal(mu_Y, Sigma_Y)
    
    # First visualization: PDF contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, rv.pdf(pos), cmap='viridis', levels=10)
    plt.title('PDF Contour Plot')
    plt.xlabel('Y1')
    plt.ylabel('Y2')
    plt.grid(True)
    plt.plot(mu_Y[0], mu_Y[1], 'ro', markersize=8, label='Mean')
    plt.legend()
    plt.tight_layout()
    
    # Save the contour plot
    save_path1 = os.path.join(images_dir, "example1_pdf_contour.png")
    plt.savefig(save_path1, bbox_inches='tight', dpi=300)
    print(f"\nContour plot saved to: {save_path1}")
    plt.close()
    
    # Second visualization: Covariance ellipses
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, rv.pdf(pos), cmap='viridis', levels=20, alpha=0.6)
    plt.title('Covariance Ellipses')
    plt.xlabel('Y1')
    plt.ylabel('Y2')
    plt.grid(True)
    
    # Plot mean point
    plt.plot(mu_Y[0], mu_Y[1], 'ro', markersize=8)
    
    # Add covariance ellipses (1σ, 2σ, 3σ)
    eigenvalues, eigenvectors = np.linalg.eig(Sigma_Y)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    for n_std in [1, 2, 3]:
        # Calculate angle of ellipse
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Calculate width and height
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        
        # Create ellipse
        ellipse = patches.Ellipse(xy=(mu_Y[0], mu_Y[1]), 
                                width=width, 
                                height=height, 
                                angle=angle,
                                fill=False, 
                                edgecolor='red', 
                                linewidth=1,
                                alpha=0.7,
                                label=f'{n_std} sigma' if n_std==1 else None)
        plt.gca().add_patch(ellipse)
    
    plt.legend()
    plt.tight_layout()
    
    # Save the ellipses plot
    save_path2 = os.path.join(images_dir, "example1_covariance_ellipses.png")
    plt.savefig(save_path2, bbox_inches='tight', dpi=300)
    print(f"Covariance ellipses plot saved to: {save_path2}")
    plt.close()
    
    # Third visualization: 3D surface plot of PDF
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, rv.pdf(pos), cmap='viridis', alpha=0.7)
    ax.set_title('3D PDF Surface')
    ax.set_xlabel('Y1')
    ax.set_ylabel('Y2')
    ax.set_zlabel('Density')
    
    # Mark the mean point projection on the x-y plane
    ax.scatter([mu_Y[0]], [mu_Y[1]], [0], color='red', s=50)
    
    # Calculate the PDF value at the mean (the highest point)
    # For a multivariate normal, this is simply 1/sqrt((2π)^k|Σ|)
    k = 2  # bivariate case
    det_sigma = np.linalg.det(Sigma_Y)
    max_pdf = 1 / np.sqrt((2 * np.pi)**k * det_sigma)
    
    # Draw a line from the x-y plane to the PDF height at the mean
    ax.plot([mu_Y[0], mu_Y[0]], [mu_Y[1], mu_Y[1]], [0, max_pdf], 'r-', linewidth=2)
    
    plt.tight_layout()
    
    # Save the 3D plot
    save_path3 = os.path.join(images_dir, "example1_3d_surface.png")
    plt.savefig(save_path3, bbox_inches='tight', dpi=300)
    print(f"3D surface plot saved to: {save_path3}")
    plt.close()
    
    # Fourth visualization: Joint Distribution (scatter plot with marginals)
    plt.figure(figsize=(8, 6))
    
    # Generate random samples from the bivariate normal distribution
    np.random.seed(42)  # Set seed for reproducibility
    samples = np.random.multivariate_normal(mu_Y, Sigma_Y, 1000)
    
    # Create a joint distribution plot using seaborn
    joint_plot = sns.jointplot(
        x=samples[:, 0], 
        y=samples[:, 1], 
        kind="scatter",
        marginal_kws=dict(bins=20, fill=True),
        joint_kws=dict(alpha=0.5)
    )
    
    # Add regression line to show the relationship
    joint_plot.plot_joint(sns.regplot, scatter=False, line_kws={'color': 'red'})
    
    # Set labels
    joint_plot.set_axis_labels('Y1', 'Y2')
    
    # Add a title
    plt.suptitle(f'Joint and Marginal Distributions (Correlation = {corr:.3f})', y=1.02)
    
    # Save the joint plot
    save_path4 = os.path.join(images_dir, "example1_joint_distribution.png")
    plt.savefig(save_path4, bbox_inches='tight', dpi=300)
    print(f"Joint distribution plot saved to: {save_path4}")
    plt.close()
    
    # Summary
    print("\n" + "="*60)
    print("Summary of Example 1:")
    print("="*60)
    print(f"1. Y follows a bivariate normal distribution with mean vector μ_Y = {mu_Y}")
    print(f"2. The covariance matrix of Y is Σ_Y = \n{Sigma_Y}")
    print(f"3. Cov(Y₁, Y₂) = {cov_Y1_Y2}, which is non-zero")
    print(f"4. The correlation coefficient between Y₁ and Y₂ is ρ = {corr:.3f}")
    print(f"5. Y₁ and Y₂ are NOT independent because their covariance is non-zero")
    print("6. This example demonstrates how linear transformations affect multivariate normal distributions")
    print("7. Key insight: A linear transformation of a multivariate normal distribution remains multivariate normal")
    
    # Return the file paths for the markdown
    return [save_path1, save_path2, save_path3, save_path4]

if __name__ == "__main__":
    example1_linear_transformation() 