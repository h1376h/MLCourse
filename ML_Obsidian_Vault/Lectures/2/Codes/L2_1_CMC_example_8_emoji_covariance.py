import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def emoji_covariance_example():
    """Create a fun example using emoji-like shapes to show covariance concepts."""
    # Print detailed step-by-step solution
    print("\n" + "="*80)
    print("Example: Positive vs Negative Correlation (The Emoji Edition)")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How can we intuitively understand positive and negative correlation using everyday visual metaphors?")
    
    print("\nStep 1: Creating Visual Metaphors for Correlation Patterns")
    print("We will use emoji-like faces to represent different correlation patterns:")
    print("- Smiley face for quadratic relationship with points following a smile curve")
    print("- Sad face for quadratic relationship with points following a frown curve")
    
    # Create figure with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    n_points = 100
    np.random.seed(42) # for reproducibility of noise
    noise_level = 0.2 # Adjust noise level for scatter

    print("\nStep 2: Drawing the Smiley Face Pattern Example")
    print("Key Concepts Illustrated for the Smiley Face Pattern:")
    print("- Points follow a U-shaped pattern (quadratic relationship)")
    print("- Data points visually follow the smiling curve")
    print("- The covariance ellipse reflects the overall spread of data")
    print("- This pattern has very low linear correlation because it's not a linear relationship")
    
    # Create a smiley face for the positive correlation that matches the data pattern
    theta = np.linspace(0, 2*np.pi, 100)
    # Face circle
    face_x = 3 * np.cos(theta)
    face_y = 3 * np.sin(theta)
    
    # Eyes (positioned to match data pattern)
    eye_left_x = -1.5 + 0.5 * np.cos(theta)
    eye_left_y = 1 + 0.5 * np.sin(theta)
    
    eye_right_x = 1.5 + 0.5 * np.cos(theta)
    eye_right_y = 1 + 0.5 * np.sin(theta)
    
    # Smiling mouth curve (parabola-like)
    mouth_x_range = np.linspace(-2, 2, 50)
    mouth_y_curve = -1 + 0.3 * mouth_x_range**2 # U-shape for smile

    # Plot the happy face on the left subplot
    ax1.plot(face_x, face_y, 'k-', linewidth=2)
    ax1.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax1.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax1.plot(mouth_x_range, mouth_y_curve, 'k-', linewidth=2) # Plot the actual mouth curve

    # Generate data points along the smiling mouth curve with noise
    data_x_pos = np.random.uniform(-2, 2, n_points)
    data_y_pos = -1 + 0.3 * data_x_pos**2 + np.random.normal(0, noise_level, n_points)
    rv_pos = np.vstack((data_x_pos, data_y_pos)).T # Combine into (n_points, 2) array

    # Calculate the actual covariance matrix and correlation from the generated data
    mean_pos = np.mean(rv_pos, axis=0)
    cov_pos = np.cov(rv_pos, rowvar=False)
    corr_pos = np.corrcoef(rv_pos, rowvar=False)[0, 1]

    # Print detailed calculation steps for positive correlation based on generated data
    print("\nDetailed Calculation for Smiley Face Pattern (from generated data):")
    print(f"Mean vector = {mean_pos}")
    print(f"Covariance Matrix = \n{cov_pos}")
    print(f"Variance of X = {cov_pos[0, 0]}")
    print(f"Variance of Y = {cov_pos[1, 1]}")
    print(f"Covariance(X,Y) = {cov_pos[0, 1]}")
    print(f"Correlation Coefficient = Covariance(X,Y) / (√Var(X) × √Var(Y))")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / (√{cov_pos[0, 0]} × √{cov_pos[1, 1]})")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / ({np.sqrt(cov_pos[0, 0])} × {np.sqrt(cov_pos[1, 1])})")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / {np.sqrt(cov_pos[0, 0]) * np.sqrt(cov_pos[1, 1])}")
    print(f"Calculated Correlation Coefficient = {corr_pos}")
    
    # Plot the positive correlation data points
    ax1.scatter(rv_pos[:, 0], rv_pos[:, 1], color='blue', alpha=0.3, s=30)
    
    # Add positive covariance ellipse based on calculated covariance
    eigenvalues_pos, eigenvectors_pos = np.linalg.eig(cov_pos)
    
    # Print eigenvalue decomposition details
    print("\nEigenvalue Decomposition (Smiley Face Pattern - from data):")
    print(f"Eigenvalues = {eigenvalues_pos}")
    print(f"Eigenvectors = \n{eigenvectors_pos}")
    print("Interpretation: The principal axes of the data's spread")
    # Use 2*std dev for ellipse boundary (approx 95% interval for normal)
    width_pos = 2 * np.sqrt(eigenvalues_pos[0]) * 2 # width corresponding to 2 std devs
    height_pos = 2 * np.sqrt(eigenvalues_pos[1]) * 2 # height corresponding to 2 std devs
    print(f"Ellipse semi-axis 1 (2*sqrt(lambda1)) = {2*np.sqrt(eigenvalues_pos[0])}")
    print(f"Ellipse semi-axis 2 (2*sqrt(lambda2)) = {2*np.sqrt(eigenvalues_pos[1])}")

    # Calculate the angle of the ellipse from the eigenvectors
    # Angle of the first eigenvector
    ellipse_angle_pos = np.rad2deg(np.arctan2(eigenvectors_pos[1, 0], eigenvectors_pos[0, 0]))
    print(f"Ellipse angle = {ellipse_angle_pos} degrees")
    
    # Draw the covariance ellipse centered at the data mean
    ell_pos = Ellipse(xy=mean_pos,
                     width=width_pos,
                     height=height_pos,
                     angle=ellipse_angle_pos,
                     edgecolor='blue', facecolor='none', linestyle='--')
    ax1.add_patch(ell_pos)
    
    # Set title and labels
    ax1.set_title(f'Smiley Face Pattern\nCorrelation = {corr_pos:.2f} (Near Zero!)')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    print("\nStep 3: Drawing the Frowny Face Pattern Example")
    print("Key Concepts Illustrated for the Frowny Face Pattern:")
    print("- Points follow an inverted U-shaped pattern (quadratic relationship)")
    print("- Data points visually follow the frowning curve")
    print("- The covariance ellipse reflects the overall spread of data")
    print("- This pattern has almost zero linear correlation despite the clear pattern")
    
    # Sad mouth curve (inverted parabola-like)
    sad_mouth_x_range = np.linspace(-2, 2, 50)
    sad_mouth_y_curve = -1 - 0.3 * sad_mouth_x_range**2 # Inverted U-shape for frown
    
    # Plot the sad face
    ax2.plot(face_x, face_y, 'k-', linewidth=2)
    ax2.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax2.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax2.plot(sad_mouth_x_range, sad_mouth_y_curve, 'k-', linewidth=2) # Plot the actual mouth curve

    # Generate data points along the sad mouth curve with noise
    data_x_neg = np.random.uniform(-2, 2, n_points)
    data_y_neg = -1 - 0.3 * data_x_neg**2 + np.random.normal(0, noise_level, n_points)
    rv_neg = np.vstack((data_x_neg, data_y_neg)).T # Combine into (n_points, 2) array

    # Calculate the actual covariance matrix and correlation from the generated data
    mean_neg = np.mean(rv_neg, axis=0)
    cov_neg = np.cov(rv_neg, rowvar=False)
    corr_neg = np.corrcoef(rv_neg, rowvar=False)[0, 1]

    # Print detailed calculation steps for negative correlation based on generated data
    print("\nDetailed Calculation for Frowny Face Pattern (from generated data):")
    print(f"Mean vector = {mean_neg}")
    print(f"Covariance Matrix = \n{cov_neg}")
    print(f"Variance of X = {cov_neg[0, 0]}")
    print(f"Variance of Y = {cov_neg[1, 1]}")
    print(f"Covariance(X,Y) = {cov_neg[0, 1]}")
    print(f"Correlation Coefficient = Covariance(X,Y) / (√Var(X) × √Var(Y))")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / (√{cov_neg[0, 0]} × √{cov_neg[1, 1]})")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / ({np.sqrt(cov_neg[0, 0])} × {np.sqrt(cov_neg[1, 1])})")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / {np.sqrt(cov_neg[0, 0]) * np.sqrt(cov_neg[1, 1])}")
    print(f"Calculated Correlation Coefficient = {corr_neg}")
    
    # Plot the negative correlation data points
    ax2.scatter(rv_neg[:, 0], rv_neg[:, 1], color='red', alpha=0.3, s=30)
    
    # Add negative covariance ellipse based on calculated covariance
    eigenvalues_neg, eigenvectors_neg = np.linalg.eig(cov_neg)
    
    # Print eigenvalue decomposition details
    print("\nEigenvalue Decomposition (Frowny Face Pattern - from data):")
    print(f"Eigenvalues = {eigenvalues_neg}")
    print(f"Eigenvectors = \n{eigenvectors_neg}")
    print("Interpretation: The principal axes of the data's spread")
    # Use 2*std dev for ellipse boundary
    width_neg = 2 * np.sqrt(eigenvalues_neg[0]) * 2
    height_neg = 2 * np.sqrt(eigenvalues_neg[1]) * 2
    print(f"Ellipse semi-axis 1 (2*sqrt(lambda1)) = {2*np.sqrt(eigenvalues_neg[0])}")
    print(f"Ellipse semi-axis 2 (2*sqrt(lambda2)) = {2*np.sqrt(eigenvalues_neg[1])}")

    # Calculate the angle of the ellipse
    ellipse_angle_neg = np.rad2deg(np.arctan2(eigenvectors_neg[1, 0], eigenvectors_neg[0, 0]))
    print(f"Ellipse angle = {ellipse_angle_neg} degrees")
    
    # Draw the covariance ellipse centered at the data mean
    ell_neg = Ellipse(xy=mean_neg,
                     width=width_neg,
                     height=height_neg,
                     angle=ellipse_angle_neg,
                     edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(ell_neg)
    
    # Set title and labels
    ax2.set_title(f'Frowny Face Pattern\nCorrelation = {corr_neg:.2f} (Almost Zero!)')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    print("\nStep 4: Visual Mnemonic and Interpretation")
    print("The smiley/sad faces provide an intuitive memory aid:")
    print("- Smile curves upward ⌣, data points follow this shape, showing a general trend where y decreases then increases as x changes.")
    print("- Frown curves downward ⌢, data points follow this shape, showing a trend where y increases then decreases as x changes.")
    print("- Note: The *linear* correlation coefficient (ρ) is near zero in both cases because the relationships aren't linear, but the visual patterns are clear.")

    plt.tight_layout()
    
    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save the original emoji faces figure
        save_path = os.path.join(images_dir, "ex8_emoji_covariance_example.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
        
    # Create a simple educational visualization showing why linear correlation fails to capture non-linear relationships
    fig_explanation = plt.figure(figsize=(12, 5))
    
    # Left panel - Linear scatter plot with high correlation
    ax1 = fig_explanation.add_subplot(131)
    
    # Generate linearly correlated data
    x_linear = np.linspace(-3, 3, 30)
    y_linear = 0.8 * x_linear + np.random.normal(0, 0.5, len(x_linear))
    corr_linear = np.corrcoef(x_linear, y_linear)[0, 1]
    
    ax1.scatter(x_linear, y_linear, color='blue', s=30)
    ax1.axline((0, 0), slope=0.8, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Linear Relationship\nCorrelation = {corr_linear:.2f}')
    
    # Middle panel - Quadratic scatter plot (U shape)
    ax2 = fig_explanation.add_subplot(132)
    
    # Generate quadratic pattern
    x_quad = np.linspace(-3, 3, 30)
    y_quad = 0.8 * x_quad**2 + np.random.normal(0, 0.5, len(x_quad))
    corr_quad = np.corrcoef(x_quad, y_quad)[0, 1]
    
    ax2.scatter(x_quad, y_quad, color='green', s=30)
    xx = np.linspace(-3, 3, 100)
    ax2.plot(xx, 0.8 * xx**2, 'r--', alpha=0.7)
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-0.5, 8)
    ax2.set_aspect(0.6)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Quadratic Relationship\nCorrelation = {corr_quad:.2f}')
    
    # Right panel - Sinusoidal scatter plot
    ax3 = fig_explanation.add_subplot(133)
    
    # Generate sinusoidal pattern
    x_sin = np.linspace(-3, 3, 30)
    y_sin = 2 * np.sin(x_sin * 1.5) + np.random.normal(0, 0.5, len(x_sin))
    corr_sin = np.corrcoef(x_sin, y_sin)[0, 1]
    
    ax3.scatter(x_sin, y_sin, color='purple', s=30)
    xx = np.linspace(-3, 3, 100)
    ax3.plot(xx, 2 * np.sin(xx * 1.5), 'r--', alpha=0.7)
    ax3.set_xlim(-3.5, 3.5)
    ax3.set_ylim(-3, 3)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Sinusoidal Relationship\nCorrelation = {corr_sin:.2f}')
    
    plt.tight_layout()
    
    # Save the educational figure
    try:
        explanation_path = os.path.join(images_dir, "ex8_linear_vs_nonlinear_explanation.png")
        fig_explanation.savefig(explanation_path, bbox_inches='tight', dpi=300)
        print(f"Educational explanation about linear vs non-linear correlation saved to: {explanation_path}")
    except Exception as e:
        print(f"\nError saving educational figure: {e}")
        
    # Print summary message
    print("\nEXAMPLE: EMOJI VISUALIZATION OF CORRELATION")
    print("\nKey Insights:")
    print("1. The linear correlation coefficient is near zero for both emoji patterns despite clear visual relationships")
    print("2. This demonstrates a limitation of linear correlation for capturing non-linear patterns")
    print("3. Always visualize your data - plotting can reveal patterns that correlation coefficients miss")
    print("4. Consider applying non-linear correlation measures for curved relationships")

if __name__ == "__main__":
    emoji_covariance_example()