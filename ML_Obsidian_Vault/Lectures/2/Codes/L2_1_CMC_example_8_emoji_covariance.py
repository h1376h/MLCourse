import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    print("- Smiley face for positive correlation: variables tend to increase or decrease together")
    print("- Sad face for negative correlation: as one variable increases, the other decreases")
    
    # Create figure with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    print("\nStep 2: Drawing the Positive Correlation (Smiley Face) Example")
    print("Key Concepts Illustrated for Positive Correlation:")
    print("- Variables tend to increase or decrease together")
    print("- Data points form a pattern from bottom-left to top-right")
    print("- The covariance ellipse is tilted along the y = x direction")
    print("- Common in naturally related quantities (height-weight, study time-grades)")
    
    # Create a smiley face for the positive correlation
    theta = np.linspace(0, 2*np.pi, 100)
    # Face circle
    face_x = 3 * np.cos(theta)
    face_y = 3 * np.sin(theta)
    
    # Eyes (ellipses showing covariance)
    eye_left_x = -1.2 + 0.5 * np.cos(theta)
    eye_left_y = 1 + 0.5 * np.sin(theta)
    
    eye_right_x = 1.2 + 0.5 * np.cos(theta)
    eye_right_y = 1 + 0.5 * np.sin(theta)
    
    # Smiling mouth (showing positive correlation)
    mouth_theta = np.linspace(0, np.pi, 50)
    mouth_x = 2 * np.cos(mouth_theta)
    mouth_y = -1 + 1.2 * np.sin(mouth_theta)
    
    # Plot the happy face on the left subplot
    ax1.plot(face_x, face_y, 'k-', linewidth=2)
    ax1.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax1.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax1.plot(mouth_x, mouth_y, 'k-', linewidth=2)
    
    # Generate and plot positive correlated data
    np.random.seed(42)
    cov_pos = np.array([[1.0, 0.8], [0.8, 1.0]])  # Positive correlation
    mean = np.array([0, 0])
    
    # Calculate the correlation coefficient
    corr_pos = cov_pos[0, 1] / np.sqrt(cov_pos[0, 0] * cov_pos[1, 1])
    
    # Print detailed calculation steps for positive correlation
    print("\nDetailed Calculation for Positive Correlation:")
    print(f"Covariance Matrix = \n{cov_pos}")
    print(f"Variance of X = {cov_pos[0, 0]}")
    print(f"Variance of Y = {cov_pos[1, 1]}")
    print(f"Covariance(X,Y) = {cov_pos[0, 1]}")
    print(f"Correlation Coefficient = Covariance(X,Y) / (√Var(X) × √Var(Y))")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / (√{cov_pos[0, 0]} × √{cov_pos[1, 1]})")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / ({np.sqrt(cov_pos[0, 0])} × {np.sqrt(cov_pos[1, 1])})")
    print(f"Correlation Coefficient = {cov_pos[0, 1]} / {np.sqrt(cov_pos[0, 0]) * np.sqrt(cov_pos[1, 1])}")
    print(f"Correlation Coefficient = {corr_pos}")
    
    # Generate random data with positive correlation
    rv_pos = np.random.multivariate_normal(mean, cov_pos, 100)
    ax1.scatter(rv_pos[:, 0], rv_pos[:, 1], color='blue', alpha=0.3, s=10)
    
    # Add positive covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)
    
    # Print eigenvalue decomposition details
    print("\nEigenvalue Decomposition (Positive Correlation):")
    print(f"Eigenvalues = {eigenvalues}")
    print(f"Eigenvectors = \n{eigenvectors}")
    print("Interpretation: The principal axes of the correlation ellipse")
    print(f"Major axis length = {4*np.sqrt(max(eigenvalues))}")
    print(f"Minor axis length = {4*np.sqrt(min(eigenvalues))}")
    
    # Calculate the angle of the ellipse
    ellipse_angle = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    print(f"Ellipse angle = {ellipse_angle} degrees")
    
    ell_pos = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=ellipse_angle,
                    edgecolor='blue', facecolor='none', linestyle='--')
    ax1.add_patch(ell_pos)
    
    ax1.set_title(f'Positive Correlation: Happy Data! 😊\nPoints tend to increase together (ρ = {corr_pos:.2f})')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    print("\nStep 3: Drawing the Negative Correlation (Sad Face) Example")
    print("Key Concepts Illustrated for Negative Correlation:")
    print("- As one variable increases, the other tends to decrease")
    print("- Data points form a pattern from top-left to bottom-right")
    print("- The covariance ellipse is tilted along the y = -x direction")
    print("- Common in trade-off relationships (speed-accuracy, price-demand)")
    
    # Create a sad face for the negative correlation
    # Face circle (reuse from above)
    
    # Eyes (reuse from above)
    
    # Sad mouth (showing negative correlation)
    sad_mouth_theta = np.linspace(np.pi, 2*np.pi, 50)
    sad_mouth_x = 2 * np.cos(sad_mouth_theta)
    sad_mouth_y = -1 + 1.2 * np.sin(sad_mouth_theta)
    
    # Plot the sad face
    ax2.plot(face_x, face_y, 'k-', linewidth=2)
    ax2.plot(eye_left_x, eye_left_y, 'k-', linewidth=2)
    ax2.plot(eye_right_x, eye_right_y, 'k-', linewidth=2)
    ax2.plot(sad_mouth_x, sad_mouth_y, 'k-', linewidth=2)
    
    # Generate and plot negative correlated data
    cov_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])  # Negative correlation
    
    # Calculate the correlation coefficient
    corr_neg = cov_neg[0, 1] / np.sqrt(cov_neg[0, 0] * cov_neg[1, 1])
    
    # Print detailed calculation steps for negative correlation
    print("\nDetailed Calculation for Negative Correlation:")
    print(f"Covariance Matrix = \n{cov_neg}")
    print(f"Variance of X = {cov_neg[0, 0]}")
    print(f"Variance of Y = {cov_neg[1, 1]}")
    print(f"Covariance(X,Y) = {cov_neg[0, 1]}")
    print(f"Correlation Coefficient = Covariance(X,Y) / (√Var(X) × √Var(Y))")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / (√{cov_neg[0, 0]} × √{cov_neg[1, 1]})")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / ({np.sqrt(cov_neg[0, 0])} × {np.sqrt(cov_neg[1, 1])})")
    print(f"Correlation Coefficient = {cov_neg[0, 1]} / {np.sqrt(cov_neg[0, 0]) * np.sqrt(cov_neg[1, 1])}")
    print(f"Correlation Coefficient = {corr_neg}")
    
    # Generate random data with negative correlation
    rv_neg = np.random.multivariate_normal(mean, cov_neg, 100)
    ax2.scatter(rv_neg[:, 0], rv_neg[:, 1], color='red', alpha=0.3, s=10)
    
    # Add negative covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_neg)
    
    # Print eigenvalue decomposition details
    print("\nEigenvalue Decomposition (Negative Correlation):")
    print(f"Eigenvalues = {eigenvalues}")
    print(f"Eigenvectors = \n{eigenvectors}")
    print("Interpretation: The principal axes of the correlation ellipse")
    print(f"Major axis length = {4*np.sqrt(max(eigenvalues))}")
    print(f"Minor axis length = {4*np.sqrt(min(eigenvalues))}")
    
    # Calculate the angle of the ellipse
    ellipse_angle = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    print(f"Ellipse angle = {ellipse_angle} degrees")
    
    ell_neg = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=ellipse_angle,
                    edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(ell_neg)
    
    ax2.set_title(f'Negative Correlation: Sad Data! 😢\nAs one variable increases, the other decreases (ρ = {corr_neg:.2f})')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    print("\nStep 4: Visual Mnemonic and Interpretation")
    print("The smiley/sad faces provide an intuitive memory aid:")
    print("- Smile curves upward ⌣ like positive correlation")
    print("- Frown curves downward ⌢ like negative correlation")
    
    print("\nEmotional Interpretation (Just for Fun):")
    print("- Positive correlation makes data 'happy' because variables 'agree' with each other")
    print("- Negative correlation makes data 'sad' because variables 'disagree' with each other")
    print("- Zero correlation is 'emotionless' data - variables show no relationship")
    
    print("\nStep 5: Practical Value of Visual Metaphors")
    print("Learning Value:")
    print("- Memorable visualizations help anchor abstract statistical concepts")
    print("- Connecting emotional resonance to mathematical patterns enhances retention")
    print("- Visual intuition complements formal mathematical understanding")
    print("- Analogies make complex concepts more accessible and engaging")
    
    plt.tight_layout()
    
    # Additional visualizations for different correlation strengths and zero correlation
    print("\nStep 6: Creating Additional Visualizations for Different Correlation Strengths")
    
    # Create a new figure for the "correlation spectrum"
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    # Define correlation values to visualize
    correlations = [-0.95, -0.5, 0.0, 0.2, 0.6, 0.95]
    titles = ["Very Strong Negative", "Moderate Negative", "Zero Correlation", 
              "Weak Positive", "Moderate Positive", "Very Strong Positive"]
    colors = ["darkred", "lightcoral", "purple", "lightblue", "blue", "darkblue"]
    
    # For tracking detailed calculations
    print("\nDetailed Calculations for Multiple Correlation Values:")
    
    # Create a scatter plot for each correlation value
    for i, (corr, title, color) in enumerate(zip(correlations, titles, colors)):
        # Calculate the corresponding covariance matrix
        cov = np.array([[1.0, corr], [corr, 1.0]])
        
        print(f"\n{title} Correlation (ρ = {corr}):")
        print(f"Covariance Matrix = \n{cov}")
        
        # Generate random data with the specified correlation
        data = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=100)
        
        # Plot the data
        axes[i].scatter(data[:, 0], data[:, 1], color=color, alpha=0.5, s=20)
        axes[i].set_title(f"{title}\n(ρ = {corr:.2f})")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].grid(True)
        axes[i].set_aspect('equal')
        
        # Add a covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        print(f"Eigenvalues = {eigenvalues}")
        print(f"Eigenvectors = \n{eigenvectors}")
        
        # For zero correlation, handle the special case (circle)
        if corr == 0.0:
            print("For zero correlation, the ellipse becomes a circle")
            print("Both eigenvalues are equal, indicating equal spread in all directions")
            ellipse = Ellipse(xy=(0, 0),
                            width=4*np.sqrt(eigenvalues[0]),
                            height=4*np.sqrt(eigenvalues[1]),
                            angle=0,  # Angle doesn't matter for a circle
                            edgecolor=color, facecolor='none', linestyle='--')
        else:
            ellipse_angle = np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            print(f"Ellipse angle = {ellipse_angle} degrees")
            
            ellipse = Ellipse(xy=(0, 0),
                            width=4*np.sqrt(eigenvalues[0]),
                            height=4*np.sqrt(eigenvalues[1]),
                            angle=ellipse_angle,
                            edgecolor=color, facecolor='none', linestyle='--')
        
        axes[i].add_patch(ellipse)
        
        # Calculate and show the sample correlation
        sample_corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        print(f"Sample correlation from generated data = {sample_corr}")
        print(f"Difference from target correlation = {abs(sample_corr - corr)}")
    
    # Add a neutral face for zero correlation in the middle plot
    neutral_face_index = 2  # Index of the zero correlation plot
    
    # Draw a neutral face
    center_x, center_y = 0, 0
    face_radius = 2
    
    # Face circle
    face_theta = np.linspace(0, 2*np.pi, 100)
    face_x = center_x + face_radius * np.cos(face_theta)
    face_y = center_y + face_radius * np.sin(face_theta)
    
    # Eyes
    left_eye_x = center_x - 0.8 + 0.3 * np.cos(face_theta)
    left_eye_y = center_y + 0.7 + 0.3 * np.sin(face_theta)
    
    right_eye_x = center_x + 0.8 + 0.3 * np.cos(face_theta)
    right_eye_y = center_y + 0.7 + 0.3 * np.sin(face_theta)
    
    # Straight mouth for neutral correlation
    mouth_x = np.array([center_x - 1.3, center_x + 1.3])
    mouth_y = np.array([center_y - 0.5, center_y - 0.5])
    
    # Plot the neutral face
    axes[neutral_face_index].plot(face_x, face_y, 'k-', linewidth=2, alpha=0.5)
    axes[neutral_face_index].plot(left_eye_x, left_eye_y, 'k-', linewidth=2, alpha=0.5)
    axes[neutral_face_index].plot(right_eye_x, right_eye_y, 'k-', linewidth=2, alpha=0.5)
    axes[neutral_face_index].plot(mouth_x, mouth_y, 'k-', linewidth=2, alpha=0.5)
    
    # Add explanation for the neutral face
    print("\nNeutral Face for Zero Correlation:")
    print("- The straight mouth represents no relationship between variables")
    print("- This corresponds to a circular distribution (equal spread in all directions)")
    print("- No linear predictive relationship between X and Y")
    
    # Add overall title
    fig2.suptitle('The Correlation Spectrum: From Strong Negative to Strong Positive', fontsize=16)
    
    # Create an additional figure to demonstrate the calculation of covariance
    print("\nStep 7: Demonstrating Covariance Calculation with a Simple Example")
    
    # Create a small dataset for demonstration
    np.random.seed(123)
    n_samples = 8
    
    # For positive correlation
    x_pos = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y_pos = np.array([2, 3, 3, 5, 6, 7, 8, 9]) + np.random.normal(0, 0.5, n_samples)
    
    # For negative correlation
    x_neg = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y_neg = np.array([9, 8, 7, 6, 5, 4, 3, 2]) + np.random.normal(0, 0.5, n_samples)
    
    # Create a figure to visualize the calculation
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot positive correlation example
    ax1.scatter(x_pos, y_pos, color='blue', s=100)
    ax1.set_title('Positive Correlation Example')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True)
    
    # Plot negative correlation example
    ax2.scatter(x_neg, y_neg, color='red', s=100)
    ax2.set_title('Negative Correlation Example')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True)
    
    # Calculate and print detailed steps for covariance/correlation
    print("\nDetailed Calculation Example - Step by Step:")
    
    # Positive correlation calculation
    print("\nPositive Correlation Example:")
    print(f"Data points: \nx = {x_pos}\ny = {y_pos}")
    
    # Calculate means
    mean_x_pos = np.mean(x_pos)
    mean_y_pos = np.mean(y_pos)
    print(f"Mean of x = {mean_x_pos}")
    print(f"Mean of y = {mean_y_pos}")
    
    # Calculate deviations from means
    dev_x_pos = x_pos - mean_x_pos
    dev_y_pos = y_pos - mean_y_pos
    print(f"Deviations of x from mean = {dev_x_pos}")
    print(f"Deviations of y from mean = {dev_y_pos}")
    
    # Calculate products of deviations
    prod_devs_pos = dev_x_pos * dev_y_pos
    print(f"Products of deviations = {prod_devs_pos}")
    print(f"Sum of products = {np.sum(prod_devs_pos)}")
    
    # Calculate covariance
    cov_pos_sample = np.sum(prod_devs_pos) / (n_samples - 1)  # Sample covariance
    print(f"Sample Covariance = Sum of products / (n - 1) = {np.sum(prod_devs_pos)} / {n_samples - 1} = {cov_pos_sample}")
    
    # Calculate variances
    var_x_pos = np.sum(dev_x_pos ** 2) / (n_samples - 1)
    var_y_pos = np.sum(dev_y_pos ** 2) / (n_samples - 1)
    print(f"Variance of x = {var_x_pos}")
    print(f"Variance of y = {var_y_pos}")
    
    # Calculate correlation coefficient
    corr_pos_sample = cov_pos_sample / (np.sqrt(var_x_pos) * np.sqrt(var_y_pos))
    print(f"Correlation coefficient = {cov_pos_sample} / (√{var_x_pos} × √{var_y_pos}) = {corr_pos_sample}")
    
    # Negative correlation calculation
    print("\nNegative Correlation Example:")
    print(f"Data points: \nx = {x_neg}\ny = {y_neg}")
    
    # Calculate means
    mean_x_neg = np.mean(x_neg)
    mean_y_neg = np.mean(y_neg)
    print(f"Mean of x = {mean_x_neg}")
    print(f"Mean of y = {mean_y_neg}")
    
    # Calculate deviations from means
    dev_x_neg = x_neg - mean_x_neg
    dev_y_neg = y_neg - mean_y_neg
    print(f"Deviations of x from mean = {dev_x_neg}")
    print(f"Deviations of y from mean = {dev_y_neg}")
    
    # Calculate products of deviations
    prod_devs_neg = dev_x_neg * dev_y_neg
    print(f"Products of deviations = {prod_devs_neg}")
    print(f"Sum of products = {np.sum(prod_devs_neg)}")
    
    # Calculate covariance
    cov_neg_sample = np.sum(prod_devs_neg) / (n_samples - 1)  # Sample covariance
    print(f"Sample Covariance = Sum of products / (n - 1) = {np.sum(prod_devs_neg)} / {n_samples - 1} = {cov_neg_sample}")
    
    # Calculate variances
    var_x_neg = np.sum(dev_x_neg ** 2) / (n_samples - 1)
    var_y_neg = np.sum(dev_y_neg ** 2) / (n_samples - 1)
    print(f"Variance of x = {var_x_neg}")
    print(f"Variance of y = {var_y_neg}")
    
    # Calculate correlation coefficient
    corr_neg_sample = cov_neg_sample / (np.sqrt(var_x_neg) * np.sqrt(var_y_neg))
    print(f"Correlation coefficient = {cov_neg_sample} / (√{var_x_neg} × √{var_y_neg}) = {corr_neg_sample}")
    
    # Add annotations to the plots
    ax1.annotate(f"Correlation = {corr_pos_sample:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    
    ax2.annotate(f"Correlation = {corr_neg_sample:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    print("\nStep 8: Summarizing Key Insights")
    print("\nCorrelation Coefficient Properties:")
    print("- Ranges from -1 to 1")
    print("- Sign indicates direction of relationship")
    print("- Magnitude indicates strength of relationship")
    print("- Equals 0 for no linear relationship")
    print("- Is invariant to linear transformations of variables")
    
    return fig, fig2, fig3

def create_arrow_visualization():
    """Create a visual explanation of correlation with arrows showing direction and strength."""
    print("\n" + "="*80)
    print("Additional Visualization: Arrow Representation of Correlation")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How can we visualize correlation using directional arrows to reinforce the concept?")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate data for different correlations
    np.random.seed(42)
    n_points = 100
    
    # Positive correlation (ρ ≈ 0.8)
    cov_pos = np.array([[1.0, 0.8], [0.8, 1.0]])
    data_pos = np.random.multivariate_normal([0, 0], cov_pos, n_points)
    
    # No correlation (ρ ≈ 0)
    cov_zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    data_zero = np.random.multivariate_normal([0, 0], cov_zero, n_points)
    
    # Negative correlation (ρ ≈ -0.8)
    cov_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    data_neg = np.random.multivariate_normal([0, 0], cov_neg, n_points)
    
    # Plot data
    ax1.scatter(data_pos[:, 0], data_pos[:, 1], color='blue', alpha=0.6)
    ax2.scatter(data_zero[:, 0], data_zero[:, 1], color='purple', alpha=0.6)
    ax3.scatter(data_neg[:, 0], data_neg[:, 1], color='red', alpha=0.6)
    
    # Print step-by-step calculations
    print("\nStep 1: Calculating Correlation for Arrow Direction")
    
    # Calculate correlations
    corr_pos = np.corrcoef(data_pos[:, 0], data_pos[:, 1])[0, 1]
    corr_zero = np.corrcoef(data_zero[:, 0], data_zero[:, 1])[0, 1]
    corr_neg = np.corrcoef(data_neg[:, 0], data_neg[:, 1])[0, 1]
    
    print(f"\nPositive Correlation Dataset:")
    print(f"Correlation coefficient = {corr_pos:.4f}")
    print("Arrow Direction: Points up and right (↗)")
    
    print(f"\nZero Correlation Dataset:")
    print(f"Correlation coefficient = {corr_zero:.4f}")
    print("Arrow Direction: No clear direction, arrows point in all directions")
    
    print(f"\nNegative Correlation Dataset:")
    print(f"Correlation coefficient = {corr_neg:.4f}")
    print("Arrow Direction: Points down and right (↘)")
    
    # Add arrows to visualize correlation direction
    print("\nStep 2: Drawing Directional Arrows to Represent Correlation")
    
    # For positive correlation: draw arrows in the correlation direction
    for i in range(0, n_points, 10):
        # Start at the data point
        x, y = data_pos[i, 0], data_pos[i, 1]
        # Arrow length proportional to distance from origin
        arrow_length = 0.4
        angle = np.pi/4  # 45 degrees for positive correlation
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)
    
    # For zero correlation: draw arrows in random directions
    for i in range(0, n_points, 10):
        x, y = data_zero[i, 0], data_zero[i, 1]
        arrow_length = 0.4
        angle = np.random.uniform(0, 2*np.pi)  # Random direction
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        ax2.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='purple', ec='purple', alpha=0.7)
    
    # For negative correlation: draw arrows in the correlation direction
    for i in range(0, n_points, 10):
        x, y = data_neg[i, 0], data_neg[i, 1]
        arrow_length = 0.4
        angle = -np.pi/4  # -45 degrees for negative correlation
        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)
        ax3.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    # Add titles and labels
    ax1.set_title(f'Positive Correlation\n(ρ = {corr_pos:.2f})\nMovement: Together ↗', fontsize=12)
    ax2.set_title(f'Zero Correlation\n(ρ = {corr_zero:.2f})\nMovement: Random', fontsize=12)
    ax3.set_title(f'Negative Correlation\n(ρ = {corr_neg:.2f})\nMovement: Opposite ↘', fontsize=12)
    
    # Set equal aspect ratio for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    # Add a main title
    fig.suptitle('Correlation as Direction of Movement', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    print("\nStep 3: Interpreting the Arrow Visualization")
    print("- For positive correlation: Arrows point upward and to the right ↗")
    print("  This shows that as x increases, y tends to increase too")
    print("- For zero correlation: Arrows point in random directions")
    print("  This shows that knowing x tells us nothing about y's direction")
    print("- For negative correlation: Arrows point downward and to the right ↘")
    print("  This shows that as x increases, y tends to decrease")
    
    return fig

def create_3d_correlation_visualization():
    """Create a 3D visualization of correlation showing the joint probability density."""
    print("\n" + "="*80)
    print("Additional Visualization: 3D Representation of Correlation")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How can we visualize the joint probability density function in 3D to understand correlation?")
    
    # Create figure with three 3D subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Generate data for different correlations
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Function to compute 2D Gaussian PDF
    def multivariate_gaussian(pos, mu, Sigma):
        """Calculate multivariate Gaussian distribution."""
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        
        # Calculate exponent for each point
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
        return np.exp(-fac / 2) / N
    
    # Print step-by-step calculations for the PDF
    print("\nStep 1: Setting up the 3D Visualization Parameters")
    print("We're using a bivariate normal distribution to show correlation in 3D.")
    print("The probability density function (PDF) is:")
    print("f(x,y) = (1/(2π|Σ|^0.5)) * exp(-0.5 * [(x,y) - μ]ᵀ Σ⁻¹ [(x,y) - μ])")
    
    # Parameters for different correlations
    corrs = [0.8, 0.0, -0.8]
    titles = ["Positive Correlation", "Zero Correlation", "Negative Correlation"]
    colors = ["Blues", "Purples", "Reds"]
    
    # Loop through the three correlation values
    for i, (corr, title, cmap) in enumerate(zip(corrs, titles, colors)):
        # Create covariance matrix
        cov = np.array([[1.0, corr], [corr, 1.0]])
        
        # Calculate PDF
        mu = np.array([0.0, 0.0])
        
        print(f"\n{title} (ρ = {corr}):")
        print(f"Covariance Matrix Σ = \n{cov}")
        print(f"Mean Vector μ = {mu}")
        print(f"Determinant |Σ| = {np.linalg.det(cov)}")
        print(f"Inverse Matrix Σ⁻¹ = \n{np.linalg.inv(cov)}")
        
        # Sample point to illustrate calculation
        sample_point = np.array([1.0, 1.0])
        
        # Manual calculation of PDF for a sample point
        n = 2  # bivariate
        Sigma_det = np.linalg.det(cov)
        Sigma_inv = np.linalg.inv(cov)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        diff = sample_point - mu
        fac = diff @ Sigma_inv @ diff
        pdf_value = np.exp(-fac / 2) / N
        
        print(f"\nDetailed PDF Calculation for point {sample_point}:")
        print(f"1. Compute normalization factor: N = √((2π)² × |Σ|) = {N:.4f}")
        print(f"2. Compute difference from mean: (x,y) - μ = {diff}")
        print(f"3. Compute Mahalanobis distance: (x,y - μ)ᵀ Σ⁻¹ (x,y - μ) = {fac:.4f}")
        print(f"4. Compute PDF: f(x,y) = exp(-0.5 × {fac:.4f}) / {N:.4f} = {pdf_value:.6f}")
        
        # Compute the PDF over the grid
        Z = multivariate_gaussian(pos, mu, cov)
        
        # Create the 3D subplot
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.7)
        
        # Add contour plot at the bottom
        offset = np.min(Z) - 0.05
        ax.contour(X, Y, Z, zdir='z', offset=offset, cmap=cmap)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Probability Density')
        ax.set_title(f'{title}\n(ρ = {corr})')
        
        # Adjust view angle
        ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    
    print("\nStep 3: Interpreting the 3D Visualization")
    print("- For positive correlation: The PDF forms a ridge along the y=x direction")
    print("  High probability density is concentrated where x and y move together")
    print("- For zero correlation: The PDF forms a circular, symmetric bell shape")
    print("  No preferred direction of relationship between x and y")
    print("- For negative correlation: The PDF forms a ridge along the y=-x direction")
    print("  High probability density where x and y move in opposite directions")
    
    return fig

def create_geometric_area_visualization():
    """Create a visualization showing correlation as geometric area."""
    print("\n" + "="*80)
    print("Additional Visualization: Geometric Area Interpretation of Covariance")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How can we visualize covariance as a geometric area to provide an intuitive understanding?")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate data for different correlations
    np.random.seed(42)
    n_points = 200
    
    # Positive correlation (ρ ≈ 0.8)
    cov_pos = np.array([[1.0, 0.8], [0.8, 1.0]])
    data_pos = np.random.multivariate_normal([0, 0], cov_pos, n_points)
    
    # No correlation (ρ ≈ 0)
    cov_zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    data_zero = np.random.multivariate_normal([0, 0], cov_zero, n_points)
    
    # Negative correlation (ρ ≈ -0.8)
    cov_neg = np.array([[1.0, -0.8], [-0.8, 1.0]])
    data_neg = np.random.multivariate_normal([0, 0], cov_neg, n_points)
    
    # Step 1: Calculate means
    print("\nStep 1: Calculate Means and Center the Data")
    
    # For demonstrations, we'll select a subset of points to keep the visualization clean
    selected_indices = np.random.choice(range(n_points), 20, replace=False)
    
    # Function to calculate and display centered data
    def process_data(data, ax, color, title):
        mean_x = np.mean(data[:, 0])
        mean_y = np.mean(data[:, 1])
        
        print(f"\n{title}:")
        print(f"Mean X = {mean_x:.4f}")
        print(f"Mean Y = {mean_y:.4f}")
        
        # Center the data (subtract means)
        centered_data = data.copy()
        centered_data[:, 0] = data[:, 0] - mean_x
        centered_data[:, 1] = data[:, 1] - mean_y
        
        # Plot all data points in light color
        ax.scatter(data[:, 0], data[:, 1], color=color, alpha=0.2, s=20)
        
        # Plot selected points in darker color
        selected_data = data[selected_indices]
        selected_centered = centered_data[selected_indices]
        
        ax.scatter(selected_data[:, 0], selected_data[:, 1], color=color, s=50)
        
        # Plot mean point
        ax.scatter(mean_x, mean_y, color='black', s=100, marker='X', label='Mean')
        
        # Draw lines to mean for selected points to visualize deviations
        for i in range(len(selected_data)):
            ax.plot([selected_data[i, 0], mean_x], [selected_data[i, 1], mean_y], 
                    color=color, alpha=0.5, linestyle='--')
        
        print(f"Centered a subset of data points by subtracting the mean")
        
        # Return the centered data for further calculations
        return selected_centered, selected_data
    
    # Process and plot data for each correlation type
    centered_pos, selected_pos = process_data(data_pos, ax1, 'blue', "Positive Correlation")
    centered_zero, selected_zero = process_data(data_zero, ax2, 'purple', "Zero Correlation")
    centered_neg, selected_neg = process_data(data_neg, ax3, 'red', "Negative Correlation")
    
    # Step 2: Calculate and visualize the areas representing covariance
    print("\nStep 2: Calculate Covariance as Area")
    
    # Function to calculate and visualize covariance as area
    def visualize_covariance_area(centered_data, selected_data, mean_x, mean_y, ax, color, title):
        # Calculate covariance
        cov_xy = np.mean(centered_data[:, 0] * centered_data[:, 1])
        
        # Sample calculation for a few points
        n_samples = min(5, len(centered_data))
        
        print(f"\n{title} - Covariance Calculation:")
        print(f"Cov(X,Y) = (1/n) * Σ[(x_i - μ_x) * (y_i - μ_y)]")
        print("For the first few points:")
        
        sum_product = 0
        for i in range(n_samples):
            dx = centered_data[i, 0]
            dy = centered_data[i, 1]
            product = dx * dy
            sum_product += product
            
            print(f"Point {i+1}: (x={selected_data[i, 0]:.2f}, y={selected_data[i, 1]:.2f})")
            print(f"  Deviation x: {dx:.4f}, Deviation y: {dy:.4f}")
            print(f"  Product: {dx:.4f} × {dy:.4f} = {product:.4f}")
            
            # Draw rectangle representing the area (dx * dy)
            if abs(product) > 0.1:  # Only draw rectangles for non-tiny areas
                # Determine rectangle corners
                if dx > 0 and dy > 0:  # Positive area (1st quadrant)
                    rect = plt.Rectangle((0, 0), dx, dy, alpha=0.2, color=color)
                elif dx < 0 and dy < 0:  # Positive area (3rd quadrant)
                    rect = plt.Rectangle((dx, dy), -dx, -dy, alpha=0.2, color=color)
                elif dx > 0 and dy < 0:  # Negative area (4th quadrant)
                    rect = plt.Rectangle((0, dy), dx, -dy, alpha=0.2, color='gray')
                else:  # Negative area (2nd quadrant)
                    rect = plt.Rectangle((dx, 0), -dx, dy, alpha=0.2, color='gray')
                
                ax.add_patch(rect)
                
                # Add text annotation for the area
                area_text = f"{product:.2f}"
                ax.text(dx/2 if dx > 0 else dx/2, dy/2 if dy > 0 else dy/2, 
                        area_text, ha='center', va='center', fontsize=8)
        
        print(f"Sum of products = {sum_product:.4f}")
        print(f"Covariance = (1/{n_samples}) × {sum_product:.4f} = {sum_product/n_samples:.4f}")
        
        # Calculate correlation
        var_x = np.var(centered_data[:, 0], ddof=0)
        var_y = np.var(centered_data[:, 1], ddof=0)
        corr_xy = cov_xy / np.sqrt(var_x * var_y)
        
        print(f"Variance X = {var_x:.4f}")
        print(f"Variance Y = {var_y:.4f}")
        print(f"Correlation = {cov_xy:.4f} / (√{var_x:.4f} × √{var_y:.4f}) = {corr_xy:.4f}")
        
        return cov_xy, corr_xy
    
    # Visualize covariance as area for each correlation type
    cov_pos, corr_pos = visualize_covariance_area(centered_pos, selected_pos, 
                                                 np.mean(data_pos[:, 0]), np.mean(data_pos[:, 1]), 
                                                 ax1, 'blue', "Positive Correlation")
    
    cov_zero, corr_zero = visualize_covariance_area(centered_zero, selected_zero, 
                                                   np.mean(data_zero[:, 0]), np.mean(data_zero[:, 1]), 
                                                   ax2, 'purple', "Zero Correlation")
    
    cov_neg, corr_neg = visualize_covariance_area(centered_neg, selected_neg, 
                                                 np.mean(data_neg[:, 0]), np.mean(data_neg[:, 1]), 
                                                 ax3, 'red', "Negative Correlation")
    
    # Set titles and labels
    ax1.set_title(f'Positive Correlation\nCov(X,Y) = {cov_pos:.2f}, ρ = {corr_pos:.2f}\nMostly Positive Areas', fontsize=12)
    ax2.set_title(f'Zero Correlation\nCov(X,Y) = {cov_zero:.2f}, ρ = {corr_zero:.2f}\nPositive & Negative Areas Balance', fontsize=12)
    ax3.set_title(f'Negative Correlation\nCov(X,Y) = {cov_neg:.2f}, ρ = {corr_neg:.2f}\nMostly Negative Areas', fontsize=12)
    
    # Set equal aspect ratio for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.legend()
    
    # Add a main title
    fig.suptitle('Covariance as Geometric Area', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    print("\nStep 3: Interpreting the Geometric Area Visualization")
    print("- In positive correlation: Most rectangles contribute positive area")
    print("  Points tend to be in 1st and 3rd quadrants (x,y both positive or both negative)")
    print("- In zero correlation: Positive and negative areas roughly cancel out")
    print("  Points are evenly distributed across all quadrants")
    print("- In negative correlation: Most rectangles contribute negative area")
    print("  Points tend to be in 2nd and 4th quadrants (x,y have opposite signs)")
    
    return fig

def create_correlation_map_visualization():
    """Create a correlation map to show how correlation changes with different data patterns."""
    print("\n" + "="*80)
    print("Additional Visualization: Correlation Map")
    print("="*80)
    
    print("\nProblem Statement:")
    print("How do different data patterns result in different correlation values?")
    
    # Create a large figure for different patterns
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Function to generate and plot data with specified pattern
    def create_pattern(ax, pattern_type, n_points=100, noise=0.3):
        np.random.seed(42)  # For reproducibility
        
        # Generate pattern data
        x = np.linspace(-3, 3, n_points)
        
        if pattern_type == "linear_pos":
            # Strong positive linear relationship
            y = x + np.random.normal(0, noise, n_points)
            title = "Strong Positive Linear"
            color = "darkblue"
        elif pattern_type == "linear_neg":
            # Strong negative linear relationship
            y = -x + np.random.normal(0, noise, n_points)
            title = "Strong Negative Linear"
            color = "darkred"
        elif pattern_type == "quadratic":
            # Quadratic relationship (parabola)
            y = x**2 + np.random.normal(0, noise, n_points)
            title = "Quadratic (U-shaped)"
            color = "purple"
        elif pattern_type == "sine":
            # Sine wave relationship
            y = np.sin(x) + np.random.normal(0, noise/2, n_points)
            title = "Sine Wave"
            color = "green"
        elif pattern_type == "circle":
            # Circular pattern
            theta = np.linspace(0, 2*np.pi, n_points)
            radius = 2
            x = radius * np.cos(theta) + np.random.normal(0, noise/3, n_points)
            y = radius * np.sin(theta) + np.random.normal(0, noise/3, n_points)
            title = "Circular Pattern"
            color = "orange"
        elif pattern_type == "cubic":
            # Cubic relationship
            y = x**3 + np.random.normal(0, noise*3, n_points)
            title = "Cubic Relationship"
            color = "brown"
        elif pattern_type == "exponential":
            # Exponential relationship
            y = np.exp(x/2) + np.random.normal(0, noise, n_points)
            title = "Exponential Growth"
            color = "magenta"
        elif pattern_type == "logarithmic":
            # Logarithmic relationship
            x = np.abs(x) + 0.1  # Ensure positive values
            y = np.log(x) + np.random.normal(0, noise/2, n_points)
            title = "Logarithmic Pattern"
            color = "cyan"
        elif pattern_type == "no_relation":
            # No relationship (random)
            y = np.random.normal(0, 1, n_points)
            title = "No Relationship"
            color = "gray"
        
        # Plot the data
        ax.scatter(x, y, color=color, alpha=0.7, s=15)
        
        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        # Print step-by-step calculation for this pattern
        print(f"\n{title} Pattern:")
        print(f"1. Generate x and y according to the pattern")
        
        # Sample five points for detailed calculation
        indices = np.linspace(0, n_points-1, 5, dtype=int)
        print("Sample points (x, y):")
        for i in indices:
            print(f"  ({x[i]:.2f}, {y[i]:.2f})")
        
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        print(f"2. Calculate means: mean_x = {mean_x:.4f}, mean_y = {mean_y:.4f}")
        
        # Calculate deviations
        dev_x = x - mean_x
        dev_y = y - mean_y
        
        # Print sample deviations
        print("3. Calculate deviations from means for sample points:")
        for i in indices:
            print(f"  Point {i}: dev_x = {dev_x[i]:.4f}, dev_y = {dev_y[i]:.4f}")
        
        # Calculate products of deviations
        prod_devs = dev_x * dev_y
        
        # Print sample products
        print("4. Calculate products of deviations for sample points:")
        for i in indices:
            print(f"  Point {i}: {dev_x[i]:.4f} × {dev_y[i]:.4f} = {prod_devs[i]:.4f}")
        
        # Calculate covariance
        cov = np.mean(prod_devs)
        print(f"5. Calculate covariance: Cov(X,Y) = mean of products = {cov:.4f}")
        
        # Calculate variances
        var_x = np.var(x)
        var_y = np.var(y)
        print(f"6. Calculate variances: Var(X) = {var_x:.4f}, Var(Y) = {var_y:.4f}")
        
        # Calculate correlation coefficient
        print(f"7. Calculate correlation: Corr(X,Y) = {cov:.4f} / (√{var_x:.4f} × √{var_y:.4f}) = {corr:.4f}")
        
        # Set title with correlation value
        ax.set_title(f"{title}\nρ = {corr:.2f}", fontsize=10)
        
        # Make the plot prettier
        ax.grid(True, alpha=0.3)
        ax.set_xlim(min(x)-0.5, max(x)+0.5)
        ax.set_ylim(min(y)-0.5, max(y)+0.5)
        
        return corr
    
    # Define patterns for each subplot
    patterns = [
        "linear_pos", "linear_neg", "quadratic",
        "sine", "circle", "cubic",
        "exponential", "logarithmic", "no_relation"
    ]
    
    # Create all patterns
    print("\nStep 1: Generating Different Data Patterns and Calculating Correlations")
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Create each pattern
    correlations = []
    for i, pattern in enumerate(patterns):
        corr = create_pattern(axes[i], pattern)
        correlations.append(corr)
    
    # Set overall title
    fig.suptitle('Correlation Map: How Different Patterns Affect Correlation (ρ)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    print("\nStep 2: Analyzing the Pattern-Correlation Relationships")
    print("\nObservations about Linear Correlation Coefficient:")
    print("1. Perfect for linear relationships: ρ ≈ 1 or -1 for perfect linear patterns")
    print("2. Completely misses non-linear associations: ρ ≈ 0 for perfect quadratic patterns")
    print("3. Averages to zero for oscillating patterns: sine wave, circular pattern")
    print("4. Sensitive to outliers in higher-order relationships: cubic, exponential")
    print("5. Captures monotonic (always increasing/decreasing) relationships better than others")
    
    print("\nKey Insights:")
    print("- Linear correlation is NOT a measure of general association")
    print("- Always visualize your data - correlation alone can be deceiving")
    print("- Different patterns can yield similar correlation values for very different relationships")
    print("- Consider non-linear measures of association (Spearman, mutual information) for complex patterns")
    
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 8: EMOJI VISUALIZATION OF CORRELATION")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig, fig2, fig3 = emoji_covariance_example()
    
    # Run the additional visualizations
    print("\n\n" + "*"*80)
    print("ADDITIONAL VISUALIZATIONS FOR CORRELATION")
    print("*"*80)
    
    fig4 = create_arrow_visualization()
    fig5 = create_3d_correlation_visualization()
    
    # Create the geometric area visualization
    fig6 = create_geometric_area_visualization()
    
    # Create the correlation map visualization
    fig7 = create_correlation_map_visualization()
    
    # Save the figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        # Save the original emoji faces figure
        save_path = os.path.join(images_dir, "ex8_emoji_covariance_example.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure 1 saved to: {save_path}")
        
        # Save the correlation spectrum figure
        save_path2 = os.path.join(images_dir, "ex8_correlation_spectrum.png")
        fig2.savefig(save_path2, bbox_inches='tight', dpi=300)
        print(f"Figure 2 saved to: {save_path2}")
        
        # Save the calculation example figure
        save_path3 = os.path.join(images_dir, "ex8_correlation_calculation.png")
        fig3.savefig(save_path3, bbox_inches='tight', dpi=300)
        print(f"Figure 3 saved to: {save_path3}")
        
        # Save the new visualizations
        save_path4 = os.path.join(images_dir, "ex8_correlation_arrows.png")
        fig4.savefig(save_path4, bbox_inches='tight', dpi=300)
        print(f"Figure 4 saved to: {save_path4}")
        
        save_path5 = os.path.join(images_dir, "ex8_correlation_3d.png")
        fig5.savefig(save_path5, bbox_inches='tight', dpi=300)
        print(f"Figure 5 saved to: {save_path5}")
        
        save_path6 = os.path.join(images_dir, "ex8_correlation_geometric.png")
        fig6.savefig(save_path6, bbox_inches='tight', dpi=300)
        print(f"Figure 6 saved to: {save_path6}")
        
        save_path7 = os.path.join(images_dir, "ex8_correlation_map.png")
        fig7.savefig(save_path7, bbox_inches='tight', dpi=300)
        print(f"Figure 7 saved to: {save_path7}")
        
    except Exception as e:
        print(f"\nError saving figures: {e}")
    