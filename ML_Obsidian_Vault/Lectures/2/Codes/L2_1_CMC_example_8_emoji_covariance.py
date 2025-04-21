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
    
    # Generate random data with positive correlation
    rv_pos = np.random.multivariate_normal(mean, cov_pos, 100)
    ax1.scatter(rv_pos[:, 0], rv_pos[:, 1], color='blue', alpha=0.3, s=10)
    
    # Add positive covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_pos)
    ell_pos = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
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
    
    # Generate random data with negative correlation
    rv_neg = np.random.multivariate_normal(mean, cov_neg, 100)
    ax2.scatter(rv_neg[:, 0], rv_neg[:, 1], color='red', alpha=0.3, s=10)
    
    # Add negative covariance ellipse
    eigenvalues, eigenvectors = np.linalg.eig(cov_neg)
    ell_neg = Ellipse(xy=(0, 0),
                    width=4*np.sqrt(eigenvalues[0]), 
                    height=4*np.sqrt(eigenvalues[1]),
                    angle=np.rad2deg(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])),
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
    return fig

if __name__ == "__main__":
    print("\n\n" + "*"*80)
    print("EXAMPLE 8: EMOJI VISUALIZATION OF CORRELATION")
    print("*"*80)
    
    # Run the example with detailed step-by-step printing
    fig = emoji_covariance_example()
    
    # Save the figure if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "Images", "Contour_Plots")
    ensure_directory_exists(images_dir)
    
    try:
        save_path = os.path.join(images_dir, "emoji_covariance_example.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"\nFigure saved to: {save_path}")
    except Exception as e:
        print(f"\nError saving figure: {e}")
    