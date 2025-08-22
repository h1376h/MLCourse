import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

def create_simple_1d_data_visualization():
    """Create simple visualization showing only data points for L5.1 Quiz Question 25."""
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Data points
    X = np.array([1, 2, 3.5, 4, 5])
    y = np.array([-1, -1, -1, 1, 1])
    
    # Plot data points only
    for i, (x_val, label) in enumerate(zip(X, y)):
        if label == -1:
            ax.scatter(x_val, 0, c='blue', s=150, marker='o',
                      facecolors='none', edgecolors='blue', linewidth=3, 
                      label='Class -1' if i == 0 else "")
        else:
            ax.scatter(x_val, 0, c='red', s=150, marker='o',
                      facecolors='red', edgecolors='black', linewidth=2,
                      label='Class +1' if i == 3 else "")
    
    # Annotate points with their values
    for x_val, label in zip(X, y):
        ax.annotate(f'x={x_val}\ny={label}', 
                   (x_val, 0), 
                   xytext=(0, 20), textcoords='offset points', 
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel('x')
    ax.set_title('1D Dataset for Linear SVM')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Remove y-axis ticks since this is 1D
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1d_svm_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple 1D data visualization saved to {save_dir}")

if __name__ == "__main__":
    create_simple_1d_data_visualization()
