import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_1_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

def create_simple_svm_data_visualization():
    """Create simple visualization showing only data points for L5.1 Quiz Question 24."""
    
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create linearly separable data 
    class_x = np.array([[1.5, 3], [2, 3.5], [3, 2.5], [2.5, 2], [3.5, 1.5]])  # 'x' markers
    class_o = np.array([[0.5, 1], [1, 0.5], [0, 1.5], [1.5, 0.8]])  # 'o' markers
    
    # Plot data points only
    ax.scatter(class_x[:, 0], class_x[:, 1], c='red', marker='x', s=150, 
               linewidth=3, label="Class 'x'")
    ax.scatter(class_o[:, 0], class_o[:, 1], c='blue', marker='o', s=100, 
               label="Class 'o'", facecolors='none', edgecolors='blue', linewidth=2)
    
    ax.set_xlim(-0.2, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Linearly Separable Dataset')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'svm_loocv_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple SVM data visualization saved to {save_dir}")

if __name__ == "__main__":
    create_simple_svm_data_visualization()
