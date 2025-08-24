import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

def create_simple_xor_visualization():
    """Create simple visualization showing only XOR data points for L5.3 Quiz Question 1."""
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # XOR data points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])  # XOR labels
    
    # Colors and markers
    colors = ['blue' if label == -1 else 'red' for label in y]
    markers = ['o' if label == -1 else '^' for label in y]
    
    # Plot data points only
    for i, (point, color, marker) in enumerate(zip(X, colors, markers)):
        ax.scatter(point[0], point[1], c=color, marker=marker, s=200, 
                   edgecolors='black', linewidth=2)
        ax.annotate(f'({point[0]}, {point[1]})\ny={y[i]}', 
                    (point[0], point[1]), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=10, ha='left')
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('XOR Problem Dataset')
    ax.grid(True, alpha=0.3)
    
    # Add legend for classes
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Class +1')
    blue_patch = mpatches.Patch(color='blue', label='Class -1')
    ax.legend(handles=[red_patch, blue_patch])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xor_kernel_transformation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple XOR data visualization saved to {save_dir}")

if __name__ == "__main__":
    create_simple_xor_visualization()
