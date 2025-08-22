import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

def create_simple_1d_data_visualization():
    """Create simple visualization showing only data points for L5.2 Quiz Question 22."""
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Data points from the problem
    negative_points = [-3, -2, -1]
    positive_points = [0, 1, 2, 3]
    
    # Plot negative class points
    for x_val in negative_points:
        ax.scatter(x_val, 0, c='blue', marker='o', s=150, 
                  facecolors='none', edgecolors='blue', linewidth=3)
        ax.annotate(f'{x_val}', 
                   (x_val, 0), 
                   xytext=(0, -25), textcoords='offset points', 
                   ha='center', fontsize=10)
    
    # Plot positive class points  
    for x_val in positive_points:
        ax.scatter(x_val, 0, c='red', marker='o', s=150, 
                  facecolors='red', edgecolors='black', linewidth=2)
        ax.annotate(f'{x_val}', 
                   (x_val, 0), 
                   xytext=(0, -25), textcoords='offset points', 
                   ha='center', fontsize=10)
    
    # Add class labels
    ax.text(-2, 0.08, 'Negative Class', ha='center', fontsize=12, weight='bold', color='blue')
    ax.text(1.5, 0.08, 'Positive Class', ha='center', fontsize=12, weight='bold', color='red')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('x')
    ax.set_title('1D Dataset for Soft-Margin SVM')
    ax.grid(True, alpha=0.3)
    
    # Remove y-axis elements since this is 1D
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'soft_margin_1d_svm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simple soft-margin 1D data visualization saved to {save_dir}")

if __name__ == "__main__":
    create_simple_1d_data_visualization()
