import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_learning_algorithm_diagram():
    """
    Generate a visualization of the learning algorithm flow for linear regression.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training set visualization
    ax.text(0.2, 0.95, "Training Set D", fontsize=14, ha='center')
    ax.add_patch(patches.FancyArrow(0.2, 0.93, 0, -0.1, width=0.01, 
                                  head_width=0.03, head_length=0.03, 
                                  facecolor='black'))
    
    # Learning algorithm box
    rect = patches.Rectangle((0.1, 0.7), 0.2, 0.1, linewidth=2,
                            edgecolor='blue', facecolor='lightblue', alpha=0.5)
    ax.add_patch(rect)
    ax.text(0.2, 0.75, "Learning\nAlgorithm", ha='center', va='center', fontsize=12)
    
    # Parameters arrow and text
    ax.add_patch(patches.FancyArrow(0.2, 0.7, 0, -0.1, width=0.01, 
                                  head_width=0.03, head_length=0.03, 
                                  facecolor='black'))
    ax.text(0.2, 0.55, r"$w_0, w_1$", fontsize=14, ha='center')
    
    # Function box
    rect = patches.Rectangle((0.1, 0.35), 0.2, 0.1, linewidth=2,
                            edgecolor='green', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(rect)
    ax.text(0.2, 0.4, r"$\hat{f}(x) = f(x; \boldsymbol{w})$", ha='center', va='center', fontsize=12)
    
    # Input/output arrows
    ax.text(0.05, 0.4, "Size of\nhouse\n$x$", fontsize=10, ha='right')
    ax.add_patch(patches.FancyArrow(0.05, 0.4, 0.04, 0, width=0.005, 
                                  head_width=0.02, head_length=0.02, 
                                  facecolor='black'))
    
    ax.add_patch(patches.FancyArrow(0.3, 0.4, 0.04, 0, width=0.005, 
                                  head_width=0.02, head_length=0.02, 
                                  facecolor='black'))
    ax.text(0.4, 0.4, "Estimated\nprice", fontsize=10, ha='left')
    
    # Description annotations
    ax.text(0.6, 0.8, "We need to:", fontsize=12, fontweight='bold')
    ax.text(0.6, 0.74, "(1) measure how well $f(x; \\boldsymbol{w})$", fontsize=12)
    ax.text(0.6, 0.7, "approximates the target", fontsize=12)
    
    ax.text(0.6, 0.6, "(2) choose $\\boldsymbol{w}$ to minimize", fontsize=12)
    ax.text(0.6, 0.56, "the error measure", fontsize=12)
    
    # Remove axes
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/learning_algorithm_diagram.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_learning_algorithm_diagram()
    print("Learning algorithm diagram generated successfully.") 