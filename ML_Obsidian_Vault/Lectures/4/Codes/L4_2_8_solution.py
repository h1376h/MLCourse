import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Check if necessary directories exist
print(f"Script directory: {script_dir}")
print(f"Images directory: {images_dir}")
print(f"Save directory: {save_dir}")

# Enable LaTeX style plotting if available
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    print("LaTeX rendering enabled")
except:
    plt.rcParams['text.usetex'] = False
    print("LaTeX rendering disabled - using standard text")

# Question parameters
w = np.array([1, 2, -1])  # Initial weight vector [w1, w2, w0]
x = np.array([2, 0, 1])   # Feature vector with bias term [x1, x2, 1]
y = -1                    # True label
eta = 0.5                 # Learning rate

print("===== Perceptron Learning Rule Calculation =====")
print(f"Initial weight vector (w): {w}")
print(f"Feature vector (x): {x}")
print(f"True label (y): {y}")
print(f"Learning rate (η): {eta}")
print("=" * 45)

# Task 2: Check if perceptron would make a prediction error
activation = np.dot(w, x)
prediction = np.sign(activation)

print("\n===== Task 2: Prediction Before Update =====")
print(f"Activation = w · x = {w} · {x} = {activation}")
print(f"Prediction = sign({activation}) = {prediction}")
print(f"True label = {y}")

if prediction != y:
    print("✓ The perceptron WOULD make a prediction error!")
    print(f"   The prediction ({prediction}) is different from true label ({y}).")
else:
    print("✗ The perceptron would NOT make a prediction error.")
    print(f"   The prediction ({prediction}) matches the true label ({y}).")
print("=" * 45)

# Task 1: Calculate updated weight vector
print("\n===== Task 1: Weight Update Calculation =====")
print(f"w_new = w_old + η · y · x")
print(f"w_new = {w} + {eta} · ({y}) · {x}")

# Calculate the update term
update_term = eta * y * x
print(f"Update term = {eta} · ({y}) · {x} = {update_term}")

# Calculate new weights
w_new = w + update_term
print(f"w_new = {w} + ({update_term}) = {w_new}")
print("=" * 45)

# Visualize the decision boundaries (2D projection)
def plot_decision_boundary():
    plt.figure(figsize=(10, 8))
    
    # Set up the grid for visualization
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Original decision boundary: w1*x1 + w2*x2 + w0 = 0
    Z1 = w[0]*X1 + w[1]*X2 + w[2]
    
    # New decision boundary: w_new1*x1 + w_new2*x2 + w_new0 = 0
    Z2 = w_new[0]*X1 + w_new[1]*X2 + w_new[2]
    
    # Plot the contours at the zero level (decision boundaries)
    cs1 = plt.contour(X1, X2, Z1, levels=[0], colors='blue', linestyles='-', linewidths=2)
    cs2 = plt.contour(X1, X2, Z2, levels=[0], colors='red', linestyles='--', linewidths=2)
    
    # Add labels for the contours
    plt.clabel(cs1, inline=1, fontsize=10, fmt={0: 'Original boundary'})
    plt.clabel(cs2, inline=1, fontsize=10, fmt={0: 'New boundary'})
    
    # Plot the feature point (without the bias term)
    plt.scatter(x[0], x[1], s=200, color='green', marker='o')
    
    # Fill the regions corresponding to the classifications
    plt.contourf(X1, X2, Z1, levels=[-100, 0, 100], colors=['#FFCCCC', '#CCCCFF'], alpha=0.3)
    
    # Add arrows to indicate the direction of the weight vectors
    origin = [0, 0]
    # Scale down the weights for better visualization
    scale = 1.5
    plt.arrow(origin[0], origin[1], w[0]/scale, w[1]/scale, head_width=0.15, 
              head_length=0.15, fc='blue', ec='blue')
    plt.arrow(origin[0], origin[1], w_new[0]/scale, w_new[1]/scale, head_width=0.15, 
              head_length=0.15, fc='red', ec='red')
    
    # Add weight vector labels
    plt.annotate('$w$', xy=(w[0]/scale + 0.1, w[1]/scale), color='blue', fontsize=12)
    plt.annotate('$w_{new}$', xy=(w_new[0]/scale + 0.1, w_new[1]/scale), color='red', fontsize=12)
    
    # Add regions labels
    plt.annotate('Predict $+1$', xy=(2.5, 2.5), color='#0000FF', fontsize=12)
    plt.annotate('Predict $-1$', xy=(-2.5, -2.5), color='#FF0000', fontsize=12)
    
    # Set up the plot
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Perceptron Decision Boundary Before and After Update')
    
    # Create a legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label=f'Original boundary: {w[0]}$x_1$ + {w[1]}$x_2$ + {w[2]} = 0'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, label=f'New boundary: {w_new[0]}$x_1$ + {w_new[1]}$x_2$ + {w_new[2]} = 0'),
        plt.scatter([], [], marker='o', color='green', s=100, label=f'Data point: $x = [{x[0]}, {x[1]}]^T$, $y = {y}$')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Add explanation text box
    textbox_text = (
        f"Initial prediction: {prediction}\n"
        f"True label: {y}\n"
        f"Prediction error: {prediction != y}\n"
        f"Update rule: $w_{{new}} = w + \\eta \\cdot y \\cdot x$\n"
        f"$w_{{new}} = {w} + {eta} \\cdot ({y}) \\cdot {x}$\n"
        f"$w_{{new}} = {w_new}$"
    )
    plt.figtext(0.02, 0.02, textbox_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perceptron_decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Visualize the feature space before and after update
plot_decision_boundary()

print("\nPlot showing the decision boundaries before and after the weight update has been saved.")

# Create a 3D visualization to better show the weight update in feature space
def plot_3d_visualization():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for visualization
    x1_range = np.linspace(-3, 3, 20)
    x2_range = np.linspace(-3, 3, 20)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Points for the original hyperplane
    Z1 = (-w[0]*X1 - w[1]*X2 - w[2]) / 1  # Assuming the coefficient for bias is 1
    
    # Points for the new hyperplane
    Z2 = (-w_new[0]*X1 - w_new[1]*X2 - w_new[2]) / 1
    
    # Plot the hyperplanes
    ax.plot_surface(X1, X2, Z1, color='blue', alpha=0.3)
    ax.plot_surface(X1, X2, Z2, color='red', alpha=0.3)
    
    # Plot the feature point (extended to the hyperplane)
    z1_point = (-w[0]*x[0] - w[1]*x[1] - w[2]) / 1
    z2_point = (-w_new[0]*x[0] - w_new[1]*x[1] - w_new[2]) / 1
    
    # Plot the feature point and its projections
    ax.scatter(x[0], x[1], 0, s=100, color='green', marker='o')
    
    # Connect the point to its projections on both hyperplanes
    ax.plot([x[0], x[0]], [x[1], x[1]], [0, z1_point], 'g--', alpha=0.5)
    ax.plot([x[0], x[0]], [x[1], x[1]], [0, z2_point], 'g--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$z$')
    ax.set_title('3D Visualization of Perceptron Decision Boundaries')
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='Original hyperplane'),
        Patch(facecolor='red', edgecolor='red', alpha=0.3, label='Updated hyperplane'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Feature point')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add explanation text
    textbox_text = (
        f"Initial weight: $w = {w}$\n"
        f"Updated weight: $w_{{new}} = {w_new}$\n"
        f"Feature point: $x = [{x[0]}, {x[1]}, {x[2]}]^T$ ($y={y}$)\n"
        f"Learning rate: $\\eta = {eta}$"
    )
    plt.figtext(0.02, 0.02, textbox_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perceptron_3d_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create 3D visualization
plot_3d_visualization()

print("\n3D visualization of the hyperplanes has been saved.")

# Additional visualization: Weight update vector
def plot_weight_update_vector():
    plt.figure(figsize=(10, 8))
    
    # Plot the original weight vector
    plt.arrow(0, 0, w[0], w[1], head_width=0.15, head_length=0.15, 
              fc='blue', ec='blue', label='Original weight vector')
    
    # Plot the update term
    plt.arrow(w[0], w[1], update_term[0], update_term[1], head_width=0.15, head_length=0.15, 
              fc='green', ec='green', label='Update term')
    
    # Plot the new weight vector
    plt.arrow(0, 0, w_new[0], w_new[1], head_width=0.15, head_length=0.15, 
              fc='red', ec='red', label='New weight vector')
    
    # Add vector labels
    plt.annotate('$w$', xy=(w[0] + 0.1, w[1]), color='blue', fontsize=12)
    plt.annotate('$\\eta \\cdot y \\cdot x$', xy=(w[0] + update_term[0]/2, w[1] + update_term[1]/2), 
                color='green', fontsize=12)
    plt.annotate('$w_{{new}}$', xy=(w_new[0] + 0.1, w_new[1]), color='red', fontsize=12)
    
    # Set up the plot
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 3)
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.title('Weight Vector Update Visualization')
    plt.legend(loc='upper left')
    
    # Add explanation text box
    textbox_text = (
        f"Original weight: $w = [{w[0]}, {w[1]}, {w[2]}]^T$\n"
        f"Update term: $\\eta \\cdot y \\cdot x = {eta} \\cdot ({y}) \\cdot {x} = {update_term}$\n"
        f"New weight: $w_{{new}} = w + \\eta \\cdot y \\cdot x = {w_new}$"
    )
    plt.figtext(0.02, 0.02, textbox_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perceptron_weight_update.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Visualize the weight update
plot_weight_update_vector()

print("\nVisualization of the weight update vector has been saved.")

# Summary
print("\n===== Summary =====")
print(f"Original weight vector: w = {w}")
print(f"Feature vector: x = {x}")
print(f"True label: y = {y}")
print(f"Learning rate: η = {eta}")
print(f"Activation before update: {activation}")
print(f"Prediction before update: {prediction}")
print(f"Was there a prediction error? {'Yes' if prediction != y else 'No'}")
print(f"Update rule: w_new = w + η · y · x")
print(f"Update term: η · y · x = {update_term}")
print(f"Updated weight vector: w_new = {w_new}")
print("=" * 45) 