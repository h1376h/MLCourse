import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_2_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Given weight vector [w1, w2, w0]
w = np.array([2, -1, -3])
print("Given weight vector w = [w1, w2, w0]^T = ", w)

# 1. Find the equation of the decision boundary
# Decision boundary is where w1*x1 + w2*x2 + w0 = 0
# Rewrite this as x2 = (-w1*x1 - w0)/w2
def decision_boundary(x1, w):
    return (-w[0]*x1 - w[2])/w[1] if w[1] != 0 else None

# Calculate the decision boundary in term of x2 = mx1 + b
slope = -w[0]/w[1] if w[1] != 0 else float('inf')
intercept = -w[2]/w[1] if w[1] != 0 else float('inf')

print("\n1. Decision Boundary Equation:")
print(f"w1*x1 + w2*x2 + w0 = 0")
print(f"{w[0]}*x1 + {w[1]}*x2 + {w[2]} = 0")
if w[1] != 0:
    print(f"x2 = {slope}*x1 + {intercept}")
else:
    x1_value = -w[2]/w[0]
    print(f"x1 = {x1_value}")

# 2. Sketch the decision boundary
plt.figure(figsize=(10, 8))

# Plot decision boundary
x1_range = np.linspace(-5, 5, 100)
if w[1] != 0:  # Non-vertical line
    x2_boundary = decision_boundary(x1_range, w)
    plt.plot(x1_range, x2_boundary, 'g-', label='Decision Boundary')
else:  # Vertical line
    plt.axvline(x=-w[2]/w[0], color='g', linestyle='-', label='Decision Boundary')

# Shade positive and negative regions
if w[1] != 0:
    # Positive region (above or below the line depending on sign of w[1])
    pos_color = 'lightblue' if w[1] < 0 else 'lightpink'
    neg_color = 'lightpink' if w[1] < 0 else 'lightblue'
    
    # Create mesh grid for shading
    x1_grid = np.linspace(-5, 5, 100)
    x2_grid = np.linspace(-5, 5, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Compute the activation for each point
    Z = w[0]*X1 + w[1]*X2 + w[2]
    
    # Shade the regions based on activation sign
    plt.contourf(X1, X2, Z, levels=[-100, 0], colors=[neg_color], alpha=0.3)
    plt.contourf(X1, X2, Z, levels=[0, 100], colors=[pos_color], alpha=0.3)
    
    # Add legend for regions
    plt.fill_between([], [], [], color=pos_color, alpha=0.3, label='Class 1 (positive)')
    plt.fill_between([], [], [], color=neg_color, alpha=0.3, label='Class -1 (negative)')

# Add labels and title
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Perceptron Decision Boundary')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Add equation of the decision boundary to the plot
eq_str = f'$w_1 x_1 + w_2 x_2 + w_0 = 0$\n${w[0]} x_1 + {w[1]} x_2 + {w[2]} = 0$'
plt.annotate(eq_str, xy=(0.05, 0.95), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# 3. Classify the given points
points = np.array([
    [2, 1],   # Point 1
    [1, 3],   # Point 2
])

# Function to predict class using sign function
def predict(x, w):
    # Augment x with 1 for bias term
    x_augmented = np.append(x, 1)
    activation = np.dot(w, x_augmented)
    return np.sign(activation), activation

# Test the points and add to plot
print("\n3. Classification of Points:")
colors = ['blue', 'red']  # Blue for positive class, red for negative class
markers = ['o', 's']      # Circle for positive, square for negative

for i, point in enumerate(points):
    prediction, activation = predict(point, w)
    class_label = 1 if prediction > 0 else -1
    marker_idx = 0 if class_label == 1 else 1
    
    print(f"Point {chr(65+i)} = {point}:")
    print(f"  Activation = w1*x1 + w2*x2 + w0 = {w[0]}*{point[0]} + {w[1]}*{point[1]} + {w[2]} = {activation}")
    print(f"  Prediction = {class_label}")
    
    # Plot the point
    plt.scatter(point[0], point[1], s=150, color=colors[marker_idx], 
                marker=markers[marker_idx], edgecolor='black', linewidth=1.5, 
                label=f'Point {chr(65+i)} ({class_label})')
    
    # Add point label
    plt.annotate(f'({point[0]}, {point[1]})',
                 (point[0], point[1]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# 4. Update the weight vector for the misclassified point
print("\n4. Weight Update for Misclassified Point:")
# Point (1, 3) with true label y = 1
point = np.array([1, 3])
true_label = 1
learning_rate = 1

# Augment point with 1 for bias term
point_augmented = np.append(point, 1)

# Get current prediction
prediction, activation = predict(point, w)
class_label = 1 if prediction > 0 else -1

print(f"Point = {point}, True label = {true_label}")
print(f"Current prediction = {class_label}")

if class_label != true_label:
    # Update weights
    w_old = w.copy()
    w_new = w + learning_rate * true_label * point_augmented
    
    print(f"The point is misclassified! Updating weights:")
    print(f"w_new = w_old + η * y * x")
    print(f"w_new = {w_old} + {learning_rate} * {true_label} * {point_augmented}")
    print(f"w_new = {w_new}")
    
    # Plot updated decision boundary
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundaries
    if w[1] != 0:  # Original boundary
        x2_boundary = decision_boundary(x1_range, w)
        plt.plot(x1_range, x2_boundary, 'g-', label='Original Boundary')
    
    if w_new[1] != 0:  # Updated boundary
        x2_new_boundary = decision_boundary(x1_range, w_new)
        plt.plot(x1_range, x2_new_boundary, 'b--', label='Updated Boundary')
        
        # Shade positive and negative regions for updated boundary
        Z_new = w_new[0]*X1 + w_new[1]*X2 + w_new[2]
        plt.contourf(X1, X2, Z_new, levels=[-100, 0], colors=['lightblue'], alpha=0.2)
        plt.contourf(X1, X2, Z_new, levels=[0, 100], colors=['lightpink'], alpha=0.2)
        
    # Plot points
    for i, p in enumerate(points):
        pred, _ = predict(p, w_old)
        class_label = 1 if pred > 0 else -1
        marker_idx = 0 if class_label == 1 else 1
        plt.scatter(p[0], p[1], s=150, color=colors[marker_idx], 
                    marker=markers[marker_idx], edgecolor='black', linewidth=1.5,
                    label=f'Point {chr(65+i)} ({class_label})')
        
        # Add point label
        plt.annotate(f'({p[0]}, {p[1]})',
                     (p[0], p[1]),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    # Add labels and title
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Updated Perceptron Decision Boundary')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    
    # Add equations to the plot
    eq_old = f'Original: ${w_old[0]} x_1 + {w_old[1]} x_2 + {w_old[2]} = 0$'
    eq_new = f'Updated: ${w_new[0]} x_1 + {w_new[1]} x_2 + {w_new[2]} = 0$'
    plt.annotate(eq_old + '\n' + eq_new, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Save updated boundary figure
    plt.savefig(os.path.join(save_dir, 'perceptron_updated_boundary.png'), dpi=300, bbox_inches='tight')
else:
    print("The point is correctly classified! No weight update needed.")

# Add legend to original plot
plt.figure(1)
plt.legend()

# Save original figure
plt.savefig(os.path.join(save_dir, 'perceptron_decision_boundary.png'), dpi=300, bbox_inches='tight')

print(f"\nPlots saved to: {save_dir}") 