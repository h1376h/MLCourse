import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
# Enable math text rendering with appropriate font
plt.rcParams.update({
    'text.usetex': False,  # Using matplotlib's math rendering instead of full LaTeX
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans'
})

print("Question 28: Fruit Classification with Perceptron and Fisher's LDA")
print("=================================================================")

# Given data
fruit_data = {
    'SA1': {'Sweetness': 8, 'Sourness': 2, 'Class': 1, 'Label': 'SweetApple'},
    'SA2': {'Sweetness': 7, 'Sourness': 3, 'Class': 1, 'Label': 'SweetApple'},
    'SO1': {'Sweetness': 3, 'Sourness': 8, 'Class': -1, 'Label': 'SourOrange'},
    'SO2': {'Sweetness': 2, 'Sourness': 7, 'Class': -1, 'Label': 'SourOrange'}
}

# Convert to numpy arrays for easier processing
X = np.array([[fruit_data[f]['Sweetness'], fruit_data[f]['Sourness']] for f in fruit_data])
y = np.array([fruit_data[f]['Class'] for f in fruit_data])
fruit_ids = list(fruit_data.keys())

# Print the data in a table
print("\nFruit Data:")
print("----------")
print(f"{'Fruit ID':<10} {'Sweetness':<12} {'Sourness':<12} {'Class':<10}")
print("-" * 50)
for fruit_id, info in fruit_data.items():
    print(f"{fruit_id:<10} {info['Sweetness']:<12} {info['Sourness']:<12} {info['Class']:<10}")

print("\nStep 1: Sketch the points in a 2D coordinate system")
print("--------------------------------------------------")

plt.figure(figsize=(10, 8))
# Plot SweetApples (Class +1)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', s=150, edgecolor='black', label='SweetApple (+1)')

# Plot SourOranges (Class -1)
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='orange', marker='s', s=150, edgecolor='black', label='SourOrange (-1)')

# Label each point with its ID
for i, fruit_id in enumerate(fruit_ids):
    plt.annotate(fruit_id, (X[i, 0] + 0.2, X[i, 1] + 0.2), fontsize=12)

plt.xlabel('Sweetness ($x_1$)', fontsize=14)
plt.ylabel('Sourness ($x_2$)', fontsize=14)
plt.title('Fruit Classification: SweetApples vs SourOranges', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(0, 10)
plt.ylim(0, 10)

# Save the figure
plt.savefig(os.path.join(save_dir, "fruit_data_plot.png"), dpi=300, bbox_inches='tight')
print("Created visualization of the fruit data points.")

print("\nStep 2: Draw a linear decision boundary")
print("-------------------------------------")

print("Looking at the data points in the 2D feature space:")
print("  SweetApples (Class +1): (8,2), (7,3)")
print("  SourOranges (Class -1): (3,8), (2,7)")
print("\nWe observe that SweetApples have higher sweetness than sourness,")
print("while SourOranges have higher sourness than sweetness.")
print("This suggests a decision boundary of the form: $x_1 - x_2 = k$")

# Extract the coordinates for better readability
sweet_apples_x1 = X[y == 1, 0]  # Sweetness of SweetApples
sweet_apples_x2 = X[y == 1, 1]  # Sourness of SweetApples
sour_oranges_x1 = X[y == -1, 0]  # Sweetness of SourOranges
sour_oranges_x2 = X[y == -1, 1]  # Sourness of SourOranges

# Calculate x₁ - x₂ for each point to find a suitable k
print("\nCalculating $x_1 - x_2$ for each point to determine a suitable k:")
for i, fruit_id in enumerate(fruit_ids):
    diff = X[i, 0] - X[i, 1]  # x₁ - x₂
    print(f"  {fruit_id}: $x_1 - x_2$ = {X[i, 0]} - {X[i, 1]} = {diff}")

# Find the midpoint between the maximum difference for negative class and minimum difference for positive class
min_diff_positive = min(sweet_apples_x1 - sweet_apples_x2)
max_diff_negative = max(sour_oranges_x1 - sour_oranges_x2)

print(f"\nMinimum difference for SweetApples (positive class): {min_diff_positive}")
print(f"Maximum difference for SourOranges (negative class): {max_diff_negative}")

# Calculate the midpoint as the optimal k
k = (min_diff_positive + max_diff_negative) / 2
print(f"Optimal value for k is the midpoint: k = (${min_diff_positive}$ + ${max_diff_negative}$) / 2 = ${k}$")

# Convert the boundary equation x₁ - x₂ = k to standard form w₁x₁ + w₂x₂ + b = 0
w1 = 1
w2 = -1
b = -k

print(f"\nConverting $x_1 - x_2 = ${k}$ to standard form $w_1x_1 + w_2x_2 + b = 0$:")
print(f"  $x_1 - x_2 = ${k}$")
print(f"  $x_1 - x_2 - ${k}$ = 0")
print(f"  ${w1}x_1 + {w2}x_2 + {b} = 0$")

# Let's check if this boundary correctly separates the points
print("\nVerifying the decision boundary:")
for fruit_id, info in fruit_data.items():
    x1, x2 = info['Sweetness'], info['Sourness']
    decision_value = w1*x1 + w2*x2 + b
    predicted_class = 1 if decision_value > 0 else -1
    correct = predicted_class == info['Class']
    print(f"  {fruit_id}: ${w1}*{x1} + {w2}*{x2} + {b} = {decision_value:.2f} -> Predicted: {predicted_class}, Actual: {info['Class']}, Correct: {correct}")

# Plot the decision boundary
plt.figure(figsize=(10, 8))

# Plot the points as before
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', s=150, edgecolor='black', label='SweetApple (+1)')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='orange', marker='s', s=150, edgecolor='black', label='SourOrange (-1)')

# Label each point with its ID
for i, fruit_id in enumerate(fruit_ids):
    plt.annotate(fruit_id, (X[i, 0] + 0.2, X[i, 1] + 0.2), fontsize=12)

# Create a meshgrid to visualize the decision boundary
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = w1*xx + w2*yy + b
plt.contour(xx, yy, Z, [0], colors='blue', linewidths=2)

# Plot the equation of the boundary line
boundary_text = f"Decision Boundary: ${w1}x_1 + {w2}x_2 + {b:.2f} = 0$"
plt.text(1, 9, boundary_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('Sweetness ($x_1$)', fontsize=14)
plt.ylabel('Sourness ($x_2$)', fontsize=14)
plt.title('Fruit Classification with Linear Decision Boundary', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add shaded regions for the classes
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['orange', 'red']))

# Save the figure
plt.savefig(os.path.join(save_dir, "decision_boundary.png"), dpi=300, bbox_inches='tight')
print("Created visualization of the decision boundary.")

print("\nStep 3: Apply Perceptron algorithm")
print("--------------------------------")

# Initialize Perceptron weights
w1_init, w2_init, b_init = 0, 0, 0
learning_rate = 1

print(f"Initialize Perceptron with weights: w₁ = ${w1_init}$, w₂ = ${w2_init}$, b = ${b_init}$")
print(f"Learning rate η = ${learning_rate}$")
print("The Perceptron algorithm follows these steps:")
print("1. Initialize weights w₁ = 0, w₂ = 0, b = 0")
print("2. For each training example ($x_1$, $x_2$, $y$):")
print("   a. Calculate the decision function: $f(x) = w_1x_1 + w_2x_2 + b$")
print("   b. Determine the predicted class: ŷ = sign(f(x))")
print("   c. Check for misclassification: if y·f(x) ≤ 0")
print("   d. Update weights if misclassified:")
print("      $w_1 = w_1 + η·y·x_1$")
print("      $w_2 = w_2 + η·y·x_2$")
print("      $b = b + η·y$")
print("3. Repeat until convergence or maximum iterations")

# Process points in the given order: SA1, SA2, SO1, SO2
processing_order = ['SA1', 'SA2', 'SO1', 'SO2']
print("\nWe will process the points in the order:", ", ".join(processing_order))

# Initialize vectors for clearer calculation
w_init = np.array([w1_init, w2_init])
b_init = b_init

# Detailed calculation for the first point (SA1)
first_point = processing_order[0]
x = np.array([fruit_data[first_point]['Sweetness'], fruit_data[first_point]['Sourness']])
y_val = fruit_data[first_point]['Class']

print(f"\nDetailed calculation for the first point (${first_point}$):")
print(f"  Input values: x = [$x_1 = {x[0]}$], $x_2 = {x[1]}$], y = ${y_val}$")

# Step 1: Calculate the decision function
print(f"  Step 1: Calculate the decision function f(x) = w·x + b")
print(f"    w·x = [$w_1 = {w_init[0]}$ $w_2 = {w_init[1]}$]·[$x_1 = {x[0]}$ $x_2 = {x[1]}$] = ${w_init[0]}·{x[0]} + {w_init[1]}·{x[1]} = {np.dot(w_init, x)}$")
decision = np.dot(w_init, x) + b_init
print(f"    f(x) = w·x + b = ${np.dot(w_init, x)}$ + ${b_init} = ${decision}$")

# Step 2: Determine the predicted class
print(f"  Step 2: Determine the predicted class ŷ = sign(f(x))")
if decision > 0:
    predicted_class = 1
    print(f"    sign(${decision}$) = +1")
elif decision < 0:
    predicted_class = -1
    print(f"    sign(${decision}$) = -1")
else:
    predicted_class = 0
    print(f"    sign(${decision}$) = 0 (ambiguous, treated as a misclassification)")

# Step 3: Check if an update is needed
print(f"  Step 3: Check if an update is needed by calculating y·f(x)")
update_condition = y_val * decision
print(f"    y·f(x) = ${y_val}·{decision} = ${update_condition}$")

# Step 4: Update weights if needed
print(f"  Step 4: Update weights if y·f(x) ≤ 0")
if update_condition <= 0:
    # Calculate the new weights
    w_new = w_init + learning_rate * y_val * x
    b_new = b_init + learning_rate * y_val
    
    print("    Since y·f(x) ≤ 0, we update the weights:")
    print(f"    $w_1 = w_1 + η·y·x_1 = {w_init[0]} + {learning_rate}·{y_val}·{x[0]} = {w_init[0] + learning_rate * y_val * x[0]}$")
    print(f"    $w_2 = w_2 + η·y·x_2 = {w_init[1]} + {learning_rate}·{y_val}·{x[1]} = {w_init[1] + learning_rate * y_val * x[1]}$")
    print(f"    $b = b + η·y = {b_init} + {learning_rate}·{y_val} = {b_init + learning_rate * y_val}$")
    
    # Update the current weights
    w_init = w_new
    b_init = b_new
else:
    print(f"    Since y·f(x) = ${update_condition} > 0$, no update is needed.")
    print(f"    w remains [$w_1 = {w_init[0]}$ $w_2 = {w_init[1]}$], b remains ${b_init}$")

# Show the results of the update
print(f"\nAfter processing the first point, the weights are:")
print(f"  w = [$w_1 = {w_init[0]}$ $w_2 = {w_init[1]}$], b = ${b_init}$")
print(f"  The corresponding decision boundary equation is: ${w_init[0]}x_1 + {w_init[1]}x_2 + {b_init} = 0$")

# Visualize the perceptron's decision boundary after the first update
plt.figure(figsize=(10, 8))

# Plot the points
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', s=150, edgecolor='black', label='SweetApple (+1)')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='orange', marker='s', s=150, edgecolor='black', label='SourOrange (-1)')

# Label each point with its ID
for i, fruit_id in enumerate(fruit_ids):
    plt.annotate(fruit_id, (X[i, 0] + 0.2, X[i, 1] + 0.2), fontsize=12)

# Highlight the point that was processed
first_point_idx = fruit_ids.index(first_point)
plt.scatter(X[first_point_idx, 0], X[first_point_idx, 1], color='none', edgecolor='green', s=250, linewidth=3)
plt.annotate("Processed", (X[first_point_idx, 0], X[first_point_idx, 1] + 0.5), fontsize=12, color='green', weight='bold')

# Plot the perceptron's decision boundary after the first update
if w_init[0] != 0 or w_init[1] != 0:  # Only plot if we have a non-zero weight vector
    Z_perceptron = w_init[0] * xx + w_init[1] * yy + b_init
    plt.contour(xx, yy, Z_perceptron, [0], colors='green', linewidths=2, linestyles='--')
    
    # Add shaded regions
    plt.contourf(xx, yy, Z_perceptron, alpha=0.1, cmap=ListedColormap(['orange', 'red']))
    
    # Plot the equation of the boundary line
    perceptron_text = f"Perceptron Boundary: ${w_init[0]}x_1 + {w_init[1]}x_2 + {b_init} = 0$"
    plt.text(1, 8.5, perceptron_text, fontsize=14, color='green', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

plt.xlabel('Sweetness ($x_1$)', fontsize=14)
plt.ylabel('Sourness ($x_2$)', fontsize=14)
plt.title('Perceptron Decision Boundary After First Update', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Save the figure
plt.savefig(os.path.join(save_dir, "perceptron_first_update.png"), dpi=300, bbox_inches='tight')
print("Created visualization of the perceptron's decision boundary after the first update.")

print("\nStep 4: Determine if the dataset is linearly separable")
print("----------------------------------------------------")

# We can determine if the dataset is linearly separable by inspecting the plot
# or by checking if we can find a hyperplane that separates the classes
print("Looking at the data points in the 2D space:")
print("  SweetApples (Class +1): (8,2), (7,3)")
print("  SourOranges (Class -1): (3,8), (2,7)")
print("\nObservation: The two classes can be separated by a straight line.")
print("Therefore, this dataset is linearly separable.")

print("\nMathematical verification:")
print(f"We found a decision boundary: $x_1 - x_2 = ${k}$ that correctly separates all points.")
print("Since we found a linear decision boundary that perfectly separates the classes, the dataset is linearly separable.")

print("\nStep 5: Apply Fisher's Linear Discriminant Analysis (LDA)")
print("---------------------------------------------------------")
print("Fisher's LDA seeks the optimal projection direction w that maximizes")
print("the ratio of between-class to within-class scatter: J(w) = (w^T S_B w) / (w^T S_W w)")

# Extract features for each class
X_sweet_apples = X[y == 1]
X_sour_oranges = X[y == -1]

# Calculate mean vectors for each class
mean_sweet_apples = np.mean(X_sweet_apples, axis=0)
mean_sour_oranges = np.mean(X_sour_oranges, axis=0)

print("\nStep 5.1: Calculate mean vectors for each class")
print("  SweetApples data points:")
for i, point in enumerate(X_sweet_apples):
    print(f"    Point {i+1}: [$x_1 = {point[0]}$], $x_2 = {point[1]}$")
print(f"  Calculate the mean: μ₁ = (1/n₁)·Σx_i = (1/2)·([$x_1 = {X_sweet_apples[0][0]}$], $x_2 = {X_sweet_apples[0][1]}$] + [$x_1 = {X_sweet_apples[1][0]}$], $x_2 = {X_sweet_apples[1][1]}$])")
print(f"  Mean for SweetApples: μ₁ = [$x_1 = {mean_sweet_apples[0]}$], $x_2 = {mean_sweet_apples[1]}$]")

print("\n  SourOranges data points:")
for i, point in enumerate(X_sour_oranges):
    print(f"    Point {i+1}: [$x_1 = {point[0]}$], $x_2 = {point[1]}$")
print(f"  Calculate the mean: μ₂ = (1/n₂)·Σx_i = (1/2)·([$x_1 = {X_sour_oranges[0][0]}$], $x_2 = {X_sour_oranges[0][1]}$] + [$x_1 = {X_sour_oranges[1][0]}$], $x_2 = {X_sour_oranges[1][1]}$])")
print(f"  Mean for SourOranges: μ₂ = [$x_1 = {mean_sour_oranges[0]}$], $x_2 = {mean_sour_oranges[1]}$]")

print("\nStep 5.2: Calculate within-class scatter matrices (S_W)")
print("  For each class, we compute the scatter matrix S = Σ(x - μ)(x - μ)^T")

# Initialize scatter matrices for detailed calculation
S_w_sweet_apples = np.zeros((2, 2))
print("  For SweetApples:")
for i, x in enumerate(X_sweet_apples):
    diff = x - mean_sweet_apples
    print(f"    Point {i+1}: x - μ₁ = [$x_1 = {x[0]}$ - $x_1 = {mean_sweet_apples[0]}$ = {diff[0]:.2f}$], $x_2 = {x[1]} - x_2 = {mean_sweet_apples[1]}$ = {diff[1]:.2f}$]")
    
    # Detailed outer product calculation
    print(f"    (x - μ₁)(x - μ₁)^T = [$x_1 = {diff[0]:.2f}$ $x_2 = {diff[1]:.2f}$]^T · [$x_1 = {diff[0]:.2f}$ $x_2 = {diff[1]:.2f}$]")
    outer = np.outer(diff, diff)
    print(f"    = [($x_1 = {diff[0]:.2f}$)^2 {diff[0]:.2f}·{diff[1]:.2f}; {diff[0]:.2f}·{diff[1]:.2f} ($x_2 = {diff[1]:.2f}$)^2]")
    print(f"    = [$x_1 = {outer[0,0]:.4f}$ $x_2 = {outer[0,1]:.4f}$ $x_2 = {outer[1,0]:.4f}$ $x_2 = {outer[1,1]:.4f}$]")
    
    S_w_sweet_apples += outer

print(f"  S_W(sweet apples) = [$x_1 = {S_w_sweet_apples[0,0]:.4f}$ $x_2 = {S_w_sweet_apples[0,1]:.4f}$ $x_2 = {S_w_sweet_apples[1,0]:.4f}$ $x_2 = {S_w_sweet_apples[1,1]:.4f}$]")

S_w_sour_oranges = np.zeros((2, 2))
print("\n  For SourOranges:")
for i, x in enumerate(X_sour_oranges):
    diff = x - mean_sour_oranges
    print(f"    Point {i+1}: x - μ₂ = [$x_1 = {x[0]}$ - $x_1 = {mean_sour_oranges[0]}$ = {diff[0]:.2f}$], $x_2 = {x[1]} - x_2 = {mean_sour_oranges[1]}$ = {diff[1]:.2f}$]")
    
    # Detailed outer product calculation
    print(f"    (x - μ₂)(x - μ₂)^T = [$x_1 = {diff[0]:.2f}$ $x_2 = {diff[1]:.2f}$]^T · [$x_1 = {diff[0]:.2f}$ $x_2 = {diff[1]:.2f}$]")
    outer = np.outer(diff, diff)
    print(f"    = [($x_1 = {diff[0]:.2f}$)^2 {diff[0]:.2f}·{diff[1]:.2f}; {diff[0]:.2f}·{diff[1]:.2f} ($x_2 = {diff[1]:.2f}$)^2]")
    print(f"    = [$x_1 = {outer[0,0]:.4f}$ $x_2 = {outer[0,1]:.4f}$ $x_2 = {outer[1,0]:.4f}$ $x_2 = {outer[1,1]:.4f}$]")
    
    S_w_sour_oranges += outer

print(f"  S_W(sour oranges) = [$x_1 = {S_w_sour_oranges[0,0]:.4f}$ $x_2 = {S_w_sour_oranges[0,1]:.4f}$ $x_2 = {S_w_sour_oranges[1,0]:.4f}$ $x_2 = {S_w_sour_oranges[1,1]:.4f}$]")

# Calculate total within-class scatter matrix
S_w = S_w_sweet_apples + S_w_sour_oranges
print("\n  Total within-class scatter matrix:")
print(f"  S_W = S_W(sweet apples) + S_W(sour oranges)")
print(f"  S_W = [$x_1 = {S_w_sweet_apples[0,0]:.4f}$ $x_2 = {S_w_sweet_apples[0,1]:.4f}$ $x_2 = {S_w_sweet_apples[1,0]:.4f}$ $x_2 = {S_w_sweet_apples[1,1]:.4f}$] + [$x_1 = {S_w_sour_oranges[0,0]:.4f}$ $x_2 = {S_w_sour_oranges[0,1]:.4f}$ $x_2 = {S_w_sour_oranges[1,0]:.4f}$ $x_2 = {S_w_sour_oranges[1,1]:.4f}$]")
print(f"  S_W = [$x_1 = {S_w[0,0]:.4f}$ $x_2 = {S_w[0,1]:.4f}$ $x_2 = {S_w[1,0]:.4f}$ $x_2 = {S_w[1,1]:.4f}$]")

print("\nStep 5.3: Calculate between-class scatter matrix (S_B)")
print("  The between-class scatter matrix S_B = (μ₁ - μ₂)(μ₁ - μ₂)^T")

# Calculate between-class scatter matrix with detailed steps
mean_diff = mean_sweet_apples - mean_sour_oranges
print(f"  μ₁ - μ₂ = [$x_1 = {mean_sweet_apples[0]}$ - $x_1 = {mean_sour_oranges[0]}$ = {mean_diff[0]}$], $x_2 = {mean_sweet_apples[1]} - x_2 = {mean_sour_oranges[1]}$ = {mean_diff[1]}$]")

print(f"  S_B = (μ₁ - μ₂)(μ₁ - μ₂)^T = [$x_1 = {mean_diff[0]}$ $x_2 = {mean_diff[1]}$]^T · [$x_1 = {mean_diff[0]}$ $x_2 = {mean_diff[1]}$]")
print(f"  S_B = [$x_1 = {mean_diff[0]}^2 {mean_diff[0]}·{mean_diff[1]}; {mean_diff[0]}·{mean_diff[1]} {mean_diff[1]}^2$]")
S_b = np.outer(mean_diff, mean_diff)
print(f"  S_B = [$x_1 = {S_b[0,0]}$ $x_2 = {S_b[0,1]}$ $x_2 = {S_b[1,0]}$ $x_2 = {S_b[1,1]}$]")

print("\nStep 5.4: Calculate the optimal projection direction")
print("  The optimal projection direction w is given by w = S_W^(-1)(μ₁ - μ₂)")

# Check if S_w is invertible for detailed calculations
try:
    S_w_inv = np.linalg.inv(S_w)
    print(f"  First, we find S_W^(-1) = [$x_1 = {S_w_inv[0,0]:.4f}$ $x_2 = {S_w_inv[0,1]:.4f}$ $x_2 = {S_w_inv[1,0]:.4f}$ $x_2 = {S_w_inv[1,1]:.4f}$]")
    
    # Detailed matrix multiplication step by step
    print(f"  Then we compute w = S_W^(-1)(μ₁ - μ₂):")
    print(f"  w = [$x_1 = {S_w_inv[0,0]:.4f}$ $x_2 = {S_w_inv[0,1]:.4f}$ $x_2 = {S_w_inv[1,0]:.4f}$ $x_2 = {S_w_inv[1,1]:.4f}$] · [$x_1 = {mean_diff[0]}$ $x_2 = {mean_diff[1]}$]")
    
    # First row calculation
    w1 = S_w_inv[0,0] * mean_diff[0] + S_w_inv[0,1] * mean_diff[1]
    print(f"  w[0] = $x_1 = {S_w_inv[0,0]:.4f}$ · $x_1 = {mean_diff[0]}$ + $x_2 = {S_w_inv[0,1]:.4f}$ · $x_2 = {mean_diff[1]}$ = $x_1 = {w1:.4f}$")
    
    # Second row calculation
    w2 = S_w_inv[1,0] * mean_diff[0] + S_w_inv[1,1] * mean_diff[1]
    print(f"  w[1] = $x_1 = {S_w_inv[1,0]:.4f}$ · $x_1 = {mean_diff[0]}$ + $x_2 = {S_w_inv[1,1]:.4f}$ · $x_2 = {mean_diff[1]}$ = $x_2 = {w2:.4f}$")
    
    w_fisher = np.array([w1, w2])
    print(f"  w = [$x_1 = {w_fisher[0]:.4f}$ $x_2 = {w_fisher[1]:.4f}$]")
    
    # Normalize to unit length with detailed steps
    w_fisher_norm = np.linalg.norm(w_fisher)
    print(f"  Normalize w to unit length:")
    print(f"  ||w|| = √($x_1 = {w_fisher[0]:.4f}$^2 + $x_2 = {w_fisher[1]:.4f}$^2) = √{w_fisher[0]**2 + w_fisher[1]**2:.4f} = $x_1 = {w_fisher_norm:.4f}$")
    
    w_fisher_normalized = w_fisher / w_fisher_norm
    print(f"  w_normalized = w / ||w|| = [$x_1 = {w_fisher[0]:.4f}$ $x_2 = {w_fisher[1]:.4f}$] / $x_1 = {w_fisher_norm:.4f}$ = [$x_1 = {w_fisher_normalized[0]:.4f}$ $x_2 = {w_fisher_normalized[1]:.4f}$]")
    
except np.linalg.LinAlgError:
    print("  Warning: The within-class scatter matrix is singular (not invertible).")
    print("  This can happen with small datasets. We'll use regularization.")

# Alternative approach with covariance matrices
print("\nStep 5.5: Alternative approach using covariance matrices")
print("  For small datasets, we can calculate covariance matrices directly:")

# Calculate class covariance matrices with more detail
cov_sweet_apples = np.cov(X_sweet_apples.T)
cov_sour_oranges = np.cov(X_sour_oranges.T)

print(f"  Covariance matrix for SweetApples:")
print(f"  Σ₁ = [$x_1 = {cov_sweet_apples[0,0]:.4f}$ $x_2 = {cov_sweet_apples[0,1]:.4f}$ $x_2 = {cov_sweet_apples[1,0]:.4f}$ $x_2 = {cov_sweet_apples[1,1]:.4f}$]")

print(f"  Covariance matrix for SourOranges:")
print(f"  Σ₂ = [$x_1 = {cov_sour_oranges[0,0]:.4f}$ $x_2 = {cov_sour_oranges[0,1]:.4f}$ $x_2 = {cov_sour_oranges[1,0]:.4f}$ $x_2 = {cov_sour_oranges[1,1]:.4f}$]")

# Calculate pooled covariance matrix with steps
print("\n  Calculate pooled covariance matrix (assuming equal class sizes):")
print(f"  Σ_pooled = (Σ₁ + Σ₂) / 2")
print(f"  Σ_pooled = [$x_1 = {cov_sweet_apples[0,0]:.4f}$ $x_2 = {cov_sweet_apples[0,1]:.4f}$ $x_2 = {cov_sweet_apples[1,0]:.4f}$ $x_2 = {cov_sweet_apples[1,1]:.4f}$] + [$x_1 = {cov_sour_oranges[0,0]:.4f}$ $x_2 = {cov_sour_oranges[0,1]:.4f}$ $x_2 = {cov_sour_oranges[1,0]:.4f}$ $x_2 = {cov_sour_oranges[1,1]:.4f}$] / 2")

cov_pooled = (cov_sweet_apples + cov_sour_oranges) / 2
print(f"  Σ_pooled = [$x_1 = {cov_pooled[0,0]:.4f}$ $x_2 = {cov_pooled[0,1]:.4f}$ $x_2 = {cov_pooled[1,0]:.4f}$ $x_2 = {cov_pooled[1,1]:.4f}$]")

# Compute the inverse of the pooled covariance matrix with steps
print("\n  Calculate the inverse of the pooled covariance matrix:")
try:
    cov_pooled_inv = np.linalg.inv(cov_pooled)
    print(f"  Σ_pooled^(-1) = [$x_1 = {cov_pooled_inv[0,0]:.4f}$ $x_2 = {cov_pooled_inv[0,1]:.4f}$ $x_2 = {cov_pooled_inv[1,0]:.4f}$ $x_2 = {cov_pooled_inv[1,1]:.4f}$]")
    
    # Calculate LDA direction with detailed steps
    print("\n  Calculate the LDA direction w = Σ_pooled^(-1)(μ₁ - μ₂):")
    print(f"  w = [$x_1 = {cov_pooled_inv[0,0]:.4f}$ $x_2 = {cov_pooled_inv[0,1]:.4f}$ $x_2 = {cov_pooled_inv[1,0]:.4f}$ $x_2 = {cov_pooled_inv[1,1]:.4f}$] · [$x_1 = {mean_diff[0]}$ $x_2 = {mean_diff[1]}$]")
    
    # First component
    w_direct_1 = cov_pooled_inv[0,0] * mean_diff[0] + cov_pooled_inv[0,1] * mean_diff[1]
    print(f"  w[0] = $x_1 = {cov_pooled_inv[0,0]:.4f}$ · $x_1 = {mean_diff[0]}$ + $x_2 = {cov_pooled_inv[0,1]:.4f}$ · $x_2 = {mean_diff[1]}$ = $x_1 = {w_direct_1:.4f}$")
    
    # Second component
    w_direct_2 = cov_pooled_inv[1,0] * mean_diff[0] + cov_pooled_inv[1,1] * mean_diff[1]
    print(f"  w[1] = $x_1 = {cov_pooled_inv[1,0]:.4f}$ · $x_1 = {mean_diff[0]}$ + $x_2 = {cov_pooled_inv[1,1]:.4f}$ · $x_2 = {mean_diff[1]}$ = $x_2 = {w_direct_2:.4f}$")
    
    w_direct = np.array([w_direct_1, w_direct_2])
    print(f"  w = [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$]")
    
    # Normalize the vector with detailed steps
    w_direct_norm = np.linalg.norm(w_direct)
    print(f"\n  Normalize the vector to unit length:")
    print(f"  ||w|| = √($x_1 = {w_direct[0]:.4f}$^2 + $x_2 = {w_direct[1]:.4f}$^2) = √{w_direct[0]**2 + w_direct[1]**2:.4f} = $x_1 = {w_direct_norm:.4f}$")
    
    w_direct = w_direct / w_direct_norm
    print(f"  w_normalized = w / ||w|| = [$x_1 = {w_direct_1:.4f}$ $x_2 = {w_direct_2:.4f}$] / $x_1 = {w_direct_norm:.4f}$ = [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$]")
    
except np.linalg.LinAlgError:
    print("  The pooled covariance matrix is singular. Using regularization.")
    epsilon = 1e-5
    cov_pooled_reg = cov_pooled + epsilon * np.eye(2)
    cov_pooled_inv_reg = np.linalg.inv(cov_pooled_reg)
    
    w_direct = cov_pooled_inv_reg @ mean_diff
    w_direct = w_direct / np.linalg.norm(w_direct)
    
    print(f"  Regularized LDA direction: w = [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$]")

# Calculate the LDA decision boundary with detailed steps
print("\nStep 5.6: Calculate the LDA decision boundary")
print("  The decision boundary is perpendicular to w and passes through the midpoint")
print("  between the class means.")

# Calculate the midpoint
center = (mean_sweet_apples + mean_sour_oranges) / 2
print(f"  Midpoint between class means = (μ₁ + μ₂) / 2")
print(f"  = [$x_1 = {mean_sweet_apples[0]}$ + $x_1 = {mean_sour_oranges[0]}$ = {center[0]}$], $x_2 = {mean_sweet_apples[1]} + x_2 = {mean_sour_oranges[1]} = {center[1]}$] / 2")
print(f"  = [$x_1 = {center[0]}$ $x_2 = {center[1]}$]")

# For a line perpendicular to w passing through the midpoint
print(f"  For a line perpendicular to w = [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$] passing through [$x_1 = {center[0]}$ $x_2 = {center[1]}$],")
print(f"  the equation is: w⊥·(x - x₀) = 0, where w⊥ = [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$] and x₀ = [$x_1 = {center[0]}$ $x_2 = {center[1]}$]")
print(f"  This becomes: $x_1 = {w_direct[0]:.4f}(x_1 - {center[0]}) + x_2 = {w_direct[1]:.4f}(x_2 - {center[1]}) = 0$")
print(f"  Simplifying: $x_1 = {w_direct[0]:.4f}x_1 + x_2 = {w_direct[1]:.4f}x_2 - {w_direct[0]:.4f}·{center[0]} - x_2 = {w_direct[1]:.4f}·{center[1]} = 0$")

# Calculate the intercept term
intercept = w_direct[0] * center[0] + w_direct[1] * center[1]
print(f"  Further simplifying: $x_1 = {w_direct[0]:.4f}x_1 + x_2 = {w_direct[1]:.4f}x_2 - {intercept:.4f} = 0$")

# Visualize the LDA projection
plt.figure(figsize=(10, 8))

# Plot the points
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='o', s=150, edgecolor='black', label='SweetApple (+1)')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='orange', marker='s', s=150, edgecolor='black', label='SourOrange (-1)')

# Label each point with its ID
for i, fruit_id in enumerate(fruit_ids):
    plt.annotate(fruit_id, (X[i, 0] + 0.2, X[i, 1] + 0.2), fontsize=12)

# Plot the class means
plt.scatter(mean_sweet_apples[0], mean_sweet_apples[1], color='red', marker='*', s=300, edgecolor='black', label='Mean SweetApple')
plt.scatter(mean_sour_oranges[0], mean_sour_oranges[1], color='orange', marker='*', s=300, edgecolor='black', label='Mean SourOrange')

# Plot the LDA direction from the center of the data
center = (mean_sweet_apples + mean_sour_oranges) / 2
plt.arrow(center[0], center[1], w_direct[0]*2, w_direct[1]*2, 
          head_width=0.3, head_length=0.5, fc='blue', ec='blue', linewidth=2, label='LDA Direction')

# Plot a line perpendicular to the LDA direction (decision boundary)
# The perpendicular line passes through the midpoint between the projected means
# The equation of the perpendicular line: w_direct[1] * (x - center[0]) - w_direct[0] * (y - center[1]) = 0
# Simplifying: w_direct[1] * x - w_direct[0] * y = w_direct[1] * center[0] - w_direct[0] * center[1]
print("\nStep 5.6: Calculate the LDA decision boundary")
print(f"  The decision boundary is perpendicular to the LDA direction and passes")
print(f"  through the midpoint between the class means.")

print(f"  Midpoint = (μ₁ + μ₂) / 2 = [$x_1 = {mean_sweet_apples[0]:.2f}$ + $x_1 = {mean_sour_oranges[0]:.2f}$ = {center[0]:.2f}$], $x_2 = {mean_sweet_apples[1]:.2f} + x_2 = {mean_sour_oranges[1]:.2f}$ = {center[1]:.2f}$] / 2 = [$x_1 = {center[0]:.2f}$ $x_2 = {center[1]:.2f}$]")

print(f"  For a line perpendicular to vector [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$] passing through [$x_1 = {center[0]:.2f}$ $x_2 = {center[1]:.2f}$]:")
print(f"  The equation is: $x_2 = {w_direct[1]:.4f}(x_1 - {center[0]:.2f}) - $x_1 = {w_direct[0]:.4f}(x_2 - {center[1]:.2f}) = 0$")

# Get two points on the perpendicular line
k = w_direct[1] * center[0] - w_direct[0] * center[1]
if abs(w_direct[0]) > 1e-10:  # Avoid division by zero
    y1, y2 = 0, 10
    x1 = (k + w_direct[0] * y1) / w_direct[1]
    x2 = (k + w_direct[0] * y2) / w_direct[1]
    print(f"  Simplified: $x_2 = {w_direct[1]:.4f}x_1 - $x_1 = {w_direct[0]:.4f}x_2 = {k:.4f}$")
    print(f"  For x_2 = 0: x_1 = {k:.4f}/{w_direct[1]:.4f} = {x1:.4f}$")
    print(f"  For x_2 = 10: x_1 = ($k = {k:.4f} + $x_1 = {w_direct[0]:.4f} · 10)/$x_2 = {w_direct[1]:.4f}$ = {x2:.4f}$")
else:
    x1, x2 = 0, 10
    y1 = (k + w_direct[1] * x1) / w_direct[0]
    y2 = (k + w_direct[1] * x2) / w_direct[0]
    print(f"  Simplified: $x_2 = {w_direct[1]:.4f}x_1 - $x_1 = {w_direct[0]:.4f}x_2 = {k:.4f}$")
    print(f"  For x_1 = 0: x_2 = {k:.4f}/{w_direct[0]:.4f} = {y1:.4f}$")
    print(f"  For x_1 = 10: x_2 = ($k = {k:.4f} + $x_2 = {w_direct[1]:.4f} · 10)/$x_1 = {w_direct[0]:.4f}$ = {y2:.4f}$")

plt.plot([x1, x2], [y1, y2], 'b--', linewidth=2, label='LDA Decision Boundary')

# Add the LDA equation
lda_eq = f"LDA Direction: [{w_direct[0]:.4f}, {w_direct[1]:.4f}]"
plt.text(1, 9, lda_eq, fontsize=14, color='blue', bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('Sweetness ($x_1$)', fontsize=14)
plt.ylabel('Sourness ($x_2$)', fontsize=14)
plt.title("Fisher's Linear Discriminant Analysis (LDA)", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')
print("Created visualization of the LDA projection.")

# Create a 1D projection visualization
plt.figure(figsize=(12, 6))

# Project the data points onto the LDA direction
X_proj = np.dot(X, w_direct)
X_proj_sweet = X_proj[y == 1]
X_proj_sour = X_proj[y == -1]

# Calculate the projected means
mean_proj_sweet = np.mean(X_proj_sweet)
mean_proj_sour = np.mean(X_proj_sour)

# Calculate the threshold (midpoint between projected means)
threshold_proj = (mean_proj_sweet + mean_proj_sour) / 2

# Create a number line
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Plot the projected points
plt.scatter(X_proj_sweet, np.zeros_like(X_proj_sweet) + 0.1, color='red', marker='o', s=150, edgecolor='black', label='SweetApple (+1)')
plt.scatter(X_proj_sour, np.zeros_like(X_proj_sour) - 0.1, color='orange', marker='s', s=150, edgecolor='black', label='SourOrange (-1)')

# Label each projected point
for i, fruit_id in enumerate(fruit_ids):
    y_offset = 0.2 if y[i] == 1 else -0.2
    plt.annotate(fruit_id, (X_proj[i], y_offset), fontsize=12, ha='center')

# Plot the projected means
plt.scatter(mean_proj_sweet, 0.1, color='red', marker='*', s=300, edgecolor='black', label='Mean SweetApple (proj)')
plt.scatter(mean_proj_sour, -0.1, color='orange', marker='*', s=300, edgecolor='black', label='Mean SourOrange (proj)')

# Plot the threshold
plt.axvline(x=threshold_proj, color='blue', linestyle='--', linewidth=2, label='Decision Threshold')

# Add labels and annotation
plt.text(threshold_proj + 0.1, 0.3, f"Threshold: $x_1 = {threshold_proj:.4f}$", fontsize=12, color='blue')
plt.text(mean_proj_sweet, 0.3, f"Mean: $x_1 = {mean_proj_sweet:.4f}$", fontsize=12, color='red')
plt.text(mean_proj_sour, -0.3, f"Mean: $x_1 = {mean_proj_sour:.4f}$", fontsize=12, color='orange')

plt.xlabel('Projection onto LDA Direction', fontsize=14)
plt.title('1D Projection of Data Points Using LDA', fontsize=16)
plt.grid(True, alpha=0.3, axis='x')
plt.legend(fontsize=12)
plt.ylim(-0.5, 0.5)

# Save the figure
plt.savefig(os.path.join(save_dir, "lda_1d_projection.png"), dpi=300, bbox_inches='tight')
print("Created visualization of the 1D LDA projection.")

print("\nSummary:")
print("---------")
print("1. We plotted the fruit data points in a 2D coordinate system.")
print(f"2. We found a linear decision boundary: $x_1 - x_2 = ${k}$")
print(f"3. After the first update of the Perceptron algorithm, the weights are: w₁ = ${w1_init}$, w₂ = ${w2_init}$, b = ${b_init}$")
print("4. The dataset is linearly separable, as we can find a straight line that perfectly separates the classes.")
print(f"5. The optimal projection direction for Fisher's LDA is: [$x_1 = {w_direct[0]:.4f}$ $x_2 = {w_direct[1]:.4f}$]") 