import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

# Create directory to save figures - using consistent directory structure
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_2_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define the dataset
data = {
    'Age': [15, 65, 30, 90, 44, 20, 50, 36],
    'Tumor_Size': [20, 30, 50, 20, 35, 70, 40, 25],
    'Malignant': [0, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
print("\n" + "="*80)
print("DATASET:")
print("="*80)
print(df)
print("\n")

# Extract features and target
X = df[['Age', 'Tumor_Size']].values
y = df['Malignant'].values
m = len(y)  # number of training examples

# Print details about the extracted data
print(f"Number of examples (m): {m}")
print(f"Feature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")
print(f"Benign examples (y=0): {np.sum(y == 0)}")
print(f"Malignant examples (y=1): {np.sum(y == 1)}")
print("\n")

# Add intercept term
X_with_intercept = np.c_[np.ones((m, 1)), X]
print("Feature matrix with intercept term:")
print(X_with_intercept)
print("\n")

# Define the sigmoid function
def sigmoid(z):
    """Compute the sigmoid function for input z"""
    return 1 / (1 + np.exp(-z))

print("="*80)
print("SIGMOID FUNCTION")
print("="*80)
print("The sigmoid function is defined as: g(z) = 1 / (1 + e^(-z))")
print("Properties of the sigmoid function:")
print("  - When z = 0: g(0) = 1/(1+e^0) = 1/2 = 0.5")
print("  - As z → +∞: g(z) → 1")
print("  - As z → -∞: g(z) → 0")
print("\n")

# Plot the sigmoid function
plt.figure(figsize=(10, 6))
z = np.linspace(-10, 10, 100)
sig = sigmoid(z)
plt.plot(z, sig, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('g(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'sigmoid_function.png'), dpi=300, bbox_inches='tight')
plt.close()

# Define the cost function
def compute_cost(X, y, theta):
    """
    Compute the logistic regression cost function
    
    Parameters:
    X (ndarray): Feature matrix with intercept (m x (n+1))
    y (ndarray): Target vector (m x 1)
    theta (ndarray): Parameter vector ((n+1) x 1)
    
    Returns:
    float: The cost value
    """
    m = len(y)
    h = sigmoid(X @ theta)
    # Use small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1-epsilon)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Define the gradient function
def compute_gradient(X, y, theta):
    """
    Compute the gradient of the logistic regression cost function
    
    Parameters:
    X (ndarray): Feature matrix with intercept (m x (n+1))
    y (ndarray): Target vector (m x 1)
    theta (ndarray): Parameter vector ((n+1) x 1)
    
    Returns:
    ndarray: The gradient vector ((n+1) x 1)
    """
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1/m) * X.T @ (h - y)
    return grad

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', label='Benign')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', label='Malignant')
plt.xlabel('Age (years)')
plt.ylabel('Tumor Size (mm)')
plt.title('Tumor Classification Dataset')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'dataset_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================================================
# Task 1: Calculate the initial cost with θ0 = 0, θ1 = 0, θ2 = 0
# ====================================================================================================
print("="*80)
print("TASK 1: INITIAL COST CALCULATION")
print("="*80)
initial_theta = np.zeros(3)
print(f"Initial parameters: θ = {initial_theta}")

# Step-by-step calculation of initial cost
print("\nStep-by-step calculation of initial cost J(θ):")
print("-"*60)

# Create a DataFrame to show the calculation similar to the image
calculation_df = pd.DataFrame({
    'x₁=age': df['Age'],
    'x₂=size': df['Tumor_Size'],
    'y': df['Malignant'],
})

# Calculate hypothesis values h(x) for each example
h_values = np.zeros(m)
for i in range(m):
    z_i = 0  # θᵀx = 0 for all examples when θ = [0,0,0]
    h_i = sigmoid(z_i)
    h_values[i] = h_i
    print(f"Example {i+1}: z_{i+1} = θᵀx_{i+1} = {initial_theta} @ {X_with_intercept[i]} = {z_i}")
    print(f"             h(x_{i+1}) = g(z_{i+1}) = g({z_i}) = {h_i}")

calculation_df['h(x)'] = h_values

# Calculate the y*log(h(x)) term for each example
y_log_h = np.zeros(m)
for i in range(m):
    if y[i] == 1:
        y_log_h[i] = y[i] * np.log(h_values[i])
        print(f"Example {i+1} (y_{i+1}=1): y*log(h(x)) = {y[i]}*log({h_values[i]:.4f}) = {y_log_h[i]:.5f}")
    else:
        y_log_h[i] = 0  # Will be 0 when y=0

calculation_df['y*log(h(x))'] = y_log_h
# Fill NaN values with empty string for better display
calculation_df['y*log(h(x))'] = calculation_df['y*log(h(x))'].apply(lambda x: f"{x:.5f}" if x != 0 else "")

# Calculate the (1-y)*log(1-h(x)) term for each example
one_minus_y_log_one_minus_h = np.zeros(m)
for i in range(m):
    if y[i] == 0:
        one_minus_y_log_one_minus_h[i] = (1-y[i]) * np.log(1-h_values[i])
        print(f"Example {i+1} (y_{i+1}=0): (1-y)*log(1-h(x)) = {1-y[i]}*log(1-{h_values[i]:.4f}) = {one_minus_y_log_one_minus_h[i]:.5f}")
    else:
        one_minus_y_log_one_minus_h[i] = 0  # Will be 0 when y=1

calculation_df['(1-y)*log(1-h(x))'] = one_minus_y_log_one_minus_h
# Fill NaN values with empty string for better display
calculation_df['(1-y)*log(1-h(x))'] = calculation_df['(1-y)*log(1-h(x))'].apply(lambda x: f"{x:.5f}" if x != 0 else "")

# Display the calculation table
print("\nCost function calculation table:")
print(calculation_df.to_string(index=False))

# Calculate the total cost (sum of all terms)
total_cost = np.sum(y_log_h) + np.sum(one_minus_y_log_one_minus_h)
print(f"\nSum of cost terms: {total_cost:.5f}")

# For this example we're using the sum directly to match the image showing J(θ) = -5.55
# rather than averaging by dividing by m
cost_j_theta = total_cost
print(f"Initial cost J(θ): {cost_j_theta:.2f}")

# For reference, show the traditional average cost as well
avg_cost = total_cost / m
print(f"Average cost (traditional calculation): {avg_cost:.5f}")

# Double-check with function but adjust to not divide by m to match the image
initial_cost = compute_cost(X_with_intercept, y, initial_theta) * m
print(f"\nVerification (total cost, not averaged): J(θ) = {initial_cost:.2f}")
print("\n")

# ====================================================================================================
# Task 2: Calculate the first two iterations of gradient descent
# ====================================================================================================
print("="*80)
print("TASK 2: GRADIENT DESCENT ITERATIONS")
print("="*80)
learning_rate = 0.01
num_iterations = 2
theta = initial_theta.copy()
cost_history = [initial_cost]
theta_history = [theta.copy()]

print(f"Learning rate α: {learning_rate}")
print(f"Initial parameters θ: {theta}")
print(f"Initial cost J(θ): {initial_cost:.4f}")

for i in range(num_iterations):
    print(f"\n{'='*30} Iteration {i+1} {'='*30}")
    
    # Step 1: Calculate the hypothesis values h(x) for each example
    print("\nStep 1: Calculate hypothesis values h(x)")
    print("-"*60)
    h_values = np.zeros(m)
    for j in range(m):
        z_j = X_with_intercept[j] @ theta
        h_j = sigmoid(z_j)
        h_values[j] = h_j
        print(f"Example {j+1}: z_{j+1} = θᵀx_{j+1} = {theta} @ {X_with_intercept[j]} = {z_j:.4f}")
        print(f"             h(x_{j+1}) = g(z_{j+1}) = g({z_j:.4f}) = {h_j:.4f}")
    
    # Step 2: Calculate the error (h(x) - y) for each example
    print("\nStep 2: Calculate errors (h(x) - y)")
    print("-"*60)
    errors = np.zeros(m)
    for j in range(m):
        error_j = h_values[j] - y[j]
        errors[j] = error_j
        print(f"Example {j+1}: error_{j+1} = h(x_{j+1}) - y_{j+1} = {h_values[j]:.4f} - {y[j]} = {error_j:.4f}")
    
    # Step 3: Calculate the gradients
    print("\nStep 3: Calculate gradients")
    print("-"*60)
    gradients = np.zeros(3)
    
    # Gradient for θ₀
    gradient_0 = 0
    for j in range(m):
        gradient_0 += errors[j] * X_with_intercept[j, 0]  # X_with_intercept[j, 0] is always 1
    gradient_0 /= m
    print(f"∂J/∂θ₀ = (1/{m}) * Σ[(h(x) - y) * x₀] = (1/{m}) * ({' + '.join([f'{errors[j]:.4f}*1' for j in range(m)])})")
    print(f"∂J/∂θ₀ = (1/{m}) * {np.sum(errors):.4f} = {gradient_0:.4f}")
    gradients[0] = gradient_0
    
    # Gradient for θ₁
    gradient_1 = 0
    for j in range(m):
        gradient_1 += errors[j] * X_with_intercept[j, 1]  # Age
    gradient_1 /= m
    print(f"∂J/∂θ₁ = (1/{m}) * Σ[(h(x) - y) * x₁] = (1/{m}) * ({' + '.join([f'{errors[j]:.4f}*{X_with_intercept[j, 1]}' for j in range(m)])})")
    print(f"∂J/∂θ₁ = (1/{m}) * {np.sum(errors * X_with_intercept[:, 1]):.4f} = {gradient_1:.4f}")
    gradients[1] = gradient_1
    
    # Gradient for θ₂
    gradient_2 = 0
    for j in range(m):
        gradient_2 += errors[j] * X_with_intercept[j, 2]  # Tumor Size
    gradient_2 /= m
    print(f"∂J/∂θ₂ = (1/{m}) * Σ[(h(x) - y) * x₂] = (1/{m}) * ({' + '.join([f'{errors[j]:.4f}*{X_with_intercept[j, 2]}' for j in range(m)])})")
    print(f"∂J/∂θ₂ = (1/{m}) * {np.sum(errors * X_with_intercept[:, 2]):.4f} = {gradient_2:.4f}")
    gradients[2] = gradient_2
    
    # Double-check with function
    func_gradients = compute_gradient(X_with_intercept, y, theta)
    print(f"\nVerification using gradient function: ∇J(θ) = {func_gradients}")
    
    # Step 4: Update parameters
    print("\nStep 4: Update parameters")
    print("-"*60)
    new_theta = np.zeros(3)
    for j in range(3):
        new_theta[j] = theta[j] - learning_rate * gradients[j]
        print(f"θ_{j} := θ_{j} - α * ∂J/∂θ_{j} = {theta[j]:.6f} - {learning_rate} * {gradients[j]:.6f} = {new_theta[j]:.6f}")
    
    theta = new_theta
    theta_history.append(theta.copy())
    
    # Step 5: Calculate new cost
    new_cost = compute_cost(X_with_intercept, y, theta)
    cost_history.append(new_cost)
    print(f"\nNew parameters: θ = {theta}")
    print(f"New cost: J(θ) = {new_cost:.6f}")
    print(f"Cost change: {new_cost - cost_history[-2]:.6f}")

print("\n")

# Plot cost over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history)), cost_history, 'b-', linewidth=2, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Cost Function over Gradient Descent Iterations')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'gradient_descent_cost.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================================================
# Task 3: Stochastic Gradient Descent (SGD)
# ====================================================================================================
print("="*80)
print("TASK 3: STOCHASTIC GRADIENT DESCENT")
print("="*80)
sgd_learning_rate = 0.1
sgd_iterations = 2
sgd_theta = initial_theta.copy()
sgd_cost_history = [initial_cost]
sgd_theta_history = [sgd_theta.copy()]

print(f"Learning rate α: {sgd_learning_rate}")
print(f"Initial parameters θ: {sgd_theta}")
print(f"Initial cost J(θ): {initial_cost:.4f}")

for i in range(sgd_iterations):
    print(f"\n{'='*30} Iteration {i+1} {'='*30}")
    
    # Step 1: Randomly select a training example
    np.random.seed(42 + i)  # Ensure reproducibility but different each iteration
    random_index = np.random.randint(0, m)
    xi = X_with_intercept[random_index:random_index+1]
    yi = y[random_index:random_index+1]
    
    print(f"Randomly selected example index: {random_index}")
    print(f"Selected example x: {xi[0]}")
    print(f"Selected example y: {yi[0]}")
    
    # Step 2: Calculate the hypothesis value for this example
    zi = xi @ sgd_theta
    hi = sigmoid(zi)
    print(f"\nStep 2: Calculate hypothesis value h(x)")
    print("-"*60)
    print(f"z = θᵀx = {sgd_theta} @ {xi[0]} = {zi[0]:.4f}")
    print(f"h(x) = g(z) = g({zi[0]:.4f}) = {hi[0]:.4f}")
    
    # Step 3: Calculate the error for this example
    error = hi - yi
    print(f"\nStep 3: Calculate error (h(x) - y)")
    print("-"*60)
    print(f"error = h(x) - y = {hi[0]:.4f} - {yi[0]} = {error[0]:.4f}")
    
    # Step 4: Calculate the gradient for this example
    print(f"\nStep 4: Calculate gradient")
    print("-"*60)
    sgd_grad = np.zeros(3)  # Initialize gradient vector
    
    for j in range(3):
        sgd_grad[j] = error[0] * xi[0, j]  # Calculate gradient for each parameter
        print(f"∂J/∂θ_{j} = error * x_{j} = {error[0]:.4f} * {xi[0, j]} = {sgd_grad[j]:.4f}")
    
    # Step 5: Update parameters
    print(f"\nStep 5: Update parameters")
    print("-"*60)
    new_sgd_theta = np.zeros(3)
    for j in range(3):
        new_sgd_theta[j] = sgd_theta[j] - sgd_learning_rate * sgd_grad[j]
        print(f"θ_{j} := θ_{j} - α * ∂J/∂θ_{j} = {sgd_theta[j]:.6f} - {sgd_learning_rate} * {sgd_grad[j]:.6f} = {new_sgd_theta[j]:.6f}")
    
    sgd_theta = new_sgd_theta
    sgd_theta_history.append(sgd_theta.copy())
    
    # Step 6: Calculate new cost on full dataset
    sgd_cost = compute_cost(X_with_intercept, y, sgd_theta)
    sgd_cost_history.append(sgd_cost)
    print(f"\nUpdated parameters: θ = {sgd_theta}")
    print(f"New cost (full dataset): J(θ) = {sgd_cost:.6f}")
    print(f"Cost change: {sgd_cost - sgd_cost_history[-2]:.6f}")

print("\n")

# ====================================================================================================
# Task 4: Decision Boundary Explanation
# ====================================================================================================
print("="*80)
print("TASK 4: DECISION BOUNDARY EXPLANATION")
print("="*80)
print("The decision boundary equation θᵀx = 0 represents the set of points where P(y=1|x) = 0.5.")
print("For the logistic regression model, P(y=1|x) = 1/(1+e^(-θᵀx)).")
print("When θᵀx = 0, P(y=1|x) = 1/(1+e^0) = 1/2 = 0.5.")
print("Geometrically, this creates a boundary in the feature space that separates the")
print("regions where the model predicts class 0 (below 0.5 probability) and class 1 (above 0.5 probability).")
print("For a model with two features, this boundary is a line. With more features, it becomes a hyperplane.")
print("\n")

# ====================================================================================================
# Task 5: Decision Boundary with Final Parameters
# ====================================================================================================
print("="*80)
print("TASK 5: DECISION BOUNDARY WITH FINAL PARAMETERS")
print("="*80)
final_theta = np.array([-136.95, 1.1, 2.2])
print(f"Final optimized parameters: θ = {final_theta}")

# Calculate the decision boundary equation: θ0 + θ1*x1 + θ2*x2 = 0
print("\nStep-by-step derivation of decision boundary equation:")
print("1. The decision boundary is defined by the equation: θ₀ + θ₁*Age + θ₂*Tumor_Size = 0")
print(f"2. Substituting our parameters: {final_theta[0]:.2f} + {final_theta[1]:.2f}*Age + {final_theta[2]:.2f}*Tumor_Size = 0")
print(f"3. Solving for Tumor_Size: {final_theta[2]:.2f}*Tumor_Size = -({final_theta[0]:.2f} + {final_theta[1]:.2f}*Age)")
print(f"4. Dividing both sides by {final_theta[2]:.2f}: Tumor_Size = -({final_theta[0]:.2f} + {final_theta[1]:.2f}*Age)/{final_theta[2]:.2f}")
print(f"5. Simplifying: Tumor_Size = {-final_theta[0]/final_theta[2]:.2f} - {final_theta[1]/final_theta[2]:.2f}*Age")
print("\n")

# Plot the decision boundary with final parameters
x1_min, x1_max = X[:, 0].min() - 10, X[:, 0].max() + 10
x2_min, x2_max = X[:, 1].min() - 10, X[:, 1].max() + 10

# Create a meshgrid for visualization
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
grid = np.c_[xx1.ravel(), xx2.ravel()]
grid_with_intercept = np.c_[np.ones(grid.shape[0]), grid]

# Calculate predictions on the grid
Z = sigmoid(grid_with_intercept @ final_theta)
Z = Z.reshape(xx1.shape)

# Plot the decision boundary and dataset
plt.figure(figsize=(12, 8))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(['blue', 'red']))
plt.contour(xx1, xx2, Z, [0.5], linewidths=2, colors='black')

# Plot the decision boundary as a line
x1_line = np.array([x1_min, x1_max])
x2_line = -(final_theta[0] + final_theta[1] * x1_line) / final_theta[2]
plt.plot(x1_line, x2_line, 'k--', linewidth=2)

# Add annotation for the decision boundary equation
plt.annotate(f'Tumor_Size = {-final_theta[0]/final_theta[2]:.2f} - {final_theta[1]/final_theta[2]:.2f}*Age',
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# Plot the data points
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', s=100, edgecolors='k', label='Benign')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', s=100, linewidth=2, label='Malignant')

# Mark the predicted point for Task 6
new_point = np.array([50, 30])
plt.scatter(new_point[0], new_point[1], c='green', marker='*', s=200, label='New Patient')

plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.xlabel('Age (years)')
plt.ylabel('Tumor Size (mm)')
plt.title('Logistic Regression Decision Boundary for Tumor Classification')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================================================
# Task 6: Predict for a new patient
# ====================================================================================================
print("="*80)
print("TASK 6: PREDICTION FOR NEW PATIENT")
print("="*80)
new_patient_age = 50
new_patient_tumor_size = 30
new_patient = np.array([1, new_patient_age, new_patient_tumor_size])  # Age=50, Tumor_Size=30 with intercept term
print(f"New patient data: [intercept, Age, Tumor_Size] = {new_patient}")

# Detailed calculation of the linear combination
print("\nStep-by-step prediction calculation:")
print("1. Compute the linear combination z = θᵀx:")
print(f"   z = θ₀ + θ₁ × Age + θ₂ × Tumor_Size")
print(f"   z = {final_theta[0]:.2f} + {final_theta[1]:.2f} × {new_patient_age} + {final_theta[2]:.2f} × {new_patient_tumor_size}")
print(f"   Substituting the values:")
print(f"   z = {final_theta[0]:.2f} + {final_theta[1]:.2f} × {new_patient_age} + {final_theta[2]:.2f} × {new_patient_tumor_size}")
print(f"   Term 1: θ₀ = {final_theta[0]:.2f}")
print(f"   Term 2: θ₁ × Age = {final_theta[1]:.2f} × {new_patient_age} = {final_theta[1]*new_patient_age:.2f}")
print(f"   Term 3: θ₂ × Tumor_Size = {final_theta[2]:.2f} × {new_patient_tumor_size} = {final_theta[2]*new_patient_tumor_size:.2f}")
print(f"   Sum of terms: z = {final_theta[0]:.2f} + {final_theta[1]*new_patient_age:.2f} + {final_theta[2]*new_patient_tumor_size:.2f} = {final_theta[0] + final_theta[1]*new_patient_age + final_theta[2]*new_patient_tumor_size:.2f}")
new_z = new_patient @ final_theta
print(f"   Final value: z = {new_z:.4f}")

# Decision boundary interpretation
print("\n1.1. Decision boundary interpretation:")
print(f"   The decision boundary is given by z = 0, which is where P(y=1|x) = 0.5")
print(f"   For our patient, z = {new_z:.4f} < 0, which means P(y=1|x) < 0.5")
print(f"   Given the patient's age ({new_patient_age}), the tumor size threshold for malignancy is:")
print(f"   Tumor_Size = 62.25 - 0.5 × {new_patient_age} = {62.25 - 0.5*new_patient_age:.2f} mm")
print(f"   Since the patient's tumor size ({new_patient_tumor_size} mm) is less than {62.25 - 0.5*new_patient_age:.2f} mm,")
print(f"   we predict the tumor is benign.")

print("\n2. Compute probability P(y=1|x) = g(z) using the sigmoid function:")
print(f"   P(y=1|x) = 1 / (1 + e^(-z))")
print(f"   P(y=1|x) = 1 / (1 + e^(-({new_z:.4f})))")

# Calculate the exponent term in detail
neg_z = -new_z
print(f"\n   Step 2.1: Compute e^(-z) = e^({neg_z:.4f}):")
print(f"   e^({neg_z:.4f}) is a very large number since {neg_z:.4f} is positive and large.")
print(f"   We can compute this as:")
# Using the properties of exponents for numerical stability
powers_of_ten = int(neg_z // 1)
remainder = neg_z - powers_of_ten
exp_remainder = np.exp(remainder)
exp_neg_z = np.exp(neg_z)

print(f"   e^({neg_z:.4f}) = e^({powers_of_ten} + {remainder:.4f}) = e^{powers_of_ten} × e^{remainder:.4f}")
print(f"   e^{powers_of_ten} ≈ {10**powers_of_ten:.6e}") 
print(f"   e^{remainder:.4f} ≈ {exp_remainder:.6f}")
print(f"   Therefore, e^({neg_z:.4f}) ≈ {10**powers_of_ten:.6e} × {exp_remainder:.6f} ≈ {exp_neg_z:.6e}")

# Show the full calculation
denominator = 1 + exp_neg_z
print(f"\n   Step 2.2: Compute the sigmoid function:")
print(f"   g(z) = 1 / (1 + e^(-z)) = 1 / (1 + {exp_neg_z:.6e}) = 1 / {denominator:.6e}")
print(f"   Since the denominator is very large, we can approximate: 1 / {denominator:.6e} ≈ {1/denominator:.10f}")
new_probability = 1 / denominator
print(f"   P(y=1|x) ≈ {new_probability:.10f}")

# Alternative notation to match the image
print("\n   Using the notation from the image:")
print(f"   h(x) = \\frac{{1}}{{1 + e^{{-({new_z:.2f})}}}} ≈ {new_probability:.10f} ≈ 0")

print("\n   We can also write this out more explicitly:")
print(f"   h(x) = \\frac{{1}}{{1 + e^{{-(-{final_theta[0]:.2f} + {final_theta[1]:.2f}*{new_patient_age} + {final_theta[2]:.2f}*{new_patient_tumor_size})}}}} ≈ {new_probability:.10f}")

print("\n3. Make classification decision:")
print(f"   Classification threshold = 0.5")
print(f"   Since P(y=1|x) = {new_probability:.10f} {'>' if new_probability > 0.5 else '<'} 0.5")
new_prediction = 1 if new_probability >= 0.5 else 0
print(f"   Classification: {'Malignant (y=1)' if new_prediction == 1 else 'Benign (y=0)'}")
print(f"   Confidence in prediction: {max(new_probability, 1-new_probability):.10f} = {max(new_probability, 1-new_probability)*100:.8f}%")
print("\n")

# Create a visual explanation of the prediction
plt.figure(figsize=(10, 6))
# Plot the sigmoid function
z_range = np.linspace(-10, 10, 1000)
sig_values = sigmoid(z_range)
plt.plot(z_range, sig_values, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
plt.axvline(x=0, color='g', linestyle='--', label='Decision Boundary (z=0)')

# Mark our calculated z value
plt.scatter([new_z], [sigmoid(new_z)], color='red', s=100, zorder=5, label=f'New Patient (z={new_z:.2f})')

plt.xlabel('z = θᵀx')
plt.ylabel('P(y=1|x) = g(z)')
plt.title('Sigmoid Function and New Patient Prediction')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_dir, 'new_patient_prediction.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a visualization of the logistic function over a range of values
plt.figure(figsize=(12, 5))

# Create a meshgrid of age and tumor size values
age_range = np.linspace(10, 100, 100)
tumor_range = np.linspace(10, 80, 100)
age_grid, tumor_grid = np.meshgrid(age_range, tumor_range)

# Calculate the probability for each point in the grid
z_grid = final_theta[0] + final_theta[1] * age_grid + final_theta[2] * tumor_grid
prob_grid = sigmoid(z_grid)

# Create a contour plot of probabilities
plt.subplot(1, 2, 1)
contour = plt.contourf(age_grid, tumor_grid, prob_grid, 20, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='P(malignant)')
plt.contour(age_grid, tumor_grid, prob_grid, levels=[0.5], colors='red', linestyles='dashed', linewidths=2)
plt.xlabel('Age (years)')
plt.ylabel('Tumor Size (mm)')
plt.title('Probability of Malignancy')
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', label='Benign', edgecolors='k')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', label='Malignant', s=80)
plt.scatter(new_patient_age, new_patient_tumor_size, c='green', marker='*', s=200, label='New Patient')
plt.legend(loc='upper right')
plt.grid(True)

# Add a 3D visualization
ax = plt.subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(age_grid, tumor_grid, prob_grid, cmap='viridis', alpha=0.8)
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='P(malignant)')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Tumor Size (mm)')
ax.set_zlabel('Probability')
ax.set_title('3D Probability Surface')
ax.view_init(30, 45)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'probability_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================================================
# Task 7: Interpretation of coefficients
# ====================================================================================================
print("="*80)
print("TASK 7: INTERPRETATION OF COEFFICIENTS")
print("="*80)
print(f"θ₁ (Age coefficient) = {final_theta[1]:.2f}")
print(f"θ₂ (Tumor Size coefficient) = {final_theta[2]:.2f}")

print("\nInterpretation in terms of log-odds:")
print(f"- For each additional year of age, the log-odds of malignancy increase by {final_theta[1]:.2f},")
print("  holding tumor size constant.")
print(f"- For each additional mm in tumor size, the log-odds of malignancy increase by {final_theta[2]:.2f},")
print("  holding age constant.")

print("\nInterpretation in terms of odds ratios:")
print(f"- The odds ratio for a 1-year increase in age is e^{final_theta[1]:.2f} = {np.exp(final_theta[1]):.2f}.")
print("  This means the odds of malignancy are multiplied by this factor for each year of age.")
print(f"- The odds ratio for a 1-mm increase in tumor size is e^{final_theta[2]:.2f} = {np.exp(final_theta[2]):.2f}.")
print("  This means the odds of malignancy are multiplied by this factor for each mm in tumor size.")

print("\nRelative importance:")
print(f"Since θ₂ ({final_theta[2]:.2f}) > θ₁ ({final_theta[1]:.2f}), tumor size has a stronger effect")
print(f"on the probability of malignancy than age. The effect of tumor size is approximately")
print(f"{final_theta[2]/final_theta[1]:.2f} times stronger than the effect of age.")
print("\n")

# Create a probability surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create mesh for feature space
age_range = np.linspace(10, 100, 50)
size_range = np.linspace(10, 80, 50)
age_grid, size_grid = np.meshgrid(age_range, size_range)
prob_grid = np.zeros_like(age_grid)

# Calculate probability at each point
for i in range(len(age_range)):
    for j in range(len(size_range)):
        x_point = np.array([1, age_grid[i, j], size_grid[i, j]])
        z = x_point @ final_theta
        prob_grid[i, j] = sigmoid(z)

# Plot probability surface
surf = ax.plot_surface(age_grid, size_grid, prob_grid, cmap='coolwarm', alpha=0.8)

# Add the training data points
for i in range(m):
    if y[i] == 0:  # Benign
        ax.scatter(X[i, 0], X[i, 1], 0, c='blue', marker='o', s=100, label='Benign' if i == 0 else "")
    else:  # Malignant
        ax.scatter(X[i, 0], X[i, 1], 1, c='red', marker='x', s=100, label='Malignant' if i == 2 else "")

# Add the decision boundary (0.5 probability contour)
ax.contour(age_grid, size_grid, prob_grid, [0.5], colors='k', linestyles='dashed')

# Add the new patient point
ax.scatter(50, 30, new_probability, c='green', marker='*', s=200, label='New Patient')

ax.set_xlabel('Age (years)')
ax.set_ylabel('Tumor Size (mm)')
ax.set_zlabel('Probability of Malignancy')
ax.set_title('Probability Surface for Tumor Classification')
ax.legend()
plt.savefig(os.path.join(save_dir, 'probability_surface.png'), dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================================================
# Task 8: Effect of learning rate
# ====================================================================================================
print("="*80)
print("TASK 8: EFFECT OF LEARNING RATE")
print("="*80)
print("Increasing the learning rate:")
print("- Advantages:")
print("  * Faster convergence if the rate is well-tuned")
print("  * Requires fewer iterations to reach the optimum")
print("  * Can escape local minima more easily")
print("- Disadvantages:")
print("  * Risk of overshooting the minimum and divergence if too large")
print("  * May oscillate around the minimum without reaching it")
print("  * Can cause numerical instability")
print("\n")

print("Decreasing the learning rate:")
print("- Advantages:")
print("  * More stable and reliable convergence")
print("  * Less sensitive to noise in the data")
print("  * Better precision near the optimum")
print("- Disadvantages:")
print("  * Slower progress, requiring more iterations")
print("  * May get stuck in local minima or plateau regions")
print("  * Very small rates may make progress imperceptibly slow")
print("\n")

# Visualize the effect of different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
num_iterations = 20
plt.figure(figsize=(12, 8))

for lr in learning_rates:
    theta = initial_theta.copy()
    cost_history = [compute_cost(X_with_intercept, y, theta)]
    
    for i in range(num_iterations):
        gradients = compute_gradient(X_with_intercept, y, theta)
        theta = theta - lr * gradients
        cost_history.append(compute_cost(X_with_intercept, y, theta))
    
    plt.plot(range(len(cost_history)), cost_history, marker='o', linewidth=2, label=f'α = {lr}')

plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Effect of Learning Rate on Convergence')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'learning_rate_effect.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a 3D visualization of the cost function
# Create a simplified version of the cost function surface (for θ1 and θ2, fixing θ0)
theta0_fixed = -10  # Fixed value for demonstration
theta1_range = np.linspace(-2, 2, 50)
theta2_range = np.linspace(-2, 2, 50)
theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
cost_grid = np.zeros_like(theta1_grid)

for i in range(len(theta1_range)):
    for j in range(len(theta2_range)):
        theta_test = np.array([theta0_fixed, theta1_grid[i, j], theta2_grid[i, j]])
        cost_grid[i, j] = compute_cost(X_with_intercept, y, theta_test)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta1_grid, theta2_grid, cost_grid, cmap='viridis', alpha=0.8)

ax.set_xlabel('θ₁ (Age coefficient)')
ax.set_ylabel('θ₂ (Tumor Size coefficient)')
ax.set_zlabel('Cost J(θ)')
ax.set_title('Logistic Regression Cost Function Surface (θ₀ fixed)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(os.path.join(save_dir, 'cost_function_surface.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations and calculations completed. Images saved to:", save_dir) 