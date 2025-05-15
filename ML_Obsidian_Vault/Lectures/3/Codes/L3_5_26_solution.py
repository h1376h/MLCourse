import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# Set up matplotlib for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_26")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Given data from the problem
time_indices = np.array([1, 2, 3, 4, 5])
time_of_day = np.array([0.25, 0.50, 0.75, 0.25, 0.50])  # x1
day_of_week = np.array([0.14, 0.14, 0.14, 0.28, 0.28])  # x2
temperature = np.array([0.6, 0.7, 0.5, 0.6, 0.8])       # x3
consumption = np.array([15, 22, 18, 14, 25])            # y (target)

# Create a dataframe to display the data
data = pd.DataFrame({
    'Time index': time_indices,
    'Time of day ($x_1$)': time_of_day,
    'Day of week ($x_2$)': day_of_week,
    'Temperature ($x_3$)': temperature,
    'Consumption ($y$)': consumption
})

print("Original data:")
print(data)
print()

# Initial weight vector
w_initial = np.array([0, 0, 0, 0])
print(f"Initial weight vector w^(0): {w_initial}")
print()

# Learning rate
alpha = 0.1
print(f"Learning rate α: {alpha}")
print()

# Create the design matrix X (with a column of ones for the intercept)
X = np.column_stack((np.ones(len(time_of_day)), time_of_day, day_of_week, temperature))
y = consumption

print("Design matrix X (with intercept column):")
print(X)
print()

print("Target vector y:")
print(y)
print()

# Function to make prediction
def predict(w, x):
    """Calculate prediction using w^T * x"""
    return np.dot(w, x)

# Function to update weights using LMS rule
def update_weights(w, x, y, alpha):
    """
    Update weights using LMS rule:
    w^(t+1) = w^(t) + α(y - w^(t)^T * x) * x
    """
    prediction = predict(w, x)
    error = y - prediction
    w_new = w + alpha * error * x
    return w_new, prediction, error

# Function to calculate the cost function (MSE)
def calculate_cost(w, X, y):
    """Calculate the mean squared error for the given weights"""
    predictions = np.dot(X, w)
    errors = y - predictions
    return np.mean(errors**2)

#################################
# Detailed pen-and-paper style calculations
#################################

print("DETAILED PEN-AND-PAPER STYLE CALCULATIONS")
print("=========================================")
print()

print("MATHEMATICAL FORMULATION:")
print("-----------------------")
print("Linear model: ŷ = w₀ + w₁x₁ + w₂x₂ + w₃x₃")
print("In vector form: ŷ = w^T * x")
print("Cost function: J(w) = (1/2) * E[(y - ŷ)²]")
print("LMS update rule: w^(t+1) = w^(t) + α(y - w^(t)^T * x) * x")
print()

# Step 1: Calculate prediction for the first data point
x1 = X[0]  # First data point
y1 = y[0]  # First target value

# Detailed calculation for first prediction
print("STEP 1: Calculate prediction for the first data point")
print("-------------------------------------------------")
print(f"Data point 1: x₁ = {x1[1]}, x₂ = {x1[2]}, x₃ = {x1[3]}, y = {y1}")
print("Initial weights: w₀ = 0, w₁ = 0, w₂ = 0, w₃ = 0")
print()

print("Prediction calculation:")
print(f"ŷ = w₀ + w₁·x₁ + w₂·x₂ + w₃·x₃")
print(f"ŷ = {w_initial[0]} + {w_initial[1]}·{x1[1]} + {w_initial[2]}·{x1[2]} + {w_initial[3]}·{x1[3]}")
print(f"ŷ = {w_initial[0]} + {w_initial[1] * x1[1]} + {w_initial[2] * x1[2]} + {w_initial[3] * x1[3]}")
prediction1 = predict(w_initial, x1)
print(f"ŷ = {prediction1}")
print()

# Step 2: Update weights after processing the first data point
print("STEP 2: Update weights after processing the first data point")
print("----------------------------------------------------------")
print("LMS update rule: w^(t+1) = w^(t) + α(y - ŷ) * x")
print()

print("Calculate error:")
error1 = y1 - prediction1
print(f"error = y - ŷ = {y1} - {prediction1} = {error1}")
print()

print("Calculate weight updates:")
w1 = np.zeros(4)  # Initialize w1 to store updated weights

# Calculate and display each weight update component
print("For weight w₀:")
w1[0] = w_initial[0] + alpha * error1 * x1[0]
print(f"w₀^(1) = w₀^(0) + α·error·x₀")
print(f"w₀^(1) = {w_initial[0]} + {alpha}·{error1}·{x1[0]}")
print(f"w₀^(1) = {w_initial[0]} + {alpha * error1 * x1[0]}")
print(f"w₀^(1) = {w1[0]}")
print()

print("For weight w₁:")
w1[1] = w_initial[1] + alpha * error1 * x1[1]
print(f"w₁^(1) = w₁^(0) + α·error·x₁")
print(f"w₁^(1) = {w_initial[1]} + {alpha}·{error1}·{x1[1]}")
print(f"w₁^(1) = {w_initial[1]} + {alpha * error1 * x1[1]}")
print(f"w₁^(1) = {w1[1]}")
print()

print("For weight w₂:")
w1[2] = w_initial[2] + alpha * error1 * x1[2]
print(f"w₂^(1) = w₂^(0) + α·error·x₂")
print(f"w₂^(1) = {w_initial[2]} + {alpha}·{error1}·{x1[2]}")
print(f"w₂^(1) = {w_initial[2]} + {alpha * error1 * x1[2]}")
print(f"w₂^(1) = {w1[2]}")
print()

print("For weight w₃:")
w1[3] = w_initial[3] + alpha * error1 * x1[3]
print(f"w₃^(1) = w₃^(0) + α·error·x₃")
print(f"w₃^(1) = {w_initial[3]} + {alpha}·{error1}·{x1[3]}")
print(f"w₃^(1) = {w_initial[3]} + {alpha * error1 * x1[3]}")
print(f"w₃^(1) = {w1[3]}")
print()

print("Updated weight vector:")
print(f"w^(1) = [{w1[0]}, {w1[1]}, {w1[2]}, {w1[3]}]")
print()

# Verify with the update_weights function
w1_func, pred1, error1_func = update_weights(w_initial, x1, y1, alpha)
assert np.allclose(w1, w1_func), "Manual calculation differs from function result"

# Step 3: Calculate prediction for the second data point using updated weights
x2 = X[1]  # Second data point
y2 = y[1]  # Second target value

print("STEP 3: Calculate prediction for the second data point")
print("----------------------------------------------------")
print(f"Data point 2: x₁ = {x2[1]}, x₂ = {x2[2]}, x₃ = {x2[3]}, y = {y2}")
print(f"Updated weights: w₀ = {w1[0]}, w₁ = {w1[1]}, w₂ = {w1[2]}, w₃ = {w1[3]}")
print()

print("Prediction calculation:")
print(f"ŷ = w₀ + w₁·x₁ + w₂·x₂ + w₃·x₃")
print(f"ŷ = {w1[0]} + {w1[1]}·{x2[1]} + {w1[2]}·{x2[2]} + {w1[3]}·{x2[3]}")
print(f"ŷ = {w1[0]} + {w1[1] * x2[1]} + {w1[2] * x2[2]} + {w1[3] * x2[3]}")
prediction2 = predict(w1, x2)
print(f"ŷ = {prediction2}")
print()

# Step 4: Update weights after processing the second data point
print("STEP 4: Update weights after processing the second data point")
print("-----------------------------------------------------------")
print("LMS update rule: w^(t+1) = w^(t) + α(y - ŷ) * x")
print()

print("Calculate error:")
error2 = y2 - prediction2
print(f"error = y - ŷ = {y2} - {prediction2} = {error2}")
print()

print("Calculate weight updates:")
w2 = np.zeros(4)  # Initialize w2 to store updated weights

# Calculate and display each weight update component
print("For weight w₀:")
w2[0] = w1[0] + alpha * error2 * x2[0]
print(f"w₀^(2) = w₀^(1) + α·error·x₀")
print(f"w₀^(2) = {w1[0]} + {alpha}·{error2}·{x2[0]}")
print(f"w₀^(2) = {w1[0]} + {alpha * error2 * x2[0]}")
print(f"w₀^(2) = {w2[0]}")
print()

print("For weight w₁:")
w2[1] = w1[1] + alpha * error2 * x2[1]
print(f"w₁^(2) = w₁^(1) + α·error·x₁")
print(f"w₁^(2) = {w1[1]} + {alpha}·{error2}·{x2[1]}")
print(f"w₁^(2) = {w1[1]} + {alpha * error2 * x2[1]}")
print(f"w₁^(2) = {w2[1]}")
print()

print("For weight w₂:")
w2[2] = w1[2] + alpha * error2 * x2[2]
print(f"w₂^(2) = w₂^(1) + α·error·x₂")
print(f"w₂^(2) = {w1[2]} + {alpha}·{error2}·{x2[2]}")
print(f"w₂^(2) = {w1[2]} + {alpha * error2 * x2[2]}")
print(f"w₂^(2) = {w2[2]}")
print()

print("For weight w₃:")
w2[3] = w1[3] + alpha * error2 * x2[3]
print(f"w₃^(2) = w₃^(1) + α·error·x₃")
print(f"w₃^(2) = {w1[3]} + {alpha}·{error2}·{x2[3]}")
print(f"w₃^(2) = {w1[3]} + {alpha * error2 * x2[3]}")
print(f"w₃^(2) = {w2[3]}")
print()

print("Updated weight vector:")
print(f"w^(2) = [{w2[0]}, {w2[1]}, {w2[2]}, {w2[3]}]")
print()

# Verify with the update_weights function
w2_func, pred2, error2_func = update_weights(w1, x2, y2, alpha)
assert np.allclose(w2, w2_func), "Manual calculation differs from function result"

# Step 5: Calculate gradient of the cost function
print("STEP 5: Calculate gradient of the cost function")
print("---------------------------------------------")
print("Gradient of the cost function: ∇J(w) = -E[(y - w^T·x)·x]")
print("For a single observation: ∇J(w) = -(y - w^T·x)·x = -(error)·x")
print()

print("Gradient for the first observation:")
gradient1 = -error1 * x1
print(f"∇J(w^(0)) = -(error)·x = -({error1})·{x1} = {gradient1}")
print()

print("Gradient for the second observation:")
gradient2 = -error2 * x2
print(f"∇J(w^(1)) = -(error)·x = -({error2})·{x2} = {gradient2}")
print()

print("Note: The LMS update is proportional to negative gradient:")
print("w^(t+1) = w^(t) - α·∇J(w^(t))")
print("w^(t+1) = w^(t) + α·error·x")
print()

# End of detailed calculations
print("End of detailed calculations")
print("=========================================")
print()

#################################
# Original calculation steps
#################################

# Store weights, predictions, and errors for each iteration
weights_history = [w_initial]
predictions_history = []
errors_history = []

# Step 1: Calculate prediction for the first data point
prediction1 = predict(w_initial, x1)
print(f"Step 1: Prediction for the first data point using w^(0):")
print(f"x_1 = {x1}")
print(f"y_1 = {y1}")
print(f"prediction = w^(0)^T * x_1 = {' + '.join([f'{w:.1f} * {x:.2f}' for w, x in zip(w_initial, x1)])} = {prediction1}")
print()

# Step 2: Update weights after processing the first data point
w1_func, pred1, error1_func = update_weights(w_initial, x1, y1, alpha)
weights_history.append(w1_func)
predictions_history.append(pred1)
errors_history.append(error1_func)

print(f"Step 2: Update weights after processing the first data point:")
print(f"Error = y_1 - prediction = {y1} - {pred1} = {error1_func}")
print(f"w^(1) = w^(0) + α * error * x_1")
print(f"w^(1) = {w_initial} + {alpha} * {error1_func} * {x1}")
print(f"w^(1) = {w_initial} + {[alpha * error1_func * xi for xi in x1]}")
print(f"w^(1) = {w1_func}")
print()

# Step 3: Calculate prediction for the second data point using updated weights
prediction2 = predict(w1_func, x2)
print(f"Step 3: Prediction for the second data point using w^(1):")
print(f"x_2 = {x2}")
print(f"y_2 = {y2}")
print(f"prediction = w^(1)^T * x_2 = {' + '.join([f'{w:.1f} * {x:.2f}' for w, x in zip(w1_func, x2)])} = {prediction2}")
print()

# Step 4: Update weights after processing the second data point
w2_func, pred2, error2_func = update_weights(w1_func, x2, y2, alpha)
weights_history.append(w2_func)
predictions_history.append(pred2)
errors_history.append(error2_func)

print(f"Step 4: Update weights after processing the second data point:")
print(f"Error = y_2 - prediction = {y2} - {pred2} = {error2_func}")
print(f"w^(2) = w^(1) + α * error * x_2")
print(f"w^(2) = {w1_func} + {alpha} * {error2_func} * {x2}")
print(f"w^(2) = {w1_func} + {[alpha * error2_func * xi for xi in x2]}")
print(f"w^(2) = {w2_func}")
print()

# Step 5: Analyze feature importance based on the first two updates
print("Step 5: Analyze feature importance based on the first two updates")
print("Let's look at the absolute weight changes for each feature:")
initial_weights = np.array([0, 0, 0, 0])
weight_changes = np.abs(w2_func - initial_weights)
feature_names = ['Intercept', 'Time of day ($x_1$)', 'Day of week ($x_2$)', 'Temperature ($x_3$)']
latex_feature_names = ['Intercept', 'Time of day ($x_1$)', 'Day of week ($x_2$)', 'Temperature ($x_3$)']

# Create a table of weight changes
weight_change_data = pd.DataFrame({
    'Feature': feature_names,
    'Initial Weight (w^(0))': initial_weights,
    'Weight after first update (w^(1))': w1_func,
    'Weight after second update (w^(2))': w2_func,
    'Absolute Change |w^(2) - w^(0)|': weight_changes
})

print(weight_change_data)
print()

# Determine the feature with the largest weight change
max_change_idx = np.argmax(weight_changes)
print(f"The feature with the largest weight change is: {feature_names[max_change_idx]}")
print(f"This suggests that {feature_names[max_change_idx]} has the strongest influence on electricity consumption so far.")
print()

# Continue the LMS algorithm for all data points (for visualization purposes)
w_current = w2_func
for i in range(2, len(X)):
    xi = X[i]
    yi = y[i]
    w_current, pred_i, error_i = update_weights(w_current, xi, yi, alpha)
    weights_history.append(w_current)
    predictions_history.append(pred_i)
    errors_history.append(error_i)

# Convert weights history to numpy array for easier slicing
weights_history = np.array(weights_history)

# Visualization 1: Weight evolution over iterations
plt.figure(figsize=(10, 6))
for i, feat_name in enumerate(latex_feature_names):
    plt.plot(range(len(weights_history)), weights_history[:, i], marker='o', label=feat_name)

plt.xlabel('Iteration')
plt.ylabel('Weight Value')
plt.title('Evolution of Weights During LMS Algorithm Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'weight_evolution.png'), dpi=300)
plt.close()

# Visualization 2: Actual vs Predicted values
# Make predictions using the weights at each iteration
all_predictions = []
for i in range(len(X)):
    if i == 0:
        all_predictions.append(0)  # Initial prediction is 0
    else:
        all_predictions.append(predict(weights_history[i], X[i]))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(X)+1), y, 'bo-', label='Actual Consumption')
plt.plot(range(1, len(X)+1), all_predictions, 'ro--', label='Predicted Consumption')
plt.xlabel('Time Index')
plt.ylabel('Electricity Consumption (kWh)')
plt.title('Actual vs Predicted Electricity Consumption')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'), dpi=300)
plt.close()

# Visualization 3: Bar chart of final weights
final_weights = weights_history[-1]
plt.figure(figsize=(10, 6))
bars = plt.bar(latex_feature_names, final_weights, color='skyblue')

# Add values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Features')
plt.ylabel('Weight Value')
plt.title('Final Feature Weights After LMS Training')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'final_weights.png'), dpi=300)
plt.close()

# Visualization 4: Feature importance (based on absolute weight values)
plt.figure(figsize=(10, 6))
abs_weights = np.abs(final_weights)
sorted_idx = np.argsort(abs_weights)
bars = plt.barh([latex_feature_names[i] for i in sorted_idx], abs_weights[sorted_idx], color='lightgreen')

# Add values to the right of the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
             f'{width:.2f}', ha='left', va='center')

plt.xlabel('Absolute Weight Value')
plt.ylabel('Features')
plt.title('Feature Importance Based on Absolute Weight Values')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
plt.close()

# Visualization 5: Error reduction during training
# Calculate MSE after each weight update
mse_values = []
for i in range(len(weights_history)):
    w = weights_history[i]
    errors = []
    for j in range(len(X)):
        pred = predict(w, X[j])
        error = y[j] - pred
        errors.append(error**2)
    mse = np.mean(errors)
    mse_values.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(range(len(mse_values)), mse_values, 'g-o', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve: Error Reduction During Training')
plt.grid(True)
plt.annotate(f'Initial MSE: {mse_values[0]:.2f}', 
             xy=(0, mse_values[0]), 
             xytext=(0.5, mse_values[0]+50),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
plt.annotate(f'Final MSE: {mse_values[-1]:.2f}', 
             xy=(len(mse_values)-1, mse_values[-1]), 
             xytext=(len(mse_values)-1.5, mse_values[-1]+50),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'learning_curve.png'), dpi=300)
plt.close()

# NEW VISUALIZATION 6: 3D Plot showing gradient descent for a simplified model
print("Creating 3D visualization of gradient descent...")

# We'll create a simplified 3D visualization using just 2 weights
# (we need to simplify because the full model has 4 weights which would require 5D visualization)
# We'll use intercept and temperature (the most important features) for this visualization

# Create a meshgrid for visualizing the cost function
w0_range = np.linspace(-5, 10, 100)
w3_range = np.linspace(-2, 8, 100)
W0, W3 = np.meshgrid(w0_range, w3_range)

# Calculate cost function for each weight combination (fixing other weights at their final values)
Z = np.zeros_like(W0)
for i in range(len(w0_range)):
    for j in range(len(w3_range)):
        # Create weight vector with fixed values for w1 and w2
        w_test = np.array([w0_range[i], final_weights[1], final_weights[2], w3_range[j]])
        Z[j, i] = calculate_cost(w_test, X, y)

# Plot the cost function surface
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(W0, W3, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

# Track the path of gradient descent for w0 and w3 specifically
w0_path = weights_history[:, 0]
w3_path = weights_history[:, 3]
cost_path = [calculate_cost(w, X, y) for w in weights_history]

# Plot the path taken by gradient descent algorithm
ax.plot(w0_path, w3_path, cost_path, 'r-o', linewidth=2, markersize=5, label='Gradient Descent Path')

# Mark the start and end points
ax.scatter(w0_path[0], w3_path[0], cost_path[0], color='r', s=100, label='Start')
ax.scatter(w0_path[-1], w3_path[-1], cost_path[-1], color='g', s=100, label='End')

# Fix the annotations for initial and final points to avoid LaTeX subscript issues
ax.text(w0_path[0], w3_path[0], cost_path[0], f'Initial weights\nw0={w0_path[0]:.2f}, w3={w3_path[0]:.2f}\nMSE={cost_path[0]:.2f}', size=9, color='r')
ax.text(w0_path[-1], w3_path[-1], cost_path[-1], f'Final weights\nw0={w0_path[-1]:.2f}, w3={w3_path[-1]:.2f}\nMSE={cost_path[-1]:.2f}', size=9, color='g')

# Customize the plot
ax.set_xlabel('Weight $w_0$ (Intercept)')
ax.set_ylabel('Weight $w_3$ (Temperature)')
ax.set_zlabel('Cost (MSE)')
ax.set_title('3D Visualization of Gradient Descent Path\n(simplified to show only Intercept and Temperature weights)')
ax.view_init(elev=30, azim=120)  # Set viewing angle
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='Cost Value')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'gradient_descent_3d.png'), dpi=300)
plt.close()

print("3D visualization created successfully.")
print()

# Summary
print("Summary:")
print(f"1. Initial weights: w^(0) = {w_initial}")
print(f"2. Prediction for the first data point: {prediction1}")
print(f"3. Updated weights after first data point: w^(1) = {w1_func}")
print(f"4. Prediction for the second data point: {prediction2}")
print(f"5. Updated weights after second data point: w^(2) = {w2_func}")
print(f"6. Feature with strongest influence: {feature_names[max_change_idx]}")
print(f"7. Initial MSE: {mse_values[0]:.2f}, Final MSE: {mse_values[-1]:.2f}")
print()

print(f"Visualizations saved to: {save_dir}")
print("Generated images:")
print("- weight_evolution.png: Shows how each weight changes during training")
print("- actual_vs_predicted.png: Compares actual and predicted consumption values")
print("- final_weights.png: Bar chart of final weight values")
print("- feature_importance.png: Ranks features by importance (absolute weight value)")
print("- learning_curve.png: Shows how the mean squared error decreases during training")
print("- gradient_descent_3d.png: 3D visualization of the gradient descent process") 