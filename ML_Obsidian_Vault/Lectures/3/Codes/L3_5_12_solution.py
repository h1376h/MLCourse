import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_12")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Question 12: Online Learning for Linear Regression")
print("\n## Problem Statement")
print("You're implementing online learning for linear regression where data arrives sequentially, and you need to decide how to process it.")
print("\nIn this problem:")
print("- New data points arrive every minute")
print("- You need to make predictions in real-time")
print("- You've trained an initial model on historical data")
print("- You have limited computational resources")

print("\n## Task 1: Write down the LMS update rule for online learning")
print("\nThe Least Mean Squares (LMS) update rule for online learning is:")
print("\nFor weights:")
print("θ_j := θ_j + α(y - h_θ(x))x_j")
print("\nIn matrix form for all parameters:")
print("θ := θ + α(y - h_θ(x))x")
print("\nWhere:")
print("- θ is the parameter vector [θ₀, θ₁, ..., θₙ]")
print("- α is the learning rate")
print("- y is the true target value")
print("- h_θ(x) is the predicted value")
print("- x is the feature vector [1, x₁, x₂, ..., xₙ] (with 1 for the intercept)")

print("\nFor our specific case with linear regression, h_θ(x) = θ₀ + θ₁x, so:")
print("θ₀ := θ₀ + α(y - (θ₀ + θ₁x)) · 1")
print("θ₁ := θ₁ + α(y - (θ₀ + θ₁x)) · x")
print("\nSimplifying:")
print("θ₀ := θ₀ + α(y - (θ₀ + θ₁x))")
print("θ₁ := θ₁ + α(y - (θ₀ + θ₁x)) · x")

# Step 2: Calculate prediction error
print("\n## Task 2: Calculate the prediction error for the new data point")
print("\nGiven:")
print("- Current model: h(x) = 1 + 1.5x")
print("- New data point: (x=3, y=7)")

# Current model parameters
theta_0 = 1
theta_1 = 1.5

# New data point
x_new = 3
y_new = 7

# Calculate prediction
prediction = theta_0 + theta_1 * x_new

# Calculate error
error = y_new - prediction

print("\nPrediction calculation:")
print(f"h(x) = θ₀ + θ₁x = {theta_0} + {theta_1} · {x_new} = {prediction}")

print("\nPrediction error calculation:")
print(f"error = y - h(x) = {y_new} - {prediction} = {error}")

# Step 3: Calculate updated parameters
print("\n## Task 3: Calculate the updated parameters")
print("\nGiven:")
print("- Learning rate α = 0.1")
print("- Error = y - h(x) = {}".format(error))

# Learning rate
alpha = 0.1

# Update parameters
new_theta_0 = theta_0 + alpha * error
new_theta_1 = theta_1 + alpha * error * x_new

print("\nUpdating θ₀:")
print(f"θ₀ := θ₀ + α(y - h(x)) = {theta_0} + {alpha} · {error} = {new_theta_0}")

print("\nUpdating θ₁:")
print(f"θ₁ := θ₁ + α(y - h(x)) · x = {theta_1} + {alpha} · {error} · {x_new} = {new_theta_1}")

print("\nNew model after update:")
print(f"h(x) = {new_theta_0} + {new_theta_1}x")

# Step 4: Visualize the update process
print("\n## Visualizing the update process")

# Create a range of x values for plotting
x_range = np.linspace(0, 5, 100)

# Calculate predictions using the old and new models
y_old = theta_0 + theta_1 * x_range
y_new_model = new_theta_0 + new_theta_1 * x_range

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the models
plt.plot(x_range, y_old, label=f'Old model: h(x) = {theta_0} + {theta_1}x', color='blue')
plt.plot(x_range, y_new_model, label=f'New model: h(x) = {new_theta_0:.3f} + {new_theta_1:.3f}x', color='green')

# Plot the new data point
plt.scatter(x_new, y_new, color='red', s=100, label='New data point (3, 7)')

# Plot the prediction on the old model
plt.scatter(x_new, prediction, color='orange', s=100, label=f'Prediction: h(3) = {prediction}')

# Draw a line from the prediction to the actual value to visualize the error
plt.plot([x_new, x_new], [prediction, y_new], 'r--', label=f'Error: {error}')

# Add labels and legend
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Online Learning: Model Update Process', fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(save_dir, "model_update.png"), dpi=300)
plt.close()

# Step 5: Compare online learning and batch retraining
print("\n## Task 4: Compare online learning with batch retraining")

# Create a table for comparison
comparison_data = {
    'Aspect': [
        'Processing Time', 
        'Memory Usage', 
        'Adaptability to New Data', 
        'Stability', 
        'Convergence to Optimal Solution',
        'Handling Non-Stationary Data',
        'Implementation Complexity'
    ],
    'Online Learning': [
        'Fast - O(d) per update, where d is the number of features',
        'Low - only needs to store model parameters',
        'High - immediately incorporates new information',
        'Lower - can oscillate or be sensitive to outliers',
        'May not reach the exact optimum but approaches it',
        'Good - continuously adapts to changing data distributions',
        'Simple - straightforward update rule'
    ],
    'Batch Retraining': [
        'Slow - O(nd) or more, where n is the number of samples',
        'High - needs to store entire dataset',
        'Low - requires complete retraining to incorporate new data',
        'Higher - less affected by individual outliers',
        'Can find the exact optimum for the given data',
        'Poor - slower to adapt to changing data distributions',
        'Complex - requires solving full optimization problem'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
pd.set_option('display.max_colwidth', None)
print("\nComparison Table:")
print(comparison_df.to_string(index=False))

# Create a visualization comparing computational resources over time
print("\n## Visualizing computational resources over time")

# Simulate data arrival and computational resources
time_points = np.arange(0, 50)
batch_computation = np.zeros_like(time_points, dtype=float)
online_computation = np.zeros_like(time_points, dtype=float)

# Every 10 time steps, batch retraining happens and uses a lot of resources
for i in range(len(time_points)):
    if i % 10 == 0:
        batch_computation[i] = 95  # 95% of resources
    online_computation[i] = 5  # Consistently low resource usage

# Create the plot
plt.figure(figsize=(12, 6))

plt.plot(time_points, batch_computation, label='Batch Retraining', color='blue', marker='o')
plt.plot(time_points, online_computation, label='Online Learning', color='green', marker='s')

plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Computational Resources (%)', fontsize=12)
plt.title('Computational Resource Usage Over Time', fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(save_dir, "resource_comparison.png"), dpi=300)
plt.close()

# Create a visualization showing model quality over time
print("\n## Visualizing model quality over time")

# Simulate model quality over time
time_points = np.arange(0, 50)
batch_quality = np.zeros_like(time_points, dtype=float)
online_quality = np.zeros_like(time_points, dtype=float)

for i in range(len(time_points)):
    # Batch quality is high right after retraining but degrades until next retraining
    batch_idx = i % 10
    batch_quality[i] = 95 - batch_idx * 3 if batch_idx > 0 else 95
    
    # Online quality gradually improves with small fluctuations
    if i == 0:
        online_quality[i] = 80
    else:
        # Add some randomness to the model quality improvement
        random_factor = np.random.uniform(-1, 1)
        online_quality[i] = min(95, online_quality[i-1] + 0.4 + random_factor)

# Create the plot
plt.figure(figsize=(12, 6))

plt.plot(time_points, batch_quality, label='Batch Retraining', color='blue', marker='o')
plt.plot(time_points, online_quality, label='Online Learning', color='green', marker='s')

plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Model Quality (%)', fontsize=12)
plt.title('Model Quality Over Time', fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig(os.path.join(save_dir, "quality_comparison.png"), dpi=300)
plt.close()

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- model_update.png: Visualization of the model update process")
print("- resource_comparison.png: Comparison of computational resource usage over time")
print("- quality_comparison.png: Comparison of model quality over time")

# Summary
print("\n## Summary")
print("1. The LMS update rule for online learning is: θ := θ + α(y - h_θ(x))x")
print("2. For the given data point (x=3, y=7) and model h(x) = 1 + 1.5x:")
print(f"   - Prediction: h(3) = {prediction}")
print(f"   - Error: y - h(x) = {error}")
print("3. After applying the update with learning rate α = 0.1:")
print(f"   - New θ₀ = {new_theta_0}")
print(f"   - New θ₁ = {new_theta_1}")
print(f"   - New model: h(x) = {new_theta_0} + {new_theta_1}x")
print("4. Comparing online learning with batch retraining:")
print("   - Online learning: More adaptive, uses fewer resources, simpler implementation")
print("   - Batch retraining: More stable, can find optimal solution, requires more resources")