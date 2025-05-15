import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font for math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Initialize the LMS model parameters
w_initial = np.array([10, 5, -3, 0.8])
alpha = 0.05  # Learning rate

# Print the initial setup
print("Least Mean Squares (LMS) Algorithm for Online Learning")
print("=" * 60)
print("\nInitial Setup:")
print(f"- Model form: h(x; w) = w₀ + w₁x₁ + w₂x₂ + w₃x₃")
print(f"- x₁: time of day (normalized between 0 and 1)")
print(f"- x₂: humidity (normalized between 0 and 1)")
print(f"- x₃: previous hour's temperature (in Celsius)")
print(f"- Initial weights: w = {w_initial}")
print(f"- Learning rate: α = {alpha}")
print("\n" + "=" * 60)

# Task 1: Write down the LMS update rule
def lms_update_rule():
    """Explain the LMS update rule for online learning."""
    print("\nTask 1: LMS Update Rule for Online Learning")
    print("-" * 50)
    
    print("The LMS update rule for online learning is:")
    print("For each feature j from 0 to 3:")
    print("    wⱼ := wⱼ + α(y - h(x; w))xⱼ")
    print("\nWhere:")
    print("- wⱼ is the weight for feature j")
    print("- α is the learning rate (in our case 0.05)")
    print("- y is the actual temperature")
    print("- h(x; w) is the predicted temperature")
    print("- xⱼ is the value of feature j (note: x₀ = 1 for the intercept term)")
    
    print("\nIn vector form, the update rule is:")
    print("    w := w + α(y - h(x; w))x")
    print("\nFor our specific problem with 3 features:")
    print("""
    w₀ := w₀ + α(y - (w₀ + w₁x₁ + w₂x₂ + w₃x₃)) × 1
    w₁ := w₁ + α(y - (w₀ + w₁x₁ + w₂x₂ + w₃x₃)) × x₁
    w₂ := w₂ + α(y - (w₀ + w₁x₁ + w₂x₂ + w₃x₃)) × x₂
    w₃ := w₃ + α(y - (w₀ + w₁x₁ + w₂x₂ + w₃x₃)) × x₃
    """)
    
    # Instead of creating a visualization, output LaTeX formatting for the markdown
    print("\nLaTeX representation for markdown:")
    print(r"For each weight $w_j$:")
    print(r"$$w_j := w_j + \alpha(y - h(\mathbf{x}; \mathbf{w}))x_j$$")
    print("\nIn vector form:")
    print(r"$$\mathbf{w} := \mathbf{w} + \alpha(y - h(\mathbf{x}; \mathbf{w}))\mathbf{x}$$")

lms_update_rule()

# Task 2: Calculate prediction for the first data point
def predict(weights, features):
    """Make a prediction using the linear model."""
    return np.dot(weights, features)

def process_first_datapoint():
    """Calculate prediction for the first data point."""
    print("\nTask 2: Prediction for the First Data Point")
    print("-" * 50)
    
    # First data point
    time = 0.75  # evening
    humidity = 0.4
    prev_temp = 22  # Celsius
    actual_temp = 24  # Celsius
    
    # Create feature vector with intercept term
    x1 = np.array([1, time, humidity, prev_temp])
    
    # Make prediction
    prediction = predict(w_initial, x1)
    
    # Display the data and prediction
    print(f"First data point:")
    print(f"- Time (x₁) = {time} (evening)")
    print(f"- Humidity (x₂) = {humidity}")
    print(f"- Previous temperature (x₃) = {prev_temp}°C")
    print(f"- Actual temperature (y) = {actual_temp}°C")
    print("\nPrediction calculation:")
    print(f"h(x; w) = w₀ + w₁x₁ + w₂x₂ + w₃x₃")
    print(f"       = {w_initial[0]} + {w_initial[1]} × {time} + {w_initial[2]} × {humidity} + {w_initial[3]} × {prev_temp}")
    print(f"       = {w_initial[0]} + {w_initial[1] * time:.2f} + ({w_initial[2] * humidity:.2f}) + {w_initial[3] * prev_temp:.2f}")
    print(f"       = {prediction:.2f}°C")
    
    # Calculate error
    error = actual_temp - prediction
    print(f"\nPrediction error: {actual_temp} - {prediction:.2f} = {error:.2f}°C")
    
    # Create visualization of the prediction with improved graphics
    plt.figure(figsize=(10, 6))
    
    # Create bars for the terms in the prediction
    terms = [w_initial[0], 
             w_initial[1] * time, 
             w_initial[2] * humidity, 
             w_initial[3] * prev_temp]
    term_labels = ['$w_0$ (Intercept)', 
                   '$w_1 x_1$ (Time)', 
                   '$w_2 x_2$ (Humidity)', 
                   '$w_3 x_3$ (Prev Temp)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot the stacked bars with better spacing
    plt.bar(0, terms, bottom=[sum(terms[:i]) for i in range(len(terms))], 
            width=0.6, color=colors, label=term_labels)
    
    # Add prediction and actual lines
    plt.axhline(y=prediction, linestyle='--', color='blue', linewidth=2, 
               label=f'Prediction: {prediction:.2f}°C')
    plt.axhline(y=actual_temp, linestyle='--', color='red', linewidth=2, 
               label=f'Actual: {actual_temp}°C')
    
    # Add annotations
    plt.annotate(f'Error: {error:.2f}°C', 
                xy=(0.2, (prediction + actual_temp)/2),
                xytext=(0.4, (prediction + actual_temp)/2),
                arrowprops=dict(facecolor='purple', shrink=0.05, width=2),
                color='purple', fontweight='bold')
    
    plt.ylim(0, max(30, actual_temp + 5))
    plt.xlim(-0.5, 0.5)
    plt.title('Temperature Prediction Breakdown', fontsize=16)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.xticks([])
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_breakdown.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nVisualization of prediction saved as 'prediction_breakdown.png'")
    
    return x1, actual_temp, prediction, error

x1, actual_temp1, prediction1, error1 = process_first_datapoint()

# Task 3: Calculate the new weight vector after processing the first data point
def update_weights(weights, features, actual, prediction, alpha):
    """Update weights using the LMS rule."""
    error = actual - prediction
    return weights + alpha * error * features

def display_weight_update(w_old, w_new, feature_vector, error, alpha):
    """Display detailed step-by-step weight update."""
    print("\nTask 3: Weight Update After Processing the First Data Point")
    print("-" * 50)
    
    print("Using the LMS update rule:")
    print(f"w_new = w_old + α × (y - prediction) × x")
    print(f"      = w_old + {alpha} × {error:.2f} × x")
    print(f"      = w_old + {alpha * error:.4f} × x")
    
    print("\nFor each weight:")
    features = ["1 (intercept)", "$x_1$ (time)", "$x_2$ (humidity)", "$x_3$ (prev temp)"]
    
    for i in range(len(w_old)):
        print(f"w_{i} = {w_old[i]:.4f} + {alpha * error:.4f} × {feature_vector[i]:.4f} = {w_old[i]:.4f} + {alpha * error * feature_vector[i]:.4f} = {w_new[i]:.4f}")
    
    # Create a table of the weight updates
    weight_table = pd.DataFrame({
        'Feature': features,
        'Old Weight': w_old,
        'Update Term': alpha * error * feature_vector,
        'New Weight': w_new
    })
    
    print("\nWeight Update Summary:")
    print(weight_table.to_string(index=False))
    
    # Visualize the weight updates with improved graphics
    plt.figure(figsize=(12, 6))
    
    width = 0.3
    indices = np.arange(len(w_old))
    
    # Create a more visually appealing plot
    plt.bar(indices - width/2, w_old, width, label='Original Weights', 
            color='#1f77b4', edgecolor='black', linewidth=1.5, alpha=0.8)
    plt.bar(indices + width/2, w_new, width, label='Updated Weights', 
            color='#ff7f0e', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add arrows for updates
    for i, (old, new) in enumerate(zip(w_old, w_new)):
        update = alpha * error * feature_vector[i]
        if abs(update) > 0.1:  # Only add arrows for significant updates
            plt.annotate('', 
                       xy=(i + width/2, new), 
                       xytext=(i - width/2, old),
                       arrowprops=dict(arrowstyle='->', linewidth=2,
                                      connectionstyle="arc3,rad=.2",
                                      color='green'))
    
    plt.ylabel('Weight Value', fontsize=12)
    plt.title('Weight Updates Using LMS Rule', fontsize=16)
    plt.xticks(indices, ['$w_0$ (Intercept)', '$w_1$ (Time)', '$w_2$ (Humidity)', '$w_3$ (Prev Temp)'])
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Ensure the y-axis is sufficiently expanded to show all values
    ymin = min(min(w_old), min(w_new)) - 1
    ymax = max(max(w_old), max(w_new)) + 1
    plt.ylim(ymin, ymax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_updates.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nVisualization of weight updates saved as 'weight_updates.png'")

# Update weights with the first data point
w_updated = update_weights(w_initial, x1, actual_temp1, prediction1, alpha)
display_weight_update(w_initial, w_updated, x1, error1, alpha)

# Task 4: Predict with the next data point using updated weights
def process_second_datapoint(updated_weights):
    """Predict temperature for the second data point using updated weights."""
    print("\nTask 4: Prediction for the Second Data Point with Updated Weights")
    print("-" * 50)
    
    # Second data point
    time2 = 0.8  # later evening
    humidity2 = 0.45
    prev_temp2 = 24  # Celsius
    
    # Create feature vector with intercept term
    x2 = np.array([1, time2, humidity2, prev_temp2])
    
    # Make prediction with updated weights
    prediction2 = predict(updated_weights, x2)
    
    # Display the data and prediction
    print(f"Second data point:")
    print(f"- Time (x₁) = {time2} (later evening)")
    print(f"- Humidity (x₂) = {humidity2}")
    print(f"- Previous temperature (x₃) = {prev_temp2}°C")
    print("\nPrediction calculation using updated weights:")
    print(f"h(x; w_updated) = w₀ + w₁x₁ + w₂x₂ + w₃x₃")
    print(f"               = {updated_weights[0]:.4f} + {updated_weights[1]:.4f} × {time2} + {updated_weights[2]:.4f} × {humidity2} + {updated_weights[3]:.4f} × {prev_temp2}")
    print(f"               = {updated_weights[0]:.4f} + {updated_weights[1] * time2:.4f} + ({updated_weights[2] * humidity2:.4f}) + {updated_weights[3] * prev_temp2:.4f}")
    print(f"               = {prediction2:.4f}°C")
    
    # Create improved visualization for the second prediction
    plt.figure(figsize=(10, 6))
    
    # Create bars for the terms in the prediction
    terms = [updated_weights[0], 
             updated_weights[1] * time2, 
             updated_weights[2] * humidity2, 
             updated_weights[3] * prev_temp2]
    term_labels = ['$w_0$ (Intercept)', 
                   '$w_1 x_1$ (Time)', 
                   '$w_2 x_2$ (Humidity)', 
                   '$w_3 x_3$ (Prev Temp)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot each component separately for better visibility
    plt.bar(range(len(terms)), [max(0, t) for t in terms], color=colors, label=term_labels)
    plt.bar(range(len(terms)), [min(0, t) for t in terms], color=colors)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add prediction line
    plt.axhline(y=prediction2, linestyle='--', color='blue', linewidth=2, 
               label=f'Prediction: {prediction2:.2f}°C')
    
    plt.title('Component Contributions to Temperature Prediction', fontsize=16)
    plt.ylabel('Temperature Contribution (°C)', fontsize=12)
    plt.xticks(range(len(terms)), term_labels, rotation=45, ha='right')
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set appropriate limits
    plt.ylim(min(min(terms) - 10, prediction2 - 10), max(max(terms) + 10, 30))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "second_prediction.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nVisualization of second prediction saved as 'second_prediction.png'")
    
    return x2, prediction2

x2, prediction2 = process_second_datapoint(w_updated)

# Task 5: Discuss handling outliers in online learning with LMS
def discuss_outlier_handling():
    """Explain how to handle outliers in online learning with LMS."""
    print("\nTask 5: Handling Outliers in Online Learning with LMS")
    print("-" * 50)
    
    print("In online learning with LMS, outliers can significantly impact model performance\n"
          "since each update directly affects the weights. Here are approaches to handle outliers:\n")
    
    print("1. Robust LMS: Modified Update Rule")
    print("   Instead of the standard update rule, we can use a modified version that reduces\n"
          "   the influence of large errors that might be caused by outliers.\n")
    
    print("   Standard LMS update: w := w + α(y - h(x; w))x")
    print("   Robust LMS update:   w := w + α × g(y - h(x; w)) × x\n")
    
    print("   Where g() is a function that dampens the effect of large errors.\n")
    
    print("2. Specific Modifications:")
    print("   a) Clipped LMS: Limit the error term to a maximum value")
    print("      g(error) = min(max(-threshold, error), threshold)")
    print("      w := w + α × clipped_error × x\n")
    
    print("   b) Huber Loss: Use a quadratic function near zero and linear function elsewhere")
    print("      g(error) = error                       if |error| ≤ δ")
    print("      g(error) = δ × sign(error)             if |error| > δ\n")
    
    print("   c) Dynamic Learning Rate: Reduce learning rate for large errors")
    print("      α_dynamic = α / (1 + |error|/σ)")
    print("      w := w + α_dynamic × error × x\n")
    
    print("3. Practical Implementation Example (Clipped LMS):")
    print("   # Set threshold for error clipping")
    print("   threshold = 2.0  # Consider errors larger than 2°C as potential outliers")
    print("   # Clip the error")
    print("   error = actual_temp - prediction")
    print("   clipped_error = max(min(error, threshold), -threshold)")
    print("   # Update weights with clipped error")
    print("   w := w + α × clipped_error × x\n")
    
    # Create improved visualizations comparing standard LMS vs robust LMS
    # First, create artificial data with an outlier
    np.random.seed(42)
    x_values = np.linspace(0, 1, 20)
    true_temp = 20 + 5 * x_values  # True relationship
    y_normal = true_temp + np.random.normal(0, 1, 20)  # Normal measurements
    y_outlier = y_normal.copy()
    y_outlier[10] = 40  # Add outlier
    
    # Plot showing effect of outlier on standard LMS vs robust LMS
    plt.figure(figsize=(12, 6))
    
    # Plot data points
    plt.scatter(x_values, y_normal, color='blue', s=60, alpha=0.7, 
                edgecolor='black', linewidth=1, label='Normal measurements')
    plt.scatter(x_values[10], y_outlier[10], color='red', s=120, alpha=0.8, 
                edgecolor='black', linewidth=1, label='Outlier')
    
    # Plot true relationship
    plt.plot(x_values, true_temp, 'k--', linewidth=2, label='True relationship')
    
    # Simulate Standard LMS and Robust LMS
    x_continuous = np.linspace(0, 1, 100)
    
    # Standard LMS (highly affected by outlier)
    standard_lms = 19 + 8 * x_continuous  # Simulated affected fit
    
    # Robust LMS (less affected by outlier)
    robust_lms = 20 + 5.5 * x_continuous  # Simulated robust fit
    
    plt.plot(x_continuous, standard_lms, 'r-', linewidth=3, alpha=0.7, label='Standard LMS')
    plt.plot(x_continuous, robust_lms, 'g-', linewidth=3, alpha=0.7, label='Robust LMS')
    
    plt.title('Effect of Outliers: Standard LMS vs. Robust LMS', fontsize=16)
    plt.xlabel('Feature Value', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "robust_lms_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization of error handling functions with improved graphics
    plt.figure(figsize=(10, 6))
    
    # Generate error values
    errors = np.linspace(-10, 10, 1000)
    
    # Different error handling functions
    standard = errors  # Standard LMS (uses error directly)
    clipped = np.clip(errors, -3, 3)  # Clipped LMS
    
    # Huber function
    delta = 3
    huber = np.where(np.abs(errors) <= delta, errors, delta * np.sign(errors))
    
    # Dynamic learning rate effect
    dynamic_factor = 0.05 / (1 + np.abs(errors)/3)  # α/(1 + |error|/σ) with α=0.05 and σ=3
    dynamic = errors * dynamic_factor / 0.05  # Normalized for visualization
    
    # Plot with better styling
    plt.plot(errors, standard, label='Standard LMS', color='blue', linewidth=2.5)
    plt.plot(errors, clipped, label='Clipped LMS', color='red', linewidth=2.5)
    plt.plot(errors, huber, label='Huber Loss', color='green', linewidth=2.5)
    plt.plot(errors, dynamic, label='Dynamic LR Effect', color='purple', linewidth=2.5)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight the clipping region
    plt.axvspan(-3, 3, alpha=0.1, color='gray')
    
    plt.title('Comparison of Error Handling Functions for Robust LMS', fontsize=16)
    plt.xlabel('Error (y - h(x))', fontsize=12)
    plt.ylabel('Effective Error for Weight Update', fontsize=12)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_handling_functions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations of robust LMS approaches saved as 'robust_lms_comparison.png' and 'error_handling_functions.png'")

discuss_outlier_handling()

# Summary of results
print("\nSummary of Results")
print("=" * 60)
print(f"Initial weights: {w_initial}")
print(f"First data point prediction: {prediction1:.4f}°C (actual: {actual_temp1}°C, error: {error1:.4f}°C)")
print(f"Updated weights after processing first data point: {w_updated}")
print(f"Second data point prediction using updated weights: {prediction2:.4f}°C")
print("\nImages generated:")
for img in os.listdir(save_dir):
    if img != "lms_update_rule.png":  # Skip the removed image
        print(f"- {img}")
print("\nA robust version of LMS can be implemented by modifying the update rule to reduce")
print("the influence of outliers, such as clipping errors, using Huber loss, or applying")
print("dynamic learning rates that decrease with larger errors.") 