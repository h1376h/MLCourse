import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

# Enable LaTeX rendering for all plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]
})

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Given data from the problem
w_current = np.array([0.1, 0.5, -0.3, 0.2, 0.4])  # Current weight vector
x = np.array([1, 0.8, 0.6, 0.4, 0.7])  # New data point
y_actual = 0.15  # Actual price change
alpha = 0.1  # Learning rate

print("=" * 80)
print("QUESTION 24: LMS ALGORITHM FOR FINANCIAL PREDICTION SYSTEM")
print("=" * 80)
print()

def task1_learning_rate_tradeoff():
    """Discuss increasing or decreasing learning rate for rapid market movements"""
    print("TASK 1: LEARNING RATE IMPACT ON RAPID MARKET MOVEMENTS")
    print("-" * 75)
    
    print("Problem: Model predictions lag behind rapid market movements")
    print("\nAnalysis:")
    print("1. Higher learning rate (α) allows the model to adapt more quickly to new data")
    print("2. Current behavior suggests the model is not updating quickly enough to capture rapid changes")
    
    print("\nRecommendation: INCREASE the learning rate α")
    
    print("\nTrade-offs:")
    print("- Advantage: Faster adaptation to new market conditions and trends")
    print("- Risk: Higher sensitivity to noise and potential instability")
    print("- Risk: May lead to overshooting the optimal weights")
    
    # Create visualization comparing different learning rates
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate a rapid market movement scenario
    t = np.linspace(0, 10, 100)
    
    # True signal with a sudden shift
    true_signal = np.zeros_like(t)
    true_signal[t >= 5] = 1.0
    
    # Add some noise to make it realistic
    np.random.seed(42)
    noisy_signal = true_signal + 0.1 * np.random.randn(len(t))
    
    # Simulate LMS tracking with different learning rates
    predictions_low_lr = np.zeros_like(t)
    predictions_medium_lr = np.zeros_like(t)
    predictions_high_lr = np.zeros_like(t)
    
    # Initial prediction starts at 0
    for i in range(1, len(t)):
        # Low learning rate (0.1)
        error_low = noisy_signal[i-1] - predictions_low_lr[i-1]
        predictions_low_lr[i] = predictions_low_lr[i-1] + 0.1 * error_low
        
        # Medium learning rate (0.3)
        error_medium = noisy_signal[i-1] - predictions_medium_lr[i-1]
        predictions_medium_lr[i] = predictions_medium_lr[i-1] + 0.3 * error_medium
        
        # High learning rate (0.7)
        error_high = noisy_signal[i-1] - predictions_high_lr[i-1]
        predictions_high_lr[i] = predictions_high_lr[i-1] + 0.7 * error_high
    
    # Plot everything
    ax.plot(t, true_signal, 'k-', linewidth=2, label='True Market Movement')
    ax.plot(t, noisy_signal, 'k:', alpha=0.5, label='Noisy Observations')
    ax.plot(t, predictions_low_lr, 'r-', linewidth=2, label=r'Low $\alpha = 0.1$')
    ax.plot(t, predictions_medium_lr, 'g-', linewidth=2, label=r'Medium $\alpha = 0.3$')
    ax.plot(t, predictions_high_lr, 'b-', linewidth=2, label=r'High $\alpha = 0.7$')
    
    # Highlight the rapid change region
    ax.axvspan(5, 5.5, color='gray', alpha=0.3, label='Rapid Market Change')
    
    # Add labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Effect of Learning Rate on Adapting to Rapid Market Movements')
    ax.legend(loc='best')
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "learning_rate_comparison.png"), dpi=300)
    plt.close()
    
    print("\nFigure saved: learning_rate_comparison.png")
    print()

def task2_calculate_update():
    """Calculate prediction and weight updates using the LMS algorithm"""
    print("TASK 2: UPDATING WEIGHTS WITH LMS ALGORITHM")
    print("-" * 75)
    
    print(f"Current weight vector w = {w_current}")
    print(f"New data point x = {x}")
    print(f"Actual price change y = {y_actual}")
    print(f"Learning rate α = {alpha}")
    
    # Detailed calculation of prediction
    terms = []
    for i in range(len(w_current)):
        term = w_current[i] * x[i]
        terms.append(term)
        
    # Calculate prediction with detailed steps
    prediction = np.sum(terms)
    detailed_calculation = " + ".join([f"{w_current[i]} × {x[i]} = {terms[i]:.4f}" for i in range(len(terms))])
    
    print("\nStep 2a: Calculate model's prediction")
    print(f"y_pred = w^T · x = {detailed_calculation} = {prediction:.4f}")
    
    # Calculate error with detailed steps
    error = y_actual - prediction
    
    print("\nStep 2b: Calculate prediction error")
    print(f"error = y - y_pred = {y_actual} - {prediction:.4f} = {error:.4f}")
    
    # Calculate weight updates with detailed steps
    gradient = error * x
    
    # Calculate detailed gradient computation
    gradient_calculation = [f"{error:.4f} × {x[i]} = {gradient[i]:.4f}" for i in range(len(gradient))]
    gradient_detailed = ", ".join(gradient_calculation)
    
    w_updates = alpha * gradient
    
    # Calculate detailed update computation
    updates_calculation = [f"{alpha} × {gradient[i]:.4f} = {w_updates[i]:.4f}" for i in range(len(w_updates))]
    updates_detailed = ", ".join(updates_calculation)
    
    w_new = w_current + w_updates
    
    # Calculate detailed weight update
    new_weights_calculation = [f"{w_current[i]} + {w_updates[i]:.4f} = {w_new[i]:.4f}" for i in range(len(w_new))]
    new_weights_detailed = ", ".join(new_weights_calculation)
    
    print("\nStep 2c: Calculate updated weight vector")
    print(f"Gradient = error * x = {error:.4f} * {x} = [{gradient_detailed}]")
    print(f"Weight updates = α * Gradient = {alpha} * [{gradient_detailed}] = [{updates_detailed}]")
    print(f"New weights = w + updates = {w_current} + [{updates_detailed}] = [{new_weights_detailed}]")
    
    # Create visualization of the prediction and update process
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Feature values
    ax1 = fig.add_subplot(gs[0, 0])
    feature_names = [r'Bias', r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
    ax1.bar(feature_names, x, color='skyblue')
    ax1.set_title('Feature Values')
    ax1.set_ylabel('Value')
    
    # Plot 2: Current weights vs New weights
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    x_pos = np.arange(len(feature_names))
    ax2.bar(x_pos - width/2, w_current, width, label='Current Weights', color='lightcoral')
    ax2.bar(x_pos + width/2, w_new, width, label='Updated Weights', color='lightgreen')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names)
    ax2.set_title('Weight Update')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    
    # Plot 3: Error and prediction
    ax3 = fig.add_subplot(gs[1, :])
    bars = ax3.bar(['Predicted', 'Actual'], [prediction, y_actual], color=['lightcoral', 'lightgreen'])
    
    # Add error text
    ax3.annotate(f'Error: {error:.4f}', 
                 xy=(0.5, max(prediction, y_actual) + 0.01),
                 xytext=(0.5, max(prediction, y_actual) + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 ha='center')
    
    # Add red line to show the error visually
    y_min = min(prediction, y_actual)
    y_max = max(prediction, y_actual)
    ax3.plot([0, 1], [y_max, y_max], 'r--', alpha=0.5)
    
    ax3.set_title('Prediction vs Actual')
    ax3.set_ylabel('Price Change')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_and_update.png"), dpi=300)
    plt.close()
    
    print("\nFigure saved: prediction_and_update.png")
    print()
    
    return prediction, error, w_new

def task3_gradient_clipping():
    """Derive a modified LMS update rule with gradient clipping"""
    print("TASK 3: GRADIENT CLIPPING FOR NOISE AND OUTLIERS")
    print("-" * 75)
    
    print("Standard LMS update rule:")
    print(r"    w^{(t+1)} = w^{(t)} + \alpha(y^{(i)} - w^T·x^{(i)})x^{(i)}")
    
    print("\nDeriving a modified LMS update rule with gradient clipping:")
    print(r"1. The gradient in LMS algorithm is: g = (y^{(i)} - w^T·x^{(i)})x^{(i)} = error·x^{(i)}")
    print(r"2. With gradient clipping, we cap the gradient if it exceeds a threshold τ")
    print("3. The clipped gradient is:")
    print(r"    g_clipped = g                              if ||g|| ≤ τ")
    print(r"                τ·g/||g||                      if ||g|| > τ")
    
    print("\nThe modified LMS update rule with gradient clipping is:")
    print(r"    w^{(t+1)} = w^{(t)} + \alpha·g_clipped")
    
    print("\nWhen expanded:")
    print(r"    w^{(t+1)} = w^{(t)} + \alpha·(y^{(i)} - w^T·x^{(i)})·x^{(i)}                 if ||(y^{(i)} - w^T·x^{(i)})·x^{(i)}|| ≤ τ")
    print(r"              w^{(t)} + \alpha·τ·(y^{(i)} - w^T·x^{(i)})·x^{(i)}/||(y^{(i)} - w^T·x^{(i)})·x^{(i)}||  if ||(y^{(i)} - w^T·x^{(i)})·x^{(i)}|| > τ")
    
    # Create visualization of gradient clipping
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a range of error values
    errors = np.linspace(-3, 3, 1000)
    
    # Standard LMS update (assuming x=1 for simplicity to visualize 1D)
    std_updates = errors
    
    # Gradient clipping with threshold τ = 1
    tau = 1.0
    clipped_updates = np.copy(std_updates)
    clipped_updates[std_updates > tau] = tau
    clipped_updates[std_updates < -tau] = -tau
    
    # Plot
    ax.plot(errors, std_updates, 'b-', label='Standard LMS Update')
    ax.plot(errors, clipped_updates, 'r-', label=r'Gradient Clipping ($\tau=1$)')
    
    # Add threshold lines
    ax.axhline(y=tau, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-tau, color='r', linestyle='--', alpha=0.5)
    
    # Highlight outlier region
    ax.axvspan(2, 3, color='yellow', alpha=0.3)
    ax.axvspan(-3, -2, color='yellow', alpha=0.3)
    ax.text(2.5, 0, 'Outlier\nRegion', ha='center', va='center', rotation=90)
    ax.text(-2.5, 0, 'Outlier\nRegion', ha='center', va='center', rotation=90)
    
    ax.set_xlabel('Error (y - prediction)')
    ax.set_ylabel('Weight Update Magnitude')
    ax.set_title('Gradient Clipping Effect on Weight Updates')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradient_clipping.png"), dpi=300)
    plt.close()
    
    print("\nFigure saved: gradient_clipping.png")
    print()

def task4_per_feature_learning_rate():
    """Propose a per-feature learning rate approach for the LMS algorithm"""
    print("TASK 4: PER-FEATURE LEARNING RATE APPROACH")
    print("-" * 75)
    
    print(r"Problem: Feature x_1 has high variance causing weight oscillations")
    
    print("\nProposed solution: Use different learning rates for different features")
    print(r"Instead of a single α for all weights, use a vector of learning rates α = [α_0, α_1, α_2, ..., α_n]")
    
    print("\nStandard LMS update rule (element-wise):")
    print(r"    w_j^{(t+1)} = w_j^{(t)} + \alpha·(y^{(i)} - w^T·x^{(i)})·x_j^{(i)}")
    
    print("\nModified LMS update rule with per-feature learning rates:")
    print(r"    w_j^{(t+1)} = w_j^{(t)} + \alpha_j·(y^{(i)} - w^T·x^{(i)})·x_j^{(i)}")
    
    print("\nIn vector form:")
    print(r"    w^{(t+1)} = w^{(t)} + (α ⊙ (y^{(i)} - w^T·x^{(i)})·x^{(i)})")
    print(r"    where ⊙ represents element-wise multiplication (Hadamard product)")
    
    print("\nFor our problem with high variance in x_1, we would set:")
    print(r"    α = [α_0, α_1_small, α_2, α_3, α_4]")
    print(r"    where α_1_small < α_0, α_2, α_3, α_4 to reduce oscillations in w_1")
    
    # Create sample data to demonstrate the concept
    np.random.seed(42)
    
    # Feature with high variance and others with normal variance
    high_var_feature = 2 + 1.5 * np.random.randn(100)  # High variance for x₁
    normal_features = 0.5 * np.random.randn(100, 3)    # Normal variance for x₂, x₃, x₄
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Feature distributions to show the high variance
    sns.histplot(high_var_feature, kde=True, label=r'$x_1$ (High Variance)', color='red', bins=20, alpha=0.5, ax=ax1)
    for i in range(normal_features.shape[1]):
        sns.histplot(normal_features[:, i], kde=True, label=f'$x_{i+2}$ (Normal)', alpha=0.5, bins=20, ax=ax1)
    
    ax1.set_title('Feature Value Distributions')
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Plot 2: Learning rate effect on weights
    # Simulate weight updates with standard and adaptive learning rates
    iterations = 50
    w_standard = np.zeros(5)  # Bias, x_1, x_2, x_3, x_4
    w_adaptive = np.zeros(5)   # Bias, x_1, x_2, x_3, x_4
    
    w_standard_history = np.zeros((iterations, 5))
    w_adaptive_history = np.zeros((iterations, 5))
    
    # Standard learning rate (same for all features)
    alpha_standard = 0.1
    
    # Adaptive learning rates (lower for high variance feature)
    alpha_adaptive = np.array([0.1, 0.02, 0.1, 0.1, 0.1])  # Lower rate for x_1
    
    for i in range(iterations):
        # Generate synthetic data point
        x_i = np.array([1.0, high_var_feature[i % len(high_var_feature)]] + 
                      list(normal_features[i % len(normal_features)]))
        
        # True relationship (assuming some known relationship)
        y_i = 1.0 + 0.5 * x_i[1] + 0.3 * x_i[2] + 0.2 * x_i[3] + 0.4 * x_i[4] + 0.1 * np.random.randn()
        
        # Standard LMS update
        error_standard = y_i - np.dot(w_standard, x_i)
        w_standard = w_standard + alpha_standard * error_standard * x_i
        w_standard_history[i] = w_standard
        
        # Adaptive LMS update
        error_adaptive = y_i - np.dot(w_adaptive, x_i)
        w_adaptive = w_adaptive + alpha_adaptive * error_adaptive * x_i
        w_adaptive_history[i] = w_adaptive
    
    # Plot weights trajectories
    feature_indices = [1]  # Only plot x₁ weight for clarity
    
    for j in feature_indices:
        ax2.plot(range(iterations), w_standard_history[:, j], 'r-', 
                 label=fr'Standard LMS ($\alpha={alpha_standard}$)')
        ax2.plot(range(iterations), w_adaptive_history[:, j], 'g-', 
                 label=fr'Adaptive LMS ($\alpha_1={alpha_adaptive[j]}$)')
    
    ax2.axhline(y=0.5, color='k', linestyle='--', label=r'True weight ($w_1=0.5$)')
    ax2.set_title(r'Weight Trajectory for High Variance Feature ($x_1$)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Weight Value')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_feature_learning_rate.png"), dpi=300)
    plt.close()
    
    print("\nFigure saved: per_feature_learning_rate.png")
    print()

def task5_comparison_diagram():
    """Create a diagram comparing standard LMS with the modified approaches"""
    print("TASK 5: COMPARISON OF STANDARD AND MODIFIED LMS APPROACHES")
    print("-" * 75)
    
    print("Comparing behavior when encountering an outlier data point:")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot titles
    titles = [
        'Standard LMS Update',
        'Gradient Clipping Approach',
        'Per-Feature Learning Rate Approach', 
        'Comparison of Weight Trajectories'
    ]
    
    # 1. Standard LMS visualization
    ax = axes[0, 0]
    
    # Create normal and outlier data
    np.random.seed(42)
    normal_points_x = np.random.randn(20)
    normal_points_y = 0.5 * normal_points_x + 0.1 * np.random.randn(20)
    outlier_x = 5.0
    outlier_y = -5.0
    
    # Plot normal data points
    ax.scatter(normal_points_x, normal_points_y, color='blue', alpha=0.6, label='Normal Data')
    
    # Plot outlier point
    ax.scatter(outlier_x, outlier_y, color='red', s=100, label='Outlier')
    
    # Plot initial regression line
    x_line = np.linspace(-3, 5, 100)
    initial_w = 0.5  # Initial slope
    initial_b = 0.0  # Initial intercept
    y_initial = initial_w * x_line + initial_b
    ax.plot(x_line, y_initial, 'g-', label='Initial Model')
    
    # Plot updated regression line after outlier
    # Simulating a single LMS update with large step due to outlier
    updated_w = initial_w - 0.1 * (initial_w * outlier_x + initial_b - outlier_y) * outlier_x
    updated_b = initial_b - 0.1 * (initial_w * outlier_x + initial_b - outlier_y)
    y_updated = updated_w * x_line + updated_b
    ax.plot(x_line, y_updated, 'r-', label='After Outlier (Large Change)')
    
    # Add big arrow to show the large weight update
    ax.annotate('', xy=(2, updated_w * 2 + updated_b), 
                xytext=(2, initial_w * 2 + initial_b),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, alpha=0.7))
    
    ax.set_title(titles[0])
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Target Value')
    ax.legend()
    ax.grid(True)
    
    # 2. Gradient Clipping approach
    ax = axes[0, 1]
    
    # Same data setup as before
    ax.scatter(normal_points_x, normal_points_y, color='blue', alpha=0.6, label='Normal Data')
    ax.scatter(outlier_x, outlier_y, color='red', s=100, label='Outlier')
    ax.plot(x_line, y_initial, 'g-', label='Initial Model')
    
    # Calculate clipped gradient update
    error = initial_w * outlier_x + initial_b - outlier_y
    gradient = error * np.array([1, outlier_x])  # [bias_gradient, weight_gradient]
    gradient_norm = np.linalg.norm(gradient)
    
    tau = 0.5  # Threshold for clipping
    if gradient_norm > tau:
        gradient = tau * gradient / gradient_norm
    
    # Apply clipped update
    clipped_w = initial_w - 0.1 * gradient[1] 
    clipped_b = initial_b - 0.1 * gradient[0]
    y_clipped = clipped_w * x_line + clipped_b
    ax.plot(x_line, y_clipped, 'purple', label='After Outlier (Clipped)')
    
    # Add smaller arrow to show the clipped update
    ax.annotate('', xy=(2, clipped_w * 2 + clipped_b), 
                xytext=(2, initial_w * 2 + initial_b),
                arrowprops=dict(facecolor='purple', shrink=0.05, width=1.5, alpha=0.7))
    
    ax.set_title(titles[1])
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Target Value')
    ax.legend()
    ax.grid(True)
    
    # 3. Per-Feature Learning Rate approach
    ax = axes[1, 0]
    
    # Same data setup as before
    ax.scatter(normal_points_x, normal_points_y, color='blue', alpha=0.6, label='Normal Data')
    ax.scatter(outlier_x, outlier_y, color='red', s=100, label='Outlier')
    ax.plot(x_line, y_initial, 'g-', label='Initial Model')
    
    # Apply per-feature learning rate update
    # Reduced learning rate for the feature weight and standard for bias
    alpha_feature = 0.02  # Reduced learning rate for the feature
    alpha_bias = 0.1      # Standard learning rate for the bias
    
    feature_w = initial_w - alpha_feature * (initial_w * outlier_x + initial_b - outlier_y) * outlier_x
    feature_b = initial_b - alpha_bias * (initial_w * outlier_x + initial_b - outlier_y)
    y_feature = feature_w * x_line + feature_b
    ax.plot(x_line, y_feature, 'orange', label='After Outlier (Per-Feature LR)')
    
    # Add arrow to show the effect
    ax.annotate('', xy=(2, feature_w * 2 + feature_b), 
                xytext=(2, initial_w * 2 + initial_b),
                arrowprops=dict(facecolor='orange', shrink=0.05, width=1.5, alpha=0.7))
    
    ax.set_title(titles[2])
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Target Value')
    ax.legend()
    ax.grid(True)
    
    # 4. Weight trajectory comparison
    ax = axes[1, 1]
    
    # Simulate multiple iterations with an outlier
    iterations = 50
    outlier_at = 25  # Outlier introduced at iteration 25
    
    # Initial weights
    w_standard = 0.5
    w_clipped = 0.5
    w_feature = 0.5
    
    w_standard_history = np.zeros(iterations)
    w_clipped_history = np.zeros(iterations)
    w_feature_history = np.zeros(iterations)
    
    for i in range(iterations):
        if i == outlier_at:
            # Outlier data point
            x_i = outlier_x
            y_i = outlier_y
        else:
            # Normal data point
            idx = i % len(normal_points_x)
            x_i = normal_points_x[idx]
            y_i = normal_points_y[idx]
        
        # Standard LMS update
        error_standard = w_standard * x_i - y_i
        w_standard = w_standard - 0.1 * error_standard * x_i
        w_standard_history[i] = w_standard
        
        # Gradient clipping update
        error_clipped = w_clipped * x_i - y_i
        gradient = error_clipped * x_i
        
        # Apply clipping
        if abs(gradient) > tau:
            gradient = tau * (gradient / abs(gradient))
        
        w_clipped = w_clipped - 0.1 * gradient
        w_clipped_history[i] = w_clipped
        
        # Per-feature learning rate update
        error_feature = w_feature * x_i - y_i
        w_feature = w_feature - (0.02 if abs(x_i) > 2.0 else 0.1) * error_feature * x_i
        w_feature_history[i] = w_feature
    
    # Plot weight trajectories
    ax.plot(range(iterations), w_standard_history, 'r-', label='Standard LMS')
    ax.plot(range(iterations), w_clipped_history, 'purple', label='Gradient Clipping')
    ax.plot(range(iterations), w_feature_history, 'orange', label='Per-Feature LR')
    
    # Mark outlier position
    ax.axvline(x=outlier_at, color='red', linestyle='--', alpha=0.5, label='Outlier Encountered')
    
    ax.set_title(titles[3])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Value')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lms_approaches_comparison.png"), dpi=300)
    plt.close()
    
    print("\nFigure saved: lms_approaches_comparison.png")
    print("\nKey observations:")
    print("1. Standard LMS: Outliers cause large weight updates, potentially destabilizing the model")
    print("2. Gradient Clipping: Limits the effect of outliers by capping the update magnitude")
    print("3. Per-Feature Learning Rate: Reduces sensitivity to high-variance features")
    print("4. Both modified approaches show more stable behavior compared to standard LMS")
    print()

# Execute all tasks
task1_learning_rate_tradeoff()
prediction, error, w_new = task2_calculate_update()
task3_gradient_clipping()
task4_per_feature_learning_rate()
task5_comparison_diagram()

# Summary of results
print("=" * 80)
print("QUESTION 24 SUMMARY")
print("=" * 80)
print(f"Task 1: For rapid market movements, INCREASE the learning rate α")
print(f"Task 2a: Model prediction = {prediction:.4f}")
print(f"Task 2b: Prediction error = {error:.4f}")
print(f"Task 2c: Updated weight vector = {w_new}")
print("Task 3: Derived gradient clipping LMS update rule:")
print("       w^(t+1) = w^t + α·g_clipped")
print("       where g_clipped = g if ||g|| ≤ τ, or τ·g/||g|| if ||g|| > τ")
print("Task 4: Derived per-feature learning rate update rule:")
print("       w_j^(t+1) = w_j^t + α_j·(y^(i) - w^T·x^(i))·x_j^(i)")
print("Task 5: Compared the behavior of different approaches with outliers")
print()
print(f"All visualizations saved to: {save_dir}") 