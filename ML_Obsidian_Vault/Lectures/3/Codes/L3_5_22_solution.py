import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Question 22: Online Learning for Linear Regression")
print("\n## Understanding Online Learning vs. Batch Learning")

# Comparing online learning vs batch learning
print("### Online Learning vs. Batch Learning Comparison")
print("\nBatch Learning:")
print("- Processes all training data at once")
print("- Updates model parameters after seeing all examples")
print("- Requires storing the entire dataset in memory")
print("- More computationally intensive for large datasets")
print("- Generally more stable parameter updates")
print("\nOnline Learning:")
print("- Processes data points one at a time as they arrive")
print("- Updates model parameters after each example")
print("- Only needs to store the current data point")
print("- More efficient for large or streaming datasets")
print("- May have less stable updates but adapts quickly to new patterns")

# Create a comparison visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data points
np.random.seed(42)
X = np.linspace(0, 10, 20)
y = 2*X + 1 + np.random.normal(0, 1, 20)

# Display batch vs online learning
ax.scatter(X, y, color='blue', alpha=0.7, label='Data points')

# Initial line (before any learning)
initial_w = np.array([0, 0])
initial_x = np.array([0, 10])
initial_y = initial_w[0] + initial_w[1] * initial_x
ax.plot(initial_x, initial_y, 'r--', alpha=0.5, label='Initial model')

# Batch learning line (final model after seeing all data)
batch_w = np.array([1.05, 1.96])  # Simulated batch learning result
batch_y = batch_w[0] + batch_w[1] * initial_x
ax.plot(initial_x, batch_y, 'g-', linewidth=2, label='Batch learning (after all data)')

# Online learning intermediate lines (after seeing part of the data)
online_w_intermediate = np.array([0.5, 1.2])  # Simulated online learning after some data
online_y_intermediate = online_w_intermediate[0] + online_w_intermediate[1] * initial_x
ax.plot(initial_x, online_y_intermediate, 'y--', alpha=0.7, label='Online learning (intermediate)')

# Online learning final line (after seeing all data)
online_w_final = np.array([1.1, 1.9])  # Slightly different from batch due to sequential updates
online_y_final = online_w_final[0] + online_w_final[1] * initial_x
ax.plot(initial_x, online_y_final, 'm-', linewidth=2, label='Online learning (after all data)')

ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Batch Learning vs. Online Learning for Linear Regression')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "batch_vs_online.png"), dpi=300)
plt.close()

print("\n## Least Mean Squares (LMS) Update Rule")
print("\nThe Least Mean Squares (LMS) update rule for online learning in linear regression is:")
print("\nw ← w + α(y - wᵀx)x")
print("\nWhere:")
print("- w is the weight vector")
print("- α is the learning rate")
print("- y is the actual target value")
print("- x is the feature vector")
print("- (y - wᵀx) is the prediction error")

# Create a visualization of the LMS update rule
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data
x_sample = np.array([1, 2])  # Feature vector [bias term, x]
y_sample = 5  # Target value
w_sample = np.array([1, 1])  # Initial weights

# Current prediction
y_pred = np.dot(w_sample, x_sample)
error = y_sample - y_pred

# Plot the data point
ax.scatter([x_sample[1]], [y_sample], color='blue', s=100, label=f'Data point: ({x_sample[1]}, {y_sample})')

# Plot current model
x_range = np.array([0, 4])
y_range = w_sample[0] + w_sample[1] * x_range
ax.plot(x_range, y_range, 'r--', label=f'Current model: y = {w_sample[0]} + {w_sample[1]}x')

# Calculate updated weights
alpha = 0.1
w_new = w_sample + alpha * error * x_sample

# Plot updated model
y_range_new = w_new[0] + w_new[1] * x_range
ax.plot(x_range, y_range_new, 'g-', label=f'Updated model: y = {w_new[0]:.2f} + {w_new[1]:.2f}x')

# Add error visualization
ax.plot([x_sample[1], x_sample[1]], [y_pred, y_sample], 'k--', label=f'Error: {error}')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Least Mean Squares (LMS) Update Example')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "lms_update.png"), dpi=300)
plt.close()

print("\n## Calculating Updated Parameters for a New Data Point")
print("\nGiven information:")
print("- Current model parameters: w = [1, 2, 1]ᵀ")
print("- New data point: x = [1, 2, 3]ᵀ, y = 14")
print("- Learning rate: α = 0.1")

# Calculate the parameter update for the given values
w_current = np.array([1, 2, 1])
x_new = np.array([1, 2, 3])
y_new = 14
alpha = 0.1

# Current prediction
y_pred = np.dot(w_current, x_new)

# Calculate error
error = y_new - y_pred

# Calculate update
update = alpha * error * x_new

# Calculate new weights
w_updated = w_current + update

print("\nStep-by-step calculation:")
print(f"1. Current prediction: wᵀx = {w_current[0]}*{x_new[0]} + {w_current[1]}*{x_new[1]} + {w_current[2]}*{x_new[2]} = {y_pred}")
print(f"2. Error: (y - wᵀx) = {y_new} - {y_pred} = {error}")
print(f"3. Update term: α(y - wᵀx)x = {alpha} * {error} * [1, 2, 3]ᵀ = [{update[0]}, {update[1]}, {update[2]}]ᵀ")
print(f"4. Updated weights: w + α(y - wᵀx)x = [{w_current[0]}, {w_current[1]}, {w_current[2]}]ᵀ + [{update[0]}, {update[1]}, {update[2]}]ᵀ = [{w_updated[0]}, {w_updated[1]}, {w_updated[2]}]ᵀ")

# Visualization of the update
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart for weights
bar_width = 0.35
x_pos = np.arange(3)
labels = ['w₀', 'w₁', 'w₂']

bars1 = ax.bar(x_pos - bar_width/2, w_current, bar_width, color='skyblue', label='Current weights')
bars2 = ax.bar(x_pos + bar_width/2, w_updated, bar_width, color='lightgreen', label='Updated weights')

# Add update direction arrows and values
for i in range(3):
    if update[i] > 0:
        ax.arrow(x_pos[i] - bar_width/2, w_current[i], 0, update[i]*0.8, 
                width=0.03, head_width=0.1, head_length=0.1, fc='red', ec='red')
    else:
        ax.arrow(x_pos[i] - bar_width/2, w_current[i], 0, update[i]*0.8, 
                width=0.03, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax.text(x_pos[i] - bar_width/2 + 0.1, w_current[i] + update[i]/2, f'{update[i]:.2f}', 
            color='red', ha='center', fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Weight Value')
ax.set_title('LMS Update of Model Parameters')
ax.legend()

# Add text box with calculation details
textbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_str = (f"Current w = [{w_current[0]}, {w_current[1]}, {w_current[2]}]ᵀ\n"
            f"x = [{x_new[0]}, {x_new[1]}, {x_new[2]}]ᵀ, y = {y_new}\n"
            f"Prediction = {y_pred}\n"
            f"Error = {error}\n"
            f"α = {alpha}\n"
            f"Update = [{update[0]:.2f}, {update[1]:.2f}, {update[2]:.2f}]ᵀ\n"
            f"New w = [{w_updated[0]:.2f}, {w_updated[1]:.2f}, {w_updated[2]:.2f}]ᵀ")

ax.text(0.05, 0.65, text_str, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=textbox_props)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "parameter_update.png"), dpi=300)
plt.close()

print("\n## Trade-offs in Online Learning")

print("\n### A. Large Learning Rate vs. Small Learning Rate")
print("\nLarge Learning Rate (α):")
print("- Advantages:")
print("  - Faster initial convergence")
print("  - Adapts quickly to new patterns")
print("  - Can escape shallow local minima")
print("- Disadvantages:")
print("  - Risk of overshooting the minimum")
print("  - May not converge (oscillate around the minimum)")
print("  - Less stable updates, higher variance")
print("  - Can be more sensitive to outliers")
print("\nSmall Learning Rate (α):")
print("- Advantages:")
print("  - More stable updates, lower variance")
print("  - More likely to converge")
print("  - Less sensitive to noisy data or outliers")
print("- Disadvantages:")
print("  - Slower convergence")
print("  - May get stuck in local minima")
print("  - May take too long to adapt to new patterns")

# Create a visualization for learning rate trade-offs
fig, ax = plt.subplots(figsize=(12, 6))

# Simple loss function visualization (parabola)
theta = np.linspace(-4, 4, 100)
loss = theta**2 + 2  # Simple quadratic loss function

ax.plot(theta, loss, 'b-', linewidth=2)
ax.set_xlabel('Model parameter (θ)')
ax.set_ylabel('Loss function')
ax.set_title('Learning Rate Trade-offs in Gradient Descent')

# Starting point
start_theta = -3
start_loss = start_theta**2 + 2

# Learning paths
# High learning rate
high_alpha = 0.8
high_thetas = [start_theta]
high_losses = [start_loss]

# Medium learning rate
medium_alpha = 0.4
medium_thetas = [start_theta]
medium_losses = [start_loss]

# Low learning rate
low_alpha = 0.1
low_thetas = [start_theta]
low_losses = [start_loss]

# Simulate gradient descent steps
for i in range(10):
    # Gradient of loss function (derivative of theta^2 + 2 is 2*theta)
    gradient = 2 * high_thetas[-1]
    high_theta_new = high_thetas[-1] - high_alpha * gradient
    high_loss_new = high_theta_new**2 + 2
    high_thetas.append(high_theta_new)
    high_losses.append(high_loss_new)
    
    gradient = 2 * medium_thetas[-1]
    medium_theta_new = medium_thetas[-1] - medium_alpha * gradient
    medium_loss_new = medium_theta_new**2 + 2
    medium_thetas.append(medium_theta_new)
    medium_losses.append(medium_loss_new)
    
    gradient = 2 * low_thetas[-1]
    low_theta_new = low_thetas[-1] - low_alpha * gradient
    low_loss_new = low_theta_new**2 + 2
    low_thetas.append(low_theta_new)
    low_losses.append(low_loss_new)

# Plot learning paths
ax.scatter(high_thetas, high_losses, color='red', s=50, alpha=0.7)
ax.plot(high_thetas, high_losses, 'r--', alpha=0.5, linewidth=1)
for i in range(len(high_thetas)-1):
    ax.annotate('', xy=(high_thetas[i+1], high_losses[i+1]), 
                xytext=(high_thetas[i], high_losses[i]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.scatter(medium_thetas, medium_losses, color='green', s=50, alpha=0.7)
ax.plot(medium_thetas, medium_losses, 'g--', alpha=0.5, linewidth=1)
for i in range(len(medium_thetas)-1):
    ax.annotate('', xy=(medium_thetas[i+1], medium_losses[i+1]), 
                xytext=(medium_thetas[i], medium_losses[i]),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

ax.scatter(low_thetas, low_losses, color='purple', s=50, alpha=0.7)
ax.plot(low_thetas, low_losses, 'm--', alpha=0.5, linewidth=1)
for i in range(len(low_thetas)-1):
    ax.annotate('', xy=(low_thetas[i+1], low_losses[i+1]), 
                xytext=(low_thetas[i], low_losses[i]),
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

# Add a legend
high_patch = mpatches.Patch(color='red', label=f'Large α = {high_alpha}')
medium_patch = mpatches.Patch(color='green', label=f'Medium α = {medium_alpha}')
low_patch = mpatches.Patch(color='purple', label=f'Small α = {low_alpha}')
ax.legend(handles=[high_patch, medium_patch, low_patch])

# Add annotations
ax.annotate('Overshooting & Oscillation', xy=(high_thetas[2], high_losses[2]), 
            xytext=(high_thetas[2]-1, high_losses[2]+5),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('Optimal Convergence', xy=(medium_thetas[-1], medium_losses[-1]), 
            xytext=(medium_thetas[-1]-2, medium_losses[-1]+3),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('Slow Convergence', xy=(low_thetas[-1], low_losses[-1]), 
            xytext=(low_thetas[-1]-2, low_losses[-1]+1),
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "learning_rate_tradeoffs.png"), dpi=300)
plt.close()

print("\n### B. Online Learning vs. Batch Learning Trade-offs")
print("\nOnline Learning Advantages:")
print("- Memory efficiency: only stores current data point")
print("- Computational efficiency for large datasets")
print("- Can handle streaming data and adapt to changes")
print("- Works well when data is non-stationary (distribution changes over time)")
print("- Can be more robust to redundant data points")
print("\nOnline Learning Disadvantages:")
print("- Potentially less stable parameter updates")
print("- Sensitive to the order of data points")
print("- May converge to a suboptimal solution")
print("- More sensitive to learning rate selection")
print("- Performance can degrade with noisy data")
print("\nBatch Learning Advantages:")
print("- More stable and optimal parameter updates")
print("- Not affected by the order of data presentation")
print("- Generally more accurate for fixed datasets")
print("- Better utilization of vectorization and parallelization")
print("- Better theoretical guarantees of convergence")
print("\nBatch Learning Disadvantages:")
print("- High memory requirements for large datasets")
print("- Computationally intensive for large datasets")
print("- Cannot adapt to changing data distributions without retraining")
print("- Not suitable for streaming data scenarios")

# Create a visualization comparing online vs batch learning
fig, ax = plt.subplots(figsize=(12, 7))

# Parameters for the visualization
np.random.seed(42)
num_features = 2
num_samples = 50
learning_rate = 0.01

# Generate synthetic data
X = np.random.rand(num_samples, num_features) * 10
true_w = np.array([2, -1])
y = np.dot(X, true_w) + np.random.normal(0, 1, num_samples)

# Initialize weights
batch_w = np.zeros(num_features)
online_w = np.zeros(num_features)

# Arrays to store loss history
batch_loss_history = []
online_loss_history = []

# Function to compute loss
def compute_loss(w, X, y):
    y_pred = np.dot(X, w)
    return np.mean((y - y_pred) ** 2)

# Initial losses
batch_loss_history.append(compute_loss(batch_w, X, y))
online_loss_history.append(compute_loss(online_w, X, y))

# Simulate batch learning - one full batch gradient descent step
y_pred_batch = np.dot(X, batch_w)
errors_batch = y - y_pred_batch
batch_update = learning_rate * np.dot(X.T, errors_batch) / num_samples
batch_w += batch_update
batch_loss_history.append(compute_loss(batch_w, X, y))

# Simulate online learning - update weights for each sample one by one
for i in range(num_samples):
    x_i = X[i]
    y_i = y[i]
    y_pred_i = np.dot(online_w, x_i)
    error_i = y_i - y_pred_i
    online_w += learning_rate * error_i * x_i
    # Store loss after each update
    online_loss_history.append(compute_loss(online_w, X, y))

# Plot loss vs updates
x_batch = [0, 1]  # Only two points for batch: initial and after update
x_online = np.arange(len(online_loss_history))

ax.plot(x_batch, batch_loss_history, 'b-', marker='o', markersize=8, linewidth=2, label='Batch Learning')
ax.plot(x_online, online_loss_history, 'r-', linewidth=2, label='Online Learning')

ax.set_xlabel('Number of Updates')
ax.set_ylabel('Mean Squared Error Loss')
ax.set_title('Loss Convergence: Online vs Batch Learning')
ax.grid(True)
ax.legend()

# Add annotations
ax.annotate('Single batch update', xy=(1, batch_loss_history[1]), 
            xytext=(5, batch_loss_history[1]+2),
            arrowprops=dict(arrowstyle='->', color='blue'))

ax.annotate('Online updates\n(one per data point)', xy=(25, online_loss_history[25]), 
            xytext=(30, online_loss_history[25]+3),
            arrowprops=dict(arrowstyle='->', color='red'))

# Add a text box highlighting key observations
textbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_str = ("Key Observations:\n"
            "• Batch learning: single update using all data\n"
            "• Online learning: multiple updates (one per sample)\n"
            "• Online: initially less stable, more updates\n"
            "• Batch: more stable, fewer updates\n"
            "• Final performance depends on learning rate & data")

ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=textbox_props)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "online_vs_batch.png"), dpi=300)
plt.close()

print("\n## Real-World Scenarios for Online Learning")
print("\nHere are scenarios where online learning is particularly valuable:")
print("\n1. **Recommendation Systems**")
print("   - User preferences constantly evolve")
print("   - New items are continuously added")
print("   - Model needs to quickly adapt to user behavior changes")
print("   - Example: Netflix movie recommendations or Amazon product suggestions")
print("\n2. **Financial Market Prediction**")
print("   - Market conditions change rapidly")
print("   - Historical patterns may become obsolete")
print("   - New data arrives in real-time")
print("   - Example: Stock price prediction or algorithmic trading")
print("\n3. **Internet of Things (IoT) and Sensor Data**")
print("   - Continuous stream of sensor readings")
print("   - Limited memory and computational power on devices")
print("   - Example: Smart home devices, industrial sensors, or wearable health monitors")
print("\n4. **Social Media Analysis**")
print("   - Trends and topics change quickly")
print("   - Massive volume of new content")
print("   - Example: Sentiment analysis, trend detection, or content recommendation")
print("\n5. **Fraud Detection**")
print("   - Fraudsters continuously develop new strategies")
print("   - Requires immediate adaptation to new patterns")
print("   - Example: Credit card fraud detection or cybersecurity systems")
print("\n6. **Traffic Prediction**")
print("   - Traffic patterns change based on time, events, weather")
print("   - Requires real-time updates")
print("   - Example: Google Maps predicting travel times or ride-sharing pricing")
print("\n7. **Ad Click-Through Rate (CTR) Prediction**")
print("   - User interests change over time")
print("   - New ads are constantly being created")
print("   - Example: Real-time bidding systems for digital advertising")

# Create a visualization of a real-world scenario: Real-time Fraud Detection
fig, ax = plt.subplots(figsize=(12, 6))

# Simulate fraud detection scenario
np.random.seed(42)
num_days = 100
time_points = np.arange(num_days)

# Legitimate transaction amount distributions
legitimate_mean_initial = 50
legitimate_std = 15
legitimate_amounts = np.random.normal(legitimate_mean_initial, legitimate_std, num_days)

# Fraudulent transaction patterns - changing over time to simulate fraudsters adapting
fraud_amounts = np.zeros(num_days)
for i in range(num_days):
    if i < 30:
        # Initial fraud pattern: very high amounts
        fraud_amounts[i] = 300 + np.random.normal(0, 30)
    elif i < 60:
        # Fraudsters adapt: amounts closer to legitimate to avoid detection
        fraud_amounts[i] = 100 + np.random.normal(0, 20)
    else:
        # Fraudsters adapt again: amounts very close to legitimate
        fraud_amounts[i] = 70 + np.random.normal(0, 15)

# Detection threshold with batch learning (fixed model, rarely updated)
batch_threshold = np.ones(num_days) * 150
batch_threshold[40:] = 90  # One manual update after 40 days

# Detection threshold with online learning (continuously adapting)
online_threshold = np.zeros(num_days)
online_threshold[0] = 150  # Start with the same threshold

# Simple online learning algorithm for threshold
for i in range(1, num_days):
    # Adapt threshold based on recent fraud patterns
    # (simplified for illustration - real systems would be more complex)
    if i < 10:
        online_threshold[i] = online_threshold[i-1]  # Not enough data yet
    else:
        # Moving average of fraud amounts plus margin
        recent_fraud_mean = np.mean(fraud_amounts[max(0, i-10):i])
        recent_legitimate_mean = np.mean(legitimate_amounts[max(0, i-10):i])
        # Set threshold between legitimate and fraud means
        online_threshold[i] = (recent_legitimate_mean + recent_fraud_mean) / 2

# Plot the transactions and thresholds
ax.scatter(time_points, legitimate_amounts, color='blue', alpha=0.6, s=30, label='Legitimate Transactions')
ax.scatter(time_points, fraud_amounts, color='red', alpha=0.6, s=30, label='Fraudulent Transactions')

ax.plot(time_points, batch_threshold, 'g-', linewidth=2, label='Batch Learning Threshold (rarely updated)')
ax.plot(time_points, online_threshold, 'm-', linewidth=2, label='Online Learning Threshold (continuously adapting)')

# Highlight missed frauds and false positives
missed_frauds_batch = (fraud_amounts < batch_threshold)
missed_frauds_online = (fraud_amounts < online_threshold)

false_positives_batch = (legitimate_amounts > batch_threshold)
false_positives_online = (legitimate_amounts > online_threshold)

# Highlight areas where batch model would miss frauds but online wouldn't
for i in range(num_days):
    if missed_frauds_batch[i] and not missed_frauds_online[i]:
        ax.scatter(time_points[i], fraud_amounts[i], marker='o', s=100, 
                  facecolors='none', edgecolors='black', linewidth=2,
                  alpha=0.7)

# Add annotations
ax.annotate('Fraudsters change\ntheir behavior', xy=(30, fraud_amounts[30]), 
            xytext=(20, fraud_amounts[30]+70),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('Online model adapts\nquickly', xy=(35, online_threshold[35]), 
            xytext=(20, online_threshold[35]-50),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('Batch model updated\nmanually', xy=(40, batch_threshold[40]), 
            xytext=(45, batch_threshold[40]+30),
            arrowprops=dict(arrowstyle='->', color='black'))

ax.annotate('Fraudsters adapt again', xy=(60, fraud_amounts[60]), 
            xytext=(65, fraud_amounts[60]+50),
            arrowprops=dict(arrowstyle='->', color='black'))

# Set labels and title
ax.set_xlabel('Time (Days)')
ax.set_ylabel('Transaction Amount')
ax.set_title('Real-World Scenario: Fraud Detection with Online Learning')
ax.legend(loc='upper right')

# Add key insights text box
textbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_str = ("Key Insights:\n"
            "• Fraudsters continually adapt their behavior\n"
            "• Online learning model adjusts threshold in real-time\n"
            "• Batch learning model requires manual updates\n"
            "• Circles: frauds detected by online but missed by batch\n"
            "• Online learning is critical when patterns evolve rapidly")

ax.text(0.02, 0.97, text_str, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=textbox_props)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fraud_detection_scenario.png"), dpi=300)
plt.close()

print("\n## Conclusion")
print("\nOnline learning is a powerful approach for linear regression when data arrives sequentially:")
print("- The LMS update rule provides a simple way to update model parameters with each new data point")
print("- The learning rate controls the balance between stability and adaptability")
print("- Online learning is particularly valuable in scenarios with streaming data or changing patterns")
print("- The trade-offs between online and batch learning should be considered based on the specific problem requirements")

print(f"\nAll visualizations have been saved to: {save_dir}")
print("Generated images:")
print("- batch_vs_online.png: Comparison of batch and online learning approaches")
print("- lms_update.png: Visualization of the LMS update rule")
print("- parameter_update.png: Visualization of weight updates for the given example")
print("- learning_rate_tradeoffs.png: Impact of different learning rates")
print("- online_vs_batch.png: Convergence comparison between online and batch learning")
print("- fraud_detection_scenario.png: Real-world application of online learning") 