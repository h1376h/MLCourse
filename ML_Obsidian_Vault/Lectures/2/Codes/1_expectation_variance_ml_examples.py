import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

print("\n=== EXPECTATION, VARIANCE & MOMENTS IN ML: STEP-BY-STEP EXAMPLES ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "L2_1_ML")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Expectation in Model Evaluation
print("Example 1: Expectation in Model Evaluation")
print("===========================================\n")

# Simulate data for a regression problem
np.random.seed(42)
n_samples = 200

# Create feature X with 2 dimensions
X = np.random.randn(n_samples, 2)

# Create target y with nonlinear relationship + noise
y = 2 + 3 * X[:, 0]**2 - 0.5 * X[:, 1] + np.random.normal(0, 1, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Step 1: Train multiple models with different random seeds")
print("--------------------------------------------------------\n")

# Train multiple Random Forest models with different random states
n_models = 10
models = []
predictions = []
mse_scores = []

for i in range(n_models):
    # Create and train a model with a different random state
    model = RandomForestRegressor(n_estimators=100, random_state=i)
    model.fit(X_train, y_train)
    models.append(model)
    
    # Make predictions
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    
    print(f"Model {i+1}: MSE = {mse:.4f}")

# Convert predictions to a numpy array for easier calculations
predictions = np.array(predictions)

print("\nStep 2: Calculate the expected prediction and variance")
print("----------------------------------------------------\n")

# Calculate the expected prediction (ensemble mean) for each test instance
expected_prediction = np.mean(predictions, axis=0)

# Calculate the prediction variance for each test instance
prediction_variance = np.var(predictions, axis=0)

# Calculate overall metrics
avg_mse = np.mean(mse_scores)
ensemble_mse = mean_squared_error(y_test, expected_prediction)

print(f"Average Individual Model MSE: {avg_mse:.4f}")
print(f"Ensemble Model MSE: {ensemble_mse:.4f}")
print(f"Improvement: {(avg_mse - ensemble_mse) / avg_mse * 100:.2f}%")

print("\nExample of expectation and variance for the first 5 test samples:")
print("| Test # | True Value | E[Prediction] | Var[Prediction] |")
print("|--------|------------|---------------|-----------------|")
for i in range(5):
    print(f"| {i+1:6d} | {y_test[i]:10.4f} | {expected_prediction[i]:13.4f} | {prediction_variance[i]:15.4f} |")

# Visualization for expectation in model evaluation
plt.figure(figsize=(12, 6))

# Plot 1: Individual models vs Ensemble
plt.subplot(1, 2, 1)
for i in range(min(5, n_models)):  # Plot only first 5 models for clarity
    plt.scatter(y_test, predictions[i], alpha=0.3, label=f'Model {i+1}')
plt.scatter(y_test, expected_prediction, color='red', label='Ensemble (E[Prediction])')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Predictions')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Individual Models vs Ensemble')
plt.legend()

# Plot 2: Prediction Variance
plt.subplot(1, 2, 2)
plt.scatter(y_test, prediction_variance, alpha=0.7)
plt.xlabel('True Values')
plt.ylabel('Prediction Variance')
plt.title('Prediction Variance Across Models')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'model_evaluation_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Law of Total Expectation - Customer Lifetime Value
print("\nExample 2: Law of Total Expectation - Customer Lifetime Value")
print("===========================================================\n")

print("Problem: Calculate the expected Customer Lifetime Value (CLV) across customer segments")
print("-----------------------------------------------------------------------------------\n")

# Define customer segments and their corresponding probabilities
segments = ['High-Value', 'Medium-Value', 'Low-Value']
segment_probabilities = [0.2, 0.3, 0.5]  # P(Segment)

# Define expected CLV for each segment
expected_clv_given_segment = [1000, 500, 100]  # E[CLV|Segment]

print("Step 1: Define the problem")
print("-----------------------\n")

print("Given information:")
print("- Customer segments: High-Value, Medium-Value, Low-Value")
for i, segment in enumerate(segments):
    print(f"- P({segment}) = {segment_probabilities[i]}")
    print(f"- E[CLV|{segment}] = ${expected_clv_given_segment[i]}")

print("\nStep 2: Apply the Law of Total Expectation")
print("---------------------------------------\n")
print("E[CLV] = Σ E[CLV|Segment_i] × P(Segment_i)")

# Calculate expected CLV using the Law of Total Expectation
expected_clv = 0
calculation_steps = []

for i, segment in enumerate(segments):
    segment_contribution = expected_clv_given_segment[i] * segment_probabilities[i]
    expected_clv += segment_contribution
    calculation_steps.append(f"E[CLV|{segment}] × P({segment}) = ${expected_clv_given_segment[i]} × {segment_probabilities[i]} = ${segment_contribution:.2f}")

print("Calculation:")
for step in calculation_steps:
    print(f"  {step}")
print(f"E[CLV] = {' + '.join([f'${expected_clv_given_segment[i] * segment_probabilities[i]:.2f}' for i in range(len(segments))])} = ${expected_clv:.2f}")

# Create a visualization for the Law of Total Expectation
plt.figure(figsize=(12, 6))

# Plot 1: Segment Probabilities
plt.subplot(1, 3, 1)
plt.bar(segments, segment_probabilities, color='skyblue')
plt.title('Segment Probabilities')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, prob in enumerate(segment_probabilities):
    plt.text(i, prob + 0.02, f'{prob:.1f}', ha='center')

# Plot 2: Expected CLV per Segment
plt.subplot(1, 3, 2)
plt.bar(segments, expected_clv_given_segment, color='lightgreen')
plt.title('E[CLV|Segment]')
plt.ylabel('Expected CLV ($)')
for i, val in enumerate(expected_clv_given_segment):
    plt.text(i, val + 20, f'${val}', ha='center')

# Plot 3: Weighted Contributions to E[CLV]
plt.subplot(1, 3, 3)
weighted_clv = [expected_clv_given_segment[i] * segment_probabilities[i] for i in range(len(segments))]
plt.bar(segments, weighted_clv, color='salmon')
plt.title('Weighted Contribution to E[CLV]')
plt.ylabel('Contribution ($)')
for i, val in enumerate(weighted_clv):
    plt.text(i, val + 10, f'${val:.1f}', ha='center')
plt.axhline(y=expected_clv, color='red', linestyle='--', label=f'E[CLV] = ${expected_clv:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'law_total_expectation.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Law of Total Variance - A/B Testing with User Segments
print("\nExample 3: Law of Total Variance - A/B Testing with User Segments")
print("=================================================================\n")

print("Problem: Analyze variance in conversion rates across user segments in an A/B test")
print("-----------------------------------------------------------------------------\n")

# Define user segments
user_segments = ['New', 'Returning']
segment_probs = [0.6, 0.4]  # P(Segment)

# Define conditional conversion means for each segment
mean_conv_given_segment = [0.05, 0.12]  # E[Conversion|Segment]

# Define conditional conversion variances for each segment
# Using variance of Bernoulli: p(1-p)
var_conv_given_segment = [
    mean_conv_given_segment[0] * (1 - mean_conv_given_segment[0]),
    mean_conv_given_segment[1] * (1 - mean_conv_given_segment[1])
]

print("Step 1: Define the problem")
print("-----------------------\n")

print("Given information:")
print("- User segments: New users, Returning users")
for i, segment in enumerate(user_segments):
    print(f"- P({segment}) = {segment_probs[i]}")
    print(f"- E[Conversion|{segment}] = {mean_conv_given_segment[i]:.2f}")
    print(f"- Var[Conversion|{segment}] = {var_conv_given_segment[i]:.4f}")

print("\nStep 2: Calculate the overall expected conversion rate using Law of Total Expectation")
print("-----------------------------------------------------------------------------------\n")
print("E[Conversion] = Σ E[Conversion|Segment_i] × P(Segment_i)")

# Calculate overall expected conversion using the Law of Total Expectation
expected_conversion = 0
for i, segment in enumerate(user_segments):
    expected_conversion += mean_conv_given_segment[i] * segment_probs[i]

print(f"E[Conversion] = {mean_conv_given_segment[0]} × {segment_probs[0]} + {mean_conv_given_segment[1]} × {segment_probs[1]}")
print(f"             = {mean_conv_given_segment[0] * segment_probs[0]:.4f} + {mean_conv_given_segment[1] * segment_probs[1]:.4f}")
print(f"             = {expected_conversion:.4f}")

print("\nStep 3: Calculate variance using the Law of Total Variance")
print("-------------------------------------------------------\n")
print("Var[Conversion] = E[Var[Conversion|Segment]] + Var[E[Conversion|Segment]]")

# Calculate terms for the Law of Total Variance
# Term 1: E[Var[Conversion|Segment]]
expected_conditional_variance = 0
for i, segment in enumerate(user_segments):
    expected_conditional_variance += var_conv_given_segment[i] * segment_probs[i]

print(f"Term 1: E[Var[Conversion|Segment]]")
print(f"      = {var_conv_given_segment[0]} × {segment_probs[0]} + {var_conv_given_segment[1]} × {segment_probs[1]}")
print(f"      = {var_conv_given_segment[0] * segment_probs[0]:.6f} + {var_conv_given_segment[1] * segment_probs[1]:.6f}")
print(f"      = {expected_conditional_variance:.6f}")

# Term 2: Var[E[Conversion|Segment]]
variance_conditional_means = 0
for i, segment in enumerate(user_segments):
    variance_conditional_means += (mean_conv_given_segment[i] - expected_conversion)**2 * segment_probs[i]

print(f"\nTerm 2: Var[E[Conversion|Segment]]")
print(f"      = Σ (E[Conversion|Segment_i] - E[Conversion])² × P(Segment_i)")
print(f"      = ({mean_conv_given_segment[0]} - {expected_conversion:.4f})² × {segment_probs[0]} + ({mean_conv_given_segment[1]} - {expected_conversion:.4f})² × {segment_probs[1]}")
print(f"      = {(mean_conv_given_segment[0] - expected_conversion)**2:.6f} × {segment_probs[0]} + {(mean_conv_given_segment[1] - expected_conversion)**2:.6f} × {segment_probs[1]}")
print(f"      = {(mean_conv_given_segment[0] - expected_conversion)**2 * segment_probs[0]:.6f} + {(mean_conv_given_segment[1] - expected_conversion)**2 * segment_probs[1]:.6f}")
print(f"      = {variance_conditional_means:.6f}")

# Total variance
total_variance = expected_conditional_variance + variance_conditional_means

print(f"\nTotal variance: Var[Conversion] = {expected_conditional_variance:.6f} + {variance_conditional_means:.6f} = {total_variance:.6f}")

# Create a visualization for the Law of Total Variance
plt.figure(figsize=(14, 7))

# Plot 1: Segment probabilities
plt.subplot(2, 2, 1)
plt.bar(user_segments, segment_probs, color='skyblue')
plt.title('Segment Probabilities')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, prob in enumerate(segment_probs):
    plt.text(i, prob + 0.02, f'{prob:.1f}', ha='center')

# Plot 2: Conditional conversion rates
plt.subplot(2, 2, 2)
plt.bar(user_segments, mean_conv_given_segment, color='lightgreen')
plt.title('E[Conversion|Segment]')
plt.ylabel('Expected Conversion Rate')
plt.ylim(0, max(mean_conv_given_segment) * 1.2)
for i, val in enumerate(mean_conv_given_segment):
    plt.text(i, val + 0.005, f'{val:.2f}', ha='center')
plt.axhline(y=expected_conversion, color='red', linestyle='--', label=f'E[Conversion] = {expected_conversion:.4f}')
plt.legend()

# Plot 3: Conditional variances
plt.subplot(2, 2, 3)
plt.bar(user_segments, var_conv_given_segment, color='salmon')
plt.title('Var[Conversion|Segment]')
plt.ylabel('Conditional Variance')
for i, val in enumerate(var_conv_given_segment):
    plt.text(i, val + 0.001, f'{val:.4f}', ha='center')
plt.axhline(y=expected_conditional_variance, color='red', linestyle='--', label=f'E[Var[X|Y]] = {expected_conditional_variance:.6f}')
plt.legend()

# Plot 4: Law of Total Variance components
plt.subplot(2, 2, 4)
components = [expected_conditional_variance, variance_conditional_means]
plt.bar(['E[Var[X|Y]]', 'Var[E[X|Y]]'], components, color=['lightblue', 'lightgreen'])
plt.title('Law of Total Variance Components')
plt.ylabel('Variance Component')
for i, val in enumerate(components):
    plt.text(i, val + 0.001, f'{val:.6f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'law_total_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Statistical Moments in ML Anomaly Detection
print("\nExample 4: Statistical Moments in ML Anomaly Detection")
print("=====================================================\n")

print("Problem: Using higher-order moments for anomaly detection in network traffic data")
print("-----------------------------------------------------------------------------\n")

# Generate synthetic network traffic data (packets per second)
np.random.seed(42)
n_samples = 1000

# Normal traffic (lognormal distribution)
normal_traffic = np.random.lognormal(mean=3, sigma=0.5, size=n_samples)

# Anomalous traffic (mixture of normal and spikes)
anomalous_traffic = np.copy(normal_traffic)
# Add some spikes (anomalies)
anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
anomalous_traffic[anomaly_indices] *= np.random.uniform(3, 10, size=50)

print("Step 1: Calculate moments for normal and anomalous traffic")
print("-------------------------------------------------------\n")

# Calculate first four moments for normal traffic
normal_mean = np.mean(normal_traffic)
normal_variance = np.var(normal_traffic)
normal_skewness = stats.skew(normal_traffic)
normal_kurtosis = stats.kurtosis(normal_traffic)  # Excess kurtosis (normal = 0)

# Calculate first four moments for anomalous traffic
anomalous_mean = np.mean(anomalous_traffic)
anomalous_variance = np.var(anomalous_traffic)
anomalous_skewness = stats.skew(anomalous_traffic)
anomalous_kurtosis = stats.kurtosis(anomalous_traffic)

print("Normal Traffic Moments:")
print(f"- Mean (1st moment): {normal_mean:.2f}")
print(f"- Variance (2nd central moment): {normal_variance:.2f}")
print(f"- Skewness (3rd standardized moment): {normal_skewness:.2f}")
print(f"- Kurtosis (4th standardized moment): {normal_kurtosis:.2f}")

print("\nAnomalous Traffic Moments:")
print(f"- Mean (1st moment): {anomalous_mean:.2f}")
print(f"- Variance (2nd central moment): {anomalous_variance:.2f}")
print(f"- Skewness (3rd standardized moment): {anomalous_skewness:.2f}")
print(f"- Kurtosis (4th standardized moment): {anomalous_kurtosis:.2f}")

print("\nStep 2: Analyzing moment changes due to anomalies")
print("------------------------------------------------\n")

print(f"Mean increase: {(anomalous_mean - normal_mean) / normal_mean * 100:.2f}%")
print(f"Variance increase: {(anomalous_variance - normal_variance) / normal_variance * 100:.2f}%")
print(f"Skewness increase: {(anomalous_skewness - normal_skewness) / abs(normal_skewness) * 100:.2f}%")
print(f"Kurtosis increase: {(anomalous_kurtosis - normal_kurtosis) / abs(normal_kurtosis) * 100:.2f}%")

print("\nStep 3: Use moments for anomaly detection")
print("-----------------------------------------\n")

# Create a simple anomaly detector based on rolling window statistics
window_size = 50
anomaly_scores = []

# Define the threshold multipliers for each moment
mean_threshold = 1.5
var_threshold = 2.0
skew_threshold = 2.0
kurt_threshold = 2.0

for i in range(window_size, len(anomalous_traffic)):
    window = anomalous_traffic[i-window_size:i]
    
    # Calculate moments in the current window
    window_mean = np.mean(window)
    window_var = np.var(window)
    window_skew = stats.skew(window)
    window_kurt = stats.kurtosis(window)
    
    # Calculate a composite anomaly score
    mean_score = abs(window_mean - normal_mean) / normal_mean
    var_score = abs(window_var - normal_variance) / normal_variance
    skew_score = abs(window_skew - normal_skewness) / max(1e-10, abs(normal_skewness))
    kurt_score = abs(window_kurt - normal_kurtosis) / max(1e-10, abs(normal_kurtosis))
    
    # Combine scores with weights emphasizing higher moments
    composite_score = (mean_score + 
                       var_score * 1.5 + 
                       skew_score * 2.0 + 
                       kurt_score * 2.5)
    
    anomaly_scores.append(composite_score)

# Determine a threshold for anomaly detection
threshold = np.percentile(anomaly_scores, 95)
detections = np.array(anomaly_scores) > threshold

# Calculate detection accuracy
true_anomalies = np.zeros(len(anomaly_scores), dtype=bool)
for idx in anomaly_indices:
    if idx >= window_size:
        true_anomalies[idx - window_size] = True

# Calculate precision and recall
true_positives = np.sum(detections & true_anomalies)
false_positives = np.sum(detections & ~true_anomalies)
false_negatives = np.sum(~detections & true_anomalies)

precision = true_positives / max(1, true_positives + false_positives)
recall = true_positives / max(1, true_positives + false_negatives)

print(f"Anomaly detection performance:")
print(f"- Precision: {precision:.2f}")
print(f"- Recall: {recall:.2f}")
print(f"- F1-score: {2 * precision * recall / max(1e-10, precision + recall):.2f}")

# Create visualizations for the statistical moments anomaly detection
plt.figure(figsize=(15, 10))

# Plot 1: Normal vs Anomalous Traffic
plt.subplot(2, 2, 1)
plt.hist(normal_traffic, bins=50, alpha=0.5, label='Normal Traffic')
plt.hist(anomalous_traffic, bins=50, alpha=0.5, label='Anomalous Traffic')
plt.xlabel('Packets per Second')
plt.ylabel('Frequency')
plt.title('Network Traffic Distribution')
plt.legend()

# Plot 2: Traffic Time Series with Anomalies
plt.subplot(2, 2, 2)
plt.plot(anomalous_traffic, color='blue', alpha=0.6, label='Traffic')
plt.scatter(anomaly_indices, anomalous_traffic[anomaly_indices], color='red', label='True Anomalies')
plt.xlabel('Time')
plt.ylabel('Packets per Second')
plt.title('Network Traffic with Anomalies')
plt.legend()

# Plot 3: Anomaly Scores
plt.subplot(2, 2, 3)
plt.plot(anomaly_scores, color='purple')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.xlabel('Time')
plt.ylabel('Anomaly Score')
plt.title('Composite Moment-based Anomaly Score')
plt.legend()

# Plot 4: Moment Comparison
plt.subplot(2, 2, 4)
moments = ['Mean', 'Variance', 'Skewness', 'Kurtosis']
normal_moments = [normal_mean, normal_variance, normal_skewness, normal_kurtosis]
anomalous_moments = [anomalous_mean, anomalous_variance, anomalous_skewness, anomalous_kurtosis]

# Normalize for better visualization
normal_normalized = [normal_moments[i]/max(abs(normal_moments[i]), abs(anomalous_moments[i])) for i in range(4)]
anomalous_normalized = [anomalous_moments[i]/max(abs(normal_moments[i]), abs(anomalous_moments[i])) for i in range(4)]

x = np.arange(len(moments))
width = 0.35
plt.bar(x - width/2, normal_normalized, width, label='Normal Traffic', color='blue', alpha=0.6)
plt.bar(x + width/2, anomalous_normalized, width, label='Anomalous Traffic', color='red', alpha=0.6)
plt.xlabel('Statistical Moment')
plt.ylabel('Normalized Value')
plt.title('Normalized Moments Comparison')
plt.xticks(x, moments)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'statistical_moments_anomaly.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll expectation, variance, and moments example images created successfully.") 