import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_3_Quiz_38")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 38: SVM FRAUD DETECTION WITH POLYNOMIAL KERNEL")
print("=" * 80)

# Given parameters
print("\n1. GIVEN PARAMETERS:")
print("-" * 50)

# Support vectors
support_vectors = np.array([
    [5.2, 14.5],  # SV1: fraud
    [8.1, 23.2],  # SV2: fraud  
    [1.8, 9.3],   # SV3: legitimate
    [12.5, 16.8]  # SV4: legitimate
])

labels = np.array([+1, +1, -1, -1])  # +1 for fraud, -1 for legitimate
alphas = np.array([1.2, 0.9, 0.8, 0.6])

# Kernel parameters
gamma = 0.1
r = 1
d = 3

print(f"Support Vectors:")
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    print(f"  SV{i+1}: x^({i+1}) = ({sv[0]:.1f}, {sv[1]:.1f}), y^({i+1}) = {label:+d}, α_{i+1} = {alpha:.1f}")

print(f"\nKernel Parameters:")
print(f"  γ = {gamma}")
print(f"  r = {r}")
print(f"  d = {d}")

# Polynomial kernel function
def polynomial_kernel(x1, x2, gamma=0.1, r=1, d=3):
    """Compute polynomial kernel k(x1, x2) = (γ⟨x1, x2⟩ + r)^d"""
    dot_product = np.dot(x1, x2)
    return (gamma * dot_product + r) ** d

print("\n2. POLYNOMIAL KERNEL FUNCTION:")
print("-" * 50)
print(f"k(x^(i), x) = (γ⟨x^(i), x⟩ + r)^d")
print(f"k(x^(i), x) = ({gamma}⟨x^(i), x⟩ + {r})^{d}")

# Step 1: Calculate bias term w0
print("\n3. CALCULATING BIAS TERM w0:")
print("-" * 50)

# Using support vector x^(s) = (5.2, 14.5) with y^(s) = +1
x_s = np.array([5.2, 14.5])
y_s = +1

print(f"Using support vector x^(s) = ({x_s[0]:.1f}, {x_s[1]:.1f}) with y^(s) = {y_s:+d}")

# Calculate kernel values k(x^(n), x^(s)) for all support vectors
kernel_values = []
print(f"\nCalculating kernel values k(x^(n), x^(s)):")
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    k_val = polynomial_kernel(sv, x_s, gamma, r, d)
    kernel_values.append(k_val)
    print(f"  k(x^({i+1}), x^(s)) = ({gamma}⟨({sv[0]:.1f}, {sv[1]:.1f}), ({x_s[0]:.1f}, {x_s[1]:.1f})⟩ + {r})^{d}")
    print(f"  k(x^({i+1}), x^(s)) = ({gamma}×({sv[0]:.1f}×{x_s[0]:.1f} + {sv[1]:.1f}×{x_s[1]:.1f}) + {r})^{d}")
    print(f"  k(x^({i+1}), x^(s)) = ({gamma}×({sv[0]*x_s[0]:.2f} + {sv[1]*x_s[1]:.2f}) + {r})^{d}")
    print(f"  k(x^({i+1}), x^(s)) = ({gamma}×{sv[0]*x_s[0] + sv[1]*x_s[1]:.2f} + {r})^{d}")
    print(f"  k(x^({i+1}), x^(s)) = ({gamma*(sv[0]*x_s[0] + sv[1]*x_s[1]) + r:.3f})^{d}")
    print(f"  k(x^({i+1}), x^(s)) = {k_val:.6f}")

# Calculate w0 using the formula
print(f"\nCalculating w0 using the formula:")
print(f"w0 = y^(s) - Σ(α_n > 0) α_n y^(n) k(x^(n), x^(s))")

sum_term = 0
print(f"\nSummation term:")
for i, (label, alpha, k_val) in enumerate(zip(labels, alphas, kernel_values)):
    if alpha > 0:  # Only consider support vectors with α > 0
        term = alpha * label * k_val
        sum_term += term
        print(f"  α_{i+1} × y^({i+1}) × k(x^({i+1}), x^(s)) = {alpha:.1f} × {label:+d} × {k_val:.6f} = {term:.6f}")

w0 = y_s - sum_term
print(f"\nw0 = {y_s:+d} - {sum_term:.6f} = {w0:.6f}")

# Step 2: Classify new transaction
print("\n4. CLASSIFYING NEW TRANSACTION:")
print("-" * 50)

# New transaction to classify
x_new = np.array([7.5, 22.0])
print(f"New transaction: x = ({x_new[0]:.1f}, {x_new[1]:.1f})")

# Calculate decision function
print(f"\nDecision function:")
print(f"ŷ = sign(w0 + Σ(α_n > 0) α_n y^(n) k(x^(n), x))")

# Calculate kernel values for new transaction
kernel_values_new = []
print(f"\nCalculating kernel values k(x^(n), x):")
for i, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
    k_val = polynomial_kernel(sv, x_new, gamma, r, d)
    kernel_values_new.append(k_val)
    print(f"  k(x^({i+1}), x) = {k_val:.6f}")

# Calculate decision score
decision_score = w0
print(f"\nDecision score calculation:")
print(f"Score = w0 + Σ(α_n > 0) α_n y^(n) k(x^(n), x)")
print(f"Score = {w0:.6f}")

for i, (label, alpha, k_val) in enumerate(zip(labels, alphas, kernel_values_new)):
    if alpha > 0:
        term = alpha * label * k_val
        decision_score += term
        print(f"  + α_{i+1} × y^({i+1}) × k(x^({i+1}), x) = {alpha:.1f} × {label:+d} × {k_val:.6f} = {term:.6f}")
        print(f"  Score = {decision_score:.6f}")

# Make prediction
prediction = np.sign(decision_score)
print(f"\nFinal decision score: {decision_score:.6f}")
print(f"Prediction: ŷ = sign({decision_score:.6f}) = {prediction:+1.0f}")
print(f"Classification: {'FRAUD' if prediction > 0 else 'LEGITIMATE'}")

# Step 3: Fraud probability score interpretation
print("\n5. FRAUD PROBABILITY SCORE INTERPRETATION:")
print("-" * 50)
print(f"Fraud probability score: {decision_score:.6f}")
print(f"Magnitude: {abs(decision_score):.6f}")

if abs(decision_score) > 1.0:
    confidence = "HIGH"
elif abs(decision_score) > 0.5:
    confidence = "MEDIUM"
else:
    confidence = "LOW"

print(f"Confidence level: {confidence}")
print(f"Interpretation: The model is {confidence.lower()}ly confident in its prediction.")

# Step 4: Performance metrics calculation
print("\n6. PERFORMANCE METRICS CALCULATION:")
print("-" * 50)

# Given parameters
total_transactions = 10000
fraud_rate = 0.005  # 0.5%
flagged_transactions = 150
review_cost = 50
fraud_loss = 10000

# Calculate expected values
expected_fraudulent = total_transactions * fraud_rate
expected_legitimate = total_transactions - expected_fraudulent

print(f"Daily transaction volume: {total_transactions:,}")
print(f"Expected fraudulent transactions: {expected_fraudulent:.1f}")
print(f"Expected legitimate transactions: {expected_legitimate:.1f}")
print(f"Flagged transactions: {flagged_transactions}")

# Assuming the model flags all fraudulent transactions and some legitimate ones
# Let's assume the model has high recall (catches most fraud)
assumed_recall = 0.95  # 95% of fraud is caught
true_positives = expected_fraudulent * assumed_recall
false_positives = flagged_transactions - true_positives

precision = true_positives / flagged_transactions if flagged_transactions > 0 else 0
recall = true_positives / expected_fraudulent if expected_fraudulent > 0 else 0

print(f"\nAssumed recall: {assumed_recall:.1%}")
print(f"True positives (fraud caught): {true_positives:.1f}")
print(f"False positives (legitimate flagged): {false_positives:.1f}")
print(f"Precision: {precision:.3f} ({precision:.1%})")
print(f"Recall: {recall:.3f} ({recall:.1%})")

# Cost analysis
daily_review_cost = flagged_transactions * review_cost
fraud_prevention_savings = true_positives * fraud_loss
net_savings = fraud_prevention_savings - daily_review_cost

print(f"\nCost Analysis:")
print(f"Daily review cost: {flagged_transactions} × ${review_cost} = ${daily_review_cost:,}")
print(f"Fraud prevention savings: {true_positives:.1f} × ${fraud_loss:,} = ${fraud_prevention_savings:,.0f}")
print(f"Net daily savings: ${fraud_prevention_savings:,.0f} - ${daily_review_cost:,} = ${net_savings:,.0f}")

# Step 5: Threshold adjustment analysis
print("\n7. THRESHOLD ADJUSTMENT ANALYSIS:")
print("-" * 50)

new_threshold = 0.5
print(f"New confidence threshold: {new_threshold}")

# For this analysis, we need to estimate how many transactions would have scores above 0.5
# Let's assume a distribution of scores and estimate the impact

# Simulate some transaction scores to estimate the distribution
np.random.seed(42)
n_samples = 10000

# Generate synthetic scores for demonstration
# In practice, you would use actual model predictions
legitimate_scores = np.random.normal(-0.8, 0.5, int(expected_legitimate))
fraudulent_scores = np.random.normal(1.2, 0.3, int(expected_fraudulent))

all_scores = np.concatenate([legitimate_scores, fraudulent_scores])
scores_above_threshold = np.sum(all_scores > new_threshold)

print(f"Estimated transactions above threshold {new_threshold}: {scores_above_threshold}")

# Calculate new metrics
new_review_cost = scores_above_threshold * review_cost
new_fraud_prevention_savings = true_positives * fraud_loss  # Assuming same recall
new_net_savings = new_fraud_prevention_savings - new_review_cost

print(f"\nNew Cost Analysis:")
print(f"New daily review cost: {scores_above_threshold} × ${review_cost} = ${new_review_cost:,}")
print(f"Fraud prevention savings: {true_positives:.1f} × ${fraud_loss:,} = ${new_fraud_prevention_savings:,.0f}")
print(f"New net daily savings: ${new_fraud_prevention_savings:,.0f} - ${new_review_cost:,} = ${new_net_savings:,.0f}")

cost_effectiveness = new_net_savings - net_savings
print(f"\nCost-effectiveness change: ${cost_effectiveness:,.0f}")
print(f"Recommendation: {'ADOPT' if cost_effectiveness > 0 else 'REJECT'} the new threshold")

# Create visualizations
print("\n8. CREATING VISUALIZATIONS:")
print("-" * 50)

# Visualization 1: Support vectors and decision boundary
plt.figure(figsize=(12, 10))

# Plot support vectors
colors = ['red' if label == 1 else 'blue' for label in labels]
markers = ['o' if label == 1 else 's' for label in labels]

for i, (sv, label, alpha, color, marker) in enumerate(zip(support_vectors, labels, alphas, colors, markers)):
    plt.scatter(sv[0], sv[1], s=200, c=color, marker=marker, 
                edgecolors='black', linewidth=2, alpha=0.7,
                label=f'SV{i+1}: {"Fraud" if label == 1 else "Legitimate"} ($\\alpha_{i+1}={alpha:.1f}$)')

# Plot new transaction
plt.scatter(x_new[0], x_new[1], s=300, c='green', marker='*', 
            edgecolors='black', linewidth=2, label=f'New Transaction (Score={decision_score:.3f})')

# Create decision boundary visualization
x1_min, x1_max = 0, 15
x2_min, x2_max = 5, 25
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))

# Calculate decision scores for grid points
Z = np.zeros_like(xx1)
for i in range(xx1.shape[0]):
    for j in range(xx1.shape[1]):
        x_grid = np.array([xx1[i, j], xx2[i, j]])
        score = w0
        for k, (sv, label, alpha) in enumerate(zip(support_vectors, labels, alphas)):
            if alpha > 0:
                k_val = polynomial_kernel(sv, x_grid, gamma, r, d)
                score += alpha * label * k_val
        Z[i, j] = score

# Plot decision boundary
contour = plt.contour(xx1, xx2, Z, levels=[0], colors='green', linewidths=3)
plt.contourf(xx1, xx2, Z, levels=[-10, 0, 10], colors=['lightblue', 'lightpink'], alpha=0.3)
# Add decision boundary to legend manually
plt.plot([], [], color='green', linewidth=3, label='Decision Boundary')

plt.xlabel('Transaction Amount (thousands of dollars)')
plt.ylabel('Time of Day (hours from midnight)')
plt.title('SVM Fraud Detection: Support Vectors and Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# Add annotations
for i, (sv, label) in enumerate(zip(support_vectors, labels)):
    plt.annotate(f'({sv[0]:.1f}, {sv[1]:.1f})', 
                 (sv[0], sv[1]), xytext=(5, 5), textcoords='offset points')

plt.savefig(os.path.join(save_dir, 'svm_fraud_detection_decision_boundary.png'), 
            dpi=300, bbox_inches='tight')

# Visualization 2: Kernel values heatmap
plt.figure(figsize=(10, 8))

# Calculate kernel matrix
kernel_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        kernel_matrix[i, j] = polynomial_kernel(support_vectors[i], support_vectors[j], gamma, r, d)

# Create heatmap
sns.heatmap(kernel_matrix, annot=True, fmt='.4f', cmap='viridis',
            xticklabels=[f'SV{i+1}' for i in range(4)],
            yticklabels=[f'SV{i+1}' for i in range(4)])
plt.title('Polynomial Kernel Matrix $K(\\mathbf{x}^{(i)}, \\mathbf{x}^{(j)})$')
plt.xlabel('Support Vector $j$')
plt.ylabel('Support Vector $i$')

plt.savefig(os.path.join(save_dir, 'polynomial_kernel_matrix.png'), 
            dpi=300, bbox_inches='tight')

# Visualization 3: Cost-benefit analysis
plt.figure(figsize=(12, 8))

# Data for cost-benefit analysis
scenarios = ['Current\nThreshold', 'New\nThreshold\n(0.5)']
review_costs = [daily_review_cost, new_review_cost]
fraud_savings = [fraud_prevention_savings, new_fraud_prevention_savings]
net_savings_values = [net_savings, new_net_savings]

x = np.arange(len(scenarios))
width = 0.25

plt.bar(x - width, review_costs, width, label='Review Cost', color='red', alpha=0.7)
plt.bar(x, fraud_savings, width, label='Fraud Prevention Savings', color='green', alpha=0.7)
plt.bar(x + width, net_savings_values, width, label='Net Savings', color='blue', alpha=0.7)

plt.xlabel('Threshold Scenario')
plt.ylabel('Daily Cost/Savings (\\$)')
plt.title('Cost-Benefit Analysis: Current vs New Threshold')
plt.xticks(x, scenarios)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (cost, savings, net) in enumerate(zip(review_costs, fraud_savings, net_savings_values)):
    plt.text(i - width, cost + 1000, f'\\${cost:,.0f}', ha='center', va='bottom')
    plt.text(i, savings + 1000, f'\\${savings:,.0f}', ha='center', va='bottom')
    plt.text(i + width, net + 1000, f'\\${net:,.0f}', ha='center', va='bottom')

plt.savefig(os.path.join(save_dir, 'cost_benefit_analysis.png'), 
            dpi=300, bbox_inches='tight')

# Visualization 4: Score distribution
plt.figure(figsize=(10, 6))

# Plot score distributions
plt.hist(legitimate_scores, bins=30, alpha=0.7, label='Legitimate Transactions', 
         color='blue', density=True)
plt.hist(fraudulent_scores, bins=30, alpha=0.7, label='Fraudulent Transactions', 
         color='red', density=True)

# Add threshold lines
plt.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Current Threshold (0)')
plt.axvline(x=new_threshold, color='orange', linestyle='--', linewidth=2, label=f'New Threshold ({new_threshold})')

plt.xlabel('Decision Score')
plt.ylabel('Density')
plt.title('Distribution of SVM Decision Scores')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(save_dir, 'score_distribution.png'), 
            dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Bias term w0 = {w0:.6f}")
print(f"2. New transaction ({x_new[0]:.1f}, {x_new[1]:.1f}) classified as: {'FRAUD' if prediction > 0 else 'LEGITIMATE'}")
print(f"3. Decision score: {decision_score:.6f} (confidence: {confidence})")
print(f"4. Current precision: {precision:.1%}")
print(f"5. Current net daily savings: ${net_savings:,.0f}")
print(f"6. New threshold would flag {scores_above_threshold} transactions")
print(f"7. Cost-effectiveness change: ${cost_effectiveness:,.0f}")
print("=" * 80)
