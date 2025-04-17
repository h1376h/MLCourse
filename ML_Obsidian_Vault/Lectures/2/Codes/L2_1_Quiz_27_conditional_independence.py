import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('default')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Introducing the problem
print_step_header(1, "Understanding Conditional Independence in ML")

print("""
Problem Statement:
In a machine learning context, consider three events:
- A: "The input feature X exceeds threshold t"
- B: "The model prediction is positive"
- C: "The true label is positive"

Tasks:
1. Explain what it means for events A and B to be conditionally independent given event C.
2. If A and B are conditionally independent given C, write the mathematical equation.
3. In a classification scenario, provide an example where conditional independence is reasonable.
4. How does conditional independence differ from marginal independence?
5. How might violating a conditional independence assumption impact model performance?
""")

# Step 2: Explaining conditional independence
print_step_header(2, "Defining Conditional Independence")

print("""
Conditional Independence Definition:
Two events A and B are conditionally independent given event C if knowing whether 
A occurred provides no additional information about the occurrence of B (and vice versa) 
once we already know that C has occurred.

Mathematical Definition:
Events A and B are conditionally independent given C if and only if:
P(A ∩ B | C) = P(A | C) × P(B | C)

Alternatively, this can be expressed as:
P(A | B, C) = P(A | C)
P(B | A, C) = P(B | C)
""")

# Step 3: Create a visual explanation of conditional independence
print_step_header(3, "Visual Representation of Conditional Independence")

# Create a visualization to explain conditional independence
plt.figure(figsize=(10, 6))

# Draw Venn diagram-like visualization
circle1 = plt.Circle((0.3, 0.6), 0.25, color='skyblue', alpha=0.5)
circle2 = plt.Circle((0.7, 0.6), 0.25, color='lightgreen', alpha=0.5)
circle3 = plt.Circle((0.5, 0.4), 0.35, color='salmon', alpha=0.3)

fig = plt.gcf()
ax = fig.gca()
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)

# Add text labels
plt.text(0.3, 0.6, 'A', ha='center', va='center', fontsize=14)
plt.text(0.7, 0.6, 'B', ha='center', va='center', fontsize=14)
plt.text(0.5, 0.4, 'C', ha='center', va='center', fontsize=14)

# Add annotations
plt.annotate('P(A|C)', xy=(0.3, 0.6), xytext=(0.1, 0.8),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7))
plt.annotate('P(B|C)', xy=(0.7, 0.6), xytext=(0.9, 0.8),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7))
plt.annotate('A ∩ B|C', xy=(0.5, 0.6), xytext=(0.5, 0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.7))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Conditional Independence: P(A ∩ B|C) = P(A|C) × P(B|C)', fontsize=14)
plt.axis('off')

# Add explanatory text box
textstr = '''Conditional Independence:
When A and B are conditionally 
independent given C, the overlap of A and B 
within C is exactly what we would expect 
if A and B were randomly distributed within C.'''
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.05, textstr, fontsize=10, verticalalignment='bottom', bbox=props)

file_path = os.path.join(save_dir, "conditional_independence_venn.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Simulating a scenario with conditional independence
print_step_header(4, "Simulating a Conditional Independence Scenario")

print("""
Simulation Scenario:
- A medical diagnosis system where:
  - A: Patient's age > 50 (feature exceeds threshold)
  - B: Model predicts disease (positive prediction)
  - C: Patient actually has the disease (true positive)

We'll simulate data to demonstrate when A and B are conditionally independent given C.
""")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 10000

# True disease status (C)
has_disease = np.random.binomial(1, 0.3, n_samples)  # 30% prevalence

# Age distribution - conditionally independent from model prediction given disease status
# Different for sick vs healthy patients
age = np.zeros(n_samples)
age[has_disease == 1] = np.random.normal(60, 10, np.sum(has_disease == 1))  # Older average age for sick
age[has_disease == 0] = np.random.normal(40, 15, np.sum(has_disease == 0))  # Younger average age for healthy

# Feature A: Age > 50
age_above_threshold = (age > 50).astype(int)

# Model predictions (B) depend only on disease status (C) in conditionally independent case
# To simulate conditional independence, prediction accuracy is the same regardless of age
prediction_accuracy = 0.8  # 80% accuracy

# Generate predictions based only on disease status (conditionally independent from age given disease)
predictions = np.zeros(n_samples)
for i in range(n_samples):
    if has_disease[i] == 1:
        # True positive rate (80%)
        predictions[i] = np.random.binomial(1, prediction_accuracy)
    else:
        # False positive rate (20%)
        predictions[i] = np.random.binomial(1, 1 - prediction_accuracy)

# Create arrays to store the data
data = np.column_stack((age, age_above_threshold, has_disease, predictions))

# Calculate joint and marginal probabilities
# P(A=1, B=1 | C=1)
disease_samples = data[data[:, 2] == 1]
p_a1_b1_given_c1 = np.sum((disease_samples[:, 1] == 1) & (disease_samples[:, 3] == 1)) / len(disease_samples)
# P(A=1 | C=1)
p_a1_given_c1 = np.sum(disease_samples[:, 1] == 1) / len(disease_samples)
# P(B=1 | C=1)
p_b1_given_c1 = np.sum(disease_samples[:, 3] == 1) / len(disease_samples)

# Product P(A=1|C=1) × P(B=1|C=1)
p_product = p_a1_given_c1 * p_b1_given_c1

print(f"For patients with disease (C=1):")
print(f"P(A=1, B=1 | C=1) = {p_a1_b1_given_c1:.4f}")
print(f"P(A=1 | C=1) = {p_a1_given_c1:.4f}")
print(f"P(B=1 | C=1) = {p_b1_given_c1:.4f}")
print(f"P(A=1 | C=1) × P(B=1 | C=1) = {p_product:.4f}")
print(f"Difference: {abs(p_a1_b1_given_c1 - p_product):.4f}")
print()

# Step 5: Visualizing the conditional independence
print_step_header(5, "Visualizing Conditional Independence")

# Prepare data for 2x2 contingency table
disease_samples = data[data[:, 2] == 1]
table = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        table[i, j] = np.sum((disease_samples[:, 1] == i) & (disease_samples[:, 3] == j))
table = table / np.sum(table)  # Normalize to get probabilities

# Create a heatmap for the contingency table
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(table, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks([0, 1], ['Neg Pred', 'Pos Pred'])
plt.yticks([0, 1], ['Age ≤ 50', 'Age > 50'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'{table[i, j]:.3f}', ha='center', va='center', color='black')
plt.title('Joint Probabilities P(A,B|C=1)\nFor Patients with Disease', fontsize=12)

# Create a comparison plot of expected vs. observed probabilities
plt.subplot(1, 2, 2)
x = np.arange(2)
y = [p_a1_b1_given_c1, p_product]
plt.bar(x, y, width=0.4, color=['skyblue', 'lightgreen'])
plt.xticks(x, ['P(A=1,B=1|C=1)', 'P(A=1|C=1)×P(B=1|C=1)'])
plt.ylim(0, max(y) * 1.2)
for i, value in enumerate(y):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
plt.title('Testing Conditional Independence\nGiven Disease Status (C=1)', fontsize=12)
plt.ylabel('Probability')
plt.tight_layout()

file_path = os.path.join(save_dir, "conditional_independence_heatmap.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Comparing with a Non-Conditionally Independent Scenario
print_step_header(6, "Comparing with Non-Conditionally Independent Scenario")

print("""
Now, let's modify our simulation to create a scenario where A and B are NOT conditionally independent given C.
In this case, the model's prediction accuracy will depend both on disease status AND age.
""")

# Create non-conditionally independent case (prediction depends on both disease and age)
predictions_dependent = np.zeros(n_samples)
for i in range(n_samples):
    if has_disease[i] == 1:
        if age[i] > 50:
            # Higher true positive rate for older patients (90%)
            predictions_dependent[i] = np.random.binomial(1, 0.9)
        else:
            # Lower true positive rate for younger patients (70%)
            predictions_dependent[i] = np.random.binomial(1, 0.7)
    else:
        if age[i] > 50:
            # Higher false positive rate for older patients (30%)
            predictions_dependent[i] = np.random.binomial(1, 0.3)
        else:
            # Lower false positive rate for younger patients (10%)
            predictions_dependent[i] = np.random.binomial(1, 0.1)

# Update data array
data_dep = np.column_stack((age, age_above_threshold, has_disease, predictions_dependent))

# Calculate joint and marginal probabilities for the non-conditionally independent case
disease_samples_dep = data_dep[data_dep[:, 2] == 1]
p_a1_b1_given_c1_dep = np.sum((disease_samples_dep[:, 1] == 1) & (disease_samples_dep[:, 3] == 1)) / len(disease_samples_dep)
p_a1_given_c1_dep = np.sum(disease_samples_dep[:, 1] == 1) / len(disease_samples_dep)
p_b1_given_c1_dep = np.sum(disease_samples_dep[:, 3] == 1) / len(disease_samples_dep)

# Product P(A=1|C=1) × P(B=1|C=1)
p_product_dep = p_a1_given_c1_dep * p_b1_given_c1_dep

print(f"For patients with disease (C=1) - Non-Conditionally Independent Case:")
print(f"P(A=1, B=1 | C=1) = {p_a1_b1_given_c1_dep:.4f}")
print(f"P(A=1 | C=1) = {p_a1_given_c1_dep:.4f}")
print(f"P(B=1 | C=1) = {p_b1_given_c1_dep:.4f}")
print(f"P(A=1 | C=1) × P(B=1 | C=1) = {p_product_dep:.4f}")
print(f"Difference: {abs(p_a1_b1_given_c1_dep - p_product_dep):.4f}")
print()

# Prepare contingency table for non-independent case
table_dep = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        table_dep[i, j] = np.sum((disease_samples_dep[:, 1] == i) & (disease_samples_dep[:, 3] == j))
table_dep = table_dep / np.sum(table_dep)  # Normalize to get probabilities

# Visualize the comparison
plt.figure(figsize=(12, 10))

# Compare Conditionally Independent vs. Non-Independent
plt.subplot(2, 2, 1)
plt.imshow(table, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks([0, 1], ['Neg Pred', 'Pos Pred'])
plt.yticks([0, 1], ['Age ≤ 50', 'Age > 50'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'{table[i, j]:.3f}', ha='center', va='center', color='black')
plt.title('Conditionally Independent Case\nJoint Probabilities P(A,B|C=1)', fontsize=12)

plt.subplot(2, 2, 2)
x = np.arange(2)
y = [p_a1_b1_given_c1, p_product]
plt.bar(x, y, width=0.4, color=['skyblue', 'lightgreen'])
plt.xticks(x, ['P(A=1,B=1|C=1)', 'P(A=1|C=1)×P(B=1|C=1)'])
plt.ylim(0, max(y) * 1.2)
for i, value in enumerate(y):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
plt.title('Testing Conditional Independence\nSmall Difference = Independent', fontsize=12)
plt.ylabel('Probability')

plt.subplot(2, 2, 3)
plt.imshow(table_dep, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks([0, 1], ['Neg Pred', 'Pos Pred'])
plt.yticks([0, 1], ['Age ≤ 50', 'Age > 50'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'{table_dep[i, j]:.3f}', ha='center', va='center', color='black')
plt.title('Non-Conditionally Independent Case\nJoint Probabilities P(A,B|C=1)', fontsize=12)

plt.subplot(2, 2, 4)
x = np.arange(2)
y = [p_a1_b1_given_c1_dep, p_product_dep]
plt.bar(x, y, width=0.4, color=['salmon', 'lightpink'])
plt.xticks(x, ['P(A=1,B=1|C=1)', 'P(A=1|C=1)×P(B=1|C=1)'])
plt.ylim(0, max(y) * 1.2)
for i, value in enumerate(y):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
plt.title('Testing Conditional Independence\nLarge Difference = Not Independent', fontsize=12)
plt.ylabel('Probability')

plt.tight_layout()
file_path = os.path.join(save_dir, "comparison_independence.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Comparing Marginal vs Conditional Independence
print_step_header(7, "Comparing Marginal vs Conditional Independence")

# Calculate marginal independence
p_a1 = np.sum(data[:, 1] == 1) / n_samples
p_b1 = np.sum(data[:, 3] == 1) / n_samples
p_a1_b1 = np.sum((data[:, 1] == 1) & (data[:, 3] == 1)) / n_samples

print(f"Testing Marginal Independence (Entire Population):")
print(f"P(A=1, B=1) = {p_a1_b1:.4f}")
print(f"P(A=1) = {p_a1:.4f}")
print(f"P(B=1) = {p_b1:.4f}")
print(f"P(A=1) × P(B=1) = {p_a1 * p_b1:.4f}")
print(f"Difference: {abs(p_a1_b1 - (p_a1 * p_b1)):.4f}")
print()

# Create visualization comparing marginal and conditional independence
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
x = np.arange(2)
y = [p_a1_b1, p_a1 * p_b1]
plt.bar(x, y, width=0.4, color=['teal', 'lightseagreen'])
plt.xticks(x, ['P(A=1,B=1)', 'P(A=1)×P(B=1)'])
plt.ylim(0, max(y) * 1.2)
for i, value in enumerate(y):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center')
plt.title('Testing Marginal Independence\nA: Age > 50, B: Positive Prediction', fontsize=12)
plt.ylabel('Probability')

plt.subplot(2, 1, 2)
# Prepare data for conditional independence comparison by disease status
no_disease_samples = data[data[:, 2] == 0]
p_a1_b1_given_c0 = np.sum((no_disease_samples[:, 1] == 1) & (no_disease_samples[:, 3] == 1)) / len(no_disease_samples)
p_a1_given_c0 = np.sum(no_disease_samples[:, 1] == 1) / len(no_disease_samples)
p_b1_given_c0 = np.sum(no_disease_samples[:, 3] == 1) / len(no_disease_samples)
p_product_c0 = p_a1_given_c0 * p_b1_given_c0

x_labels = ['C=0: No Disease', 'C=1: Has Disease']
ind = np.arange(len(x_labels))
width = 0.35

plt.bar(ind - width/2, [p_a1_b1_given_c0, p_a1_b1_given_c1], width, 
        label='P(A=1,B=1|C)')
plt.bar(ind + width/2, [p_product_c0, p_product], width, 
        label='P(A=1|C)×P(B=1|C)')
plt.xticks(ind, x_labels)
plt.ylabel('Probability')
plt.title('Conditional Independence Given Disease Status', fontsize=12)
plt.legend()

plt.tight_layout()
file_path = os.path.join(save_dir, "marginal_vs_conditional.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 8: Model Performance Impact
print_step_header(8, "Impact on Model Performance")

# Calculate simple performance metrics for both models
def calculate_metrics(predictions, true_labels):
    tp = np.sum((true_labels == 1) & (predictions == 1))
    fp = np.sum((true_labels == 0) & (predictions == 1))
    tn = np.sum((true_labels == 0) & (predictions == 0))
    fn = np.sum((true_labels == 1) & (predictions == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    confusion = np.array([[tn, fp], [fn, tp]])
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': confusion
    }

metrics_independent = calculate_metrics(data[:, 3], data[:, 2])
metrics_dependent = calculate_metrics(data_dep[:, 3], data_dep[:, 2])

print("Model Performance Comparison:")
print("\nConditionally Independent Model:")
print(f"Accuracy: {metrics_independent['Accuracy']:.4f}")
print(f"Precision: {metrics_independent['Precision']:.4f}")
print(f"Recall: {metrics_independent['Recall']:.4f}")
print(f"F1 Score: {metrics_independent['F1']:.4f}")
print("Confusion Matrix:")
print(metrics_independent['Confusion Matrix'])

print("\nNon-Conditionally Independent Model:")
print(f"Accuracy: {metrics_dependent['Accuracy']:.4f}")
print(f"Precision: {metrics_dependent['Precision']:.4f}")
print(f"Recall: {metrics_dependent['Recall']:.4f}")
print(f"F1 Score: {metrics_dependent['F1']:.4f}")
print("Confusion Matrix:")
print(metrics_dependent['Confusion Matrix'])

# Visualize performance comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
ind_values = [metrics_independent[m] for m in metrics_names]
dep_values = [metrics_dependent[m] for m in metrics_names]

x = np.arange(len(metrics_names))
width = 0.35
plt.bar(x - width/2, ind_values, width, label='Conditionally Independent Model')
plt.bar(x + width/2, dep_values, width, label='Non-Conditionally Independent Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics_names)
plt.legend()

plt.subplot(1, 2, 2)
def plot_confusions(ax, cm, title):
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Loop over data dimensions and create text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2. else "black")

subplot_positions = [223, 224]
for i, (metrics, title) in enumerate([(metrics_independent, 'Conditionally Independent Model'), 
                                     (metrics_dependent, 'Non-Conditionally Independent Model')]):
    plt.subplot(subplot_positions[i])
    plot_confusions(plt.gca(), metrics['Confusion Matrix'], title)

plt.tight_layout()
file_path = os.path.join(save_dir, "model_performance.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

print("\nQuestion 27 Solution Summary:")
print("1. Conditional independence between events A and B given C means that once we know C,")
print("   knowledge about A provides no additional information about B and vice versa.")
print("2. The mathematical equation is: P(A ∩ B | C) = P(A | C) × P(B | C)")
print("3. An example in classification: in a medical diagnosis system, patient's age exceeding a")
print("   threshold and model prediction might be conditionally independent given the true disease status.")
print("4. Conditional independence differs from marginal independence in that the former considers")
print("   the relationship between A and B only within subpopulations defined by C, while the latter")
print("   considers their relationship across the entire population.")
print("5. Violating conditional independence assumptions can lead to biased or suboptimal models,")
print("   as it indicates relevant feature interactions are being ignored, potentially resulting in")
print("   decreased performance as demonstrated in our simulation.") 