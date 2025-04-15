import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import PercentFormatter

print("\n=== CONDITIONAL PROBABILITY EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Set better aesthetics for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Medical Testing
print("Example 1: Medical Testing")
# Define probabilities
prevalence = 0.01  # P(D)
sensitivity = 0.95  # P(T|D)
specificity = 0.90  # P(¬T|¬D)
false_positive_rate = 1 - specificity  # P(T|¬D)

# Step-by-step calculation
print("\nStep-by-step calculation:")
print(f"Disease prevalence: P(D) = {prevalence}")
print(f"Test sensitivity: P(T|D) = {sensitivity}")
print(f"Test specificity: P(¬T|¬D) = {specificity}")
print(f"False positive rate: P(T|¬D) = {false_positive_rate}")

# Calculate the probability of getting a positive test
p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)
print(f"\nStep 1: Calculate P(T) using the law of total probability:")
print(f"P(T) = P(T|D)×P(D) + P(T|¬D)×P(¬D)")
print(f"P(T) = {sensitivity} × {prevalence} + {false_positive_rate} × {1-prevalence}")
print(f"P(T) = {sensitivity * prevalence:.4f} + {false_positive_rate * (1-prevalence):.4f}")
print(f"P(T) = {p_positive:.4f}")

# Calculate the posterior probability
posterior = (sensitivity * prevalence) / p_positive
print(f"\nStep 2: Calculate P(D|T) using Bayes' theorem:")
print(f"P(D|T) = [P(T|D)×P(D)] / P(T)")
print(f"P(D|T) = [{sensitivity} × {prevalence}] / {p_positive:.4f}")
print(f"P(D|T) = {sensitivity * prevalence:.4f} / {p_positive:.4f}")
print(f"P(D|T) = {posterior:.4f}")

print(f"\nTherefore, if a person tests positive, there's only a {posterior*100:.1f}% chance they actually have the disease.")

# Create a more academic visual representation for medical testing
fig, ax = plt.subplots(figsize=(12, 8))

# Create a 2x2 table for the confusion matrix
tp = sensitivity * prevalence
fn = (1 - sensitivity) * prevalence
fp = false_positive_rate * (1 - prevalence)
tn = specificity * (1 - prevalence)

# Create the table data
table_data = np.array([
    [tp, fp, tp + fp],
    [fn, tn, fn + tn],
    [tp + fn, fp + tn, 1.0]
])

# Row and column labels
row_labels = ['Positive Test', 'Negative Test', 'Total']
col_labels = ['Disease', 'No Disease', 'Total']

# Define cell colors
colors = np.array([
    ['#ffcccc', '#ffcccc', '#f2f2f2'],  # Light red for positive test
    ['#ccffcc', '#ccffcc', '#f2f2f2'],  # Light green for negative test
    ['#f2f2f2', '#f2f2f2', '#f2f2f2']   # Light gray for totals
])

# Create the table
the_table = plt.table(
    cellText=[[f'{val:.4f}' for val in row] for row in table_data],
    rowLabels=row_labels,
    colLabels=col_labels,
    cellColours=colors,
    cellLoc='center',
    loc='center'
)

# Modify table appearance
the_table.scale(1, 1.5)
the_table.set_fontsize(12)

# Add title and remove axes
ax.set_title('Medical Test Results by Disease Status', fontsize=16, pad=20)
ax.axis('off')

plt.savefig(os.path.join(images_dir, 'bayes_theorem_medical.png'), dpi=120, bbox_inches='tight')
plt.close()

# Example 2: Email Classification
print("\n\nExample 2: Email Classification")
# Define probabilities
prior_spam = 0.30  # P(S)
true_positive_rate = 0.92  # P(C|S)
false_positive_rate = 0.02  # P(C|¬S)

# Step-by-step calculation
print("\nStep-by-step calculation:")
print(f"Prior probability of spam: P(S) = {prior_spam}")
print(f"True positive rate: P(C|S) = {true_positive_rate}")
print(f"False positive rate: P(C|¬S) = {false_positive_rate}")

# Calculate the probability of getting a classified as spam
p_classified_spam = true_positive_rate * prior_spam + false_positive_rate * (1 - prior_spam)
print(f"\nStep 1: Calculate P(C) using the law of total probability:")
print(f"P(C) = P(C|S)×P(S) + P(C|¬S)×P(¬S)")
print(f"P(C) = {true_positive_rate} × {prior_spam} + {false_positive_rate} × {1-prior_spam}")
print(f"P(C) = {true_positive_rate * prior_spam:.4f} + {false_positive_rate * (1-prior_spam):.4f}")
print(f"P(C) = {p_classified_spam:.4f}")

# Calculate the posterior probability
posterior_spam = (true_positive_rate * prior_spam) / p_classified_spam
print(f"\nStep 2: Calculate P(S|C) using Bayes' theorem:")
print(f"P(S|C) = [P(C|S)×P(S)] / P(C)")
print(f"P(S|C) = [{true_positive_rate} × {prior_spam}] / {p_classified_spam:.4f}")
print(f"P(S|C) = {true_positive_rate * prior_spam:.4f} / {p_classified_spam:.4f}")
print(f"P(S|C) = {posterior_spam:.4f}")

print(f"\nTherefore, if an email is classified as spam, there's a {posterior_spam*100:.1f}% chance it actually is spam.")

# Create a more academic visual representation for email classification
fig, ax = plt.subplots(figsize=(12, 8))

# Create a 2x2 table for the confusion matrix
tp = true_positive_rate * prior_spam
fn = (1 - true_positive_rate) * prior_spam
fp = false_positive_rate * (1 - prior_spam)
tn = (1 - false_positive_rate) * (1 - prior_spam)

# Create the table data
table_data = np.array([
    [tp, fp, tp + fp],
    [fn, tn, fn + tn],
    [tp + fn, fp + tn, 1.0]
])

# Row and column labels
row_labels = ['Classified as Spam', 'Classified as Not Spam', 'Total']
col_labels = ['Actually Spam', 'Actually Not Spam', 'Total']

# Define cell colors
colors = np.array([
    ['#ffcccc', '#ffcccc', '#f2f2f2'],  # Light red for classified as spam
    ['#ccffcc', '#ccffcc', '#f2f2f2'],  # Light green for classified as not spam
    ['#f2f2f2', '#f2f2f2', '#f2f2f2']   # Light gray for totals
])

# Create the table
the_table = plt.table(
    cellText=[[f'{val:.4f}' for val in row] for row in table_data],
    rowLabels=row_labels,
    colLabels=col_labels,
    cellColours=colors,
    cellLoc='center',
    loc='center'
)

# Modify table appearance
the_table.scale(1, 1.5)
the_table.set_fontsize(12)

# Add title and remove axes
ax.set_title('Email Classification Results', fontsize=16, pad=20)
ax.axis('off')

plt.savefig(os.path.join(images_dir, 'bayes_theorem_email.png'), dpi=120, bbox_inches='tight')
plt.close()

# Example 3: Disease Diagnosis with Multiple Tests
print("\n\nExample 3: Disease Diagnosis with Multiple Tests")
# Define probabilities
prevalence = 0.001  # P(D)
test_a_sensitivity = 0.99  # P(A|D)
test_a_specificity = 0.95  # P(¬A|¬D)
test_a_false_positive = 1 - test_a_specificity  # P(A|¬D)
test_b_sensitivity = 0.95  # P(B|D)
test_b_specificity = 0.99  # P(¬B|¬D)
test_b_false_positive = 1 - test_b_specificity  # P(B|¬D)

# Step-by-step calculation
print("\nStep-by-step calculation:")
print(f"Disease prevalence: P(D) = {prevalence}")
print(f"Test A sensitivity: P(A|D) = {test_a_sensitivity}")
print(f"Test A false positive rate: P(A|¬D) = {test_a_false_positive}")
print(f"Test B sensitivity: P(B|D) = {test_b_sensitivity}")
print(f"Test B false positive rate: P(B|¬D) = {test_b_false_positive}")

# Calculate P(A∩B|D) and P(A∩B|¬D) assuming conditional independence
p_both_positive_given_d = test_a_sensitivity * test_b_sensitivity
p_both_positive_given_not_d = test_a_false_positive * test_b_false_positive

print(f"\nStep 1: Calculate probabilities assuming conditional independence:")
print(f"P(A∩B|D) = P(A|D) × P(B|D) = {test_a_sensitivity} × {test_b_sensitivity} = {p_both_positive_given_d:.4f}")
print(f"P(A∩B|¬D) = P(A|¬D) × P(B|¬D) = {test_a_false_positive} × {test_b_false_positive} = {p_both_positive_given_not_d:.4f}")

# Calculate P(A∩B) using law of total probability
p_both_positive = p_both_positive_given_d * prevalence + p_both_positive_given_not_d * (1 - prevalence)
print(f"\nStep 2: Calculate P(A∩B) using the law of total probability:")
print(f"P(A∩B) = P(A∩B|D)×P(D) + P(A∩B|¬D)×P(¬D)")
print(f"P(A∩B) = {p_both_positive_given_d} × {prevalence} + {p_both_positive_given_not_d} × {1-prevalence}")
print(f"P(A∩B) = {p_both_positive_given_d * prevalence:.6f} + {p_both_positive_given_not_d * (1-prevalence):.6f}")
print(f"P(A∩B) = {p_both_positive:.6f}")

# Calculate posterior probability using Bayes' theorem
posterior_both = (p_both_positive_given_d * prevalence) / p_both_positive
print(f"\nStep 3: Calculate P(D|A∩B) using Bayes' theorem:")
print(f"P(D|A∩B) = [P(A∩B|D)×P(D)] / P(A∩B)")
print(f"P(D|A∩B) = [{p_both_positive_given_d} × {prevalence}] / {p_both_positive:.6f}")
print(f"P(D|A∩B) = {p_both_positive_given_d * prevalence:.6f} / {p_both_positive:.6f}")
print(f"P(D|A∩B) = {posterior_both:.4f}")

print(f"\nTherefore, if a patient tests positive on both tests, there's a {posterior_both*100:.1f}% probability they have the disease.")

# Calculate single test posteriors for comparison
posterior_test_a = (test_a_sensitivity * prevalence) / (test_a_sensitivity * prevalence + test_a_false_positive * (1 - prevalence))
posterior_test_b = (test_b_sensitivity * prevalence) / (test_b_sensitivity * prevalence + test_b_false_positive * (1 - prevalence))

# Create a more academic visual representation for multiple tests
fig, ax = plt.subplots(figsize=(10, 8))

# Create bar chart showing the improvement in probability from prior to each test to both tests
probabilities = [prevalence, posterior_test_a, posterior_test_b, posterior_both]
labels = ['Prior P(D)', 'P(D|Test A+)', 'P(D|Test B+)', 'P(D|Test A+ & B+)']
colors = ['gray', 'lightblue', 'lightgreen', 'salmon']

bars = ax.bar(labels, probabilities, color=colors, width=0.6, alpha=0.8, edgecolor='black')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom', fontsize=12)

# Set y-axis to percentage format
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylim(0, min(1.0, posterior_both * 1.2))  # Set upper limit with some padding

# Add a horizontal line at 50% for reference
ax.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7)

# Add title and labels
ax.set_title('Power of Multiple Diagnostic Tests', fontsize=16, pad=20)
ax.set_ylabel('Probability of Disease', fontsize=14)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multiple_tests_bayes.png'), dpi=120, bbox_inches='tight')
plt.close()

print("\nAll conditional probability example images created successfully.") 