import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import beta

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_5_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Disease prevalence: 1% of the population")
print("- Test sensitivity (true positive rate): 95%")
print("- Test specificity (true negative rate): 90%")
print()
print("We need to calculate:")
print("1. Probability of disease given a positive test result")
print("2. Updated probability after a second positive test")
print("3. Number of positive tests needed for 95% disease probability")
print("4. Effect of disease prevalence on test interpretation")
print()

# Step 2: Define variables and Bayes' theorem
print_step_header(2, "Applying Bayes' Theorem")

# Define variables
prevalence = 0.01  # P(D) - prior probability of disease
sensitivity = 0.95  # P(+|D) - probability of positive test given disease
specificity = 0.90  # P(-|¬D) - probability of negative test given no disease
false_positive_rate = 1 - specificity  # P(+|¬D) - probability of positive test given no disease

# Calculate probability of a positive test: P(+) = P(+|D)P(D) + P(+|¬D)P(¬D)
p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)

# Apply Bayes' theorem: P(D|+) = P(+|D)P(D) / P(+)
p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"Prior probability of disease (prevalence): P(D) = {prevalence:.4f}")
print(f"Sensitivity (true positive rate): P(+|D) = {sensitivity:.4f}")
print(f"Specificity (true negative rate): P(-|¬D) = {specificity:.4f}")
print(f"False positive rate: P(+|¬D) = {false_positive_rate:.4f}")
print()
print(f"Probability of a positive test: P(+) = {p_positive:.4f}")
print(f"Probability of disease given a positive test: P(D|+) = {p_disease_given_positive:.4f}")
print(f"This means that despite a positive test, there's only a {p_disease_given_positive*100:.2f}% chance the person has the disease!")
print()

# Create visualization of Bayes' theorem application
fig, ax = plt.subplots(figsize=(12, 7))

# Population sizes for visualization
population_size = 10000
disease_count = int(population_size * prevalence)
no_disease_count = population_size - disease_count

# Test results
true_positives = int(disease_count * sensitivity)
false_negatives = disease_count - true_positives
false_positives = int(no_disease_count * false_positive_rate)
true_negatives = no_disease_count - false_positives

# Calculate PPV
ppv = true_positives / (true_positives + false_positives)

# Create diagram data
categories = ['Disease\n(1%)', 'No Disease\n(99%)']
values = [disease_count, no_disease_count]
colors = ['#ff9999', '#66b3ff']

# First level: Disease prevalence
ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)

# Second level: Test results for disease group
test_categories = ['Disease &\nTest +', 'Disease &\nTest -']
test_values = [true_positives, false_negatives]
test_colors = ['#ff5555', '#ffcccc']

bar_width = 0.4
ax.bar([categories[0]], [true_positives], width=bar_width, 
       bottom=[0], color=test_colors[0], edgecolor='black', alpha=0.7)
ax.bar([categories[0]], [false_negatives], width=bar_width, 
       bottom=[true_positives], color=test_colors[1], edgecolor='black', alpha=0.7)

# Second level: Test results for no disease group
test_categories_no = ['No Disease &\nTest +', 'No Disease &\nTest -']
test_values_no = [false_positives, true_negatives]
test_colors_no = ['#3399ff', '#99ccff']

ax.bar([categories[1]], [false_positives], width=bar_width, 
       bottom=[0], color=test_colors_no[0], edgecolor='black', alpha=0.7)
ax.bar([categories[1]], [true_negatives], width=bar_width, 
       bottom=[false_positives], color=test_colors_no[1], edgecolor='black', alpha=0.7)

# Add annotations
ax.text(0, disease_count/2, f"D+\n{disease_count}", ha='center', va='center', fontweight='bold')
ax.text(1, no_disease_count/2, f"D-\n{no_disease_count}", ha='center', va='center', fontweight='bold')

ax.text(0, true_positives/2, f"TP\n{true_positives}", ha='center', va='center')
ax.text(0, true_positives + false_negatives/2, f"FN\n{false_negatives}", ha='center', va='center')

ax.text(1, false_positives/2, f"FP\n{false_positives}", ha='center', va='center')
ax.text(1, false_positives + true_negatives/2, f"TN\n{true_negatives}", ha='center', va='center')

# Draw attention to the test positives
plt.annotate('All test positives', xy=(0.5, false_positives + 50), 
             xytext=(0.5, false_positives + 1000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             horizontalalignment='center')

# Highlight PPV calculation
plt.text(0.5, false_positives + 1500, 
         f"PPV = TP / (TP + FP) = {true_positives} / {true_positives + false_positives} = {ppv:.4f}",
         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.title('Bayes\' Theorem Applied to Medical Testing (Population of 10,000)', fontsize=14)
plt.ylabel('Number of People', fontsize=12)
plt.ylim(0, 10100)  # Slightly higher than population size to fit annotations
plt.xticks(fontsize=12)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bayes_theorem_medical.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 3: Calculate updated probability after second positive test
print_step_header(3, "Updated Probability After Second Positive Test")

# The posterior from the first test becomes the prior for the second test
updated_prevalence = p_disease_given_positive

# Calculate the new posterior probability
p_positive_2 = sensitivity * updated_prevalence + false_positive_rate * (1 - updated_prevalence)
p_disease_given_positive_2 = (sensitivity * updated_prevalence) / p_positive_2

print(f"Prior probability for second test (posterior from first test): P(D) = {updated_prevalence:.4f}")
print(f"Probability of disease after two positive tests: P(D|+,+) = {p_disease_given_positive_2:.4f}")
print(f"This means after two positive tests, there's a {p_disease_given_positive_2*100:.2f}% chance the person has the disease.")
print()

# Visualization of probability update
plt.figure(figsize=(10, 6))

# Posterior probabilities for multiple positive tests
posteriors = [prevalence]  # Start with prior
current_posterior = prevalence

for i in range(10):  # Calculate for 10 tests
    current_posterior = (sensitivity * current_posterior) / (sensitivity * current_posterior + false_positive_rate * (1 - current_posterior))
    posteriors.append(current_posterior)

tests = range(11)  # 0 means prior, before any tests
plt.plot(tests, posteriors, 'bo-', linewidth=2, markersize=8)

# Mark the first and second test results
plt.scatter([1, 2], [p_disease_given_positive, p_disease_given_positive_2], color='red', s=100, zorder=5)
plt.annotate(f"First test: {p_disease_given_positive:.4f}", xy=(1, p_disease_given_positive), 
             xytext=(1.5, p_disease_given_positive - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10)
plt.annotate(f"Second test: {p_disease_given_positive_2:.4f}", xy=(2, p_disease_given_positive_2), 
             xytext=(2.5, p_disease_given_positive_2 - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10)

# Add horizontal line at 0.95 probability
plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Probability Threshold')

# Add annotations
for i, posterior in enumerate(posteriors):
    if i > 0:  # Skip the prior
        plt.text(i, posterior + 0.02, f"{posterior:.4f}", ha='center', fontsize=8)

plt.title('Probability of Disease After Multiple Positive Test Results', fontsize=14)
plt.xlabel('Number of Positive Tests', fontsize=12)
plt.ylabel('Probability of Disease', fontsize=12)
plt.grid(True)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sequential_testing.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Calculate number of positive tests needed for 95% probability
print_step_header(4, "Number of Tests Needed for 95% Confidence")

# Find the number of tests needed for 95% probability
def calculate_posterior(prior, sensitivity, specificity, positive_tests):
    """Calculate posterior probability after multiple positive tests."""
    posterior = prior
    false_positive_rate = 1 - specificity
    
    for _ in range(positive_tests):
        likelihood_ratio = sensitivity / false_positive_rate
        posterior_odds = (posterior / (1 - posterior)) * likelihood_ratio
        posterior = posterior_odds / (1 + posterior_odds)
    
    return posterior

# Search for the number of tests needed
tests_needed = 1
while calculate_posterior(prevalence, sensitivity, specificity, tests_needed) < 0.95:
    tests_needed += 1

print(f"Number of consecutive positive tests needed for at least 95% probability: {tests_needed}")
print(f"Exact probability after {tests_needed} positive tests: {calculate_posterior(prevalence, sensitivity, specificity, tests_needed):.4f}")
print()

# Step 5: Effect of disease prevalence on test interpretation
print_step_header(5, "Effect of Disease Prevalence on Test Interpretation")

# Calculate probability of disease given positive test for different prevalence values
prevalence_range = np.logspace(-3, -1, 30)  # From 0.1% to 10%
ppv_values = []

for prev in prevalence_range:
    p_positive = sensitivity * prev + false_positive_rate * (1 - prev)
    ppv = (sensitivity * prev) / p_positive
    ppv_values.append(ppv)

plt.figure(figsize=(10, 6))
plt.plot(prevalence_range * 100, np.array(ppv_values) * 100, 'b-', linewidth=3)

# Mark our original problem with 1% prevalence
original_ppv = p_disease_given_positive * 100
plt.plot(1, original_ppv, 'ro', markersize=10)
plt.annotate(f"1% prevalence → {original_ppv:.1f}% PPV", 
             xy=(1, original_ppv), xytext=(2, original_ppv + 10),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1),
             fontsize=10)

# Add more example points
example_prevalences = [0.2, 5]  # 0.2% and 5%
for prev in example_prevalences:
    p_positive = sensitivity * prev/100 + false_positive_rate * (1 - prev/100)
    ppv = (sensitivity * prev/100) / p_positive * 100
    plt.plot(prev, ppv, 'go', markersize=8)
    plt.annotate(f"{prev}% prevalence → {ppv:.1f}% PPV", 
                 xy=(prev, ppv), xytext=(prev + (1 if prev < 3 else -3), ppv - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=10)

plt.title('Effect of Disease Prevalence on Positive Predictive Value (PPV)', fontsize=14)
plt.xlabel('Disease Prevalence (%)', fontsize=12)
plt.ylabel('Positive Predictive Value (%)', fontsize=12)
plt.xscale('log')
plt.grid(True, which="both", ls="-")
plt.xlim(0.1, 10)
plt.ylim(0, 100)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "prevalence_effect.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Comparison of different sensitivity/specificity values
print_step_header(6, "Comparison of Different Test Characteristics")

# Create grid for different sensitivity/specificity values
sensitivity_values = [0.8, 0.9, 0.95, 0.99]
specificity_values = [0.8, 0.9, 0.95, 0.99]

plt.figure(figsize=(12, 8))

for sens in sensitivity_values:
    ppv_values = []
    for prev in prevalence_range:
        false_pos_rate = 1 - specificity  # Keep specificity constant at original value
        p_positive = sens * prev + false_pos_rate * (1 - prev)
        ppv = (sens * prev) / p_positive
        ppv_values.append(ppv)
    plt.plot(prevalence_range * 100, np.array(ppv_values) * 100, '-', linewidth=2, 
             label=f"Sensitivity = {sens}")

plt.title('Effect of Different Sensitivity Values on PPV\n(Specificity fixed at 0.9)', fontsize=14)
plt.xlabel('Disease Prevalence (%)', fontsize=12)
plt.ylabel('Positive Predictive Value (%)', fontsize=12)
plt.xscale('log')
plt.grid(True, which="both", ls="-")
plt.xlim(0.1, 10)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "sensitivity_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Now vary specificity
plt.figure(figsize=(12, 8))

for spec in specificity_values:
    ppv_values = []
    for prev in prevalence_range:
        false_pos_rate = 1 - spec
        p_positive = sensitivity * prev + false_pos_rate * (1 - prev)  # Keep sensitivity constant at original value
        ppv = (sensitivity * prev) / p_positive
        ppv_values.append(ppv)
    plt.plot(prevalence_range * 100, np.array(ppv_values) * 100, '-', linewidth=2, 
             label=f"Specificity = {spec}")

plt.title('Effect of Different Specificity Values on PPV\n(Sensitivity fixed at 0.95)', fontsize=14)
plt.xlabel('Disease Prevalence (%)', fontsize=12)
plt.ylabel('Positive Predictive Value (%)', fontsize=12)
plt.xscale('log')
plt.grid(True, which="both", ls="-")
plt.xlim(0.1, 10)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "specificity_comparison.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion and summary
print_step_header(7, "Conclusion and Summary")

print("Summary of findings:")
print(f"1. Probability of disease given a single positive test: {p_disease_given_positive:.4f} or {p_disease_given_positive*100:.2f}%")
print(f"2. Probability of disease after two positive tests: {p_disease_given_positive_2:.4f} or {p_disease_given_positive_2*100:.2f}%")
print(f"3. Number of consecutive positive tests needed for at least 95% probability: {tests_needed}")
print("4. Disease prevalence has a major impact on the interpretation of test results:")
print("   - Lower prevalence → lower positive predictive value")
print("   - Higher prevalence → higher positive predictive value")
print()
print("5. Important insights:")
print("   - A positive test result must be interpreted in the context of disease prevalence")
print("   - For rare diseases, even with good tests, the PPV can be surprisingly low")
print("   - Multiple tests can significantly increase confidence in the diagnosis")
print("   - Test specificity becomes extremely important for rare diseases") 