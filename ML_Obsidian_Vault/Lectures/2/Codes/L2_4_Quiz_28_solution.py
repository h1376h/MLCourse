import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Create directory to save figures if it doesn't exist
os.makedirs("images", exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join("images", filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
    plt.close(fig)

# Problem data
categories = ['Cat', 'Dog', 'Bird']
samples = [
    {'probabilities': [0.7, 0.2, 0.1], 'true_label': 'Cat', 'true_one_hot': [1, 0, 0]},
    {'probabilities': [0.3, 0.5, 0.2], 'true_label': 'Dog', 'true_one_hot': [0, 1, 0]},
    {'probabilities': [0.1, 0.3, 0.6], 'true_label': 'Bird', 'true_one_hot': [0, 0, 1]},
    {'probabilities': [0.4, 0.4, 0.2], 'true_label': 'Cat', 'true_one_hot': [1, 0, 0]},
    {'probabilities': [0.2, 0.1, 0.7], 'true_label': 'Bird', 'true_one_hot': [0, 0, 1]}
]

# ==============================
# STEP 1: Log-Likelihood of the Observed Data
# ==============================
print_section_header("STEP 1: Log-Likelihood of the Observed Data")

print("The log-likelihood measures how well the model's predicted probabilities match the observed data.")
print("For our multinomial classifier, we need to calculate:")
print("log L = ∑(i=1 to n) log(ŷ_{i,c_i})")
print("where c_i is the correct class index for sample i, and ŷ_{i,c_i} is the predicted probability")
print("for the correct class of sample i.")

print("\nStep-by-step log-likelihood calculation:")
# Calculate log-likelihood with detailed steps
log_likelihood = 0
likelihood_details = []

print("\nSample details:")
for i, sample in enumerate(samples):
    true_label_index = sample['true_one_hot'].index(1)
    prob_of_true_class = sample['probabilities'][true_label_index]
    log_prob = math.log(prob_of_true_class)
    log_likelihood += log_prob
    
    # Store details for display
    likelihood_details.append({
        'sample': i+1,
        'true_class': categories[true_label_index],
        'prob': prob_of_true_class,
        'log_prob': log_prob
    })
    
    # Print detailed calculations
    print(f"Sample {i+1}:")
    print(f"  True class: {categories[true_label_index]}")
    print(f"  Predicted probabilities: {[round(p, 2) for p in sample['probabilities']]}")
    print(f"  Probability assigned to true class: {prob_of_true_class:.2f}")
    print(f"  log({prob_of_true_class:.2f}) = {log_prob:.4f}")

print(f"\nTotal log-likelihood = sum of all log probabilities = {log_likelihood:.4f}")
print("This value quantifies how well the model's predictions match the true labels.")
print("Higher (less negative) values indicate better predictive performance.")

# Visualization of log-likelihood - SIMPLIFIED
fig1 = plt.figure(figsize=(10, 5))

# Plot: Bar chart of probabilities for true classes (simplified)
ax = plt.subplot()
sample_labels = [f"Sample {i+1}\n({detail['true_class']})" for i, detail in enumerate(likelihood_details)]
probs = [detail['prob'] for detail in likelihood_details]
colors = sns.color_palette("pastel", len(samples))

bars = ax.bar(sample_labels, probs, color=colors)
ax.set_title('Probability Assigned to True Class')
ax.set_xlabel('Sample')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1.0)

# Add cleaner labels
for bar, prob, log_prob in zip(bars, probs, [d['log_prob'] for d in likelihood_details]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height/2, 
            f"{prob:.2f}\nlog: {log_prob:.2f}", 
            ha='center', va='center', fontweight='bold', color='black')

# Add log-likelihood value
plt.text(0.5, 0.9, f"Total Log-Likelihood: {log_likelihood:.4f}", 
         transform=ax.transAxes, ha='center', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
save_figure(fig1, "step1_log_likelihood.png")

# ==============================
# STEP 2: MLE Threshold for Classification
# ==============================
print_section_header("STEP 2: MLE Threshold for Classification")

print("Using Maximum Likelihood Estimation (MLE) principles, we determine the optimal")
print("classification strategy that maximizes the likelihood of the observed data.")
print("\nIn MLE framework, we classify each sample according to the class with the highest probability.")
print("This corresponds to a relative threshold rather than an absolute fixed threshold value.")

print("\nAnalyzing each sample's classification using the maximum probability rule:")
# Check highest probability class for each sample with detailed explanation
correct_count = 0
for i, sample in enumerate(samples):
    probs = sample['probabilities']
    max_prob_index = np.argmax(probs)
    max_prob = probs[max_prob_index]
    true_label_index = sample['true_one_hot'].index(1)
    
    is_correct = max_prob_index == true_label_index
    if is_correct:
        correct_count += 1
    
    result = "Correct" if is_correct else "Incorrect"
    
    print(f"\nSample {i+1}:")
    print(f"  Probabilities: {[round(p, 2) for p in probs]}")
    print(f"  Maximum probability: {max_prob:.2f} for class '{categories[max_prob_index]}'")
    print(f"  True class: '{categories[true_label_index]}'")
    print(f"  Classification result: {result}")

accuracy = correct_count / len(samples)
print(f"\nAccuracy using maximum probability classification: {correct_count}/{len(samples)} = {accuracy:.2f} or {accuracy*100:.0f}%")

print("\nThis demonstrates that from an MLE perspective, the optimal strategy is to classify each sample")
print("according to the class with the highest probability, which maximizes the likelihood of the data.")

# Visualization of max probability classifications - SIMPLIFIED
fig2 = plt.figure(figsize=(10, 5))

# Create a clear visualization of probabilities and classifications
sample_indices = np.arange(len(samples))
bar_width = 0.8 / len(categories)

# For each sample, show the probabilities for each category
for i, category in enumerate(categories):
    # Extract probabilities for this category across all samples
    probs = [sample['probabilities'][i] for sample in samples]
    
    # Calculate offset for grouped bars
    offset = (i - (len(categories)-1)/2) * bar_width
    
    # Plot bars
    plt.bar(sample_indices + offset, probs, bar_width, 
            label=category, alpha=0.7)
    
    # Mark the true labels with a star
    for j, sample in enumerate(samples):
        if sample['true_one_hot'][i] == 1:  # This is the true class
            plt.plot(j + offset, probs[j] + 0.05, 'k*', markersize=10)

# Formatting
plt.xlabel('Sample')
plt.ylabel('Probability')
plt.title('Model Probabilities by Class')
plt.xticks(sample_indices, [f'Sample {i+1}' for i in range(len(samples))])
plt.legend(title='Class')
plt.ylim(0, 1.0)

# Add a note about the MLE approach
plt.text(0.5, 0.95, f"MLE Strategy: Choose highest probability class (Accuracy: {accuracy*100:.0f}%)", 
         transform=plt.gca().transAxes, ha='center', fontsize=11,
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
save_figure(fig2, "step2_mle_threshold.png")

# ==============================
# STEP 3: Fixed Probability Threshold for Maximum Accuracy
# ==============================
print_section_header("STEP 3: Fixed Probability Threshold for Maximum Accuracy")

print("Now we'll determine a fixed probability threshold that maximizes accuracy on this dataset.")
print("With a fixed threshold, we only classify a sample to a class if the probability exceeds the threshold.")
print("We'll test different threshold values to find the optimal one.\n")

# Generate a range of threshold values to test
threshold_values = np.arange(0.05, 1.0, 0.05)
print(f"Testing threshold values from {threshold_values[0]:.2f} to {threshold_values[-1]:.2f}:")

# Test each threshold with detailed results
threshold_results = []

for threshold in threshold_values:
    print(f"\nThreshold {threshold:.2f}:")
    correct_classifications = 0
    total_classifications = 0
    classifications = []
    
    for i, sample in enumerate(samples):
        probs = sample['probabilities']
        true_label_index = sample['true_one_hot'].index(1)
        
        # Check which classes exceed the threshold
        classes_above_threshold = [j for j, p in enumerate(probs) if p > threshold]
        
        # Store classification result for this sample
        if not classes_above_threshold:
            cls_result = "Unclassified (no class exceeds threshold)"
            correct = False
        elif len(classes_above_threshold) == 1:
            # One class exceeds threshold - straightforward case
            predicted_class = classes_above_threshold[0]
            cls_result = f"Classified as '{categories[predicted_class]}'"
            correct = predicted_class == true_label_index
            total_classifications += 1
        else:
            # Multiple classes exceed threshold - take the highest
            predicted_class = np.argmax(probs)
            cls_result = f"Multiple classes exceed threshold, classified as highest: '{categories[predicted_class]}'"
            correct = predicted_class == true_label_index
            total_classifications += 1
        
        if correct:
            correct_classifications += 1
            
        classifications.append(cls_result)
        
        # Print sample-specific results
        print(f"  Sample {i+1} (True: {categories[true_label_index]}): {cls_result}")
    
    # Calculate accuracy for this threshold
    accuracy = correct_classifications / len(samples) if len(samples) > 0 else 0
    
    # Store results
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'correct': correct_classifications,
        'total_classified': total_classifications,
        'classifications': classifications
    })
    
    # Print threshold-specific results
    classified_percent = (total_classifications / len(samples)) * 100 if len(samples) > 0 else 0
    print(f"  Correct: {correct_classifications}/{len(samples)}, Classified: {total_classifications}/{len(samples)} ({classified_percent:.0f}%)")
    print(f"  Accuracy: {accuracy:.4f} or {accuracy*100:.0f}%")

# Find the threshold that gives the highest accuracy
best_results = [r for r in threshold_results if r['accuracy'] == max([r['accuracy'] for r in threshold_results])]
best_threshold = min([r['threshold'] for r in best_results])  # Choose lowest threshold with best accuracy

print("\nAnalysis of threshold performance:")
print(f"The best accuracy of {best_results[0]['accuracy']*100:.0f}% is achieved with thresholds: ", 
      ', '.join([f"{r['threshold']:.2f}" for r in best_results]))
print(f"The optimal threshold is {best_threshold:.2f}, which is the lowest threshold that achieves maximum accuracy.")
print("A lower threshold is preferable when multiple thresholds give the same accuracy because it")
print("classifies more samples while maintaining the same level of accuracy.")

# Visualization of accuracy vs threshold - SIMPLIFIED
fig3 = plt.figure(figsize=(10, 5))

# Plot accuracy vs threshold
thresholds = [r['threshold'] for r in threshold_results]
accuracies = [r['accuracy'] for r in threshold_results]
classified_percentages = [r['total_classified']/len(samples) for r in threshold_results]

plt.plot(thresholds, accuracies, 'bo-', label='Accuracy', linewidth=2)
plt.plot(thresholds, classified_percentages, 'ro-', label='% Classified', linewidth=2)

# Highlight the best threshold
plt.axvline(x=best_threshold, color='g', linestyle='--', 
            label=f'Optimal Threshold = {best_threshold:.2f}')

# Add annotations
plt.annotate(f'Max Accuracy: {best_results[0]["accuracy"]*100:.0f}%', 
             xy=(best_threshold, best_results[0]['accuracy']), 
             xytext=(best_threshold+0.1, best_results[0]['accuracy']-0.1),
             arrowprops=dict(arrowstyle='->'))

plt.title('Accuracy vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
save_figure(fig3, "step3_fixed_threshold.png")

# ==============================
# SUMMARY
# ==============================
print_section_header("SUMMARY")

print("Key findings from this analysis of one-hot encoding and maximum likelihood estimation:")

print("\n1. Log-Likelihood of the Observed Data:")
print(f"   - The log-likelihood is {log_likelihood:.4f}")
print(f"   - This measures how well the model's predictions align with the true labels")
print(f"   - Higher (less negative) values indicate better model performance")

print("\n2. MLE Threshold for Classification:")
print(f"   - From a maximum likelihood perspective, we should classify each sample")
print(f"     to the class with the highest probability")
print(f"   - This strategy achieved {correct_count}/{len(samples)} correct classifications ({correct_count/len(samples)*100:.0f}% accuracy)")

print("\n3. Fixed Probability Threshold:")
print(f"   - The optimal fixed threshold is {best_threshold:.2f}")
print(f"   - This threshold achieves {best_results[0]['accuracy']*100:.0f}% accuracy")
print(f"   - Lower thresholds (≤ {best_threshold:.2f}) classify more samples while maintaining accuracy")
print(f"   - Higher thresholds result in fewer classifications and lower accuracy")
print(f"   - Fixed thresholds introduce trade-offs between classification confidence and coverage") 