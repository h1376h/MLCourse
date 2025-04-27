import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.gridspec import GridSpec
import os

# Set a nice style for the plots
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 11})  # Slightly smaller font size for cleaner plots

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

# Create directory for saving images
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_4_Quiz_28")
os.makedirs(save_dir, exist_ok=True)

def save_figure(fig, filename):
    """Save figure to the specified directory."""
    filepath = os.path.join(save_dir, filename)
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

# Visualization of log-likelihood - ENHANCED
fig1 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Top: Bar chart of probabilities for true classes
ax1 = plt.subplot(gs[0])
sample_labels = [f"Sample {i+1}\n({detail['true_class']})" for i, detail in enumerate(likelihood_details)]
probs = [detail['prob'] for detail in likelihood_details]
colors = sns.color_palette("pastel", len(samples))

bars = ax1.bar(sample_labels, probs, color=colors)
ax1.set_title('Probability Assigned to True Class')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Probability')
ax1.set_ylim(0, 1.0)

# Add cleaner labels
for bar, prob, log_prob in zip(bars, probs, [d['log_prob'] for d in likelihood_details]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height/2, 
            f"p = {prob:.2f}\nlog(p) = {log_prob:.2f}", 
            ha='center', va='center', fontweight='bold', color='black')

# Bottom: Visual explanation of log-likelihood calculation
ax2 = plt.subplot(gs[1])
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, "Log-Likelihood Calculation Explained", 
         ha='center', va='center', fontsize=14, fontweight='bold')

# Create a step-by-step explanation with formula and calculation
formula = r"$\log L = \sum_{i=1}^{n} \log(P(y_i|x_i))$"
ax2.text(0.5, 0.85, formula, ha='center', va='center', fontsize=14)

# Show the calculation step by step
step_y_position = 0.7
ax2.text(0.05, step_y_position, "Calculation:", ha='left', va='center', fontsize=12, fontweight='bold')

calculation = "log L = "
for i, detail in enumerate(likelihood_details):
    if i > 0:
        calculation += " + "
    calculation += f"log({detail['prob']:.2f})"
step_y_position -= 0.07
ax2.text(0.1, step_y_position, calculation, ha='left', va='center', fontsize=12)

calculation = "log L = "
for i, detail in enumerate(likelihood_details):
    if i > 0:
        calculation += " + "
    calculation += f"{detail['log_prob']:.4f}"
step_y_position -= 0.07
ax2.text(0.1, step_y_position, calculation, ha='left', va='center', fontsize=12)

step_y_position -= 0.07
ax2.text(0.1, step_y_position, f"log L = {log_likelihood:.4f}", ha='left', va='center', fontsize=12, fontweight='bold', 
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round'))

# Interpretation
ax2.text(0.05, 0.35, "Interpretation:", ha='left', va='center', fontsize=12, fontweight='bold')
interpretations = [
    "• The log-likelihood measures how well the model assigns probability to the true classes",
    "• Values closer to 0 (less negative) indicate better performance",
    "• This value can be used to compare different models on the same dataset",
    "• Maximizing log-likelihood is equivalent to minimizing cross-entropy loss"
]
for i, interp in enumerate(interpretations):
    ax2.text(0.1, 0.28 - i*0.07, interp, ha='left', va='center', fontsize=11)

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

# Visualization of max probability classifications - ENHANCED
fig2 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Top: Grouped bar chart of probabilities
ax1 = plt.subplot(gs[0])
x = np.arange(len(samples))
width = 0.25

# For each sample, show the probabilities for each category
for i, category in enumerate(categories):
    # Extract probabilities for this category across all samples
    probs = [sample['probabilities'][i] for sample in samples]
    
    # Calculate offset for grouped bars
    offset = (i - 1) * width
    
    # Plot bars with custom properties based on correctness
    bars = ax1.bar(x + offset, probs, width, label=category, alpha=0.7)
    
    # Mark the true labels with a star
    for j, sample in enumerate(samples):
        if sample['true_one_hot'][i] == 1:  # This is the true class
            ax1.plot(j + offset, probs[j] + 0.05, 'k*', markersize=12, label='_nolegend_')
        
        # Add annotation for the winner (highest probability)
        if np.argmax(sample['probabilities']) == i:
            ax1.text(j + offset, probs[j] + 0.02, "Max", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

# Formatting
ax1.set_xlabel('Sample')
ax1.set_ylabel('Probability')
ax1.set_title('Model Probabilities by Class')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Sample {i+1}' for i in range(len(samples))])
ax1.legend(title='Class')
ax1.set_ylim(0, 1.0)

# Bottom: Explanation of MLE classification
ax2 = plt.subplot(gs[1])
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, "Maximum Likelihood Estimation (MLE) for Classification", 
         ha='center', va='center', fontsize=14, fontweight='bold')

# MLE principle explanation
ax2.text(0.05, 0.85, "MLE Classification Principle:", ha='left', va='center', fontsize=12, fontweight='bold')
ax2.text(0.1, 0.78, "• Choose the class with the highest probability (argmax rule)", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.72, "• For a sample with probabilities [p₁, p₂, ..., pₖ], predict class j where j = argmax(pᵢ)", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.66, "• No fixed threshold is needed - only the relative ordering of probabilities matters", ha='left', va='center', fontsize=11)

# Mathematical justification
ax2.text(0.05, 0.55, "Mathematical Justification:", ha='left', va='center', fontsize=12, fontweight='bold')
ax2.text(0.1, 0.48, "• For one-hot encoded labels y = [0,0,...,1,0,...], only one position has y_j = 1", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.42, "• Log-likelihood for a sample is log(p_j) where j is the true class", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.36, "• To maximize likelihood across all samples, we choose class with highest probability", ha='left', va='center', fontsize=11)

# Sample classification results
ax2.text(0.05, 0.25, "Classification Results:", ha='left', va='center', fontsize=12, fontweight='bold')
ax2.text(0.1, 0.18, f"• Correct classifications: {correct_count}/{len(samples)}", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.12, f"• Accuracy: {accuracy:.2f} or {accuracy*100:.0f}%", ha='left', va='center', fontsize=11,
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round'))

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

# Visualization of accuracy vs threshold - ENHANCED
fig3 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 1, height_ratios=[1, 1.2])

# Top: Plot accuracy and classification rate vs threshold
ax1 = plt.subplot(gs[0])
thresholds = [r['threshold'] for r in threshold_results]
accuracies = [r['accuracy'] for r in threshold_results]
classified_percentages = [r['total_classified']/len(samples) for r in threshold_results]

ax1.plot(thresholds, accuracies, 'bo-', label='Accuracy', linewidth=2)
ax1.plot(thresholds, classified_percentages, 'ro-', label='% Classified', linewidth=2)

# Highlight the best threshold
ax1.axvline(x=best_threshold, color='g', linestyle='--', 
           label=f'Optimal Threshold = {best_threshold:.2f}')

# Add annotations
ax1.annotate(f'Max Accuracy: {best_results[0]["accuracy"]*100:.0f}%', 
             xy=(best_threshold, best_results[0]['accuracy']), 
             xytext=(best_threshold+0.1, best_results[0]['accuracy']-0.1),
             arrowprops=dict(arrowstyle='->'))

ax1.set_title('Accuracy vs. Threshold')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Rate')
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bottom: Explanation of fixed threshold approach
ax2 = plt.subplot(gs[1])
ax2.axis('off')

# Title
ax2.text(0.5, 0.95, "Fixed Probability Threshold Analysis", 
         ha='center', va='center', fontsize=14, fontweight='bold')

# Fixed threshold explanation
ax2.text(0.05, 0.85, "Classification Strategy with Fixed Threshold:", ha='left', va='center', fontsize=12, fontweight='bold')
ax2.text(0.1, 0.78, "• Only classify a sample to class i if probability p_i > threshold", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.72, "• If multiple classes exceed threshold, choose the one with highest probability", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.66, "• If no class exceeds threshold, the sample remains unclassified", ha='left', va='center', fontsize=11)

# Trade-offs explanation
ax2.text(0.05, 0.55, "Trade-offs in Threshold Selection:", ha='left', va='center', fontsize=12, fontweight='bold')
ax2.text(0.1, 0.48, "• Lower threshold → More samples classified → Potentially more errors", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.42, "• Higher threshold → Fewer samples classified → Higher confidence in classifications", ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.36, "• Optimal threshold balances accuracy and classification coverage", ha='left', va='center', fontsize=11)

# Optimal threshold results
ax2.text(0.05, 0.25, "Results for Optimal Threshold:", ha='left', va='center', fontsize=12, fontweight='bold')
opt_result = next(r for r in threshold_results if r['threshold'] == best_threshold)
ax2.text(0.1, 0.18, f"• Optimal threshold: {best_threshold:.2f}", ha='left', va='center', fontsize=11, 
         bbox=dict(facecolor='lightgreen', alpha=0.3, boxstyle='round'))
ax2.text(0.1, 0.12, f"• Accuracy: {opt_result['accuracy']*100:.0f}%, Classification rate: {opt_result['total_classified']/len(samples)*100:.0f}%", 
         ha='left', va='center', fontsize=11)
ax2.text(0.1, 0.06, f"• Correctly classified: {opt_result['correct']}/{len(samples)}, Total classified: {opt_result['total_classified']}/{len(samples)}", 
         ha='left', va='center', fontsize=11)

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

# Create summary visualization
fig4 = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5])

# Top-left: Log-likelihood visualization
ax1 = fig4.add_subplot(gs[0, 0])
sample_labels = [f"S{i+1}" for i in range(len(samples))]
probs = [detail['prob'] for detail in likelihood_details]
ax1.bar(sample_labels, probs, color=colors)
ax1.set_title('Probabilities of True Classes')
ax1.set_ylim(0, 1.0)
for i, (prob, log_prob) in enumerate(zip(probs, [d['log_prob'] for d in likelihood_details])):
    ax1.text(i, prob/2, f"{prob:.2f}", ha='center', va='center', fontweight='bold')
ax1.text(0.5, 0.9, f"Log-Likelihood: {log_likelihood:.2f}", transform=ax1.transAxes, ha='center',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

# Top-right: MLE classification
ax2 = fig4.add_subplot(gs[0, 1])
correct_colors = ['green' if np.argmax(s['probabilities']) == s['true_one_hot'].index(1) else 'red' 
                 for s in samples]
ax2.bar(sample_labels, [1]*len(samples), color=correct_colors)
ax2.set_title('MLE Classification Results')
ax2.set_ylim(0, 1.2)
for i, correct in enumerate(correct_colors):
    result = "✓" if correct == 'green' else "✗"
    ax2.text(i, 0.5, result, ha='center', va='center', fontsize=16, fontweight='bold')
ax2.text(0.5, 0.9, f"Accuracy: {correct_count/len(samples)*100:.0f}%", transform=ax2.transAxes, ha='center',
         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

# Middle-left: Threshold accuracy plot
ax3 = fig4.add_subplot(gs[1, 0])
ax3.plot(thresholds, accuracies, 'bo-', label='Accuracy')
ax3.axvline(x=best_threshold, color='g', linestyle='--', label=f'Threshold={best_threshold:.2f}')
ax3.set_title('Accuracy vs. Threshold')
ax3.set_ylabel('Accuracy')
ax3.set_ylim(0, 1.1)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Middle-right: Threshold classification rate
ax4 = fig4.add_subplot(gs[1, 1])
ax4.plot(thresholds, classified_percentages, 'ro-', label='% Classified')
ax4.axvline(x=best_threshold, color='g', linestyle='--', label=f'Threshold={best_threshold:.2f}')
ax4.set_title('Classification Rate vs. Threshold')
ax4.set_ylabel('% Classified')
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3)
ax4.legend()

# Bottom: Key insights
ax5 = fig4.add_subplot(gs[2, :])
ax5.axis('off')
ax5.text(0.5, 0.9, "Key Insights", ha='center', va='center', fontsize=14, fontweight='bold')

insights = [
    "• Log-likelihood evaluates how well a model assigns probability to true classes",
    "• MLE classification simply chooses the class with highest probability",
    f"• The optimal fixed threshold for this dataset is {best_threshold:.2f}",
    "• Fixed thresholds allow controlling the trade-off between confidence and coverage"
]

for i, insight in enumerate(insights):
    ax5.text(0.5, 0.7 - i*0.2, insight, ha='center', va='center', fontsize=12)

plt.tight_layout()
save_figure(fig4, "step4_summary.png")

print("\nAll visualizations have been saved to the Images/L2_4_Quiz_28 directory.") 