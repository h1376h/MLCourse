import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 6: RANDOM FOREST ENSEMBLE PREDICTION ANALYSIS")
print("=" * 80)

# Given data
print("\nGIVEN DATA:")
print("-" * 50)

# Sample 1: Tree predictions [0, 1, 0, 1, 0] with confidences [0.8, 0.6, 0.9, 0.7, 0.85]
sample1_predictions = np.array([0, 1, 0, 1, 0])
sample1_confidences = np.array([0.8, 0.6, 0.9, 0.7, 0.85])

# Sample 2: Tree predictions [1, 1, 1, 1, 1] with confidences [0.95, 0.88, 0.92, 0.89, 0.91]
sample2_predictions = np.array([1, 1, 1, 1, 1])
sample2_confidences = np.array([0.95, 0.88, 0.92, 0.89, 0.91])

# Sample 3: Tree predictions [0, 1, 0, 1, 0] with confidences [0.55, 0.65, 0.45, 0.75, 0.60]
sample3_predictions = np.array([0, 1, 0, 1, 0])
sample3_confidences = np.array([0.55, 0.65, 0.45, 0.75, 0.60])

samples = [
    ("Sample 1", sample1_predictions, sample1_confidences),
    ("Sample 2", sample2_predictions, sample2_confidences),
    ("Sample 3", sample3_predictions, sample3_confidences)
]

for name, preds, confs in samples:
    print(f"{name}:")
    print(f"  Tree Predictions: {preds}")
    print(f"  Tree Confidences: {confs}")
    print()

print("=" * 80)
print("SOLUTION")
print("=" * 80)

# Task 1: Hard Voting
print("\nTASK 1: HARD VOTING ANALYSIS")
print("-" * 40)

def hard_voting(predictions):
    """Perform hard voting on binary predictions"""
    # Count votes for each class
    vote_counts = Counter(predictions)
    # Return the majority class
    majority_class = max(vote_counts, key=vote_counts.get)
    # Calculate vote percentage
    total_votes = len(predictions)
    vote_percentage = (vote_counts[majority_class] / total_votes) * 100
    return majority_class, vote_counts, vote_percentage

print("Hard voting counts the number of trees predicting each class:")
print("Class with majority votes wins the final prediction.\n")

for name, preds, confs in samples:
    final_pred, vote_counts, vote_percentage = hard_voting(preds)
    print(f"{name}:")
    print(f"  Tree predictions: {preds}")
    print(f"  Vote counts: Class 0 = {vote_counts.get(0, 0)}, Class 1 = {vote_counts.get(1, 0)}")
    print(f"  Final prediction: Class {final_pred} (wins with {vote_percentage:.1f}% of votes)")
    print()

# Task 2: Soft Voting
print("\nTASK 2: SOFT VOTING ANALYSIS")
print("-" * 40)

def soft_voting(predictions, confidences):
    """Perform soft voting using confidence scores"""
    # Calculate weighted average confidence for each class
    class0_confidence = 0
    class1_confidence = 0
    
    for pred, conf in zip(predictions, confidences):
        if pred == 0:
            class0_confidence += conf
        else:
            class1_confidence += conf
    
    # Normalize by number of trees
    n_trees = len(predictions)
    class0_avg = class0_confidence / n_trees
    class1_avg = class1_confidence / n_trees
    
    # Final prediction is the class with higher average confidence
    final_pred = 0 if class0_avg > class1_avg else 1
    final_confidence = max(class0_avg, class1_avg)
    
    return final_pred, class0_avg, class1_avg, final_confidence

print("Soft voting uses confidence scores to calculate weighted averages:")
print("Final prediction is the class with higher average confidence.\n")

for name, preds, confs in samples:
    final_pred, conf0, conf1, final_conf = soft_voting(preds, confs)
    print(f"{name}:")
    print(f"  Tree predictions: {preds}")
    print(f"  Tree confidences: {confs}")
    print(f"  Average confidence for Class 0: {conf0:.3f}")
    print(f"  Average confidence for Class 1: {conf1:.3f}")
    print(f"  Final prediction: Class {final_pred} (confidence: {final_conf:.3f})")
    print()

# Task 3: Highest Confidence Analysis
print("\nTASK 3: HIGHEST CONFIDENCE ANALYSIS")
print("-" * 40)

ensemble_confidences = []
for name, preds, confs in samples:
    _, _, _, final_conf = soft_voting(preds, confs)
    ensemble_confidences.append((name, final_conf))

# Sort by confidence
ensemble_confidences.sort(key=lambda x: x[1], reverse=True)

print("Ensemble confidence for each sample (sorted by confidence):")
for name, conf in ensemble_confidences:
    print(f"  {name}: {conf:.3f}")

highest_confidence_sample = ensemble_confidences[0]
print(f"\nSample with highest confidence: {highest_confidence_sample[0]} ({highest_confidence_sample[1]:.3f})")

# Task 4: Trust Analysis
print("\nTASK 4: TRUST ANALYSIS")
print("-" * 40)

print("For high confidence requirements, we should consider:")
print("1. Ensemble confidence (from soft voting)")
print("2. Agreement among trees (consensus)")
print("3. Individual tree confidence levels\n")

trust_scores = []
for name, preds, confs in samples:
    # Ensemble confidence
    _, _, _, ensemble_conf = soft_voting(preds, confs)
    
    # Agreement among trees (percentage of trees agreeing with ensemble)
    final_pred, _, _, _ = soft_voting(preds, confs)
    agreement = np.mean(preds == final_pred)
    
    # Average individual tree confidence
    avg_tree_conf = np.mean(confs)
    
    # Overall trust score (weighted combination)
    trust_score = 0.4 * ensemble_conf + 0.4 * agreement + 0.2 * avg_tree_conf
    
    trust_scores.append((name, ensemble_conf, agreement, avg_tree_conf, trust_score))
    
    print(f"{name}:")
    print(f"  Ensemble confidence: {ensemble_conf:.3f}")
    print(f"  Tree agreement: {agreement:.3f} ({agreement*100:.1f}%)")
    print(f"  Average tree confidence: {avg_tree_conf:.3f}")
    print(f"  Trust score: {trust_score:.3f}")
    print()

# Sort by trust score
trust_scores.sort(key=lambda x: x[4], reverse=True)
most_trusted = trust_scores[0]
print(f"Most trusted sample for high confidence: {most_trusted[0]} (trust score: {most_trusted[4]:.3f})")

# Task 5: Variance Analysis
print("\nTASK 5: VARIANCE ANALYSIS")
print("-" * 40)

def calculate_prediction_variance(predictions, confidences):
    """Calculate variance in predictions and confidences"""
    # Variance in predictions (binary, so we can use proportion of disagreements)
    pred_variance = np.var(predictions.astype(float))
    
    # Variance in confidences
    conf_variance = np.var(confidences)
    
    # Disagreement measure (how much trees disagree)
    disagreement = 1 - np.mean(predictions == predictions[0])  # 0 if all agree, 1 if split
    
    return pred_variance, conf_variance, disagreement

print("Variance analysis shows disagreement among trees:")
print("Higher variance indicates more disagreement and uncertainty.\n")

variance_results = []
for name, preds, confs in samples:
    pred_var, conf_var, disagreement = calculate_prediction_variance(preds, confs)
    variance_results.append((name, pred_var, conf_var, disagreement))
    
    print(f"{name}:")
    print(f"  Prediction variance: {pred_var:.3f}")
    print(f"  Confidence variance: {conf_var:.3f}")
    print(f"  Disagreement measure: {disagreement:.3f}")
    print(f"  Interpretation: {'High disagreement' if disagreement > 0.5 else 'Low disagreement'}")
    print()

# Find sample with highest disagreement
highest_disagreement = max(variance_results, key=lambda x: x[3])
print(f"Sample with highest disagreement: {highest_disagreement[0]} (disagreement: {highest_disagreement[3]:.3f})")

# Visualization 1: Tree Predictions and Confidences
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

plt.figure(figsize=(15, 10))

# Subplot 1: Tree Predictions
plt.subplot(2, 2, 1)
tree_labels = [f'Tree {i+1}' for i in range(5)]
x_pos = np.arange(5)

for i, (name, preds, confs) in enumerate(samples):
    plt.subplot(2, 2, i+1)
    
    # Plot predictions as bars
    colors = ['red' if p == 0 else 'blue' for p in preds]
    bars = plt.bar(tree_labels, preds, color=colors, alpha=0.7)
    
    # Add confidence values on top of bars
    for j, (bar, conf) in enumerate(zip(bars, confs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{conf:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'{name}: Tree Predictions and Confidences')
    plt.ylabel('Prediction (0 or 1)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Class 0'),
                      Patch(facecolor='blue', alpha=0.7, label='Class 1')]
    plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tree_predictions_confidences.png'), dpi=300, bbox_inches='tight')

# Visualization 2: Voting Comparison
plt.figure(figsize=(12, 8))

# Hard vs Soft voting comparison
methods = ['Hard Voting', 'Soft Voting']
samples_names = ['Sample 1', 'Sample 2', 'Sample 3']

hard_results = []
soft_results = []

for name, preds, confs in samples:
    # Hard voting
    final_pred, vote_counts, vote_percentage = hard_voting(preds)
    hard_results.append(vote_percentage)
    
    # Soft voting
    final_pred, _, _, final_conf = soft_voting(preds, confs)
    soft_results.append(final_conf)

x = np.arange(len(samples_names))
width = 0.35

plt.bar(x - width/2, hard_results, width, label='Hard Voting (% votes)', alpha=0.8)
plt.bar(x + width/2, [conf * 100 for conf in soft_results], width, label='Soft Voting (% confidence)', alpha=0.8)

plt.xlabel('Samples')
plt.ylabel('Percentage / Confidence')
plt.title('Hard vs Soft Voting Comparison')
plt.xticks(x, samples_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (hard, soft) in enumerate(zip(hard_results, soft_results)):
    plt.text(i - width/2, hard + 1, f'{hard:.1f}%', ha='center', va='bottom')
    plt.text(i + width/2, soft * 100 + 1, f'{soft*100:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'voting_comparison.png'), dpi=300, bbox_inches='tight')

# Visualization 3: Confidence and Agreement Analysis
plt.figure(figsize=(15, 5))

# Subplot 1: Ensemble Confidence
plt.subplot(1, 3, 1)
sample_names = [name for name, _, _, _, _ in trust_scores]
ensemble_confs = [conf for _, conf, _, _, _ in trust_scores]
colors = ['lightblue', 'lightgreen', 'lightcoral']

bars = plt.bar(sample_names, ensemble_confs, color=colors, alpha=0.8)
plt.title('Ensemble Confidence by Sample')
plt.ylabel('Confidence')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, conf in zip(bars, ensemble_confs):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{conf:.3f}', ha='center', va='bottom')

# Subplot 2: Tree Agreement
plt.subplot(1, 3, 2)
agreements = [agreement for _, _, agreement, _, _ in trust_scores]

bars = plt.bar(sample_names, agreements, color=colors, alpha=0.8)
plt.title('Tree Agreement by Sample')
plt.ylabel('Agreement Ratio')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, agreement in zip(bars, agreements):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{agreement:.3f}', ha='center', va='bottom')

# Subplot 3: Trust Scores
plt.subplot(1, 3, 3)
trust_values = [trust for _, _, _, _, trust in trust_scores]

bars = plt.bar(sample_names, trust_values, color=colors, alpha=0.8)
plt.title('Overall Trust Score by Sample')
plt.ylabel('Trust Score')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, trust in zip(bars, trust_values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{trust:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confidence_agreement_analysis.png'), dpi=300, bbox_inches='tight')

# Visualization 4: Variance and Disagreement
plt.figure(figsize=(12, 8))

# Subplot 1: Prediction and Confidence Variance
plt.subplot(2, 2, 1)
sample_names = [name for name, _, _, _ in variance_results]
pred_vars = [pred_var for _, pred_var, _, _ in variance_results]
conf_vars = [conf_var for _, _, conf_var, _ in variance_results]

x = np.arange(len(sample_names))
width = 0.35

plt.bar(x - width/2, pred_vars, width, label='Prediction Variance', alpha=0.8)
plt.bar(x + width/2, conf_vars, width, label='Confidence Variance', alpha=0.8)

plt.xlabel('Samples')
plt.ylabel('Variance')
plt.title('Prediction vs Confidence Variance')
plt.xticks(x, sample_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Disagreement Measure
plt.subplot(2, 2, 2)
disagreements = [disagreement for _, _, _, disagreement in variance_results]

bars = plt.bar(sample_names, disagreements, color=['red', 'orange', 'green'], alpha=0.8)
plt.title('Tree Disagreement Measure')
plt.ylabel('Disagreement (0=Agree, 1=Disagree)')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels
for bar, disagreement in zip(bars, disagreements):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{disagreement:.3f}', ha='center', va='bottom')

# Subplot 3: Confidence Distribution
plt.subplot(2, 2, 3)
for name, preds, confs in samples:
    plt.hist(confs, alpha=0.7, label=name, bins=5)

plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Tree Confidences')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Summary Statistics
plt.subplot(2, 2, 4)
plt.axis('off')

# Create summary table
summary_data = []
for name, preds, confs in samples:
    # Hard voting
    final_pred, vote_counts, vote_percentage = hard_voting(preds)
    # Soft voting
    _, _, _, final_conf = soft_voting(preds, confs)
    # Variance
    pred_var, conf_var, disagreement = calculate_prediction_variance(preds, confs)
    
    summary_data.append([name, final_pred, f"{vote_percentage:.1f}%", f"{final_conf:.3f}", f"{disagreement:.3f}"])

# Create table
table = plt.table(cellText=summary_data,
                  colLabels=['Sample', 'Hard Vote', 'Vote %', 'Soft Conf', 'Disagreement'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.title('Summary of All Results', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'variance_analysis.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# Final Summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nANSWERS TO ALL TASKS:")
print("-" * 50)

print("1. HARD VOTING PREDICTIONS:")
for name, preds, confs in samples:
    final_pred, vote_counts, vote_percentage = hard_voting(preds)
    print(f"   {name}: Class {final_pred} (wins with {vote_percentage:.1f}% of votes)")

print("\n2. SOFT VOTING PREDICTIONS:")
for name, preds, confs in samples:
    final_pred, _, _, final_conf = soft_voting(preds, confs)
    print(f"   {name}: Class {final_pred} (confidence: {final_conf:.3f})")

print(f"\n3. HIGHEST CONFIDENCE: {highest_confidence_sample[0]} ({highest_confidence_sample[1]:.3f})")

print(f"\n4. MOST TRUSTED FOR HIGH CONFIDENCE: {most_trusted[0]} (trust score: {most_trusted[4]:.3f})")

print(f"\n5. HIGHEST DISAGREEMENT: {highest_disagreement[0]} (disagreement: {highest_disagreement[3]:.3f})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
