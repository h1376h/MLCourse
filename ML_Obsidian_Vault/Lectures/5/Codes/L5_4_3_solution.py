import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.special import softmax

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

print("=" * 80)
print("QUESTION 3: OvR (One-vs-Rest) Concrete Example")
print("=" * 80)

# Given OvR classifier outputs
f1 = 1.2  # Class 1 vs Rest
f2 = -0.3 # Class 2 vs Rest
f3 = 0.8  # Class 3 vs Rest

print(f"Given OvR classifier outputs for test point x:")
print(f"$f_1(x) = {f1}$ (Class 1 vs Rest)")
print(f"$f_2(x) = {f2}$ (Class 2 vs Rest)")
print(f"$f_3(x) = {f3}$ (Class 3 vs Rest)")

# Task 1: Standard OvR rule
print("\n" + "="*50)
print("TASK 1: Standard OvR Rule")
print("="*50)

print("STEP-BY-STEP DERIVATION:")
print("1. Recall the OvR decision rule:")
print("   $\\hat{y} = \\arg\\max_k f_k(x)$")
print("   This means: predict the class k that maximizes $f_k(x)$")
print()
print("2. Given decision values:")
print(f"   $f_1(x) = {f1}$")
print(f"   $f_2(x) = {f2}$")
print(f"   $f_3(x) = {f3}$")
print()
print("3. Find the maximum value:")
print(f"   max([{f1}, {f2}, {f3}]) = {max([f1, f2, f3])}")
print()
print("4. Find the index of the maximum:")
print(f"   argmax([{f1}, {f2}, {f3}]) = {np.argmax([f1, f2, f3])}")
print("   Note: Python uses 0-based indexing, so we add 1 for class labels")
print()
print("5. Final prediction:")
print(f"   $\\hat{{y}} = {np.argmax([f1, f2, f3]) + 1}$")

# Standard OvR rule: predict class with highest decision value
decision_values = [f1, f2, f3]
predicted_class = np.argmax(decision_values) + 1  # +1 because classes are 1,2,3
max_decision = max(decision_values)

print(f"\nRESULT: Predicted class = {predicted_class}")
print(f"Mathematical notation: $\\hat{{y}} = \\arg\\max_k f_k(x) = {predicted_class}$")

# Task 2: Confidence score
print("\n" + "="*50)
print("TASK 2: Confidence Score")
print("="*50)

print("STEP-BY-STEP DERIVATION:")
print("1. Confidence measures how decisive the prediction is")
print("2. We use margin-based confidence:")
print("   Confidence = Margin / (|Highest| + |Second Highest|)")
print("   where Margin = Highest - Second Highest")
print()
print("3. Sort decision values in descending order:")
print(f"   Original: [{f1}, {f2}, {f3}]")
sorted_values = sorted(decision_values, reverse=True)
print(f"   Sorted: {sorted_values}")
print()
print("4. Calculate margin:")
print(f"   Highest = {sorted_values[0]}")
print(f"   Second Highest = {sorted_values[1]}")
margin = sorted_values[0] - sorted_values[1]
print(f"   Margin = {sorted_values[0]} - {sorted_values[1]} = {margin}")
print()
print("5. Calculate confidence:")
print(f"   |Highest| = |{sorted_values[0]}| = {abs(sorted_values[0])}")
print(f"   |Second Highest| = |{sorted_values[1]}| = {abs(sorted_values[1])}")
print(f"   Denominator = {abs(sorted_values[0])} + {abs(sorted_values[1])} = {abs(sorted_values[0]) + abs(sorted_values[1])}")
confidence = margin / (abs(sorted_values[0]) + abs(sorted_values[1]) + 1e-8)
print(f"   Confidence = {margin} / {abs(sorted_values[0]) + abs(sorted_values[1])} = {confidence:.3f}")
print()
print("6. Interpretation:")
print(f"   Confidence = {confidence:.3f}")
if confidence < 0.1:
    print("   Low confidence: prediction is uncertain")
elif confidence < 0.3:
    print("   Moderate confidence: prediction is somewhat certain")
else:
    print("   High confidence: prediction is very certain")

# Task 3: Handle ambiguous case
print("\n" + "="*50)
print("TASK 3: Ambiguous Case Handling")
print("="*50)

print("STEP-BY-STEP DERIVATION:")
print("1. Consider the ambiguous case:")
f1_amb = 0.9
f2_amb = -0.3
f3_amb = 0.9
print(f"   $f_1(x) = {f1_amb}$")
print(f"   $f_2(x) = {f2_amb}$")
print(f"   $f_3(x) = {f3_amb}$")
print()
print("2. Apply OvR rule:")
print(f"   argmax([{f1_amb}, {f2_amb}, {f3_amb}])")
print(f"   Both $f_1(x)$ and $f_3(x)$ are equal ({f1_amb})")
print("   This creates a TIE!")
print()
print("3. Strategies to handle ties:")
print()
print("   Strategy 1: Random Selection")
print("   - Choose randomly between tied classes")
print("   - Simple but arbitrary")
print("   - Probability of each tied class = 1/number_of_ties")
print()
print("   Strategy 2: Class Priors")
class_priors = [0.4, 0.3, 0.3]  # Example priors
print(f"   - Use prior knowledge: P(y=1)={class_priors[0]}, P(y=2)={class_priors[1]}, P(y=3)={class_priors[2]}")
print(f"   - Choose class with higher prior: Class {np.argmax([class_priors[0], class_priors[2]]) + 1}")
print("   - Incorporates domain knowledge")
print()
print("   Strategy 3: Decision Value Magnitude")
print(f"   - $|f_1(x)| = |{f1_amb}| = {abs(f1_amb)}$")
print(f"   - $|f_3(x)| = |{f3_amb}| = {abs(f3_amb)}$")
print(f"   - Both equal, so still ambiguous")
print()
print("   Strategy 4: Ensemble Methods")
print("   - Train multiple OvR classifiers")
print("   - Use different random seeds or sampling")
print("   - Combine predictions via voting")
print()
print("   Strategy 5: Cost-Sensitive Decision")
print("   - Consider misclassification costs")
print("   - Choose class with lower expected cost")
print()
print("4. Mathematical formulation:")
print("   For tie between classes i and j:")
print("   - Random: P(y=i) = P(y=j) = 0.5")
print("   - Priors: $P(y=i) \\propto P(y=i|\\text{prior})$")
print("   - Ensemble: P(y=i) = average over multiple classifiers")

# Task 4: Softmax probabilities
print("\n" + "="*50)
print("TASK 4: Softmax Probabilities")
print("="*50)

print("STEP-BY-STEP DERIVATION:")
print("1. Softmax function converts decision values to probabilities:")
print("   $P(y=k|x) = \\exp(f_k(x)) / \\sum_j \\exp(f_j(x))$")
print()
print("2. Given decision values:")
decision_vector = np.array([f1, f2, f3])
print(f"   $f_1(x) = {f1}$")
print(f"   $f_2(x) = {f2}$")
print(f"   $f_3(x) = {f3}$")
print()
print("3. Calculate exponentials:")
exp_f1 = np.exp(f1)
exp_f2 = np.exp(f2)
exp_f3 = np.exp(f3)
print(f"   $\\exp(f_1(x)) = \\exp({f1}) = {exp_f1:.3f}$")
print(f"   $\\exp(f_2(x)) = \\exp({f2}) = {exp_f2:.3f}$")
print(f"   $\\exp(f_3(x)) = \\exp({f3}) = {exp_f3:.3f}$")
print()
print("4. Calculate denominator (sum of exponentials):")
denominator = exp_f1 + exp_f2 + exp_f3
print(f"   $\\sum_j \\exp(f_j(x)) = {exp_f1:.3f} + {exp_f2:.3f} + {exp_f3:.3f} = {denominator:.3f}$")
print()
print("5. Calculate individual probabilities:")
prob1 = exp_f1 / denominator
prob2 = exp_f2 / denominator
prob3 = exp_f3 / denominator
print(f"   $P(y=1|x) = {exp_f1:.3f} / {denominator:.3f} = {prob1:.3f}$")
print(f"   $P(y=2|x) = {exp_f2:.3f} / {denominator:.3f} = {prob2:.3f}$")
print(f"   $P(y=3|x) = {exp_f3:.3f} / {denominator:.3f} = {prob3:.3f}$")
print()
print("6. Verify sum equals 1:")
probabilities = softmax(decision_vector)
print(f"   $\\sum_k P(y=k|x) = {prob1:.3f} + {prob2:.3f} + {prob3:.3f} = {np.sum(probabilities):.6f}$")
print()
print("7. Properties of softmax:")
print("   - Preserves ranking: highest $f_k(x)$ → highest $P(y=k|x)$")
print("   - Sum to 1: $\\sum_k P(y=k|x) = 1$")
print("   - Monotonic: if $f_i(x) > f_j(x)$, then $P(y=i|x) > P(y=j|x)$")
print()
print("8. Mathematical verification:")
print(f"   argmax([{f1}, {f2}, {f3}]) = {np.argmax([f1, f2, f3]) + 1}")
print(f"   argmax([{prob1:.3f}, {prob2:.3f}, {prob3:.3f}]) = {np.argmax([prob1, prob2, prob3]) + 1}")
print("   ✓ Ranking preserved!")

# Task 5: Additional information for resolving ambiguities
print("\n" + "="*50)
print("TASK 5: Additional Information for Resolving Ambiguities")
print("="*50)

print("STEP-BY-STEP ANALYSIS:")
print("1. Problem: OvR can produce ambiguous predictions when:")
print("   - Multiple classes have identical decision values")
print("   - Decision values are very close to each other")
print("   - Classifiers have low confidence")
print()
print("2. Additional information sources:")
print()
print("   A. Class Priors P(y=k):")
print("      - Prior probability of each class")
print("      - Example: P(y=1) = 0.4, P(y=2) = 0.3, P(y=3) = 0.3")
print("      - Use: Choose class with higher prior in case of tie")
print("      - Mathematical: P(y=k|x) ∝ P(y=k) × P(x|y=k)")
print()
print("   B. Feature Importance Weights:")
print("      - Different features may have different reliability")
print("      - Weight decisions based on feature confidence")
print("      - Example: $w_1 = 0.8$, $w_2 = 0.6$, $w_3 = 0.9$")
print("      - Use: Weighted voting or confidence-weighted decisions")
print()
print("   C. Individual Classifier Confidence:")
print("      - Each OvR classifier provides its own confidence")
print("      - Combine confidences for robust decisions")
print("      - Example: $\\text{conf}_1 = 0.8$, $\\text{conf}_2 = 0.6$, $\\text{conf}_3 = 0.7$")
print("      - Use: Confidence-weighted ensemble")
print()
print("   D. Ensemble Methods:")
print("      - Train multiple OvR classifiers")
print("      - Use different random seeds, sampling, or algorithms")
print("      - Combine via voting, averaging, or stacking")
print("      - Mathematical: $\\hat{y} = \\text{majority\\_vote}(\\{\\hat{y}_1, \\hat{y}_2, \\ldots, \\hat{y}_M\\})$")
print()
print("   E. Domain Knowledge:")
print("      - Understanding of class relationships")
print("      - Business rules or constraints")
print("      - Hierarchical class structure")
print("      - Example: Class 1 and 3 are mutually exclusive")
print()
print("   F. Cost Matrix:")
print("      - Different costs for different misclassifications")
print("      - Optimize for minimum expected cost")
print("      - Example: C(i,j) = cost of predicting class i when true is j")
print("      - Mathematical: $\\hat{y} = \\arg\\min_k \\sum_j C(k,j) \\times P(y=j|x)$")
print()
print("3. Mathematical formulation for combining information:")
print("   $P(y=k|x) = \\alpha \\times P_{\\text{ovr}}(y=k|x) + \\beta \\times P_{\\text{prior}}(y=k) + \\gamma \\times P_{\\text{ensemble}}(y=k|x)$")
print("   where $\\alpha + \\beta + \\gamma = 1$ (convex combination)")
print()
print("4. Practical implementation:")
print("   - Start with simple strategies (priors, random)")
print("   - Progress to more complex methods (ensemble, cost-sensitive)")
print("   - Validate on holdout data")
print("   - Monitor performance in production")

# Visualizations
print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Figure 1: Decision values comparison
plt.figure(figsize=(12, 8))

# Subplot 1: Decision values bar plot
plt.subplot(2, 2, 1)
classes = ['Class 1', 'Class 2', 'Class 3']
colors = ['#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(classes, decision_values, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.ylabel('Decision Value $f(x)$')
plt.title('OvR Decision Values')
plt.grid(True, alpha=0.3)

# Highlight the winning class
winning_idx = predicted_class - 1
bars[winning_idx].set_alpha(1.0)
bars[winning_idx].set_edgecolor('red')
bars[winning_idx].set_linewidth(2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, decision_values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Softmax probabilities
plt.subplot(2, 2, 2)
prob_bars = plt.bar(classes, probabilities, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Probability $P(y|x)$')
plt.title('Softmax Probabilities')
plt.grid(True, alpha=0.3)

# Add probability labels
for i, (bar, prob) in enumerate(zip(prob_bars, probabilities)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Margin visualization
plt.subplot(2, 2, 3)
sorted_indices = np.argsort(decision_values)[::-1]
sorted_classes = [classes[i] for i in sorted_indices]
sorted_values = [decision_values[i] for i in sorted_indices]

bars = plt.bar(sorted_classes, sorted_values, color=[colors[i] for i in sorted_indices], alpha=0.7)
plt.ylabel('Decision Value')
plt.title('Decision Values (Sorted)')
plt.grid(True, alpha=0.3)

# Highlight margin
margin_bars = bars[:2]
margin_bars[0].set_alpha(1.0)
margin_bars[0].set_edgecolor('red')
margin_bars[0].set_linewidth(2)
margin_bars[1].set_alpha(1.0)
margin_bars[1].set_edgecolor('blue')
margin_bars[1].set_linewidth(2)

# Add margin annotation
plt.annotate(f'Margin = {margin:.1f}', 
             xy=(0.5, (sorted_values[0] + sorted_values[1])/2),
             xytext=(1.5, sorted_values[0] + 0.2),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, fontweight='bold', color='red')

# Subplot 4: Ambiguous case
plt.subplot(2, 2, 4)
amb_decision_values = [f1_amb, f2_amb, f3_amb]
amb_bars = plt.bar(classes, amb_decision_values, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.ylabel('Decision Value $f(x)$')
plt.title('Ambiguous Case (Tie)')
plt.grid(True, alpha=0.3)

# Highlight tied classes
amb_bars[0].set_alpha(1.0)
amb_bars[0].set_edgecolor('red')
amb_bars[0].set_linewidth(2)
amb_bars[2].set_alpha(1.0)
amb_bars[2].set_edgecolor('red')
amb_bars[2].set_linewidth(2)

# Add tie annotation
plt.annotate('TIE!', 
             xy=(1, f1_amb),
             xytext=(1.5, f1_amb + 0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_decision_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 2: Confidence analysis
plt.figure(figsize=(15, 10))

# Subplot 1: Confidence vs margin
plt.subplot(2, 3, 1)
margins = np.linspace(0, 2, 100)
confidences = margins / (2 + margins)  # Simplified confidence formula
plt.plot(margins, confidences, 'b-', linewidth=2)
plt.scatter([margin], [confidence], color='red', s=100, zorder=5, label=f'Our case: {confidence:.3f}')
plt.xlabel('Margin')
plt.ylabel('Confidence Score')
plt.title('Confidence vs Margin')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Decision space visualization
plt.subplot(2, 3, 2)
# Create a 2D decision space (using f1 and f3 as axes)
f1_range = np.linspace(-2, 2, 50)
f3_range = np.linspace(-2, 2, 50)
F1, F3 = np.meshgrid(f1_range, f3_range)
F2 = np.zeros_like(F1)  # Fixed f2 value

# Determine winning class for each point
winning_class = np.zeros_like(F1)
for i in range(len(f1_range)):
    for j in range(len(f3_range)):
        decisions = [F1[j, i], F2[j, i], F3[j, i]]
        winning_class[j, i] = np.argmax(decisions) + 1

# Plot decision regions
contour = plt.contourf(F1, F3, winning_class, levels=[0.5, 1.5, 2.5, 3.5], 
                       colors=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.6)
plt.contour(F1, F3, winning_class, levels=[1, 2, 3], colors='black', linewidths=1)

# Plot our point
plt.scatter([f1], [f3], color='black', s=200, marker='*', label='Our point', zorder=5)
plt.xlabel('$f_1(x)$')
plt.ylabel('$f_3(x)$')
plt.title('Decision Space ($f_1$ vs $f_3$)')
plt.legend()

# Subplot 3: Probability distribution
plt.subplot(2, 3, 3)
plt.pie(probabilities, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Class Probability Distribution')

# Subplot 4: Decision value comparison with thresholds
plt.subplot(2, 3, 4)
thresholds = np.linspace(-1, 1, 100)
class1_above = np.array([np.sum(f1 > t) for t in thresholds])
class2_above = np.array([np.sum(f2 > t) for t in thresholds])
class3_above = np.array([np.sum(f3 > t) for t in thresholds])

plt.plot(thresholds, class1_above/len(thresholds), label='Class 1', color=colors[0], linewidth=2)
plt.plot(thresholds, class2_above/len(thresholds), label='Class 2', color=colors[1], linewidth=2)
plt.plot(thresholds, class3_above/len(thresholds), label='Class 3', color=colors[2], linewidth=2)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero threshold')
plt.xlabel('Threshold')
plt.ylabel('Fraction above threshold')
plt.title('Decision Value vs Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Confidence distribution
plt.subplot(2, 3, 5)
# Generate random decision values to show confidence distribution
np.random.seed(42)
n_samples = 1000
random_decisions = np.random.normal(0, 1, (n_samples, 3))
random_confidences = []

for decisions in random_decisions:
    sorted_vals = sorted(decisions, reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    conf = margin / (abs(sorted_vals[0]) + abs(sorted_vals[1]) + 1e-8)
    random_confidences.append(conf)

plt.hist(random_confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(x=confidence, color='red', linestyle='--', linewidth=2, 
            label=f'Our confidence: {confidence:.3f}')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 6: Ambiguous case analysis
plt.subplot(2, 3, 6)
amb_probabilities = softmax([f1_amb, f2_amb, f3_amb])
plt.pie(amb_probabilities, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Ambiguous Case Probabilities')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_confidence_analysis.png'), dpi=300, bbox_inches='tight')

# Figure 3: Detailed step-by-step analysis
plt.figure(figsize=(16, 12))

# Step 1: Show decision values
plt.subplot(3, 3, 1)
bars = plt.bar(classes, decision_values, color=colors, alpha=0.7)
bars[predicted_class-1].set_alpha(1.0)
bars[predicted_class-1].set_edgecolor('red')
bars[predicted_class-1].set_linewidth(2)
plt.ylabel('Decision Value')
plt.title('Step 1: Decision Values')
plt.grid(True, alpha=0.3)

# Step 2: Show argmax operation
plt.subplot(3, 3, 2)
plt.text(0.5, 0.5, f'argmax([{f1}, {f2}, {f3}]) = {predicted_class}', 
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Step 2: Argmax Operation')

# Step 3: Show margin calculation
plt.subplot(3, 3, 3)
sorted_vals = sorted(decision_values, reverse=True)
plt.bar(['1st', '2nd', '3rd'], sorted_vals, color=['red', 'blue', 'gray'], alpha=0.7)
plt.ylabel('Decision Value')
plt.title('Step 3: Margin Calculation')
plt.grid(True, alpha=0.3)
plt.text(0.5, sorted_vals[0] + 0.1, f'Margin = {margin:.1f}', 
         ha='center', fontweight='bold', color='red')

# Step 4: Show confidence calculation
plt.subplot(3, 3, 4)
plt.text(0.5, 0.5, f'Confidence = {margin:.1f} / ({abs(sorted_vals[0]):.1f} + {abs(sorted_vals[1]):.1f})\n= {confidence:.3f}', 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Step 4: Confidence Calculation')

# Step 5: Show softmax calculation
plt.subplot(3, 3, 5)
plt.text(0.5, 0.5, f'Softmax([{f1}, {f2}, {f3}])\n= [{probabilities[0]:.3f}, {probabilities[1]:.3f}, {probabilities[2]:.3f}]', 
         ha='center', va='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Step 5: Softmax Calculation')

# Step 6: Show ambiguous case
plt.subplot(3, 3, 6)
amb_bars = plt.bar(classes, [f1_amb, f2_amb, f3_amb], color=colors, alpha=0.7)
amb_bars[0].set_edgecolor('red')
amb_bars[0].set_linewidth(2)
amb_bars[2].set_edgecolor('red')
amb_bars[2].set_linewidth(2)
plt.ylabel('Decision Value')
plt.title('Step 6: Ambiguous Case')
plt.grid(True, alpha=0.3)
plt.text(1, f1_amb + 0.1, 'TIE!', ha='center', fontweight='bold', color='red')

# Step 7: Show tie-breaking strategies
plt.subplot(3, 3, 7)
strategies = ['Random', 'Priors', 'Magnitude', 'Ensemble']
strategy_scores = [0.5, 0.4, 0.5, 0.6]  # Example scores
plt.bar(strategies, strategy_scores, color=['orange', 'green', 'blue', 'purple'], alpha=0.7)
plt.ylabel('Effectiveness Score')
plt.title('Step 7: Tie-breaking Strategies')
plt.grid(True, alpha=0.3)

# Step 8: Show additional information sources
plt.subplot(3, 3, 8)
info_sources = ['Class\nPriors', 'Feature\nWeights', 'Ensemble\nScores', 'Domain\nKnowledge']
info_importance = [0.8, 0.7, 0.9, 0.6]
plt.bar(info_sources, info_importance, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
plt.ylabel('Importance Score')
plt.title('Step 8: Additional Information')
plt.grid(True, alpha=0.3)

# Step 9: Final prediction summary
plt.subplot(3, 3, 9)
summary_text = f"""Final Prediction:
Class: {predicted_class}
Confidence: {confidence:.3f}
Probability: {probabilities[predicted_class-1]:.3f}

Decision Values:
Class 1: {f1:.1f}
Class 2: {f2:.1f}
Class 3: {f3:.1f}"""

plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')
plt.title('Step 9: Final Summary')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ovr_step_by_step.png'), dpi=300, bbox_inches='tight')

print(f"\nVisualizations saved to: {save_dir}")
print("\nGenerated files:")
print("1. ovr_decision_analysis.png - Basic decision analysis")
print("2. ovr_confidence_analysis.png - Confidence and probability analysis")
print("3. ovr_step_by_step.png - Detailed step-by-step breakdown")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Predicted class using standard OvR rule: Class {predicted_class}")
print(f"2. Confidence score: {confidence:.3f}")
print(f"3. Ambiguous case handling: Multiple strategies available")
print(f"4. Softmax probabilities: {probabilities}")
print(f"5. Additional information: Class priors, feature weights, ensemble methods, etc.")
print("="*80)
