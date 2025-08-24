import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import comb
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("FEATURE SELECTION STRATEGY GAME - DETAILED SOLUTION")
print("=" * 80)

# Game parameters
total_features = 100
useful_features = 15
useless_features = total_features - useful_features
features_to_select = 20
useful_points = 10
useless_points = -2

print(f"\nGame Setup:")
print(f"- Total features: {total_features}")
print(f"- Useful features: {useful_features}")
print(f"- Useless features: {useless_features}")
print(f"- Features to select: {features_to_select}")
print(f"- Points per useful feature: +{useful_points}")
print(f"- Points per useless feature: {useless_points}")

# Question 1: Best possible score
print("\n" + "="*60)
print("QUESTION 1: BEST POSSIBLE SCORE")
print("="*60)

max_useful_selected = min(useful_features, features_to_select)
min_useless_selected = features_to_select - max_useful_selected

best_score = max_useful_selected * useful_points + min_useless_selected * useless_points

print(f"\nTo maximize score, select as many useful features as possible:")
print(f"- Maximum useful features we can select: min({useful_features}, {features_to_select}) = {max_useful_selected}")
print(f"- Remaining selections must be useless: {features_to_select} - {max_useful_selected} = {min_useless_selected}")
print(f"- Best score = {max_useful_selected} × {useful_points} + {min_useless_selected} × {useless_points}")
print(f"- Best score = {max_useful_selected * useful_points} + {min_useless_selected * useless_points} = {best_score}")

# Question 2: Worst possible score
print("\n" + "="*60)
print("QUESTION 2: WORST POSSIBLE SCORE")
print("="*60)

min_useful_selected = max(0, features_to_select - useless_features)
max_useless_selected = features_to_select - min_useful_selected

worst_score = min_useful_selected * useful_points + max_useless_selected * useless_points

print(f"\nTo minimize score, select as few useful features as possible:")
print(f"- Minimum useful features we must select: max(0, {features_to_select} - {useless_features}) = {min_useful_selected}")
print(f"- Maximum useless features we can select: {features_to_select} - {min_useful_selected} = {max_useless_selected}")
print(f"- Worst score = {min_useful_selected} × {useful_points} + {max_useless_selected} × {useless_points}")
print(f"- Worst score = {min_useful_selected * useful_points} + {max_useless_selected * useless_points} = {worst_score}")

# Question 3: Expected score with random selection
print("\n" + "="*60)
print("QUESTION 3: EXPECTED SCORE WITH RANDOM SELECTION")
print("="*60)

# Using hypergeometric distribution
# X = number of useful features selected (follows hypergeometric distribution)
expected_useful = features_to_select * (useful_features / total_features)
expected_useless = features_to_select - expected_useful

expected_score_random = expected_useful * useful_points + expected_useless * useless_points

print(f"\nWith random selection, the number of useful features follows a hypergeometric distribution:")
print(f"- Population size (N): {total_features}")
print(f"- Success states (K): {useful_features}")
print(f"- Sample size (n): {features_to_select}")
print(f"\nExpected number of useful features selected:")
print(f"E[X] = n × (K/N) = {features_to_select} × ({useful_features}/{total_features}) = {expected_useful:.2f}")
print(f"\nExpected number of useless features selected:")
print(f"E[Y] = {features_to_select} - E[X] = {features_to_select} - {expected_useful:.2f} = {expected_useless:.2f}")
print(f"\nExpected score:")
print(f"E[Score] = E[X] × {useful_points} + E[Y] × {useless_points}")
print(f"E[Score] = {expected_useful:.2f} × {useful_points} + {expected_useless:.2f} × {useless_points} = {expected_score_random:.2f}")

# Question 4: Strategy to maximize score
print("\n" + "="*60)
print("QUESTION 4: STRATEGY TO MAXIMIZE SCORE")
print("="*60)

print(f"\nOptimal Strategy:")
print(f"1. Identify all {useful_features} useful features with 100% accuracy")
print(f"2. Select all {useful_features} useful features")
print(f"3. Select {features_to_select - useful_features} additional features from the remaining {useless_features} useless features")
print(f"4. This guarantees the maximum possible score of {best_score}")
print(f"\nIn practice, since perfect identification is impossible:")
print(f"- Use feature selection methods (correlation analysis, mutual information, etc.)")
print(f"- Apply domain expertise to identify likely useful features")
print(f"- Use cross-validation to evaluate feature subsets")
print(f"- Consider ensemble methods that can handle some useless features")

# Question 5: Probability of positive score with random selection
print("\n" + "="*60)
print("QUESTION 5: PROBABILITY OF POSITIVE SCORE WITH RANDOM SELECTION")
print("="*60)

# Calculate probability distribution for random selection
possible_useful_counts = np.arange(max(0, features_to_select - useless_features), 
                                  min(useful_features, features_to_select) + 1)

probabilities = []
scores = []

print(f"\nCalculating probabilities for each possible outcome:")
print(f"Useful features selected | Probability | Score")
print(f"-" * 45)

for k in possible_useful_counts:
    # Hypergeometric probability
    prob = comb(useful_features, k) * comb(useless_features, features_to_select - k) / comb(total_features, features_to_select)
    score = k * useful_points + (features_to_select - k) * useless_points
    
    probabilities.append(prob)
    scores.append(score)
    
    print(f"{k:^23} | {prob:^11.4f} | {score:^5}")

probabilities = np.array(probabilities)
scores = np.array(scores)

# Probability of positive score
positive_score_prob = np.sum(probabilities[scores > 0])

print(f"\nProbability of positive score: {positive_score_prob:.4f} = {positive_score_prob*100:.2f}%")

# Find threshold
threshold_useful = None
for i, score in enumerate(scores):
    if score > 0:
        threshold_useful = possible_useful_counts[i]
        break

if threshold_useful is not None:
    print(f"Need at least {threshold_useful} useful features for positive score")
    threshold_score = threshold_useful * useful_points + (features_to_select - threshold_useful) * useless_points
    print(f"Threshold score: {threshold_useful} × {useful_points} + {features_to_select - threshold_useful} × {useless_points} = {threshold_score}")

# Question 6: Imperfect selection with 80% accuracy
print("\n" + "="*60)
print("QUESTION 6: IMPERFECT SELECTION (80% ACCURACY)")
print("="*60)

accuracy = 0.8
false_positive_rate = 0.2

print(f"\nImperfect selection parameters:")
print(f"- True positive rate (correctly identify useful): {accuracy}")
print(f"- False positive rate (incorrectly identify useless as useful): {false_positive_rate}")

# Calculate expected performance with imperfect selection
# Strategy: Select top 20 features based on imperfect identification

# Expected number of truly useful features among those identified as useful
# True useful identified correctly: useful_features * accuracy
# Useless features incorrectly identified as useful: useless_features * false_positive_rate
# Total identified as useful: useful_features * accuracy + useless_features * false_positive_rate

true_useful_identified = useful_features * accuracy
false_useful_identified = useless_features * false_positive_rate
total_identified_useful = true_useful_identified + false_useful_identified

print(f"\nExpected identification results:")
print(f"- Truly useful features correctly identified: {useful_features} × {accuracy} = {true_useful_identified}")
print(f"- Useless features incorrectly identified as useful: {useless_features} × {false_positive_rate} = {false_useful_identified}")
print(f"- Total features identified as useful: {true_useful_identified} + {false_useful_identified} = {total_identified_useful}")

# If we select top 20 from identified useful features
if total_identified_useful >= features_to_select:
    # We have enough identified features, select proportionally
    selected_true_useful = (true_useful_identified / total_identified_useful) * features_to_select
    selected_false_useful = (false_useful_identified / total_identified_useful) * features_to_select
else:
    # We don't have enough identified features, take all and fill randomly
    selected_true_useful = true_useful_identified
    selected_false_useful = false_useful_identified
    remaining_to_select = features_to_select - total_identified_useful
    
    # Remaining features to select from unidentified pool
    remaining_useful = useful_features - true_useful_identified
    remaining_useless = useless_features - false_useful_identified
    total_remaining = remaining_useful + remaining_useless
    
    # Select proportionally from remaining
    additional_useful = (remaining_useful / total_remaining) * remaining_to_select
    additional_useless = (remaining_useless / total_remaining) * remaining_to_select
    
    selected_true_useful += additional_useful
    selected_false_useful += additional_useless

expected_score_imperfect = selected_true_useful * useful_points + selected_false_useful * useless_points

print(f"\nWith imperfect selection strategy:")
if total_identified_useful >= features_to_select:
    print(f"- Select {features_to_select} features from {total_identified_useful:.1f} identified as useful")
    print(f"- Expected truly useful selected: ({true_useful_identified:.1f}/{total_identified_useful:.1f}) × {features_to_select} = {selected_true_useful:.2f}")
    print(f"- Expected falsely identified selected: ({false_useful_identified:.1f}/{total_identified_useful:.1f}) × {features_to_select} = {selected_false_useful:.2f}")
else:
    print(f"- Take all {total_identified_useful:.1f} identified features")
    print(f"- Randomly select {features_to_select - total_identified_useful:.1f} more from remaining {total_remaining} features")

print(f"- Expected truly useful features selected: {selected_true_useful:.2f}")
print(f"- Expected useless features selected: {selected_false_useful:.2f}")
print(f"- Expected score: {selected_true_useful:.2f} × {useful_points} + {selected_false_useful:.2f} × {useless_points} = {expected_score_imperfect:.2f}")

# Comparison
print(f"\n" + "="*60)
print("COMPARISON OF STRATEGIES")
print("="*60)

print(f"\nScore Comparison:")
print(f"- Perfect selection: {best_score}")
print(f"- Imperfect selection (80% accuracy): {expected_score_imperfect:.2f}")
print(f"- Random selection: {expected_score_random:.2f}")
print(f"- Worst case: {worst_score}")

improvement_over_random = expected_score_imperfect - expected_score_random
improvement_percentage = (improvement_over_random / abs(expected_score_random)) * 100

print(f"\nImperfect vs Random:")
print(f"- Improvement: {improvement_over_random:.2f} points")
print(f"- Percentage improvement: {improvement_percentage:.1f}%")

# Create visualizations
print(f"\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Visualization 1: Score distribution for random selection
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Probability distribution
ax1.bar(possible_useful_counts, probabilities, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(expected_useful, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_useful:.1f}')
ax1.set_xlabel('Number of Useful Features Selected')
ax1.set_ylabel('Probability')
ax1.set_title('Distribution of Useful Features Selected (Random)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Score distribution
score_colors = ['lightcoral' if s <= 0 else 'lightgreen' for s in scores]
bars = ax2.bar(possible_useful_counts, scores, alpha=0.7, color=score_colors, edgecolor='black')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axvline(expected_useful, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_useful:.1f}')
ax2.set_xlabel('Number of Useful Features Selected')
ax2.set_ylabel('Score')
ax2.set_title('Score vs Useful Features Selected')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add score values on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
             f'{score}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

# Plot 3: Strategy comparison
strategies = ['Perfect', 'Imperfect\n(80% accuracy)', 'Random', 'Worst Case']
strategy_scores = [best_score, expected_score_imperfect, expected_score_random, worst_score]
colors = ['gold', 'orange', 'lightblue', 'lightcoral']

bars = ax3.bar(strategies, strategy_scores, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Expected Score')
ax3.set_title('Comparison of Selection Strategies')
ax3.grid(True, alpha=0.3)

# Add score values on bars
for bar, score in zip(bars, strategy_scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -8),
             f'{score:.1f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

# Plot 4: Cumulative probability of achieving at least a certain score
sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
sorted_scores = scores[sorted_indices]
sorted_probs = probabilities[sorted_indices]
cumulative_probs = np.cumsum(sorted_probs)

ax4.step(sorted_scores, cumulative_probs, where='post', linewidth=2, color='blue')
ax4.fill_between(sorted_scores, cumulative_probs, alpha=0.3, color='blue', step='post')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Score = 0')
ax4.axhline(positive_score_prob, color='red', linestyle=':', linewidth=2, 
           label=f'P(Score $\\geq$ 0) = {positive_score_prob:.3f}')
ax4.set_xlabel('Score')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Probability of Achieving At Least Score X')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_analysis.png'), dpi=300, bbox_inches='tight')
print(f"Saved: feature_selection_analysis.png")

# Visualization 2: Detailed comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Expected useful features by strategy
strategies_detailed = ['Perfect\nSelection', 'Imperfect\nSelection', 'Random\nSelection']
useful_selected = [max_useful_selected, selected_true_useful, expected_useful]
useless_selected = [min_useless_selected, selected_false_useful, expected_useless]

x = np.arange(len(strategies_detailed))
width = 0.35

bars1 = ax1.bar(x - width/2, useful_selected, width, label='Useful Features', 
                color='lightgreen', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, useless_selected, width, label='Useless Features', 
                color='lightcoral', alpha=0.7, edgecolor='black')

ax1.set_xlabel('Strategy')
ax1.set_ylabel('Expected Number of Features')
ax1.set_title('Expected Feature Composition by Strategy')
ax1.set_xticks(x)
ax1.set_xticklabels(strategies_detailed)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Accuracy analysis for imperfect selection
accuracy_range = np.linspace(0.5, 1.0, 51)
expected_scores_accuracy = []

for acc in accuracy_range:
    fpr = 1 - acc  # Assuming false positive rate = 1 - accuracy for simplicity
    
    true_useful_id = useful_features * acc
    false_useful_id = useless_features * fpr
    total_id_useful = true_useful_id + false_useful_id
    
    if total_id_useful >= features_to_select:
        sel_true_useful = (true_useful_id / total_id_useful) * features_to_select
        sel_false_useful = (false_useful_id / total_id_useful) * features_to_select
    else:
        sel_true_useful = true_useful_id
        sel_false_useful = false_useful_id
        remaining = features_to_select - total_id_useful
        rem_useful = useful_features - true_useful_id
        rem_useless = useless_features - false_useful_id
        total_rem = rem_useful + rem_useless
        
        if total_rem > 0:
            sel_true_useful += (rem_useful / total_rem) * remaining
            sel_false_useful += (rem_useless / total_rem) * remaining
    
    expected_score = sel_true_useful * useful_points + sel_false_useful * useless_points
    expected_scores_accuracy.append(expected_score)

ax2.plot(accuracy_range * 100, expected_scores_accuracy, linewidth=2, color='blue')
ax2.axhline(expected_score_random, color='red', linestyle='--', linewidth=2, 
           label=f'Random Selection: {expected_score_random:.1f}')
ax2.axhline(best_score, color='green', linestyle='--', linewidth=2, 
           label=f'Perfect Selection: {best_score}')
ax2.axvline(80, color='orange', linestyle=':', linewidth=2, 
           label=f'80% Accuracy: {expected_score_imperfect:.1f}')

ax2.set_xlabel('Feature Identification Accuracy (%)')
ax2.set_ylabel('Expected Score')
ax2.set_title('Expected Score vs Identification Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(50, 100)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'strategy_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved: strategy_comparison.png")

# Visualization 3: Game theory visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create a heatmap showing score for different combinations
useful_range = np.arange(0, min(useful_features, features_to_select) + 1)
score_matrix = []

for u in useful_range:
    useless_count = features_to_select - u
    score = u * useful_points + useless_count * useless_points
    score_matrix.append(score)

# Create bar chart with color coding
colors = ['red' if s < 0 else 'yellow' if s == 0 else 'green' for s in score_matrix]
bars = ax.bar(useful_range, score_matrix, color=colors, alpha=0.7, edgecolor='black')

# Add score labels
for i, (bar, score) in enumerate(zip(bars, score_matrix)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (2 if height >= 0 else -8),
            f'{score}', ha='center', va='bottom' if height >= 0 else 'top', 
            fontweight='bold', fontsize=10)

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Number of Useful Features Selected')
ax.set_ylabel('Score')
ax.set_title('Score for Each Possible Outcome\n(Selecting 20 Features Total)')
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Negative Score'),
                  Patch(facecolor='yellow', alpha=0.7, label='Zero Score'),
                  Patch(facecolor='green', alpha=0.7, label='Positive Score')]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'score_outcomes.png'), dpi=300, bbox_inches='tight')
print(f"Saved: score_outcomes.png")

print(f"\nAll visualizations saved to: {save_dir}")

# Summary table
print(f"\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

print(f"{'Strategy':<20} {'Expected Score':<15} {'Improvement over Random':<25}")
print(f"{'-'*20} {'-'*15} {'-'*25}")
print(f"{'Perfect Selection':<20} {best_score:<15} {best_score - expected_score_random:<25.1f}")
print(f"{'Imperfect (80%)':<20} {expected_score_imperfect:<15.1f} {expected_score_imperfect - expected_score_random:<25.1f}")
print(f"{'Random Selection':<20} {expected_score_random:<15.1f} {'0.0 (baseline)':<25}")
print(f"{'Worst Case':<20} {worst_score:<15} {worst_score - expected_score_random:<25.1f}")

print(f"\nKey Insights:")
print(f"- Perfect feature identification provides {best_score - expected_score_random:.1f} point improvement")
print(f"- 80% accurate identification provides {expected_score_imperfect - expected_score_random:.1f} point improvement")
print(f"- Random selection has {positive_score_prob:.1%} chance of positive score")
print(f"- Need at least {threshold_useful} useful features for positive score")
