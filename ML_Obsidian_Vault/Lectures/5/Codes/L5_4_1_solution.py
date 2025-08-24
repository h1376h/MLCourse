import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle, Circle
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("=" * 80)
print("QUESTION 1: MULTI-CLASS CLASSIFICATION PROBLEM ANALYSIS")
print("=" * 80)

# Given data
classes = ['A', 'B', 'C', 'D']
class_counts = [400, 300, 200, 100]
total_samples = sum(class_counts)

print(f"\nGiven 4-class problem:")
print(f"Classes: {classes}")
print(f"Sample counts: {class_counts}")
print(f"Total samples: {total_samples}")

# ============================================================================
# TASK 1: One-vs-Rest (OvR) Analysis
# ============================================================================
print("\n" + "="*60)
print("TASK 1: ONE-VS-REST (OvR) ANALYSIS")
print("="*60)

ovr_problems = []
ovr_distributions = []

for i, class_name in enumerate(classes):
    positive_samples = class_counts[i]
    negative_samples = total_samples - positive_samples
    
    ovr_problems.append(f"{class_name} vs Rest")
    ovr_distributions.append({
        'positive': positive_samples,
        'negative': negative_samples,
        'total': total_samples
    })
    
    print(f"\n{i+1}. {class_name} vs Rest:")
    print(f"   Positive class ({class_name}): {positive_samples} samples")
    print(f"   Negative class (Rest): {negative_samples} samples")
    print(f"   Total: {total_samples} samples")

# ============================================================================
# TASK 2: One-vs-One (OvO) Analysis
# ============================================================================
print("\n" + "="*60)
print("TASK 2: ONE-VS-ONE (OvO) ANALYSIS")
print("="*60)

ovo_problems = []
ovo_distributions = []

problem_count = 0
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        problem_count += 1
        class1, class2 = classes[i], classes[j]
        samples1, samples2 = class_counts[i], class_counts[j]
        
        ovo_problems.append(f"{class1} vs {class2}")
        ovo_distributions.append({
            'class1': samples1,
            'class2': samples2,
            'total': samples1 + samples2
        })
        
        print(f"\n{problem_count}. {class1} vs {class2}:")
        print(f"   {class1}: {samples1} samples")
        print(f"   {class2}: {samples2} samples")
        print(f"   Total: {samples1 + samples2} samples")

print(f"\nTotal OvO problems: {len(ovo_problems)} = C(4,2) = 6")

# ============================================================================
# TASK 3: Class Imbalance Ratio Analysis
# ============================================================================
print("\n" + "="*60)
print("TASK 3: CLASS IMBALANCE RATIO ANALYSIS")
print("="*60)

imbalance_ratios = []
for i, problem in enumerate(ovr_problems):
    pos = ovr_distributions[i]['positive']
    neg = ovr_distributions[i]['negative']
    ratio = max(pos, neg) / min(pos, neg)
    imbalance_ratios.append(ratio)
    
    print(f"\n{problem}:")
    print(f"   Positive: {pos}, Negative: {neg}")
    print(f"   Imbalance ratio: {ratio:.2f}:1")

# ============================================================================
# TASK 4: OvR vs OvO Imbalance Comparison
# ============================================================================
print("\n" + "="*60)
print("TASK 4: OvR vs OvO IMBALANCE COMPARISON")
print("="*60)

# Calculate imbalance ratios for OvO
ovo_imbalance_ratios = []
for i, problem in enumerate(ovo_problems):
    class1_samples = ovo_distributions[i]['class1']
    class2_samples = ovo_distributions[i]['class2']
    ratio = max(class1_samples, class2_samples) / min(class1_samples, class2_samples)
    ovo_imbalance_ratios.append(ratio)

print(f"\nOvR Imbalance Ratios: {[f'{r:.2f}' for r in imbalance_ratios]}")
print(f"OvO Imbalance Ratios: {[f'{r:.2f}' for r in ovo_imbalance_ratios]}")
print(f"\nOvR Average Imbalance: {np.mean(imbalance_ratios):.2f}")
print(f"OvO Average Imbalance: {np.mean(ovo_imbalance_ratios):.2f}")
print(f"OvR Max Imbalance: {np.max(imbalance_ratios):.2f}")
print(f"OvO Max Imbalance: {np.max(ovo_imbalance_ratios):.2f}")

if np.mean(imbalance_ratios) > np.mean(ovo_imbalance_ratios):
    print("\nCONCLUSION: OvR suffers more from class imbalance in this scenario.")
else:
    print("\nCONCLUSION: OvO suffers more from class imbalance in this scenario.")

# ============================================================================
# TASK 5: Cost-Sensitive Modification
# ============================================================================
print("\n" + "="*60)
print("TASK 5: COST-SENSITIVE MODIFICATION")
print("="*60)

# Calculate class weights for cost-sensitive learning
class_weights = {}
for i, class_name in enumerate(classes):
    # Weight inversely proportional to class frequency
    weight = total_samples / (len(classes) * class_counts[i])
    class_weights[class_name] = weight
    print(f"{class_name}: weight = {weight:.3f}")

# For OvR, calculate weights for each binary problem
ovr_weights = []
for i, problem in enumerate(ovr_problems):
    pos_weight = total_samples / (2 * ovr_distributions[i]['positive'])
    neg_weight = total_samples / (2 * ovr_distributions[i]['negative'])
    ovr_weights.append({'positive': pos_weight, 'negative': neg_weight})
    
    print(f"\n{problem} weights:")
    print(f"   Positive class weight: {pos_weight:.3f}")
    print(f"   Negative class weight: {neg_weight:.3f}")

# ============================================================================
# TASK 6: Tournament Design and Analysis
# ============================================================================
print("\n" + "="*60)
print("TASK 6: TOURNAMENT DESIGN AND ANALYSIS")
print("="*60)

# Tournament data
teams = ['Alpha', 'Beta', 'Gamma', 'Delta']
team_fans = [400, 300, 200, 100]
total_fans = sum(team_fans)

print(f"\nTournament Teams:")
for i, (team, fans) in enumerate(zip(teams, team_fans)):
    print(f"   {team}: {fans} fans ({fans/total_fans*100:.1f}%)")

# Format A: One-vs-Rest Analysis
print(f"\nFORMAT A: One-vs-Rest (Each team vs 'All Others')")
format_a_games = []
for i, team in enumerate(teams):
    team_fans_count = team_fans[i]
    others_fans = total_fans - team_fans_count
    format_a_games.append({
        'team': team,
        'team_fans': team_fans_count,
        'others_fans': others_fans,
        'imbalance_ratio': others_fans / team_fans_count
    })
    
    print(f"\n   {team} vs All Others:")
    print(f"      {team}: {team_fans_count} fans")
    print(f"      All Others: {others_fans} fans")
    print(f"      Imbalance ratio: {others_fans/team_fans_count:.2f}:1")

# Format B: One-vs-One Analysis
print(f"\nFORMAT B: One-vs-One (Round-robin)")
format_b_games = []
game_count = 0
for i in range(len(teams)):
    for j in range(i+1, len(teams)):
        game_count += 1
        team1, team2 = teams[i], teams[j]
        fans1, fans2 = team_fans[i], team_fans[j]
        imbalance = max(fans1, fans2) / min(fans1, fans2)
        
        format_b_games.append({
            'game': f"{team1} vs {team2}",
            'team1': team1,
            'team2': team2,
            'fans1': fans1,
            'fans2': fans2,
            'imbalance_ratio': imbalance
        })
        
        print(f"\n   Game {game_count}: {team1} vs {team2}")
        print(f"      {team1}: {fans1} fans")
        print(f"      {team2}: {fans2} fans")
        print(f"      Imbalance ratio: {imbalance:.2f}:1")

# Fairness scoring system
def calculate_fairness_score(team_fans, total_fans, win_probability=0.5):
    """Calculate fairness score based on team size and win probability"""
    expected_fans = team_fans * win_probability
    fairness_score = expected_fans / total_fans
    return fairness_score

print(f"\nFAIRNESS SCORING SYSTEM:")
print(f"Fairness Score = (Team Fans × Win Probability) / Total Fans")

# Calculate competition balance for each format
print(f"\nCOMPETITION BALANCE ANALYSIS:")

# Format A balance
format_a_balance = []
for game in format_a_games:
    team_fairness = calculate_fairness_score(game['team_fans'], total_fans, 0.5)
    others_fairness = calculate_fairness_score(game['others_fans'], total_fans, 0.5)
    balance = abs(team_fairness - others_fairness)
    format_a_balance.append(balance)
    
    print(f"\n   {game['team']} vs All Others:")
    print(f"      {game['team']} fairness: {team_fairness:.4f}")
    print(f"      All Others fairness: {others_fairness:.4f}")
    print(f"      Balance score: {balance:.4f}")

# Format B balance
format_b_balance = []
for game in format_b_games:
    team1_fairness = calculate_fairness_score(game['fans1'], total_fans, 0.5)
    team2_fairness = calculate_fairness_score(game['fans2'], total_fans, 0.5)
    balance = abs(team1_fairness - team2_fairness)
    format_b_balance.append(balance)
    
    print(f"\n   {game['game']}:")
    print(f"      {game['team1']} fairness: {team1_fairness:.4f}")
    print(f"      {game['team2']} fairness: {team2_fairness:.4f}")
    print(f"      Balance score: {balance:.4f}")

avg_format_a_balance = np.mean(format_a_balance)
avg_format_b_balance = np.mean(format_b_balance)

print(f"\nFORMAT COMPARISON:")
print(f"Format A (OvR) average balance: {avg_format_a_balance:.4f}")
print(f"Format B (OvO) average balance: {avg_format_b_balance:.4f}")

if avg_format_b_balance < avg_format_a_balance:
    print(f"CONCLUSION: Format B (OvO) gives smaller teams better chances to win.")
else:
    print(f"CONCLUSION: Format A (OvR) gives smaller teams better chances to win.")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# 1. Class Distribution Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Multi-class Classification Problem Analysis', fontsize=16, fontweight='bold')

# Original class distribution
ax1.bar(classes, class_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('Original Class Distribution')
ax1.set_ylabel('Number of Samples')
ax1.set_xlabel('Classes')
for i, v in enumerate(class_counts):
    ax1.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# OvR class distributions
ovr_pos = [d['positive'] for d in ovr_distributions]
ovr_neg = [d['negative'] for d in ovr_distributions]

x = np.arange(len(ovr_problems))
width = 0.35

ax2.bar(x - width/2, ovr_pos, width, label='Positive Class', color='#FF6B6B')
ax2.bar(x + width/2, ovr_neg, width, label='Negative Class (Rest)', color='#4ECDC4')
ax2.set_title('One-vs-Rest (OvR) Class Distributions')
ax2.set_ylabel('Number of Samples')
ax2.set_xlabel('Binary Problems')
ax2.set_xticks(x)
ax2.set_xticklabels(ovr_problems, rotation=45)
ax2.legend()

# OvO class distributions
ovo_class1 = [d['class1'] for d in ovo_distributions]
ovo_class2 = [d['class2'] for d in ovo_distributions]

x = np.arange(len(ovo_problems))
ax3.bar(x - width/2, ovo_class1, width, label='Class 1', color='#45B7D1')
ax3.bar(x + width/2, ovo_class2, width, label='Class 2', color='#96CEB4')
ax3.set_title('One-vs-One (OvO) Class Distributions')
ax3.set_ylabel('Number of Samples')
ax3.set_xlabel('Binary Problems')
ax3.set_xticks(x)
ax3.set_xticklabels(ovo_problems, rotation=45)
ax3.legend()

# Imbalance ratio comparison
x = np.arange(len(classes))
ax4.bar(x - width/2, imbalance_ratios, width, label='OvR Imbalance', color='#FF6B6B', alpha=0.7)
# For OvO, we need to show average imbalance per class, not per problem
ovo_avg_by_class = []
for i, cls in enumerate(classes):
    class_imbalances = []
    for j, problem in enumerate(ovo_problems):
        if cls in problem:
            class_imbalances.append(ovo_imbalance_ratios[j])
    ovo_avg_by_class.append(np.mean(class_imbalances))
ax4.bar(x + width/2, ovo_avg_by_class, width, label='OvO Avg Imbalance', color='#4ECDC4', alpha=0.7)
ax4.set_title('Class Imbalance Ratio Comparison')
ax4.set_ylabel('Imbalance Ratio')
ax4.set_xlabel('Classes')
ax4.set_xticks(x)
ax4.set_xticklabels(classes)
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'class_distribution_analysis.png'), dpi=300, bbox_inches='tight')

# 2. Tournament Analysis Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Tournament Format Analysis', fontsize=16, fontweight='bold')

# Team fan distribution
ax1.pie(team_fans, labels=teams, autopct='%1.1f%%', startangle=90, 
        colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('Team Fan Distribution')

# Format A vs Format B comparison
format_a_imbalances = [game['imbalance_ratio'] for game in format_a_games]
format_b_imbalances = [game['imbalance_ratio'] for game in format_b_games]

x = np.arange(len(teams))
ax2.bar(x - width/2, format_a_imbalances, width, label='Format A (OvR)', color='#FF6B6B', alpha=0.7)
ax2.bar(x + width/2, format_b_imbalances[:4], width, label='Format B (OvO) - First 4 games', color='#4ECDC4', alpha=0.7)
ax2.set_title('Tournament Imbalance Ratios')
ax2.set_ylabel('Imbalance Ratio')
ax2.set_xlabel('Teams')
ax2.set_xticks(x)
ax2.set_xticklabels(teams)
ax2.legend()

# Competition balance comparison
ax3.bar(['Format A (OvR)', 'Format B (OvO)'], [avg_format_a_balance, avg_format_b_balance], 
        color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('Average Competition Balance')
ax3.set_ylabel('Balance Score (Lower is Better)')
for i, v in enumerate([avg_format_a_balance, avg_format_b_balance]):
    ax3.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# Fairness scores for each team
team_fairness_scores = [calculate_fairness_score(fans, total_fans) for fans in team_fans]
ax4.bar(teams, team_fairness_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax4.set_title('Team Fairness Scores')
ax4.set_ylabel('Fairness Score')
ax4.set_xlabel('Teams')
for i, v in enumerate(team_fairness_scores):
    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tournament_analysis.png'), dpi=300, bbox_inches='tight')

# 3. Cost-sensitive weights visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Cost-Sensitive Learning Weights', fontsize=16, fontweight='bold')

# Class weights
ax1.bar(classes, list(class_weights.values()), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('Class Weights for Cost-Sensitive Learning')
ax1.set_ylabel('Weight')
ax1.set_xlabel('Classes')
for i, v in enumerate(class_weights.values()):
    ax1.text(i, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# OvR weights comparison
ovr_pos_weights = [w['positive'] for w in ovr_weights]
ovr_neg_weights = [w['negative'] for w in ovr_weights]

x = np.arange(len(ovr_problems))
ax2.bar(x - width/2, ovr_pos_weights, width, label='Positive Class Weight', color='#FF6B6B')
ax2.bar(x + width/2, ovr_neg_weights, width, label='Negative Class Weight', color='#4ECDC4')
ax2.set_title('OvR Binary Classifier Weights')
ax2.set_ylabel('Weight')
ax2.set_xlabel('Binary Problems')
ax2.set_xticks(x)
ax2.set_xticklabels(ovr_problems, rotation=45)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_sensitive_weights.png'), dpi=300, bbox_inches='tight')

print(f"\n" + "="*80)
print("VISUALIZATIONS GENERATED")
print("="*80)
print(f"Plots saved to: {save_dir}")
print("1. class_distribution_analysis.png - Class distributions and imbalance analysis")
print("2. tournament_analysis.png - Tournament format comparison")
print("3. cost_sensitive_weights.png - Cost-sensitive learning weights")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print(f"\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nClass Distribution:")
for i, (cls, count) in enumerate(zip(classes, class_counts)):
    print(f"   {cls}: {count} samples ({count/total_samples*100:.1f}%)")

print(f"\nOvR Analysis:")
print(f"   Number of binary classifiers: {len(ovr_problems)}")
print(f"   Average imbalance ratio: {np.mean(imbalance_ratios):.2f}")
print(f"   Maximum imbalance ratio: {np.max(imbalance_ratios):.2f}")

print(f"\nOvO Analysis:")
print(f"   Number of binary classifiers: {len(ovo_problems)}")
print(f"   Average imbalance ratio: {np.mean(ovo_imbalance_ratios):.2f}")
print(f"   Maximum imbalance ratio: {np.max(ovo_imbalance_ratios):.2f}")

print(f"\nTournament Analysis:")
print(f"   Format A (OvR) average balance: {avg_format_a_balance:.4f}")
print(f"   Format B (OvO) average balance: {avg_format_b_balance:.4f}")
print(f"   Better format for small teams: {'Format B (OvO)' if avg_format_b_balance < avg_format_a_balance else 'Format A (OvR)'}")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
