import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_1_Quiz_14")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False  # Disable LaTeX to avoid issues
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

print("Question 14: Feature Cost Analysis")
print("=" * 50)

# Define problem parameters as variables
print("Problem Parameters:")
print("-" * 40)

# Individual features
feature_a_cost = 10
feature_a_improvement = 2  # 2%
feature_b_cost = 100
feature_b_improvement = 5  # 5%

# Budget and feature sets
budget = 500
base_accuracy = 80

feature_sets = {
    'Set 1': {'cost': 200, 'accuracy': 85},
    'Set 2': {'cost': 300, 'accuracy': 87},
    'Set 3': {'cost': 400, 'accuracy': 89}
}

print(f"Individual Features:")
print(f"  Feature A: Cost = {feature_a_cost}, Improvement = {feature_a_improvement}%")
print(f"  Feature B: Cost = {feature_b_cost}, Improvement = {feature_b_improvement}%")
print(f"Budget: {budget}")
print(f"Base accuracy (without features): {base_accuracy}%")
print()

# 1. Feature A vs Feature B Analysis
print("1. Feature A vs Feature B Analysis:")
print("-" * 40)

# Calculate cost-effectiveness (improvement per cost unit)
ce_a = feature_a_improvement / feature_a_cost
ce_b = feature_b_improvement / feature_b_cost

print(f"Cost-effectiveness calculation:")
print(f"  CE = Improvement / Cost")
print(f"  CE_A = {feature_a_improvement}% / {feature_a_cost} = {ce_a:.3f}% per cost unit")
print(f"  CE_B = {feature_b_improvement}% / {feature_b_cost} = {ce_b:.3f}% per cost unit")

if ce_a > ce_b:
    print(f"Result: Feature A is more cost-effective ({ce_a:.3f} vs {ce_b:.3f})")
    better_feature = "Feature A"
    better_ce = ce_a
else:
    print(f"Result: Feature B is more cost-effective ({ce_b:.3f} vs {ce_a:.3f})")
    better_feature = "Feature B"
    better_ce = ce_b

# Visualize Feature A vs Feature B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart comparison
features = ['Feature A', 'Feature B']
costs = [feature_a_cost, feature_b_cost]
improvements = [feature_a_improvement, feature_b_improvement]
cost_effectiveness = [ce_a, ce_b]

# Cost comparison
bars1 = ax1.bar(features, costs, color=['lightblue', 'lightcoral'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Cost')
ax1.set_title('Feature Cost Comparison')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, cost in zip(bars1, costs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{cost}', ha='center', va='bottom', fontweight='bold')

# Improvement comparison
bars2 = ax2.bar(features, improvements, color=['lightgreen', 'lightyellow'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy Improvement (%)')
ax2.set_title('Feature Performance Comparison')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, improvement in zip(bars2, improvements):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{improvement}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_comparison.png'), dpi=300, bbox_inches='tight')

# 2. Cost-effectiveness analysis
print("\n2. Cost-effectiveness Analysis:")
print("-" * 40)

# Create a comprehensive cost-effectiveness plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot points for each feature
ax.scatter(feature_a_cost, feature_a_improvement, s=200, color='blue', 
           label=f'Feature A (CE: {ce_a:.3f})', zorder=5)
ax.scatter(feature_b_cost, feature_b_improvement, s=200, color='red', 
           label=f'Feature B (CE: {ce_b:.3f})', zorder=5)

# Add cost-effectiveness lines
x_range = np.linspace(0, 120, 100)
ce_lines = [0.1, 0.2, 0.3, 0.4, 0.5]
colors = ['lightgray', 'gray', 'darkgray', 'black', 'darkred']

for i, ce in enumerate(ce_lines):
    y = ce * x_range
    ax.plot(x_range, y, '--', color=colors[i], alpha=0.7, 
            label=f'CE = {ce:.1f}% per cost unit')

# Highlight the most cost-effective feature
if ce_a > ce_b:
    ax.scatter(feature_a_cost, feature_a_improvement, s=300, color='blue', 
               edgecolor='gold', linewidth=3, zorder=6, label='Most Cost-Effective')
else:
    ax.scatter(feature_b_cost, feature_b_improvement, s=300, color='red', 
               edgecolor='gold', linewidth=3, zorder=6, label='Most Cost-Effective')

ax.set_xlabel('Cost')
ax.set_ylabel('Accuracy Improvement (%)')
ax.set_title('Cost-Effectiveness Analysis')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(0, 120)
ax.set_ylim(0, 6)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cost_effectiveness_analysis.png'), dpi=300, bbox_inches='tight')

# 3. Budget optimization problem
print("\n3. Budget Optimization Problem:")
print("-" * 40)

print(f"Budget: {budget}")
print(f"Base accuracy (without features): {base_accuracy}%")
print()

# Calculate improvements and cost-effectiveness
for set_name, set_data in feature_sets.items():
    improvement = set_data['accuracy'] - base_accuracy
    cost_effectiveness = improvement / set_data['cost']
    set_data['improvement'] = improvement
    set_data['cost_effectiveness'] = cost_effectiveness
    
    print(f"{set_name}:")
    print(f"  Cost: {set_data['cost']}")
    print(f"  Accuracy: {set_data['accuracy']}%")
    print(f"  Improvement: {improvement}%")
    print(f"  Cost-effectiveness: {cost_effectiveness:.4f}% per cost unit")
    print()

# Find the best value for money
best_set = max(feature_sets.keys(), key=lambda x: feature_sets[x]['cost_effectiveness'])
print(f"Best value for money: {best_set}")
print(f"Cost-effectiveness: {feature_sets[best_set]['cost_effectiveness']:.4f}% per cost unit")

# 4. Budget allocation visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Cost-effectiveness comparison
set_names = list(feature_sets.keys())
cost_effectiveness_values = [feature_sets[name]['cost_effectiveness'] for name in set_names]
improvements = [feature_sets[name]['improvement'] for name in set_names]
costs = [feature_sets[name]['cost'] for name in set_names]

# Bar chart for cost-effectiveness
bars = ax1.bar(set_names, cost_effectiveness_values, 
                color=['lightblue', 'lightgreen', 'lightcoral'], 
                alpha=0.7, edgecolor='black')
ax1.set_ylabel('Cost-Effectiveness (% per cost unit)')
ax1.set_title('Cost-Effectiveness Comparison')
ax1.grid(True, alpha=0.3)

# Highlight the best option
best_idx = set_names.index(best_set)
bars[best_idx].set_color('gold')
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(2)

# Add value labels
for bar, value in zip(bars, cost_effectiveness_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Budget utilization
ax2.pie(costs, labels=set_names, autopct='%1.1f%%', startangle=90,
         colors=['lightblue', 'lightgreen', 'lightcoral'])
ax2.set_title('Budget Utilization by Feature Set')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'budget_optimization.png'), dpi=300, bbox_inches='tight')

# 5. Comprehensive analysis with Pareto frontier
print("\n4. Comprehensive Analysis:")
print("-" * 40)

# Create a comprehensive scatter plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all feature sets
colors = ['blue', 'green', 'red']
for i, (set_name, set_data) in enumerate(feature_sets.items()):
    ax.scatter(set_data['cost'], set_data['accuracy'], s=200, 
               color=colors[i], label=f'{set_name}', zorder=5)

# Add the individual features
ax.scatter(feature_a_cost, base_accuracy + feature_a_improvement, s=150, 
           color='purple', marker='s', label='Feature A', zorder=5)
ax.scatter(feature_b_cost, base_accuracy + feature_b_improvement, s=150, 
           color='orange', marker='^', label='Feature B', zorder=5)

# Add budget constraint line
budget_line_x = [0, budget]
budget_line_y = [base_accuracy, base_accuracy]
ax.plot(budget_line_x, budget_line_y, 'k--', alpha=0.7, label='Budget Constraint')

# Add cost-effectiveness lines
x_range = np.linspace(0, budget, 100)
for ce in [0.1, 0.2, 0.3, 0.4]:
    y = base_accuracy + ce * x_range
    ax.plot(x_range, y, ':', color='gray', alpha=0.5, 
            label=f'CE = {ce:.1f}% per cost unit')

# Highlight the best option
ax.scatter(feature_sets[best_set]['cost'], feature_sets[best_set]['accuracy'], 
           s=300, color=colors[set_names.index(best_set)], 
           edgecolor='gold', linewidth=3, zorder=6, label='Best Value')

ax.set_xlabel('Cost')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Feature Selection: Cost vs Performance')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlim(0, budget + 50)
ax.set_ylim(base_accuracy - 2, max([set_data['accuracy'] for set_data in feature_sets.values()]) + 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')

# 6. Decision matrix
print("\n5. Decision Matrix:")
print("-" * 40)

# Create decision matrix
decision_data = []
for set_name, set_data in feature_sets.items():
    decision_data.append({
        'Feature Set': set_name,
        'Cost': set_data['cost'],
        'Accuracy (%)': set_data['accuracy'],
        'Improvement (%)': set_data['improvement'],
        'Cost-Effectiveness (%/cost unit)': f"{set_data['cost_effectiveness']:.4f}",
        'Budget Used (%)': f"{set_data['cost']/budget*100:.1f}%"
    })

# Add individual features
decision_data.append({
    'Feature Set': 'Feature A',
    'Cost': feature_a_cost,
    'Accuracy (%)': base_accuracy + feature_a_improvement,
    'Improvement (%)': feature_a_improvement,
    'Cost-Effectiveness (%/cost unit)': f"{ce_a:.4f}",
    'Budget Used (%)': f"{feature_a_cost/budget*100:.1f}%"
})

decision_data.append({
    'Feature Set': 'Feature B',
    'Cost': feature_b_cost,
    'Accuracy (%)': base_accuracy + feature_b_improvement,
    'Improvement (%)': feature_b_improvement,
    'Cost-Effectiveness (%/cost unit)': f"{ce_b:.4f}",
    'Budget Used (%)': f"{feature_b_cost/budget*100:.1f}%"
})

df = pd.DataFrame(decision_data)
print(df.to_string(index=False))

# Save decision matrix as image
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Highlight the best option
for i in range(len(df)):
    if df.iloc[i]['Feature Set'] == best_set:
        for j in range(len(df.columns)):
            table[(i+1, j)].set_facecolor('lightyellow')

ax.set_title('Feature Selection Decision Matrix', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'decision_matrix.png'), dpi=300, bbox_inches='tight')

print(f"\nAll plots saved to: {save_dir}")

# Summary of findings
print("\n" + "="*60)
print("SUMMARY OF FINDINGS")
print("="*60)
print(f"1. {better_feature} is more cost-effective than the other ({better_ce:.3f} vs {min(ce_a, ce_b):.3f})")
print(f"2. Best feature set for budget optimization: {best_set}")
print(f"3. Cost-effectiveness ranking:")
sorted_sets = sorted(feature_sets.items(), key=lambda x: x[1]['cost_effectiveness'], reverse=True)
for i, (set_name, set_data) in enumerate(sorted_sets):
    print(f"   {i+1}. {set_name}: {set_data['cost_effectiveness']:.4f}% per cost unit")
print(f"4. Budget utilization: {budget} available, {feature_sets[best_set]['cost']} used")
print(f"5. Best accuracy achievable: {feature_sets[best_set]['accuracy']}%")
print("="*60)

# Additional analysis: What if we combine features?
print("\n6. Feature Combination Analysis:")
print("-" * 40)

# Calculate if combining individual features would be better
combined_cost = feature_a_cost + feature_b_cost
combined_improvement = feature_a_improvement + feature_b_improvement
combined_ce = combined_improvement / combined_cost

print(f"Combining Feature A and Feature B:")
print(f"  Total cost: {feature_a_cost} + {feature_b_cost} = {combined_cost}")
print(f"  Total improvement: {feature_a_improvement}% + {feature_b_improvement}% = {combined_improvement}%")
print(f"  Combined cost-effectiveness: {combined_ce:.4f}% per cost unit")

# Compare with feature sets
best_feature_set_ce = feature_sets[best_set]['cost_effectiveness']
if combined_ce > best_feature_set_ce:
    print(f"  Result: Combining features is more cost-effective than {best_set}")
else:
    print(f"  Result: {best_set} is more cost-effective than combining features")

# Budget efficiency analysis
print(f"\n7. Budget Efficiency Analysis:")
print("-" * 40)

for set_name, set_data in feature_sets.items():
    efficiency = (set_data['accuracy'] - base_accuracy) / set_data['cost']
    budget_efficiency = efficiency * budget
    print(f"{set_name}:")
    print(f"  If we had {budget} budget, we could achieve: {base_accuracy + budget_efficiency:.1f}% accuracy")
    print(f"  Current: {set_data['accuracy']}% accuracy with {set_data['cost']} cost")
    print(f"  Efficiency ratio: {budget_efficiency / (set_data['accuracy'] - base_accuracy):.2f}")
    print()
