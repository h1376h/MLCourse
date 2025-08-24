import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_38")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("QUESTION 38: SPLIT QUALITY DETECTIVE - DECISION TREE AUDIT")
print("=" * 80)

# Dataset information
total_samples = 16
positive_class = 10
negative_class = 6

print(f"\nDATASET OVERVIEW:")
print(f"Total samples: {total_samples}")
print(f"Class distribution: [+: {positive_class}, -: {negative_class}]")
print(f"Base entropy: {-(positive_class/total_samples)*math.log2(positive_class/total_samples) - (negative_class/total_samples)*math.log2(negative_class/total_samples):.4f} bits")

# Define the three splits
splits = {
    'Split A (Weather)': {
        'branches': {
            'Sunny': {'positive': 4, 'negative': 1, 'total': 5},
            'Cloudy': {'positive': 3, 'negative': 2, 'total': 5},
            'Rainy': {'positive': 3, 'negative': 3, 'total': 6}
        }
    },
    'Split B (Customer_ID)': {
        'branches': {
            'ID_001-100': {'positive': 2, 'negative': 0, 'total': 2},
            'ID_101-200': {'positive': 2, 'negative': 0, 'total': 2},
            'ID_201-300': {'positive': 2, 'negative': 0, 'total': 2},
            'ID_301-400': {'positive': 2, 'negative': 0, 'total': 2},
            'ID_401-500': {'positive': 2, 'negative': 6, 'total': 8}
        }
    },
    'Split C (Purchase_Amount $\\leq$ $50)': {
        'branches': {
            '$\\leq$ $50$': {'positive': 6, 'negative': 4, 'total': 10},
            '$>$ $50$': {'positive': 4, 'negative': 2, 'total': 6}
        }
    }
}

def calculate_entropy(positive, negative):
    """Calculate entropy for a binary classification node"""
    total = positive + negative
    if total == 0:
        return 0
    
    p_pos = positive / total
    p_neg = negative / total
    
    entropy = 0
    if p_pos > 0:
        entropy -= p_pos * math.log2(p_pos)
    if p_neg > 0:
        entropy -= p_neg * math.log2(p_neg)
    
    return entropy

def calculate_gini_impurity(positive, negative):
    """Calculate Gini impurity for a binary classification node"""
    total = positive + negative
    if total == 0:
        return 0
    
    p_pos = positive / total
    p_neg = negative / total
    
    return 1 - (p_pos**2 + p_neg**2)

def calculate_information_gain(split_data, base_entropy):
    """Calculate information gain for a split"""
    weighted_entropy = 0
    
    for branch_name, branch_data in split_data['branches'].items():
        branch_entropy = calculate_entropy(branch_data['positive'], branch_data['negative'])
        weight = branch_data['total'] / total_samples
        weighted_entropy += weight * branch_entropy
    
    information_gain = base_entropy - weighted_entropy
    return information_gain, weighted_entropy

def calculate_gain_ratio(split_data, base_entropy):
    """Calculate gain ratio for a split"""
    information_gain, weighted_entropy = calculate_information_gain(split_data, base_entropy)
    
    # Calculate split information (intrinsic value)
    split_info = 0
    for branch_name, branch_data in split_data['branches'].items():
        weight = branch_data['total'] / total_samples
        if weight > 0:
            split_info -= weight * math.log2(weight)
    
    if split_info == 0:
        return 0, split_info
    
    gain_ratio = information_gain / split_info
    return gain_ratio, split_info

def calculate_gini_gain(split_data, base_gini):
    """Calculate Gini gain for a split"""
    weighted_gini = 0
    
    for branch_name, branch_data in split_data['branches'].items():
        branch_gini = calculate_gini_impurity(branch_data['positive'], branch_data['negative'])
        weight = branch_data['total'] / total_samples
        weighted_gini += weight * branch_gini
    
    gini_gain = base_gini - weighted_gini
    return gini_gain, weighted_gini

# Calculate base metrics
base_entropy = calculate_entropy(positive_class, negative_class)
base_gini = calculate_gini_impurity(positive_class, negative_class)

print(f"\nBASE METRICS:")
print(f"Base entropy: {base_entropy:.4f} bits")
print(f"Base Gini impurity: {base_gini:.4f}")

# Calculate metrics for each split
results = {}
print(f"\n" + "="*80)
print(f"STEP-BY-STEP CALCULATIONS")
print(f"="*80)

for split_name, split_data in splits.items():
    print(f"\n{split_name}")
    print("-" * 50)
    
    # Calculate information gain
    info_gain, weighted_entropy = calculate_information_gain(split_data, base_entropy)
    
    # Calculate gain ratio
    gain_ratio, split_info = calculate_gain_ratio(split_data, base_entropy)
    
    # Calculate Gini gain
    gini_gain, weighted_gini = calculate_gini_gain(split_data, base_gini)
    
    # Store results
    results[split_name] = {
        'info_gain': info_gain,
        'weighted_entropy': weighted_entropy,
        'gain_ratio': gain_ratio,
        'split_info': split_info,
        'gini_gain': gini_gain,
        'weighted_gini': weighted_gini
    }
    
    print(f"Information Gain: {info_gain:.4f} bits")
    print(f"Gain Ratio: {gain_ratio:.4f}")
    print(f"Gini Gain: {gini_gain:.4f}")
    
    # Show detailed calculations
    print(f"\nDetailed calculations:")
    print(f"Base entropy: {base_entropy:.4f}")
    print(f"Base Gini: {base_gini:.4f}")
    
    for branch_name, branch_data in split_data['branches'].items():
        branch_entropy = calculate_entropy(branch_data['positive'], branch_data['negative'])
        branch_gini = calculate_gini_impurity(branch_data['positive'], branch_data['negative'])
        weight = branch_data['total'] / total_samples
        
        print(f"  {branch_name}: [+{branch_data['positive']}, -{branch_data['negative']}] "
              f"Entropy: {branch_entropy:.4f}, Gini: {branch_gini:.4f}, Weight: {weight:.3f}")
    
    print(f"Weighted entropy: {weighted_entropy:.4f}")
    print(f"Weighted Gini: {weighted_gini:.4f}")
    print(f"Split information: {split_info:.4f}")

# Create comprehensive comparison table
print(f"\n" + "="*80)
print(f"ALGORITHM COMPARISON SUMMARY")
print(f"="*80)

comparison_data = []
for split_name, metrics in results.items():
    comparison_data.append({
        'Split': split_name,
        'Information Gain': f"{metrics['info_gain']:.4f}",
        'Gain Ratio': f"{metrics['gain_ratio']:.4f}",
        'Gini Gain': f"{metrics['gini_gain']:.4f}",
        'Split Info': f"{metrics['split_info']:.4f}"
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# Determine algorithm preferences
print(f"\n" + "="*80)
print(f"ALGORITHM PREFERENCES")
print(f"="*80)

# ID3 preference (highest information gain)
id3_preference = max(results.keys(), key=lambda x: results[x]['info_gain'])
print(f"ID3 preference: {id3_preference} (IG: {results[id3_preference]['info_gain']:.4f})")

# C4.5 preference (highest gain ratio)
c45_preference = max(results.keys(), key=lambda x: results[x]['gain_ratio'])
print(f"C4.5 preference: {c45_preference} (Gain Ratio: {results[c45_preference]['gain_ratio']:.4f})")

# CART with Gini preference (highest Gini gain)
cart_gini_preference = max(results.keys(), key=lambda x: results[x]['gini_gain'])
print(f"CART (Gini) preference: {cart_gini_preference} (Gini Gain: {results[cart_gini_preference]['gini_gain']:.4f})")

# CART with Entropy preference (highest information gain, same as ID3)
cart_entropy_preference = max(results.keys(), key=lambda x: results[x]['info_gain'])
print(f"CART (Entropy) preference: {cart_entropy_preference} (IG: {results[cart_entropy_preference]['info_gain']:.4f})")

# Create separate visualizations

# Plot 1: Information Gain Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
split_names = list(results.keys())
info_gains = [results[name]['info_gain'] for name in split_names]
gain_ratios = [results[name]['gain_ratio'] for name in split_names]
gini_gains = [results[name]['gini_gain'] for name in split_names]

x = np.arange(len(split_names))
width = 0.25

bars1 = ax1.bar(x - width, info_gains, width, label='Information Gain', color='skyblue', alpha=0.7)
bars2 = ax1.bar(x, gain_ratios, width, label='Gain Ratio', color='lightgreen', alpha=0.7)
bars3 = ax1.bar(x + width, gini_gains, width, label='Gini Gain', color='lightcoral', alpha=0.7)

ax1.set_xlabel('Splits')
ax1.set_ylabel('Metric Value')
ax1.set_title('Split Quality Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([name.split('(')[0].strip() for name in split_names], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'split_quality_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Algorithm Preferences
fig2, ax2 = plt.subplots(figsize=(10, 6))
algorithms = ['ID3', 'C4.5', 'CART (Gini)', 'CART (Entropy)']
preferences = [id3_preference, c45_preference, cart_gini_preference, cart_entropy_preference]

# Color code based on split type
colors = []
for pref in preferences:
    if 'Weather' in pref:
        colors.append('skyblue')
    elif 'Customer_ID' in pref:
        colors.append('lightcoral')
    else:
        colors.append('lightgreen')

bars = ax2.barh(algorithms, [1]*len(algorithms), color=colors, alpha=0.7)
ax2.set_xlabel('Preference')
ax2.set_title('Algorithm Preferences')
ax2.set_xlim(0, 1)

# Add split labels on bars
for i, (bar, pref) in enumerate(zip(bars, preferences)):
    split_short = pref.split('(')[0].strip()
    ax2.text(0.5, i, split_short, ha='center', va='center', fontweight='bold')

# Create legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.7, label='Weather'),
    plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.7, label='Customer_ID'),
    plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7, label='Purchase_Amount')
]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_preferences.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Detailed Split Analysis
fig3, ax3 = plt.subplots(figsize=(10, 6))
split_metrics = ['Information\nGain', 'Gain\nRatio', 'Gini\nGain']
split_a_metrics = [results['Split A (Weather)']['info_gain'], 
                   results['Split A (Weather)']['gain_ratio'],
                   results['Split A (Weather)']['gini_gain']]
split_b_metrics = [results['Split B (Customer_ID)']['info_gain'],
                   results['Split B (Customer_ID)']['gain_ratio'],
                   results['Split B (Customer_ID)']['gini_gain']]
split_c_metrics = [results['Split C (Purchase_Amount $\\leq$ $50)']['info_gain'],
                   results['Split C (Purchase_Amount $\\leq$ $50)']['gain_ratio'],
                   results['Split C (Purchase_Amount $\\leq$ $50)']['gini_gain']]

x = np.arange(len(split_metrics))
width = 0.25

ax3.bar(x - width, split_a_metrics, width, label='Split A (Weather)', color='skyblue', alpha=0.7)
ax3.bar(x, split_b_metrics, width, label='Split B (Customer_ID)', color='lightcoral', alpha=0.7)
ax3.bar(x + width, split_c_metrics, width, label='Split C (Purchase_Amount)', color='lightgreen', alpha=0.7)

ax3.set_xlabel('Metrics')
ax3.set_ylabel('Value')
ax3.set_title('Detailed Split Analysis')
ax3.set_xticks(x)
ax3.set_xticklabels(split_metrics)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_split_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Branch Purity Analysis
fig4, ax4 = plt.subplots(figsize=(12, 6))
branch_purities = []
branch_names = []
split_colors = []

for split_name, split_data in splits.items():
    for branch_name, branch_data in split_data['branches'].items():
        purity = branch_data['positive'] / branch_data['total']
        branch_purities.append(purity)
        branch_names.append(f"{split_name.split('(')[0].strip()}\n{branch_name}")
        
        if 'Weather' in split_name:
            split_colors.append('skyblue')
        elif 'Customer_ID' in split_name:
            split_colors.append('lightcoral')
        else:
            split_colors.append('lightgreen')

bars = ax4.bar(range(len(branch_purities)), branch_purities, color=split_colors, alpha=0.7)
ax4.set_xlabel('Branches')
ax4.set_ylabel('Positive Class Proportion')
ax4.set_title('Branch Purity Analysis')
ax4.set_xticks(range(len(branch_names)))
ax4.set_xticklabels(branch_names, rotation=45, ha='right')
ax4.axhline(y=0.625, color='red', linestyle='--', alpha=0.7, label='Base Rate (10/16)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Add value labels
for i, (bar, purity) in enumerate(zip(bars, branch_purities)):
    ax4.text(bar.get_x() + bar.get_width()/2., purity + 0.01,
             f'{purity:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'branch_purity_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Overfitting Analysis
fig5, ax5 = plt.subplots(figsize=(10, 6))
overfitting_indicators = {
    'Split A (Weather)': {
        'high_cardinality': False,
        'perfect_separation': False,
        'unbalanced_branches': False,
        'overfitting_score': 0.2
    },
    'Split B (Customer_ID)': {
        'high_cardinality': True,
        'perfect_separation': True,
        'unbalanced_branches': True,
        'overfitting_score': 0.9
    },
    'Split C (Purchase_Amount $\\leq$ $50)': {
        'high_cardinality': False,
        'perfect_separation': False,
        'unbalanced_branches': False,
        'overfitting_score': 0.3
    }
}

overfitting_scores = [overfitting_indicators[name]['overfitting_score'] for name in split_names]
overfitting_colors = ['green' if score < 0.5 else 'orange' if score < 0.8 else 'red' for score in overfitting_scores]

bars = ax5.bar(range(len(split_names)), overfitting_scores, color=overfitting_colors, alpha=0.7)
ax5.set_xlabel('Splits')
ax5.set_ylabel('Overfitting Risk Score')
ax5.set_title('Overfitting Risk Analysis')
ax5.set_xticks(range(len(split_names)))
ax5.set_xticklabels([name.split('(')[0].strip() for name in split_names], rotation=45, ha='right')
ax5.set_ylim(0, 1)
ax5.grid(True, alpha=0.3)

# Add risk labels
for i, (bar, score) in enumerate(zip(bars, overfitting_scores)):
    risk_level = 'Low' if score < 0.5 else 'Medium' if score < 0.8 else 'High'
    ax5.text(bar.get_x() + bar.get_width()/2., score + 0.02,
             risk_level, ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_risk_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Summary Table
fig6, ax6 = plt.subplots(figsize=(10, 6))
summary_data = {
    'Algorithm': ['ID3', 'C4.5', 'CART (Gini)', 'CART (Entropy)'],
    'Preferred Split': [id3_preference.split('(')[0].strip(), 
                       c45_preference.split('(')[0].strip(),
                       cart_gini_preference.split('(')[0].strip(),
                       cart_entropy_preference.split('(')[0].strip()],
    'Key Reason': ['Highest IG', 'Highest GR', 'Highest GG', 'Highest IG']
}

df_summary = pd.DataFrame(summary_data)
ax6.axis('tight')
ax6.axis('off')

table = ax6.table(cellText=df_summary.values,
                 colLabels=df_summary.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code based on split preference
for i, pref in enumerate([id3_preference, c45_preference, cart_gini_preference, cart_entropy_preference]):
    if 'Weather' in pref:
        table[(i+1, 1)].set_facecolor('skyblue')
    elif 'Customer_ID' in pref:
        table[(i+1, 1)].set_facecolor('lightcoral')
    else:
        table[(i+1, 1)].set_facecolor('lightgreen')
    table[(i+1, 1)].set_text_props(weight='bold')

# Header styling
for j in range(len(df_summary.columns)):
    table[(0, j)].set_facecolor('#2E8B57')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax6.set_title('Algorithm Selection Summary', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'algorithm_selection_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create detailed comparison matrix
fig2, ax = plt.subplots(figsize=(14, 10))
ax.axis('tight')
ax.axis('off')

# Extended comparison table
extended_data = {
    'Metric': ['Information Gain', 'Gain Ratio', 'Gini Gain', 'Split Info', 'Overfitting Risk'],
    'Split A (Weather)': [
        f"{results['Split A (Weather)']['info_gain']:.4f}",
        f"{results['Split A (Weather)']['gain_ratio']:.4f}",
        f"{results['Split A (Weather)']['gini_gain']:.4f}",
        f"{results['Split A (Weather)']['split_info']:.4f}",
        'Low (0.2)'
    ],
    'Split B (Customer_ID)': [
        f"{results['Split B (Customer_ID)']['info_gain']:.4f}",
        f"{results['Split B (Customer_ID)']['gain_ratio']:.4f}",
        f"{results['Split B (Customer_ID)']['gini_gain']:.4f}",
        f"{results['Split B (Customer_ID)']['split_info']:.4f}",
        'High (0.9)'
    ],
    'Split C (Purchase_Amount)': [
        f"{results['Split C (Purchase_Amount $\\leq$ $50)']['info_gain']:.4f}",
        f"{results['Split C (Purchase_Amount $\\leq$ $50)']['gain_ratio']:.4f}",
        f"{results['Split C (Purchase_Amount $\\leq$ $50)']['gini_gain']:.4f}",
        f"{results['Split C (Purchase_Amount $\\leq$ $50)']['split_info']:.4f}",
        'Medium (0.3)'
    ]
}

df_extended = pd.DataFrame(extended_data)
table2 = ax.table(cellText=df_extended.values,
                 colLabels=df_extended.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table2.auto_set_font_size(False)
table2.set_fontsize(11)
table2.scale(1, 2.5)

# Color coding for overfitting risk
for i in range(1, len(df_extended) + 1):
    for j in range(1, len(df_extended.columns)):
        cell_value = df_extended.iloc[i-1, j]
        if 'Low' in str(cell_value):
            table2[(i, j)].set_facecolor('#4CAF50')
            table2[(i, j)].set_text_props(color='white', weight='bold')
        elif 'Medium' in str(cell_value):
            table2[(i, j)].set_facecolor('#FFC107')
            table2[(i, j)].set_text_props(weight='bold')
        elif 'High' in str(cell_value):
            table2[(i, j)].set_facecolor('#F44336')
            table2[(i, j)].set_text_props(color='white', weight='bold')

# Header styling
for j in range(len(df_extended.columns)):
    table2[(0, j)].set_facecolor('#1976D2')
    table2[(0, j)].set_text_props(weight='bold', color='white')

ax.set_title('Detailed Split Comparison Matrix', fontsize=16, fontweight='bold', pad=20)

plt.savefig(os.path.join(save_dir, 'detailed_comparison_matrix.png'), dpi=300, bbox_inches='tight')

# Print final analysis
print(f"\n" + "="*80)
print(f"FINAL ANALYSIS AND INSIGHTS")
print(f"="*80)

print(f"\n1. INFORMATION GAIN ANALYSIS:")
print(f"   - Split A (Weather): {results['Split A (Weather)']['info_gain']:.4f} bits")
print(f"   - Split B (Customer_ID): {results['Split B (Customer_ID)']['info_gain']:.4f} bits")
print(f"   - Split C (Purchase_Amount): {results['Split C (Purchase_Amount $\\leq$ $50)']['info_gain']:.4f} bits")

print(f"\n2. GAIN RATIO ANALYSIS:")
print(f"   - Split A (Weather): {results['Split A (Weather)']['gain_ratio']:.4f}")
print(f"   - Split B (Customer_ID): {results['Split B (Customer_ID)']['gain_ratio']:.4f}")
print(f"   - Split C (Purchase_Amount): {results['Split C (Purchase_Amount $\\leq$ $50)']['gain_ratio']:.4f}")

print(f"\n3. GINI GAIN ANALYSIS:")
print(f"   - Split A (Weather): {results['Split A (Weather)']['gini_gain']:.4f}")
print(f"   - Split B (Customer_ID): {results['Split B (Customer_ID)']['gini_gain']:.4f}")
print(f"   - Split C (Purchase_Amount): {results['Split C (Purchase_Amount $\\leq$ $50)']['gini_gain']:.4f}")

print(f"\n4. ALGORITHM AGREEMENT:")
print(f"   - ID3 and CART (Entropy) agree: Both prefer {id3_preference.split('(')[0].strip()}")
print(f"   - C4.5 prefers: {c45_preference.split('(')[0].strip()}")
print(f"   - CART (Gini) prefers: {cart_gini_preference.split('(')[0].strip()}")

print(f"\n5. OVERFITTING ANALYSIS:")
print(f"   - Split B (Customer_ID) shows HIGH overfitting risk due to:")
print(f"     * Perfect separation in most branches")
print(f"     * High cardinality (5 branches)")
print(f"     * Unbalanced branch sizes")
print(f"     * Customer ID is not a meaningful business feature")

print(f"\n6. REAL-WORLD DEPLOYMENT ISSUES:")
print(f"   - Split B would fail on new customer IDs not in training data")
print(f"   - No generalization capability")
print(f"   - Business logic violation (customer ID shouldn't predict behavior)")
print(f"   - High maintenance cost for new customers")

print(f"\n7. PRODUCTION DECISION ANALYSIS:")
print(f"   - All algorithms prefer Split B (Customer_ID)")
print(f"   - Split B shows severe overfitting (risk score: 0.9)")
print(f"   - Split A (Weather) provides balanced performance")
print(f"   - Split C (Purchase_Amount) shows minimal improvement")

print(f"\n8. PRODUCTION DEPLOYMENT RECOMMENDATION:")
print(f"   - REJECT Split B despite algorithm preference")
print(f"   - CHOOSE Split A (Weather) for production")
print(f"   - REASONING:")
print(f"     * Balanced information gain (0.0504 bits)")
print(f"     * Low overfitting risk (0.2)")
print(f"     * Meaningful business feature")
print(f"     * Good generalization capability")
print(f"     * Interpretable decision rules")
print(f"     * Low maintenance cost")
print(f"     * Handles new weather conditions gracefully")

print(f"\nImages saved to: {save_dir}")
