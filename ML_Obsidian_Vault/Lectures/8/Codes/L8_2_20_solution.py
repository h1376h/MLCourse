import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import entropy
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 20: Advanced Mutual Information Analysis")
print("=" * 60)

# Given joint frequency distribution
print("\nGiven Joint Frequency Distribution:")
print("Activity Level (X) vs Engagement Score (Y) across 100 users")
print("-" * 60)

# Create the joint frequency table
data = {
    'Activity Level (X)': ['Inactive (a)', 'Active (b)', 'Column Sum'],
    'Low Engagement (Y=0)': [30, 20, 50],
    'High Engagement (Y=1)': [10, 40, 50],
    'Row Sum': [40, 60, 100]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

# Extract the joint frequencies
n_aa = 30  # P(X=a, Y=0)
n_ab = 10  # P(X=a, Y=1)
n_ba = 20  # P(X=b, Y=0)
n_bb = 40  # P(X=b, Y=1)
N = 100    # Total number of users

print("\n" + "=" * 60)
print("STEP-BY-STEP SOLUTION")
print("=" * 60)

# Step 1: Calculate marginal probabilities
print("\n1. Calculating Marginal Probabilities:")
print("-" * 40)

P_X_a = (n_aa + n_ab) / N
P_X_b = (n_ba + n_bb) / N
P_Y_0 = (n_aa + n_ba) / N
P_Y_1 = (n_ab + n_bb) / N

print(f"P(X=a) = (n_aa + n_ab) / N = ({n_aa} + {n_ab}) / {N} = {P_X_a:.4f}")
print(f"P(X=b) = (n_ba + n_bb) / N = ({n_ba} + {n_bb}) / {N} = {P_X_b:.4f}")
print(f"P(Y=0) = (n_aa + n_ba) / N = ({n_aa} + {n_ba}) / {N} = {P_Y_0:.4f}")
print(f"P(Y=1) = (n_ab + n_bb) / N = ({n_ab} + {n_bb}) / {N} = {P_Y_1:.4f}")

# Step 2: Calculate conditional probabilities and check independence
print("\n2. Calculating Conditional Probabilities and Checking Independence:")
print("-" * 60)

P_X_a_given_Y_1 = n_ab / (n_ab + n_bb)
P_Y_1_given_X_a = n_ab / (n_aa + n_ab)

print(f"P(X=a|Y=1) = n_ab / (n_ab + n_bb) = {n_ab} / ({n_ab} + {n_bb}) = {P_X_a_given_Y_1:.4f}")
print(f"P(Y=1|X=a) = n_ab / (n_aa + n_ab) = {n_ab} / ({n_aa} + {n_ab}) = {P_Y_1_given_X_a:.4f}")

# Check independence: P(X=a|Y=1) should equal P(X=a) if independent
print(f"\nChecking Independence:")
print(f"P(X=a|Y=1) = {P_X_a_given_Y_1:.4f}")
print(f"P(X=a) = {P_X_a:.4f}")
print(f"Are they equal? {abs(P_X_a_given_Y_1 - P_X_a) < 1e-10}")

if abs(P_X_a_given_Y_1 - P_X_a) < 1e-10:
    print("X and Y are INDEPENDENT")
else:
    print("X and Y are NOT INDEPENDENT")

# Step 3: Calculate entropies H(X) and H(Y)
print("\n3. Calculating Entropies H(X) and H(Y):")
print("-" * 40)

def entropy_binary(p):
    """Calculate binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)"""
    if p == 0 or p == 1:
        return 0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

H_X = entropy_binary(P_X_a)
H_Y = entropy_binary(P_Y_0)

print(f"H(X) = -P(X=a)*log2(P(X=a)) - P(X=b)*log2(P(X=b))")
print(f"H(X) = -{P_X_a:.4f}*log2({P_X_a:.4f}) - {P_X_b:.4f}*log2({P_X_b:.4f})")
print(f"H(X) = {H_X:.4f}")

print(f"\nH(Y) = -P(Y=0)*log2(P(Y=0)) - P(Y=1)*log2(P(Y=1))")
print(f"H(Y) = -{P_Y_0:.4f}*log2({P_Y_0:.4f}) - {P_Y_1:.4f}*log2({P_Y_1:.4f})")
print(f"H(Y) = {H_Y:.4f}")

# Step 4: Calculate conditional entropy H(X|Y)
print("\n4. Calculating Conditional Entropy H(X|Y):")
print("-" * 40)

# Calculate P(X|Y) for each combination
P_X_a_given_Y_0 = n_aa / (n_aa + n_ba)
P_X_b_given_Y_0 = n_ba / (n_aa + n_ba)
P_X_a_given_Y_1 = n_ab / (n_ab + n_bb)
P_X_b_given_Y_1 = n_bb / (n_ab + n_bb)

print(f"P(X=a|Y=0) = n_aa / (n_aa + n_ba) = {n_aa} / ({n_aa} + {n_ba}) = {P_X_a_given_Y_0:.4f}")
print(f"P(X=b|Y=0) = n_ba / (n_aa + n_ba) = {n_ba} / ({n_aa} + {n_ba}) = {P_X_b_given_Y_0:.4f}")
print(f"P(X=a|Y=1) = n_ab / (n_ab + n_bb) = {n_ab} / ({n_ab} + {n_bb}) = {P_X_a_given_Y_1:.4f}")
print(f"P(X=b|Y=1) = n_bb / (n_ab + n_bb) = {n_bb} / ({n_ab} + {n_bb}) = {P_X_b_given_Y_1:.4f}")

# Calculate H(X|Y=0) and H(X|Y=1)
H_X_given_Y_0 = entropy_binary(P_X_a_given_Y_0)
H_X_given_Y_1 = entropy_binary(P_X_a_given_Y_1)

print(f"\nH(X|Y=0) = -P(X=a|Y=0)*log2(P(X=a|Y=0)) - P(X=b|Y=0)*log2(P(X=b|Y=0))")
print(f"H(X|Y=0) = -{P_X_a_given_Y_0:.4f}*log2({P_X_a_given_Y_0:.4f}) - {P_X_b_given_Y_0:.4f}*log2({P_X_b_given_Y_0:.4f})")
print(f"H(X|Y=0) = {H_X_given_Y_0:.4f}")

print(f"\nH(X|Y=1) = -P(X=a|Y=1)*log2(P(X=a|Y=1)) - P(X=b|Y=1)*log2(P(X=b|Y=1))")
print(f"H(X|Y=1) = -{P_X_a_given_Y_1:.4f}*log2({P_X_a_given_Y_1:.4f}) - {P_X_b_given_Y_1:.4f}*log2({P_X_b_given_Y_1:.4f})")
print(f"H(X|Y=1) = {H_X_given_Y_1:.4f}")

# Calculate H(X|Y) = sum over y of P(y) * H(X|Y=y)
H_X_given_Y = P_Y_0 * H_X_given_Y_0 + P_Y_1 * H_X_given_Y_1

print(f"\nH(X|Y) = P(Y=0)*H(X|Y=0) + P(Y=1)*H(X|Y=1)")
print(f"H(X|Y) = {P_Y_0:.4f}*{H_X_given_Y_0:.4f} + {P_Y_1:.4f}*{H_X_given_Y_1:.4f}")
print(f"H(X|Y) = {H_X_given_Y:.4f}")

# Step 5: Calculate mutual information I(X;Y) = H(X) - H(X|Y)
print("\n5. Calculating Mutual Information I(X;Y) = H(X) - H(X|Y):")
print("-" * 60)

I_XY_method1 = H_X - H_X_given_Y

print(f"I(X;Y) = H(X) - H(X|Y)")
print(f"I(X;Y) = {H_X:.4f} - {H_X_given_Y:.4f}")
print(f"I(X;Y) = {I_XY_method1:.4f}")

# Step 6: Verify using alternative formula I(X;Y) = H(Y) - H(Y|X)
print("\n6. Verifying using I(X;Y) = H(Y) - H(Y|X):")
print("-" * 50)

# Calculate P(Y|X) for each combination
P_Y_0_given_X_a = n_aa / (n_aa + n_ab)
P_Y_1_given_X_a = n_ab / (n_aa + n_ab)
P_Y_0_given_X_b = n_ba / (n_ba + n_bb)
P_Y_1_given_X_b = n_bb / (n_ba + n_bb)

print(f"P(Y=0|X=a) = n_aa / (n_aa + n_ab) = {n_aa} / ({n_aa} + {n_ab}) = {P_Y_0_given_X_a:.4f}")
print(f"P(Y=1|X=a) = n_ab / (n_aa + n_ab) = {n_ab} / ({n_aa} + {n_ab}) = {P_Y_1_given_X_a:.4f}")
print(f"P(Y=0|X=b) = n_ba / (n_ba + n_bb) = {n_ba} / ({n_ba} + {n_bb}) = {P_Y_0_given_X_b:.4f}")
print(f"P(Y=1|X=b) = n_bb / (n_ba + n_bb) = {n_bb} / ({n_ba} + {n_bb}) = {P_Y_1_given_X_b:.4f}")

# Calculate H(Y|X=a) and H(Y|X=b)
H_Y_given_X_a = entropy_binary(P_Y_1_given_X_a)
H_Y_given_X_b = entropy_binary(P_Y_1_given_X_b)

print(f"\nH(Y|X=a) = -P(Y=0|X=a)*log2(P(Y=0|X=a)) - P(Y=1|X=a)*log2(P(Y=1|X=a))")
print(f"H(Y|X=a) = -{P_Y_0_given_X_a:.4f}*log2({P_Y_0_given_X_a:.4f}) - {P_Y_1_given_X_a:.4f}*log2({P_Y_1_given_X_a:.4f})")
print(f"H(Y|X=a) = {H_Y_given_X_a:.4f}")

print(f"\nH(Y|X=b) = -P(Y=0|X=b)*log2(P(Y=0|X=b)) - P(Y=1|X=b)*log2(P(Y=1|X=b))")
print(f"H(Y|X=b) = -{P_Y_0_given_X_b:.4f}*log2({P_Y_0_given_X_b:.4f}) - {P_Y_1_given_X_b:.4f}*log2({P_Y_1_given_X_b:.4f})")
print(f"H(Y|X=b) = {H_Y_given_X_b:.4f}")

# Calculate H(Y|X) = sum over x of P(x) * H(Y|X=x)
H_Y_given_X = P_X_a * H_Y_given_X_a + P_X_b * H_Y_given_X_b

print(f"\nH(Y|X) = P(X=a)*H(Y|X=a) + P(X=b)*H(Y|X=b)")
print(f"H(Y|X) = {P_X_a:.4f}*{H_Y_given_X_a:.4f} + {P_X_b:.4f}*{H_Y_given_X_b:.4f}")
print(f"H(Y|X) = {H_Y_given_X:.4f}")

# Calculate I(X;Y) using alternative method
I_XY_method2 = H_Y - H_Y_given_X

print(f"\nI(X;Y) = H(Y) - H(Y|X)")
print(f"I(X;Y) = {H_Y:.4f} - {H_Y_given_X:.4f}")
print(f"I(X;Y) = {I_XY_method2:.4f}")

print(f"\nVerification: Method 1 = {I_XY_method1:.4f}, Method 2 = {I_XY_method2:.4f}")
print(f"Difference: {abs(I_XY_method1 - I_XY_method2):.10f}")

# Step 7: Calculate I(X;Y) using direct formula
print("\n7. Calculating I(X;Y) using direct formula:")
print("-" * 50)

def mutual_info_direct():
    """Calculate mutual information using direct formula"""
    # Joint probabilities
    P_aa = n_aa / N  # P(X=a, Y=0)
    P_ab = n_ab / N  # P(X=a, Y=1)
    P_ba = n_ba / N  # P(X=b, Y=0)
    P_bb = n_bb / N  # P(X=b, Y=1)
    
    # Marginal probabilities
    P_X_a = P_aa + P_ab
    P_X_b = P_ba + P_bb
    P_Y_0 = P_aa + P_ba
    P_Y_1 = P_ab + P_bb
    
    # Calculate each term in the sum
    term1 = P_aa * math.log2(P_aa / (P_X_a * P_Y_0)) if P_aa > 0 else 0
    term2 = P_ab * math.log2(P_ab / (P_X_a * P_Y_1)) if P_ab > 0 else 0
    term3 = P_ba * math.log2(P_ba / (P_X_b * P_Y_0)) if P_ba > 0 else 0
    term4 = P_bb * math.log2(P_bb / (P_X_b * P_Y_1)) if P_bb > 0 else 0
    
    return term1 + term2 + term3 + term4, (term1, term2, term3, term4)

I_XY_method3, terms = mutual_info_direct()

print(f"I(X;Y) = Σ P(x,y) * log2(P(x,y) / (P(x) * P(y)))")
print(f"\nTerm 1: P(X=a,Y=0) * log2(P(X=a,Y=0) / (P(X=a) * P(Y=0)))")
print(f"        = {n_aa/N:.4f} * log2({n_aa/N:.4f} / ({P_X_a:.4f} * {P_Y_0:.4f}))")
print(f"        = {terms[0]:.4f}")

print(f"\nTerm 2: P(X=a,Y=1) * log2(P(X=a,Y=1) / (P(X=a) * P(Y=1)))")
print(f"        = {n_ab/N:.4f} * log2({n_ab/N:.4f} / ({P_X_a:.4f} * {P_Y_1:.4f}))")
print(f"        = {terms[1]:.4f}")

print(f"\nTerm 3: P(X=b,Y=0) * log2(P(X=b,Y=0) / (P(X=b) * P(Y=0)))")
print(f"        = {n_ba/N:.4f} * log2({n_ba/N:.4f} / ({P_X_b:.4f} * {P_Y_0:.4f}))")
print(f"        = {terms[2]:.4f}")

print(f"\nTerm 4: P(X=b,Y=1) * log2(P(X=b,Y=1) / (P(X=b) * P(Y=1)))")
print(f"        = {n_bb/N:.4f} * log2({n_bb/N:.4f} / ({P_X_b:.4f} * {P_Y_1:.4f}))")
print(f"        = {terms[3]:.4f}")

print(f"\nI(X;Y) = {terms[0]:.4f} + {terms[1]:.4f} + {terms[2]:.4f} + {terms[3]:.4f}")
print(f"I(X;Y) = {I_XY_method3:.4f}")

print(f"\nAll three methods give the same result: {I_XY_method3:.4f}")

# Step 8: Interpret mutual information value
print("\n8. Interpretation of Mutual Information Value:")
print("-" * 50)
print(f"The mutual information I(X;Y) = {I_XY_method3:.4f} indicates:")
if I_XY_method3 > 0.5:
    print("- Strong relationship between activity level and engagement")
elif I_XY_method3 > 0.2:
    print("- Moderate relationship between activity level and engagement")
elif I_XY_method3 > 0.1:
    print("- Weak relationship between activity level and engagement")
else:
    print("- Very weak relationship between activity level and engagement")

print(f"- The amount of uncertainty about X reduced by knowing Y (or vice versa)")
print(f"- Higher values indicate stronger dependence between variables")

# Step 9: Calculate normalized mutual information
print("\n9. Calculating Normalized Mutual Information:")
print("-" * 45)

NMI_XY = I_XY_method3 / math.sqrt(H_X * H_Y)

print(f"NMI(X;Y) = I(X;Y) / √(H(X) * H(Y))")
print(f"NMI(X;Y) = {I_XY_method3:.4f} / √({H_X:.4f} * {H_Y:.4f})")
print(f"NMI(X;Y) = {I_XY_method3:.4f} / √({H_X * H_Y:.4f})")
print(f"NMI(X;Y) = {I_XY_method3:.4f} / {math.sqrt(H_X * H_Y):.4f}")
print(f"NMI(X;Y) = {NMI_XY:.4f}")

# Step 10: Feature selection decision
print("\n10. Feature Selection Decision:")
print("-" * 35)
threshold = 0.1
print(f"Mutual Information Threshold: {threshold}")
print(f"Our calculated I(X;Y): {I_XY_method3:.4f}")

if I_XY_method3 > threshold:
    print(f"Decision: SELECT this feature (I(X;Y) = {I_XY_method3:.4f} > {threshold})")
    print("Justification: The feature provides sufficient information about the target")
else:
    print(f"Decision: DO NOT SELECT this feature (I(X;Y) = {I_XY_method3:.4f} ≤ {threshold})")
    print("Justification: The feature provides insufficient information about the target")

# Create visualizations
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Joint Probability Heatmap
plt.figure(figsize=(12, 10))

# Create subplots
plt.subplot(2, 2, 1)
joint_probs = np.array([[n_aa/N, n_ab/N], [n_ba/N, n_bb/N]])
sns.heatmap(joint_probs, 
            annot=True, 
            fmt='.3f',
            xticklabels=['Y=0 (Low)', 'Y=1 (High)'],
            yticklabels=['X=a (Inactive)', 'X=b (Active)'],
            cmap='Blues',
            cbar_kws={'label': 'Joint Probability P(X,Y)'})
plt.title('Joint Probability Distribution P(X,Y)')
plt.xlabel('Engagement Score (Y)')
plt.ylabel('Activity Level (X)')

# 2. Marginal Probabilities Bar Chart
plt.subplot(2, 2, 2)
categories = ['P(X=a)', 'P(X=b)', 'P(Y=0)', 'P(Y=1)']
probabilities = [P_X_a, P_X_b, P_Y_0, P_Y_1]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = plt.bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black')
plt.title('Marginal Probabilities')
plt.ylabel('Probability')
plt.ylim(0, 1)

# Add value labels on bars
for bar, prob in zip(bars, probabilities):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{prob:.3f}', ha='center', va='bottom')

# 3. Conditional Probabilities Comparison
plt.subplot(2, 2, 3)
x_pos = np.arange(4)
conditional_probs = [P_X_a_given_Y_0, P_X_a_given_Y_1, P_Y_0_given_X_a, P_Y_0_given_X_b]
labels = ['P(X=a|Y=0)', 'P(X=a|Y=1)', 'P(Y=0|X=a)', 'P(Y=0|X=b)']

bars = plt.bar(x_pos, conditional_probs, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'], 
               alpha=0.7, edgecolor='black')
plt.title('Conditional Probabilities')
plt.ylabel('Probability')
plt.xticks(x_pos, labels, rotation=45, ha='right')
plt.ylim(0, 1)

# Add value labels on bars
for bar, prob in zip(bars, conditional_probs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{prob:.3f}', ha='center', va='bottom')

# 4. Information Theory Summary
plt.subplot(2, 2, 4)
info_metrics = ['H(X)', 'H(Y)', 'H(X|Y)', 'H(Y|X)', 'I(X;Y)']
info_values = [H_X, H_Y, H_X_given_Y, H_Y_given_X, I_XY_method3]
colors_info = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orange']

bars = plt.bar(info_metrics, info_values, color=colors_info, alpha=0.7, edgecolor='black')
plt.title('Information Theory Metrics')
plt.ylabel('Bits')
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, info_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'mutual_information_analysis.png'), dpi=300, bbox_inches='tight')

# 5. Detailed Mutual Information Breakdown
plt.figure(figsize=(14, 8))

# Create a more detailed visualization
plt.subplot(1, 2, 1)
# Venn diagram representation
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Create circles for X and Y
circle_x = Circle((0.3, 0.5), 0.4, fill=False, linewidth=2, color='blue', label='H(X)')
circle_y = Circle((0.7, 0.5), 0.4, fill=False, linewidth=2, color='red', label='H(Y)')

ax.add_patch(circle_x)
ax.add_patch(circle_y)

# Add text annotations
ax.text(0.3, 0.5, f'H(X)\n{H_X:.3f}', ha='center', va='center', fontsize=12, weight='bold')
ax.text(0.7, 0.5, f'H(Y)\n{H_Y:.3f}', ha='center', va='center', fontsize=12, weight='bold')
ax.text(0.5, 0.5, f'I(X;Y)\n{I_XY_method3:.3f}', ha='center', va='center', fontsize=12, weight='bold', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
ax.text(0.1, 0.5, f'H(X|Y)\n{H_X_given_Y:.3f}', ha='center', va='center', fontsize=10)
ax.text(0.9, 0.5, f'H(Y|X)\n{H_Y_given_X:.3f}', ha='center', va='center', fontsize=10)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Information Theory Relationships\n(Venn Diagram Representation)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'information_venn_diagram.png'), dpi=300, bbox_inches='tight')

# 6. Feature Selection Decision Visualization
plt.figure(figsize=(10, 6))

# Create a decision threshold visualization
thresholds = np.linspace(0, 0.5, 100)
feature_scores = [I_XY_method3] * len(thresholds)

plt.plot(thresholds, feature_scores, 'b-', linewidth=3, label=f'I(X;Y) = {I_XY_method3:.4f}')
plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold}')
plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2)

# Shade regions
plt.fill_between(thresholds, 0, feature_scores, where=(thresholds <= threshold), 
                 alpha=0.3, color='red', label='Reject Region')
plt.fill_between(thresholds, 0, feature_scores, where=(thresholds > threshold), 
                 alpha=0.3, color='green', label='Accept Region')

plt.xlabel('Mutual Information Threshold')
plt.ylabel('Mutual Information I(X;Y)')
plt.title('Feature Selection Decision Based on Mutual Information Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, max(I_XY_method3 * 1.2, threshold * 1.2))

# Add decision text
decision_text = "SELECT" if I_XY_method3 > threshold else "REJECT"
plt.text(0.5, 0.5, f'Decision: {decision_text}', 
         transform=plt.gca().transAxes, fontsize=14, weight='bold',
         ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'feature_selection_decision.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Results:")
print(f"- Marginal Probabilities: P(X=a)={P_X_a:.4f}, P(X=b)={P_X_b:.4f}, P(Y=0)={P_Y_0:.4f}, P(Y=1)={P_Y_1:.4f}")
print(f"- Entropies: H(X)={H_X:.4f}, H(Y)={H_Y:.4f}")
print(f"- Conditional Entropies: H(X|Y)={H_X_given_Y:.4f}, H(Y|X)={H_Y_given_X:.4f}")
print(f"- Mutual Information: I(X;Y)={I_XY_method3:.4f}")
print(f"- Normalized Mutual Information: NMI(X;Y)={NMI_XY:.4f}")
print(f"- Feature Selection Decision: {'SELECT' if I_XY_method3 > threshold else 'REJECT'}")
