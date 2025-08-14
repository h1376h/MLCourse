import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_3_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Question 22: CART's Surrogate Splits for Missing Value Handling")
print("We need to understand how CART handles missing values using surrogate splits")
print("and compare different surrogate split strategies.")
print()
print("Tasks:")
print("1. Define surrogate splits in one sentence")
print("2. Rank surrogate splits by quality given primary split accuracy")
print("3. Explain why surrogate splits are more robust than simple imputation")
print("4. Show how to use the best surrogate for new samples")
print()

# Step 2: Understanding Surrogate Splits
print_step_header(2, "Understanding Surrogate Splits")

print("Surrogate Splits Definition:")
print("Surrogate splits are alternative splitting rules that closely mimic the behavior")
print("of the primary split when the primary feature is missing, allowing the tree")
print("to make predictions even when key features are unavailable.")
print()

print("Key Concepts:")
print("- Primary split: The best split based on the primary feature")
print("- Surrogate splits: Alternative splits that approximate the primary split")
print("- Agreement rate: How well the surrogate split agrees with the primary split")
print("- Quality ranking: Surrogates are ranked by their agreement with the primary split")
print()

# Step 3: Given Data Analysis
print_step_header(3, "Given Data Analysis")

print("Primary Split: 'Income > $50K' with 80% accuracy")
print("Surrogate Splits:")
print("1. 'Education = Graduate': 70% agreement")
print("2. 'Age > 40': 65% agreement") 
print("3. 'Experience > 8': 75% agreement")
print()

# Step 4: Ranking Surrogate Splits
print_step_header(4, "Ranking Surrogate Splits by Quality")

print("Surrogate splits are ranked by their agreement rate with the primary split.")
print("Higher agreement means the surrogate is more reliable.")
print()

agreements = [
    ("Experience > 8", 75),
    ("Education = Graduate", 70),
    ("Age > 40", 65)
]

print("Ranking by Agreement Rate (highest to lowest):")
print("-" * 50)
for i, (surrogate, agreement) in enumerate(agreements, 1):
    print(f"{i}. {surrogate}: {agreement}% agreement")
print()

print("Explanation of Ranking:")
print("- Experience > 8 (75%): Best surrogate, highest agreement with primary split")
print("- Education = Graduate (70%): Second best, good agreement")
print("- Age > 40 (65%): Weakest surrogate, lowest agreement")
print()

# Step 5: Why Surrogate Splits are More Robust
print_step_header(5, "Why Surrogate Splits are More Robust Than Simple Imputation")

print("1. Data-Driven Approach:")
print("   - Surrogates are learned from the actual data patterns")
print("   - They capture the relationship between features and the target")
print("   - No assumptions about missing value distribution")
print()
print("2. Multiple Fallback Options:")
print("   - Multiple surrogates provide redundancy")
print("   - If one surrogate fails, others can be used")
print("   - Gradual degradation rather than complete failure")
print()
print("3. Context-Aware Predictions:")
print("   - Surrogates consider the specific split context")
print("   - They adapt to different regions of the tree")
print("   - More accurate than global imputation strategies")
print()
print("4. Preserves Tree Structure:")
print("   - Maintains the original tree topology")
print("   - No need to rebuild or modify the tree")
print("   - Consistent prediction paths")
print()

# Step 6: Using Surrogate Splits for New Samples
print_step_header(6, "Using Surrogate Splits for New Samples")

print("When Income is missing for a new sample, we use the surrogate splits in order:")
print()
print("Step 1: Try the primary split (Income > $50K)")
print("   - If Income is available, use it directly")
print("   - If Income is missing, proceed to surrogate splits")
print()
print("Step 2: Try the best surrogate (Experience > 8)")
print("   - If Experience > 8, follow the 'Yes' branch")
print("   - If Experience ≤ 8, follow the 'No' branch")
print()
print("Step 3: If Experience is also missing, try the second surrogate (Education = Graduate)")
print("   - If Education = Graduate, follow the 'Yes' branch")
print("   - If Education ≠ Graduate, follow the 'No' branch")
print()
print("Step 4: If Education is missing, try the third surrogate (Age > 40)")
print("   - If Age > 40, follow the 'Yes' branch")
print("   - If Age ≤ 40, follow the 'No' branch")
print()
print("Step 5: If all features are missing, use the majority class at that node")
print()

# Step 7: Visualizing Primary Split
print_step_header(7, "Visualizing Primary Split")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r'Income (\$K)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(r'Primary Split: Income $> \$50K$', fontsize=16, fontweight='bold')

# Create sample income distribution
np.random.seed(42)
income_low = np.random.normal(35, 8, 400)
income_high = np.random.normal(65, 12, 600)

# Plot histogram
ax.hist(income_low, bins=30, alpha=0.7, color='lightcoral', label=r'Income $\leq \$50K$')
ax.hist(income_high, bins=30, alpha=0.7, color='lightblue', label=r'Income $> \$50K$')
ax.axvline(x=50, color='red', linestyle='--', linewidth=3, label=r'Split: $\$50K$')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'primary_split_visualization.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 8: Visualizing Surrogate Split 1 - Experience
print_step_header(8, "Visualizing Surrogate Split 1 - Experience")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('Experience (Years)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(r'Surrogate 1: Experience $> 8$ (75\% agreement)', fontsize=16, fontweight='bold')

# Create sample experience distribution
exp_low = np.random.normal(4, 2, 400)
exp_high = np.random.normal(12, 3, 600)

ax.hist(exp_low, bins=30, alpha=0.7, color='lightcoral', label=r'Experience $\leq 8$')
ax.hist(exp_high, bins=30, alpha=0.7, color='lightblue', label=r'Experience $> 8$')
ax.axvline(x=8, color='blue', linestyle='--', linewidth=3, label=r'Split: $8$ years')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'surrogate_split_1_experience.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 9: Visualizing Surrogate Split 2 - Education
print_step_header(9, "Visualizing Surrogate Split 2 - Education")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('Education Level', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(r'Surrogate 2: Education = Graduate (70\% agreement)', fontsize=16, fontweight='bold')

# Create sample education distribution
education_levels = ['High School', 'Bachelor', 'Graduate']
education_counts = [300, 400, 300]

bars = ax.bar(education_levels, education_counts, color=['lightcoral', 'lightcoral', 'lightblue'], alpha=0.7)
ax.set_ylim(0, 500)
ax.axhline(y=300, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Split threshold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'surrogate_split_2_education.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 10: Visualizing Surrogate Split 3 - Age
print_step_header(10, "Visualizing Surrogate Split 3 - Age")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('Age (Years)', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(r'Surrogate 3: Age $> 40$ (65\% agreement)', fontsize=16, fontweight='bold')

# Create sample age distribution
age_young = np.random.normal(28, 6, 400)
age_old = np.random.normal(52, 8, 600)

ax.hist(age_young, bins=30, alpha=0.7, color='lightcoral', label=r'Age $\leq 40$')
ax.hist(age_old, bins=30, alpha=0.7, color='lightblue', label=r'Age $> 40$')
ax.axvline(x=40, color='green', linestyle='--', linewidth=3, label=r'Split: $40$ years')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'surrogate_split_3_age.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 11: Decision Tree with Surrogate Splits
print_step_header(11, "Decision Tree with Surrogate Splits Visualization")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw the decision tree structure
def draw_node(x, y, text, color='lightblue', size=0.8):
    box = FancyBboxPatch(
        (x - size/2, y - size/2), size, size,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(start, end, label='', color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8))

# Root node
draw_node(5, 9, 'Income > \\$50K?', 'lightgreen', 1.2)

# Primary split branches
draw_arrow((5, 8.4), (3, 7.5))
draw_arrow((5, 8.4), (7, 7.5))

# Primary split nodes
draw_node(3, 7, 'No\n($\\leq$ \\$50K)', 'lightcoral')
draw_node(7, 7, 'Yes\n($>$ \\$50K)', 'lightblue')

# Surrogate splits for missing income
ax.text(5, 8.8, r'If Income Missing:', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8))

# Surrogate 1
draw_arrow((5, 8.2), (2, 6.5))
draw_arrow((5, 8.2), (8, 6.5))

draw_node(2, 6, 'Experience\n($\\leq$ 8)', 'lightcoral')
draw_node(8, 6, 'Experience\n($>$ 8)', 'lightblue')

ax.text(5, 6.8, r'Surrogate 1 (75\% agreement)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="black", alpha=0.8))

# Surrogate 2
draw_arrow((2, 5.6), (1, 4.5))
draw_arrow((2, 5.6), (3, 4.5))

draw_node(1, 4, 'Education\n($\\neq$ Graduate)', 'lightcoral')
draw_node(3, 4, 'Education\n= Graduate', 'lightblue')

ax.text(2, 4.8, r'Surrogate 2 (70\% agreement)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="black", alpha=0.8))

# Surrogate 3
draw_arrow((8, 5.6), (7, 4.5))
draw_arrow((8, 5.6), (9, 4.5))

draw_node(7, 4, 'Age\n($\\leq$ 40)', 'lightcoral')
draw_node(9, 4, 'Age\n($>$ 40)', 'lightblue')

ax.text(8, 4.8, r'Surrogate 3 (65\% agreement)', ha='center', va='center', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="black", alpha=0.8))

# Leaf nodes
draw_node(1, 3, 'Class 0\n(Default)', 'lightcoral')
draw_node(3, 3, 'Class 1\n(Default)', 'lightblue')
draw_node(7, 3, 'Class 0\n(Default)', 'lightcoral')
draw_node(9, 3, 'Class 1\n(Default)', 'lightblue')

# Connect to leaf nodes
draw_arrow((1, 4.4), (1, 3.4))
draw_arrow((3, 4.4), (3, 3.4))
draw_arrow((7, 4.4), (7, 3.4))
draw_arrow((9, 4.4), (9, 3.4))

# Add title and explanation
ax.text(5, 0.5, r'Surrogate splits provide fallback options when primary features are missing', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", alpha=0.8))

plt.savefig(os.path.join(save_dir, 'decision_tree_with_surrogates.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 12: Key Insights
print_step_header(12, "Key Insights")

print("1. Surrogate Split Quality:")
print("   - Higher agreement rate = better surrogate")
print("   - Multiple surrogates provide redundancy")
print("   - Quality decreases as we go down the surrogate list")
print()
print("2. Robustness Benefits:")
print("   - Handles missing values without data imputation")
print("   - Preserves tree structure and interpretability")
print("   - Provides graceful degradation when features are missing")
print()
print("3. Practical Implementation:")
print("   - Try surrogates in order of quality")
print("   - Use majority class as final fallback")
print("   - Monitor surrogate performance on validation data")
print()

# Step 13: Answer Summary
print_step_header(13, "Answer Summary")

print("1. Surrogate splits definition:")
print("   Surrogate splits are alternative splitting rules that closely mimic the")
print("   behavior of the primary split when the primary feature is missing.")
print()
print("2. Ranking by quality:")
print("   1. Experience > 8: 75% agreement (best)")
print("   2. Education = Graduate: 70% agreement")
print("   3. Age > 40: 65% agreement (worst)")
print()
print("3. Why more robust than simple imputation:")
print("   - Data-driven approach based on actual feature relationships")
print("   - Multiple fallback options with graceful degradation")
print("   - Preserves tree structure and interpretability")
print()
print("4. Using best surrogate for new samples:")
print("   - Try primary split first (Income > $50K)")
print("   - If missing, use Experience > 8 (75% agreement)")
print("   - Continue down surrogate list if needed")
print("   - Use majority class as final fallback")
print()

print(f"\nVisualizations saved to: {save_dir}")
print("The plots show the concept of surrogate splits and how they handle missing values.")
print("Each plot is saved as a separate file for better clarity and use.")
