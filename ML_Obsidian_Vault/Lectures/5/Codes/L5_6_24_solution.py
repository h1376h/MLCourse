import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L5_6_Quiz_24")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=" * 60)
print("Question 24: SVM Algorithm Understanding")
print("=" * 60)

# Task 1: What does SMO stand for?
print("\n1. What does SMO stand for?")
print("   SMO stands for Sequential Minimal Optimization")
print("   - Sequential: Optimizes variables in sequence")
print("   - Minimal: Uses the smallest possible working set")
print("   - Optimization: Solves the SVM dual optimization problem")

# Task 2: Main idea behind SMO algorithm
print("\n2. Main idea behind SMO algorithm:")
print("   SMO breaks down the large quadratic programming problem into")
print("   a series of smallest possible sub-problems that can be solved analytically.")
print("   - Instead of optimizing all α variables simultaneously")
print("   - SMO optimizes exactly 2 variables at each iteration")
print("   - Each 2-variable sub-problem has an analytical solution")

# Task 3: Why optimize two variables at a time?
print("\n3. Why does SMO optimize two variables at a time instead of one?")
print("   Mathematical proof:")
print("   Given constraint: Σᵢ αᵢyᵢ = 0")
print("   If we fix all αₖ for k ≠ j, then:")
print("   αⱼyⱼ + Σₖ≠ⱼ αₖyₖ = 0")
print("   Therefore: αⱼ = -yⱼ Σₖ≠ⱼ αₖyₖ")
print("   Since yⱼ ∈ {-1, +1}, we have yⱼ² = 1")
print("   So αⱼ is completely determined by other variables")
print("   Need at least 2 free variables for optimization freedom")

# Visualization 1: SMO Concept
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Traditional QP approach
ax1.set_title('Traditional QP Approach', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)

# Draw a large optimization problem box
large_box = FancyBboxPatch((1, 2), 8, 4, boxstyle="round,pad=0.1", 
                          facecolor='lightcoral', edgecolor='darkred', linewidth=2)
ax1.add_patch(large_box)
ax1.text(5, 4, r'Optimize all $\alpha_1, \alpha_2, ..., \alpha_n$' + '\nsimultaneously\n(Large QP Problem)', 
         ha='center', va='center', fontsize=11, fontweight='bold')

# Add complexity annotation
ax1.text(5, 1, r'Complexity: $O(n^3)$ for $n$ variables', ha='center', va='center', 
         fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Right plot: SMO approach
ax2.set_title('SMO Approach', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)

# Draw small optimization problems
colors = ['lightblue', 'lightgreen', 'lightyellow']
for i, color in enumerate(colors):
    small_box = FancyBboxPatch((1 + i*2.5, 3), 2, 2, boxstyle="round,pad=0.1", 
                              facecolor=color, edgecolor='darkblue', linewidth=1.5)
    ax2.add_patch(small_box)
    ax2.text(2 + i*2.5, 4, f'Optimize\n' + r'$\alpha_i, \alpha_j$', 
             ha='center', va='center', fontsize=10, fontweight='bold')

# Add arrows between boxes
for i in range(len(colors)-1):
    ax2.annotate('', xy=(3.5 + i*2.5, 4), xytext=(3 + i*2.5, 4),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

ax2.text(5, 1.5, 'Analytical solution for each 2-variable subproblem', 
         ha='center', va='center', fontsize=10, style='italic',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))

ax2.text(5, 0.5, r'Complexity: $O(1)$ per iteration', ha='center', va='center', 
         fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))

ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'smo_concept.png'), dpi=300, bbox_inches='tight')
plt.close()

# Task 4: What is a "working set" in SVM optimization?
print("\n4. What is a 'working set' in SVM optimization?")
print("   A working set is a subset of variables that are actively optimized")
print("   while keeping the remaining variables fixed.")
print("   - In SMO: working set size = 2 (minimal possible)")
print("   - In chunking methods: working set size > 2")
print("   - Variables not in working set remain at their current values")
print("   - Working set selection affects convergence speed")

# Task 5: Type of optimization problem
print("\n5. What type of optimization problem is the SVM dual formulation?")
print("   The SVM dual formulation is a Quadratic Programming (QP) problem:")
print("   ")
print("   Objective function (quadratic in α):")
print("   L_D(α) = Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)")
print("   ")
print("   Subject to constraints:")
print("   1. Equality: Σᵢ αᵢyᵢ = 0")
print("   2. Box constraints: 0 ≤ αᵢ ≤ C for all i = 1,...,n")
print("   ")
print("   QP characteristics:")
print("   - Quadratic objective function (convex)")
print("   - Linear constraints")
print("   - Guaranteed global optimum")

# Visualization 2: Working Set Concept
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_title('Working Set in SVM Optimization', fontsize=16, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)

# Draw all variables
n_vars = 8
var_width = 1.2
var_height = 1.5
start_x = 1
start_y = 6

# Draw all alpha variables
for i in range(n_vars):
    x = start_x + i * var_width
    if i in [2, 5]:  # Working set variables
        color = 'lightgreen'
        edge_color = 'darkgreen'
        linewidth = 3
    else:  # Fixed variables
        color = 'lightgray'
        edge_color = 'gray'
        linewidth = 1
    
    var_box = Rectangle((x, start_y), var_width*0.8, var_height, 
                       facecolor=color, edgecolor=edge_color, linewidth=linewidth)
    ax.add_patch(var_box)
    ax.text(x + var_width*0.4, start_y + var_height*0.5, f'$\\alpha_{{{i+1}}}$', 
            ha='center', va='center', fontsize=12, fontweight='bold')

# Add working set label
working_set_box = FancyBboxPatch((start_x + 2*var_width - 0.1, start_y - 0.5), 
                                var_width*0.8 + 3*var_width + 0.2, var_height + 1, 
                                boxstyle="round,pad=0.1", facecolor='none', 
                                edgecolor='red', linewidth=2, linestyle='--')
ax.add_patch(working_set_box)
ax.text(start_x + 3.5*var_width, start_y + var_height + 0.7, 'Working Set', 
        ha='center', va='center', fontsize=12, fontweight='bold', color='red')

# Add legend
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='darkgreen', 
                      linewidth=2, label='Variables being optimized'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='gray', 
                      linewidth=1, label='Fixed variables')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

# Add constraint equation
ax.text(6, 4, r'Constraint: $\sum_{i=1}^n \alpha_i y_i = 0$', 
        ha='center', va='center', fontsize=14, 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', edgecolor='orange'))

# Add explanation text
explanation = ("In SMO:\n"
              "• Working set size = 2\n"
              "• Optimize exactly 2 variables per iteration\n"
              "• All other variables remain fixed\n"
              "• Constraint automatically satisfied")
ax.text(6, 2, explanation, ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', edgecolor='blue'))

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'working_set_concept.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: QP Problem Structure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_title('SVM Dual as Quadratic Programming Problem', fontsize=16, fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)

# Objective function
obj_box = FancyBboxPatch((1, 9), 10, 1.5, boxstyle="round,pad=0.1", 
                        facecolor='lightcoral', edgecolor='darkred', linewidth=2)
ax.add_patch(obj_box)
ax.text(6, 9.75, r'Maximize: $\sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j K(x_i, x_j)$', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Constraints
const1_box = FancyBboxPatch((1, 6.5), 10, 1, boxstyle="round,pad=0.1", 
                           facecolor='lightblue', edgecolor='darkblue', linewidth=2)
ax.add_patch(const1_box)
ax.text(6, 7, r'Subject to: $\sum_{i=1}^n \alpha_i y_i = 0$ (Equality constraint)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

const2_box = FancyBboxPatch((1, 5), 10, 1, boxstyle="round,pad=0.1", 
                           facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
ax.add_patch(const2_box)
ax.text(6, 5.5, r'$0 \leq \alpha_i \leq C$ for all $i$ (Box constraints)', 
        ha='center', va='center', fontsize=12, fontweight='bold')

# Problem characteristics
char_box = FancyBboxPatch((1, 2), 10, 2, boxstyle="round,pad=0.1", 
                         facecolor='lightyellow', edgecolor='orange', linewidth=2)
ax.add_patch(char_box)
characteristics = ("Quadratic Programming (QP) Problem Characteristics:\n"
                  "• Quadratic objective function (convex)\n"
                  "• Linear constraints\n"
                  "• Guaranteed global optimum\n"
                  "• Can be solved by specialized QP solvers")
ax.text(6, 3, characteristics, ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'qp_problem_structure.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll visualizations saved to: {save_dir}")
print("\nSummary of Question 24:")
print("1. SMO = Sequential Minimal Optimization")
print("2. Main idea: Break large QP into smallest analytical subproblems")
print("3. Two variables needed due to equality constraint")
print("4. Working set = subset of variables being optimized")
print("5. SVM dual = Quadratic Programming problem")
