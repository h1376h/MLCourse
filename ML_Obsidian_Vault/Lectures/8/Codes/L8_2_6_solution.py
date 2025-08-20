import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L8_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("=" * 80)
print("QUESTION 6: CHI-SQUARE TEST FOR INDEPENDENCE")
print("=" * 80)

# ============================================================================
# PART 1: Chi-square test statistic formula
# ============================================================================
print("\n1. CHI-SQUARE TEST STATISTIC FORMULA")
print("-" * 50)

print("The chi-square test statistic formula is:")
print("chi^2 = Σ[(O - E)^2 / E]")
print("where:")
print("  O = Observed frequency")
print("  E = Expected frequency")
print("  Σ = Sum over all cells in the contingency table")

print("\nAlternative forms:")
print("chi^2 = Σ[(O - E)^2 / E] = Σ[O^2/E] - N")
print("where N is the total sample size")

# ============================================================================
# PART 2: When is chi-square test appropriate
# ============================================================================
print("\n2. WHEN IS CHI-SQUARE TEST APPROPRIATE?")
print("-" * 50)

print("The chi-square test is appropriate when:")
print("• Both variables are categorical (nominal or ordinal)")
print("• Observations are independent")
print("• Expected frequency in each cell ≥ 5 (for small samples)")
print("• Sample size is sufficiently large (typically N ≥ 20)")
print("• Data is randomly sampled from the population")

print("\nWhen NOT to use chi-square test:")
print("• When variables are continuous")
print("• When expected frequencies are too small (< 5)")
print("• When observations are not independent")
print("• For very small sample sizes")

# ============================================================================
# PART 3: Contingency table analysis
# ============================================================================
print("\n3. CONTINGENCY TABLE ANALYSIS")
print("-" * 50)

# Given contingency table
print("Given contingency table:")
print("| Feature\\Target | Class 0 | Class 1 |")
print("|----------------|---------|---------|")
print("| Category A     |    25   |    15   |")
print("| Category B     |    20   |    30   |")

# Create the observed data
observed = np.array([[25, 15], [20, 30]])
print(f"\nObserved frequencies matrix:")
print(observed)

# Calculate row and column totals
row_totals = np.sum(observed, axis=1)
col_totals = np.sum(observed, axis=0)
total = np.sum(observed)

print(f"\nRow totals: {row_totals}")
print(f"Column totals: {col_totals}")
print(f"Total sample size: {total}")

# Calculate expected frequencies
expected = np.outer(row_totals, col_totals) / total
print(f"\nExpected frequencies matrix:")
print(expected)

# Calculate chi-square contributions for each cell
chi_square_contributions = (observed - expected)**2 / expected
print(f"\nChi-square contributions for each cell:")
print(chi_square_contributions)

# Calculate total chi-square statistic
chi_square_stat = np.sum(chi_square_contributions)
print(f"\nChi-square statistic: chi^2 = {chi_square_stat:.4f}")

# Calculate degrees of freedom
df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
print(f"Degrees of freedom: df = {df}")

# Critical value at α = 0.05
alpha = 0.05
critical_value = stats.chi2.ppf(1 - alpha, df)
print(f"Critical value at alpha = {alpha}: chi^2_critical = {critical_value:.4f}")

# Decision
if chi_square_stat > critical_value:
    decision = "REJECT"
    conclusion = "The feature is NOT independent of the target"
else:
    decision = "FAIL TO REJECT"
    conclusion = "The feature IS independent of the target"

print(f"\nDecision: {decision} the null hypothesis")
print(f"Conclusion: {conclusion}")

# P-value
p_value = 1 - stats.chi2.cdf(chi_square_stat, df)
print(f"P-value: p = {p_value:.6f}")

# ============================================================================
# PART 4: 4x3 contingency table analysis
# ============================================================================
print("\n4. 4×3 CONTINGENCY TABLE ANALYSIS")
print("-" * 50)

# For a 4×3 contingency table
rows, cols = 4, 3
df_4x3 = (rows - 1) * (cols - 1)
print(f"Degrees of freedom for 4×3 table: df = ({rows}-1) × ({cols}-1) = {df_4x3}")

# Given observed chi-square statistic
observed_chi_square = 18.5
alpha_4x3 = 0.01
critical_value_4x3 = stats.chi2.ppf(1 - alpha_4x3, df_4x3)

print(f"Observed chi-square statistic: chi^2 = {observed_chi_square}")
print(f"Critical value at alpha = {alpha_4x3}: chi^2_critical = {critical_value_4x3:.4f}")

# Decision for 4×3 table
if observed_chi_square > critical_value_4x3:
    decision_4x3 = "REJECT"
    conclusion_4x3 = "The null hypothesis of independence should be REJECTED"
else:
    decision_4x3 = "FAIL TO REJECT"
    conclusion_4x3 = "The null hypothesis of independence should NOT be rejected"

print(f"\nDecision: {decision_4x3} the null hypothesis")
print(f"Conclusion: {conclusion_4x3}")

# P-value for 4×3 table
p_value_4x3 = 1 - stats.chi2.cdf(observed_chi_square, df_4x3)
print(f"P-value: p = {p_value_4x3:.6f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Contingency table heatmap
plt.figure(figsize=(12, 10))

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Observed frequencies heatmap
im1 = ax1.imshow(observed, cmap='Blues', aspect='auto')
ax1.set_title('Observed Frequencies', fontsize=14, fontweight='bold')
ax1.set_xlabel('Target Class')
ax1.set_ylabel('Feature Category')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Class 0', 'Class 1'])
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Category A', 'Category B'])

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, f'{observed[i, j]}', ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='white')
        if observed[i, j] < 25:  # Make text black for light cells
            text.set_color('black')

plt.colorbar(im1, ax=ax1, shrink=0.8)

# Plot 2: Expected frequencies heatmap
im2 = ax2.imshow(expected, cmap='Greens', aspect='auto')
ax2.set_title('Expected Frequencies', fontsize=14, fontweight='bold')
ax2.set_xlabel('Target Class')
ax2.set_ylabel('Feature Category')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Class 0', 'Class 1'])
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Category A', 'Category B'])

# Add text annotations
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{expected[i, j]:.1f}', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')

plt.colorbar(im2, ax=ax2, shrink=0.8)

# Plot 3: Chi-square contributions heatmap
im3 = ax3.imshow(chi_square_contributions, cmap='Reds', aspect='auto')
ax3.set_title('Chi-Square Contributions', fontsize=14, fontweight='bold')
ax3.set_xlabel('Target Class')
ax3.set_ylabel('Feature Category')
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Class 0', 'Class 1'])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Category A', 'Category B'])

# Add text annotations
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{chi_square_contributions[i, j]:.3f}', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')

plt.colorbar(im3, ax=ax3, shrink=0.8)

# Plot 4: Chi-square distribution with critical value
x = np.linspace(0, 10, 1000)
chi2_pdf = stats.chi2.pdf(x, df)
ax4.plot(x, chi2_pdf, 'b-', linewidth=2, label=f'$\\chi^2$ distribution (df={df})')
ax4.axvline(chi_square_stat, color='red', linestyle='--', linewidth=2, 
            label=f'Observed $\\chi^2$ = {chi_square_stat:.4f}')
ax4.axvline(critical_value, color='green', linestyle='--', linewidth=2, 
            label=f'Critical value = {critical_value:.4f}')

# Shade rejection region
x_reject = np.linspace(critical_value, 10, 100)
ax4.fill_between(x_reject, stats.chi2.pdf(x_reject, df), alpha=0.3, color='red', 
                label=f'Rejection region (alpha={alpha})')

ax4.set_title('Chi-Square Distribution and Test Results', fontsize=14, fontweight='bold')
ax4.set_xlabel('Chi-Square Value')
ax4.set_ylabel('Probability Density')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'chi_square_analysis.png'), dpi=300, bbox_inches='tight')

# 2. Detailed contingency table visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Create a more detailed table visualization
table_data = [
    ['Feature/Target', 'Class 0', 'Class 1', 'Row Total'],
    ['Category A', '25', '15', '40'],
    ['Category B', '20', '30', '50'],
    ['Column Total', '45', '45', '90']
]

# Create table
table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Style the table
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        if i == 0 or j == 0:  # Header row and column
            table[(i, j)].set_facecolor('#4CAF50')
            table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            table[(i, j)].set_facecolor('#E8F5E8')

ax.set_title('Contingency Table with Margins', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'contingency_table.png'), dpi=300, bbox_inches='tight')

# 3. Chi-square test results summary
fig, ax = plt.subplots(figsize=(12, 8))

# Create summary text
summary_text = f"""
Chi-Square Test Results Summary

Test Statistic: chi^2 = {chi_square_stat:.4f}
Degrees of Freedom: df = {df}
Critical Value (alpha = {alpha}): {critical_value:.4f}
P-value: {p_value:.6f}

Decision: {decision}
Conclusion: {conclusion}

Expected Frequencies:
Category A, Class 0: {expected[0,0]:.1f}
Category A, Class 1: {expected[0,1]:.1f}
Category B, Class 0: {expected[1,0]:.1f}
Category B, Class 1: {expected[1,1]:.1f}

Chi-Square Contributions:
Category A, Class 0: {chi_square_contributions[0,0]:.4f}
Category A, Class 1: {chi_square_contributions[0,1]:.4f}
Category B, Class 0: {chi_square_contributions[1,0]:.4f}
Category B, Class 1: {chi_square_contributions[1,1]:.4f}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

ax.set_title('Chi-Square Test Results Summary', fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

plt.savefig(os.path.join(save_dir, 'test_results_summary.png'), dpi=300, bbox_inches='tight')

# 4. Degrees of freedom visualization for different table sizes
fig, ax = plt.subplots(figsize=(10, 8))

# Create sample table sizes
table_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (5, 4)]
df_values = [(r-1)*(c-1) for r, c in table_sizes]

# Create bar plot
x_pos = np.arange(len(table_sizes))
bars = ax.bar(x_pos, df_values, color='skyblue', edgecolor='navy', alpha=0.7)

# Add labels
ax.set_xlabel('Table Size (rows × columns)')
ax.set_ylabel('Degrees of Freedom')
ax.set_title('Degrees of Freedom for Different Contingency Table Sizes', 
            fontsize=14, fontweight='bold')

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{r}×{c}' for r, c in table_sizes])

# Add value labels on bars
for bar, df_val in zip(bars, df_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{df_val}', ha='center', va='bottom', fontweight='bold')

# Highlight the 4×3 case
highlight_idx = 4  # Index of 4×3 table
bars[highlight_idx].set_color('red')
bars[highlight_idx].set_alpha(0.8)

ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'degrees_of_freedom.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"\nPart 3 Results:")
print(f"• Chi-square statistic: {chi_square_stat:.4f}")
print(f"• Degrees of freedom: {df}")
print(f"• Critical value (alpha = {alpha}): {critical_value:.4f}")
print(f"• P-value: {p_value:.6f}")
print(f"• Decision: {decision}")
print(f"• Conclusion: {conclusion}")

print(f"\nPart 4 Results:")
print(f"• Degrees of freedom for 4×3 table: {df_4x3}")
print(f"• Critical value (alpha = {alpha_4x3}): {critical_value_4x3:.4f}")
print(f"• Decision: {decision_4x3}")
print(f"• Conclusion: {conclusion_4x3}")

print(f"\nFiles generated:")
print(f"• chi_square_analysis.png - Main analysis plots")
print(f"• contingency_table.png - Contingency table visualization")
print(f"• test_results_summary.png - Test results summary")
print(f"• degrees_of_freedom.png - Degrees of freedom comparison")
