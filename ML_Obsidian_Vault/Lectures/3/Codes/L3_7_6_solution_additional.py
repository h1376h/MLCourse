import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Create a theoretical comparison visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Define the space for visualization
rho_values = np.linspace(0, 1, 100)  # L1 ratio from 0 (Ridge) to 1 (Lasso)
properties = ['Grouping Effect', 'Sparsity', 'Stability', 'Handles Multicollinearity']

# Define how properties vary with l1_ratio
grouping_effect = 1 - rho_values**2  # Strong at rho=0, decreases as rho increases
sparsity = rho_values  # Increases linearly with rho
stability = 1 - 0.7*rho_values  # Decreases as rho increases
multicollinearity = 1 - 0.8*rho_values  # Decreases as rho increases

# Plot the properties
ax.plot(rho_values, grouping_effect, 'b-', linewidth=3, label='Grouping Effect')
ax.plot(rho_values, sparsity, 'r-', linewidth=3, label='Sparsity')
ax.plot(rho_values, stability, 'g-', linewidth=3, label='Stability')
ax.plot(rho_values, multicollinearity, 'purple', linewidth=3, label='Handles Multicollinearity')

# Add vertical lines for key values
ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Ridge')
ax.axvline(x=1, color='k', linestyle=':', alpha=0.5, label='Lasso')

# Add horizontal line at y=0.5 for reference
ax.axhline(y=0.5, color='k', linestyle='-', alpha=0.2)

# Add regions with annotations
ax.fill_between([0, 0.3], [0, 0], [1, 1], color='blue', alpha=0.1)
ax.fill_between([0.3, 0.7], [0, 0], [1, 1], color='green', alpha=0.1)
ax.fill_between([0.7, 1], [0, 0], [1, 1], color='red', alpha=0.1)

ax.text(0.15, 0.05, "Ridge-like\nBest for\nmulticollinearity", ha='center', fontsize=10)
ax.text(0.5, 0.05, "Balanced\nCompromise", ha='center', fontsize=10)
ax.text(0.85, 0.05, "Lasso-like\nBest for\nsparsity", ha='center', fontsize=10)

# Customize the plot
ax.set_xlabel(r'Mixing Parameter ($\rho$)', fontsize=14)
ax.set_ylabel('Relative Strength of Property', fontsize=14)
ax.set_title('Theoretical Properties of Elastic Net by Mixing Parameter', fontsize=16)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12)

# Add annotations for parameter regions
ax.annotate('Ridge ($\\rho = 0$)', xy=(0, -0.05), xytext=(0, -0.05), xycoords=('data', 'axes fraction'),
           ha='center', va='top', fontsize=10)
ax.annotate('Lasso ($\\rho = 1$)', xy=(1, -0.05), xytext=(1, -0.05), xycoords=('data', 'axes fraction'), 
           ha='center', va='top', fontsize=10)

# Add parameter choice guidance
parameter_text = (
    "Parameter Selection Guidance:\n"
    "- High multicollinearity → Lower $\\rho$ (Ridge-like)\n"
    "- Need feature selection → Higher $\\rho$ (Lasso-like)\n"
    "- Need grouping effect → Lower $\\rho$\n"
    "- Balanced needs → Middle $\\rho$ values (0.3-0.7)"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.5, 0.8, parameter_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', horizontalalignment='center', bbox=props)

# Add alpha guidance
plt.figtext(0.5, 0.18, 
           "$\\alpha$ (total regularization strength) should be selected via cross-validation\n"
           "and depends on noise level, sample size, and number of features",
           ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "elastic_net_theoretical_comparison.png"), dpi=300, bbox_inches='tight')
print(f"Theoretical comparison figure saved to: {save_dir}/elastic_net_theoretical_comparison.png") 