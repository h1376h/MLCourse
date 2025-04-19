import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_23")
os.makedirs(save_dir, exist_ok=True)

# Set LaTeX style for plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': True if os.system('which latex') == 0 else False,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Define creative functions for this question
# Creating a scenario with a multimodal likelihood with an interesting prior

# Marginal distributions
def f_X(x):
    """Marginal PDF of X - bimodal distribution"""
    return 0.1 + 0.3 * np.exp(-(x-1)**2/0.4) + 0.5 * np.exp(-(x-3)**2/0.6)

def f_Y(y):
    """Marginal PDF of Y - mixture of exponential and normal"""
    return 0.5 * np.exp(-y) + 0.5 * np.exp(-(y-2)**2/1.0)

# Conditional distributions
def f_X_given_Y_3(y):
    """PDF of X given Y=3"""
    # Sharp peak at y=0.5 and smaller peak at y=3
    return 0.3 * np.exp(-(y-0.5)**2/0.1) + 0.1 * np.exp(-(y-3)**2/0.3)

def f_Y_given_X_3(y):
    """PDF of Y given X=3 - multimodal distribution"""
    return 0.2 * np.exp(-(y-0.5)**2/0.1) + 0.1 * np.exp(-(y-2)**2/0.2) + 0.05 * np.exp(-(y-3.5)**2/0.15)

# Conditional expectations 
def E_Y_given_X(x):
    """Conditional expectation of Y given X=x - distinctive curve with inflection"""
    return 1.0 + 1.0 * np.sin(x * np.pi/2)

def E_X_given_Y(y):
    """Conditional expectation of X given Y=y - distinctive curve with inflection"""
    return 2.0 + 0.5 * np.cos(y * np.pi/2)

# Create grids for x and y values
x = np.linspace(0, 4, 200)
y = np.linspace(0, 4, 200)
X, Y = np.meshgrid(x, y)

# Plot each graph separately
def save_individual_plot(func, x_data, y_data, xlabel, ylabel, title, filename, xlim=(0, 4), ylim=None, grid=False):
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if grid:
        plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Plot f_X|Y(3|y)
save_individual_plot(
    f_X_given_Y_3, y, f_X_given_Y_3(y),
    r"$y$", r"$f_{X|Y}(3|y)$", r"$f_{X|Y}(3|Y=y)$",
    "graph1_f_X_given_Y.png", ylim=(0, 0.8)
)

# 2. Plot f_Y(y)
save_individual_plot(
    f_Y, y, f_Y(y),
    r"$y$", r"$f_Y(y)$", r"$f_Y(y)$",
    "graph2_f_Y.png", ylim=(0, 1)
)

# 3. Plot E[Y|X=x]
save_individual_plot(
    E_Y_given_X, x, E_Y_given_X(x),
    r"$x$", r"$E[Y|X=x]$", r"$E[Y|X=x]$",
    "graph3_E_Y_given_X.png", ylim=(0, 3), grid=True
)

# 4. Plot f_Y|X(y|X=3)
save_individual_plot(
    f_Y_given_X_3, y, f_Y_given_X_3(y),
    r"$y$", r"$f_{Y|X}(y|X=3)$", r"$f_{Y|X}(y|X=3)$",
    "graph4_f_Y_given_X.png", ylim=(0, 0.25)
)

# 5. Plot f_X(x)
save_individual_plot(
    f_X, x, f_X(x),
    r"$x$", r"$f_X(x)$", r"$f_X(x)$",
    "graph5_f_X.png", ylim=(0, 1)
)

# 6. Plot E[X|Y=y]
save_individual_plot(
    E_X_given_Y, y, E_X_given_Y(y),
    r"$y$", r"$E[X|Y=y]$", r"$E[X|Y=y]$",
    "graph6_E_X_given_Y.png", ylim=(1.5, 3), grid=True
)

print(f"Individual graphs saved in '{save_dir}'")

# Calculate and visualize ML, MAP, and MMSE estimates
y_values = np.linspace(0, 4, 1000)
likelihood = f_Y_given_X_3(y_values)
prior = f_Y(y_values)
posterior = likelihood * prior / np.trapz(likelihood * prior, y_values)

# Find ML estimate (maximizes likelihood)
ml_index = np.argmax(likelihood)
ml_estimate = y_values[ml_index]

# Find MAP estimate (maximizes posterior)
map_index = np.argmax(posterior)
map_estimate = y_values[map_index]

# Find MMSE estimate (expected value of posterior)
mmse_estimate = np.trapz(y_values * posterior, y_values)

print(f"ML Estimate: y = {ml_estimate:.2f}")
print(f"MAP Estimate: y = {map_estimate:.2f}")
print(f"MMSE Estimate: y = {mmse_estimate:.2f}")

# Create visualizations for the solution explanation
plt.figure(figsize=(10, 6))
plt.plot(y_values, likelihood / np.max(likelihood), 'g--', label='Likelihood (normalized)')
plt.axvline(x=ml_estimate, color='g', linestyle='-', 
            label=f'ML Estimate: y = {ml_estimate:.2f}')
plt.title('Maximum Likelihood Estimation for Y given X=3')
plt.xlabel('y')
plt.ylabel('Normalized Likelihood')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, "ml_estimate.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(y_values, likelihood / np.max(likelihood), 'g--', label='Likelihood (normalized)')
plt.plot(y_values, prior / np.max(prior), 'r-.', label='Prior (normalized)')
plt.plot(y_values, posterior / np.max(posterior), 'b-', label='Posterior (normalized)')
plt.axvline(x=ml_estimate, color='g', linestyle='-', 
            label=f'ML Estimate: y = {ml_estimate:.2f}')
plt.axvline(x=map_estimate, color='b', linestyle='-', 
            label=f'MAP Estimate: y = {map_estimate:.2f}')
plt.title('MAP Estimation: Likelihood, Prior, and Posterior for Y given X=3')
plt.xlabel('y')
plt.ylabel('Normalized Density')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, "map_estimate.png"), dpi=300, bbox_inches='tight')
plt.close()

# MMSE estimate visualization using E[Y|X=x]
x_range = np.linspace(0, 4, 100)
expected_y = E_Y_given_X(x_range)
plt.figure(figsize=(10, 6))
plt.plot(x_range, expected_y, 'b-', linewidth=2, label='E[Y|X=x]')
plt.scatter([3], [E_Y_given_X(3)], color='r', s=100, 
            label=f'MMSE Estimate at X=3: y = {E_Y_given_X(3):.2f}')
plt.xlabel('x')
plt.ylabel('E[Y|X=x]')
plt.title('MMSE Estimate: Conditional Expectation E[Y|X=x]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, "mmse_estimate.png"), dpi=300, bbox_inches='tight')
plt.close()

# Comparison of all estimates
plt.figure(figsize=(10, 6))
plt.plot(y_values, likelihood / np.max(likelihood), 'g--', linewidth=2, label='Likelihood (normalized)')
plt.plot(y_values, prior / np.max(prior), 'r-.', linewidth=2, label='Prior (normalized)')
plt.plot(y_values, posterior / np.max(posterior), 'b-', linewidth=2, label='Posterior (normalized)')
plt.axvline(x=ml_estimate, color='g', linestyle='-', 
            label=f'ML Estimate: y = {ml_estimate:.2f}', linewidth=2)
plt.axvline(x=map_estimate, color='b', linestyle='-', 
            label=f'MAP Estimate: y = {map_estimate:.2f}', linewidth=2)
plt.axvline(x=mmse_estimate, color='purple', linestyle='-', 
            label=f'MMSE Estimate: y = {mmse_estimate:.2f}', linewidth=2)
plt.xlabel('y')
plt.ylabel('Normalized Density')
plt.title('Comparison of ML, MAP, and MMSE Estimates for Y given X=3')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(os.path.join(save_dir, "all_estimates_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved in '{save_dir}'") 