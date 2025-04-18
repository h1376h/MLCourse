import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Set LaTeX style for plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Define functions with simple, clear distributions that make estimates easy to spot
# Marginal distributions
def f_X(x):
    """Marginal PDF of X - simple quadratic function peaking at x=3"""
    return 0.1 + 0.2 * (1 - (x - 3)**2 / 9)

def f_Y(y):
    """Marginal PDF of Y - simple triangular distribution peaking at y=1"""
    if y <= 1:
        return 0.5 * y
    else:
        return 0.5 * (4 - y) / 3

# Conditional distributions - designed for clear ML and MAP estimates
def f_X_given_Y_2(y):
    """PDF of X given Y=2"""
    # Simple gaussian-like function
    return 0.3 * np.exp(-(y - 2.5)**2 / 0.5)

def f_Y_given_X_2(y):
    """PDF of Y given X=2 - clear peak at y=2"""
    # Simple gaussian with clear peak at y=2
    return 0.3 * np.exp(-(y - 2)**2 / 0.3)

# Conditional expectations - simple linear functions
def E_Y_given_X(x):
    """Conditional expectation of Y given X=x - exactly E[Y|X=2] = 3"""
    return 1 + x # This makes E[Y|X=2] = 3

def E_X_given_Y(y):
    """Conditional expectation of X given Y=y"""
    return 2 + 0.5 * (y - 2) # Centers at (2,2)

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
    # Add a thicker vertical line at x=2 or y=2 for easier reading
    if 'E[Y|X=x]' in title:
        plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=3, color='r', linestyle='--', alpha=0.5)
        plt.plot(2, 3, 'ro', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Plot f_X|Y(2|y)
save_individual_plot(
    f_X_given_Y_2, y, f_X_given_Y_2(y),
    r"$y$", r"$f_{X|Y}(2|y)$", r"$f_{X|Y}(2|Y=y)$",
    "graph1_f_X_given_Y.png", ylim=(0, 0.5)
)

# 2. Plot f_Y(y)
save_individual_plot(
    f_Y, y, [f_Y(yi) for yi in y],
    r"$y$", r"$f_Y(y)$", r"$f_Y(y)$",
    "graph2_f_Y.png", ylim=(0, 0.8)
)

# 3. Plot E[Y|X=x]
save_individual_plot(
    E_Y_given_X, x, E_Y_given_X(x),
    r"$x$", r"$E[Y|X=x]$", r"$E[Y|X=x]$",
    "graph3_E_Y_given_X.png", ylim=(0, 5), grid=True
)

# 4. Plot f_Y|X(y|X=2)
save_individual_plot(
    f_Y_given_X_2, y, f_Y_given_X_2(y),
    r"$y$", r"$f_{Y|X}(y|X=2)$", r"$f_{Y|X}(y|X=2)$",
    "graph4_f_Y_given_X.png", ylim=(0, 0.4)
)

# 5. Plot f_X(x)
save_individual_plot(
    f_X, x, f_X(x),
    r"$x$", r"$f_X(x)$", r"$f_X(x)$",
    "graph5_f_X.png", ylim=(0, 0.4)
)

# 6. Plot E[X|Y=y]
save_individual_plot(
    E_X_given_Y, y, E_X_given_Y(y),
    r"$y$", r"$E[X|Y=y]$", r"$E[X|Y=y]$",
    "graph6_E_X_given_Y.png", ylim=(0, 4), grid=True
)

print(f"Individual graphs saved in '{save_dir}'")

# Create "answer key" info for the generated graphs
print("\n==== ANSWER KEY ====")

# For ML estimate - the peak of the likelihood is exactly at y=2
ml_y = 2
print(f"Maximum Likelihood Estimate: {ml_y}")

# For MAP estimate - we need to multiply likelihood by prior and find the peak
# The triangular prior should pull the MAP estimate to y=1
# Compute actual MAP estimate to verify
y_fine = np.linspace(0, 4, 1000)
likelihood = np.array([f_Y_given_X_2(yi) for yi in y_fine])
prior = np.array([f_Y(yi) for yi in y_fine])
posterior = likelihood * prior
map_y = y_fine[np.argmax(posterior)]
print(f"Maximum A Posteriori Estimate: {map_y:.1f}")

# For MMSE estimate - read off the value from E[Y|X=x] at x=2
mmse_y = E_Y_given_X(2)
print(f"Minimum Mean Squared Error Estimate: {mmse_y}")

# Create a visualization for debugging/answer verification
plt.figure(figsize=(10, 6))
plt.plot(y_fine, likelihood / np.max(likelihood), 'g--', linewidth=2, label='Likelihood (normalized)')
plt.plot(y_fine, prior / np.max(prior), 'r-.', linewidth=2, label='Prior (normalized)')
plt.plot(y_fine, posterior / np.max(posterior), 'b-', linewidth=2, label='Posterior (normalized)')
plt.axvline(x=ml_y, color='green', linestyle='--', label=f'ML Estimate: y = {ml_y}')
plt.axvline(x=map_y, color='blue', linestyle='--', label=f'MAP Estimate: y = {map_y:.1f}')
plt.axvline(x=mmse_y, color='purple', linestyle='--', label=f'MMSE Estimate: y = {mmse_y}')
plt.xlabel('y')
plt.ylabel('Normalized Density')
plt.title('Comparison of ML, MAP, and MMSE Estimates for Y given X=2')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'answer_verification.png'), dpi=300, bbox_inches='tight')

print(f"All visualizations saved in '{save_dir}'")
print(f"Key values: ML={ml_y}, MAP={map_y:.1f}, MMSE={mmse_y}") 