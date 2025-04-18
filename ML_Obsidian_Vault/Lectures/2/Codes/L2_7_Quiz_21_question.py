import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_21")
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

# Define functions for the new question - different from the original 
# These create a scenario where ML, MAP, and MMSE estimates are more clearly different

# Marginal distributions
def f_X(x):
    """Marginal PDF of X, uniform distribution"""
    return np.ones_like(x) * 0.25  # Uniform over [0, 4]

def f_Y(y):
    """Marginal PDF of Y, decreasing exponential"""
    return 0.5 * np.exp(-0.5 * y)

# Conditional distributions
def f_X_given_Y_2(y):
    """PDF of X given Y=2"""
    return 0.25 + 0.1 * np.sin(np.pi * y)

def f_Y_given_X_2(y):
    """PDF of Y given X=2, bimodal distribution"""
    # Bimodal distribution with peaks at y=1 and y=3
    return 0.15 * np.exp(-(y-1)**2/0.2) + 0.1 * np.exp(-(y-3)**2/0.3)

# Conditional expectations
def E_Y_given_X(x):
    """Conditional expectation of Y given X=x"""
    return 1.0 + 0.75 * x  # Linear relationship with positive slope

def E_X_given_Y(y):
    """Conditional expectation of X given Y=y"""
    return 2.0 + 0.5 * np.sin(np.pi * y / 2)  # Sinusoidal relationship

# Create grids for x and y values
x = np.linspace(0, 4, 200)
y = np.linspace(0, 4, 200)

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

# 1. Plot f_X|Y(2|y)
save_individual_plot(
    f_X_given_Y_2, y, f_X_given_Y_2(y),
    r"$y$", r"$f_{X|Y}(2|y)$", r"$f_{X|Y}(2|Y=y)$",
    "graph1_f_X_given_Y.png", ylim=(0, 0.5)
)

# 2. Plot f_Y(y)
save_individual_plot(
    f_Y, y, f_Y(y),
    r"$y$", r"$f_Y(y)$", r"$f_Y(y)$",
    "graph2_f_Y.png", ylim=(0, 0.5)
)

# 3. Plot E[Y|X=x]
save_individual_plot(
    E_Y_given_X, x, E_Y_given_X(x),
    r"$x$", r"$E[Y|X=x]$", r"$E[Y|X=x]$",
    "graph3_E_Y_given_X.png", ylim=(0, 4), grid=True
)

# 4. Plot f_Y|X(y|X=2)
save_individual_plot(
    f_Y_given_X_2, y, f_Y_given_X_2(y),
    r"$y$", r"$f_{Y|X}(y|X=2)$", r"$f_{Y|X}(y|X=2)$",
    "graph4_f_Y_given_X.png", ylim=(0, 0.2)
)

# 5. Plot f_X(x)
save_individual_plot(
    f_X, x, f_X(x),
    r"$x$", r"$f_X(x)$", r"$f_X(x)$",
    "graph5_f_X.png", ylim=(0, 0.5)
)

# 6. Plot E[X|Y=y]
save_individual_plot(
    E_X_given_Y, y, E_X_given_Y(y),
    r"$y$", r"$E[X|Y=y]$", r"$E[X|Y=y]$",
    "graph6_E_X_given_Y.png", ylim=(0, 4), grid=True
)

print(f"Individual graphs saved in '{save_dir}'")

# Calculate and print the key answers for reference
print("\nSolution Reference Values:")

# ML estimate - maximum value of f_Y|X(y|X=2)
y_values = np.linspace(0, 4, 1000)
likelihood_values = f_Y_given_X_2(y_values)
ml_estimate = y_values[np.argmax(likelihood_values)]
print(f"ML Estimate: {ml_estimate:.2f}")

# MAP estimate - maximum value of f_Y|X(y|X=2) * f_Y(y)
prior_values = f_Y(y_values)
posterior_values = likelihood_values * prior_values
map_estimate = y_values[np.argmax(posterior_values)]
print(f"MAP Estimate: {map_estimate:.2f}")

# MMSE estimate - E[Y|X=2]
mmse_estimate = E_Y_given_X(2)
print(f"MMSE Estimate: {mmse_estimate:.2f}") 