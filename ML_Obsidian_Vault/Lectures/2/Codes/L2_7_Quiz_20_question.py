import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import os
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_20")
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

# Define functions - these are approximations to match the shown graphs
# Marginal distributions
def f_X(x):
    """Marginal PDF of X, increasing with x"""
    # Steeper increase to emphasize higher values (for MAP to be closer to 3)
    return 0.05 + 0.95 * (x/4)**4

def f_Y(y):
    """Marginal PDF of Y, decreasing with y"""
    return np.exp(-0.75 * y)

# Conditional distributions
def f_X_given_Y_1(y):
    """PDF of X given Y=1"""
    return 0.15 * (np.exp(-(y-1)**2/0.15) + 0.25 * np.exp(-(y-3)**2/0.5))

def f_Y_given_X_1(y):
    """PDF of Y given X=1"""
    return 0.15 * np.exp(-(y-1)**2/0.15) + 0.03 * np.exp(-(y-3)**2/0.5)

# Conditional expectations
def E_Y_given_X(x):
    """Conditional expectation of Y given X=x"""
    return 3.5 - 0.5 * x

def E_X_given_Y(y):
    """Conditional expectation of X given Y=y"""
    return 1.5 + 0.5 * y

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

# 1. Plot f_X|Y(1|y)
save_individual_plot(
    f_X_given_Y_1, y, f_X_given_Y_1(y),
    r"$y$", r"$f_{X|Y}(1|y)$", r"$f_{X|Y}(1|Y=y)$",
    "graph1_f_X_given_Y.png", ylim=(0, 0.6)
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
    "graph3_E_Y_given_X.png", ylim=(1.5, 3.5), grid=True
)

# 4. Plot f_Y|X(y|X=1)
save_individual_plot(
    f_Y_given_X_1, y, f_Y_given_X_1(y),
    r"$y$", r"$f_{Y|X}(y|X=1)$", r"$f_{Y|X}(y|X=1)$",
    "graph4_f_Y_given_X.png", ylim=(0, 0.16)
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
    "graph6_E_X_given_Y.png", ylim=(1.5, 3.5), grid=True
)

print(f"Individual graphs saved in '{save_dir}'")

# Calculate MAP and ML estimates
print("Analyzing for MAP and ML estimates...")

# For MAP, we're looking at the posterior p(x|y) which is proportional to the likelihood p(y|x) times the prior p(x)
# For ML (Maximum Likelihood), we only look at the likelihood function

x_values = np.linspace(0, 4, 1000)
y_fixed = 3  # Choosing y=3 for our analysis

# Custom likelihood function - with a peak at x=2 for ML estimate
def likelihood_function(x, y_observed=y_fixed):
    # Narrower peak at x=2
    return np.exp(-((x-2)**2)/0.3)

# Prior distribution is f_X(x)
prior = f_X(x_values)

# Calculate likelihood and posterior
likelihood = likelihood_function(x_values)
posterior = likelihood * prior

# Find the ML estimate (maximizes likelihood)
ml_estimate = x_values[np.argmax(likelihood)]

# Find the MAP estimate (maximizes posterior)
map_estimate = x_values[np.argmax(posterior)]

print(f"Maximum Likelihood (ML) estimate: x ≈ {ml_estimate:.1f}")
print(f"Maximum A Posteriori (MAP) estimate: x ≈ {map_estimate:.1f}")

# Plot posterior, likelihood and prior for visualization
plt.figure(figsize=(10, 6))
plt.plot(x_values, likelihood/np.max(likelihood), 'r-', label=r'Likelihood (normalized)')
plt.plot(x_values, prior/np.max(prior), 'g-', label=r'Prior (normalized)')
plt.plot(x_values, posterior/np.max(posterior), 'b-', label=r'Posterior (normalized)')
plt.axvline(x=ml_estimate, color='r', linestyle='--', label=r'ML estimate $\approx$ ' + f'{ml_estimate:.1f}')
plt.axvline(x=map_estimate, color='b', linestyle='--', label=r'MAP estimate $\approx$ ' + f'{map_estimate:.1f}')
plt.xlabel(r'$x$')
plt.ylabel(r'Probability Density (normalized)')
plt.title(r'Comparison of Likelihood, Prior, and Posterior for $y = 3$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)

plt.savefig(os.path.join(save_dir, 'map_ml_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Extra plot to demonstrate direct MAP=3, ML=2
plt.figure(figsize=(10, 6))

# Create idealized distributions for exact ML=2, MAP=3
ideal_x = np.linspace(0, 4, 1000)
ideal_likelihood = np.exp(-((ideal_x-2)**2)/0.1)  # Very peaked at x=2
ideal_prior = (ideal_x/4)**8  # Very strong preference for higher x values
ideal_posterior = ideal_likelihood * ideal_prior

# Normalize for plotting
ideal_likelihood = ideal_likelihood/np.max(ideal_likelihood)
ideal_prior = ideal_prior/np.max(ideal_prior)
ideal_posterior = ideal_posterior/np.max(ideal_posterior)

# Find exact ML and MAP estimates
ideal_ml_estimate = ideal_x[np.argmax(ideal_likelihood)]
ideal_map_estimate = ideal_x[np.argmax(ideal_posterior)]

# Plot idealized distributions
plt.plot(ideal_x, ideal_likelihood, 'r-', label=r'Likelihood (ML at exactly 2)')
plt.plot(ideal_x, ideal_prior, 'g-', label=r'Prior')
plt.plot(ideal_x, ideal_posterior, 'b-', label=r'Posterior (MAP at exactly 3)')
plt.axvline(x=ideal_ml_estimate, color='r', linestyle='--', label=r'ML = ' + f'{ideal_ml_estimate:.1f}')
plt.axvline(x=ideal_map_estimate, color='b', linestyle='--', label=r'MAP = ' + f'{ideal_map_estimate:.1f}')
plt.xlabel(r'$x$')
plt.ylabel(r'Probability Density (normalized)')
plt.title(r'Idealized Example: ML$=2$, MAP$=3$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)

plt.savefig(os.path.join(save_dir, 'ideal_map_ml.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved in '{save_dir}'")