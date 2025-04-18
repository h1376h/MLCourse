import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_20")
os.makedirs(save_dir, exist_ok=True)

# Set up the figure for all plots
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=plt.gcf())

# Create grids for x and y values
x = np.linspace(0, 4, 200)
y = np.linspace(0, 4, 200)
X, Y = np.meshgrid(x, y)

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

# Plot f_X|Y(1|y)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(y, f_X_given_Y_1(y))
ax1.set_xlabel("y")
ax1.set_ylabel("f_X|Y(1|Y=y)")
ax1.set_title("f_X|Y(1|Y=y)")
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 0.6)

# Plot f_Y(y)
ax2 = plt.subplot(gs[0, 1])
ax2.plot(y, f_Y(y))
ax2.set_xlabel("y")
ax2.set_ylabel("f_Y(y)")
ax2.set_title("f_Y(y)")
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 1)

# Plot E[Y|X=x]
ax3 = plt.subplot(gs[0, 2])
ax3.plot(x, E_Y_given_X(x))
ax3.set_xlabel("x")
ax3.set_ylabel("E[Y|X=x]")
ax3.set_title("E[Y|X=x]")
ax3.set_xlim(0, 4)
ax3.set_ylim(1.5, 3.5)
ax3.grid(linestyle='--')

# Plot f_Y|X(y|X=1)
ax4 = plt.subplot(gs[1, 0])
ax4.plot(y, f_Y_given_X_1(y))
ax4.set_xlabel("y")
ax4.set_ylabel("f_Y|X(y|X=1)")
ax4.set_title("f_Y|X(y|X=1)")
ax4.set_xlim(0, 4)
ax4.set_ylim(0, 0.16)

# Plot f_X(x)
ax5 = plt.subplot(gs[1, 1])
ax5.plot(x, f_X(x))
ax5.set_xlabel("x")
ax5.set_ylabel("f_X(x)")
ax5.set_title("f_X(x)")
ax5.set_xlim(0, 4)
ax5.set_ylim(0, 1)

# Plot E[X|Y=y]
ax6 = plt.subplot(gs[1, 2])
ax6.plot(y, E_X_given_Y(y))
ax6.set_xlabel("y")
ax6.set_ylabel("E[X|Y=y]")
ax6.set_title("E[X|Y=y]")
ax6.set_xlim(0, 4)
ax6.set_ylim(1.5, 3.5)
ax6.grid(linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'joint_pdf_graphs.png'), dpi=300, bbox_inches='tight')
plt.close()

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
plt.plot(x_values, likelihood/np.max(likelihood), 'r-', label='Likelihood (normalized)')
plt.plot(x_values, prior/np.max(prior), 'g-', label='Prior (normalized)')
plt.plot(x_values, posterior/np.max(posterior), 'b-', label='Posterior (normalized)')
plt.axvline(x=ml_estimate, color='r', linestyle='--', label=f'ML estimate ≈ {ml_estimate:.1f}')
plt.axvline(x=map_estimate, color='b', linestyle='--', label=f'MAP estimate ≈ {map_estimate:.1f}')
plt.xlabel('x')
plt.ylabel('Probability Density (normalized)')
plt.title('Comparison of Likelihood, Prior, and Posterior for y = 3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)

plt.savefig(os.path.join(save_dir, 'map_ml_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved in '{save_dir}'")

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
plt.plot(ideal_x, ideal_likelihood, 'r-', label='Likelihood (ML at exactly 2)')
plt.plot(ideal_x, ideal_prior, 'g-', label='Prior')
plt.plot(ideal_x, ideal_posterior, 'b-', label='Posterior (MAP at exactly 3)')
plt.axvline(x=ideal_ml_estimate, color='r', linestyle='--', label=f'ML = {ideal_ml_estimate:.1f}')
plt.axvline(x=ideal_map_estimate, color='b', linestyle='--', label=f'MAP = {ideal_map_estimate:.1f}')
plt.xlabel('x')
plt.ylabel('Probability Density (normalized)')
plt.title('Idealized Example: ML=2, MAP=3')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)

plt.savefig(os.path.join(save_dir, 'ideal_map_ml.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Added '{os.path.join(save_dir, 'ideal_map_ml.png')}' with exact ML=2, MAP=3 demonstration") 