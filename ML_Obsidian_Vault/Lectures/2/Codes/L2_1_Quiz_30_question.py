import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_1_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Set general plot style
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Create a bivariate normal distribution with:
# - Positive correlation (0.6)
# - Different variances (more variance in x than y)
mu = [0, 0]  # Mean vector at the origin
rho = 0.6    # Positive correlation coefficient
sigma_x = 2  # Higher variance for x
sigma_y = 1  # Lower variance for y

# Calculate covariance matrix
cov = [[sigma_x**2, rho*sigma_x*sigma_y], 
       [rho*sigma_x*sigma_y, sigma_y**2]]

# Create grid of points for contour plot
x = np.linspace(-5, 5, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Generate the bivariate normal distribution
rv = stats.multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# Create the contour plot
plt.figure(figsize=(10, 8))

# Add contour lines and filled contours
contour_levels = np.linspace(0, 0.15, 10)
plt.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.7)
CS = plt.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.8)

# Label the innermost contour with 0.15
plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f', manual=[(0, 0)])

# Add grid, axis labels and title
plt.grid(linestyle='--', alpha=0.6)
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.title('Bivariate Normal Distribution')
plt.colorbar(label='Probability Density')

# Add annotation to highlight correlation
plt.annotate('Elliptical contours indicate\ncorrelation between variables', 
             xy=(2, 1.5), xytext=(2.5, 2), fontsize=9,
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Add a mark at (0,0) to show the distribution center
plt.scatter(0, 0, color='red', s=50, marker='x', label='Mean Vector (μ)')
plt.legend(loc='upper right')

# Keep axis equal to prevent distortion
plt.axis('equal')
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(save_dir, 'contour_question.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Contour plot saved to {os.path.join(save_dir, 'contour_question.png')}") 