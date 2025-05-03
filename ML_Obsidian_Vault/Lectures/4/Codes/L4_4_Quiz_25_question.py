import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set plot style to closely match the target image
plt.style.use('default')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-'
})

def create_lda_visualization():
    """Create data and compute LDA for visualization."""
    np.random.seed(42)
    
    # Generate data
    # Class 0 (blue dots) - create a diagonal line of points
    n_samples_class0 = 50
    X_class0 = np.zeros((n_samples_class0, 2))
    
    # Create diagonal pattern for blue points
    for i in range(n_samples_class0):
        # Start at bottom left, going up diagonally
        t = i / (n_samples_class0 - 1)
        X_class0[i, 0] = -2 + t * 2  # x from -2 to 0
        X_class0[i, 1] = -1 + t * 1  # y from -1 to 0
        # Add random variation but limit how far points can stray
        X_class0[i, 0] += np.clip(np.random.normal(0, 0.15), -0.3, 0.3)
        X_class0[i, 1] += np.clip(np.random.normal(0, 0.15), -0.3, 0.3)
    
    # Class 1 (red triangles) - more scattered in top right
    n_samples_class1 = 40
    X_class1 = np.zeros((n_samples_class1, 2))
    
    # Create scattered pattern for red points (ensuring they stay in the top right quadrant)
    for i in range(n_samples_class1):
        # Generate points further away from the blue cluster
        X_class1[i, 0] = np.random.uniform(0.5, 4)  # x from 0.5 to 4 (minimum 0.5 to avoid overlap)
        X_class1[i, 1] = np.random.uniform(0, 2)    # y from 0 to 2
    
    # Combine data
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(n_samples_class0), np.ones(n_samples_class1)])
    
    # Compute class means
    mean0 = np.mean(X[y == 0], axis=0)
    mean1 = np.mean(X[y == 1], axis=0)
    
    # Compute within-class scatter matrix
    S_w = np.zeros((2, 2))
    for i in range(len(X)):
        if y[i] == 0:
            diff = X[i] - mean0
        else:
            diff = X[i] - mean1
        S_w += np.outer(diff, diff)
    
    # Compute Fisher's linear discriminant direction
    try:
        w = np.linalg.solve(S_w, mean1 - mean0)
        w = w / np.linalg.norm(w)  # Normalize
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        w = mean1 - mean0
        w = w / np.linalg.norm(w)
    
    # Calculate plot bounds based on data
    x_padding = (np.max(X[:, 0]) - np.min(X[:, 0])) * 0.2
    y_padding = (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.2
    x_min = np.min(X[:, 0]) - x_padding
    x_max = np.max(X[:, 0]) + x_padding
    y_min = np.min(X[:, 1]) - y_padding
    y_max = np.max(X[:, 1]) + y_padding
    
    # Compute decision boundary (perpendicular to w, passing through midpoint)
    midpoint = (mean0 + mean1) / 2
    slope = -w[0] / w[1]
    xx_line = np.linspace(x_min, x_max, 100)
    yy_line = midpoint[1] + slope * (xx_line - midpoint[0])
    
    # Create grid for decision regions
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Compute decision function values
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            # Project point onto w
            proj = np.dot(point - midpoint, w)
            Z[i, j] = 1 if proj > 0 else 0
    
    # Force-check data points to make sure there are no misclassifications
    # Calculate linear classification for each data point
    for i in range(len(X)):
        point_class = 1 if np.dot(X[i] - midpoint, w) > 0 else 0
        # If point is misclassified according to the LDA boundary
        if point_class != y[i]:
            # Move the point further into its correct class region
            if y[i] == 0:  # Blue class
                # Move blue point more to the left/bottom
                X[i] = X[i] - 0.5 * w
            else:  # Red class
                # Move red point more to the right/top
                X[i] = X[i] + 0.5 * w
    
    # Create projection line (in direction perpendicular to decision boundary)
    # Position it below the data points (calculate based on data bounds)
    proj_line_y = y_min + 0.2 * (y_max - y_min)  # Position relative to data bounds
    # Calculate the direction vector of the projection line
    # This is the direction of w itself (perpendicular to decision boundary)
    proj_direction = w / np.linalg.norm(w)
    # Create the projection line endpoints
    # Calculate line length based on data bounds
    line_length = 1.2 * (x_max - x_min)
    # Place line horizontally centered in the plot
    proj_center = np.array([(x_min + x_max) / 2, proj_line_y])
    proj_start = proj_center - line_length/2 * proj_direction
    proj_end = proj_center + line_length/2 * proj_direction
    
    return X, y, mean0, mean1, w, xx, yy, Z, xx_line, yy_line, proj_start, proj_end, midpoint

def plot_visualization(X, y, mean0, mean1, w, xx, yy, Z, xx_line, yy_line, proj_start, proj_end, midpoint):
    """Plot the LDA visualization with proper projections."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    # Plot decision regions
    plt.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.RdBu_r, levels=1)
    
    # Plot data points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='.', c='blue', s=30, alpha=0.9)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='^', c='red', s=30, alpha=0.5)
    
    # Plot decision boundary
    plt.plot(xx_line, yy_line, 'k-', linewidth=1)
    
    # Add "Decision boundary" label with improved placement and visibility
    midpoint_idx = len(xx_line) // 2
    # Calculate a good position for the label that's away from data points
    label_x = xx_line[midpoint_idx + 15]
    label_y = yy_line[midpoint_idx + 15]
    
    # Create a white background for the text for better readability
    plt.annotate('Decision boundary', 
               xy=(xx_line[midpoint_idx], yy_line[midpoint_idx]),
               xytext=(label_x, label_y),
               fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
               arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"))
    
    # Plot projection line
    plt.plot([proj_start[0], proj_end[0]], [proj_start[1], proj_end[1]], 'k-', linewidth=1)
    
    # Choose a subset of points from each class
    # Use stratified sampling to ensure good coverage
    class0_indices = np.where(y == 0)[0]
    class1_indices = np.where(y == 1)[0]
    
    # Select points at regular intervals for better distribution
    step0 = max(1, len(class0_indices) // 6)
    step1 = max(1, len(class1_indices) // 6)
    
    selected_indices0 = class0_indices[::step0][:6]  # Take up to 6 points
    selected_indices1 = class1_indices[::step1][:6]  # Take up to 6 points
    
    indices = np.concatenate([selected_indices0, selected_indices1])
    
    # Unit vector along projection line
    proj_vec = (proj_end - proj_start) / np.linalg.norm(proj_end - proj_start)
    # Unit vector perpendicular to projection line
    perp_vec = np.array([-proj_vec[1], proj_vec[0]])  # rotate 90 degrees
    
    # Draw green perpendicular projection lines and project points
    projected_points = []
    for idx in indices:
        point = X[idx]
        
        # Vector from proj_start to the point
        v = point - proj_start
        # Projection distance along the projection line
        proj_dist = np.dot(v, proj_vec)
        # Projected point on the line
        proj_point = proj_start + proj_dist * proj_vec
        projected_points.append((proj_point, y[idx]))
        
        # Draw green perpendicular projection line
        plt.plot([point[0], proj_point[0]], [point[1], proj_point[1]], 'g-', linewidth=1, alpha=0.5)
    
    # Plot the projected points on the line with different style
    for proj_point, label in projected_points:
        if label == 0:
            # Blue class projections
            plt.plot(proj_point[0], proj_point[1], 'o', color='blue', markersize=4, alpha=0.9)
        else:
            # Red class projections
            plt.plot(proj_point[0], proj_point[1], 'o', color='red', markersize=4, alpha=0.9)
    
    # Set axis limits based on data
    x_padding = (np.max(X[:, 0]) - np.min(X[:, 0])) * 0.2
    y_padding = (np.max(X[:, 1]) - np.min(X[:, 1])) * 0.2
    plt.xlim(np.min(X[:, 0]) - x_padding, np.max(X[:, 0]) + x_padding)
    plt.ylim(np.min(X[:, 1]) - y_padding, np.max(X[:, 1]) + y_padding)
    
    # Add labels
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lda_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional check - print any potentially misclassified points
    misclassified = []
    for i in range(len(X)):
        point_class = 1 if np.dot(X[i] - midpoint, w) > 0 else 0
        if point_class != y[i]:
            misclassified.append(i)
    
    if misclassified:
        print(f"Warning: {len(misclassified)} points might be misclassified")

# Generate and save visualization
print("Generating final LDA visualization for quiz question...")
X, y, mean0, mean1, w, xx, yy, Z, xx_line, yy_line, proj_start, proj_end, midpoint = create_lda_visualization()
plot_visualization(X, y, mean0, mean1, w, xx, yy, Z, xx_line, yy_line, proj_start, proj_end, midpoint)

print(f"Visualization saved to: {save_dir}")
print(f"LDA direction vector: {w}")
print(f"Class means: {mean0}, {mean1}") 