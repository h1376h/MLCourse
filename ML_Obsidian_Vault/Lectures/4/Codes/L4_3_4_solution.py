import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots and enable LaTeX support
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
plt.rcParams['font.family'] = 'serif'

# Step 1: Define the dataset
X = np.array([
    [1, 1],  # Class 0
    [2, 1],  # Class 0
    [1, 2],  # Class 0
    [3, 3],  # Class 1
    [4, 3],  # Class 1
    [3, 4]   # Class 1
])

y = np.array([0, 0, 0, 1, 1, 1])

# Step 2: Plot the dataset
plt.figure(figsize=(10, 8))
plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='o', s=100, label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='x', s=100, label='Class 1')

# Add point labels
for i, (x1, x2) in enumerate(X):
    plt.annotate(f'({x1},{x2})', (x1, x2), xytext=(10, 5), textcoords='offset points')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Points for Binary Classification')
plt.grid(True)
plt.legend()

# Save the figure
plt.savefig(os.path.join(save_dir, 'data_points.png'), dpi=300, bbox_inches='tight')

# Step 3: Explain the form of the decision boundary for a linear probabilistic classifier
# For a linear probabilistic classifier, the decision boundary has the form: w0 + w1*x1 + w2*x2 = 0

print("Step 3: Form of the decision boundary for a linear probabilistic classifier")
print("------------------------------------------------------------------------")
print("For a linear probabilistic classifier, the decision boundary has the form:")
print("w0 + w1*x1 + w2*x2 = 0")
print("\nThis is equivalent to:")
print("x2 = -(w0/w2) - (w1/w2)*x1")
print("\nThe probability of class 1 is given by the logistic function:")
print("P(y=1|x) = 1 / (1 + exp(-(w0 + w1*x1 + w2*x2)))")
print("The decision boundary occurs where P(y=1|x) = 0.5, which is exactly where w0 + w1*x1 + w2*x2 = 0")

# Step 4: Draw the decision boundary with the given parameters
w0, w1, w2 = -5, 1, 1

# Create a function to plot the decision boundary
def plot_decision_boundary(w0, w1, w2, X, y, title, filename):
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='o', s=100, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='x', s=100, label='Class 1')
    
    # Add point labels
    for i, (x1, x2) in enumerate(X):
        plt.annotate(f'({x1},{x2})', (x1, x2), xytext=(10, 5), textcoords='offset points')
    
    # Plot the decision boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0/w2) - (w1/w2)*x1
    x1_min, x1_max = 0, 5
    x1_range = np.array([x1_min, x1_max])
    x2_boundary = -(w0/w2) - (w1/w2) * x1_range
    
    plt.plot(x1_range, x2_boundary, 'k-', label=f'Decision Boundary: {w2}$x_2$ = -{w0} - {w1}$x_1$')
    
    # Highlight the point (2,2) if requested
    highlight_point = (2, 2)
    plt.scatter(highlight_point[0], highlight_point[1], color='green', marker='*', s=200, label='Point (2,2)')
    
    # Calculate and annotate the value of w0 + w1*x1 + w2*x2 at (2,2)
    log_odds_value = w0 + w1*highlight_point[0] + w2*highlight_point[1]
    probability = 1 / (1 + np.exp(-log_odds_value))
    
    plt.annotate(f'Log-odds: {log_odds_value}\nP(y=1|x) = {probability:.3f}', 
                 (highlight_point[0], highlight_point[1]), 
                 xytext=(30, 30), 
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Add grid lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Create a meshgrid to visualize the decision regions
    x1_min, x1_max = 0, 5
    x2_min, x2_max = 0, 5
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    
    # Calculate the classification for each point in the grid
    Z = w0 + w1*xx + w2*yy
    Z = 1 / (1 + np.exp(-Z))  # Apply logistic function to get probabilities
    Z = (Z > 0.5).astype(int)  # Classify based on probability threshold
    
    # Plot the decision regions with transparency
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(['blue', 'red']))
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    
    return log_odds_value, probability

# Plot the decision boundary
log_odds_value, probability = plot_decision_boundary(
    w0, w1, w2, X, y, 
    f'Decision Boundary: $w_0 + w_1x_1 + w_2x_2 = 0$ with $w_0={w0}$, $w_1={w1}$, $w_2={w2}$',
    'decision_boundary.png'
)

# Step 5: Calculate the log-odds ratio for the point (2,2)
point = np.array([2, 2])
print("\nStep 5: Calculate the log-odds ratio for the point (2,2)")
print("----------------------------------------------------------")
print(f"For the point (x1,x2) = ({point[0]},{point[1]}):")
print(f"log-odds ratio = w0 + w1*x1 + w2*x2 = {w0} + {w1}*{point[0]} + {w2}*{point[1]} = {log_odds_value}")
print(f"This corresponds to a probability P(y=1|x) = 1 / (1 + exp(-({log_odds_value}))) = {probability:.5f}")

# Explanation of the results
print("\nExplanation of the Results")
print("-------------------------")
print("1. The dataset consists of 6 points with coordinates and labels as given in the problem statement.")
print("2. The decision boundary for a linear probabilistic classifier has the form: w0 + w1*x1 + w2*x2 = 0.")
print(f"3. With parameters w0 = {w0}, w1 = {w1}, and w2 = {w2}, the decision boundary is:")
print(f"   {w0} + {w1}*x1 + {w2}*x2 = 0, which can be rewritten as x2 = {-(w0/w2):.1f} - {(w1/w2):.1f}*x1")
print(f"4. The log-odds ratio for the point (2,2) is {log_odds_value}, which corresponds to a probability of {probability:.5f}")

# Step 6: Demonstrate effect of changing the parameters
# Create a function to visualize different decision boundaries
def visualize_different_parameters():
    plt.figure(figsize=(12, 10))
    
    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', marker='o', s=100, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', marker='x', s=100, label='Class 1')
    
    # Add point labels
    for i, (x1, x2) in enumerate(X):
        plt.annotate(f'({x1},{x2})', (x1, x2), xytext=(10, 5), textcoords='offset points')
    
    # Original parameters
    w0_orig, w1_orig, w2_orig = -5, 1, 1
    
    # Define different parameter sets
    param_sets = [
        {'w0': -5, 'w1': 1, 'w2': 1, 'color': 'black', 'style': '-', 'label': 'Original: $w_0=-5, w_1=1, w_2=1$'},
        {'w0': -6, 'w1': 1, 'w2': 1, 'color': 'green', 'style': '--', 'label': 'Changed $w_0$: $w_0=-6, w_1=1, w_2=1$'},
        {'w0': -5, 'w1': 2, 'w2': 1, 'color': 'purple', 'style': '-.', 'label': 'Changed $w_1$: $w_0=-5, w_1=2, w_2=1$'},
        {'w0': -5, 'w1': 1, 'w2': 2, 'color': 'brown', 'style': ':', 'label': 'Changed $w_2$: $w_0=-5, w_1=1, w_2=2$'},
    ]
    
    # Plot multiple decision boundaries
    x1_min, x1_max = 0, 5
    x1_range = np.array([x1_min, x1_max])
    
    for params in param_sets:
        w0, w1, w2 = params['w0'], params['w1'], params['w2']
        x2_boundary = -(w0/w2) - (w1/w2) * x1_range
        plt.plot(x1_range, x2_boundary, color=params['color'], linestyle=params['style'], 
                 linewidth=2, label=params['label'])
    
    # Highlight the point (2,2)
    plt.scatter(2, 2, color='green', marker='*', s=200, label='Point (2,2)')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Effect of Different Parameters on Decision Boundary')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'different_parameters.png'), dpi=300, bbox_inches='tight')

# Visualize different parameters
visualize_different_parameters()

# Step 7: Visualize the logistic function and decision probabilities
def visualize_logistic_function():
    plt.figure(figsize=(10, 6))
    
    # Plot the logistic function
    z = np.linspace(-10, 10, 1000)
    p = 1 / (1 + np.exp(-z))
    
    plt.plot(z, p, 'b-', linewidth=2, label='Logistic Function: $P(y=1|x) = \\frac{1}{1+e^{-z}}$')
    
    # Highlight the point where z = log-odds value for (2,2)
    plt.scatter(log_odds_value, probability, color='green', marker='*', s=200, 
                label=f'Point (2,2): z = {log_odds_value:.2f}, P = {probability:.3f}')
    
    # Highlight the decision threshold
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold: P = 0.5')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Log-Odds = 0')
    
    plt.xlabel('z = $w_0 + w_1x_1 + w_2x_2$ (Log-Odds)')
    plt.ylabel('P(y=1|x)')
    plt.title('Logistic Function and Classification of Point (2,2)')
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'logistic_function.png'), dpi=300, bbox_inches='tight')

visualize_logistic_function()

print("\nAll visualizations have been saved to:", save_dir)
print("Final output: The point (2,2) has a log-odds ratio of", log_odds_value, "and a probability of", probability) 