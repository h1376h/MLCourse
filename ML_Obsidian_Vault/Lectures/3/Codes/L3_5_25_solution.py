import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_5_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Question 25: Adaptive Learning Rate Strategies for LMS Algorithm")
print("\n## Part 1: Implementing an Error-Dependent Adaptive Learning Rate\n")

print("### Modified LMS Update Rule with Adaptive Learning Rate")
print("The standard LMS update rule is:")
print("$$\\mathbf{w}^{(t+1)} = \\mathbf{w}^{(t)} + \\eta(y^{(i)} - \\mathbf{w}^{(t)T}\\mathbf{x}^{(i)})\\mathbf{x}^{(i)}$$")
print("\nA modified LMS update rule where the learning rate depends on recent prediction errors could be:")
print("$$\\mathbf{w}^{(t+1)} = \\mathbf{w}^{(t)} + \\alpha_t(y^{(i)} - \\mathbf{w}^{(t)T}\\mathbf{x}^{(i)})\\mathbf{x}^{(i)}$$")
print("\nWhere $\\alpha_t$ is the adaptive learning rate at time step $t$")

print("\nProposed formula for adaptive learning rate:")
print("$$\\alpha_t = \\alpha_0 \\cdot (1 + \\gamma|e_t|)$$")
print("\nWhere:")
print("- $\\alpha_0$ is the base learning rate")
print("- $\\gamma$ is a scaling factor")
print("- $e_t = y^{(i)} - \\mathbf{w}^{(t)T}\\mathbf{x}^{(i)}$ is the prediction error at time $t$")
print("- $|e_t|$ is the absolute error")

print("\nThis formula increases the learning rate when errors are large and keeps it close to $\\alpha_0$ when errors are small.")
print("The scaling factor $\\gamma$ controls how strongly the error influences the learning rate.")

print("\n## Part 2: Tracing Algorithm on Given Data Points\n")

# Define the data points
x1 = np.array([1, 2])
y1 = 5
x2 = np.array([1, 3])
y2 = 8
x3 = np.array([1, 4])
y3 = 9

# Initial weights and learning rate parameters
w0 = np.array([0, 1])
alpha0 = 0.1
gamma = 0.5  # Scaling factor for error-dependent learning rate

# Create a detailed function to implement the adaptive LMS update with verbose calculations
def adaptive_lms_update_detailed(w, x, y, alpha0, gamma):
    # Step 1: Calculate prediction
    prediction = np.dot(w, x)
    print(f"   - w[0] * x[0] = {w[0]} * {x[0]} = {w[0] * x[0]}")
    print(f"   - w[1] * x[1] = {w[1]} * {x[1]} = {w[1] * x[1]}")
    print(f"   - Prediction: w^T * x = {prediction}")
    
    # Step 2: Calculate error
    error = y - prediction
    print(f"   - Error: e = y - prediction = {y} - {prediction} = {error}")
    
    # Step 3: Calculate adaptive learning rate
    abs_error = abs(error)
    gamma_times_error = gamma * abs_error
    one_plus_gamma_error = 1 + gamma_times_error
    alpha_t = alpha0 * one_plus_gamma_error
    print(f"   - |error| = |{error}| = {abs_error}")
    print(f"   - γ * |error| = {gamma} * {abs_error} = {gamma_times_error}")
    print(f"   - 1 + γ * |error| = 1 + {gamma_times_error} = {one_plus_gamma_error}")
    print(f"   - α_t = α_0 * (1 + γ * |error|) = {alpha0} * {one_plus_gamma_error} = {alpha_t}")
    
    # Step 4: Update weights
    update_term = alpha_t * error * x
    w_new = w + update_term
    print(f"   - Calculating new weights:")
    print(f"     - Original weights: w = {w}")
    print(f"     - Update term: α_t * error * x = {alpha_t} * {error} * {x}")
    print(f"     - Which equals: [{', '.join([str(val) for val in update_term])}]")
    
    for i in range(len(w)):
        print(f"     - w_new[{i}] = w[{i}] + update_term[{i}] = {w[i]} + {update_term[i]} = {w_new[i]}")
    
    print(f"   - New weights: w_new = {w_new}")
    
    return w_new, error, alpha_t, prediction

# Create a DataFrame to store results
columns = ["Step", "w", "x", "y", "Prediction", "Error", "α_t", "Updated w"]
results = []

# Initial step
w = w0.copy()
results.append([0, w.tolist(), "-", "-", "-", "-", alpha0, w.tolist()])

print("Initial weights: $\\mathbf{w}^{(0)} = [0, 1]$")
print("Initial learning rate: $\\alpha_0 = 0.1$")
print("Scaling factor: $\\gamma = 0.5$")
print("\nTracing through the updates for each data point:")

print("\n### Step 1: First data point")
print(f"$\\mathbf{{x}}^{{(1)}} = [1, 2], y^{{(1)}} = 5$")

print("\nDetailed calculations:")
w_new, error, alpha_t, prediction = adaptive_lms_update_detailed(w, x1, y1, alpha0, gamma)
results.append([1, w.tolist(), x1.tolist(), y1, prediction, error, alpha_t, w_new.tolist()])

print(f"\nPrediction: $\\mathbf{{w}}^{{(0)T}}\\mathbf{{x}}^{{(1)}} = [0, 1] \\cdot [1, 2] = 0 \\cdot 1 + 1 \\cdot 2 = {prediction}$")
print(f"Error: $e_1 = y^{{(1)}} - \\mathbf{{w}}^{{(0)T}}\\mathbf{{x}}^{{(1)}} = {y1} - {prediction} = {error}$")
print(f"Adaptive learning rate: $\\alpha_1 = \\alpha_0 \\cdot (1 + \\gamma|e_1|) = {alpha0} \\cdot (1 + {gamma}\\cdot{abs(error)}) = {alpha_t}$")
print(f"Weight update: $\\mathbf{{w}}^{{(1)}} = \\mathbf{{w}}^{{(0)}} + \\alpha_1 \\cdot e_1 \\cdot \\mathbf{{x}}^{{(1)}} = [0, 1] + {alpha_t} \\cdot {error} \\cdot [1, 2] = {w_new}$")

w = w_new.copy()

print("\n### Step 2: Second data point")
print(f"$\\mathbf{{x}}^{{(2)}} = [1, 3], y^{{(2)}} = 8$")

print("\nDetailed calculations:")
w_new, error, alpha_t, prediction = adaptive_lms_update_detailed(w, x2, y2, alpha0, gamma)
results.append([2, w.tolist(), x2.tolist(), y2, prediction, error, alpha_t, w_new.tolist()])

print(f"\nPrediction: $\\mathbf{{w}}^{{(1)T}}\\mathbf{{x}}^{{(2)}} = {w} \\cdot [1, 3] = {prediction}$")
print(f"Error: $e_2 = y^{{(2)}} - \\mathbf{{w}}^{{(1)T}}\\mathbf{{x}}^{{(2)}} = {y2} - {prediction} = {error}$")
print(f"Adaptive learning rate: $\\alpha_2 = \\alpha_0 \\cdot (1 + \\gamma|e_2|) = {alpha0} \\cdot (1 + {gamma}\\cdot{abs(error)}) = {alpha_t}$")
print(f"Weight update: $\\mathbf{{w}}^{{(2)}} = \\mathbf{{w}}^{{(1)}} + \\alpha_2 \\cdot e_2 \\cdot \\mathbf{{x}}^{{(2)}} = {w} + {alpha_t} \\cdot {error} \\cdot [1, 3] = {w_new}$")

w = w_new.copy()

print("\n### Step 3: Third data point")
print(f"$\\mathbf{{x}}^{{(3)}} = [1, 4], y^{{(3)}} = 9$")

print("\nDetailed calculations:")
w_new, error, alpha_t, prediction = adaptive_lms_update_detailed(w, x3, y3, alpha0, gamma)
results.append([3, w.tolist(), x3.tolist(), y3, prediction, error, alpha_t, w_new.tolist()])

print(f"\nPrediction: $\\mathbf{{w}}^{{(2)T}}\\mathbf{{x}}^{{(3)}} = {w} \\cdot [1, 4] = {prediction}$")
print(f"Error: $e_3 = y^{{(3)}} - \\mathbf{{w}}^{{(2)T}}\\mathbf{{x}}^{{(3)}} = {y3} - {prediction} = {error}$")
print(f"Adaptive learning rate: $\\alpha_3 = \\alpha_0 \\cdot (1 + \\gamma|e_3|) = {alpha0} \\cdot (1 + {gamma}\\cdot{abs(error)}) = {alpha_t}$")
print(f"Weight update: $\\mathbf{{w}}^{{(3)}} = \\mathbf{{w}}^{{(2)}} + \\alpha_3 \\cdot e_3 \\cdot \\mathbf{{x}}^{{(3)}} = {w} + {alpha_t} \\cdot {error} \\cdot [1, 4] = {w_new}$")

w = w_new.copy()

# Create a summary table
df = pd.DataFrame(results, columns=columns)
print("\n### Summary of iterations:")
print(df.to_string(index=False))

# Create visualization of the adaptive learning rate process
# First visualization: Learning rates and errors
plt.figure(figsize=(12, 5))

# Plot 1: Learning rates over iterations
plt.subplot(1, 2, 1)
iterations = np.arange(len(results))
learning_rates = [row[6] for row in results]
plt.plot(iterations, learning_rates, 'o-', color='blue', markersize=8)
plt.title('Adaptive Learning Rate Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate ($\\alpha_t$)')
plt.grid(True)

# Plot 2: Errors over iterations
plt.subplot(1, 2, 2)
errors = [row[5] if row[5] != '-' else 0 for row in results]
errors = [0 if isinstance(e, str) else e for e in errors]
plt.plot(iterations[1:], errors[1:], 'o-', color='red', markersize=8)
plt.title('Prediction Error Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Error ($e_t$)')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaptive_lms_rates_errors.png'), dpi=300)
plt.close()

# Second visualization: Weight vector trajectory
plt.figure(figsize=(8, 6))
w_trajectories = [row[1] if row[1] != '-' else [0, 0] for row in results]
w0_values = [w[0] for w in w_trajectories]
w1_values = [w[1] for w in w_trajectories]

# Plot weight trajectory more clearly
plt.scatter(w0_values, w1_values, c=range(len(w0_values)), cmap='viridis', s=100, zorder=5)
plt.plot(w0_values, w1_values, '-', color='gray', alpha=0.7, zorder=1)

# Add arrows to show direction of weight updates
for i in range(len(w0_values)-1):
    plt.annotate("",
                xy=(w0_values[i+1], w1_values[i+1]),
                xytext=(w0_values[i], w1_values[i]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# Add labels to points
for i, (w0, w1) in enumerate(zip(w0_values, w1_values)):
    plt.annotate(f"$\\mathbf{{w}}^{{({i})}}$", (w0, w1), textcoords="offset points", 
                 xytext=(0, 10), ha='center')

plt.title('Weight Vector Trajectory During Training')
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaptive_lms_weight_trajectory.png'), dpi=300)
plt.close()

# New visualization: Data points and final decision boundary
plt.figure(figsize=(8, 6))

# Create data points from the problem
data_x = np.array([
    [1, 2],  # x1
    [1, 3],  # x2
    [1, 4]   # x3
])
data_y = np.array([5, 8, 9])  # y1, y2, y3

# Plot the data points
plt.scatter(data_x[:, 1], data_y, color='blue', s=100, label='Training Data')

# Get the final weights
final_w = w_trajectories[-1]

# Create a range of x values for plotting the decision line
x_range = np.linspace(1.5, 4.5, 100)
y_pred = final_w[0] + final_w[1] * x_range

# Plot the decision line
plt.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Final Model: $y = {final_w[0]:.3f} + {final_w[1]:.3f}x$')

# Plot the initial model
initial_w = w_trajectories[0]
y_initial = initial_w[0] + initial_w[1] * x_range
plt.plot(x_range, y_initial, 'g--', linewidth=2, label=f'Initial Model: $y = {initial_w[0]:.3f} + {initial_w[1]:.3f}x$')

plt.title('LMS Model Fitting with Adaptive Learning Rate')
plt.xlabel('Feature Value ($x_1$)')
plt.ylabel('Target Value ($y$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaptive_lms_model_fit.png'), dpi=300)
plt.close()

print("\nVisualizations saved to:")
print("- 'adaptive_lms_rates_errors.png' (learning rates and errors)")
print("- 'adaptive_lms_weight_trajectory.png' (weight vector trajectory)")
print("- 'adaptive_lms_model_fit.png' (data points and fitted model)")

print("\n## Part 3: Annealing Learning Rate\n")

# Implement annealing learning rate with more detailed calculations
print("The annealing learning rate formula is:")
print("$$\\alpha_t = \\frac{\\alpha_0}{1 + \\beta \\cdot t}$$")
print("\nWhere:")
print("- $\\alpha_0$ is the initial learning rate")
print("- $\\beta$ is the decay parameter")
print("- $t$ is the time step or iteration number")

# Parameters
alpha0 = 0.2
beta = 0.1

# Calculate learning rates for the first 5 time steps
annealing_rates = []
print("\nDetailed calculations for first 5 time steps:")

for t in range(5):
    print(f"\nTime step $t = {t}$:")
    print(f"- Formula: $\\alpha_{{{t}}} = \\frac{{\\alpha_0}}{{1 + \\beta \\cdot t}}$")
    print(f"- Numerator: $\\alpha_0 = {alpha0}$")
    beta_t = beta * t
    print(f"- Denominator part 1: $\\beta \\cdot t = {beta} \\cdot {t} = {beta_t}$")
    denominator = 1 + beta_t
    print(f"- Denominator full: $1 + \\beta \\cdot t = 1 + {beta_t} = {denominator}$")
    alpha_t = alpha0 / denominator
    print(f"- Final calculation: $\\alpha_{{{t}}} = \\frac{{{alpha0}}}{{{denominator}}} = {alpha_t}$")
    
    annealing_rates.append((t, alpha_t))

print("\nSummary of learning rates for first 5 time steps:")
for t, rate in annealing_rates:
    print(f"$\\alpha_{{{t}}} = {rate:.6f}$")

# Visualize annealing learning rate
plt.figure(figsize=(10, 6))
time_steps = np.arange(0, 50)
learning_rates = [alpha0 / (1 + beta * t) for t in time_steps]

plt.plot(time_steps, learning_rates, 'o-', markersize=5)
plt.title('Annealing Learning Rate Schedule')
plt.xlabel('Time Step ($t$)')
plt.ylabel('Learning Rate ($\\alpha_t$)')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'annealing_learning_rate.png'), dpi=300)
plt.close()

print("\nVisualization of the annealing learning rate saved to 'annealing_learning_rate.png'")

print("\n## Part 4: Constant vs. Annealing Learning Rate for Non-Stationary Data\n")

print("For non-stationary data (where the underlying distribution changes over time):")
print("- A constant learning rate maintains the ability to adapt to recent changes")
print("- An annealing learning rate decreases over time, eventually becoming too small to adapt to new changes")

# Create comparative visualization
plt.figure(figsize=(12, 6))

# Generate some non-stationary data for illustration (a moving target)
np.random.seed(42)
time_steps = np.arange(100)
target_values = 5 * np.sin(0.1 * time_steps) + np.cumsum(0.05 * np.random.randn(100))

# Simulate constant and annealing learning rate tracking
constant_alpha = 0.1
annealing_alpha0 = 0.5
annealing_beta = 0.05

constant_predictions = np.zeros_like(target_values)
annealing_predictions = np.zeros_like(target_values)

constant_predictions[0] = target_values[0]
annealing_predictions[0] = target_values[0]

print("\nDetailed tracking calculations (first 5 steps):")
for t in range(1, min(6, len(time_steps))):
    print(f"\nTime step $t = {t}$:")
    
    # Constant learning rate update
    error_constant = target_values[t-1] - constant_predictions[t-1]
    update_constant = constant_alpha * error_constant
    constant_predictions[t] = constant_predictions[t-1] + update_constant
    
    print(f"Constant learning rate calculations:")
    print(f"- Previous prediction: $\\hat{{y}}_{{{t-1}}} = {constant_predictions[t-1]:.6f}$")
    print(f"- Target value: $y_{{{t-1}}} = {target_values[t-1]:.6f}$")
    print(f"- Error: $e_{{{t-1}}} = y_{{{t-1}}} - \\hat{{y}}_{{{t-1}}} = {target_values[t-1]:.6f} - {constant_predictions[t-1]:.6f} = {error_constant:.6f}$")
    print(f"- Update: $\\alpha \\cdot e_{{{t-1}}} = {constant_alpha} \\cdot {error_constant:.6f} = {update_constant:.6f}$")
    print(f"- New prediction: $\\hat{{y}}_{{{t}}} = \\hat{{y}}_{{{t-1}}} + \\alpha \\cdot e_{{{t-1}}} = {constant_predictions[t-1]:.6f} + {update_constant:.6f} = {constant_predictions[t]:.6f}$")
    
    # Annealing learning rate update
    annealing_alpha_t = annealing_alpha0 / (1 + annealing_beta * t)
    error_annealing = target_values[t-1] - annealing_predictions[t-1]
    update_annealing = annealing_alpha_t * error_annealing
    annealing_predictions[t] = annealing_predictions[t-1] + update_annealing
    
    print(f"\nAnnealing learning rate calculations:")
    print(f"- Current learning rate: $\\alpha_{{{t}}} = \\frac{{\\alpha_0}}{{1 + \\beta \\cdot t}} = \\frac{{{annealing_alpha0}}}{{1 + {annealing_beta} \\cdot {t}}} = {annealing_alpha_t:.6f}$")
    print(f"- Previous prediction: $\\hat{{y}}_{{{t-1}}} = {annealing_predictions[t-1]:.6f}$")
    print(f"- Target value: $y_{{{t-1}}} = {target_values[t-1]:.6f}$")
    print(f"- Error: $e_{{{t-1}}} = y_{{{t-1}}} - \\hat{{y}}_{{{t-1}}} = {target_values[t-1]:.6f} - {annealing_predictions[t-1]:.6f} = {error_annealing:.6f}$")
    print(f"- Update: $\\alpha_{{{t}}} \\cdot e_{{{t-1}}} = {annealing_alpha_t:.6f} \\cdot {error_annealing:.6f} = {update_annealing:.6f}$")
    print(f"- New prediction: $\\hat{{y}}_{{{t}}} = \\hat{{y}}_{{{t-1}}} + \\alpha_{{{t}}} \\cdot e_{{{t-1}}} = {annealing_predictions[t-1]:.6f} + {update_annealing:.6f} = {annealing_predictions[t]:.6f}$")

# Continue with the rest of the calculations (without detailed printouts)
for t in range(6, len(time_steps)):
    # Constant learning rate update
    error_constant = target_values[t-1] - constant_predictions[t-1]
    constant_predictions[t] = constant_predictions[t-1] + constant_alpha * error_constant
    
    # Annealing learning rate update
    annealing_alpha_t = annealing_alpha0 / (1 + annealing_beta * t)
    error_annealing = target_values[t-1] - annealing_predictions[t-1]
    annealing_predictions[t] = annealing_predictions[t-1] + annealing_alpha_t * error_annealing

plt.plot(time_steps, target_values, 'k-', label='Non-stationary Target', linewidth=2)
plt.plot(time_steps, constant_predictions, 'b-', label=f'Constant $\\alpha={constant_alpha}$', linewidth=2)
plt.plot(time_steps, annealing_predictions, 'r-', label=f'Annealing $\\alpha_0={annealing_alpha0}, \\beta={annealing_beta}$', linewidth=2)

plt.title('Constant vs. Annealing Learning Rate for Non-Stationary Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'constant_vs_annealing.png'), dpi=300)
plt.close()

print("\nVisualization comparing constant and annealing learning rates saved to 'constant_vs_annealing.png'")

print("\n## Part 5: Momentum-Based LMS Update Rule\n")

print("A momentum-based LMS update rule incorporates information from previous updates:")
print("\nThe momentum-based LMS update consists of two steps:")
print("1. Compute the velocity vector:")
print("$$\\mathbf{v}^{(t+1)} = \\mu\\mathbf{v}^{(t)} + \\alpha(y^{(i)} - \\mathbf{w}^{(t)T}\\mathbf{x}^{(i)})\\mathbf{x}^{(i)}$$")
print("2. Update the weights using the velocity:")
print("$$\\mathbf{w}^{(t+1)} = \\mathbf{w}^{(t)} + \\mathbf{v}^{(t+1)}$$")

print("\nWhere:")
print("- $\\mu$ is the momentum coefficient (typically between 0 and 1)")
print("- $\\mathbf{v}^{(t)}$ is the velocity vector at time $t$")
print("- $\\alpha$ is the learning rate")

# Visualize the effect of momentum with detailed calculations
print("\nDetailed example of momentum vs. standard LMS updates:")

# Simulation parameters
np.random.seed(42)
n_iterations = 50

# Generate some data points with a linear relationship with noise
X = np.random.rand(100, 2)
X = np.column_stack([np.ones(100), X])  # Add bias term
true_w = np.array([1, 2, 3])
y = X.dot(true_w) + 0.5 * np.random.randn(100)

# Initialize weights
w_standard = np.zeros(3)
w_momentum = np.zeros(3)
v_momentum = np.zeros(3)

# Learning parameters
alpha = 0.01
mu = 0.9  # Momentum coefficient

print(f"\nSimulation parameters:")
print(f"- True weights: $\\mathbf{{w}}_{{true}} = {true_w}$")
print(f"- Learning rate: $\\alpha = {alpha}$")
print(f"- Momentum coefficient: $\\mu = {mu}$")
print(f"- Initial weights: $\\mathbf{{w}}^{{(0)}} = {w_standard}$")
print(f"- Initial velocity: $\\mathbf{{v}}^{{(0)}} = {v_momentum}$")

# Track weights and errors
w_standard_history = [w_standard.copy()]
w_momentum_history = [w_momentum.copy()]
error_standard = []
error_momentum = []

# Run both algorithms with detailed calculations for a few iterations
print("\nDetailed calculations for first 3 iterations:")
for i in range(3):
    # Random data point
    idx = np.random.randint(0, len(X))
    x_i, y_i = X[idx], y[idx]
    
    print(f"\nIteration {i+1}:")
    print(f"Selected data point: $\\mathbf{{x}}^{{({i+1})}} = {x_i}, y^{{({i+1})}} = {y_i:.6f}$")
    
    # Standard LMS
    print("\nStandard LMS update:")
    pred_standard = w_standard.dot(x_i)
    print(f"- Prediction: $\\mathbf{{w}}^{{({i})T}}\\mathbf{{x}}^{{({i+1})}} = {w_standard} \\cdot {x_i} = {pred_standard:.6f}$")
    
    error_std = y_i - pred_standard
    print(f"- Error: $e = y^{{({i+1})}} - \\mathbf{{w}}^{{({i})T}}\\mathbf{{x}}^{{({i+1})}} = {y_i:.6f} - {pred_standard:.6f} = {error_std:.6f}$")
    
    update = alpha * error_std * x_i
    print(f"- Update: $\\alpha \\cdot e \\cdot \\mathbf{{x}}^{{({i+1})}} = {alpha} \\cdot {error_std:.6f} \\cdot {x_i} = {update}$")
    
    w_standard = w_standard + update
    print(f"- New weights: $\\mathbf{{w}}^{{({i+1})}} = \\mathbf{{w}}^{{({i})}} + \\alpha \\cdot e \\cdot \\mathbf{{x}}^{{({i+1})}} = {w_standard_history[-1]} + {update} = {w_standard}$")
    
    error_standard.append(error_std**2)
    w_standard_history.append(w_standard.copy())
    
    # Momentum LMS
    print("\nMomentum LMS update:")
    pred_momentum = w_momentum.dot(x_i)
    print(f"- Prediction: $\\mathbf{{w}}^{{({i})T}}\\mathbf{{x}}^{{({i+1})}} = {w_momentum} \\cdot {x_i} = {pred_momentum:.6f}$")
    
    error_mom = y_i - pred_momentum
    print(f"- Error: $e = y^{{({i+1})}} - \\mathbf{{w}}^{{({i})T}}\\mathbf{{x}}^{{({i+1})}} = {y_i:.6f} - {pred_momentum:.6f} = {error_mom:.6f}$")
    
    grad_term = alpha * error_mom * x_i
    print(f"- Gradient term: $\\alpha \\cdot e \\cdot \\mathbf{{x}}^{{({i+1})}} = {alpha} \\cdot {error_mom:.6f} \\cdot {x_i} = {grad_term}$")
    
    momentum_term = mu * v_momentum
    print(f"- Momentum term: $\\mu \\cdot \\mathbf{{v}}^{{({i})}} = {mu} \\cdot {v_momentum} = {momentum_term}$")
    
    v_momentum = momentum_term + grad_term
    print(f"- New velocity: $\\mathbf{{v}}^{{({i+1})}} = \\mu \\cdot \\mathbf{{v}}^{{({i})}} + \\alpha \\cdot e \\cdot \\mathbf{{x}}^{{({i+1})}} = {momentum_term} + {grad_term} = {v_momentum}$")
    
    w_momentum = w_momentum + v_momentum
    print(f"- New weights: $\\mathbf{{w}}^{{({i+1})}} = \\mathbf{{w}}^{{({i})}} + \\mathbf{{v}}^{{({i+1})}} = {w_momentum_history[-1]} + {v_momentum} = {w_momentum}$")
    
    error_momentum.append(error_mom**2)
    w_momentum_history.append(w_momentum.copy())

# Continue running the rest of the iterations without detailed printouts
print("\nContinuing with remaining iterations...")
for i in range(3, n_iterations):
    # Random data point
    idx = np.random.randint(0, len(X))
    x_i, y_i = X[idx], y[idx]
    
    # Standard LMS
    pred_standard = w_standard.dot(x_i)
    error_std = y_i - pred_standard
    error_standard.append(error_std**2)
    w_standard = w_standard + alpha * error_std * x_i
    w_standard_history.append(w_standard.copy())
    
    # Momentum LMS
    pred_momentum = w_momentum.dot(x_i)
    error_mom = y_i - pred_momentum
    error_momentum.append(error_mom**2)
    v_momentum = mu * v_momentum + alpha * error_mom * x_i
    w_momentum = w_momentum + v_momentum
    w_momentum_history.append(w_momentum.copy())

print(f"\nFinal weights after {n_iterations} iterations:")
print(f"- Standard LMS: $\\mathbf{{w}}^{{({n_iterations})}} = {w_standard}$")
print(f"- Momentum LMS: $\\mathbf{{w}}^{{({n_iterations})}} = {w_momentum}$")
print(f"- True weights: $\\mathbf{{w}}_{{true}} = {true_w}$")

print("\nDistance to true weights:")
std_distance = np.linalg.norm(w_standard - true_w)
mom_distance = np.linalg.norm(w_momentum - true_w)
print(f"- Standard LMS: $\\|\\mathbf{{w}}^{{({n_iterations})}} - \\mathbf{{w}}_{{true}}\\| = {std_distance:.6f}$")
print(f"- Momentum LMS: $\\|\\mathbf{{w}}^{{({n_iterations})}} - \\mathbf{{w}}_{{true}}\\| = {mom_distance:.6f}$")

# Create comparison plots
plt.figure(figsize=(10, 8))

# Plot the errors
plt.subplot(2, 1, 1)
plt.plot(error_standard, 'b-', label='Standard LMS')
plt.plot(error_momentum, 'r-', label='Momentum LMS')
plt.title('Squared Error per Iteration')
plt.ylabel('Squared Error')
plt.legend()
plt.grid(True)

# Plot the distance to the true weights
plt.subplot(2, 1, 2)
w_standard_distance = [np.linalg.norm(w - true_w) for w in w_standard_history]
w_momentum_distance = [np.linalg.norm(w - true_w) for w in w_momentum_history]

plt.plot(w_standard_distance, 'b-', label='Standard LMS')
plt.plot(w_momentum_distance, 'r-', label='Momentum LMS')
plt.title('Distance to True Weights')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'momentum_lms.png'), dpi=300)
plt.close()

print("\nVisualization of momentum-based LMS compared to standard LMS saved to 'momentum_lms.png'")

# Create a 3D visualization of the momentum effect
from mpl_toolkits.mplot3d import Axes3D

# Simplify to 2D weights (exclude bias) for visualization
w_standard_history_2d = np.array(w_standard_history)[:, 1:]
w_momentum_history_2d = np.array(w_momentum_history)[:, 1:]
true_w_2d = true_w[1:]

# Create 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Add trajectories
ax.plot(w_standard_history_2d[:, 0], w_standard_history_2d[:, 1], 
        np.arange(len(w_standard_history_2d)), 'b-', label='Standard LMS')
ax.plot(w_momentum_history_2d[:, 0], w_momentum_history_2d[:, 1], 
        np.arange(len(w_momentum_history_2d)), 'r-', label='Momentum LMS')

# Mark the true weights with a star
ax.scatter([true_w_2d[0]], [true_w_2d[1]], [0], marker='*', s=200, c='green', label='True Weights')

# Set labels and title
ax.set_xlabel('Weight 1 ($w_1$)')
ax.set_ylabel('Weight 2 ($w_2$)')
ax.set_zlabel('Iteration')
ax.set_title('3D Visualization of Weight Trajectories with and without Momentum')

ax.legend()
plt.savefig(os.path.join(save_dir, 'momentum_lms_3d.png'), dpi=300)
plt.close()

print("\n3D visualization of weight trajectories saved to 'momentum_lms_3d.png'")

# Final summary
print("\n## Summary of Adaptive Learning Rate Strategies for LMS\n")

print("In this analysis, we explored several adaptive learning rate strategies for the LMS algorithm:")
print("1. Error-dependent adaptive learning rate: $\\alpha_t = \\alpha_0 \\cdot (1 + \\gamma|e_t|)$")
print("2. Annealing learning rate: $\\alpha_t = \\frac{\\alpha_0}{1 + \\beta \\cdot t}$")
print("3. Momentum-based update rule:")
print("   - $\\mathbf{v}^{(t+1)} = \\mu\\mathbf{v}^{(t)} + \\alpha(y^{(i)} - \\mathbf{w}^{(t)T}\\mathbf{x}^{(i)})\\mathbf{x}^{(i)}$")
print("   - $\\mathbf{w}^{(t+1)} = \\mathbf{w}^{(t)} + \\mathbf{v}^{(t+1)}$")

print("\nEach strategy has specific advantages:")
print("- Error-dependent rates adapt faster when errors are large")
print("- Annealing rates help convergence for stationary problems")
print("- Constant rates maintain adaptability for non-stationary problems")
print("- Momentum helps navigate complex error surfaces more efficiently")

print("\nThe choice of strategy depends on the specific characteristics of the problem at hand.") 