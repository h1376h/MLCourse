import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_10")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Understanding the Problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Model M₁: Linear regression with 3 parameters")
print("- Model M₂: Polynomial regression with 8 parameters")
print("- Both models fit to n = 50 data points")
print("- Maximum log-likelihoods:")
print("  - log p(D|M₁, θ̂₁) = -75")
print("  - log p(D|M₂, θ̂₂) = -65")
print("\nTask:")
print("1. Calculate the BIC value for each model")
print("2. Which model would be selected according to BIC?")
print("3. How does BIC penalize model complexity compared to AIC?")

# Step 2: Calculating BIC for Each Model
print_step_header(2, "Calculating BIC Values")

# Define the parameters
n = 50  # number of data points
k1 = 3  # number of parameters for Model 1
k2 = 8  # number of parameters for Model 2
log_like1 = -75  # maximum log-likelihood for Model 1
log_like2 = -65  # maximum log-likelihood for Model 2

# Calculate BIC for each model
# BIC = -2 * log-likelihood + k * log(n)
bic1 = -2 * log_like1 + k1 * np.log(n)
bic2 = -2 * log_like2 + k2 * np.log(n)

print("The Bayesian Information Criterion (BIC) is defined as:")
print("BIC = -2 * log-likelihood + k * log(n)")
print("Where:")
print("- log-likelihood is the maximum log-likelihood of the model")
print("- k is the number of parameters in the model")
print("- n is the number of data points")
print("\nFor Model M₁:")
print(f"BIC₁ = -2 * ({log_like1}) + {k1} * log({n})")
print(f"BIC₁ = {-2 * log_like1} + {k1} * {np.log(n):.4f}")
print(f"BIC₁ = {-2 * log_like1} + {k1 * np.log(n):.4f}")
print(f"BIC₁ = {bic1:.4f}")
print("\nFor Model M₂:")
print(f"BIC₂ = -2 * ({log_like2}) + {k2} * log({n})")
print(f"BIC₂ = {-2 * log_like2} + {k2} * {np.log(n):.4f}")
print(f"BIC₂ = {-2 * log_like2} + {k2 * np.log(n):.4f}")
print(f"BIC₂ = {bic2:.4f}")

# Calculate the BIC difference
delta_bic = bic2 - bic1
print("\nBIC difference (BIC₂ - BIC₁):")
print(f"Δ BIC = {bic2:.4f} - {bic1:.4f} = {delta_bic:.4f}")

# Visualize the BIC values
plt.figure(figsize=(10, 6))

# Stacked bar chart showing the components of BIC
models = ['Model 1\n(Linear, 3 params)', 'Model 2\n(Polynomial, 8 params)']
log_likelihood_terms = [-2 * log_like1, -2 * log_like2]
complexity_penalties = [k1 * np.log(n), k2 * np.log(n)]
bic_values = [bic1, bic2]

# Plot the total BIC values
bars = plt.bar(models, bic_values, color=['blue', 'red'], alpha=0.3, edgecolor='black')

# Plot stacked bars for the components
plt.bar(models, log_likelihood_terms, color=['lightblue', 'lightcoral'], 
        label='Goodness of fit: -2 * log-likelihood')
plt.bar(models, complexity_penalties, bottom=log_likelihood_terms, color=['darkblue', 'darkred'],
        label='Complexity penalty: k * log(n)')

# Add text annotations for the total BIC
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bic_values[i] + 2, 
             f'BIC = {bic_values[i]:.2f}', 
             ha='center', va='bottom', fontweight='bold')

# Add text annotations for the components
for i, (fit, penalty) in enumerate(zip(log_likelihood_terms, complexity_penalties)):
    plt.text(i, fit/2, f'-2*logL = {fit:.1f}', ha='center', va='center', color='white')
    plt.text(i, fit + penalty/2, f'Penalty = {penalty:.1f}', ha='center', va='center')

plt.ylabel('BIC Value', fontsize=12)
plt.title('Bayesian Information Criterion (BIC) Components by Model', fontsize=14)
plt.legend(loc='upper center')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bic_components.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Model Selection using BIC
print_step_header(3, "Model Selection using BIC")

selected_model = "M₁" if bic1 < bic2 else "M₂"
print(f"The model with the lower BIC value is selected. In this case, it's Model {selected_model}.")
print("\nInterpretation:")
if bic1 < bic2:
    print("Model M₁ (linear regression) has a lower BIC value, indicating it provides a better balance")
    print("between goodness of fit and model complexity for the given data.")
    print("\nEven though Model M₂ has a higher log-likelihood (better fit to the data),")
    print("the additional complexity penalty outweighs this advantage in the BIC calculation.")
else:
    print("Model M₂ (polynomial regression) has a lower BIC value, indicating it provides a better balance")
    print("between goodness of fit and model complexity for the given data.")
    print("\nThe improved fit (higher log-likelihood) of Model M₂ outweighs its")
    print("higher complexity penalty in the BIC calculation.")

# Create visualization for model selection
plt.figure(figsize=(10, 6))

# Simulated data points for visualization
np.random.seed(42)
x = np.linspace(0, 10, n)
y_true = 2 + 3*x - 0.2*x**2  # True underlying function
y_noisy = y_true + np.random.normal(0, 5, size=n)  # Add noise

# Fit models (simplified)
def linear_model(x, params):
    return params[0] + params[1]*x

def polynomial_model(x, params):
    return np.polyval(params[::-1], x)

# Visual representation of models (simplified)
x_plot = np.linspace(0, 10, 100)
y_linear = linear_model(x_plot, [2, 3])  # Simplified parameters
y_poly = polynomial_model(x_plot, [2, 3, -0.2, 0.05, -0.02, 0.001, -0.0002, 0.00001])  # Simplified
y_true_plot = 2 + 3*x_plot - 0.2*x_plot**2  # True function for plotting

plt.scatter(x, y_noisy, alpha=0.6, label='Data points (n=50)', color='gray')
plt.plot(x_plot, y_true_plot, 'k--', label='True function', linewidth=2)
plt.plot(x_plot, y_linear, 'b-', label='Model 1: Linear (3 parameters)', linewidth=2)
plt.plot(x_plot, y_poly, 'r-', label='Model 2: Polynomial (8 parameters)', linewidth=2)

plt.legend()

# Add BIC values to the legend
plt.legend(title="BIC Comparison", loc="upper left")

# Highlight the selected model
if bic1 < bic2:
    plt.plot(x_plot, y_linear, 'b-', linewidth=4, alpha=0.5)
    plt.annotate('Selected Model (by BIC)', xy=(7, y_linear[70]), xytext=(7, y_linear[70]+10),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5), fontsize=12)
else:
    plt.plot(x_plot, y_poly, 'r-', linewidth=4, alpha=0.5)
    plt.annotate('Selected Model (by BIC)', xy=(7, y_poly[70]), xytext=(7, y_poly[70]+10),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5), fontsize=12)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Model Selection using BIC', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "model_selection.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Comparing BIC and AIC Penalties
print_step_header(4, "Comparing BIC and AIC Penalties")

# Calculate AIC for comparison
aic1 = -2 * log_like1 + 2 * k1
aic2 = -2 * log_like2 + 2 * k2

print("The Akaike Information Criterion (AIC) is defined as:")
print("AIC = -2 * log-likelihood + 2 * k")
print("\nLet's compare how BIC and AIC penalize complexity:")
print("\nFor Model M₁ (k = 3, n = 50):")
print(f"AIC penalty = 2 * {k1} = {2 * k1}")
print(f"BIC penalty = {k1} * log({n}) = {k1} * {np.log(n):.4f} = {k1 * np.log(n):.4f}")
print(f"Ratio (BIC/AIC) = {(k1 * np.log(n)) / (2 * k1):.4f}")

print("\nFor Model M₂ (k = 8, n = 50):")
print(f"AIC penalty = 2 * {k2} = {2 * k2}")
print(f"BIC penalty = {k2} * log({n}) = {k2} * {np.log(n):.4f} = {k2 * np.log(n):.4f}")
print(f"Ratio (BIC/AIC) = {(k2 * np.log(n)) / (2 * k2):.4f}")

print("\nDifference in Model Selection:")
print(f"AIC for Model M₁ = {aic1:.4f}")
print(f"AIC for Model M₂ = {aic2:.4f}")
delta_aic = aic2 - aic1
print(f"Δ AIC (AIC₂ - AIC₁) = {delta_aic:.4f}")
aic_selected = "M₁" if aic1 < aic2 else "M₂"
print(f"Model selected by AIC: {aic_selected}")
print(f"Model selected by BIC: {selected_model}")

# Compare penalties across different sample sizes
plt.figure(figsize=(10, 6))

n_values = np.arange(10, 1000, 10)
bic_penalties = np.log(n_values)
aic_penalties = np.full_like(n_values, 2)

plt.plot(n_values, bic_penalties, 'b-', linewidth=2, label='BIC: log(n)')
plt.plot(n_values, aic_penalties, 'r-', linewidth=2, label='AIC: 2')
plt.axvline(x=n, color='green', linestyle='--', 
            label=f'Current n = {n}, log(n) = {np.log(n):.2f}')

# Add an annotation at the current sample size
plt.annotate(f'At n = {n}, BIC penalty = {np.log(n):.2f}\nAIC penalty = 2',
            xy=(n, np.log(n)), xytext=(n+50, np.log(n)+0.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# Add explanation of crossover point
crossover_n = np.exp(2)  # n where log(n) = 2
plt.axvline(x=crossover_n, color='gray', linestyle=':')
plt.annotate(f'Crossover point: n ≈ {crossover_n:.1f}',
            xy=(crossover_n, 2), xytext=(crossover_n+50, 2.3),
            arrowprops=dict(facecolor='gray', shrink=0.05), fontsize=10)

plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Penalty per Parameter', fontsize=12)
plt.title('BIC vs AIC: Penalty Factor per Parameter Across Sample Sizes', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bic_vs_aic_penalty.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Create a comparative visualization of model selection for different criteria
k_values = np.arange(1, 20)
n = 50  # sample size

# Calculate penalties
bic_penalties = k_values * np.log(n)
aic_penalties = 2 * k_values

# Set a fixed improvement in fit (log-likelihood) per parameter
# Assuming each additional parameter improves fit by 4 units
ll_improvement = 4 * k_values

# Calculate criteria values (lower is better)
base_ll = -75  # baseline log-likelihood (from model 1)
bic_values = -2 * (base_ll + ll_improvement) + bic_penalties
aic_values = -2 * (base_ll + ll_improvement) + aic_penalties

plt.figure(figsize=(10, 6))

plt.plot(k_values, bic_values, 'b-', linewidth=2, label='BIC')
plt.plot(k_values, aic_values, 'r-', linewidth=2, label='AIC')

# Find and mark the minima
bic_min_k = np.argmin(bic_values) + 1
aic_min_k = np.argmin(aic_values) + 1

plt.scatter(bic_min_k, bic_values[bic_min_k-1], s=100, color='blue', marker='o', zorder=5)
plt.scatter(aic_min_k, aic_values[aic_min_k-1], s=100, color='red', marker='o', zorder=5)

plt.annotate(f'BIC optimum: k = {bic_min_k}',
            xy=(bic_min_k, bic_values[bic_min_k-1]), 
            xytext=(bic_min_k+1, bic_values[bic_min_k-1]+10),
            arrowprops=dict(facecolor='blue', shrink=0.05), fontsize=10)

plt.annotate(f'AIC optimum: k = {aic_min_k}',
            xy=(aic_min_k, aic_values[aic_min_k-1]), 
            xytext=(aic_min_k+1, aic_values[aic_min_k-1]+10),
            arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)

# Mark the two models from our question
plt.scatter(k1, bic1, s=150, color='blue', marker='*', zorder=5, 
            label=f'Model 1 (k={k1}): BIC={bic1:.1f}')
plt.scatter(k2, bic2, s=150, color='red', marker='*', zorder=5,
            label=f'Model 2 (k={k2}): BIC={bic2:.1f}')

plt.xlabel('Number of Parameters (k)', fontsize=12)
plt.ylabel('Information Criterion Value', fontsize=12)
plt.title('Model Selection: BIC vs AIC with Fixed Improvement per Parameter', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bic_vs_aic_selection.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("\nKey Insights about BIC vs AIC:")
print("1. BIC applies a stronger penalty for model complexity than AIC when n > 7.4")
print("2. As sample size increases, BIC increasingly favors simpler models compared to AIC")
print("3. BIC penalty is proportional to log(n), while AIC uses a fixed penalty of 2")
print("4. BIC is consistent: it will select the true model as n → ∞ (if the true model is in the set)")
print("5. AIC is efficient: it minimizes prediction error, even if the true model isn't in the set")
print("6. The difference in penalties leads to different model selection behaviors:")
print("   - BIC tends to select simpler models (may underfit if n is large)")
print("   - AIC tends to select more complex models (may overfit if n is small)")
print(f"7. In our case, with n = {n} and log(n) = {np.log(n):.4f}, the BIC penalty is {np.log(n)/2:.4f} times stronger than AIC") 