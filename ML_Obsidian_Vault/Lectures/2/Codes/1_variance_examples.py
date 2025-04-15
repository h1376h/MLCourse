import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

print("\n=== VARIANCE EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Variance of a Discrete Random Variable
print("Example 1: Variance of a Discrete Random Variable")
x = np.array([1, 2, 3, 4])
probs = np.array([0.2, 0.3, 0.4, 0.1])
print(f"Values (x): {x}")
print(f"Probabilities P(X=x): {probs}")

# Calculate expected value
print("\nStep 1: Calculate the expected value E[X]")
expected_value = 0
for i in range(len(x)):
    term = x[i] * probs[i]
    expected_value += term
    print(f"  {x[i]} × {probs[i]:.1f} = {term:.1f}")

print(f"Sum = {expected_value}")
print(f"Therefore, E[X] = {expected_value}")

# Method 1: Using the definition
print("\nMethod 1: Calculate variance using the definition Var(X) = E[(X - E[X])²]")
variance_method1 = 0
for i in range(len(x)):
    deviation = x[i] - expected_value
    squared_deviation = deviation**2
    term = squared_deviation * probs[i]
    variance_method1 += term
    print(f"  ({x[i]} - {expected_value:.1f})² × {probs[i]:.1f} = {deviation:.1f}² × {probs[i]:.1f} = {squared_deviation:.2f} × {probs[i]:.1f} = {term:.3f}")

print(f"Sum = {variance_method1:.3f}")

# Method 2: Using the computational formula
print("\nMethod 2: Calculate variance using the computational formula Var(X) = E[X²] - (E[X])²")
expected_square = 0
for i in range(len(x)):
    squared_term = x[i]**2 * probs[i]
    expected_square += squared_term
    print(f"  {x[i]}² × {probs[i]:.1f} = {x[i]**2} × {probs[i]:.1f} = {squared_term:.1f}")

print(f"E[X²] = {expected_square:.2f}")
variance_method2 = expected_square - expected_value**2
print(f"Var(X) = E[X²] - (E[X])² = {expected_square:.2f} - ({expected_value:.1f})² = {expected_square:.2f} - {expected_value**2:.2f} = {variance_method2:.3f}")
print(f"Therefore, Var(X) = {variance_method2:.3f} = {variance_method2}")

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Value of X', fontsize=12)
plt.ylabel('Probability P(X=x)', fontsize=12)
plt.title('Discrete Random Variable Distribution', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.5)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(x[i], prob + 0.02, f'{prob:.1f}', ha='center', fontsize=10)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = {expected_value:.1f}')

# Add standard deviation lines
std_dev = np.sqrt(variance_method2)
plt.axvline(x=expected_value - std_dev, color='green', linestyle=':', label=f'μ ± σ (σ = {std_dev:.2f})')
plt.axvline(x=expected_value + std_dev, color='green', linestyle=':')
plt.legend()

# Add annotation for the variance calculation
calculation_1 = f'$E[X] = {expected_value:.1f}$'
calculation_2 = f'$E[X^2] = {expected_square:.2f}$'
calculation_3 = f'$Var(X) = E[X^2] - (E[X])^2 = {expected_square:.2f} - ({expected_value:.1f})^2 = {variance_method2:.2f}$'

plt.annotate(calculation_1 + '\n' + calculation_2 + '\n' + calculation_3, 
            xy=(2.5, 0.45),
            xytext=(2.5, 0.45),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'discrete_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Variance of Dice Rolls
print("\n\nExample 2: Variance of Dice Rolls")
x = np.arange(1, 7)  # Possible outcomes (1 to 6)
probs = np.ones(6) / 6  # Equal probability for each outcome
print(f"Values (x): {x}")
print(f"Probabilities P(X=x): {[f'{p:.3f}' for p in probs]}")

# Calculate expected value
print("\nStep 1: Calculate the expected value E[X]")
expected_value = 0
for i in range(len(x)):
    term = x[i] * probs[i]
    expected_value += term
    print(f"  {x[i]} × {probs[i]:.3f} = {term:.3f}")

print(f"Sum = {expected_value} = 21/6")
print(f"Therefore, E[X] = {expected_value}")

# Calculate E[X²]
print("\nStep 2: Calculate E[X²]")
expected_square = 0
for i in range(len(x)):
    term = (x[i]**2) * probs[i]
    expected_square += term
    print(f"  {x[i]}² × {probs[i]:.3f} = {x[i]**2} × {probs[i]:.3f} = {term:.3f}")

print(f"Sum = {expected_square:.3f} = 91/6")
print(f"Therefore, E[X²] = {expected_square:.3f}")

# Calculate Variance
print("\nStep 3: Calculate Var(X) = E[X²] - (E[X])²")
variance = expected_square - expected_value**2
print(f"Var(X) = {expected_square:.3f} - ({expected_value:.1f})² = {expected_square:.3f} - {expected_value**2:.3f} = {variance:.3f}")
print(f"Therefore, Var(X) = {variance:.3f}")
print(f"Standard deviation = √Var(X) = √{variance:.3f} = {np.sqrt(variance):.3f}")

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Dice Face', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Dice Roll Distribution with Variance', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.2)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    plt.text(i + 1, prob + 0.01, f'{prob:.3f}', ha='center', fontsize=9)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = {expected_value:.1f}')

# Add standard deviation lines
plt.axvline(x=expected_value - np.sqrt(variance), color='green', linestyle=':', label=f'μ ± σ (σ = {np.sqrt(variance):.2f})')
plt.axvline(x=expected_value + np.sqrt(variance), color='green', linestyle=':')
plt.legend()

# Add annotations for the calculations
plt.annotate(f'$E[X] = \\frac{{21}}{{6}} = {expected_value:.1f}$\n'
             f'$E[X^2] = \\frac{{91}}{{6}} = {expected_square:.2f}$\n'
             f'$Var(X) = E[X^2] - (E[X])^2 = {expected_square:.2f} - {expected_value:.1f}^2 = {variance:.2f}$', 
             xy=(3.5, 0.17),
             xytext=(3.5, 0.17),
             fontsize=9,
             ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'dice_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Variance of a Binomial Distribution
print("\n\nExample 3: Variance of a Binomial Distribution")
n = 10  # Number of trials
p = 0.7  # Probability of success
x = np.arange(0, n+1)  # Possible outcomes (0 to 10)
probs = stats.binom.pmf(x, n, p)
print(f"Number of trials (n): {n}")
print(f"Probability of success (p): {p}")

# Calculate expected value
print("\nStep 1: For a binomial distribution, E[X] = n × p")
expected_value = n * p
print(f"E[X] = {n} × {p} = {expected_value}")

# Calculate Variance
print("\nStep 2: For a binomial distribution, Var(X) = n × p × (1-p)")
variance = n * p * (1-p)
print(f"Var(X) = {n} × {p} × (1-{p}) = {n} × {p} × {1-p} = {variance}")
print(f"Therefore, Var(X) = {variance}")
print(f"Standard deviation = √Var(X) = √{variance} = {np.sqrt(variance):.2f}")

# Print the probability mass function
print("\nProbability mass function values:")
for i, prob in enumerate(probs):
    if prob > 0.01:
        print(f"  P(X={i}) = {prob:.4f}")

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(x, probs, color='skyblue', alpha=0.7)
plt.grid(True, alpha=0.3)
plt.xlabel('Number of Successful Free Throws', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('Binomial Distribution: n=10, p=0.7', fontsize=14)
plt.xticks(x)
plt.ylim(0, 0.3)

# Add probability values on top of each bar
for i, prob in enumerate(probs):
    if prob > 0.02:  # Only add text for larger bars
        plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', fontsize=8)

# Add expectation line
plt.axvline(x=expected_value, color='red', linestyle='--', label=f'E[X] = np = {expected_value:.1f}')

# Add standard deviation lines
plt.axvline(x=expected_value - np.sqrt(variance), color='green', linestyle=':', label=f'μ ± σ (σ = {np.sqrt(variance):.2f})')
plt.axvline(x=expected_value + np.sqrt(variance), color='green', linestyle=':')
plt.legend()

# Add annotation for the variance formula
plt.annotate(f'$Var(X) = n \\times p \\times (1-p) = 10 \\times 0.7 \\times 0.3 = 2.1$', 
            xy=(5, 0.28),
            xytext=(5, 0.28),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'binomial_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 4: Portfolio Diversification Effect on Variance
print("\n\nExample 4: Portfolio Diversification Effect on Variance")
# Parameters
variance_A = 100
variance_B = 100
correlation = 0.2
weight_A = 0.4
weight_B = 0.6
std_A = np.sqrt(variance_A)
std_B = np.sqrt(variance_B)
covariance = correlation * std_A * std_B

print(f"Variance of Asset A: {variance_A}")
print(f"Variance of Asset B: {variance_B}")
print(f"Correlation between assets: {correlation}")
print(f"Weight of Asset A in portfolio: {weight_A}")
print(f"Weight of Asset B in portfolio: {weight_B}")

# Calculate portfolio variance
print("\nStep-by-step calculation:")
print(f"Step 1: Calculate the covariance between assets")
print(f"Cov(A,B) = ρ × σA × σB = {correlation} × {std_A} × {std_B} = {covariance}")

print("\nStep 2: Calculate the portfolio variance")
term1 = weight_A**2 * variance_A
term2 = weight_B**2 * variance_B
term3 = 2 * weight_A * weight_B * covariance
portfolio_variance = term1 + term2 + term3

print(f"Var(Portfolio) = w_A² × Var(A) + w_B² × Var(B) + 2 × w_A × w_B × Cov(A,B)")
print(f"Var(Portfolio) = {weight_A}² × {variance_A} + {weight_B}² × {variance_B} + 2 × {weight_A} × {weight_B} × {covariance}")
print(f"Var(Portfolio) = {term1} + {term2} + {term3} = {portfolio_variance}")

print(f"\nTherefore, the portfolio variance is {portfolio_variance}")
print(f"This is less than the individual asset variances ({variance_A}), demonstrating the benefit of diversification.")

# Generate portfolio variance for different correlation values
correlations = np.linspace(-1, 1, 100)
portfolio_variances = []

for corr in correlations:
    cov = corr * std_A * std_B
    port_var = weight_A**2 * variance_A + weight_B**2 * variance_B + 2 * weight_A * weight_B * cov
    portfolio_variances.append(port_var)

# Plot the variance curve
plt.figure(figsize=(10, 6))
plt.plot(correlations, portfolio_variances, 'b-', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Correlation between Assets A and B', fontsize=12)
plt.ylabel('Portfolio Variance', fontsize=12)
plt.title('Effect of Correlation on Portfolio Variance', fontsize=14)

# Mark the current value
plt.scatter([correlation], [portfolio_variance], color='red', s=100, label=f'Portfolio Variance = {portfolio_variance:.1f}')
plt.legend()

# Add formula and calculation
formula = (f'$Var(Portfolio) = w_A^2 \\times Var(A) + w_B^2 \\times Var(B) + 2 w_A w_B \\times Cov(A,B)$\n'
           f'$Var(Portfolio) = {weight_A}^2 \\times {variance_A} + {weight_B}^2 \\times {variance_B} + '
           f'2 \\times {weight_A} \\times {weight_B} \\times {covariance}$\n'
           f'$Var(Portfolio) = {term1} + {term2} + {term3} = {portfolio_variance:.1f}$')

plt.annotate(formula, 
            xy=(0, portfolio_variance + 20),
            xytext=(0, portfolio_variance + 20),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add diversification benefit annotation
individual_variance = variance_A  # Both have variance of 100
plt.annotate(f'Diversification Benefit: Reduction from {individual_variance} to {portfolio_variance:.1f}', 
            xy=(correlation, portfolio_variance - 15),
            xytext=(correlation, portfolio_variance - 15),
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'portfolio_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 5: Bias-Variance Tradeoff
print("\n\nExample 5: Bias-Variance Tradeoff")
# Data
degrees = [1, 3, 5, 10]
training_mse = [150, 80, 40, 5]
test_mse = [155, 95, 90, 200]
bias_squared = [140, 75, 35, 3]
variance = [15, 20, 55, 197]

print("Model complexity (polynomial degree) and error decomposition:")
print("| Polynomial Degree | Training MSE | Test MSE | Bias² | Variance |")
print("|-------------------|--------------|----------|-------|----------|")
for i in range(len(degrees)):
    print(f"| {degrees[i]:17} | {training_mse[i]:12} | {test_mse[i]:8} | {bias_squared[i]:5} | {variance[i]:8} |")

print("\nObservations:")
print("1. As model complexity increases, training error consistently decreases")
print("2. The bias component decreases with increasing complexity")
print("3. The variance component increases dramatically with complexity")
print("4. Test error follows a U-shaped curve, with the best balance at polynomial degree 3")

# Create grouped bar plot
plt.figure(figsize=(10, 6))
x = np.arange(len(degrees))
width = 0.2

plt.bar(x - width*1.5, training_mse, width, label='Training MSE', color='blue', alpha=0.7)
plt.bar(x - width/2, test_mse, width, label='Test MSE', color='red', alpha=0.7)
plt.bar(x + width/2, bias_squared, width, label='Bias²', color='green', alpha=0.7)
plt.bar(x + width*1.5, variance, width, label='Variance', color='purple', alpha=0.7)

plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Error / Component Value', fontsize=12)
plt.title('Bias-Variance Tradeoff in Polynomial Regression', fontsize=14)
plt.xticks(x, degrees)
plt.legend()
plt.grid(True, alpha=0.3)

# Add annotations
plt.annotate('Underfitting\nHigh Bias', 
            xy=(0, 50),
            xytext=(0, 50),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.annotate('Best Model\nGood Balance', 
            xy=(1, 50),
            xytext=(1, 50),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

plt.annotate('Overfitting\nHigh Variance', 
            xy=(3, 140),
            xytext=(3, 140),
            fontsize=10,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'bias_variance_tradeoff.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 6: Sample Variance Calculation
print("\n\nExample 6: Sample Variance Calculation")
# Sample data
sample_data = np.array([185, 210, 195, 222, 203, 197])
print(f"Sample data: {sample_data}")

# Calculate sample mean
print("\nStep 1: Calculate the sample mean")
sample_mean = np.mean(sample_data)
print(f"Mean = ({' + '.join(map(str, sample_data))}) / {len(sample_data)} = {np.sum(sample_data)} / {len(sample_data)} = {sample_mean}")

# Calculate deviations and squared deviations
print("\nStep 2: Calculate deviations from the mean and their squares")
sample_deviations = sample_data - sample_mean
squared_deviations = sample_deviations**2

for i, (data_point, deviation, sq_deviation) in enumerate(zip(sample_data, sample_deviations, squared_deviations)):
    print(f"  ({data_point} - {sample_mean}) = {deviation:.1f}, squared: {deviation:.1f}² = {sq_deviation:.1f}")

# Calculate sample variance
print("\nStep 3: Calculate the sample variance")
sum_squared_deviations = np.sum(squared_deviations)
sample_variance = sum_squared_deviations / (len(sample_data) - 1)

print(f"Sum of squared deviations = {' + '.join([f'{sd:.1f}' for sd in squared_deviations])} = {sum_squared_deviations:.1f}")
print(f"Sample variance = sum of squared deviations / (n - 1) = {sum_squared_deviations:.1f} / {len(sample_data) - 1} = {sample_variance:.1f}")
print(f"Therefore, the sample variance is {sample_variance:.1f} (mg/dL)²")
print(f"The sample standard deviation is √{sample_variance:.1f} = {np.sqrt(sample_variance):.1f} mg/dL")

# Create the plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(sample_data)), sample_data, color='blue', s=100)
plt.axhline(y=sample_mean, color='red', linestyle='--', label=f'Mean = {sample_mean:.1f}')
plt.xlabel('Patient', fontsize=12)
plt.ylabel('Cholesterol Level (mg/dL)', fontsize=12)
plt.title('Sample Data with Mean', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Add vertical lines to show deviations
for i, (data_point, deviation) in enumerate(zip(sample_data, sample_deviations)):
    plt.plot([i, i], [sample_mean, data_point], 'g-', alpha=0.5)

# Add text labels for deviations
for i, dev in enumerate(sample_deviations):
    plt.text(i, (sample_mean + sample_data[i])/2, f'{dev:.1f}', 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# Plot the squared deviations
plt.subplot(1, 2, 2)
bars = plt.bar(range(len(sample_data)), squared_deviations, color='purple', alpha=0.7)
plt.xlabel('Patient', fontsize=12)
plt.ylabel('Squared Deviation (mg/dL)²', fontsize=12)
plt.title('Squared Deviations from Mean', fontsize=14)
plt.grid(True, alpha=0.3)

# Add text labels for squared deviations
for i, sq_dev in enumerate(squared_deviations):
    plt.text(i, sq_dev/2, f'{sq_dev:.1f}', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()

# Add variance calculation annotation
plt.figtext(0.5, 0.01, 
           f"Sample Variance = Sum of Squared Deviations / (n - 1) = {np.sum(squared_deviations):.1f} / {len(sample_data) - 1} = {sample_variance:.1f} (mg/dL)²",
           ha="center", fontsize=12, bbox={"facecolor":"yellow", "alpha":0.2, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(images_dir, 'sample_variance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 7: Quiz Example - Model Error Decomposition
print("\n\nExample 7: Bias-Variance Decomposition in Model Error")
mean_prediction = 25
true_mean = 22
mean_squared_diff = 18

print("Model prediction characteristics:")
print(f"  Mean prediction: {mean_prediction}")
print(f"  True mean target value: {true_mean}")
print(f"  Mean squared difference between predictions and their mean: {mean_squared_diff}")

print("\nStep 1: Calculate the bias")
bias = mean_prediction - true_mean
squared_bias = bias**2
print(f"Bias = Mean prediction - True mean = {mean_prediction} - {true_mean} = {bias}")
print(f"Squared bias = {bias}² = {squared_bias}")

print("\nStep 2: Identify the variance")
variance = mean_squared_diff
print(f"Variance = Mean squared difference between predictions and their mean = {variance}")

print("\nStep 3: Calculate the total mean squared error")
mse = squared_bias + variance
print(f"Total MSE = Bias² + Variance = {squared_bias} + {variance} = {mse}")

print(f"\nConclusion: The variance ({variance}) contributes more to the total error than the bias ({squared_bias}).")

print("\nAll variance example images created successfully.") 