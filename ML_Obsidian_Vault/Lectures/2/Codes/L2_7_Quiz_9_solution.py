import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import beta

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_7_Quiz_9")
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
print("- Three weather prediction models for tomorrow:")
print("  - Model 1: Predicts sunny with probability 0.7")
print("  - Model 2: Predicts sunny with probability 0.8")
print("  - Model 3: Predicts sunny with probability 0.6")
print("- Posterior probabilities of each model based on historical data:")
print("  - P(M₁|D) = 0.5")
print("  - P(M₂|D) = 0.3")
print("  - P(M₃|D) = 0.2")
print("\nTask:")
print("1. Calculate the Bayesian Model Averaged prediction for tomorrow being sunny")
print("2. If tomorrow is actually rainy, how would the posterior probabilities change?")
print("3. What advantage does Bayesian Model Averaging have over selecting the highest probability model?")

# Step 2: Calculating the Bayesian Model Averaged prediction
print_step_header(2, "Calculating the Bayesian Model Averaged Prediction")

# Model probabilities
model_probs = np.array([0.5, 0.3, 0.2])
# Probabilities of sunny prediction from each model
sunny_probs = np.array([0.7, 0.8, 0.6])

# Calculate the model-averaged probability
bma_sunny = np.sum(model_probs * sunny_probs)

print("The Bayesian Model Averaged (BMA) prediction is calculated as:")
print("P(sunny|D) = ∑ P(sunny|Mi, D) × P(Mi|D)")
print("P(sunny|D) = P(sunny|M₁, D) × P(M₁|D) + P(sunny|M₂, D) × P(M₂|D) + P(sunny|M₃, D) × P(M₃|D)")
print(f"P(sunny|D) = {sunny_probs[0]} × {model_probs[0]} + {sunny_probs[1]} × {model_probs[1]} + {sunny_probs[2]} × {model_probs[2]}")
print(f"P(sunny|D) = {sunny_probs[0] * model_probs[0]:.4f} + {sunny_probs[1] * model_probs[1]:.4f} + {sunny_probs[2] * model_probs[2]:.4f}")
print(f"P(sunny|D) = {bma_sunny:.4f}")

# Visualize the model-averaged prediction
plt.figure(figsize=(10, 6))

# Plot individual model predictions
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (prob, model_prob) in enumerate(zip(sunny_probs, model_probs)):
    plt.bar(i, prob, color=colors[i], alpha=0.7, label=f'Model {i+1}: P(sunny) = {prob:.1f}, Weight = {model_prob:.1f}')

# Add the model-averaged prediction
plt.axhline(y=bma_sunny, color='red', linestyle='-', linewidth=2, label=f'Model-Averaged: P(sunny) = {bma_sunny:.4f}')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Probability of Sunny Weather', fontsize=12)
plt.title('Model Predictions and Bayesian Model Average', fontsize=14)
plt.xticks([0, 1, 2], ['Model 1', 'Model 2', 'Model 3'])
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "model_averaged_prediction.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 3: Updating the posterior probabilities if tomorrow is rainy
print_step_header(3, "Updating Posterior Probabilities if Tomorrow is Rainy")

# Probability of rainy for each model
rainy_probs = 1 - sunny_probs

# Calculating unnormalized updated posterior probabilities
# P(Mi|D, rainy) ∝ P(rainy|Mi) × P(Mi|D)
updated_unnorm_probs = rainy_probs * model_probs

# Normalize to get proper probabilities
updated_probs = updated_unnorm_probs / np.sum(updated_unnorm_probs)

print("If tomorrow is rainy, we update the posterior probabilities using Bayes' rule:")
print("P(Mi|D, rainy) ∝ P(rainy|Mi) × P(Mi|D)")
print("\nFor each model:")
for i in range(3):
    print(f"P(M{i+1}|D, rainy) ∝ {rainy_probs[i]:.4f} × {model_probs[i]:.4f} = {updated_unnorm_probs[i]:.4f}")

print("\nAfter normalization:")
for i in range(3):
    print(f"P(M{i+1}|D, rainy) = {updated_unnorm_probs[i]:.4f} / {np.sum(updated_unnorm_probs):.4f} = {updated_probs[i]:.4f}")

# Visualize the updated posterior probabilities
plt.figure(figsize=(10, 6))

# Original posterior probabilities
plt.bar(np.arange(3) - 0.2, model_probs, width=0.4, color='blue', alpha=0.7, label='Original Posteriors')

# Updated posterior probabilities
plt.bar(np.arange(3) + 0.2, updated_probs, width=0.4, color='red', alpha=0.7, label='Updated Posteriors (after rainy day)')

# Add text annotations
for i in range(3):
    plt.text(i - 0.2, model_probs[i] + 0.02, f'{model_probs[i]:.2f}', ha='center', va='bottom', fontsize=10)
    plt.text(i + 0.2, updated_probs[i] + 0.02, f'{updated_probs[i]:.2f}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Posterior Probability P(Mi|D)', fontsize=12)
plt.title('Posterior Probabilities Before and After Observing Rainy Day', fontsize=14)
plt.xticks([0, 1, 2], ['Model 1', 'Model 2', 'Model 3'])
plt.ylim(0, max(max(model_probs), max(updated_probs)) + 0.1)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "updated_posteriors.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 4: Advantages of Bayesian Model Averaging
print_step_header(4, "Advantages of Bayesian Model Averaging over Selecting the Highest Probability Model")

# Calculate the prediction from the single best model (by posterior probability)
best_model_index = np.argmax(model_probs)
best_model_prob = sunny_probs[best_model_index]

print(f"The highest probability model is Model {best_model_index + 1} with posterior probability {model_probs[best_model_index]:.4f}")
print(f"This model predicts sunny with probability {best_model_prob:.4f}")
print(f"The Bayesian Model Averaged prediction is {bma_sunny:.4f}")
print(f"Difference: {np.abs(bma_sunny - best_model_prob):.4f}")

print("\nAdvantages of Bayesian Model Averaging:")
print("1. Incorporates the uncertainty in model selection")
print("2. Leverages predictions from all models, weighted by their posterior probabilities")
print("3. Often provides more accurate predictions than any single model")
print("4. Reduces the risk of overconfidence in a single, potentially incorrect model")
print("5. Accounts for different strengths of different models in different situations")

# Visualize the comparison
plt.figure(figsize=(10, 6))

# Plot individual model predictions with their weights as alpha
for i, (prob, model_prob) in enumerate(zip(sunny_probs, model_probs)):
    label = f'Model {i+1}: P(sunny) = {prob:.1f}, Weight = {model_prob:.1f}'
    if i == best_model_index:
        label += ' (Best Model)'
    
    # Full bar with reduced alpha
    plt.bar(i, prob, color=colors[i], alpha=0.3)
    
    # Weighted contribution with full alpha
    plt.bar(i, prob * model_prob, color=colors[i], label=label)

# Add the model-averaged prediction
plt.axhline(y=bma_sunny, color='red', linestyle='-', linewidth=2, label=f'Model-Averaged: P(sunny) = {bma_sunny:.4f}')

# Add the best model prediction
plt.axhline(y=best_model_prob, color='green', linestyle='--', linewidth=2, 
            label=f'Best Model: P(sunny) = {best_model_prob:.4f}')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Probability of Sunny Weather', fontsize=12)
plt.title('Comparison: Bayesian Model Averaging vs. Best Model', fontsize=14)
plt.xticks([0, 1, 2], ['Model 1', 'Model 2', 'Model 3'])
plt.ylim(0, 1)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
plt.grid(True)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "bma_vs_best_model.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

# Step 5: Illustrate the concept with different scenarios
print_step_header(5, "Illustrating BMA with Different Model Probabilities")

# Create a range of scenarios with different model probabilities
scenario_names = [
    "Equal model probabilities",
    "Strong confidence in one model",
    "Original scenario",
    "Balanced between two models",
    "Gradually decreasing probabilities"
]

scenario_probs = [
    [1/3, 1/3, 1/3],          # Equal
    [0.9, 0.07, 0.03],        # Strong confidence in one model
    [0.5, 0.3, 0.2],          # Original
    [0.48, 0.48, 0.04],       # Balanced between two
    [0.6, 0.3, 0.1]           # Gradually decreasing
]

# Calculate BMA for each scenario
bma_values = []
for scenario_prob in scenario_probs:
    bma_values.append(np.sum(np.array(scenario_prob) * sunny_probs))

# Visualize the scenarios
plt.figure(figsize=(12, 8))

for i, (name, probs, bma) in enumerate(zip(scenario_names, scenario_probs, bma_values)):
    plt.subplot(len(scenario_names), 1, i+1)
    
    # Plot model probabilities
    bars = plt.bar(np.arange(3), probs, alpha=0.7)
    for j, bar in enumerate(bars):
        plt.text(j, probs[j] + 0.02, f'{probs[j]:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.axhline(y=bma, color='red', linestyle='-', linewidth=2, 
                label=f'BMA = {bma:.4f}')
    
    best_model = np.argmax(probs)
    plt.axhline(y=sunny_probs[best_model], color='green', linestyle='--', linewidth=2,
                label=f'Best Model ({best_model+1}) = {sunny_probs[best_model]:.4f}')
    
    plt.ylabel('P(Mi|D)')
    plt.title(name)
    plt.xticks([0, 1, 2], ['M1', 'M2', 'M3'])
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "different_scenarios.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {file_path}")

print("\nConclusion:")
print("Bayesian Model Averaging provides a principled way to incorporate predictions from multiple models")
print("while accounting for our uncertainty about which model is correct. It automatically adapts to")
print("different scenarios, giving more weight to more probable models while still leveraging the")
print("information from all available models.")