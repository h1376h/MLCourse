import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Set up the problem
print_step_header(1, "Understanding the Problem")

print("Given:")
print("- Binary classification problem with true labels y and predicted probabilities ŷ")
print()
print("Tasks:")
print("1. Write down the formula for the cross-entropy loss")
print("2. Calculate the cross-entropy loss for 4 samples with:")
print("   - True labels y = [1, 0, 1, 0]")
print("   - Predicted probabilities ŷ = [0.8, 0.3, 0.6, 0.2]")
print("3. Calculate the KL divergence between the true distribution and predicted distribution")
print("4. Explain how minimizing cross-entropy loss relates to maximum likelihood estimation")
print()

# Step 2: Define and explain the cross-entropy loss formula
print_step_header(2, "Cross-Entropy Loss Formula")

print("In binary classification, the cross-entropy loss for a single sample is:")
print("L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]")
print()
print("For multiple samples, we take the average:")
print("L(y, ŷ) = -1/N * Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]")
print()
print("Where:")
print("- y_i is the true label (0 or 1) for sample i")
print("- ŷ_i is the predicted probability for sample i")
print("- N is the number of samples")
print()

# Step 3: Calculate cross-entropy loss for the given samples
print_step_header(3, "Calculating Cross-Entropy Loss for Given Samples")

# Define the true labels and predicted probabilities
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.8, 0.3, 0.6, 0.2])

# Function to calculate binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate binary cross-entropy loss.
    Epsilon is a small constant to avoid log(0).
    """
    # Clip predicted values to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate loss for each sample
    losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # Return average loss
    return np.mean(losses)

# Calculate the cross-entropy loss
ce_loss = binary_cross_entropy(y_true, y_pred)
print(f"Cross-Entropy Loss: {ce_loss:.6f}")

# Show the calculation step by step
print("\nCalculation steps for each sample:")
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    # Avoid log(0) or log(1)
    pred_safe = np.clip(pred, 1e-15, 1 - 1e-15)
    
    if true == 1:
        term = -np.log(pred_safe)
        print(f"Sample {i+1}: y = {true}, ŷ = {pred:.4f}, -log(ŷ) = {term:.6f}")
    else:  # true == 0
        term = -np.log(1 - pred_safe)
        print(f"Sample {i+1}: y = {true}, ŷ = {pred:.4f}, -log(1-ŷ) = {term:.6f}")

# Calculate loss for each sample separately
individual_losses = []
for true, pred in zip(y_true, y_pred):
    pred_safe = np.clip(pred, 1e-15, 1 - 1e-15)
    loss = -true * np.log(pred_safe) - (1 - true) * np.log(1 - pred_safe)
    individual_losses.append(loss)

print("\nIndividual losses for each sample:")
for i, loss in enumerate(individual_losses):
    print(f"Sample {i+1}: {loss:.6f}")

print(f"\nAverage loss (Cross-Entropy Loss): {np.mean(individual_losses):.6f}")

# Visualize the individual losses and overall loss
plt.figure(figsize=(10, 6))
bar_positions = np.arange(len(individual_losses))
bars = plt.bar(bar_positions, individual_losses, 
        color=['blue', 'green', 'orange', 'red'])

# Add a horizontal line for the mean
plt.axhline(y=np.mean(individual_losses), color='black', linestyle='--', 
           label=f'Mean Loss: {np.mean(individual_losses):.4f}')

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Sample Index')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss for Each Sample')
plt.xticks(bar_positions, [f'Sample {i+1}' for i in range(len(individual_losses))])
plt.legend()
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "individual_losses.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 4: Visualize cross-entropy loss function for different predictions
print_step_header(4, "Visualizing the Cross-Entropy Loss Function")

# We'll visualize how the loss changes for different predicted probabilities
# when true label is 0 or 1
pred_range = np.linspace(0.01, 0.99, 100)

# Calculate loss when true label is 1
loss_y1 = -np.log(pred_range)

# Calculate loss when true label is 0
loss_y0 = -np.log(1 - pred_range)

plt.figure(figsize=(10, 6))
plt.plot(pred_range, loss_y1, 'b-', linewidth=2, label='True Label y = 1')
plt.plot(pred_range, loss_y0, 'r-', linewidth=2, label='True Label y = 0')

# Mark our specific samples
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true == 1:
        loss = -np.log(max(pred, 1e-15))
        plt.scatter([pred], [loss], color='blue', s=100, zorder=10)
        plt.text(pred, loss + 0.2, f'Sample {i+1}', ha='center')
    else:  # true == 0
        loss = -np.log(max(1 - pred, 1e-15))
        plt.scatter([pred], [loss], color='red', s=100, zorder=10)
        plt.text(pred, loss + 0.2, f'Sample {i+1}', ha='center')

plt.xlabel('Predicted Probability (ŷ)')
plt.ylabel('Loss Value')
plt.title('Cross-Entropy Loss Function')
plt.grid(True)
plt.legend()
plt.ylim(0, 5)
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "loss_function.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 5: Calculate KL divergence
print_step_header(5, "Calculating KL Divergence between True and Predicted Distributions")

# For binary classification, the KL divergence is calculated for each sample
# and then averaged
def kl_divergence_binary(y_true, y_pred, epsilon=1e-15):
    """
    Calculate KL divergence for binary classification.
    Epsilon is a small constant to avoid log(0).
    """
    # Clip predicted values to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # For binary classification, the true distribution is a one-hot distribution
    # KL divergence for each sample
    kl_divs = y_true * np.log(y_true / y_pred) + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))
    
    # Return average KL divergence
    return np.mean(kl_divs)

# In the binary case, KL divergence can be simplified because y_true is either 0 or 1
# When y_true = 1: KL = log(1/y_pred)
# When y_true = 0: KL = log(1/(1-y_pred))
def kl_divergence_binary_simplified(y_true, y_pred, epsilon=1e-15):
    """
    Simplified KL divergence calculation for binary classification.
    """
    # Clip predicted values to avoid log(0) or log(1)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate KL divergence for each sample
    kl_divs = np.zeros_like(y_true, dtype=float)
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 1:
            kl_divs[i] = np.log(1 / pred)
        else:  # true == 0
            kl_divs[i] = np.log(1 / (1 - pred))
    
    # Return average KL divergence
    return np.mean(kl_divs)

# Calculate the KL divergence
kl_div = kl_divergence_binary_simplified(y_true, y_pred)
print(f"KL Divergence: {kl_div:.6f}")

# Show the calculation step by step
print("\nCalculation steps for KL divergence:")
for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    pred_safe = np.clip(pred, 1e-15, 1 - 1e-15)
    
    if true == 1:
        kl = np.log(1 / pred_safe)
        print(f"Sample {i+1}: y = {true}, ŷ = {pred:.4f}, KL = log(1/ŷ) = {kl:.6f}")
    else:  # true == 0
        kl = np.log(1 / (1 - pred_safe))
        print(f"Sample {i+1}: y = {true}, ŷ = {pred:.4f}, KL = log(1/(1-ŷ)) = {kl:.6f}")

# Calculate the individual KL divergences
individual_kl_divs = []
for true, pred in zip(y_true, y_pred):
    pred_safe = np.clip(pred, 1e-15, 1 - 1e-15)
    if true == 1:
        kl = np.log(1 / pred_safe)
    else:  # true == 0
        kl = np.log(1 / (1 - pred_safe))
    individual_kl_divs.append(kl)

print("\nIndividual KL divergences for each sample:")
for i, kl_div in enumerate(individual_kl_divs):
    print(f"Sample {i+1}: {kl_div:.6f}")

print(f"\nAverage KL divergence: {np.mean(individual_kl_divs):.6f}")

# Calculate the entropy of the true distribution
def entropy_binary(y):
    """Calculate entropy of binary labels."""
    # We need to check how many 1s and 0s are in the distribution
    p1 = np.mean(y)
    p0 = 1 - p1
    
    # Handle edge cases where p0 or p1 is 0
    if p1 == 0 or p1 == 1:
        return 0
    
    # Calculate entropy
    return -p1 * np.log(p1) - p0 * np.log(p0)

# Calculate the entropy of true labels
h_true = entropy_binary(y_true)
print(f"Entropy of true distribution: {h_true:.6f}")

# Verify that cross-entropy loss equals entropy plus KL divergence
print("\nVerifying relationship: Cross-Entropy = Entropy + KL Divergence")
print(f"Cross-Entropy: {ce_loss:.6f}")
print(f"Entropy + KL Divergence: {h_true + kl_div:.6f}")

if np.isclose(ce_loss, h_true + kl_div):
    print("✓ Relationship holds: Cross-Entropy = Entropy + KL Divergence")
else:
    print("✗ Relationship does not hold. This is because we're looking at individual samples,")
    print("  not the distribution as a whole.")
    
# Visualize KL divergence vs. cross-entropy
plt.figure(figsize=(10, 6))
bar_positions = np.arange(3)
values = [ce_loss, h_true, kl_div]
labels = ['Cross-Entropy', 'Entropy', 'KL Divergence']
colors = ['blue', 'green', 'red']

bars = plt.bar(bar_positions, values, color=colors)

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.4f}', ha='center', va='bottom')

plt.xlabel('Measure')
plt.ylabel('Value')
plt.title('Relationship between Cross-Entropy, Entropy and KL Divergence')
plt.xticks(bar_positions, labels)
plt.ylim(0, max(values) * 1.2)  # Give some space for labels
plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "entropy_kl_relationship.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 6: Explain how minimizing cross-entropy relates to maximum likelihood estimation
print_step_header(6, "Cross-Entropy Loss and Maximum Likelihood Estimation")

print("Relationship between Cross-Entropy Loss and Maximum Likelihood Estimation:")
print("\n1. In binary classification, we model P(y=1|x) = ŷ and P(y=0|x) = 1-ŷ")
print("\n2. For a dataset with labels {y_i} and predictions {ŷ_i}, the likelihood is:")
print("   L(θ) = Π[ŷ_i^(y_i) × (1-ŷ_i)^(1-y_i)]")
print("   where θ represents the model parameters")
print("\n3. Taking the negative log-likelihood:")
print("   -log(L(θ)) = -Σ[y_i×log(ŷ_i) + (1-y_i)×log(1-ŷ_i)]")
print("\n4. This is exactly the cross-entropy loss! Therefore:")
print("   * Minimizing cross-entropy loss = Maximizing likelihood")
print("   * Minimizing cross-entropy loss = Minimizing negative log-likelihood")
print("\n5. Intuitively: Cross-entropy measures the difference between:")
print("   * The true distribution (one-hot encoded true labels)")
print("   * The predicted distribution (model's probabilistic predictions)")
print("   The closer these distributions, the better the model prediction")

# Create a visualization of the relationship
plt.figure(figsize=(12, 8))

# Create a simplified diagram
plt.subplot(111)
plt.axis('off')  # Turn off the axis

# Title
plt.text(0.5, 0.95, "Relationship Between Cross-Entropy Loss and Maximum Likelihood Estimation", 
         fontsize=14, ha='center', fontweight='bold')

# Left side: Likelihood
plt.text(0.25, 0.85, "Maximum Likelihood Estimation", fontsize=12, ha='center', fontweight='bold')
plt.text(0.25, 0.8, "Likelihood:", fontsize=10, ha='center')
plt.text(0.25, 0.75, r"$L(\theta) = \prod_{i=1}^{N} \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}$", fontsize=12, ha='center')
plt.text(0.25, 0.65, "Log-Likelihood:", fontsize=10, ha='center')
plt.text(0.25, 0.6, r"$\log L(\theta) = \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)$", fontsize=12, ha='center')
plt.text(0.25, 0.5, "Negative Log-Likelihood:", fontsize=10, ha='center')
plt.text(0.25, 0.45, r"$-\log L(\theta) = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)$", fontsize=12, ha='center')
plt.text(0.25, 0.35, "Goal: MAXIMIZE Likelihood", fontsize=10, ha='center', fontweight='bold')
plt.text(0.25, 0.3, "= MINIMIZE Negative Log-Likelihood", fontsize=10, ha='center', fontweight='bold')

# Right side: Cross-Entropy
plt.text(0.75, 0.85, "Cross-Entropy Loss", fontsize=12, ha='center', fontweight='bold')
plt.text(0.75, 0.75, "Cross-Entropy Formula:", fontsize=10, ha='center')
plt.text(0.75, 0.7, r"$CE = -\frac{1}{N}\sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)$", fontsize=12, ha='center')
plt.text(0.75, 0.6, "Information Theory Interpretation:", fontsize=10, ha='center')
plt.text(0.75, 0.55, "Measures the 'distance' between true and", fontsize=10, ha='center')
plt.text(0.75, 0.5, "predicted probability distributions", fontsize=10, ha='center')
plt.text(0.75, 0.4, "KL Divergence + Entropy", fontsize=10, ha='center')
plt.text(0.75, 0.35, "Goal: MINIMIZE Cross-Entropy Loss", fontsize=10, ha='center', fontweight='bold')

# Center: Connection
plt.text(0.5, 0.5, "=", fontsize=20, ha='center')
plt.text(0.5, 0.3, "Therefore, minimizing cross-entropy loss\nis equivalent to\nmaximizing likelihood", 
         fontsize=12, ha='center', fontweight='bold')

# Add example
plt.text(0.5, 0.15, "Example with our data:", fontsize=12, ha='center', fontweight='bold')
plt.text(0.5, 0.1, f"Cross-Entropy Loss: {ce_loss:.4f} = -Log Likelihood (scaled by 1/N)", 
         fontsize=10, ha='center')

plt.tight_layout()

# Save the figure
file_path = os.path.join(save_dir, "cross_entropy_mle.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {file_path}")

# Step 7: Conclusion
print_step_header(7, "Conclusion and Answer Summary")

print("Question 4 Solution Summary:")
print("\n1. The formula for cross-entropy loss in binary classification is:")
print("   L(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]")
print("   For multiple samples, we take the average:")
print("   L(y, ŷ) = -1/N * Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]")

print("\n2. Cross-entropy loss for the given samples:")
print("   - True labels y = [1, 0, 1, 0]")
print("   - Predicted probabilities ŷ = [0.8, 0.3, 0.6, 0.2]")
print(f"   Cross-entropy loss = {ce_loss:.6f}")

print("\n3. KL divergence between true and predicted distributions:")
print(f"   KL divergence = {kl_div:.6f}")

print("\n4. Minimizing cross-entropy loss is equivalent to maximizing likelihood because:")
print("   - The negative log-likelihood formula for binary classification is identical")
print("     to the cross-entropy loss formula")
print("   - This means that finding parameters that minimize cross-entropy loss")
print("     is the same as finding parameters that maximize the likelihood of the data")
print("   - From an information theory perspective, we're minimizing the difference")
print("     between the true data distribution and our model's predicted distribution") 