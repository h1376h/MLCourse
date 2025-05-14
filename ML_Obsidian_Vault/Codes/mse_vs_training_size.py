import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate the data
num_samples = np.arange(1, 201)  # x-axis: 1 to 200 training samples
mse_values = np.zeros(len(num_samples))

# Simulate the MSE curve
# Initial high MSE
mse_values[0:5] = 0.6 + 0.15 * np.random.randn(5)
# Rapid decrease
for i in range(5, 20):
    mse_values[i] = 0.3 + 0.05 * np.random.randn(1)[0]
# More gradual decrease
for i in range(20, 40):
    mse_values[i] = 0.28 - (i-20) * 0.002 + 0.02 * np.random.randn(1)[0]
# Plateau with small fluctuations
for i in range(40, 200):
    mse_values[i] = 0.26 + 0.01 * np.random.randn(1)[0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(num_samples, mse_values, 'b-', linewidth=1.5)
plt.xlabel('Num of Training Data', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.xlim(0, 200)
plt.ylim(0.2, 0.9)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('plots/mse_vs_training_size.png', dpi=300)
plt.close()

print("MSE vs Training Size plot saved as 'plots/mse_vs_training_size.png'") 