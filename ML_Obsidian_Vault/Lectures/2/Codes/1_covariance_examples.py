import numpy as np
import matplotlib.pyplot as plt
import os

print("\n=== COVARIANCE EXAMPLES: STEP-BY-STEP SOLUTIONS ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the ML_Obsidian_Vault directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Example 1: Stock Market Returns
print("\nExample 1: Stock Market Returns")
print("Calculating covariance between Tech Corp (TC) and Financial Services Inc (FSI) returns")

# Data
tc_returns = np.array([1.2, -0.5, 0.8, -1.1, 1.6])  # TC returns in %
fsi_returns = np.array([0.8, -0.3, 0.2, -0.9, 1.2])  # FSI returns in %

print("\nStep 1: Calculate the means")
mean_tc = np.mean(tc_returns)
mean_fsi = np.mean(fsi_returns)
print(f"Mean of TC returns: {mean_tc:.1f}%")
print(f"Mean of FSI returns: {mean_fsi:.1f}%")

print("\nStep 2: Calculate deviations from means")
print("The deviation of each data point from its mean is calculated as:")
print("deviation = x_i - mean")
print("\nFor TC returns:")
tc_deviations = tc_returns - mean_tc
for i, (ret, dev) in enumerate(zip(tc_returns, tc_deviations)):
    print(f"Day {i+1}: {ret:.1f}% - {mean_tc:.1f}% = {dev:.1f}%")

print("\nFor FSI returns:")
fsi_deviations = fsi_returns - mean_fsi
for i, (ret, dev) in enumerate(zip(fsi_returns, fsi_deviations)):
    print(f"Day {i+1}: {ret:.1f}% - {mean_fsi:.1f}% = {dev:.1f}%")

print("\nStep 3: Calculate products of deviations")
print("The product of deviations for each pair is calculated as:")
print("product = (x_i - mean_x) × (y_i - mean_y)")
products = tc_deviations * fsi_deviations
for i, (tc_dev, fsi_dev, prod) in enumerate(zip(tc_deviations, fsi_deviations, products)):
    print(f"Day {i+1}: ({tc_dev:.1f}) × ({fsi_dev:.1f}) = {prod:.2f}")

print("\nStep 4: Calculate covariance")
print("Covariance is calculated as the average of the products of deviations:")
print("covariance = sum(products) / (n - 1)")
covariance = np.sum(products) / (len(tc_returns) - 1)
print(f"Covariance = {covariance:.3f}")

# Create visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(tc_returns, fsi_returns, color='blue', s=100)
plt.axvline(x=mean_tc, color='red', linestyle='--', label=f'Mean TC = {mean_tc:.1f}%')
plt.axhline(y=mean_fsi, color='green', linestyle='--', label=f'Mean FSI = {mean_fsi:.1f}%')

# Add labels and title
plt.xlabel('Tech Corp Returns (%)')
plt.ylabel('Financial Services Inc Returns (%)')
plt.title('Stock Returns Scatter Plot with Covariance')
plt.legend()

# Add covariance information
plt.figtext(0.5, 0.01, f"Covariance = {covariance:.3f}", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Add second subplot for deviations
plt.subplot(1, 2, 2)
plt.scatter(tc_deviations, fsi_deviations, color='purple', s=100)
plt.axvline(x=0, color='red', linestyle='--')
plt.axhline(y=0, color='green', linestyle='--')
plt.xlabel('TC Deviations from Mean')
plt.ylabel('FSI Deviations from Mean')
plt.title('Deviations Scatter Plot')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'stock_covariance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 2: Housing Data Analysis
print("\n\nExample 2: Housing Data Analysis")
print("Calculating covariance between house size and price")

# Data
sizes = np.array([1500, 2200, 1800, 3000, 2500])  # square feet
prices = np.array([250, 340, 275, 455, 390])  # thousands of dollars

print("\nStep 1: Calculate the means")
mean_size = np.mean(sizes)
mean_price = np.mean(prices)
print(f"Mean size: {mean_size:.0f} sq ft")
print(f"Mean price: ${mean_price:.0f}k")

print("\nStep 2: Calculate deviations from means")
print("The deviation of each data point from its mean is calculated as:")
print("deviation = x_i - mean")
print("\nFor house sizes:")
size_deviations = sizes - mean_size
for i, (size, dev) in enumerate(zip(sizes, size_deviations)):
    print(f"House {i+1}: {size} sq ft - {mean_size:.0f} sq ft = {dev:.0f} sq ft")

print("\nFor house prices:")
price_deviations = prices - mean_price
for i, (price, dev) in enumerate(zip(prices, price_deviations)):
    print(f"House {i+1}: ${price}k - ${mean_price:.0f}k = ${dev:.0f}k")

print("\nStep 3: Calculate products of deviations")
print("The product of deviations for each pair is calculated as:")
print("product = (x_i - mean_x) × (y_i - mean_y)")
products = size_deviations * price_deviations
for i, (size_dev, price_dev, prod) in enumerate(zip(size_deviations, price_deviations, products)):
    print(f"House {i+1}: ({size_dev:.0f}) × ({price_dev:.0f}) = {prod:.0f}")

print("\nStep 4: Calculate covariance")
print("Covariance is calculated as the average of the products of deviations:")
print("covariance = sum(products) / (n - 1)")
covariance = np.sum(products) / (len(sizes) - 1)
print(f"Covariance = {covariance:.0f}")

# Create visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(sizes, prices, color='blue', s=100)
plt.axvline(x=mean_size, color='red', linestyle='--', label=f'Mean Size = {mean_size:.0f} sq ft')
plt.axhline(y=mean_price, color='green', linestyle='--', label=f'Mean Price = ${mean_price:.0f}k')

# Add labels and title
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($k)')
plt.title('House Size vs Price Scatter Plot with Covariance')
plt.legend()

# Add covariance information
plt.figtext(0.5, 0.01, f"Covariance = {covariance:.0f}", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Add second subplot for deviations
plt.subplot(1, 2, 2)
plt.scatter(size_deviations, price_deviations, color='purple', s=100)
plt.axvline(x=0, color='red', linestyle='--')
plt.axhline(y=0, color='green', linestyle='--')
plt.xlabel('Size Deviations from Mean')
plt.ylabel('Price Deviations from Mean')
plt.title('Deviations Scatter Plot')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'housing_covariance.png'), dpi=100, bbox_inches='tight')
plt.close()

# Example 3: Multivariate Covariance Matrix
print("\n\nExample 3: Multivariate Covariance Matrix")
print("Calculating covariance matrix for plant characteristics")

# Data
heights = np.array([45, 60, 35, 50])  # cm
leaf_widths = np.array([12, 15, 10, 14])  # mm
chlorophyll = np.array([2.5, 3.0, 2.0, 2.8])  # mg/g

# Create data matrix
data = np.column_stack((heights, leaf_widths, chlorophyll))

print("\nStep 1: Calculate means for each variable")
means = np.mean(data, axis=0)
print(f"Mean heights: {means[0]:.1f} cm")
print(f"Mean leaf widths: {means[1]:.1f} mm")
print(f"Mean chlorophyll: {means[2]:.2f} mg/g")

print("\nStep 2: Calculate deviations from means")
print("The deviation matrix is calculated as:")
print("deviations = data - means")
deviations = data - means
print("\nDeviations matrix:")
print(deviations)

print("\nFor each plant and variable:")
for i in range(len(heights)):
    print(f"\nPlant {i+1}:")
    print(f"Height deviation: {heights[i]} cm - {means[0]:.1f} cm = {deviations[i, 0]:.1f} cm")
    print(f"Leaf width deviation: {leaf_widths[i]} mm - {means[1]:.1f} mm = {deviations[i, 1]:.1f} mm")
    print(f"Chlorophyll deviation: {chlorophyll[i]} mg/g - {means[2]:.2f} mg/g = {deviations[i, 2]:.2f} mg/g")

print("\nStep 3: Calculate covariance matrix")
print("The covariance matrix is calculated as:")
print("cov_matrix = (deviations.T @ deviations) / (n - 1)")
cov_matrix = np.cov(data, rowvar=False)
print("\nCovariance Matrix:")
print(cov_matrix)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(cov_matrix, cmap='viridis')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Covariance', rotation=-90, va="bottom")

# Add labels
ax.set_xticks(np.arange(len(['Height', 'Leaf Width', 'Chlorophyll'])))
ax.set_yticks(np.arange(len(['Height', 'Leaf Width', 'Chlorophyll'])))
ax.set_xticklabels(['Height', 'Leaf Width', 'Chlorophyll'])
ax.set_yticklabels(['Height', 'Leaf Width', 'Chlorophyll'])

# Add covariance values
for i in range(len(cov_matrix)):
    for j in range(len(cov_matrix)):
        text = ax.text(j, i, f'{cov_matrix[i, j]:.2f}',
                      ha="center", va="center", color="w")

plt.title('Covariance Matrix for Plant Characteristics')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multivariate_covariance.png'), dpi=100, bbox_inches='tight')
plt.close()

print("\nAll covariance example images created successfully.") 