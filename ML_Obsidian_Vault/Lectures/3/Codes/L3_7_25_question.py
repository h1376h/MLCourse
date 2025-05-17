import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_7_Quiz_25")
os.makedirs(save_dir, exist_ok=True)

# Generate empty plot for students to sketch on
plt.figure(figsize=(12, 8))

# Set up the plot
plt.xlabel(r'log($\lambda$)', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.title('Training and Validation Error vs Regularization Strength', fontsize=16)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Set x and y limits
plt.xlim(-5, 5)
plt.ylim(0, 1)

# Add a legend box for the future curves
plt.legend(['Training Error', 'Validation Error'], loc='upper center')

# Add blank regions for students to label
plt.text(-4, 0.9, 'Region 1: ?', fontsize=12)
plt.text(0, 0.9, 'Region 2: ?', fontsize=12)
plt.text(4, 0.9, 'Region 3: ?', fontsize=12)

# Prompt arrows to guide sketching
plt.annotate('', xy=(-4, 0.1), xytext=(-4, 0.05), 
           arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(0, 0.1), xytext=(0, 0.05), 
           arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(4, 0.1), xytext=(4, 0.05), 
           arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'regularization_paths.png'), dpi=300)
plt.close()

print(f"Quiz template image saved to: {save_dir}/regularization_paths.png")
print("TASK: Sketch the general shape of the training and validation error curves as functions of the regularization parameter λ.")
print("      Mark and label the regions of underfitting, optimal fitting, and overfitting on your diagram.") 