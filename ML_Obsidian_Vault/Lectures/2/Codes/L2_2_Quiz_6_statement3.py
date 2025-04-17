import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L2_2_Quiz_6")
os.makedirs(save_dir, exist_ok=True)

# Set a clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')

print("# Statement 3: For any two random variables X and Y, the mutual information I(X;Y) is always non-negative.")

# Helper function to calculate mutual information
def mutual_information(joint_p):
    # Ensure it's a numpy array and sums to 1
    joint_p = np.array(joint_p)
    joint_p = joint_p / np.sum(joint_p)

    # Calculate marginal probabilities
    p_x = np.sum(joint_p, axis=1)
    p_y = np.sum(joint_p, axis=0)

    # Calculate product of marginals (independence assumption)
    p_x_p_y = np.outer(p_x, p_y)

    # Calculate mutual information I(X;Y) = D_KL( P(X,Y) || P(X)P(Y) )
    mi = 0
    for i in range(joint_p.shape[0]):
        for j in range(joint_p.shape[1]):
            if joint_p[i, j] > 1e-10 and p_x_p_y[i, j] > 1e-10:
                mi += joint_p[i, j] * np.log2(joint_p[i, j] / p_x_p_y[i, j])
    return mi, joint_p, p_x, p_y, p_x_p_y

# Example 1: Dependent Variables
joint_p1 = [[0.3, 0.1],
            [0.1, 0.5]]
mi1, jp1, px1, py1, pxpy1 = mutual_information(joint_p1)

# Example 2: Independent Variables
joint_p2 = [[0.2*0.6, 0.2*0.4],
            [0.8*0.6, 0.8*0.4]] # P(X,Y) = P(X)P(Y)
mi2, jp2, px2, py2, pxpy2 = mutual_information(joint_p2)

# Example 3: Partially Dependent Variables
joint_p3 = [[0.4, 0.1],
            [0.2, 0.3]]
mi3, jp3, px3, py3, pxpy3 = mutual_information(joint_p3)

# Mathematical explanation
print("\n#### Mathematical Definition of Mutual Information:")
print("I(X;Y) = Σ_{x,y} P(x,y) * log2( P(x,y) / (P(x)P(y)) )")
print("It measures the amount of information obtained about one random variable")
print("by observing the other random variable.")
print("")
print("#### Relationship to KL Divergence:")
print("Mutual Information can be expressed as the KL divergence between the joint")
print("distribution P(X,Y) and the product of the marginal distributions P(X)P(Y):")
print("I(X;Y) = D_KL( P(X,Y) || P(X)P(Y) )")
print("")
print("#### Non-Negativity Property:")
print("The KL divergence D_KL(P || Q) is always non-negative (≥ 0). It is zero")
print("if and only if P = Q.")
print("Therefore, since I(X;Y) is a KL divergence, I(X;Y) ≥ 0.")
print("I(X;Y) = 0 if and only if P(X,Y) = P(X)P(Y), which is the definition of independence.")
print("")
print("#### Numerical Calculations:")
print(f"Example 1 (Dependent):")
print(f"  Joint P(X,Y):\n{jp1}")
print(f"  Marginal P(X): {px1}")
print(f"  Marginal P(Y): {py1}")
print(f"  Product P(X)P(Y):\n{pxpy1}")
print(f"  Mutual Information I(X;Y) = {mi1:.4f} (Positive)")
print("")
print(f"Example 2 (Independent):")
print(f"  Joint P(X,Y):\n{jp2}")
print(f"  Marginal P(X): {px2}")
print(f"  Marginal P(Y): {py2}")
print(f"  Product P(X)P(Y):\n{pxpy2}")
print(f"  Mutual Information I(X;Y) = {mi2:.4f} (Zero, due to independence)")
print("")
print(f"Example 3 (Partially Dependent):")
print(f"  Joint P(X,Y):\n{jp3}")
print(f"  Marginal P(X): {px3}")
print(f"  Marginal P(Y): {py3}")
print(f"  Product P(X)P(Y):\n{pxpy3}")
print(f"  Mutual Information I(X;Y) = {mi3:.4f} (Positive)")

# --- Visualization ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

x_labels = ['X=0', 'X=1']
y_labels = ['Y=0', 'Y=1']

def plot_heatmap(ax, data, title):
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=np.max([jp1, jp2, jp3, pxpy1, pxpy2, pxpy3])) # Use consistent color scale
    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.2f}', ha="center", va="center", color="white" if data[i, j] > 0.3 else "black")
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Plot Example 1
plot_heatmap(axes[0, 0], jp1, f'Ex 1: Joint P(X,Y)\nMI = {mi1:.3f}')
plot_heatmap(axes[1, 0], pxpy1, 'Ex 1: Product P(X)P(Y)')

# Plot Example 2
plot_heatmap(axes[0, 1], jp2, f'Ex 2: Joint P(X,Y) (Independent)\nMI = {mi2:.3f}')
plot_heatmap(axes[1, 1], pxpy2, 'Ex 2: Product P(X)P(Y)')

# Plot Example 3
plot_heatmap(axes[0, 2], jp3, f'Ex 3: Joint P(X,Y)\nMI = {mi3:.3f}')
plot_heatmap(axes[1, 2], pxpy3, 'Ex 3: Product P(X)P(Y)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.colorbar(axes[0,0].images[0], ax=axes.ravel().tolist(), shrink=0.5) # Add a common colorbar
fig.suptitle('Mutual Information: Comparing Joint P(X,Y) with Product of Marginals P(X)P(Y)', fontsize=16)

# Save the figure
img_path = os.path.join(save_dir, "statement3_mutual_information.png")
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.close()

print("\n#### Visual Verification:")
print(f"Plot comparing joint distributions and product of marginals saved to: {img_path}")
print("- Top row shows the actual joint distribution P(X,Y).")
print("- Bottom row shows the distribution assuming independence, P(X)P(Y).")
print("- Mutual Information measures the difference (KL divergence) between the top and bottom row heatmaps.")
print("- When P(X,Y) = P(X)P(Y) (Example 2, independence), the heatmaps are identical, and MI = 0.")
print("- When P(X,Y) != P(X)P(Y) (Examples 1, 3), the heatmaps differ, and MI > 0.")
print("")
print("#### Conclusion:")
print("Mutual information is fundamentally a measure of dependence between variables, expressed")
print("as the KL divergence between their joint distribution and the distribution assuming independence.")
print("Since KL divergence is always non-negative, mutual information I(X;Y) must also be non-negative.")
print("It quantifies how much knowing one variable reduces uncertainty about the other.")
print("")
print("Therefore, Statement 3 is TRUE.") 