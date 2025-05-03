import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_19")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("Question 19: LDA for Credit Approval")
print("====================================")
print("This solution demonstrates a step-by-step implementation of Linear Discriminant Analysis")
print("for a credit approval classification problem.\n")

# Dataset from the problem
# Income (thousands), Debt-to-Income (%), Credit Approved (1/0)
data = np.array([
    [65, 28, 1],
    [50, 32, 0],
    [79, 22, 1],
    [48, 40, 0],
    [95, 18, 1],
    [36, 36, 0],
    [72, 30, 1],
    [60, 34, 0],
    [85, 24, 1],
    [42, 38, 0]
])

# Display the dataset in a more readable format
print("Dataset:")
print("-" * 50)
print("| Income ($K) | Debt-to-Income (%) | Credit Approved |")
print("|" + "-" * 13 + "|" + "-" * 21 + "|" + "-" * 17 + "|")
for row in data:
    print(f"| {row[0]:11.0f} | {row[1]:19.0f} | {int(row[2]):15d} |")
print("-" * 50)

# Extract features and labels
X = data[:, :2]  # Income and Debt-to-Income ratio
y = data[:, 2]   # Credit approved (1) or denied (0)

# Step 1: Calculate class means
print("\nStep 1: Calculate class means")
print("-" * 40)

# Split the data by class
print("First, we separate the data points by class:")
X_approved = X[y == 1]  # Class 1 (approved)
X_denied = X[y == 0]    # Class 0 (denied)

print("\nApproved applications (Class 1):")
for i, x in enumerate(X_approved):
    print(f"    Sample {i+1}: Income = ${x[0]}K, Debt-to-Income = {x[1]}%")

print("\nDenied applications (Class 0):")
for i, x in enumerate(X_denied):
    print(f"    Sample {i+1}: Income = ${x[0]}K, Debt-to-Income = {x[1]}%")

# Calculate means for each class manually to show the process
print("\nCalculating class means:")
print("For the approved class (y=1), we compute:")

# Manual calculation for approved applications
sum_income_approved = sum(X_approved[:, 0])
sum_dti_approved = sum(X_approved[:, 1])
n_approved = X_approved.shape[0]

print(f"  Mean Income = (65 + 79 + 95 + 72 + 85) / 5 = {sum_income_approved} / 5 = {sum_income_approved/n_approved:.1f}")
print(f"  Mean Debt-to-Income = (28 + 22 + 18 + 30 + 24) / 5 = {sum_dti_approved} / 5 = {sum_dti_approved/n_approved:.1f}")

# Manual calculation for denied applications
sum_income_denied = sum(X_denied[:, 0])
sum_dti_denied = sum(X_denied[:, 1])
n_denied = X_denied.shape[0]

print("\nFor the denied class (y=0), we compute:")
print(f"  Mean Income = (50 + 48 + 36 + 60 + 42) / 5 = {sum_income_denied} / 5 = {sum_income_denied/n_denied:.1f}")
print(f"  Mean Debt-to-Income = (32 + 40 + 36 + 34 + 38) / 5 = {sum_dti_denied} / 5 = {sum_dti_denied/n_denied:.1f}")

# Now use NumPy for the actual values
mean_approved = np.mean(X_approved, axis=0)
mean_denied = np.mean(X_denied, axis=0)

print("\nClass means computed with NumPy:")
print(f"  Class 1 (Approved) Mean: μ₁ = [{mean_approved[0]:.1f}, {mean_approved[1]:.1f}]")
print(f"  Class 0 (Denied) Mean: μ₀ = [{mean_denied[0]:.1f}, {mean_denied[1]:.1f}]")

# Overall mean
mean_overall = np.mean(X, axis=0)
print(f"\nOverall Mean: μ = [{mean_overall[0]:.1f}, {mean_overall[1]:.1f}]")

# Step 2: Calculate the pooled within-class covariance matrix
print("\nStep 2: Calculate the pooled within-class covariance matrix")
print("-" * 60)

# Calculate covariance matrices for each class
print("First, we calculate the covariance matrix for each class separately.")
print("\nFor the approved class (y=1):")

# Manual calculation of covariance for approved class (showing the process)
X_approved_centered = X_approved - mean_approved
print("Centered data (X - μ₁):")
for i, x in enumerate(X_approved_centered):
    print(f"  Sample {i+1}: [{x[0]:.1f}, {x[1]:.1f}]")

print("\nComputing the elements of the covariance matrix:")
cov_approved_00 = np.sum(X_approved_centered[:, 0] * X_approved_centered[:, 0]) / (n_approved - 1)
cov_approved_01 = np.sum(X_approved_centered[:, 0] * X_approved_centered[:, 1]) / (n_approved - 1)
cov_approved_10 = cov_approved_01  # Symmetric
cov_approved_11 = np.sum(X_approved_centered[:, 1] * X_approved_centered[:, 1]) / (n_approved - 1)

print(f"  S₁[0,0] = Variance(Income) = {cov_approved_00:.1f}")
print(f"  S₁[0,1] = S₁[1,0] = Covariance(Income, Debt-to-Income) = {cov_approved_01:.1f}")
print(f"  S₁[1,1] = Variance(Debt-to-Income) = {cov_approved_11:.1f}")

# Using NumPy for the actual values
cov_approved = np.cov(X_approved, rowvar=False)
print("\nCovariance matrix for approved class using NumPy:")
print(f"  S₁ = [   {cov_approved[0,0]:.1f}    {cov_approved[0,1]:.1f}   ]")
print(f"       [   {cov_approved[1,0]:.1f}    {cov_approved[1,1]:.1f}   ]")

print("\nFor the denied class (y=0):")
# Manual calculation of covariance for denied class (showing the process)
X_denied_centered = X_denied - mean_denied
print("Centered data (X - μ₀):")
for i, x in enumerate(X_denied_centered):
    print(f"  Sample {i+1}: [{x[0]:.1f}, {x[1]:.1f}]")

print("\nComputing the elements of the covariance matrix:")
cov_denied_00 = np.sum(X_denied_centered[:, 0] * X_denied_centered[:, 0]) / (n_denied - 1)
cov_denied_01 = np.sum(X_denied_centered[:, 0] * X_denied_centered[:, 1]) / (n_denied - 1)
cov_denied_10 = cov_denied_01  # Symmetric
cov_denied_11 = np.sum(X_denied_centered[:, 1] * X_denied_centered[:, 1]) / (n_denied - 1)

print(f"  S₀[0,0] = Variance(Income) = {cov_denied_00:.1f}")
print(f"  S₀[0,1] = S₀[1,0] = Covariance(Income, Debt-to-Income) = {cov_denied_01:.1f}")
print(f"  S₀[1,1] = Variance(Debt-to-Income) = {cov_denied_11:.1f}")

# Using NumPy for the actual values
cov_denied = np.cov(X_denied, rowvar=False)
print("\nCovariance matrix for denied class using NumPy:")
print(f"  S₀ = [   {cov_denied[0,0]:.1f}    {cov_denied[0,1]:.1f}   ]")
print(f"       [   {cov_denied[1,0]:.1f}    {cov_denied[1,1]:.1f}   ]")

# Number of samples in each class
n_total = X.shape[0]

# Pooled covariance (weighted average of class covariances)
print("\nNow, we calculate the pooled within-class covariance matrix:")
print("  S_W = ((n₁-1)·S₁ + (n₀-1)·S₀) / (n₁+n₀-2)")
print(f"      = (({n_approved}-1)·S₁ + ({n_denied}-1)·S₀) / ({n_approved}+{n_denied}-2)")
print(f"      = (4·S₁ + 4·S₀) / 8")

print("\nSubstituting the covariance matrices:")
print(f"  S_W = 4·[   {cov_approved[0,0]:.1f}    {cov_approved[0,1]:.1f}   ] + 4·[   {cov_denied[0,0]:.1f}    {cov_denied[0,1]:.1f}   ] / 8")
print(f"       [   {cov_approved[1,0]:.1f}    {cov_approved[1,1]:.1f}   ]    [   {cov_denied[1,0]:.1f}    {cov_denied[1,1]:.1f}   ]")

S_W = ((n_approved - 1) * cov_approved + (n_denied - 1) * cov_denied) / (n_total - 2)

print("\nPooled Within-Class Covariance Matrix (S_W):")
print(f"  S_W = [   {S_W[0,0]:.1f}    {S_W[0,1]:.1f}   ]")
print(f"        [   {S_W[1,0]:.1f}    {S_W[1,1]:.1f}   ]")

# Step 3: Find the between-class scatter matrix
print("\nStep 3: Calculate the between-class scatter matrix")
print("-" * 60)

# Calculate class difference vector
mean_diff = mean_approved - mean_denied
print("First, we compute the difference between class means:")
print(f"  μ₁ - μ₀ = [{mean_approved[0]:.1f}, {mean_approved[1]:.1f}] - [{mean_denied[0]:.1f}, {mean_denied[1]:.1f}]")
print(f"         = [{mean_diff[0]:.1f}, {mean_diff[1]:.1f}]")

# Transform to column vector for matrix multiplication
mean_diff_col = mean_diff.reshape(-1, 1)
mean_diff_row = mean_diff.reshape(1, -1)

print("\nWe compute the outer product (μ₁ - μ₀)(μ₁ - μ₀)ᵀ:")
outer_product = np.dot(mean_diff_col, mean_diff_row)
print(f"  (μ₁ - μ₀)(μ₁ - μ₀)ᵀ = [{mean_diff[0]:.1f}] · [{mean_diff[0]:.1f} {mean_diff[1]:.1f}]")
print(f"                        [{mean_diff[1]:.1f}]")
print(f"                      = [{outer_product[0,0]:.1f}  {outer_product[0,1]:.1f}]")
print(f"                        [{outer_product[1,0]:.1f}  {outer_product[1,1]:.1f}]")

# Between-class scatter matrix
print("\nNow, we calculate the between-class scatter matrix:")
print("  S_B = (n₁·n₀/n)·(μ₁ - μ₀)(μ₁ - μ₀)ᵀ")
print(f"      = ({n_approved}·{n_denied}/{n_total})·(μ₁ - μ₀)(μ₁ - μ₀)ᵀ")
print(f"      = ({n_approved*n_denied}/{n_total})·(μ₁ - μ₀)(μ₁ - μ₀)ᵀ")
print(f"      = {n_approved*n_denied/n_total:.1f}·(μ₁ - μ₀)(μ₁ - μ₀)ᵀ")

S_B = n_approved * n_denied / n_total * np.dot(mean_diff_col, mean_diff_row)

print("\nBetween-Class Scatter Matrix (S_B):")
print(f"  S_B = [   {S_B[0,0]:.1f}    {S_B[0,1]:.1f}   ]")
print(f"        [   {S_B[1,0]:.1f}    {S_B[1,1]:.1f}   ]")

# Verify the rank of S_B (should be 1 for binary classification)
rank_S_B = np.linalg.matrix_rank(S_B)
print(f"\nRank of S_B: {rank_S_B} (expect 1 for binary classification)")
print("The rank is 1 because in binary classification, S_B is the outer product of a single vector.")

# Step 4: Find the optimal projection direction
print("\nStep 4: Find the optimal projection direction")
print("-" * 60)

# For binary classification, we can directly compute the projection direction
print("In LDA, we project the data onto the direction that maximizes the between-class")
print("variance while minimizing the within-class variance. For binary classification,")
print("this direction is given by w = S_W^(-1)(μ₁ - μ₀).")

# Calculate the inverse of S_W
S_W_inv = np.linalg.inv(S_W)
print("\nFirst, we calculate the inverse of the pooled within-class covariance matrix:")
print(f"  S_W^(-1) = [   {S_W_inv[0,0]:.4f}    {S_W_inv[0,1]:.4f}   ]")
print(f"             [   {S_W_inv[1,0]:.4f}    {S_W_inv[1,1]:.4f}   ]")

# Calculate the projection direction
print("\nThen, we compute the optimal projection direction:")
print(f"  w = S_W^(-1)(μ₁ - μ₀)")
print(f"    = [   {S_W_inv[0,0]:.4f}    {S_W_inv[0,1]:.4f}   ] · [{mean_diff[0]:.1f}]")
print(f"      [   {S_W_inv[1,0]:.4f}    {S_W_inv[1,1]:.4f}   ]   [{mean_diff[1]:.1f}]")

w = np.dot(S_W_inv, mean_diff)

print(f"    = [{S_W_inv[0,0]:.4f}·{mean_diff[0]:.1f} + {S_W_inv[0,1]:.4f}·{mean_diff[1]:.1f}]")
print(f"      [{S_W_inv[1,0]:.4f}·{mean_diff[0]:.1f} + {S_W_inv[1,1]:.4f}·{mean_diff[1]:.1f}]")
print(f"    = [{w[0]:.4f}]")
print(f"      [{w[1]:.4f}]")

# Normalize w to unit length for visualization and interpretation
w_norm = w / np.linalg.norm(w)
print("\nNormalized projection direction (unit vector):")
print(f"  w_norm = w / ||w|| = [{w[0]:.4f}, {w[1]:.4f}] / {np.linalg.norm(w):.4f}")
print(f"        = [{w_norm[0]:.4f}, {w_norm[1]:.4f}]")

print(f"\nOptimal LDA projection direction: w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"Normalized direction: w_norm = [{w_norm[0]:.4f}, {w_norm[1]:.4f}]")

# Alternative approach using eigendecomposition
print("\nAlternative approach: Using eigendecomposition of S_W^(-1)S_B")
print("For binary classification, the eigenvector corresponding to the largest")
print("eigenvalue of S_W^(-1)S_B points in the same direction as S_W^(-1)(μ₁ - μ₀).")

# Compute S_W^(-1)S_B
SW_inv_SB = np.dot(S_W_inv, S_B)
print("\nS_W^(-1)S_B:")
print(f"  S_W^(-1)S_B = [   {SW_inv_SB[0,0]:.4f}    {SW_inv_SB[0,1]:.4f}   ]")
print(f"                [   {SW_inv_SB[1,0]:.4f}    {SW_inv_SB[1,1]:.4f}   ]")

# Find eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(SW_inv_SB)

# Sort eigenvectors by decreasing eigenvalues
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

print("\nEigenvalues of S_W^(-1)S_B:")
for i, val in enumerate(eig_vals.real):
    print(f"  λ{i+1} = {val:.8f}")

print("\nEigenvectors of S_W^(-1)S_B:")
for i, vec in enumerate(eig_vecs.T):
    print(f"  v{i+1} = [{vec[0].real:.4f}, {vec[1].real:.4f}]")

# The first eigenvector is the LDA projection direction
w_eig = eig_vecs[:, 0].real  # Take the real part in case of complex eigenvalues
w_eig_norm = w_eig / np.linalg.norm(w_eig)

print("\nThe first eigenvector (corresponding to the largest eigenvalue) gives the LDA direction:")
print(f"  w_eig = [{w_eig[0]:.4f}, {w_eig[1]:.4f}]")
print(f"  w_eig_norm = [{w_eig_norm[0]:.4f}, {w_eig_norm[1]:.4f}]")

# Verify both methods give same direction (might differ by sign)
dot_product = np.abs(np.dot(w_norm, w_eig_norm))
print(f"\nVerifying both methods give the same direction:")
print(f"  |w_norm · w_eig_norm| = |{np.dot(w_norm, w_eig_norm):.4f}| = {dot_product:.4f}")
print(f"  This is close to 1, confirming both methods yield the same direction (possibly with opposite sign).")

# Step 5: Calculate the threshold for classification
print("\nStep 5: Calculate the threshold for classification")
print("-" * 60)

# Project class means onto w
print("To find the optimal threshold, we project the class means onto the LDA direction w.")
print("The projection of a point x onto w is given by w·x (dot product).")

proj_mean_approved = np.dot(w, mean_approved)
proj_mean_denied = np.dot(w, mean_denied)

print(f"\nProjection of μ₁ (approved class mean) onto w:")
print(f"  w·μ₁ = [{w[0]:.4f}, {w[1]:.4f}] · [{mean_approved[0]:.1f}, {mean_approved[1]:.1f}]")
print(f"       = {w[0]:.4f}·{mean_approved[0]:.1f} + {w[1]:.4f}·{mean_approved[1]:.1f}")
print(f"       = {w[0]*mean_approved[0]:.4f} + {w[1]*mean_approved[1]:.4f}")
print(f"       = {proj_mean_approved:.4f}")

print(f"\nProjection of μ₀ (denied class mean) onto w:")
print(f"  w·μ₀ = [{w[0]:.4f}, {w[1]:.4f}] · [{mean_denied[0]:.1f}, {mean_denied[1]:.1f}]")
print(f"       = {w[0]:.4f}·{mean_denied[0]:.1f} + {w[1]:.4f}·{mean_denied[1]:.1f}")
print(f"       = {w[0]*mean_denied[0]:.4f} + {w[1]*mean_denied[1]:.4f}")
print(f"       = {proj_mean_denied:.4f}")

# For equal priors, the threshold is the midpoint of projected means
threshold_equal_priors = (proj_mean_approved + proj_mean_denied) / 2
print("\nWith equal prior probabilities P(y=1) = P(y=0) = 0.5, the threshold is")
print("the midpoint of the projected class means:")
print(f"  threshold_equal = (w·μ₁ + w·μ₀) / 2")
print(f"                  = ({proj_mean_approved:.4f} + {proj_mean_denied:.4f}) / 2")
print(f"                  = {proj_mean_approved + proj_mean_denied:.4f} / 2")
print(f"                  = {threshold_equal_priors:.4f}")

# Calculate threshold with given priors
# P(y=1) = 0.3 and P(y=0) = 0.7
prior_approved = 0.3
prior_denied = 0.7

print(f"\nHowever, we're given unequal prior probabilities:")
print(f"  P(y=1) = {prior_approved} (approved)")
print(f"  P(y=0) = {prior_denied} (denied)")

print("\nFor Gaussian distributions with shared covariance, the decision boundary in the projected space")
print("includes a term that depends on the prior probabilities:")
print("  threshold = 0.5(w·μ₁ + w·μ₀) + (1/(w·Σ·w))·log(P(y=0)/P(y=1))")

# Calculate w·Σ·w
w_S_w = np.dot(np.dot(w, S_W), w)
print(f"\nCalculating w·Σ·w:")
print(f"  w·Σ·w = w·S_W·w")
print(f"        = [{w[0]:.4f}, {w[1]:.4f}] · [   {S_W[0,0]:.1f}    {S_W[0,1]:.1f}   ] · [{w[0]:.4f}]")
print(f"                         [   {S_W[1,0]:.1f}    {S_W[1,1]:.1f}   ]   [{w[1]:.4f}]")
print(f"        = {w_S_w:.4f}")

# Calculate log(P(y=0)/P(y=1))
log_prior_ratio = np.log(prior_denied / prior_approved)
print(f"\nCalculating log(P(y=0)/P(y=1)):")
print(f"  log(P(y=0)/P(y=1)) = log({prior_denied}/{prior_approved})")
print(f"                      = log({prior_denied/prior_approved:.4f})")
print(f"                      = {log_prior_ratio:.4f}")

# Calculate the threshold with priors
threshold_with_priors = (proj_mean_approved + proj_mean_denied) / 2 + \
                        log_prior_ratio / w_S_w

print(f"\nThreshold with prior probabilities:")
print(f"  threshold = 0.5(w·μ₁ + w·μ₀) + (1/(w·Σ·w))·log(P(y=0)/P(y=1))")
print(f"            = 0.5({proj_mean_approved:.4f} + {proj_mean_denied:.4f}) + (1/{w_S_w:.4f})·{log_prior_ratio:.4f}")
print(f"            = {(proj_mean_approved + proj_mean_denied)/2:.4f} + {log_prior_ratio/w_S_w:.4f}")
print(f"            = {threshold_with_priors:.4f}")

print(f"\nNote that the threshold with priors ({threshold_with_priors:.4f}) is higher than")
print(f"the threshold with equal priors ({threshold_equal_priors:.4f}). This makes sense because")
print(f"the prior probability for approval (0.3) is lower than for denial (0.7), making")
print(f"the decision boundary more selective for approvals.")

# Step 6: Predict class for a new applicant
print("\nStep 6: Predict class for a new applicant")
print("-" * 60)

# New applicant: income $55K, debt-to-income ratio 25%
new_applicant = np.array([55, 25])
print(f"We have a new applicant with the following features:")
print(f"  Income = ${new_applicant[0]}K")
print(f"  Debt-to-Income = {new_applicant[1]}%")

# Project the new point onto w
proj_new = np.dot(w, new_applicant)
print(f"\nProject this applicant onto the LDA direction:")
print(f"  w·x_new = [{w[0]:.4f}, {w[1]:.4f}] · [{new_applicant[0]}, {new_applicant[1]}]")
print(f"          = {w[0]:.4f}·{new_applicant[0]} + {w[1]:.4f}·{new_applicant[1]}")
print(f"          = {w[0]*new_applicant[0]:.4f} + {w[1]*new_applicant[1]:.4f}")
print(f"          = {proj_new:.4f}")

# Classify based on the threshold with priors
print(f"\nCompare the projected value with the threshold (with priors):")
print(f"  Projected value: {proj_new:.4f}")
print(f"  Threshold: {threshold_with_priors:.4f}")

if proj_new > threshold_with_priors:
    predicted_class = 1
    decision = "Approved"
    print(f"  {proj_new:.4f} > {threshold_with_priors:.4f}, so the applicant is classified as Class 1 (Approved)")
else:
    predicted_class = 0
    decision = "Denied"
    print(f"  {proj_new:.4f} < {threshold_with_priors:.4f}, so the applicant is classified as Class 0 (Denied)")

print(f"\nFinal prediction: Credit application is {decision}")
print(f"Note: This is a very close decision, as the projected value ({proj_new:.4f}) is only")
print(f"slightly below the threshold ({threshold_with_priors:.4f}).")

# Step 7: Visualizations
print("\nStep 7: Create visualizations")
print("-" * 40)
print("Creating visualizations to illustrate the LDA analysis...")

# Function to plot confidence ellipses
def plot_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        Ellipse facecolor.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    
    # Compute the standard deviation points
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# 1. Plot the data points with decision boundary
plt.figure(figsize=(12, 10))

# Scatter plot of the data points
plt.scatter(X_approved[:, 0], X_approved[:, 1], c='green', marker='o', s=100, label='Approved')
plt.scatter(X_denied[:, 0], X_denied[:, 1], c='red', marker='x', s=100, label='Denied')

# Label the points
for i, point in enumerate(X_approved):
    plt.annotate(f'A{i+1}', (point[0], point[1]), 
                 xytext=(7, 0), textcoords='offset points', fontsize=12)
for i, point in enumerate(X_denied):
    plt.annotate(f'D{i+1}', (point[0], point[1]), 
                 xytext=(7, 0), textcoords='offset points', fontsize=12)

# Plot the class means
plt.scatter(mean_approved[0], mean_approved[1], c='darkgreen', marker='*', s=200, label='Approved Mean')
plt.scatter(mean_denied[0], mean_denied[1], c='darkred', marker='*', s=200, label='Denied Mean')

# Plot confidence ellipses
plot_confidence_ellipse(X_approved[:, 0], X_approved[:, 1], plt.gca(), n_std=2.0, 
                        edgecolor='green', linestyle='--', alpha=0.3, label='Approved 95% Confidence')
plot_confidence_ellipse(X_denied[:, 0], X_denied[:, 1], plt.gca(), n_std=2.0, 
                        edgecolor='red', linestyle='--', alpha=0.3, label='Denied 95% Confidence')

# Calculate points for the decision boundary line
# The decision boundary is perpendicular to w and passes through the point x0
# where w^T * x0 = threshold
# We can use parametric equation of a line:
# x = x0 + t * v, where v is perpendicular to w

# Range for plotting
min_income, max_income = 30, 100
min_debt, max_debt = 15, 45

# Find two points on the decision boundary by solving w^T * x = threshold
# For x-axis value (income), get the y-axis value (debt-to-income)
x1 = min_income
y1 = (threshold_with_priors - w[0] * x1) / w[1]

x2 = max_income
y2 = (threshold_with_priors - w[0] * x2) / w[1]

# Plot the decision boundary
plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2, label='Decision Boundary')

# Plot the direction of w
# Scale w for visualization
w_scale = 10
midpoint = (mean_approved + mean_denied) / 2
plt.arrow(midpoint[0], midpoint[1], w_scale * w[0], w_scale * w[1], 
          head_width=1.5, head_length=1.5, fc='blue', ec='blue', label='LDA Direction')

# Plot the new applicant
plt.scatter(new_applicant[0], new_applicant[1], c='purple', marker='D', s=100, label='New Applicant')
plt.annotate('New', (new_applicant[0], new_applicant[1]), 
             xytext=(7, 0), textcoords='offset points', fontsize=12)

# Set labels and title
plt.xlabel('Income ($K)', fontsize=14)
plt.ylabel('Debt-to-Income Ratio (%)', fontsize=14)
plt.title('LDA for Credit Approval Decision', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Create a textbox with the results
textbox = f"LDA Results:\n" \
          f"w = [{w[0]:.4f}, {w[1]:.4f}]\n" \
          f"Threshold = {threshold_with_priors:.4f}\n" \
          f"New Applicant Projection = {proj_new:.4f}\n" \
          f"Prediction: {decision}"
          
plt.figtext(0.15, 0.15, textbox, fontsize=12, 
            bbox={"facecolor":"white", "alpha":0.8, "pad":5})

plt.xlim(min_income, max_income)
plt.ylim(min_debt, max_debt)

# Save the plot
print("\nSaving visualization: LDA Credit Approval Decision Plot")
plt.savefig(os.path.join(save_dir, "lda_credit_approval.png"), dpi=300, bbox_inches='tight')

# 2. Plot the projected data points
plt.figure(figsize=(12, 6))

# Project all data points onto w
proj_approved = np.dot(X_approved, w)
proj_denied = np.dot(X_denied, w)

# Create a scatter plot of the projected data
plt.scatter(proj_approved, np.zeros_like(proj_approved) + 0.1, c='green', marker='o', s=100, label='Approved')
plt.scatter(proj_denied, np.zeros_like(proj_denied) - 0.1, c='red', marker='x', s=100, label='Denied')

# Label the points
for i, point in enumerate(proj_approved):
    plt.annotate(f'A{i+1}', (point, 0.1), 
                 xytext=(0, 5), textcoords='offset points', fontsize=12)
for i, point in enumerate(proj_denied):
    plt.annotate(f'D{i+1}', (point, -0.1), 
                 xytext=(0, -15), textcoords='offset points', fontsize=12)

# Plot the projected means
plt.scatter(proj_mean_approved, 0.1, c='darkgreen', marker='*', s=200, label='Approved Mean')
plt.scatter(proj_mean_denied, -0.1, c='darkred', marker='*', s=200, label='Denied Mean')

# Plot the threshold
plt.axvline(x=threshold_with_priors, color='k', linestyle='-', linewidth=2, label='Threshold (with priors)')
plt.axvline(x=threshold_equal_priors, color='k', linestyle='--', linewidth=1, label='Threshold (equal priors)')

# Plot the new applicant's projection
plt.scatter(proj_new, 0, c='purple', marker='D', s=100, label='New Applicant')
plt.annotate('New', (proj_new, 0), 
             xytext=(0, 5), textcoords='offset points', fontsize=12)

# Shade regions
plt.axvspan(threshold_with_priors, plt.xlim()[1], alpha=0.2, color='green', label='Approved Region')
plt.axvspan(plt.xlim()[0], threshold_with_priors, alpha=0.2, color='red', label='Denied Region')

# Set labels and title
plt.xlabel('Projection onto LDA Direction', fontsize=14)
plt.yticks([])  # Hide y-axis ticks
plt.title('LDA Projection for Credit Approval Decision', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Save the projected data plot
print("Saving visualization: LDA Projection Plot")
plt.savefig(os.path.join(save_dir, "lda_projection.png"), dpi=300, bbox_inches='tight')

# 3. Feature importance visualization
plt.figure(figsize=(8, 6))

# Feature names
features = ['Income', 'Debt-to-Income']

# Absolute weights indicate importance
importance = np.abs(w)
normalized_importance = importance / np.sum(importance)

plt.bar(features, normalized_importance, color=['blue', 'orange'])
plt.title('Feature Importance in LDA Direction', fontsize=16)
plt.ylabel('Normalized Absolute Weight', fontsize=14)
plt.ylim(0, 1)

# Add the weights as text
for i, val in enumerate(normalized_importance):
    plt.text(i, val + 0.05, f'{val:.2f}', ha='center', fontsize=12)
    plt.text(i, val/2, f'w = {w[i]:.4f}', ha='center', fontsize=12, color='white')

print("Saving visualization: Feature Importance Plot")
plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')

# Summary of Results
print("\nSummary of Results:")
print("=" * 40)
print(f"1. The class means are:")
print(f"   - Approved: Income = ${mean_approved[0]:.1f}K, Debt-to-Income = {mean_approved[1]:.1f}%")
print(f"   - Denied: Income = ${mean_denied[0]:.1f}K, Debt-to-Income = {mean_denied[1]:.1f}%")
print(f"2. The pooled within-class covariance matrix is:")
print(f"   S_W = [   {S_W[0,0]:.1f}    {S_W[0,1]:.1f}   ]")
print(f"         [   {S_W[1,0]:.1f}    {S_W[1,1]:.1f}   ]")
print(f"3. The between-class scatter matrix is:")
print(f"   S_B = [   {S_B[0,0]:.1f}    {S_B[0,1]:.1f}   ]")
print(f"         [   {S_B[1,0]:.1f}    {S_B[1,1]:.1f}   ]")
print(f"4. The optimal LDA projection direction is:")
print(f"   w = [{w[0]:.4f}, {w[1]:.4f}]")
print(f"5. The classification threshold (with priors P(y=1)=0.3, P(y=0)=0.7) is:")
print(f"   threshold = {threshold_with_priors:.4f}")
print(f"6. For a new applicant with income $55K and debt-to-income 25%:")
print(f"   - Projected value: {proj_new:.4f}")
print(f"   - Prediction: Credit {decision}")
print(f"7. Feature importance in the LDA model:")
print(f"   - Income: {normalized_importance[0]:.2f}")
print(f"   - Debt-to-Income: {normalized_importance[1]:.2f}")
print("\nAll visualizations have been saved to the Images directory.") 