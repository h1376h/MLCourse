import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from scipy.stats import multivariate_normal
import os
from matplotlib.patches import Ellipse

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_3_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

#########################
# 1. Dataset Generation #
#########################

# Function to generate datasets with different properties
def generate_dataset(n_samples=1000, n_informative=2, class_sep=1.0, random_state=42):
    """Generate a 2D classification dataset with specified properties."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=random_state
    )
    return X, y

# Generate a standard dataset for initial comparison
X, y = generate_dataset(class_sep=2.0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##################################
# 2. Model Training and Analysis #
##################################

# Train a discriminative model (Logistic Regression)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_test_pred = lr_model.predict(X_test)
lr_test_prob = lr_model.predict_proba(X_test)
lr_accuracy = accuracy_score(y_test, lr_test_pred)
lr_loss = log_loss(y_test, lr_test_prob)

# Train a generative model (Linear Discriminant Analysis)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
lda_test_pred = lda_model.predict(X_test)
lda_test_prob = lda_model.predict_proba(X_test)
lda_accuracy = accuracy_score(y_test, lda_test_pred)
lda_loss = log_loss(y_test, lda_test_prob)

# Extract parameters estimated by LDA (as it is a generative model)
class_means = lda_model.means_
class_priors = lda_model.priors_

# Get the class-specific covariance matrices
X_c0 = X_train[y_train == 0]
X_c1 = X_train[y_train == 1]
cov_c0 = np.cov(X_c0, rowvar=False)
cov_c1 = np.cov(X_c1, rowvar=False)

# Calculate common/pooled covariance for LDA
n0 = X_c0.shape[0]
n1 = X_c1.shape[0]
common_cov = ((n0 - 1) * cov_c0 + (n1 - 1) * cov_c1) / (n0 + n1 - 2)

# Function to manually calculate posterior for LDA (to demonstrate generative approach)
def lda_posterior(X, class_means, class_priors, common_cov):
    """Calculate posterior probabilities using Bayes' rule with Gaussian densities."""
    n_samples = X.shape[0]
    n_classes = len(class_means)
    posteriors = np.zeros((n_samples, n_classes))
    
    # Calculate likelihood for each class
    for c in range(n_classes):
        mvn = multivariate_normal(mean=class_means[c], cov=common_cov)
        likelihood = mvn.pdf(X)
        posteriors[:, c] = likelihood * class_priors[c]
    
    # Normalize to get posteriors
    posteriors = posteriors / np.sum(posteriors, axis=1)[:, np.newaxis]
    return posteriors

# Calculate manual posteriors for LDA
manual_lda_probs = lda_posterior(X_test, class_means, class_priors, common_cov)
manual_lda_loss = log_loss(y_test, manual_lda_probs)

###########################################
# 3. Visualization of Decision Boundaries #
###########################################

def plot_decision_boundary(X, y, models, model_names, title, filename):
    """Plot decision boundaries for multiple models on the same dataset."""
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    plt.figure(figsize=(12, 5))
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        plt.subplot(1, 2, i+1)
        
        # Predict class labels for the grid
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
        
        # Plot the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"{name}")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

# Standard decision boundary plot
plot_decision_boundary(
    X, y, 
    [lr_model, lda_model], 
    ['Logistic Regression\n(Discriminative)', 'LDA\n(Generative)'],
    'Decision Boundaries: Discriminative vs Generative',
    'decision_boundaries.png'
)

# Add covariance ellipse to a plot
def add_covariance_ellipse(ax, mean, cov, color, alpha=0.3, label=None):
    """Add an ellipse representing the covariance matrix to the specified axis."""
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Calculate the angle of the ellipse
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    
    # Create the ellipse patch (for 95% confidence interval)
    ellipse = Ellipse(xy=mean, width=2*2*np.sqrt(evals[0]), height=2*2*np.sqrt(evals[1]),
                      angle=angle, facecolor=color, alpha=alpha, edgecolor=color, label=label)
    ax.add_patch(ellipse)

# Visualize how the models represent the data distribution
def plot_data_representation(X, y):
    """Visualize how discriminative and generative models represent the data."""
    plt.figure(figsize=(12, 5))
    
    # Separate data by class
    X_c0 = X[y == 0]
    X_c1 = X[y == 1]
    
    # Calculate class means
    mean_c0 = np.mean(X_c0, axis=0)
    mean_c1 = np.mean(X_c1, axis=0)
    
    # Calculate class covariances
    cov_c0 = np.cov(X_c0, rowvar=False)
    cov_c1 = np.cov(X_c1, rowvar=False)
    
    # LDA assumption: classes share the same covariance
    pooled_cov = ((len(X_c0) - 1) * cov_c0 + (len(X_c1) - 1) * cov_c1) / (len(X_c0) + len(X_c1) - 2)
    
    # 1. Discriminative approach (Logistic Regression)
    ax1 = plt.subplot(1, 2, 1)
    
    # Plot the data points
    plt.scatter(X_c0[:, 0], X_c0[:, 1], c='red', label='Class 0', alpha=0.6, edgecolor='k')
    plt.scatter(X_c1[:, 0], X_c1[:, 1], c='blue', label='Class 1', alpha=0.6, edgecolor='k')
    
    # Plot the decision boundary of Logistic Regression
    # Calculate the decision boundary explicitly
    coef = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    
    # Plot the decision boundary as a line
    boundary_x = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
    boundary_y = -(coef[0] * boundary_x + intercept) / coef[1]
    
    plt.plot(boundary_x, boundary_y, 'k-', label='Decision Boundary')
    
    plt.title('Discriminative Approach (Logistic Regression)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # 2. Generative approach (LDA)
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot the data points
    plt.scatter(X_c0[:, 0], X_c0[:, 1], c='red', label='Class 0', alpha=0.6, edgecolor='k')
    plt.scatter(X_c1[:, 0], X_c1[:, 1], c='blue', label='Class 1', alpha=0.6, edgecolor='k')
    
    # Plot class means
    plt.scatter(mean_c0[0], mean_c0[1], c='darkred', s=100, marker='X', label='Mean Class 0')
    plt.scatter(mean_c1[0], mean_c1[1], c='darkblue', s=100, marker='X', label='Mean Class 1')
    
    # Plot covariance ellipses for the generative model (LDA uses pooled covariance)
    add_covariance_ellipse(ax2, mean_c0, pooled_cov, 'red', alpha=0.3, label='Pooled Covariance')
    add_covariance_ellipse(ax2, mean_c1, pooled_cov, 'blue', alpha=0.3)
    
    # Plot the LDA decision boundary
    coef = lda_model.coef_[0]
    intercept = lda_model.intercept_[0]
    boundary_x = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
    boundary_y = -(coef[0] * boundary_x + intercept) / coef[1]
    
    plt.plot(boundary_x, boundary_y, 'k-', label='Decision Boundary')
    
    plt.title('Generative Approach (LDA)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_representation.png'), dpi=300, bbox_inches='tight')

plot_data_representation(X, y)

#################################################
# 4. Performance in Limited Data Scenario       #
#################################################

def compare_performance_limited_data():
    """Compare how models perform with limited training data."""
    # Generate a larger test set
    X_large, y_large = generate_dataset(n_samples=5000, class_sep=2.0, random_state=43)
    
    # Create arrays to store results
    train_sizes = np.array([20, 50, 100, 200, 500, 1000])
    lr_accuracies = []
    lda_accuracies = []
    
    # Train and evaluate models with different training set sizes
    for size in train_sizes:
        # Generate a training set of the specified size
        X_small, y_small = generate_dataset(n_samples=size, class_sep=2.0, random_state=42)
        
        # Train models
        lr = LogisticRegression(random_state=42).fit(X_small, y_small)
        lda = LinearDiscriminantAnalysis().fit(X_small, y_small)
        
        # Evaluate models
        lr_accuracies.append(accuracy_score(y_large, lr.predict(X_large)))
        lda_accuracies.append(accuracy_score(y_large, lda.predict(X_large)))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, lr_accuracies, 'o-', color='red', label='Logistic Regression')
    plt.plot(train_sizes, lda_accuracies, 'o-', color='blue', label='LDA')
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Model Performance with Limited Training Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'limited_data_performance.png'), dpi=300, bbox_inches='tight')
    
    return train_sizes, lr_accuracies, lda_accuracies

# Compare performance with limited data
train_sizes, lr_accuracies, lda_accuracies = compare_performance_limited_data()

#################################################
# 5. Performance with Distribution Shift        #
#################################################

def compare_performance_distribution_shift():
    """Compare how models perform when test data distribution shifts."""
    # Train on original distribution
    X_train, y_train = generate_dataset(n_samples=1000, class_sep=2.0, random_state=42)
    
    # Original test set (same distribution)
    X_test_original, y_test_original = generate_dataset(n_samples=500, class_sep=2.0, random_state=43)
    
    # Shifted test sets (different separations)
    test_seps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    lr_accuracies = []
    lda_accuracies = []
    
    # Train models on original distribution
    lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    
    # Test on various shifted distributions
    for sep in test_seps:
        # Generate shifted test data
        X_test_shifted, y_test_shifted = generate_dataset(n_samples=500, class_sep=sep, random_state=44)
        
        # Evaluate models
        lr_accuracies.append(accuracy_score(y_test_shifted, lr.predict(X_test_shifted)))
        lda_accuracies.append(accuracy_score(y_test_shifted, lda.predict(X_test_shifted)))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(test_seps, lr_accuracies, 'o-', color='red', label='Logistic Regression')
    plt.plot(test_seps, lda_accuracies, 'o-', color='blue', label='LDA')
    plt.axvline(x=2.0, color='gray', linestyle='--', label='Training Distribution')
    plt.grid(True)
    plt.xlabel('Class Separation in Test Data')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Model Performance with Distribution Shift')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distribution_shift_performance.png'), dpi=300, bbox_inches='tight')
    
    return test_seps, lr_accuracies, lda_accuracies

# Compare performance with distribution shift
test_seps, lr_shift_accuracies, lda_shift_accuracies = compare_performance_distribution_shift()

#################################################
# 6. Sampling from Generative Model             #
#################################################

def sample_from_generative_model():
    """Demonstrate that generative models can generate new data samples."""
    # Extract parameters from the trained LDA model
    mean_c0 = class_means[0]
    mean_c1 = class_means[1]
    cov = common_cov  # LDA uses a common covariance matrix
    
    # Sample from class distributions
    n_samples = 200
    samples_c0 = np.random.multivariate_normal(mean_c0, cov, n_samples)
    samples_c1 = np.random.multivariate_normal(mean_c1, cov, n_samples)
    
    # Plot original data and sampled data
    plt.figure(figsize=(12, 5))
    
    # Plot original data
    plt.subplot(1, 2, 1)
    X_c0 = X[y == 0]
    X_c1 = X[y == 1]
    
    plt.scatter(X_c0[:, 0], X_c0[:, 1], c='red', label='Class 0', alpha=0.6, edgecolor='k')
    plt.scatter(X_c1[:, 0], X_c1[:, 1], c='blue', label='Class 1', alpha=0.6, edgecolor='k')
    
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot sampled data
    plt.subplot(1, 2, 2)
    plt.scatter(samples_c0[:, 0], samples_c0[:, 1], c='red', label='Generated Class 0', alpha=0.6, edgecolor='k')
    plt.scatter(samples_c1[:, 0], samples_c1[:, 1], c='blue', label='Generated Class 1', alpha=0.6, edgecolor='k')
    
    plt.title('Data Generated from LDA Model')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generative_sampling.png'), dpi=300, bbox_inches='tight')

# Generate samples from the generative model
sample_from_generative_model()

#################################################
# 7. Handling Missing Features                  #
#################################################

def compare_with_missing_features():
    """Compare model performance when some features are missing at test time."""
    # Train models on the full dataset
    X_train_full, y_train = generate_dataset(n_samples=1000, class_sep=2.0, random_state=42)
    X_test_full, y_test = generate_dataset(n_samples=500, class_sep=2.0, random_state=43)
    
    # Train models on full data
    lr = LogisticRegression(random_state=42).fit(X_train_full, y_train)
    lda = LinearDiscriminantAnalysis().fit(X_train_full, y_train)
    
    # Evaluate on full data
    lr_acc_full = accuracy_score(y_test, lr.predict(X_test_full))
    lda_acc_full = accuracy_score(y_test, lda.predict(X_test_full))
    
    # Create test datasets with only one feature
    X_test_feature1 = np.copy(X_test_full)
    X_test_feature1[:, 1] = 0  # Zero out feature 2
    
    X_test_feature2 = np.copy(X_test_full)
    X_test_feature2[:, 0] = 0  # Zero out feature 1
    
    # Evaluate on partial features
    # For discriminative model, we can't easily handle missing features without re-training
    # We just use the model with zeroed features as an approximation
    lr_acc_feature1 = accuracy_score(y_test, lr.predict(X_test_feature1))
    lr_acc_feature2 = accuracy_score(y_test, lr.predict(X_test_feature2))
    
    lda_acc_feature1 = accuracy_score(y_test, lda.predict(X_test_feature1))
    lda_acc_feature2 = accuracy_score(y_test, lda.predict(X_test_feature2))
    
    # Plot the results
    feature_scenarios = ['Full Features', 'Only Feature 1', 'Only Feature 2']
    lr_accuracies = [lr_acc_full, lr_acc_feature1, lr_acc_feature2]
    lda_accuracies = [lda_acc_full, lda_acc_feature1, lda_acc_feature2]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_scenarios))
    width = 0.35
    
    plt.bar(x - width/2, lr_accuracies, width, label='Logistic Regression', color='red', alpha=0.7)
    plt.bar(x + width/2, lda_accuracies, width, label='LDA', color='blue', alpha=0.7)
    
    plt.xlabel('Available Features')
    plt.ylabel('Accuracy')
    plt.title('Model Performance with Missing Features')
    plt.xticks(x, feature_scenarios)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'missing_features_performance.png'), dpi=300, bbox_inches='tight')
    
    return feature_scenarios, lr_accuracies, lda_accuracies

# Compare performance with missing features
feature_scenarios, lr_missing_accuracies, lda_missing_accuracies = compare_with_missing_features()

#################################################
# 8. Output Summary                             #
#################################################

# Print summary of results
print(f"Results saved to: {save_dir}")
print("\nModel Performance Summary:")
print(f"Logistic Regression (Discriminative) Accuracy: {lr_accuracy:.4f}, Log Loss: {lr_loss:.4f}")
print(f"LDA (Generative) Accuracy: {lda_accuracy:.4f}, Log Loss: {lda_loss:.4f}")
print(f"Manual LDA Posterior Calculation Log Loss: {manual_lda_loss:.4f}")

print("\nPerformance with Limited Training Data:")
print("Training Sizes:", train_sizes)
print("LR Accuracies:", [f"{acc:.4f}" for acc in lr_accuracies])
print("LDA Accuracies:", [f"{acc:.4f}" for acc in lda_accuracies])

print("\nPerformance with Distribution Shift:")
print("Test Separations:", test_seps)
print("LR Accuracies:", [f"{acc:.4f}" for acc in lr_shift_accuracies])
print("LDA Accuracies:", [f"{acc:.4f}" for acc in lda_shift_accuracies])

print("\nPerformance with Missing Features:")
print("Feature Scenarios:", feature_scenarios)
print("LR Accuracies:", [f"{acc:.4f}" for acc in lr_missing_accuracies])
print("LDA Accuracies:", [f"{acc:.4f}" for acc in lda_missing_accuracies])

print("\nKey Insights:")
print("1. Discriminative models directly model the decision boundary.")
print("2. Generative models model the class distributions and apply Bayes' rule.")
print("3. Generative models typically perform better with limited data.")
print("4. Generative models can generate new data samples.")
print("5. Generative models can handle missing features more naturally.")
print("6. Discriminative models often achieve better accuracy when training data is abundant.") 