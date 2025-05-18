import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from itertools import combinations
import os
from matplotlib.colors import ListedColormap
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Step 1: Create a synthetic multi-class dataset
def generate_dataset(n_samples=300, n_classes=3, random_state=42):
    """Generate a synthetic dataset with specified number of classes."""
    print(f"Step 1: Generating a synthetic dataset with {n_classes} classes...")
    
    X, y = datasets.make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=2,
        random_state=random_state,
        cluster_std=1.0
    )
    
    # Scale the data for better visualization
    X = X * 1.5
    
    print(f"Dataset shape: X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

# Step 2: Implement One-vs-All (OVA) classification
def train_one_vs_all(X_train, y_train, n_classes, classifier=LogisticRegression):
    """
    Implement One-vs-All classification manually.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes
        classifier: Base classifier to use
        
    Returns:
        List of trained classifiers, one for each class
    """
    print("\nStep 2: Training One-vs-All (OVA) classifiers manually...")
    
    classifiers = []
    
    for i in range(n_classes):
        print(f"  Training classifier for class {i} vs rest...")
        
        # Create binary labels: 1 for current class, 0 for all other classes
        binary_y = (y_train == i).astype(int)
        
        # Train classifier
        clf = classifier(random_state=42)
        clf.fit(X_train, binary_y)
        
        classifiers.append(clf)
    
    print(f"Trained {len(classifiers)} OVA classifiers")
    return classifiers

# Step 3: Implement One-vs-One (OVO) classification
def train_one_vs_one(X_train, y_train, n_classes, classifier=LogisticRegression):
    """
    Implement One-vs-One classification manually.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_classes: Number of classes
        classifier: Base classifier to use
        
    Returns:
        List of tuples (classifier, class_i, class_j) for each pair of classes
    """
    print("\nStep 3: Training One-vs-One (OVO) classifiers manually...")
    
    classifiers = []
    
    # Generate all pairs of classes
    class_pairs = list(combinations(range(n_classes), 2))
    
    for i, j in class_pairs:
        print(f"  Training classifier for class {i} vs class {j}...")
        
        # Select only samples from classes i and j
        mask = (y_train == i) | (y_train == j)
        X_subset = X_train[mask]
        y_subset = y_train[mask]
        
        # Relabel as binary problem: class i -> 0, class j -> 1
        y_binary = (y_subset == j).astype(int)
        
        # Train classifier
        clf = classifier(random_state=42)
        clf.fit(X_subset, y_binary)
        
        classifiers.append((clf, i, j))
    
    print(f"Trained {len(classifiers)} OVO classifiers")
    return classifiers

# Step 4: Prediction functions for OVA and OVO
def predict_one_vs_all(X, classifiers):
    """
    Make predictions using OVA classifiers.
    
    Args:
        X: Features to predict
        classifiers: List of trained OVA classifiers
        
    Returns:
        Predicted class labels
    """
    # Get confidence scores for each class
    n_samples = X.shape[0]
    n_classes = len(classifiers)
    scores = np.zeros((n_samples, n_classes))
    
    for i, clf in enumerate(classifiers):
        # Get decision function values (confidence scores)
        if hasattr(clf, 'decision_function'):
            scores[:, i] = clf.decision_function(X)
        else:
            # For classifiers that don't have decision_function
            scores[:, i] = clf.predict_proba(X)[:, 1]
    
    # Return class with highest confidence score
    return np.argmax(scores, axis=1)

def predict_one_vs_one(X, classifiers, n_classes):
    """
    Make predictions using OVO classifiers.
    
    Args:
        X: Features to predict
        classifiers: List of tuples (classifier, class_i, class_j)
        n_classes: Total number of classes
        
    Returns:
        Predicted class labels
    """
    n_samples = X.shape[0]
    votes = np.zeros((n_samples, n_classes))
    
    for clf, i, j in classifiers:
        # Predict binary outcomes
        predictions = clf.predict(X)
        
        # Add votes: class i gets votes where prediction is 0
        # class j gets votes where prediction is 1
        votes[predictions == 0, i] += 1
        votes[predictions == 1, j] += 1
    
    # Return class with most votes
    return np.argmax(votes, axis=1)

# Step 5: Visualization function for decision boundaries
def plot_decision_boundaries(X, y, classifiers, classifier_type, n_classes):
    """
    Plot decision boundaries of the classifiers.
    
    Args:
        X: Features
        y: True labels
        classifiers: Trained classifiers
        classifier_type: 'OVA' or 'OVO'
        n_classes: Number of classes
    """
    print(f"\nStep 5: Plotting decision boundaries for {classifier_type} approach...")
    
    # Create a mesh grid to visualize decision boundaries
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a mesh grid of points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict classes for all points in the mesh grid
    if classifier_type == 'OVA':
        Z = predict_one_vs_all(grid_points, classifiers)
    else:  # OVO
        Z = predict_one_vs_one(grid_points, classifiers, n_classes)
    
    # Reshape the predictions to match the mesh grid
    Z = Z.reshape(xx.shape)
    
    # Create a colormap for the plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#FFAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF'])
    
    # Plot decision boundaries and scatter points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'Decision Boundaries using {classifier_type} Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'{classifier_type.lower()}_decision_boundaries.png'), 
                dpi=300, bbox_inches='tight')

# Step 6: Compare training and prediction time
def measure_time_complexity(X_train, y_train, X_test, n_classes):
    """
    Measure and compare training and prediction time for OVA and OVO.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        n_classes: Number of classes
        
    Returns:
        Dictionary with timing results
    """
    print("\nStep 6: Measuring time complexity of OVA and OVO approaches...")
    
    results = {}
    
    # Measure OVA time
    start_time = time.time()
    ova_classifiers = train_one_vs_all(X_train, y_train, n_classes)
    ova_train_time = time.time() - start_time
    
    start_time = time.time()
    _ = predict_one_vs_all(X_test, ova_classifiers)
    ova_predict_time = time.time() - start_time
    
    # Measure OVO time
    start_time = time.time()
    ovo_classifiers = train_one_vs_one(X_train, y_train, n_classes)
    ovo_train_time = time.time() - start_time
    
    start_time = time.time()
    _ = predict_one_vs_one(X_test, ovo_classifiers, n_classes)
    ovo_predict_time = time.time() - start_time
    
    results['ova_train_time'] = ova_train_time
    results['ova_predict_time'] = ova_predict_time
    results['ovo_train_time'] = ovo_train_time
    results['ovo_predict_time'] = ovo_predict_time
    
    print(f"  OVA - Training time: {ova_train_time:.4f}s, Prediction time: {ova_predict_time:.4f}s")
    print(f"  OVO - Training time: {ovo_train_time:.4f}s, Prediction time: {ovo_predict_time:.4f}s")
    
    return results

# Step 7: Calculate and compare accuracy
def evaluate_accuracy(X_train, y_train, X_test, y_test, n_classes):
    """
    Calculate and compare accuracy of OVA and OVO approaches.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_classes: Number of classes
        
    Returns:
        Dictionary with accuracy results
    """
    print("\nStep 7: Evaluating accuracy of OVA and OVO approaches...")
    
    results = {}
    
    # Train OVA classifiers
    ova_classifiers = train_one_vs_all(X_train, y_train, n_classes)
    
    # Train OVO classifiers
    ovo_classifiers = train_one_vs_one(X_train, y_train, n_classes)
    
    # Make predictions
    y_pred_ova = predict_one_vs_all(X_test, ova_classifiers)
    y_pred_ovo = predict_one_vs_one(X_test, ovo_classifiers, n_classes)
    
    # Calculate accuracy
    ova_accuracy = accuracy_score(y_test, y_pred_ova)
    ovo_accuracy = accuracy_score(y_test, y_pred_ovo)
    
    results['ova_accuracy'] = ova_accuracy
    results['ovo_accuracy'] = ovo_accuracy
    
    print(f"  OVA Accuracy: {ova_accuracy:.4f}")
    print(f"  OVO Accuracy: {ovo_accuracy:.4f}")
    
    return results

# Step 8: Calculate the number of classifiers needed
def calculate_num_classifiers(n_classes):
    """
    Calculate number of classifiers needed for OVA and OVO approaches.
    
    Args:
        n_classes: Number of classes
        
    Returns:
        Dictionary with number of classifiers
    """
    print("\nStep 8: Calculating number of classifiers needed...")
    
    ova_num_classifiers = n_classes
    ovo_num_classifiers = n_classes * (n_classes - 1) // 2
    
    print(f"  For {n_classes} classes:")
    print(f"  OVA requires {ova_num_classifiers} classifiers")
    print(f"  OVO requires {ovo_num_classifiers} classifiers")
    
    # Calculate for 10 classes specifically
    n_classes_10 = 10
    ova_num_classifiers_10 = n_classes_10
    ovo_num_classifiers_10 = n_classes_10 * (n_classes_10 - 1) // 2
    
    print(f"\n  For 10 classes:")
    print(f"  OVA requires {ova_num_classifiers_10} classifiers")
    print(f"  OVO requires {ovo_num_classifiers_10} classifiers")
    
    return {
        'ova_num_classifiers': ova_num_classifiers,
        'ovo_num_classifiers': ovo_num_classifiers,
        'ova_num_classifiers_10': ova_num_classifiers_10,
        'ovo_num_classifiers_10': ovo_num_classifiers_10
    }

# Step 9: Compare training with imbalanced data
def compare_imbalance_effect(n_classes=3, random_state=42):
    """
    Compare how OVA and OVO handle imbalanced data.
    
    Args:
        n_classes: Number of classes
        random_state: Random seed
        
    Returns:
        Dictionary with imbalance comparison results
    """
    print("\nStep 9: Comparing OVA and OVO on imbalanced data...")
    
    # Generate balanced dataset
    X_balanced, y_balanced = datasets.make_blobs(
        n_samples=300,
        centers=n_classes,
        n_features=2,
        random_state=random_state,
        cluster_std=1.0
    )
    X_balanced = X_balanced * 1.5
    
    # Generate imbalanced dataset
    samples_per_class = [50, 200, 50]  # Imbalanced class distribution
    X_imbalanced = []
    y_imbalanced = []
    
    for i in range(n_classes):
        X_class, y_class = datasets.make_blobs(
            n_samples=samples_per_class[i],
            centers=1,
            n_features=2,
            random_state=random_state + i,
            center_box=(i*4, i*4 + 2),
            cluster_std=1.0
        )
        X_imbalanced.append(X_class)
        y_imbalanced.extend([i] * samples_per_class[i])
    
    X_imbalanced = np.vstack(X_imbalanced)
    y_imbalanced = np.array(y_imbalanced)
    
    # Split data
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=random_state)
    
    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imbalanced, y_imbalanced, test_size=0.3, random_state=random_state)
    
    # Train classifiers on balanced data
    ova_bal = train_one_vs_all(X_train_bal, y_train_bal, n_classes)
    ovo_bal = train_one_vs_one(X_train_bal, y_train_bal, n_classes)
    
    # Train classifiers on imbalanced data
    ova_imb = train_one_vs_all(X_train_imb, y_train_imb, n_classes)
    ovo_imb = train_one_vs_one(X_train_imb, y_train_imb, n_classes)
    
    # Make predictions
    y_pred_ova_bal = predict_one_vs_all(X_test_bal, ova_bal)
    y_pred_ovo_bal = predict_one_vs_one(X_test_bal, ovo_bal, n_classes)
    y_pred_ova_imb = predict_one_vs_all(X_test_imb, ova_imb)
    y_pred_ovo_imb = predict_one_vs_one(X_test_imb, ovo_imb, n_classes)
    
    # Calculate accuracy
    ova_bal_acc = accuracy_score(y_test_bal, y_pred_ova_bal)
    ovo_bal_acc = accuracy_score(y_test_bal, y_pred_ovo_bal)
    ova_imb_acc = accuracy_score(y_test_imb, y_pred_ova_imb)
    ovo_imb_acc = accuracy_score(y_test_imb, y_pred_ovo_imb)
    
    print(f"  Balanced data: OVA acc={ova_bal_acc:.4f}, OVO acc={ovo_bal_acc:.4f}")
    print(f"  Imbalanced data: OVA acc={ova_imb_acc:.4f}, OVO acc={ovo_imb_acc:.4f}")
    print(f"  OVA accuracy drop: {ova_bal_acc - ova_imb_acc:.4f}")
    print(f"  OVO accuracy drop: {ovo_bal_acc - ovo_imb_acc:.4f}")
    
    # Plot imbalanced data decision boundaries
    plt.figure(figsize=(15, 6))
    
    # OVA on imbalanced data
    plt.subplot(1, 2, 1)
    h = 0.02
    x_min, x_max = X_imbalanced[:, 0].min() - 1, X_imbalanced[:, 0].max() + 1
    y_min, y_max = X_imbalanced[:, 1].min() - 1, X_imbalanced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = predict_one_vs_all(grid_points, ova_imb)
    Z = Z.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X_imbalanced[:, 0], X_imbalanced[:, 1], c=y_imbalanced, cmap=cmap_bold, 
                edgecolor='k', s=50, alpha=0.7)
    plt.title(f'OVA on Imbalanced Data (Acc={ova_imb_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # OVO on imbalanced data
    plt.subplot(1, 2, 2)
    Z = predict_one_vs_one(grid_points, ovo_imb, n_classes)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X_imbalanced[:, 0], X_imbalanced[:, 1], c=y_imbalanced, cmap=cmap_bold, 
                edgecolor='k', s=50, alpha=0.7)
    plt.title(f'OVO on Imbalanced Data (Acc={ovo_imb_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'imbalanced_comparison.png'), 
                dpi=300, bbox_inches='tight')
    
    return {
        'ova_balanced_acc': ova_bal_acc,
        'ovo_balanced_acc': ovo_bal_acc,
        'ova_imbalanced_acc': ova_imb_acc,
        'ovo_imbalanced_acc': ovo_imb_acc,
        'ova_acc_drop': ova_bal_acc - ova_imb_acc,
        'ovo_acc_drop': ovo_bal_acc - ovo_imb_acc
    }

# Step 10: Compare with nonlinear decision boundaries
def compare_nonlinear_boundaries(n_classes=3, random_state=42):
    """
    Compare OVA and OVO with nonlinear decision boundaries using SVM with RBF kernel.
    
    Args:
        n_classes: Number of classes
        random_state: Random seed
    """
    print("\nStep 10: Comparing OVA and OVO with nonlinear decision boundaries...")
    
    # Generate dataset with more complex structure
    X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=random_state)
    # Add a third class
    X_third, y_third = datasets.make_circles(n_samples=100, noise=0.1, random_state=random_state, 
                                            factor=0.5)
    X_third = X_third + [3, 0]  # Shift the circles
    y_third = y_third + 2  # Label as class 2
    
    # Combine datasets
    X = np.vstack((X, X_third))
    y = np.concatenate((y, y_third))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    # Train classifiers with linear kernel
    print("  Training with linear kernel...")
    ova_linear = []
    for i in range(n_classes):
        binary_y = (y_train == i).astype(int)
        clf = SVC(kernel='linear', random_state=random_state)
        clf.fit(X_train, binary_y)
        ova_linear.append(clf)
    
    ovo_linear = []
    for i, j in combinations(range(n_classes), 2):
        mask = (y_train == i) | (y_train == j)
        X_subset = X_train[mask]
        y_subset = y_train[mask]
        y_binary = (y_subset == j).astype(int)
        clf = SVC(kernel='linear', random_state=random_state)
        clf.fit(X_subset, y_binary)
        ovo_linear.append((clf, i, j))
    
    # Train classifiers with RBF kernel
    print("  Training with RBF kernel...")
    ova_rbf = []
    for i in range(n_classes):
        binary_y = (y_train == i).astype(int)
        clf = SVC(kernel='rbf', random_state=random_state)
        clf.fit(X_train, binary_y)
        ova_rbf.append(clf)
    
    ovo_rbf = []
    for i, j in combinations(range(n_classes), 2):
        mask = (y_train == i) | (y_train == j)
        X_subset = X_train[mask]
        y_subset = y_train[mask]
        y_binary = (y_subset == j).astype(int)
        clf = SVC(kernel='rbf', random_state=random_state)
        clf.fit(X_subset, y_binary)
        ovo_rbf.append((clf, i, j))
    
    # Make predictions
    y_pred_ova_linear = predict_one_vs_all(X_test, ova_linear)
    y_pred_ovo_linear = predict_one_vs_one(X_test, ovo_linear, n_classes)
    y_pred_ova_rbf = predict_one_vs_all(X_test, ova_rbf)
    y_pred_ovo_rbf = predict_one_vs_one(X_test, ovo_rbf, n_classes)
    
    # Calculate accuracy
    ova_linear_acc = accuracy_score(y_test, y_pred_ova_linear)
    ovo_linear_acc = accuracy_score(y_test, y_pred_ovo_linear)
    ova_rbf_acc = accuracy_score(y_test, y_pred_ova_rbf)
    ovo_rbf_acc = accuracy_score(y_test, y_pred_ovo_rbf)
    
    print(f"  Linear kernel: OVA acc={ova_linear_acc:.4f}, OVO acc={ovo_linear_acc:.4f}")
    print(f"  RBF kernel: OVA acc={ova_rbf_acc:.4f}, OVO acc={ovo_rbf_acc:.4f}")
    
    # Plot decision boundaries for nonlinear case
    plt.figure(figsize=(15, 12))
    
    # Set up the mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # OVA with linear kernel
    plt.subplot(2, 2, 1)
    Z = predict_one_vs_all(grid_points, ova_linear)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.title(f'OVA with Linear Kernel (Acc={ova_linear_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # OVO with linear kernel
    plt.subplot(2, 2, 2)
    Z = predict_one_vs_one(grid_points, ovo_linear, n_classes)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.title(f'OVO with Linear Kernel (Acc={ovo_linear_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # OVA with RBF kernel
    plt.subplot(2, 2, 3)
    Z = predict_one_vs_all(grid_points, ova_rbf)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.title(f'OVA with RBF Kernel (Acc={ova_rbf_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # OVO with RBF kernel
    plt.subplot(2, 2, 4)
    Z = predict_one_vs_one(grid_points, ovo_rbf, n_classes)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.title(f'OVO with RBF Kernel (Acc={ovo_rbf_acc:.4f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nonlinear_comparison.png'), 
                dpi=300, bbox_inches='tight')
    
    return {
        'ova_linear_acc': ova_linear_acc,
        'ovo_linear_acc': ovo_linear_acc,
        'ova_rbf_acc': ova_rbf_acc,
        'ovo_rbf_acc': ovo_rbf_acc,
    }

# Main function to run the entire demo
def main():
    """Main function to run the demo."""
    print("Starting demonstration of One-vs-All (OVA) and One-vs-One (OVO) approaches...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dataset
    n_classes = 3
    X, y = generate_dataset(n_samples=300, n_classes=n_classes)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train OVA classifiers
    ova_classifiers = train_one_vs_all(X_train, y_train, n_classes)
    
    # Train OVO classifiers
    ovo_classifiers = train_one_vs_one(X_train, y_train, n_classes)
    
    # Plot decision boundaries
    plot_decision_boundaries(X, y, ova_classifiers, 'OVA', n_classes)
    plot_decision_boundaries(X, y, ovo_classifiers, 'OVO', n_classes)
    
    # Measure time complexity
    time_results = measure_time_complexity(X_train, y_train, X_test, n_classes)
    
    # Evaluate accuracy
    accuracy_results = evaluate_accuracy(X_train, y_train, X_test, y_test, n_classes)
    
    # Calculate number of classifiers
    num_classifiers = calculate_num_classifiers(n_classes)
    
    # Compare effect of imbalance
    imbalance_results = compare_imbalance_effect(n_classes)
    
    # Compare with nonlinear decision boundaries
    nonlinear_results = compare_nonlinear_boundaries(n_classes)
    
    print("\nDemonstration complete! All results and visualizations saved.")
    
    return {
        'time_results': time_results,
        'accuracy_results': accuracy_results, 
        'num_classifiers': num_classifiers,
        'imbalance_results': imbalance_results,
        'nonlinear_results': nonlinear_results
    }

# Run the main function if script is executed directly
if __name__ == "__main__":
    results = main() 