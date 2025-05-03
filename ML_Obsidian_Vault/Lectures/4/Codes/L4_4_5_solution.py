import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
import os
from matplotlib.colors import ListedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_4_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

print("\n====================================================================")
print("Question 5: Classifier Characteristics")
print("====================================================================")

# Function to plot simple decision boundaries
def plot_simple_decision_boundary(X, y, classifier, title, filename):
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Get the prediction for each point in the meshgrid
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] - 0.5
    
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='lightblue', edgecolor='k', s=60, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    
    # Color the regions
    plt.contourf(xx, yy, Z, levels=[-999, 0, 999], alpha=0.2, colors=['lightblue', 'salmon'])
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create comparison visualization of all classifiers
def create_classifier_comparison():
    # Create a common dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                            n_informative=2, random_state=42, 
                            class_sep=1.5, n_clusters_per_class=1)
    
    # Initialize the classifiers
    classifiers = [
        Perceptron(max_iter=1000, tol=1e-3),
        LogisticRegression(C=1.0),
        LinearDiscriminantAnalysis(),
        LinearSVC(C=1.0, loss='hinge', dual="auto")
    ]
    
    titles = [
        'Perceptron',
        'Logistic Regression',
        'Linear Discriminant Analysis',
        'Support Vector Machine'
    ]
    
    filenames = [
        'perceptron.png',
        'logistic_regression.png',
        'lda.png',
        'svm.png'
    ]
    
    # Print and create plots for each classifier
    for clf, title, filename in zip(classifiers, titles, filenames):
        # Train the classifier
        clf.fit(X, y)
        
        # Print model details
        print(f"\n--------------------------------------------------------------------")
        print(f"Classifier: {title}")
        print(f"--------------------------------------------------------------------")
        
        try:
            if hasattr(clf, "coef_"):
                print(f"Coefficients: {clf.coef_[0]}")
            if hasattr(clf, "intercept_"):
                print(f"Intercept: {clf.intercept_[0]}")
            
            # Print the decision boundary equation
            if hasattr(clf, "coef_") and hasattr(clf, "intercept_"):
                w1, w2 = clf.coef_[0]
                b = clf.intercept_[0]
                print(f"Decision Boundary Equation: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
        except Exception as e:
            print(f"Could not extract model parameters: {e}")
            
        # Create the visualization
        plot_simple_decision_boundary(X, y, clf, title, filename)

# 1. Visualize SVM and the margin concept
def visualize_svm_margin():
    print("\n--------------------------------------------------------------------")
    print("1. SVM: Maximizing the margin between classes")
    print("--------------------------------------------------------------------")
    
    # Create a simple dataset
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y = 2*y - 1  # Convert to {-1, 1}
    
    # Train the SVM
    svm = LinearSVC(C=1.0, loss='hinge', dual="auto")
    svm.fit(X, y)
    
    # Print SVM details
    print(f"SVM Coefficients: {svm.coef_[0]}")
    print(f"SVM Intercept: {svm.intercept_[0]}")
    print(f"Decision Boundary Equation: {svm.coef_[0,0]:.4f}*x1 + {svm.coef_[0,1]:.4f}*x2 + {svm.intercept_[0]:.4f} = 0")
    
    # Calculate margin
    w_norm = np.linalg.norm(svm.coef_[0])
    margin = 2 / w_norm
    print(f"Margin width: {margin:.4f}")
    
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    
    # Get the decision function values
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Identify support vectors (approximation)
    margin_points = np.where(np.abs(svm.decision_function(X)) <= 1.0)[0]
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y==-1, 0], X[y==-1, 1], c='lightblue', edgecolor='k', s=60, label='Class -1')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Highlight margin points
    plt.scatter(X[margin_points, 0], X[margin_points, 1], s=100, 
               linewidth=1, facecolors='none', edgecolor='k', label='Margin Points')
    
    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k', label='Decision Boundary')
    
    # Plot the margins
    plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=1, colors=['k','k'], linestyles=['--','--'])
    
    # Color the regions
    plt.contourf(xx, yy, Z, levels=[-999, 0, 999], alpha=0.2, colors=['lightblue', 'salmon'])
    
    plt.title('SVM: Maximizing the Margin', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svm_margin.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 2. Visualize Logistic Regression probabilities
def visualize_logistic_regression():
    print("\n--------------------------------------------------------------------")
    print("2. Logistic Regression: Modeling posterior probability using sigmoid")
    print("--------------------------------------------------------------------")
    
    # Create a dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42,
                               class_sep=1.5, n_clusters_per_class=1)
    
    # Train the logistic regression model
    logreg = LogisticRegression(C=1.0)
    logreg.fit(X, y)
    
    # Print model details
    print(f"Logistic Regression Coefficients: {logreg.coef_[0]}")
    print(f"Logistic Regression Intercept: {logreg.intercept_[0]}")
    print(f"Decision Boundary Equation: {logreg.coef_[0,0]:.4f}*x1 + {logreg.coef_[0,1]:.4f}*x2 + {logreg.intercept_[0]:.4f} = 0")
    
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Calculate the probabilities
    Z = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='lightblue', edgecolor='k', s=60, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Plot the decision boundary (probability = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')
    
    # Plot probability contours
    plt.contour(xx, yy, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], linewidths=1, colors='k', linestyles='--')
    
    # Color the regions by probability
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
    
    plt.title('Logistic Regression: Posterior Probabilities', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    
    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label('Probability of Class 1', fontsize=12)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "logistic_regression_prob.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a separate plot to show the sigmoid function
    plt.figure(figsize=(8, 6))
    
    # Get the coefficients of the model
    w1, w2 = logreg.coef_[0]
    b = logreg.intercept_[0]
    
    # Create a plot of the sigmoid function
    x = np.linspace(-6, 6, 100)
    y = 1 / (1 + np.exp(-x))
    
    plt.plot(x, y, 'k-', linewidth=2)
    plt.grid(True)
    plt.title('Sigmoid Function: σ(z) = 1/(1+e^(-z))', fontsize=14)
    plt.xlabel('z = w₁x₁ + w₂x₂ + b', fontsize=12)
    plt.ylabel('Probability: P(y=1|x)', fontsize=12)
    
    # Mark the decision boundary
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.text(0.5, 0.47, 'Decision\nBoundary\n(p=0.5)', color='r')
    
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sigmoid_function.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 3. Visualize Perceptron multiple solutions
def visualize_perceptron():
    print("\n--------------------------------------------------------------------")
    print("3. Perceptron: Finding any separating boundary")
    print("--------------------------------------------------------------------")
    
    # Create a dataset
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42,
                               class_sep=1.2, n_clusters_per_class=1)
    
    # Run the Perceptron with different random initializations
    perceptrons = []
    boundaries = []
    
    for i in range(5):
        perc = Perceptron(max_iter=1000, tol=1e-3, random_state=i)
        perc.fit(X, y)
        perceptrons.append(perc)
        
        # Print model details
        print(f"\nPerceptron solution {i+1}:")
        print(f"  Coefficients: {perc.coef_[0]}")
        print(f"  Intercept: {perc.intercept_[0]}")
        print(f"  Decision Boundary Equation: {perc.coef_[0,0]:.4f}*x1 + {perc.coef_[0,1]:.4f}*x2 + {perc.intercept_[0]:.4f} = 0")
        
        # Calculate the decision boundary
        boundaries.append((perc.coef_[0], perc.intercept_[0]))
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='lightblue', edgecolor='k', s=60, label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Plot multiple decision boundaries
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    
    for i, ((w1, w2), b) in enumerate(boundaries):
        # Plot decision boundary line
        if w2 != 0:
            y_range = (-w1 * x_range - b) / w2
            plt.plot(x_range, y_range, color=colors[i], linestyle='-', linewidth=2, 
                     label=f'Perceptron {i+1}')
    
    plt.title('Perceptron: Multiple Possible Solutions', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perceptron_boundaries.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Visualize LDA with class-conditional densities
def visualize_lda_densities():
    print("\n--------------------------------------------------------------------")
    print("4. LDA: Probabilistic approach with class-conditional densities")
    print("--------------------------------------------------------------------")
    
    # Create a dataset with Gaussian distributions
    np.random.seed(42)
    n_samples = 100
    means = [[-2, -2], [2, 2]]
    cov = [[1, 0], [0, 1]]
    
    X0 = np.random.multivariate_normal(means[0], cov, n_samples // 2)
    X1 = np.random.multivariate_normal(means[1], cov, n_samples // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Print model details
    print(f"LDA Coefficients: {lda.coef_[0]}")
    print(f"LDA Intercept: {lda.intercept_[0]}")
    print(f"Decision Boundary Equation: {lda.coef_[0,0]:.4f}*x1 + {lda.coef_[0,1]:.4f}*x2 + {lda.intercept_[0]:.4f} = 0")
    print(f"Class means: Class 0 = {means[0]}, Class 1 = {means[1]}")
    print(f"Shared covariance matrix: {cov}")
    
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Calculate the probabilities
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X0[:, 0], X0[:, 1], c='lightblue', edgecolor='k', s=60, label='Class 0')
    plt.scatter(X1[:, 0], X1[:, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Plot the class means
    plt.scatter(means[0][0], means[0][1], c='blue', marker='X', s=100, label='Mean Class 0')
    plt.scatter(means[1][0], means[1][1], c='red', marker='X', s=100, label='Mean Class 1')
    
    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')
    
    # Draw the probability contours
    plt.contour(xx, yy, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], linewidths=1, colors='k', linestyles='--')
    
    # Color the regions by probability
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
    
    plt.title('LDA: Class-Conditional Densities and Bayes Rule', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    
    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label('Probability of Class 1', fontsize=12)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lda_densities.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 5. Visualize LDA between-class to within-class scatter
def visualize_lda_scatter():
    print("\n--------------------------------------------------------------------")
    print("5. LDA: Maximizing between-class to within-class scatter ratio")
    print("--------------------------------------------------------------------")
    
    # Create a dataset where the classes have different means but same covariance
    np.random.seed(42)
    n_samples = 100
    means = [[-2, -2], [2, 2]]
    covs = [[[2, 0.5], [0.5, 1]], [[2, 0.5], [0.5, 1]]]  # Same covariance for both classes
    
    X0 = np.random.multivariate_normal(means[0], covs[0], n_samples // 2)
    X1 = np.random.multivariate_normal(means[1], covs[1], n_samples // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Compute class means and overall mean
    class_means = [np.mean(X[y == i], axis=0) for i in range(2)]
    overall_mean = np.mean(X, axis=0)
    
    # Compute within-class scatter matrix
    S_W = np.zeros((2, 2))
    for i in range(2):
        class_data = X[y == i]
        class_scatter = np.zeros((2, 2))
        for x in class_data:
            x_diff = x - class_means[i]
            class_scatter += np.outer(x_diff, x_diff)
        S_W += class_scatter
    
    # Compute between-class scatter matrix
    S_B = np.zeros((2, 2))
    for i in range(2):
        n_i = np.sum(y == i)
        mean_diff = class_means[i] - overall_mean
        S_B += n_i * np.outer(mean_diff, mean_diff)
    
    # Compute the projection direction (eigenvector of S_W^-1 S_B)
    evals, evecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    lda_direction = evecs[:, 0]
    
    # Print scatter matrices and projection direction
    print(f"Class means: Class 0 = {class_means[0]}, Class 1 = {class_means[1]}")
    print(f"Overall mean: {overall_mean}")
    print(f"Within-class scatter matrix S_W:\n{S_W}")
    print(f"Between-class scatter matrix S_B:\n{S_B}")
    print(f"LDA projection direction: {lda_direction}")
    
    # Train LDA to get the decision boundary
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    
    # Print model details
    print(f"LDA Coefficients: {lda.coef_[0]}")
    print(f"LDA Intercept: {lda.intercept_[0]}")
    print(f"Decision Boundary Equation: {lda.coef_[0,0]:.4f}*x1 + {lda.coef_[0,1]:.4f}*x2 + {lda.intercept_[0]:.4f} = 0")
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(X0[:, 0], X0[:, 1], c='lightblue', edgecolor='k', s=60, label='Class 0')
    plt.scatter(X1[:, 0], X1[:, 1], c='salmon', edgecolor='k', s=60, label='Class 1')
    
    # Plot the class means
    plt.scatter(class_means[0][0], class_means[0][1], c='blue', marker='X', s=100, label='Mean Class 0')
    plt.scatter(class_means[1][0], class_means[1][1], c='red', marker='X', s=100, label='Mean Class 1')
    
    # Plot the overall mean
    plt.scatter(overall_mean[0], overall_mean[1], c='black', marker='o', s=100, label='Overall Mean')
    
    # Plot the LDA direction
    plt.arrow(overall_mean[0], overall_mean[1], 
              lda_direction[0]*3, lda_direction[1]*3, 
              head_width=0.3, head_length=0.3, fc='k', ec='k', label='LDA Direction')
    
    # Plot the decision boundary
    slope = -lda_direction[0] / lda_direction[1]
    intercept = np.dot(overall_mean, lda_direction) / lda_direction[1]
    x_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_range = slope * x_range + intercept
    plt.plot(x_range, y_range, 'k--', label='Decision Boundary')
    
    plt.title('LDA: Maximizing Between-Class to Within-Class Scatter Ratio', fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lda_scatter_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Run the analysis and create the visualizations
print("\nCreating visualizations for each classifier...")
create_classifier_comparison()
visualize_svm_margin()
visualize_logistic_regression()
visualize_perceptron()
visualize_lda_densities()
visualize_lda_scatter()

print("\n====================================================================")
print("SUMMARY")
print("====================================================================")
print("1. SVM finds a decision boundary that maximizes the margin between classes")
print("2. LDA uses a probabilistic approach based on class-conditional densities and Bayes' rule")
print("3. Perceptron simply tries to find any decision boundary that separates the classes")
print("4. Logistic Regression directly models the posterior probability P(y|x) using the sigmoid function")
print("5. LDA is a discriminative model that maximizes the ratio of between-class to within-class scatter") 