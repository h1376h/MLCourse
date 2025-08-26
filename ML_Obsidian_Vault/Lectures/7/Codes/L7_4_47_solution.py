import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_47")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX rendering with proper configuration
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

class AdaBoostManual:
    def __init__(self, T=3):
        self.T = T
        self.alphas = []
        self.weak_learners = []
        self.errors = []
        self.Z_values = []
        self.weight_history = []
        
    def fit(self, X, y):
        """
        Manual AdaBoost implementation with detailed step-by-step calculations
        """
        n_samples = len(X)
        
        # Initialize weights uniformly
        D = np.ones(n_samples) / n_samples
        self.weight_history.append(D.copy())
        
        print("=" * 80)
        print("ADABOOST MANUAL CALCULATION - QUESTION 47")
        print("=" * 80)
        print(f"Dataset: {n_samples} samples")
        print(f"Features: {X}")
        print(f"Labels: {y}")
        print(f"Initial weights: {D}")
        print()
        
        for t in range(self.T):
            print(f"ITERATION {t+1}")
            print("-" * 40)
            
            # Step 1: Find the best weak learner (decision stump)
            best_stump, best_error, best_predictions = self._find_best_stump(X, y, D)
            
            # Step 2: Calculate error rate
            epsilon_t = best_error
            self.errors.append(epsilon_t)
            
            print(f"Step 1-2: Best decision stump found")
            print(f"  Feature: {best_stump['feature']}")
            print(f"  Threshold: {best_stump['threshold']}")
            print(f"  Direction: {best_stump['direction']}")
            print(f"  Predictions: {best_predictions}")
            print(f"  Weighted error $\\epsilon_t$ = {epsilon_t:.4f}")
            
            # Step 3: Calculate alpha (weight of the weak learner)
            alpha_t = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
            self.alphas.append(alpha_t)
            
            print(f"Step 3: Calculate $\\alpha_t$")
            print(f"  $\\alpha_t = 0.5 \\cdot \\ln((1 - \\epsilon_t) / \\epsilon_t)$")
            print(f"  $\\alpha_t = 0.5 \\cdot \\ln((1 - {epsilon_t:.4f}) / {epsilon_t:.4f})$")
            print(f"  $\\alpha_t = 0.5 \\cdot \\ln({(1-epsilon_t):.4f} / {epsilon_t:.4f})$")
            print(f"  $\\alpha_t = 0.5 \\cdot \\ln({(1-epsilon_t)/epsilon_t:.4f})$")
            print(f"  $\\alpha_t = {alpha_t:.4f}$")
            
            # Step 4: Update weights
            # w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i)) / Z_t
            weight_updates = D * np.exp(-alpha_t * y * best_predictions)
            
            # Step 5: Calculate normalization factor Z_t
            Z_t = np.sum(weight_updates)
            self.Z_values.append(Z_t)
            
            print(f"Step 4-5: Update weights and calculate Z_t")
            print(f"  Weight updates before normalization:")
            for i in range(n_samples):
                print(f"    w_{i+1}^({t+2}) = {D[i]:.4f} * exp(-{alpha_t:.4f} * {y[i]} * {best_predictions[i]})")
                print(f"    w_{i+1}^({t+2}) = {D[i]:.4f} * exp({-alpha_t * y[i] * best_predictions[i]:.4f})")
                print(f"    w_{i+1}^({t+2}) = {weight_updates[i]:.4f}")
            
            print(f"  $Z_t$ = sum of weight updates = {Z_t:.4f}")
            
            # Step 6: Normalize weights
            D = weight_updates / Z_t
            self.weight_history.append(D.copy())
            
            print(f"Step 6: Normalize weights")
            print(f"  $D_{t+2}$ = {D}")
            print(f"  Sum of weights = {np.sum(D):.6f}")
            
            # Store the weak learner
            self.weak_learners.append(best_stump)
            
            print()
        
        print("=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Alphas: {[f'{a:.4f}' for a in self.alphas]}")
        print(f"Errors: {[f'{e:.4f}' for e in self.errors]}")
        print(f"Z values: {[f'{z:.4f}' for z in self.Z_values]}")
        
        return self
    
    def _find_best_stump(self, X, y, D):
        """
        Find the best decision stump by trying all possible splits
        """
        n_samples, n_features = X.shape
        best_error = float('inf')
        best_stump = None
        best_predictions = None
        
        print("  Detailed stump search:")
        
        for feature in range(n_features):
            print(f"    Feature {feature}:")
            # Get unique values for this feature
            unique_values = np.unique(X[:, feature])
            print(f"      Unique values: {unique_values}")
            
            # Try each threshold (midpoint between consecutive values)
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Try both directions
                for direction in [-1, 1]:
                    # Make predictions
                    predictions = np.where(X[:, feature] <= threshold, direction, -direction)
                    
                    # Calculate weighted error
                    misclassified = (predictions != y)
                    error = np.sum(D[misclassified])
                    
                    print(f"      Threshold {threshold:.2f}, Direction {direction}: Error = {error:.4f}")
                    
                    if error < best_error:
                        best_error = error
                        best_stump = {
                            'feature': feature,
                            'threshold': threshold,
                            'direction': direction
                        }
                        best_predictions = predictions.copy()
                        print(f"      *** New best stump found! ***")
        
        return best_stump, best_error, best_predictions
    
    def predict(self, X):
        """
        Make predictions using the ensemble
        """
        X = np.array(X)  # Ensure X is a numpy array
        predictions = np.zeros(len(X))
        
        for t, (alpha, stump) in enumerate(zip(self.alphas, self.weak_learners)):
            # Make prediction using the stump
            stump_pred = np.where(X[:, stump['feature']] <= stump['threshold'], 
                                stump['direction'], -stump['direction'])
            
            # Add weighted prediction
            predictions += alpha * stump_pred
        
        return np.sign(predictions)
    
    def predict_detailed(self, X):
        """
        Make predictions with detailed step-by-step output
        """
        X = np.array(X)
        predictions = np.zeros(len(X))
        
        print("\nDetailed Ensemble Prediction:")
        print("=" * 50)
        
        for t, (alpha, stump) in enumerate(zip(self.alphas, self.weak_learners)):
            print(f"\nWeak Learner {t+1} ($\\alpha_{t+1}$ = {alpha:.4f}):")
            print(f"  Feature: {stump['feature']}, Threshold: {stump['threshold']:.2f}, Direction: {stump['direction']}")
            
            # Make prediction using the stump
            stump_pred = np.where(X[:, stump['feature']] <= stump['threshold'], 
                                stump['direction'], -stump['direction'])
            
            print(f"  Stump predictions: {stump_pred}")
            print(f"  Weighted contributions: {alpha * stump_pred}")
            
            # Add weighted prediction
            predictions += alpha * stump_pred
            print(f"  Cumulative predictions: {predictions}")
        
        final_predictions = np.sign(predictions)
        print(f"\nFinal ensemble predictions: {final_predictions}")
        print(f"Raw ensemble scores: {predictions}")
        
        return final_predictions
    
    def get_training_error(self, X, y):
        """
        Calculate training error
        """
        predictions = self.predict(X)
        return 1 - accuracy_score(y, predictions)

def plot_adaboost_iterations(X, y, adaboost, save_dir):
    """
    Plot the decision boundaries for each iteration
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Create mesh grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Plot original data
    for i in range(4):
        ax = axes[i]
        
        # Plot data points
        for j, (x, label) in enumerate(zip(X, y)):
            color = 'red' if label == 1 else 'blue'
            marker = 'o' if label == 1 else 's'
            ax.scatter(x[0], x[1], c=color, marker=marker, s=100, 
                      edgecolors='black', linewidth=1.5, alpha=0.7)
            ax.annotate(f'X{j+1}', (x[0], x[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        if i == 0:
            ax.set_title('Original Dataset', fontsize=14, fontweight='bold')
        else:
            # Plot decision boundary for iteration i
            stump = adaboost.weak_learners[i-1]
            alpha = adaboost.alphas[i-1]
            
            # Create decision boundary
            if stump['feature'] == 0:  # x1 feature
                boundary_x = [stump['threshold'], stump['threshold']]
                boundary_y = [y_min, y_max]
            else:  # x2 feature
                boundary_x = [x_min, x_max]
                boundary_y = [stump['threshold'], stump['threshold']]
            
            ax.plot(boundary_x, boundary_y, 'g-', linewidth=2, 
                   label=f'Stump {i}: $\\alpha$={alpha:.3f}')
            
            # Shade regions
            if stump['feature'] == 0:
                if stump['direction'] == 1:
                    ax.fill_betweenx([y_min, y_max], x_min, stump['threshold'], 
                                   alpha=0.2, color='red')
                    ax.fill_betweenx([y_min, y_max], stump['threshold'], x_max, 
                                   alpha=0.2, color='blue')
                else:
                    ax.fill_betweenx([y_min, y_max], x_min, stump['threshold'], 
                                   alpha=0.2, color='blue')
                    ax.fill_betweenx([y_min, y_max], stump['threshold'], x_max, 
                                   alpha=0.2, color='red')
            else:
                if stump['direction'] == 1:
                    ax.fill_between([x_min, x_max], y_min, stump['threshold'], 
                                   alpha=0.2, color='red')
                    ax.fill_between([x_min, x_max], stump['threshold'], y_max, 
                                   alpha=0.2, color='blue')
                else:
                    ax.fill_between([x_min, x_max], y_min, stump['threshold'], 
                                   alpha=0.2, color='blue')
                    ax.fill_between([x_min, x_max], stump['threshold'], y_max, 
                                   alpha=0.2, color='red')
            
            ax.set_title(f'Iteration {i}: $\\epsilon$={adaboost.errors[i-1]:.3f}, $\\alpha$={alpha:.3f}', 
                        fontsize=14, fontweight='bold')
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaboost_iterations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_final_ensemble(X, y, adaboost, save_dir):
    """
    Plot the final ensemble decision boundary
    """
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get ensemble predictions for mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    ensemble_pred = adaboost.predict(mesh_points)
    ensemble_pred = ensemble_pred.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, ensemble_pred, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, ensemble_pred, levels=[0], colors='black', linewidths=2)
    
    # Plot data points
    for j, (x, label) in enumerate(zip(X, y)):
        color = 'red' if label == 1 else 'blue'
        marker = 'o' if label == 1 else 's'
        plt.scatter(x[0], x[1], c=color, marker=marker, s=100, 
                   edgecolors='black', linewidth=1.5, alpha=0.7)
        plt.annotate(f'X{j+1}', (x[0], x[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Final AdaBoost Ensemble Decision Boundary', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.3, label='Class +1'),
                      Patch(facecolor='blue', alpha=0.3, label='Class -1')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_ensemble.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weight_evolution(adaboost, save_dir):
    """
    Plot how weights evolve across iterations
    """
    weight_history = np.array(adaboost.weight_history)
    
    plt.figure(figsize=(12, 6))
    
    for i in range(weight_history.shape[1]):
        plt.plot(range(len(weight_history)), weight_history[:, i], 
                marker='o', linewidth=2, markersize=8, label=f'Sample X{i+1}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.title('Weight Evolution Across AdaBoost Iterations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(len(weight_history)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_new_point(X, y, adaboost, new_point, save_dir):
    """
    Analyze the effect of adding a new point
    """
    print("\n" + "=" * 80)
    print("ANALYSIS OF NEW POINT X9 = (0.25, 0.25, +1)")
    print("=" * 80)
    
    # Get predictions for existing points
    existing_pred = adaboost.predict(X)
    
    # Get prediction for new point
    new_pred = adaboost.predict([new_point[:2]])
    
    print(f"New point X9 = {new_point}")
    print(f"Prediction for X9: {new_pred[0]}")
    print(f"True label for X9: {new_point[2]}")
    print(f"Correctly classified: {new_pred[0] == new_point[2]}")
    
    # Plot with new point
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get ensemble predictions for mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    ensemble_pred = adaboost.predict(mesh_points)
    ensemble_pred = ensemble_pred.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, ensemble_pred, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx, yy, ensemble_pred, levels=[0], colors='black', linewidths=2)
    
    # Plot existing data points
    for j, (x, label) in enumerate(zip(X, y)):
        color = 'red' if label == 1 else 'blue'
        marker = 'o' if label == 1 else 's'
        plt.scatter(x[0], x[1], c=color, marker=marker, s=100, 
                   edgecolors='black', linewidth=1.5, alpha=0.7)
        plt.annotate(f'X{j+1}', (x[0], x[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    # Plot new point
    plt.scatter(new_point[0], new_point[1], c='green', marker='*', s=200, 
               edgecolors='black', linewidth=2, label='X9 (new)')
    plt.annotate('X9', (new_point[0], new_point[1]), xytext=(10, 10), 
                textcoords='offset points', fontsize=12, fontweight='bold')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('AdaBoost Decision Boundary with New Point X9', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.3, label='Class +1'),
                      Patch(facecolor='blue', alpha=0.3, label='Class -1'),
                      plt.scatter([], [], c='green', marker='*', s=200, label='X9 (new)')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'new_point_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Dataset from Question 47
    X = np.array([
        [-1, 0],      # X1
        [-0.5, 0.5],  # X2
        [0, 1],       # X3
        [0.5, 1],     # X4
        [1, 0],       # X5
        [1, -1],      # X6
        [0, -1],      # X7
        [0, 0]        # X8
    ])
    
    y = np.array([1, 1, -1, -1, 1, 1, -1, -1])  # Labels
    
    # Run AdaBoost
    adaboost = AdaBoostManual(T=3)
    adaboost.fit(X, y)
    
    # Calculate training error with detailed predictions
    print("\n" + "=" * 80)
    print("DETAILED ENSEMBLE PREDICTIONS")
    print("=" * 80)
    adaboost.predict_detailed(X)
    
    training_error = adaboost.get_training_error(X, y)
    print(f"\nTraining Error: {training_error:.4f}")
    
    # Calculate theoretical bound with detailed steps
    print(f"\nTheoretical Bound Calculation:")
    print(f"Formula: E_train ≤ ∏(t=1 to T) 2√(ε_t(1-ε_t))")
    
    theoretical_bound = 1.0
    for t, epsilon_t in enumerate(adaboost.errors):
        term = 2 * np.sqrt(epsilon_t * (1 - epsilon_t))
        theoretical_bound *= term
        print(f"  Iteration {t+1}: 2√({epsilon_t:.4f} × {1-epsilon_t:.4f}) = 2√({epsilon_t*(1-epsilon_t):.4f}) = {term:.4f}")
    
    print(f"  Final bound: {theoretical_bound:.4f}")
    print(f"Actual Training Error: {training_error:.4f}")
    print(f"Bound vs Actual: {theoretical_bound:.4f} vs {training_error:.4f}")
    
    # Generate plots
    plot_adaboost_iterations(X, y, adaboost, save_dir)
    plot_final_ensemble(X, y, adaboost, save_dir)
    plot_weight_evolution(adaboost, save_dir)
    
    # Analyze new point
    new_point = np.array([0.25, 0.25, 1])  # X9
    analyze_new_point(X, y, adaboost, new_point, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}")
    
    return adaboost, X, y

if __name__ == "__main__":
    adaboost, X, y = main()
