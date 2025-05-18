import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_5_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset
def generate_data(n_samples=1000, n_features=2, noise=0.1):
    """Generate a synthetic binary classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_redundant=0, n_informative=2,
                               random_state=42, n_clusters_per_class=1,
                               class_sep=2.0, flip_y=noise)
    return X, y

# Define sigmoid function for logistic regression
def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))

# Define loss function (binary cross-entropy)
def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss."""
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip to avoid numerical issues
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Batch Learning - Logistic Regression with Gradient Descent
class BatchLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tol=1e-4):
        """Initialize Batch Logistic Regression model."""
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []
        self.iterations = 0
        
    def fit(self, X, y, X_test=None, y_test=None, verbose=False):
        """Fit the model to the data using batch gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Track computational time
        start_time = time.time()
        
        # Training loop
        for i in range(self.max_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # Compute gradients (batch gradient descent)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Save loss and accuracy
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            
            # Calculate training accuracy
            train_predictions = (y_pred >= 0.5).astype(int)
            train_accuracy = accuracy_score(y, train_predictions)
            self.accuracy_history.append(train_accuracy)
            
            # Convergence check
            if i > 0 and abs(self.loss_history[i] - self.loss_history[i-1]) < self.tol:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break
                
            self.iterations = i + 1
            
        self.training_time = time.time() - start_time
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int)

# Online Learning - Stochastic Gradient Descent Logistic Regression
class OnlineLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1, n_epochs=10, tol=1e-4):
        """Initialize Online Logistic Regression model."""
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations  # Number of iterations per sample
        self.n_epochs = n_epochs  # Number of passes through the dataset
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []
        self.iterations = 0
        
    def fit(self, X, y, X_test=None, y_test=None, verbose=False):
        """Fit the model to the data using online gradient descent (SGD)."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Track computational time
        start_time = time.time()
        
        # Track loss for convergence
        prev_epoch_loss = float('inf')
        
        # Record loss after each sample to match batch records
        accumulated_loss = 0
        accumulated_accuracy = 0
        sample_count = 0
        
        # Training loop - multiple epochs
        for epoch in range(self.n_epochs):
            # Create a random permutation of indices
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            
            # Iterate through each sample in random order
            for i in indices:
                x_i = X[i].reshape(1, -1)  # Single sample
                y_i = np.array([y[i]])     # Single label
                
                # Forward pass for this sample
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_pred = sigmoid(linear_model)
                
                # Compute gradients (online/stochastic gradient descent)
                dw = np.dot(x_i.T, (y_pred - y_i))
                db = np.sum(y_pred - y_i)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Accumulate loss for this sample
                sample_loss = binary_cross_entropy(y_i, y_pred)
                epoch_loss += sample_loss
                
                # Accumulate for tracking
                accumulated_loss += sample_loss
                accumulated_accuracy += (y_pred >= 0.5).astype(int) == y_i
                sample_count += 1
                
                # Record accumulated metrics at regular intervals to match batch history
                if sample_count >= n_samples / 20:  # Record 20 points per epoch
                    self.loss_history.append(accumulated_loss / sample_count)
                    self.accuracy_history.append(accumulated_accuracy / sample_count)
                    accumulated_loss = 0
                    accumulated_accuracy = 0
                    sample_count = 0
                
                self.iterations += 1
            
            # Check for convergence after each epoch
            epoch_loss /= n_samples
            if abs(epoch_loss - prev_epoch_loss) < self.tol:
                if verbose:
                    print(f"Converged after {epoch+1} epochs")
                break
            
            prev_epoch_loss = epoch_loss
        
        self.training_time = time.time() - start_time
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X) >= 0.5).astype(int)

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, title, filename, show_samples=True):
    """Plot the decision boundary and data points."""
    # Set figure size
    plt.figure(figsize=(10, 8))
    
    # Determine the bounds of the grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                          np.arange(y_min, y_max, 0.01))
    
    # Predict the function value for the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Plot the data points if requested
    if show_samples:
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
        
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

# Function to compare training progress
def plot_training_progress(batch_model, online_model, filename):
    """Plot the training progress (loss and accuracy) for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Adjust x-axis for fair comparison
    batch_iterations = np.arange(len(batch_model.loss_history))
    online_iterations = np.arange(len(online_model.loss_history))
    
    # Loss plot
    axes[0].plot(batch_iterations, batch_model.loss_history, 
                 'b-', linewidth=2, label='Batch Learning')
    axes[0].plot(online_iterations, online_model.loss_history, 
                 'r-', linewidth=2, label='Online Learning')
    axes[0].set_title('Loss vs. Training Progress')
    axes[0].set_xlabel('Training Progress')
    axes[0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0].grid(True)
    axes[0].legend()
    
    # Accuracy plot
    axes[1].plot(batch_iterations, batch_model.accuracy_history, 
                 'b-', linewidth=2, label='Batch Learning')
    axes[1].plot(online_iterations, online_model.accuracy_history, 
                 'r-', linewidth=2, label='Online Learning')
    axes[1].set_title('Accuracy vs. Training Progress')
    axes[1].set_xlabel('Training Progress')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

# Function to plot learning curves for different dataset sizes
def plot_learning_curves(filename):
    """Plot learning curves showing how both approaches scale with dataset size."""
    # Dataset sizes to test
    dataset_sizes = [100, 500, 1000, 5000, 10000]
    
    # Arrays to store results
    batch_times = []
    online_times = []
    batch_accuracies = []
    online_accuracies = []
    
    # Test different dataset sizes
    for size in dataset_sizes:
        # Generate data
        X, y = generate_data(n_samples=size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train batch model
        batch_model = BatchLogisticRegression()
        batch_model.fit(X_train, y_train)
        batch_times.append(batch_model.training_time)
        batch_pred = batch_model.predict(X_test)
        batch_accuracies.append(accuracy_score(y_test, batch_pred))
        
        # Train online model
        online_model = OnlineLogisticRegression()
        online_model.fit(X_train, y_train)
        online_times.append(online_model.training_time)
        online_pred = online_model.predict(X_test)
        online_accuracies.append(accuracy_score(y_test, online_pred))
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training time vs dataset size
    axes[0].plot(dataset_sizes, batch_times, 'b-o', linewidth=2, label='Batch Learning')
    axes[0].plot(dataset_sizes, online_times, 'r-o', linewidth=2, label='Online Learning')
    axes[0].set_title('Training Time vs. Dataset Size')
    axes[0].set_xlabel('Dataset Size')
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Accuracy vs dataset size
    axes[1].plot(dataset_sizes, batch_accuracies, 'b-o', linewidth=2, label='Batch Learning')
    axes[1].plot(dataset_sizes, online_accuracies, 'r-o', linewidth=2, label='Online Learning')
    axes[1].set_title('Test Accuracy vs. Dataset Size')
    axes[1].set_xlabel('Dataset Size')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

# Function to demonstrate concept drift and adaptation
def plot_concept_drift_adaptation(filename):
    """Plot how online and batch learning adapt to concept drift."""
    # Generate initial data with a specific pattern
    np.random.seed(42)
    X1, y1 = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1,
                               class_sep=2.0, flip_y=0.1)
    
    # Generate data with a drift in concept
    np.random.seed(24)
    X2, y2 = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, random_state=24, n_clusters_per_class=1,
                               class_sep=2.0, flip_y=0.1)
    
    # Normalize both datasets to make them more different
    X1 = (X1 - X1.mean(axis=0)) / X1.std(axis=0)
    X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
    
    # Split the data for training and testing
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_test, y2_test = X2[:100], y2[:100]  # Use part of the second dataset for testing
    
    # Train batch model on initial data
    batch_model = BatchLogisticRegression()
    batch_model.fit(X1_train, y1_train)
    
    # Train online model on initial data
    online_model = OnlineLogisticRegression()
    online_model.fit(X1_train, y1_train)
    
    # Test accuracy on initial test set
    batch_acc_initial = accuracy_score(y1_test, batch_model.predict(X1_test))
    online_acc_initial = accuracy_score(y1_test, online_model.predict(X1_test))
    
    # Test accuracy on drifted test set (before adaptation)
    batch_acc_drift = accuracy_score(y2_test, batch_model.predict(X2_test))
    online_acc_drift = accuracy_score(y2_test, online_model.predict(X2_test))
    
    # Adapt online model to the new concept (incremental learning)
    online_model.fit(X2[:400], y2[:400])
    
    # Retrain batch model on combined data (needs full retraining)
    X_combined = np.vstack([X1_train, X2[:400]])
    y_combined = np.hstack([y1_train, y2[:400]])
    batch_model.fit(X_combined, y_combined)
    
    # Test accuracy after adaptation
    batch_acc_adapt = accuracy_score(y2_test, batch_model.predict(X2_test))
    online_acc_adapt = accuracy_score(y2_test, online_model.predict(X2_test))
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    index = np.arange(3)
    
    batch_accuracies = [batch_acc_initial, batch_acc_drift, batch_acc_adapt]
    online_accuracies = [online_acc_initial, online_acc_drift, online_acc_adapt]
    
    batch_bars = ax.bar(index - bar_width/2, batch_accuracies, bar_width, 
                        label='Batch Learning', color='blue', alpha=0.7)
    online_bars = ax.bar(index + bar_width/2, online_accuracies, bar_width,
                         label='Online Learning', color='red', alpha=0.7)
    
    ax.set_title('Adaptation to Concept Drift')
    ax.set_xlabel('Learning Phase')
    ax.set_ylabel('Test Accuracy')
    ax.set_xticks(index)
    ax.set_xticklabels(['Initial Data', 'After Drift\n(Before Adaptation)', 'After Adaptation'])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
    
    add_labels(batch_bars)
    add_labels(online_bars)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

# Function to visualize memory usage
def plot_memory_usage(filename):
    """Plot memory usage comparison between batch and online learning."""
    # Dataset sizes to test
    dataset_sizes = [100, 1000, 5000, 10000, 50000]
    
    # Approximate memory usage (simplified model based on what each algorithm stores)
    # For batch learning: needs to store entire dataset
    batch_memory = [size * 8 * 2 for size in dataset_sizes]  # 8 bytes per float, 2 features
    
    # For online learning: only needs to store model parameters and current sample
    online_memory = [2 * 8 * 2 for _ in dataset_sizes]  # 2 parameters, 8 bytes per float, 2 features
    
    plt.figure(figsize=(12, 8))
    plt.plot(dataset_sizes, batch_memory, 'b-o', linewidth=2, label='Batch Learning')
    plt.plot(dataset_sizes, online_memory, 'r-o', linewidth=2, label='Online Learning')
    plt.title('Memory Usage Comparison')
    plt.xlabel('Dataset Size')
    plt.ylabel('Approximate Memory Usage (bytes)')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Log scale to show the difference more clearly
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')

def main():
    print("Comparing Batch and Online Learning for Linear Classification")
    
    # Generate dataset
    print("\nGenerating dataset...")
    X, y = generate_data(n_samples=1000, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train batch model
    print("\nTraining batch logistic regression model...")
    batch_model = BatchLogisticRegression(learning_rate=0.1, max_iterations=100)
    batch_model.fit(X_train, y_train, verbose=True)
    batch_pred = batch_model.predict(X_test)
    batch_accuracy = accuracy_score(y_test, batch_pred)
    
    print(f"Batch model training time: {batch_model.training_time:.4f} seconds")
    print(f"Batch model iterations: {batch_model.iterations}")
    print(f"Batch model test accuracy: {batch_accuracy:.4f}")
    
    # Train online model
    print("\nTraining online logistic regression model...")
    online_model = OnlineLogisticRegression(learning_rate=0.01, n_epochs=5)
    online_model.fit(X_train, y_train, verbose=True)
    online_pred = online_model.predict(X_test)
    online_accuracy = accuracy_score(y_test, online_pred)
    
    print(f"Online model training time: {online_model.training_time:.4f} seconds")
    print(f"Online model updates: {online_model.iterations}")
    print(f"Online model test accuracy: {online_accuracy:.4f}")
    
    # Plot decision boundaries
    print("\nPlotting decision boundaries...")
    plot_decision_boundary(X_test, y_test, batch_model, 
                          "Decision Boundary - Batch Learning", 
                          "batch_decision_boundary.png")
    
    plot_decision_boundary(X_test, y_test, online_model, 
                          "Decision Boundary - Online Learning", 
                          "online_decision_boundary.png")
    
    # Plot training progress
    print("\nPlotting training progress...")
    plot_training_progress(batch_model, online_model, "training_progress.png")
    
    # Plot learning curves
    print("\nPlotting learning curves...")
    plot_learning_curves("learning_curves.png")
    
    # Plot concept drift adaptation
    print("\nPlotting concept drift adaptation...")
    plot_concept_drift_adaptation("concept_drift_adaptation.png")
    
    # Plot memory usage
    print("\nPlotting memory usage comparison...")
    plot_memory_usage("memory_usage.png")
    
    print(f"\nAll visualizations saved to {save_dir}")

if __name__ == "__main__":
    main() 