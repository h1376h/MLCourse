import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Set random seed for reproducibility
np.random.seed(42)

# Create directory for saving figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L4_6_Quiz_8")
os.makedirs(save_dir, exist_ok=True)

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 6)

# Generate a synthetic dataset for multi-class classification
X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
X = StandardScaler().fit_transform(X)  # Standardize features

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the generated dataset
def plot_dataset(X, y, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in np.unique(y):
        ax.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', alpha=0.7)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_dataset(X, y, 'Multi-class Classification Dataset')

# Implementation of softmax function
def softmax(z):
    """Compute softmax values for each set of scores in z."""
    # Shift the input for numerical stability
    shifted_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# One-hot encoding function
def one_hot_encode(y, num_classes):
    """Convert label array to one-hot encoded matrix."""
    y_one_hot = np.zeros((len(y), num_classes))
    y_one_hot[np.arange(len(y)), y] = 1
    return y_one_hot

# Implementation of softmax regression
class SoftmaxRegression:
    def __init__(self, num_features, num_classes, learning_rate=0.01, epochs=100, batch_size=32):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # Initialize weights and bias
        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))
        
        # For tracking training progress
        self.train_loss_history = []
        self.train_accuracy_history = []
    
    def forward(self, X):
        """Forward pass: compute logits and apply softmax."""
        logits = np.dot(X, self.W) + self.b
        probabilities = softmax(logits)
        return probabilities
    
    def compute_likelihood(self, X, y_one_hot):
        """Compute the likelihood for the dataset."""
        probabilities = self.forward(X)
        # For each sample, multiply probability of the correct class
        likelihood = np.prod([probabilities[i, np.argmax(y_one_hot[i])] for i in range(len(X))])
        return likelihood
    
    def compute_log_likelihood(self, X, y_one_hot):
        """Compute the log-likelihood for the dataset."""
        probabilities = self.forward(X)
        # For each sample, sum log probability of correct class
        log_likelihood = np.sum([np.log(probabilities[i, np.argmax(y_one_hot[i])]) for i in range(len(X))])
        return log_likelihood
    
    def compute_cross_entropy_loss(self, X, y_one_hot):
        """Compute the cross-entropy loss."""
        m = len(X)
        probabilities = self.forward(X)
        # Cross-entropy loss formula
        loss = -np.sum(y_one_hot * np.log(probabilities + 1e-15)) / m
        return loss
    
    def compute_accuracy(self, X, y):
        """Compute classification accuracy."""
        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def train(self, X, y):
        """Train the softmax regression model using batch gradient descent."""
        num_samples = len(X)
        num_batches = int(np.ceil(num_samples / self.batch_size))
        y_one_hot = one_hot_encode(y, self.num_classes)
        
        for epoch in range(self.epochs):
            # Shuffle the data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            y_one_hot_shuffled = y_one_hot[indices]
            
            epoch_loss = 0
            epoch_likelihood = 0
            epoch_log_likelihood = 0
            
            # Process mini-batches
            for b in range(num_batches):
                batch_start = b * self.batch_size
                batch_end = min((b + 1) * self.batch_size, num_samples)
                
                # Get mini-batch
                X_batch = X_shuffled[batch_start:batch_end]
                y_one_hot_batch = y_one_hot_shuffled[batch_start:batch_end]
                
                # Forward pass
                probabilities = self.forward(X_batch)
                
                # Compute gradients
                dZ = probabilities - y_one_hot_batch
                dW = np.dot(X_batch.T, dZ) / len(X_batch)
                db = np.sum(dZ, axis=0, keepdims=True) / len(X_batch)
                
                # Update parameters
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db
                
                # Track metrics for this batch
                batch_loss = self.compute_cross_entropy_loss(X_batch, y_one_hot_batch)
                epoch_loss += batch_loss * len(X_batch)
            
            # Compute metrics for the full epoch
            epoch_loss /= num_samples
            epoch_likelihood = self.compute_likelihood(X, y_one_hot)
            epoch_log_likelihood = self.compute_log_likelihood(X, y_one_hot)
            epoch_accuracy = self.compute_accuracy(X, y)
            
            # Track progress
            self.train_loss_history.append(epoch_loss)
            self.train_accuracy_history.append(epoch_accuracy)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Loss: {epoch_loss:.6f}, "
                      f"Accuracy: {epoch_accuracy:.4f}, "
                      f"Likelihood: {epoch_likelihood:.8e}, "
                      f"Log-Likelihood: {epoch_log_likelihood:.4f}")

# Train the model
num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))
model = SoftmaxRegression(num_features, num_classes, learning_rate=0.1, epochs=100, batch_size=32)

print("Starting training...")
start_time = time.time()
model.train(X_train, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# Evaluate on test set
test_loss = model.compute_cross_entropy_loss(X_test, one_hot_encode(y_test, num_classes))
test_accuracy = model.compute_accuracy(X_test, y_test)
test_likelihood = model.compute_likelihood(X_test, one_hot_encode(y_test, num_classes))
test_log_likelihood = model.compute_log_likelihood(X_test, one_hot_encode(y_test, num_classes))

print(f"\nTest Results:")
print(f"Loss: {test_loss:.6f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Likelihood: {test_likelihood:.8e}")
print(f"Log-Likelihood: {test_log_likelihood:.4f}")

# Plot the decision boundary
def plot_decision_boundary(model, X, y, title):
    # Define grid boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Get predictions for each point on the meshgrid
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot the data points
    for i in np.unique(y):
        ax.scatter(X[y == i, 0], X[y == i, 1], label=f'Class {i}', alpha=0.7)
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_decision_boundary(model, X, y, 'Softmax Regression Decision Boundaries')

# Plot the training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(range(1, model.epochs + 1), model.train_loss_history, 'b-')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, model.epochs + 1), model.train_accuracy_history, 'r-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create visualization to demonstrate the relationship between MLE and cross-entropy loss
# We'll create a simple 1D example for visualization
def create_mle_cross_entropy_visualization():
    # Create a simple scenario: true class probability vs predicted probability
    true_probs = np.array([1.0, 0.0, 0.0])  # One-hot encoded ground truth (class 0)
    p_range = np.linspace(0.01, 0.99, 100)  # Range of predicted probabilities for class 0
    
    # Arrays to store values
    likelihoods = []
    log_likelihoods = []
    cross_entropies = []
    negative_log_likelihoods = []
    
    # Calculate values for different predicted probabilities of class 0
    for p in p_range:
        # Create softmax-like predictions (p for class 0, remaining distributed among other classes)
        pred_probs = np.array([p, (1-p)/2, (1-p)/2])
        
        # Calculate likelihood (probability of observing the true class)
        likelihood = np.prod(pred_probs ** true_probs)
        likelihoods.append(likelihood)
        
        # Calculate log-likelihood
        log_likelihood = np.sum(true_probs * np.log(pred_probs))
        log_likelihoods.append(log_likelihood)
        
        # Calculate cross-entropy
        cross_entropy = -np.sum(true_probs * np.log(pred_probs))
        cross_entropies.append(cross_entropy)
        
        # Calculate negative log-likelihood
        negative_log_likelihood = -log_likelihood
        negative_log_likelihoods.append(negative_log_likelihood)
    
    # Create the visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot likelihood and cross-entropy
    ax1 = axs[0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(p_range, likelihoods, 'b-', label='Likelihood')
    line2 = ax1_twin.plot(p_range, cross_entropies, 'r-', label='Cross-Entropy Loss')
    
    ax1.set_xlabel('Predicted Probability for True Class')
    ax1.set_ylabel('Likelihood', color='b')
    ax1_twin.set_ylabel('Cross-Entropy Loss', color='r')
    ax1.set_title('Relationship between Likelihood and Cross-Entropy Loss')
    
    # Add combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax1.grid(True)
    
    # Plot log-likelihood and negative log-likelihood (equivalent to cross-entropy for one-hot)
    ax2 = axs[1]
    ax2_twin = ax2.twinx()
    
    line3 = ax2.plot(p_range, log_likelihoods, 'g-', label='Log-Likelihood')
    line4 = ax2_twin.plot(p_range, negative_log_likelihoods, 'm-', label='Negative Log-Likelihood')
    
    ax2.set_xlabel('Predicted Probability for True Class')
    ax2.set_ylabel('Log-Likelihood', color='g')
    ax2_twin.set_ylabel('Negative Log-Likelihood', color='m')
    ax2.set_title('Relationship between Log-Likelihood and Negative Log-Likelihood (Cross-Entropy)')
    
    # Add combined legend
    lines = line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mle_cross_entropy_relationship.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Create the MLE and cross-entropy relationship visualization
create_mle_cross_entropy_visualization()

# Visualize the softmax function and probability distributions
def visualize_softmax():
    # Create some sample logits (pre-softmax values)
    samples = 5
    features = 3  # Three classes
    
    # Different variations of logits to visualize softmax behavior
    scenarios = [
        {"name": "Balanced", "logits": np.array([[1, 1, 1]] * samples)},
        {"name": "Strong class 0", "logits": np.array([[5, 1, 1]] * samples)},
        {"name": "Strong class 1", "logits": np.array([[1, 5, 1]] * samples)},
        {"name": "Strong class 2", "logits": np.array([[1, 1, 5]] * samples)},
        {"name": "Mixed", "logits": np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5], [3, 3, 1], [1, 3, 3]])},
    ]
    
    # Create a figure to visualize softmax outputs
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 3*len(scenarios)))
    
    # Process each scenario
    for i, scenario in enumerate(scenarios):
        logits = scenario["logits"]
        probs = softmax(logits)
        
        ax = axes[i]
        width = 0.2  # Bar width
        
        for j in range(samples):
            x = np.arange(features) + j*width
            ax.bar(x, probs[min(j, len(probs)-1)], width=width, 
                  label=f'Sample {j+1}' if i == 0 else None)
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title(f'Softmax Probabilities: {scenario["name"]} Scenario')
        ax.set_xticks(np.arange(features) + width*(samples-1)/2)
        ax.set_xticklabels(['Class 0', 'Class 1', 'Class 2'])
        
        # Only show legend for the first subplot
        if i == 0:
            ax.legend(ncol=samples, bbox_to_anchor=(0, 1.02, 1, 0.2), 
                     loc="lower left", mode="expand")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'softmax_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Visualize softmax function
visualize_softmax()

# Print key mathematical formulations
print("\nMathematical Formulations:")
print("--------------------------")

print("\n1. Softmax Function:")
print("P(y=k|x) = softmax(z_k) = exp(z_k) / Σ_j exp(z_j)")
print("where z_k = w_k^T x + b_k are the logits")

print("\n2. Likelihood Function:")
print("L(w,b) = ∏_{i=1}^{n} ∏_{k=1}^{K} [P(y^{(i)}=k|x^{(i)})]^{y_k^{(i)}}")
print("where y_k^{(i)} is 1 if the i-th example belongs to class k, 0 otherwise")

print("\n3. Log-Likelihood Function:")
print("ℓ(w,b) = ∑_{i=1}^{n} ∑_{k=1}^{K} y_k^{(i)} log(P(y^{(i)}=k|x^{(i)}))")

print("\n4. Cross-Entropy Loss:")
print("CE = -1/n ∑_{i=1}^{n} ∑_{k=1}^{K} y_k^{(i)} log(P(y^{(i)}=k|x^{(i)}))")

print("\n5. Negative Log-Likelihood:")
print("NLL = -ℓ(w,b) = -∑_{i=1}^{n} ∑_{k=1}^{K} y_k^{(i)} log(P(y^{(i)}=k|x^{(i)}))")
print("= n * CE")

print("\nHence, maximizing the log-likelihood is equivalent to minimizing the cross-entropy loss.")

print(f"\nAll visualizations saved to: {save_dir}") 