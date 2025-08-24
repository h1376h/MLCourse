import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_27")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid Unicode issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostRace:
    def __init__(self):
        # Dataset: 4 samples with 2 features
        self.X = np.array([
            [1, 2],  # Sample 1: (x11=1, x12=2, y1=+1)
            [2, 1],  # Sample 2: (x21=2, x22=1, y2=+1)
            [3, 3],  # Sample 3: (x31=3, x32=3, y3=-1)
            [4, 4]   # Sample 4: (x41=4, x42=4, y4=-1)
        ])
        
        self.y = np.array([+1, +1, -1, -1])  # True labels
        
        # Weak learners
        self.weak_learners = [
            ("h1: x1 <= 2.5", lambda x: 1 if x[0] <= 2.5 else -1),
            ("h2: x2 <= 2.5", lambda x: 1 if x[1] <= 2.5 else -1),
            ("h3: x1 + x2 <= 5", lambda x: 1 if x[0] + x[1] <= 5 else -1)
        ]
        
        # Initial weights
        self.N = len(self.X)
        self.weights = np.ones(self.N) / self.N  # w1 = w2 = w3 = w4 = 0.25
        
        # Storage for iterations
        self.iterations = []
        
    def evaluate_weak_learner(self, weak_learner, weights):
        """Evaluate a weak learner and return predictions, errors, and weighted error"""
        name, learner = weak_learner
        predictions = np.array([learner(x) for x in self.X])
        
        # Calculate errors (1 if misclassified, 0 if correct)
        errors = (predictions != self.y).astype(int)
        
        # Calculate weighted error
        weighted_error = np.sum(weights * errors)
        
        return name, predictions, errors, weighted_error
    
    def find_best_weak_learner(self, weights, used_learners):
        """Find the best weak learner among unused ones"""
        best_learner = None
        best_error = float('inf')
        best_predictions = None
        best_errors = None
        
        for i, weak_learner in enumerate(self.weak_learners):
            if i not in used_learners:
                name, predictions, errors, weighted_error = self.evaluate_weak_learner(weak_learner, weights)
                
                if weighted_error < best_error:
                    best_error = weighted_error
                    best_learner = (i, weak_learner, predictions, errors)
        
        return best_learner
    
    def calculate_alpha(self, weighted_error):
        """Calculate the weight of the weak learner"""
        if weighted_error == 0:
            return 10  # Very large weight for perfect classifier
        elif weighted_error >= 0.5:
            return 0  # No weight for very poor classifier
        
        return 0.5 * np.log((1 - weighted_error) / weighted_error)
    
    def update_weights(self, weights, alpha, predictions, errors):
        """Update sample weights based on AdaBoost formula"""
        # w_i = w_i * exp(alpha * y_i * h(x_i))
        # Since errors[i] = 1 if misclassified, we can rewrite as:
        # w_i = w_i * exp(alpha * (1 - 2*errors[i]))
        
        new_weights = weights * np.exp(alpha * (1 - 2 * errors))
        
        # Normalize weights
        new_weights = new_weights / np.sum(new_weights)
        
        return new_weights
    
    def run_adaboost(self):
        """Run AdaBoost algorithm step by step"""
        print("=" * 80)
        print("ADABOOST ALGORITHM RACE - STEP BY STEP EXECUTION")
        print("=" * 80)
        
        # Initial setup
        print(f"\nINITIAL SETUP:")
        print(f"Dataset: {self.N} samples")
        print(f"Initial weights: {self.weights}")
        print(f"True labels: {self.y}")
        
        used_learners = set()
        final_alpha = []
        final_learners = []
        
        # First iteration
        print(f"\n{'='*50}")
        print("FIRST ITERATION")
        print(f"{'='*50}")
        
        print(f"\nStep 1: Evaluate all weak learners with current weights {self.weights}")
        print("-" * 60)
        
        for i, weak_learner in enumerate(self.weak_learners):
            name, predictions, errors, weighted_error = self.evaluate_weak_learner(weak_learner, self.weights)
            print(f"{name}:")
            print(f"  Predictions: {predictions}")
            print(f"  Errors: {errors}")
            print(f"  Weighted Error: {weighted_error:.4f}")
            print()
        
        # Find best weak learner
        print("Step 2: Find the best weak learner (lowest weighted error)")
        print("-" * 60)
        
        best_learner = self.find_best_weak_learner(self.weights, used_learners)
        learner_idx, weak_learner, predictions, errors = best_learner
        name = weak_learner[0]
        
        print(f"Best weak learner: {name}")
        print(f"Predictions: {predictions}")
        print(f"Errors: {errors}")
        
        # Calculate weighted error
        weighted_error = np.sum(self.weights * errors)
        print(f"Weighted Error: {weighted_error:.4f}")
        
        # Calculate alpha
        alpha = self.calculate_alpha(weighted_error)
        print(f"Alpha (learner weight): {alpha:.4f}")
        
        # Update weights
        print(f"\nStep 3: Update sample weights")
        print("-" * 60)
        print(f"Old weights: {self.weights}")
        
        new_weights = self.update_weights(self.weights, alpha, predictions, errors)
        print(f"New weights: {new_weights}")
        
        # Store iteration info
        self.iterations.append({
            'iteration': 1,
            'learner': name,
            'predictions': predictions,
            'errors': errors,
            'weighted_error': weighted_error,
            'alpha': alpha,
            'old_weights': self.weights.copy(),
            'new_weights': new_weights.copy()
        })
        
        # Update weights and mark learner as used
        self.weights = new_weights
        used_learners.add(learner_idx)
        final_alpha.append(alpha)
        final_learners.append(weak_learner)
        
        # Second iteration
        print(f"\n{'='*50}")
        print("SECOND ITERATION")
        print(f"{'='*50}")
        
        print(f"\nStep 1: Re-evaluate remaining weak learners with updated weights {self.weights}")
        print("-" * 60)
        
        for i, weak_learner in enumerate(self.weak_learners):
            if i not in used_learners:
                name, predictions, errors, weighted_error = self.evaluate_weak_learner(weak_learner, self.weights)
                print(f"{name}:")
                print(f"  Predictions: {predictions}")
                print(f"  Errors: {errors}")
                print(f"  Weighted Error: {weighted_error:.4f}")
                print()
        
        # Find best weak learner
        print("Step 2: Find the best weak learner among remaining ones")
        print("-" * 60)
        
        best_learner = self.find_best_weak_learner(self.weights, used_learners)
        learner_idx, weak_learner, predictions, errors = best_learner
        name = weak_learner[0]
        
        print(f"Best weak learner: {name}")
        print(f"Predictions: {predictions}")
        print(f"Errors: {errors}")
        
        # Calculate weighted error
        weighted_error = np.sum(self.weights * errors)
        print(f"Weighted Error: {weighted_error:.4f}")
        
        # Calculate alpha
        alpha = self.calculate_alpha(weighted_error)
        print(f"Alpha (learner weight): {alpha:.4f}")
        
        # Update weights
        print(f"\nStep 3: Update sample weights")
        print("-" * 60)
        print(f"Old weights: {self.weights}")
        
        new_weights = self.update_weights(self.weights, alpha, predictions, errors)
        print(f"New weights: {new_weights}")
        
        # Store iteration info
        self.iterations.append({
            'iteration': 2,
            'learner': name,
            'predictions': predictions,
            'errors': weighted_error,
            'weighted_error': weighted_error,
            'alpha': alpha,
            'old_weights': self.weights.copy(),
            'new_weights': new_weights.copy()
        })
        
        # Update weights and mark learner as used
        self.weights = new_weights
        used_learners.add(learner_idx)
        final_alpha.append(alpha)
        final_learners.append(weak_learner)
        
        # Final combination
        print(f"\n{'='*50}")
        print("FINAL COMBINATION")
        print(f"{'='*50}")
        
        print(f"\nStep 4: Combine weak learners with their weights")
        print("-" * 60)
        
        for i, (alpha, learner) in enumerate(zip(final_alpha, final_learners)):
            print(f"Learner {i+1}: {learner[0]} with weight α_{i+1} = {alpha:.4f}")
        
        # Final prediction
        print(f"\nStep 5: Final ensemble prediction")
        print("-" * 60)
        
        final_predictions = np.zeros(self.N)
        for alpha, learner in zip(final_alpha, final_learners):
            predictions = np.array([learner[1](x) for x in self.X])
            final_predictions += alpha * predictions
        
        # Convert to binary predictions
        final_binary = np.sign(final_predictions)
        
        print(f"Final predictions (before sign): {final_predictions}")
        print(f"Final binary predictions: {final_binary}")
        print(f"True labels: {self.y}")
        print(f"Final accuracy: {np.mean(final_binary == self.y):.2f}")
        
        # Analysis of hardest samples
        print(f"\n{'='*50}")
        print("ANALYSIS OF HARDEST SAMPLES")
        print(f"{'='*50}")
        
        print(f"\nWeight evolution across iterations:")
        print("-" * 60)
        
        for i in range(self.N):
            print(f"Sample {i+1} (x={self.X[i]}, y={self.y[i]}):")
            print(f"  Initial weight: {self.iterations[0]['old_weights'][i]:.4f}")
            for j, iteration in enumerate(self.iterations):
                print(f"  After iteration {j+1}: {iteration['new_weights'][i]:.4f}")
            print()
        
        # Find hardest sample (highest final weight)
        hardest_sample_idx = np.argmax(self.weights)
        print(f"Hardest sample to classify: Sample {hardest_sample_idx + 1}")
        print(f"  Features: {self.X[hardest_sample_idx]}")
        print(f"  True label: {self.y[hardest_sample_idx]}")
        print(f"  Final weight: {self.weights[hardest_sample_idx]:.4f}")
        
        return self.iterations, final_alpha, final_learners
    
    def visualize_iterations(self):
        """Create visualizations for each iteration"""
        
        # 1. Dataset visualization
        plt.figure(figsize=(15, 10))
        
        # Plot samples
        colors = ['green' if y == 1 else 'red' for y in self.y]
        markers = ['o' if y == 1 else 's' for y in self.y]
        
        for i, (x, y, color, marker) in enumerate(zip(self.X, self.y, colors, markers)):
            plt.scatter(x[0], x[1], s=200, c=color, marker=marker, 
                       edgecolors='black', linewidth=2, label=f'Sample {i+1} (y={y})')
        
        # Plot decision boundaries for each weak learner
        x1_range = np.linspace(0, 5, 100)
        
        # h1: x1 <= 2.5
        plt.axvline(x=2.5, color='blue', linestyle='--', alpha=0.7, label='h1: x1 <= 2.5')
        
        # h2: x2 <= 2.5
        plt.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='h2: x2 <= 2.5')
        
        # h3: x1 + x2 <= 5
        x2_h3 = 5 - x1_range
        plt.plot(x1_range, x2_h3, 'purple', linestyle='--', alpha=0.7, label='h3: x1 + x2 <= 5')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Dataset and Weak Learners')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        
        # Add sample coordinates
        for i, x in enumerate(self.X):
            plt.annotate(f'({x[0]}, {x[1]})', (x[0], x[1]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'dataset_and_weak_learners.png'), dpi=300, bbox_inches='tight')
        
        # 2. Weight evolution visualization
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each iteration
        for i, iteration in enumerate(self.iterations):
            plt.subplot(2, 2, i+1)
            
            # Bar plot of weights
            x_pos = np.arange(self.N)
            bars = plt.bar(x_pos, iteration['new_weights'], 
                          color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'],
                          edgecolor='black', linewidth=1)
            
            plt.xlabel('Sample Index')
            plt.ylabel('Weight')
            plt.title(f'Iteration {i+1}: {iteration["learner"]}\nalpha = {iteration["alpha"]:.4f}')
            plt.xticks(x_pos, [f'S{i+1}' for i in range(self.N)])
            plt.ylim(0, 0.6)
            
            # Add weight values on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_evolution.png'), dpi=300, bbox_inches='tight')
        
        # 3. Final ensemble visualization
        plt.figure(figsize=(12, 8))
        
        # Plot samples with final weights as size
        sizes = self.weights * 2000  # Scale weights for visualization
        
        for i, (x, y, size) in enumerate(zip(self.X, self.y, sizes)):
            color = 'green' if y == 1 else 'red'
            marker = 'o' if y == 1 else 's'
            plt.scatter(x[0], x[1], s=size, c=color, marker=marker, 
                       edgecolors='black', linewidth=2, alpha=0.7,
                       label=f'Sample {i+1} (y={y}, w={self.weights[i]:.4f})')
        
        # Plot decision boundaries with alpha weights
        x1_range = np.linspace(0, 5, 100)
        
        # Find which learners were used
        used_learners = [iter_info['learner'] for iter_info in self.iterations]
        alphas = [iter_info['alpha'] for iter_info in self.iterations]
        
        if 'h1: x1 <= 2.5' in used_learners:
            idx = used_learners.index('h1: x1 <= 2.5')
            alpha = alphas[idx]
            plt.axvline(x=2.5, color='blue', linestyle='--', alpha=0.7, 
                       linewidth=2, label=f'h1: x1 <= 2.5 (alpha={alpha:.4f})')
        
        if 'h2: x2 <= 2.5' in used_learners:
            idx = used_learners.index('h2: x2 <= 2.5')
            alpha = alphas[idx]
            plt.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7,
                       linewidth=2, label=f'h2: x2 <= 2.5 (alpha={alpha:.4f})')
        
        if 'h3: x1 + x2 <= 5' in used_learners:
            idx = used_learners.index('h3: x1 + x2 <= 5')
            alpha = alphas[idx]
            x2_h3 = 5 - x1_range
            plt.plot(x1_range, x2_h3, 'purple', linestyle='--', alpha=0.7,
                    linewidth=2, label=f'h3: x1 + x2 <= 5 (alpha={alpha:.4f})')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Final AdaBoost Ensemble\n(Sample sizes proportional to final weights)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        
        # Add sample coordinates
        for i, x in enumerate(self.X):
            plt.annotate(f'({x[0]}, {x[1]})', (x[0], x[1]), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_ensemble.png'), dpi=300, bbox_inches='tight')
        
        # 4. Weight progression visualization
        plt.figure(figsize=(12, 8))
        
        # Create weight progression matrix
        weight_matrix = np.zeros((len(self.iterations) + 1, self.N))
        weight_matrix[0] = self.iterations[0]['old_weights']  # Initial weights
        
        for i, iteration in enumerate(self.iterations):
            weight_matrix[i + 1] = iteration['new_weights']
        
        # Heatmap
        sns.heatmap(weight_matrix, 
                    annot=True, 
                    fmt='.4f',
                    xticklabels=[f'S{i+1}' for i in range(self.N)],
                    yticklabels=['Initial'] + [f'Iter {i+1}' for i in range(len(self.iterations))],
                    cmap='Blues',
                    cbar_kws={'label': 'Weight'})
        
        plt.title('Weight Evolution Across AdaBoost Iterations')
        plt.xlabel('Sample')
        plt.ylabel('Iteration')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_heatmap.png'), dpi=300, bbox_inches='tight')

def main():
    print("Starting AdaBoost Algorithm Race...")
    
    # Create AdaBoost instance
    adaboost = AdaBoostRace()
    
    # Run the algorithm
    iterations, final_alpha, final_learners = adaboost.run_adaboost()
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    adaboost.visualize_iterations()
    
    print(f"\nVisualizations saved to: {save_dir}")
    print("\nAdaBoost Algorithm Race completed successfully!")

if __name__ == "__main__":
    main()
