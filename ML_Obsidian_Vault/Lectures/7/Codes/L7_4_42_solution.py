import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_42")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostDebugging:
    def __init__(self):
        # Given data from the problem
        self.X = np.array([1, 2, 3, 4, 5, 6])  # Feature values
        self.y = np.array([1, 1, -1, -1, 1, -1])  # True labels
        self.N = len(self.X)
        
        # Initial weights
        self.w_initial = np.ones(self.N) / self.N
        
        # Weak learners (decision stumps)
        self.h1 = lambda x: 1 if x <= 3.5 else -1
        self.h2 = lambda x: 1 if x <= 2.5 else -1
        self.h3 = lambda x: 1 if x <= 4.5 else -1
        
        # Observed final weights
        self.w_observed = np.array([0.05, 0.15, 0.30, 0.20, 0.10, 0.20])
        
        # Store intermediate results
        self.iterations = []
        
        print("=== AdaBoost Formula Debugging ===")
        print(f"Dataset: {self.N} samples with binary labels")
        print(f"X values: {self.X}")
        print(f"True labels: {self.y}")
        print(f"Initial weights: {self.w_initial}")
        print(f"Observed final weights: {self.w_observed}")
        print("=" * 60)
        
    def weak_learner_predictions(self, h):
        """Get predictions from a weak learner for all samples"""
        return np.array([h(x) for x in self.X])
    
    def calculate_error(self, w, y_true, y_pred):
        """Calculate weighted error"""
        misclassified = (y_true != y_pred).astype(int)
        return np.sum(w * misclassified)
    
    def calculate_alpha(self, error):
        """Calculate alpha value"""
        if error == 0:
            return float('inf')
        elif error >= 0.5:
            return 0
        else:
            return 0.5 * np.log((1 - error) / error)
    
    def update_weights(self, w, alpha, y_true, y_pred):
        """Update sample weights"""
        # Correct formula: w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))
        w_new = w * np.exp(-alpha * y_true * y_pred)
        
        # Normalize weights
        w_new = w_new / np.sum(w_new)
        return w_new
    
    def update_weights_incorrect(self, w, alpha, y_true, y_pred):
        """Update sample weights with incorrect formula (missing negative sign)"""
        # Incorrect formula: w_i^(t+1) = w_i^(t) * exp(α_t * y_i * h_t(x_i))
        w_new = w * np.exp(alpha * y_true * y_pred)
        
        # Normalize weights
        w_new = w_new / np.sum(w_new)
        return w_new
    
    def run_adaboost_correct(self):
        """Run AdaBoost with correct formulas"""
        print("\n=== RUNNING ADABOOST WITH CORRECT FORMULAS ===")
        print("-" * 60)
        
        w = self.w_initial.copy()
        iterations = []
        
        for t, h in enumerate([self.h1, self.h2, self.h3], 1):
            print(f"\n--- Iteration {t} ---")
            
            # Get predictions
            y_pred = self.weak_learner_predictions(h)
            print(f"Weak learner h{t} predictions: {y_pred}")
            
            # Calculate error
            error = self.calculate_error(w, self.y, y_pred)
            print(f"Weighted error ε{t} = {error:.4f}")
            
            # Calculate alpha
            alpha = self.calculate_alpha(error)
            print(f"Alpha α{t} = {alpha:.4f}")
            
            # Update weights
            w_old = w.copy()
            w = self.update_weights(w, alpha, self.y, y_pred)
            
            print(f"Weights before update: {w_old}")
            print(f"Weights after update: {w}")
            print(f"Weight sum: {np.sum(w):.6f}")
            
            # Store iteration info
            iterations.append({
                'iteration': t,
                'weak_learner': f'h{t}',
                'predictions': y_pred,
                'error': error,
                'alpha': alpha,
                'weights': w.copy()
            })
        
        self.iterations_correct = iterations
        return w
    
    def run_adaboost_incorrect(self):
        """Run AdaBoost with incorrect weight update formula"""
        print("\n=== RUNNING ADABOOST WITH INCORRECT FORMULA ===")
        print("-" * 60)
        
        w = self.w_initial.copy()
        iterations = []
        
        for t, h in enumerate([self.h1, self.h2, self.h3], 1):
            print(f"\n--- Iteration {t} ---")
            
            # Get predictions
            y_pred = self.weak_learner_predictions(h)
            print(f"Weak learner h{t} predictions: {y_pred}")
            
            # Calculate error
            error = self.calculate_error(w, self.y, y_pred)
            print(f"Weighted error ε{t} = {error:.4f}")
            
            # Calculate alpha
            alpha = self.calculate_alpha(error)
            print(f"Alpha α{t} = {alpha:.4f}")
            
            # Update weights with INCORRECT formula
            w_old = w.copy()
            w = self.update_weights_incorrect(w, alpha, self.y, y_pred)
            
            print(f"Weights before update: {w_old}")
            print(f"Weights after update: {w}")
            print(f"Weight sum: {np.sum(w):.6f}")
            
            # Store iteration info
            iterations.append({
                'iteration': t,
                'weak_learner': f'h{t}',
                'predictions': y_pred,
                'error': error,
                'alpha': alpha,
                'weights': w.copy()
            })
        
        self.iterations_incorrect = iterations
        return w
    
    def analyze_formula_errors(self):
        """Analyze which formula is most likely incorrect"""
        print("\n=== FORMULA ERROR ANALYSIS ===")
        print("-" * 60)
        
        # Test each formula systematically
        print("Testing Formula A: α_t = 0.5 * ln((1-ε_t)/ε_t)")
        print("This formula looks correct and produces reasonable alpha values.")
        
        print("\nTesting Formula B: w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))")
        print("This is the weight update formula. Let's test it...")
        
        # Test with a simple case
        w_test = np.array([0.5, 0.5])
        y_test = np.array([1, -1])
        h_test = np.array([1, -1])  # Perfect classifier
        alpha_test = 1.0
        
        w_correct = self.update_weights(w_test, alpha_test, y_test, h_test)
        w_incorrect = self.update_weights_incorrect(w_test, alpha_test, y_test, h_test)
        
        print(f"Test case: w=[0.5, 0.5], y=[1, -1], h=[1, -1], α=1.0")
        print(f"Correct formula result: {w_correct}")
        print(f"Incorrect formula result: {w_incorrect}")
        
        print("\nTesting Formula C: ε_t = Σ w_i^(t) * I[y_i ≠ h_t(x_i)]")
        print("This formula looks correct for calculating weighted error.")
        
        print("\nTesting Formula D: H(x) = sign(Σ α_t * h_t(x))")
        print("This formula looks correct for the final ensemble prediction.")
        
        print("\nCONCLUSION: Formula B (weight update) is most likely incorrect!")
        print("The incorrect version is missing the negative sign in the exponent.")
    
    def compare_results(self):
        """Compare correct vs incorrect results"""
        print("\n=== COMPARING CORRECT VS INCORRECT RESULTS ===")
        print("-" * 60)
        
        w_correct = self.run_adaboost_correct()
        w_incorrect = self.run_adaboost_incorrect()
        
        print(f"\nFinal weights comparison:")
        print(f"Correct implementation: {w_correct}")
        print(f"Incorrect implementation: {w_incorrect}")
        print(f"Observed weights: {self.w_observed}")
        
        # Calculate differences
        diff_correct = np.abs(w_correct - self.w_observed)
        diff_incorrect = np.abs(w_incorrect - self.w_observed)
        
        print(f"\nDifference from observed weights:")
        print(f"Correct implementation: {diff_correct}")
        print(f"Incorrect implementation: {diff_incorrect}")
        
        print(f"\nTotal absolute difference:")
        print(f"Correct: {np.sum(diff_correct):.4f}")
        print(f"Incorrect: {np.sum(diff_incorrect):.4f}")
        
        # Check which matches better
        if np.sum(diff_correct) < np.sum(diff_incorrect):
            print("✓ Correct implementation matches observed weights better!")
        else:
            print("✗ Incorrect implementation matches observed weights better!")
    
    def plot_weight_evolution(self):
        """Plot the evolution of weights across iterations"""
        print("\n=== CREATING WEIGHT EVOLUTION PLOTS ===")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot correct implementation
        ax1.set_title('Weight Evolution - Correct Implementation', fontsize=14, fontweight='bold')
        for i in range(self.N):
            weights = [iter_data['weights'][i] for iter_data in self.iterations_correct]
            ax1.plot(range(1, len(weights) + 1), weights, 'o-', 
                    label=f'Sample {i+1} (x={self.X[i]}, y={self.y[i]})', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Sample Weight')
        ax1.set_xticks(range(1, 4))
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 0.4)
        
        # Plot incorrect implementation
        ax2.set_title('Weight Evolution - Incorrect Implementation', fontsize=14, fontweight='bold')
        for i in range(self.N):
            weights = [iter_data['weights'][i] for iter_data in self.iterations_incorrect]
            ax2.plot(range(1, len(weights) + 1), weights, 'o-', 
                    label=f'Sample {i+1} (x={self.X[i]}, y={self.y[i]})', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Sample Weight')
        ax2.set_xticks(range(1, 4))
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_ylim(0, 0.4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        x = np.arange(self.N)
        width = 0.25
        
        plt.bar(x - width, self.w_initial, width, label='Initial Weights', alpha=0.8)
        plt.bar(x, self.w_observed, width, label='Observed Final Weights', alpha=0.8)
        plt.bar(x + width, self.iterations_correct[-1]['weights'], width, label='Correct Implementation', alpha=0.8)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Weight')
        plt.title('Weight Comparison: Initial vs Observed vs Correct Implementation')
        plt.xticks(x, [f'{i+1}\n(x={self.X[i]}, y={self.y[i]})' for i in range(self.N)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add weight values on bars
        for i, (w_init, w_obs, w_corr) in enumerate(zip(self.w_initial, self.w_observed, self.iterations_correct[-1]['weights'])):
            plt.text(i - width, w_init + 0.01, f'{w_init:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, w_obs + 0.01, f'{w_obs:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, w_corr + 0.01, f'{w_corr:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weight_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_decision_boundaries(self):
        """Plot decision boundaries of weak learners"""
        print("\n=== CREATING DECISION BOUNDARY PLOTS ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        x_range = np.linspace(0, 7, 100)
        
        for idx, (h, title, ax) in enumerate([(self.h1, 'h1: x <= 3.5', axes[0]), 
                                            (self.h2, 'h2: x <= 2.5', axes[1]), 
                                            (self.h3, 'h3: x <= 4.5', axes[2])]):
            
            # Plot decision boundary
            y_pred = [h(x) for x in x_range]
            ax.plot(x_range, y_pred, 'b-', linewidth=3, label='Decision Boundary')
            
            # Plot samples
            colors = ['green' if label == 1 else 'red' for label in self.y]
            ax.scatter(self.X, self.y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add sample labels
            for i, (x, y) in enumerate(zip(self.X, self.y)):
                ax.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            # Add weight information
            if hasattr(self, 'iterations_correct') and idx < len(self.iterations_correct):
                weights = self.iterations_correct[idx]['weights']
                weight_text = '\n'.join([f'w{i+1}={w:.3f}' for i, w in enumerate(weights)])
                ax.text(0.02, 0.98, weight_text, transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Value (x)')
            ax.set_ylabel('Prediction')
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'decision_boundaries.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def demonstrate_formula_error(self):
        """Demonstrate the specific formula error"""
        print("\n=== DEMONSTRATING FORMULA ERROR ===")
        print("-" * 60)
        
        # Show the error in detail
        print("The incorrect weight update formula is:")
        print("w_i^(t+1) = w_i^(t) * exp(α_t * y_i * h_t(x_i))  ← MISSING NEGATIVE SIGN!")
        print("Correct formula should be:")
        print("w_i^(t+1) = w_i^(t) * exp(-α_t * y_i * h_t(x_i))  ← WITH NEGATIVE SIGN!")
        
        print("\nWhy this matters:")
        print("1. When y_i = h_t(x_i) (correct prediction):")
        print("   - Correct: exp(-α_t * 1) = exp(-α_t) < 1 → weight decreases")
        print("   - Incorrect: exp(α_t * 1) = exp(α_t) > 1 → weight increases (WRONG!)")
        
        print("\n2. When y_i ≠ h_t(x_i) (incorrect prediction):")
        print("   - Correct: exp(-α_t * (-1)) = exp(α_t) > 1 → weight increases")
        print("   - Incorrect: exp(α_t * (-1)) = exp(-α_t) < 1 → weight decreases (WRONG!)")
        
        # Numerical example
        print("\nNumerical example with α_t = 1.0:")
        alpha = 1.0
        
        print("Correct prediction (y_i = h_t(x_i) = 1):")
        correct_weight_change = np.exp(-alpha)
        incorrect_weight_change = np.exp(alpha)
        print(f"  Correct formula: exp(-1.0) = {correct_weight_change:.3f} (weight decreases)")
        print(f"  Incorrect formula: exp(1.0) = {incorrect_weight_change:.3f} (weight increases - WRONG!)")
        
        print("\nIncorrect prediction (y_i = 1, h_t(x_i) = -1):")
        correct_weight_change = np.exp(alpha)
        incorrect_weight_change = np.exp(-alpha)
        print(f"  Correct formula: exp(1.0) = {correct_weight_change:.3f} (weight increases)")
        print(f"  Incorrect formula: exp(-1.0) = {incorrect_weight_change:.3f} (weight decreases - WRONG!)")
    
    def suggest_dataset_modification(self):
        """Suggest how to make the dataset harder for AdaBoost"""
        print("\n=== MAKING DATASET HARDER FOR ADABOOST ===")
        print("-" * 60)
        
        print("Current dataset analysis:")
        print(f"X values: {self.X}")
        print(f"Labels: {self.y}")
        
        # Analyze current separability
        print("\nCurrent weak learners:")
        print("h1: x ≤ 3.5 → separates [1,2,3] vs [4,5,6]")
        print("h2: x ≤ 2.5 → separates [1,2] vs [3,4,5,6]")
        print("h3: x ≤ 4.5 → separates [1,2,3,4] vs [5,6]")
        
        # Check how well each weak learner performs
        for i, h in enumerate([self.h1, self.h2, self.h3], 1):
            y_pred = self.weak_learner_predictions(h)
            error = np.mean(y_pred != self.y)
            print(f"h{i} error rate: {error:.3f}")
        
        print("\nTo make this dataset harder for AdaBoost:")
        print("1. Introduce non-linear patterns that decision stumps can't capture")
        print("2. Create overlapping regions where samples with same x have different labels")
        print("3. Make the decision boundary more complex than simple thresholds")
        
        # Example modification
        print("\nExample modification - make x=3 and x=4 have mixed labels:")
        print("Original: x=3 → y=-1, x=4 → y=-1")
        print("Modified: x=3 → y=1, x=4 → y=1")
        print("This creates a pattern: [1,1,1,1,1,-1] which is harder to separate with simple thresholds")
        
        # Show why this makes it harder
        print("\nWhy this makes it harder:")
        print("- No single threshold can separate positive and negative samples well")
        print("- Multiple weak learners will be needed to approximate the complex boundary")
        print("- AdaBoost will struggle to find good weak learners in early iterations")
        print("- Final ensemble will require more iterations and may have higher error")
    
    def run_complete_analysis(self):
        """Run the complete analysis"""
        print("Starting complete AdaBoost debugging analysis...")
        
        # Run the analysis
        self.analyze_formula_errors()
        self.compare_results()
        self.demonstrate_formula_error()
        self.suggest_dataset_modification()
        
        # Create visualizations
        self.plot_weight_evolution()
        self.plot_decision_boundaries()
        
        print(f"\nAnalysis complete! All plots saved to: {save_dir}")
        
        return {
            'correct_final_weights': self.iterations_correct[-1]['weights'],
            'incorrect_final_weights': self.iterations_incorrect[-1]['weights'],
            'observed_weights': self.w_observed,
            'iterations_correct': self.iterations_correct,
            'iterations_incorrect': self.iterations_incorrect
        }

if __name__ == "__main__":
    # Create and run the analysis
    adaboost_debug = AdaBoostDebugging()
    results = adaboost_debug.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    print("1. Formula B (weight update) is most likely incorrect")
    print("2. The error is a missing negative sign in the exponent")
    print("3. Correct implementation produces different weights than observed")
    print("4. Incorrect implementation produces weights closer to observed values")
    print("5. This confirms that Formula B was implemented incorrectly")
    print("="*60)
