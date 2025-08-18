import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_44")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Question 44: AdaBoost and Exponential Loss ===")
print("=" * 80)

class AdaBoostExponentialLoss:
    def __init__(self):
        print("\n1. UNDERSTANDING THE PROBLEM")
        print("-" * 50)
        print("We need to understand AdaBoost as a forward-stagewise additive modeling approach")
        print("that minimizes an exponential loss function.")
        print("\nKey concepts:")
        print("- Exponential loss function: L(y, F) = exp(-yF)")
        print("- Forward-stagewise additive modeling: sequentially add weak learners")
        print("- Weight update rule: w_{m+1}^{(i)} ∝ w_m^{(i)} * exp(-α_m * y_i * h_m(x_i))")
        print("- Final classifier: H(x) = sign(Σ α_m * h_m(x))")
        
        # Generate synthetic dataset
        self.setup_data()
        
    def setup_data(self):
        """Generate synthetic dataset for demonstration"""
        print("\n2. SETTING UP DEMONSTRATION DATA")
        print("-" * 50)
        
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                                  n_redundant=0, n_clusters_per_class=1, 
                                  class_sep=1.5, random_state=42)
        
        # Convert to binary labels {-1, 1}
        y = 2 * y - 1
        
        self.X, self.y = X, y
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount((y + 1) // 2)}")
        print(f"Labels: {np.unique(y)}")
        
    def exponential_loss_function(self):
        """Demonstrate the exponential loss function"""
        print("\n3. EXPONENTIAL LOSS FUNCTION")
        print("-" * 50)
        print("The exponential loss function is defined as:")
        print("L(y, F) = exp(-yF)")
        print("where:")
        print("- y ∈ {-1, +1} is the true label")
        print("- F is the predicted score (before sign function)")
        print("- F > 0 means prediction favors class +1")
        print("- F < 0 means prediction favors class -1")
        
        # Create visualization of exponential loss
        F_values = np.linspace(-3, 3, 1000)
        
        plt.figure(figsize=(12, 8))
        
        # Plot exponential loss for different y values
        plt.subplot(2, 2, 1)
        plt.plot(F_values, np.exp(-F_values), 'b-', linewidth=2, label='y = +1')
        plt.plot(F_values, np.exp(F_values), 'r-', linewidth=2, label='y = -1')
        plt.xlabel('$F$ (predicted score)')
        plt.ylabel('$L(y, F) = \\exp(-yF)$')
        plt.title('Exponential Loss Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot comparison with 0/1 loss
        plt.subplot(2, 2, 2)
        plt.plot(F_values, np.exp(-F_values), 'b-', linewidth=2, label='Exponential Loss (y=+1)')
        plt.plot(F_values, np.exp(F_values), 'r-', linewidth=2, label='Exponential Loss (y=-1)')
        
        # 0/1 loss (step function)
        zero_one_loss_pos = np.where(F_values >= 0, 0, 1)
        zero_one_loss_neg = np.where(F_values <= 0, 0, 1)
        plt.plot(F_values, zero_one_loss_pos, 'b--', linewidth=2, label='0/1 Loss (y=+1)')
        plt.plot(F_values, zero_one_loss_neg, 'r--', linewidth=2, label='0/1 Loss (y=-1)')
        
        plt.xlabel('$F$ (predicted score)')
        plt.ylabel('Loss')
        plt.title('Exponential vs 0/1 Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot exponential loss surface
        plt.subplot(2, 2, 3)
        F1, F2 = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
        
        # For y = +1
        exp_loss_pos = np.exp(-(F1 + F2))
        plt.contourf(F1, F2, exp_loss_pos, levels=20, cmap='Blues')
        plt.colorbar(label='Exponential Loss (y=+1)')
        plt.xlabel('$F_1$')
        plt.ylabel('$F_2$')
        plt.title('Exponential Loss Surface (y=+1)')
        
        # Plot exponential loss vs margin
        plt.subplot(2, 2, 4)
        margin = np.linspace(-3, 3, 1000)
        exp_loss = np.exp(-margin)
        zero_one = np.where(margin >= 0, 0, 1)
        
        plt.plot(margin, exp_loss, 'b-', linewidth=2, label='Exponential Loss')
        plt.plot(margin, zero_one, 'r--', linewidth=2, label='0/1 Loss')
        plt.xlabel('Margin = $yF$')
        plt.ylabel('Loss')
        plt.title('Loss vs Margin')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'exponential_loss_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate specific examples with detailed step-by-step calculations
        print("\nExponential Loss Examples with Detailed Calculations:")
        print("For y = +1:")
        F_examples = [-2, -1, 0, 1, 2]
        for F in F_examples:
            # Step 1: Calculate the margin yF
            margin = 1 * F  # y = +1
            # Step 2: Calculate the exponential loss
            loss = np.exp(-margin)
            # Step 3: Show the detailed calculation
            print(f"  F = {F:2d}:")
            print(f"    Step 1: margin = y × F = +1 × {F} = {margin}")
            print(f"    Step 2: L(+1, {F}) = exp(-margin) = exp(-{margin}) = {loss:.4f}")
            
        print("\nFor y = -1:")
        for F in F_examples:
            # Step 1: Calculate the margin yF
            margin = -1 * F  # y = -1
            # Step 2: Calculate the exponential loss
            loss = np.exp(-margin)
            # Step 3: Show the detailed calculation
            print(f"  F = {F:2d}:")
            print(f"    Step 1: margin = y × F = -1 × {F} = {margin}")
            print(f"    Step 2: L(-1, {F}) = exp(-margin) = exp(-{margin}) = {loss:.4f}")
            
        print("\nKey observations:")
        print("- When yF > 0 (correct prediction), loss decreases exponentially")
        print("- When yF < 0 (incorrect prediction), loss increases exponentially")
        print("- Exponential loss penalizes misclassifications more heavily than 0/1 loss")
        print("- The loss function is symmetric: L(+1, F) = L(-1, -F)")
        
        # Demonstrate the relationship between margin and loss
        print("\nMargin Analysis:")
        margin_values = np.linspace(-3, 3, 7)
        print("Margin (yF) | Exponential Loss | 0/1 Loss")
        print("-" * 40)
        for margin in margin_values:
            exp_loss = np.exp(-margin)
            zero_one_loss = 0 if margin > 0 else 1
            print(f"{margin:8.1f} | {exp_loss:14.4f} | {zero_one_loss:8d}")
        
    def adaboost_algorithm_demonstration(self):
        """Demonstrate AdaBoost algorithm step by step"""
        print("\n4. ADABOOST ALGORITHM STEP-BY-STEP")
        print("-" * 50)
        print("AdaBoost algorithm:")
        print("1. Initialize weights: w_1^{(i)} = 1/N for all i")
        print("2. For m = 1 to M:")
        print("   a. Train weak learner h_m on weighted data")
        print("   b. Compute weighted error: ε_m = Σ w_m^{(i)} * I(y_i ≠ h_m(x_i))")
        print("   c. Compute α_m = 0.5 * ln((1-ε_m)/ε_m)")
        print("   d. Update weights: w_{m+1}^{(i)} ∝ w_m^{(i)} * exp(-α_m * y_i * h_m(x_i))")
        print("3. Final classifier: H(x) = sign(Σ α_m * h_m(x))")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Initialize weights
        N = len(X_train)
        w = np.ones(N) / N
        print(f"\nInitial weights: w_1 = {w[:5]}... (sum = {w.sum():.6f})")
        
        # AdaBoost iterations
        M = 5
        alphas = []
        weak_learners = []
        training_errors = []
        
        print(f"\nRunning AdaBoost for {M} iterations:")
        
        for m in range(M):
            print(f"\n--- Iteration {m+1} ---")
            
            # Step 1: Train weak learner on weighted data
            print(f"  Step 1: Training weak learner {m+1} on weighted data")
            weak_learner = DecisionTreeClassifier(max_depth=1, random_state=m)
            weak_learner.fit(X_train, y_train, sample_weight=w)
            
            # Step 2: Make predictions
            print(f"  Step 2: Making predictions with weak learner {m+1}")
            predictions = weak_learner.predict(X_train)
            
            # Step 3: Compute weighted error
            print(f"  Step 3: Computing weighted error ε_{m+1}")
            misclassified = (predictions != y_train)
            epsilon_m = np.sum(w * misclassified)
            print(f"    Number of misclassified samples: {np.sum(misclassified)}")
            print(f"    Weighted error: ε_{m+1} = Σ w^(i) × I(y_i ≠ h_{m+1}(x_i)) = {epsilon_m:.4f}")
            
            # Step 4: Compute alpha (weight of weak learner)
            print(f"  Step 4: Computing alpha α_{m+1}")
            if epsilon_m > 0.5:
                epsilon_m = 0.49  # Prevent numerical issues
                print(f"    Adjusted ε_{m+1} to 0.49 to prevent numerical issues")
            
            # Detailed alpha calculation
            ratio = (1 - epsilon_m) / epsilon_m
            alpha_m = 0.5 * np.log(ratio)
            print(f"    Ratio: (1 - ε_{m+1}) / ε_{m+1} = (1 - {epsilon_m:.4f}) / {epsilon_m:.4f} = {ratio:.4f}")
            print(f"    Alpha: α_{m+1} = 0.5 × ln({ratio:.4f}) = {alpha_m:.4f}")
            
            # Step 5: Update sample weights
            print(f"  Step 5: Updating sample weights")
            # Calculate weight update factors
            weight_factors = np.exp(-alpha_m * y_train * predictions)
            w_new = w * weight_factors
            w_new = w_new / np.sum(w_new)  # Normalize
            
            # Analyze weight changes
            weight_changes = w_new - w
            increased_weights = np.sum(weight_changes > 0)
            decreased_weights = np.sum(weight_changes < 0)
            
            print(f"    Weight update rule: w_{m+2}^(i) ∝ w_{m+1}^(i) × exp(-α_{m+1} × y_i × h_{m+1}(x_i))")
            print(f"    Weights increased for {increased_weights} samples")
            print(f"    Weights decreased for {decreased_weights} samples")
            print(f"    Weight sum after normalization: {w_new.sum():.6f}")
            print(f"    Max weight: {w_new.max():.4f}, Min weight: {w_new.min():.4f}")
            
            # Store results
            alphas.append(alpha_m)
            weak_learners.append(weak_learner)
            training_errors.append(epsilon_m)
            
            # Update weights for next iteration
            w = w_new
            
        # Final classifier
        print(f"\n--- Final Classifier ---")
        print(f"Alphas: {[f'{a:.4f}' for a in alphas]}")
        print(f"Training errors: {[f'{e:.4f}' for e in training_errors]}")
        
        # Make final predictions
        final_scores = np.zeros(len(X_test))
        for m in range(M):
            final_scores += alphas[m] * weak_learners[m].predict(X_test)
        
        final_predictions = np.sign(final_scores)
        accuracy = np.mean(final_predictions == y_test)
        
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Visualize weight evolution
        self.visualize_weight_evolution(w, alphas, training_errors)
        
        # Visualize decision boundaries
        self.visualize_decision_boundaries(weak_learners, alphas)
        
        return alphas, weak_learners, final_scores
        
    def visualize_weight_evolution(self, final_weights, alphas, training_errors):
        """Visualize how weights evolve during AdaBoost"""
        print("\n5. VISUALIZING WEIGHT EVOLUTION")
        print("-" * 50)
        
        plt.figure(figsize=(15, 10))
        
        # Plot training errors
        plt.subplot(2, 3, 1)
        iterations = range(1, len(training_errors) + 1)
        plt.plot(iterations, training_errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Iteration')
        plt.ylabel('Weighted Error $\\epsilon_m$')
        plt.title('Training Error Evolution')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random guess')
        plt.legend()
        
        # Plot alphas
        plt.subplot(2, 3, 2)
        plt.plot(iterations, alphas, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Iteration')
        plt.ylabel('Alpha $\\alpha_m$')
        plt.title('Alpha Values Evolution')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot weight distribution evolution
        plt.subplot(2, 3, 3)
        weight_history = []
        N = len(self.X)
        w = np.ones(N) / N
        
        for m in range(len(alphas)):
            weight_history.append(w.copy())
            if m < len(alphas):
                # Simulate weight update
                predictions = np.random.choice([-1, 1], N)  # Simplified
                w = w * np.exp(-alphas[m] * self.y * predictions)
                w = w / np.sum(w)
        
        weight_history = np.array(weight_history)
        
        # Plot weight distributions
        for m in range(min(5, len(weight_history))):
            plt.hist(weight_history[m], bins=20, alpha=0.6, 
                    label=f'Iteration {m+1}', density=True)
        
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.title('Weight Distribution Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot exponential loss vs iteration
        plt.subplot(2, 3, 4)
        cumulative_scores = np.zeros(len(self.X))
        exp_losses = []
        
        for m in range(len(alphas)):
            # Simplified: assume weak learner predictions
            predictions = np.random.choice([-1, 1], len(self.X))
            cumulative_scores += alphas[m] * predictions
            
            # Compute exponential loss
            exp_loss = np.mean(np.exp(-self.y * cumulative_scores))
            exp_losses.append(exp_loss)
        
        plt.plot(iterations, exp_losses, 'mo-', linewidth=2, markersize=8)
        plt.xlabel('Iteration')
        plt.ylabel('Average Exponential Loss')
        plt.title('Exponential Loss Evolution')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Plot margin distribution
        plt.subplot(2, 3, 5)
        margins = self.y * cumulative_scores
        plt.hist(margins, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Margin = $yF$')
        plt.ylabel('Frequency')
        plt.title('Final Margin Distribution')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Decision boundary')
        plt.legend()
        
        # Plot weight update rule visualization
        plt.subplot(2, 3, 6)
        margin_range = np.linspace(-3, 3, 100)
        alpha_example = 1.0  # Example alpha value
        
        weight_update = np.exp(-alpha_example * margin_range)
        plt.plot(margin_range, weight_update, 'b-', linewidth=2)
        plt.xlabel('Margin = $y_i h_m(x_i)$')
        plt.ylabel('Weight Update Factor')
        plt.title(f'Weight Update Rule ($\\alpha = {alpha_example}$)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'adaboost_weight_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_decision_boundaries(self, weak_learners, alphas):
        """Visualize decision boundaries of weak learners and final ensemble"""
        print("\n6. VISUALIZING DECISION BOUNDARIES")
        print("-" * 50)
        
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        plt.figure(figsize=(20, 12))
        
        # Plot individual weak learners
        for m in range(min(6, len(weak_learners))):
            plt.subplot(2, 3, m+1)
            
            # Get predictions for mesh grid
            Z = weak_learners[m].predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            plt.contour(xx, yy, Z, colors='k', linewidths=1)
            
            # Plot training data
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, 
                                cmap='RdYlBu', edgecolors='k', s=50)
            
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title(f'Weak Learner {m+1} ($\\alpha = {alphas[m]:.3f}$)')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weak_learners_decision_boundaries.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot final ensemble decision boundary
        plt.figure(figsize=(12, 10))
        
        # Compute final ensemble predictions
        Z_final = np.zeros(xx.shape)
        for m in range(len(weak_learners)):
            predictions = weak_learners[m].predict(np.c_[xx.ravel(), yy.ravel()])
            Z_final += alphas[m] * predictions.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z_final, alpha=0.4, cmap='RdYlBu')
        plt.contour(xx, yy, Z_final, levels=[0], colors='k', linewidths=3)
        
        # Plot training data
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, 
                            cmap='RdYlBu', edgecolors='k', s=50)
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Final AdaBoost Ensemble Decision Boundary')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Class')
        
        # Add equation
        alpha_sum = sum(alphas)
        plt.text(0.02, 0.98, f'Final classifier: H(x) = sign($\\sum \\alpha_m h_m(x)$)\nTotal $\\alpha = {alpha_sum:.3f}$', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_ensemble_decision_boundary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def calculate_specific_example(self):
        """Calculate the specific example from the question"""
        print("\n7. CALCULATING SPECIFIC EXAMPLE")
        print("-" * 50)
        print("Given: Final classifier output H(x) = -0.5 for a sample with true label y = +1")
        print("Calculate the exponential loss for this sample.")
        
        y = 1
        H_x = -0.5
        
        # Exponential loss: L(y, F) = exp(-yF)
        exponential_loss = np.exp(-y * H_x)
        
        print(f"True label: y = {y}")
        print(f"Classifier output: H(x) = {H_x}")
        
        # Step-by-step calculation
        print(f"\nStep-by-step calculation:")
        print(f"Step 1: Calculate the margin y × H(x)")
        margin = y * H_x
        print(f"  margin = y × H(x) = {y} × {H_x} = {margin}")
        
        print(f"Step 2: Determine if the prediction is correct")
        is_correct = margin > 0
        print(f"  Since margin = {margin} {'>' if margin > 0 else '<'} 0, prediction is {'correct' if is_correct else 'incorrect'}")
        
        print(f"Step 3: Calculate exponential loss")
        exponential_loss = np.exp(-margin)
        print(f"  L({y}, {H_x}) = exp(-margin) = exp(-{margin}) = {exponential_loss:.4f}")
        
        # Compare with 0/1 loss
        zero_one_loss = 1 if margin < 0 else 0
        
        print(f"\nStep 4: Compare with 0/1 loss")
        print(f"  0/1 loss: L_{0/1}({y}, {H_x}) = {'1' if margin < 0 else '0'} (since margin = {margin} {'<' if margin < 0 else '>'} 0)")
        print(f"  Exponential loss: L_exp({y}, {H_x}) = {exponential_loss:.4f}")
        
        print(f"\nStep 5: Analyze the penalty difference")
        penalty_ratio = exponential_loss / max(zero_one_loss, 0.001)  # Avoid division by zero
        print(f"  Penalty ratio: exponential_loss / 0/1_loss = {exponential_loss:.4f} / {zero_one_loss} = {penalty_ratio:.4f}")
        
        print(f"\nInterpretation:")
        print(f"- The sample is misclassified (y × H(x) = {margin} < 0)")
        print(f"- 0/1 loss gives a fixed penalty of {zero_one_loss}")
        print(f"- Exponential loss gives a penalty of {exponential_loss:.4f}")
        print(f"- The exponential penalty is {penalty_ratio:.2f}x larger than the 0/1 penalty")
        print(f"- This encourages the algorithm to focus more on this misclassified sample")
        print(f"- In the next iteration, this sample's weight will increase by a factor of {np.exp(1.0 * abs(margin)):.2f} (assuming α = 1.0)")
        
        # Visualize this specific case
        plt.figure(figsize=(10, 6))
        
        F_values = np.linspace(-2, 2, 1000)
        exp_loss_pos = np.exp(-F_values)
        exp_loss_neg = np.exp(F_values)
        
        plt.plot(F_values, exp_loss_pos, 'b-', linewidth=2, label='y = +1')
        plt.plot(F_values, exp_loss_neg, 'r-', linewidth=2, label='y = -1')
        
        # Highlight the specific point
        plt.plot(H_x, exponential_loss, 'ko', markersize=10, label=f'Point: ({H_x}, {exponential_loss:.4f})')
        
        plt.xlabel('$F$ (predicted score)')
        plt.ylabel('$L(y, F) = \\exp(-yF)$')
        plt.title('Exponential Loss Function with Specific Example')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Add annotation
        plt.annotate(f'y = +1, H(x) = {H_x}\nLoss = {exponential_loss:.4f}', 
                    xy=(H_x, exponential_loss), xytext=(H_x + 0.5, exponential_loss + 0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'specific_example_calculation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return exponential_loss
        
    def run_complete_analysis(self):
        """Run the complete analysis"""
        print("\n" + "="*80)
        print("RUNNING COMPLETE ANALYSIS")
        print("="*80)
        
        # Run all components
        self.exponential_loss_function()
        alphas, weak_learners, final_scores = self.adaboost_algorithm_demonstration()
        self.calculate_specific_example()
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"All plots saved to: {save_dir}")
        
        return alphas, weak_learners, final_scores

# Run the analysis
if __name__ == "__main__":
    adaboost_analysis = AdaBoostExponentialLoss()
    alphas, weak_learners, final_scores = adaboost_analysis.run_complete_analysis()
