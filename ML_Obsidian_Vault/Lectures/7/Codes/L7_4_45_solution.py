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
save_dir = os.path.join(images_dir, "L7_4_Quiz_45")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("=== Question 45: AdaBoost and Margin Theory ===")
print("=" * 80)

class AdaBoostMarginTheory:
    def __init__(self):
        print("\n1. UNDERSTANDING THE PROBLEM")
        print("-" * 50)
        print("We need to understand how AdaBoost's strong generalization performance")
        print("is related to its effect on the classification margin.")
        print("\nKey concepts:")
        print("- Margin: measure of confidence in classification")
        print("- AdaBoost maximizes margins of training examples")
        print("- Larger margins lead to better generalization")
        print("- Margin theory explains why AdaBoost doesn't overfit")
        
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
        
    def margin_definition(self):
        """Explain and demonstrate margin definition"""
        print("\n3. MARGIN DEFINITION")
        print("-" * 50)
        print("For a given sample (x_i, y_i), the margin is defined as:")
        print("margin(x_i, y_i) = y_i * H(x_i)")
        print("where:")
        print("- H(x_i) is the normalized final classifier output")
        print("- H(x_i) = Σ(α_m * h_m(x_i)) / Σ|α_m|")
        print("- y_i ∈ {-1, +1} is the true label")
        print("- margin ∈ [-1, +1]")
        
        # Train AdaBoost classifier
        print("\nTraining AdaBoost classifier...")
        self.adaboost = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=10,
            random_state=42
        )
        self.adaboost.fit(self.X, self.y)
        
        # Calculate margins
        self.calculate_margins()
        
    def calculate_margins(self):
        """Calculate margins for all training samples"""
        print("\n4. CALCULATING MARGINS")
        print("-" * 50)
        
        # Get AdaBoost predictions and margins
        predictions = self.adaboost.decision_function(self.X)
        margins = self.y * predictions
        
        # Normalize margins to [-1, 1] range
        max_score = np.max(np.abs(predictions))
        normalized_margins = margins / max_score
        
        self.margins = margins
        self.normalized_margins = normalized_margins
        self.predictions = predictions
        
        print(f"Raw predictions range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        print(f"Margins range: [{np.min(margins):.3f}, {np.max(margins):.3f}]")
        print(f"Normalized margins range: [{np.min(normalized_margins):.3f}, {np.max(normalized_margins):.3f}]")
        
        # Show some examples
        print("\nSample margins:")
        for i in range(5):
            print(f"Sample {i}: x={self.X[i]}, y={self.y[i]}, margin={normalized_margins[i]:.3f}")
            
    def margin_interpretation(self):
        """Explain what positive and negative margins indicate"""
        print("\n5. MARGIN INTERPRETATION")
        print("-" * 50)
        print("Margin interpretation:")
        print("- Positive margin: sample is correctly classified with confidence")
        print("- Negative margin: sample is incorrectly classified")
        print("- Margin magnitude: confidence level of classification")
        print("- Margin = 0: sample is exactly on decision boundary")
        
        # Analyze margin distribution
        positive_margins = self.normalized_margins[self.normalized_margins > 0]
        negative_margins = self.normalized_margins[self.normalized_margins < 0]
        zero_margins = self.normalized_margins[self.normalized_margins == 0]
        
        print(f"\nMargin distribution:")
        print(f"Positive margins: {len(positive_margins)} samples")
        print(f"Negative margins: {len(negative_margins)} samples")
        print(f"Zero margins: {len(zero_margins)} samples")
        
        if len(positive_margins) > 0:
            print(f"Positive margin range: [{np.min(positive_margins):.3f}, {np.max(positive_margins):.3f}]")
        if len(negative_margins) > 0:
            print(f"Negative margin range: [{np.min(negative_margins):.3f}, {np.max(negative_margins):.3f}]")
            
    def adaboost_margin_maximization(self):
        """Demonstrate how AdaBoost maximizes margins"""
        print("\n6. ADABOOST MARGIN MAXIMIZATION")
        print("-" * 50)
        print("AdaBoost focuses on misclassified samples to maximize margins:")
        print("- Misclassified samples get higher weights in next iteration")
        print("- New weak learners focus on these hard examples")
        print("- Process continues until margins are maximized")
        
        # Show margin evolution across iterations
        self.show_margin_evolution()
        
    def show_margin_evolution(self):
        """Show how margins evolve during AdaBoost training"""
        print("\n7. MARGIN EVOLUTION DURING TRAINING")
        print("-" * 50)
        
        # Train AdaBoost with different numbers of estimators
        n_estimators_list = [1, 2, 5, 10, 20, 50]
        margin_evolution = []
        
        for n_est in n_estimators_list:
            adaboost_temp = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=1, random_state=42),
                n_estimators=n_est,
                random_state=42
            )
            adaboost_temp.fit(self.X, self.y)
            
            predictions_temp = adaboost_temp.decision_function(self.X)
            margins_temp = self.y * predictions_temp
            max_score_temp = np.max(np.abs(predictions_temp))
            normalized_margins_temp = margins_temp / max_score_temp
            
            margin_evolution.append(normalized_margins_temp)
            
        self.margin_evolution = margin_evolution
        self.n_estimators_list = n_estimators_list
        
        # Calculate statistics
        print("Margin statistics across iterations:")
        for i, n_est in enumerate(n_estimators_list):
            margins = margin_evolution[i]
            min_margin = np.min(margins)
            mean_margin = np.mean(margins)
            max_margin = np.max(margins)
            print(f"{n_est:2d} estimators: min={min_margin:.3f}, mean={mean_margin:.3f}, max={max_margin:.3f}")
            
    def generalization_connection(self):
        """Explain connection between margins and generalization"""
        print("\n8. MARGINS AND GENERALIZATION")
        print("-" * 50)
        print("Larger margins suggest better generalization because:")
        print("- More confident predictions on training data")
        print("- Decision boundary is further from training points")
        print("- Less sensitive to small perturbations in data")
        print("- Better separation between classes")
        
        # Visualize margin distribution
        self.visualize_margins()
        
    def visualize_margins(self):
        """Create comprehensive visualizations"""
        print("\n9. CREATING VISUALIZATIONS")
        print("-" * 50)
        
        # Figure 1: Margin distribution
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Margin histogram
        plt.subplot(2, 3, 1)
        plt.hist(self.normalized_margins, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        plt.xlabel('Normalized Margin')
        plt.ylabel('Frequency')
        plt.title('Distribution of Training Sample Margins')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Margin vs prediction confidence
        plt.subplot(2, 3, 2)
        plt.scatter(np.abs(self.predictions), self.normalized_margins, 
                   c=self.y, cmap='RdYlBu', alpha=0.7, s=30)
        plt.xlabel('|Prediction Score|')
        plt.ylabel('Normalized Margin')
        plt.title('Margin vs Prediction Confidence')
        plt.colorbar(label='True Label')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Margin evolution
        plt.subplot(2, 3, 3)
        margin_means = [np.mean(margins) for margins in self.margin_evolution]
        margin_mins = [np.min(margins) for margins in self.margin_evolution]
        
        plt.plot(self.n_estimators_list, margin_means, 'b-o', linewidth=2, label='Mean Margin')
        plt.plot(self.n_estimators_list, margin_mins, 'r-s', linewidth=2, label='Min Margin')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Margin')
        plt.title('Margin Evolution During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Decision boundary with margins
        plt.subplot(2, 3, 4)
        self.plot_decision_boundary_with_margins()
        
        # Subplot 5: Margin vs distance to boundary
        plt.subplot(2, 3, 5)
        distances = np.abs(self.predictions)  # Distance to decision boundary
        plt.scatter(distances, self.normalized_margins, 
                   c=self.y, cmap='RdYlBu', alpha=0.7, s=30)
        plt.xlabel('Distance to Decision Boundary')
        plt.ylabel('Normalized Margin')
        plt.title('Margin vs Distance to Boundary')
        plt.colorbar(label='True Label')
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Margin distribution by class
        plt.subplot(2, 3, 6)
        class_1_margins = self.normalized_margins[self.y == 1]
        class_neg1_margins = self.normalized_margins[self.y == -1]
        
        plt.hist(class_1_margins, bins=15, alpha=0.7, label='Class +1', color='blue')
        plt.hist(class_neg1_margins, bins=15, alpha=0.7, label='Class -1', color='red')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
        plt.xlabel('Normalized Margin')
        plt.ylabel('Frequency')
        plt.title('Margin Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'margin_analysis.png'), dpi=300, bbox_inches='tight')
        
        # Figure 2: Detailed margin analysis
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Margin vs sample index
        plt.subplot(2, 2, 1)
        sample_indices = np.arange(len(self.normalized_margins))
        plt.scatter(sample_indices, self.normalized_margins, 
                   c=self.y, cmap='RdYlBu', alpha=0.7, s=30)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Margin')
        plt.title('Margins Across All Training Samples')
        plt.colorbar(label='True Label')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Margin distribution with decision boundary
        plt.subplot(2, 2, 2)
        plt.hist(self.normalized_margins, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary (margin=0)')
        plt.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='High Confidence (margin=0.5)')
        plt.axvline(x=-0.5, color='orange', linestyle='--', linewidth=2, label='Low Confidence (margin=-0.5)')
        plt.xlabel('Normalized Margin')
        plt.ylabel('Frequency')
        plt.title('Margin Distribution with Confidence Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Margin evolution heatmap
        plt.subplot(2, 2, 3)
        margin_matrix = np.array(self.margin_evolution)
        plt.imshow(margin_matrix, aspect='auto', cmap='RdYlBu', 
                   extent=[0, len(self.n_estimators_list), -1, 1])
        plt.colorbar(label='Normalized Margin')
        plt.xlabel('Number of Estimators')
        plt.ylabel('Normalized Margin')
        plt.title('Margin Evolution Heatmap')
        plt.yticks([-1, -0.5, 0, 0.5, 1])
        
        # Subplot 4: Margin statistics over iterations
        plt.subplot(2, 2, 4)
        margin_means = [np.mean(margins) for margins in self.margin_evolution]
        margin_stds = [np.std(margins) for margins in self.margin_evolution]
        
        plt.errorbar(self.n_estimators_list, margin_means, yerr=margin_stds, 
                    fmt='o-', linewidth=2, capsize=5, capthick=2)
        plt.xlabel('Number of Estimators')
        plt.ylabel('Mean Margin ± Std Dev')
        plt.title('Margin Statistics Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'detailed_margin_analysis.png'), dpi=300, bbox_inches='tight')
        
        print(f"Visualizations saved to: {save_dir}")
        
    def plot_decision_boundary_with_margins(self):
        """Plot decision boundary with margin information"""
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Get predictions for mesh grid
        Z = self.adaboost.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.3)
        
        # Plot training points with margin-based coloring
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], 
                            c=self.normalized_margins, cmap='RdYlBu', 
                            s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Decision Boundary with Sample Margins')
        plt.colorbar(scatter, label='Normalized Margin')
        plt.grid(True, alpha=0.3)
        
    def answer_questions(self):
        """Provide detailed answers to all questions"""
        print("\n10. DETAILED ANSWERS TO QUESTIONS")
        print("=" * 80)
        
        print("\nQuestion 1: Margin Definition")
        print("-" * 40)
        print("For a given sample (x_i, y_i), the margin is defined as:")
        print("margin(x_i, y_i) = y_i * H(x_i)")
        print("where H(x_i) is the normalized final classifier output.")
        print(f"In our example, margins range from {np.min(self.normalized_margins):.3f} to {np.max(self.normalized_margins):.3f}")
        
        print("\nQuestion 2: Margin Interpretation")
        print("-" * 40)
        positive_count = np.sum(self.normalized_margins > 0)
        negative_count = np.sum(self.normalized_margins < 0)
        print(f"Positive margins ({positive_count} samples): Correctly classified with confidence")
        print(f"Negative margins ({negative_count} samples): Incorrectly classified")
        print("Margin magnitude indicates confidence level")
        
        print("\nQuestion 3: AdaBoost Margin Maximization")
        print("-" * 40)
        print("AdaBoost focuses on misclassified samples by:")
        print("- Increasing weights of misclassified samples")
        print("- Training new weak learners on hard examples")
        print("- Iteratively improving margins")
        print(f"Final mean margin: {np.mean(self.normalized_margins):.3f}")
        
        print("\nQuestion 4: Generalization Connection")
        print("-" * 40)
        print("Larger margins suggest better generalization because:")
        print("- More confident predictions on training data")
        print("- Decision boundary is further from training points")
        print("- Less sensitive to data perturbations")
        print("- Better class separation")
        
        print("\nQuestion 5: Margin Near Decision Boundary")
        print("-" * 40)
        # Find samples close to decision boundary
        boundary_threshold = 0.1
        near_boundary = np.abs(self.normalized_margins) < boundary_threshold
        near_boundary_count = np.sum(near_boundary)
        print(f"Samples near decision boundary (|margin| < {boundary_threshold}): {near_boundary_count}")
        print("These samples have margins closer to 0, indicating low confidence")
        
        # Show examples
        if near_boundary_count > 0:
            near_boundary_indices = np.where(near_boundary)[0]
            print("Examples of samples near boundary:")
            for i in near_boundary_indices[:3]:
                print(f"  Sample {i}: margin={self.normalized_margins[i]:.3f}, prediction={self.predictions[i]:.3f}")

def main():
    """Main function to run the complete analysis"""
    print("Starting AdaBoost Margin Theory Analysis...")
    
    # Create and run analysis
    analysis = AdaBoostMarginTheory()
    analysis.margin_definition()
    analysis.margin_interpretation()
    analysis.adaboost_margin_maximization()
    analysis.generalization_connection()
    analysis.answer_questions()
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check the generated visualizations for detailed insights.")
    print(f"All files saved to: {save_dir}")

if __name__ == "__main__":
    main()
