import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_43")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Question 43: Bagging vs Boosting - Bias-Variance Trade-off ===")
print("=" * 80)

class BiasVarianceAnalysis:
    def __init__(self):
        print("\n1. UNDERSTANDING THE PROBLEM")
        print("-" * 50)
        print("We need to compare how Bagging and Boosting affect the bias-variance trade-off.")
        print("Key concepts:")
        print("- Bias: How far off the model's predictions are on average")
        print("- Variance: How much the model's predictions vary for different training sets")
        print("- Bagging: Bootstrap Aggregating - trains multiple models independently")
        print("- Boosting: Sequentially trains models, each focusing on previous errors")
        
        # Generate synthetic datasets
        self.setup_data()
        
    def setup_data(self):
        """Generate synthetic datasets for demonstration"""
        print("\n2. SETTING UP DEMONSTRATION DATA")
        print("-" * 50)
        
        # Classification dataset
        np.random.seed(42)
        X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                          n_redundant=5, n_clusters_per_class=1, random_state=42)
        self.X_clf, self.y_clf = X_clf, y_clf
        
        # Regression dataset
        X_reg, y_reg = make_regression(n_samples=1000, n_features=10, n_informative=8, 
                                      noise=0.5, random_state=42)
        self.X_reg, self.y_reg = X_reg, y_reg
        
        print(f"Classification dataset: {X_clf.shape[0]} samples, {X_clf.shape[1]} features")
        print(f"Regression dataset: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
        
        # Calculate dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Classification - Class distribution: {np.bincount(y_clf)}")
        print(f"Regression - Target mean: {y_reg.mean():.4f}, std: {y_reg.std():.4f}")
        
    def demonstrate_bagging(self):
        """Demonstrate Bagging and its effect on bias-variance"""
        print("\n3. DEMONSTRATING BAGGING (BOOTSTRAP AGGREGATING)")
        print("-" * 50)
        print("Bagging trains multiple models independently on bootstrap samples of the data.")
        print("Key characteristics:")
        print("- Models are trained in parallel")
        print("- Each model sees a different subset of data")
        print("- Final prediction is the average/majority vote")
        print("- Reduces variance without increasing bias")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.3, random_state=42
        )
        
        print(f"\nData split: Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Base learner (high-variance, low-bias)
        base_learner = DecisionTreeClassifier(max_depth=10, random_state=42)
        
        # Bagging ensemble
        bagging = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                       random_state=42, bootstrap=True)
        
        # Train and evaluate
        print("\nTraining models...")
        base_learner.fit(X_train, y_train)
        bagging.fit(X_train, y_train)
        
        # Cross-validation scores to estimate bias and variance
        print("\nPerforming 5-fold cross-validation...")
        base_scores = cross_val_score(base_learner, X_train, y_train, cv=5)
        bagging_scores = cross_val_score(bagging, X_train, y_train, cv=5)
        
        print(f"\nBase Learner (Decision Tree) CV scores: {base_scores}")
        print(f"Base Learner mean accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
        print(f"Bagging Ensemble CV scores: {bagging_scores}")
        print(f"Bagging Ensemble mean accuracy: {bagging_scores.mean():.4f} ± {bagging_scores.std():.4f}")
        
        # Detailed bias-variance calculation
        print("\nDetailed Bias-Variance Calculation for Bagging:")
        print("-" * 40)
        
        # Bias = 1 - mean accuracy (for classification)
        base_bias = 1 - base_scores.mean()
        bagging_bias = 1 - bagging_scores.mean()
        
        # Variance = variance of scores
        base_variance = base_scores.var()
        bagging_variance = bagging_scores.var()
        
        print(f"Base Learner:")
        print(f"  Bias = 1 - {base_scores.mean():.4f} = {base_bias:.4f}")
        print(f"  Variance = {base_variance:.6f}")
        print(f"  Standard Deviation = {base_scores.std():.4f}")
        
        print(f"\nBagging Ensemble:")
        print(f"  Bias = 1 - {bagging_scores.mean():.4f} = {bagging_bias:.4f}")
        print(f"  Variance = {bagging_variance:.6f}")
        print(f"  Standard Deviation = {bagging_scores.std():.4f}")
        
        print(f"\nImprovement Analysis:")
        bias_improvement = base_bias - bagging_bias
        variance_improvement = base_variance - bagging_variance
        print(f"  Bias improvement: {base_bias:.4f} - {bagging_bias:.4f} = {bias_improvement:.4f}")
        print(f"  Variance improvement: {base_variance:.6f} - {bagging_variance:.6f} = {variance_improvement:.6f}")
        print(f"  Relative bias improvement: {(bias_improvement/base_bias)*100:.1f}%")
        print(f"  Relative variance improvement: {(variance_improvement/base_variance)*100:.1f}%")
        
        # Test set performance
        base_pred = base_learner.predict(X_test)
        bagging_pred = bagging.predict(X_test)
        
        base_acc = accuracy_score(y_test, base_pred)
        bagging_acc = accuracy_score(y_test, bagging_pred)
        
        print(f"\nTest set performance:")
        print(f"Base Learner accuracy: {base_acc:.4f}")
        print(f"Bagging Ensemble accuracy: {bagging_acc:.4f}")
        print(f"Absolute improvement: {bagging_acc - base_acc:.4f}")
        print(f"Relative improvement: {((bagging_acc - base_acc)/base_acc)*100:.1f}%")
        
        # Store results for visualization
        self.bagging_results = {
            'base_scores': base_scores,
            'bagging_scores': bagging_scores,
            'base_acc': base_acc,
            'bagging_acc': base_acc,
            'base_bias': base_bias,
            'bagging_bias': bagging_bias,
            'base_variance': base_variance,
            'bagging_variance': bagging_variance
        }
        
        return base_learner, bagging
        
    def demonstrate_boosting(self):
        """Demonstrate Boosting and its effect on bias-variance"""
        print("\n4. DEMONSTRATING BOOSTING")
        print("-" * 50)
        print("Boosting trains models sequentially, each focusing on previous errors.")
        print("Key characteristics:")
        print("- Models are trained sequentially")
        print("- Each model focuses on samples misclassified by previous models")
        print("- Final prediction is weighted combination of all models")
        print("- Reduces bias but can increase variance")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.3, random_state=42
        )
        
        # Base learner (high-bias, low-variance)
        base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)  # Decision stump
        
        # Boosting ensemble
        boosting = AdaBoostClassifier(estimator=base_learner, n_estimators=100, 
                                    random_state=42)
        
        # Train and evaluate
        print("\nTraining models...")
        base_learner.fit(X_train, y_train)
        boosting.fit(X_train, y_train)
        
        # Cross-validation scores
        print("\nPerforming 5-fold cross-validation...")
        base_scores = cross_val_score(base_learner, X_train, y_train, cv=5)
        boosting_scores = cross_val_score(boosting, X_train, y_train, cv=5)
        
        print(f"\nBase Learner (Decision Stump) CV scores: {base_scores}")
        print(f"Base Learner mean accuracy: {base_scores.mean():.4f} ± {base_scores.std():.4f}")
        print(f"Boosting Ensemble CV scores: {boosting_scores}")
        print(f"Boosting Ensemble mean accuracy: {boosting_scores.mean():.4f} ± {boosting_scores.std():.4f}")
        
        # Detailed bias-variance calculation
        print("\nDetailed Bias-Variance Calculation for Boosting:")
        print("-" * 40)
        
        # Bias = 1 - mean accuracy (for classification)
        base_bias = 1 - base_scores.mean()
        boosting_bias = 1 - boosting_scores.mean()
        
        # Variance = variance of scores
        base_variance = base_scores.var()
        boosting_variance = boosting_scores.var()
        
        print(f"Base Learner (Decision Stump):")
        print(f"  Bias = 1 - {base_scores.mean():.4f} = {base_bias:.4f}")
        print(f"  Variance = {base_variance:.6f}")
        print(f"  Standard Deviation = {base_scores.std():.4f}")
        
        print(f"\nBoosting Ensemble:")
        print(f"  Bias = 1 - {boosting_scores.mean():.4f} = {boosting_bias:.4f}")
        print(f"  Bias = 1 - {boosting_scores.mean():.4f} = {boosting_bias:.4f}")
        print(f"  Variance = {boosting_variance:.6f}")
        print(f"  Standard Deviation = {boosting_scores.std():.4f}")
        
        print(f"\nImprovement Analysis:")
        bias_improvement = base_bias - boosting_bias
        variance_change = boosting_variance - base_variance
        print(f"  Bias improvement: {base_bias:.4f} - {boosting_bias:.4f} = {bias_improvement:.4f}")
        print(f"  Variance change: {boosting_variance:.6f} - {base_variance:.6f} = {variance_change:+.6f}")
        print(f"  Relative bias improvement: {(bias_improvement/base_bias)*100:.1f}%")
        if variance_change > 0:
            print(f"  Relative variance increase: {(variance_change/base_variance)*100:.1f}%")
        else:
            print(f"  Relative variance improvement: {(-variance_change/base_variance)*100:.1f}%")
        
        # Test set performance
        base_pred = base_learner.predict(X_test)
        boosting_pred = boosting.predict(X_test)
        
        base_acc = accuracy_score(y_test, base_pred)
        boosting_acc = accuracy_score(y_test, boosting_pred)
        
        print(f"\nTest set performance:")
        print(f"Base Learner accuracy: {base_acc:.4f}")
        print(f"Boosting Ensemble accuracy: {boosting_acc:.4f}")
        print(f"Absolute improvement: {boosting_acc - base_acc:.4f}")
        print(f"Relative improvement: {((boosting_acc - base_acc)/base_acc)*100:.1f}%")
        
        # Store results for visualization
        self.boosting_results = {
            'base_scores': base_scores,
            'boosting_scores': boosting_scores,
            'base_acc': base_acc,
            'boosting_acc': boosting_acc,
            'base_bias': base_bias,
            'boosting_bias': boosting_bias,
            'base_variance': base_variance,
            'boosting_variance': boosting_variance
        }
        
        return base_learner, boosting
        
    def analyze_bias_variance_tradeoff(self):
        """Analyze the bias-variance trade-off in detail"""
        print("\n5. DETAILED BIAS-VARIANCE ANALYSIS")
        print("-" * 50)
        
        # Calculate bias and variance estimates
        print("Estimating bias and variance from cross-validation scores...")
        
        # For bagging
        base_bias_bag = self.bagging_results['base_bias']
        base_var_bag = self.bagging_results['base_variance']
        bagging_bias = self.bagging_results['bagging_bias']
        bagging_var = self.bagging_results['bagging_variance']
        
        # For boosting
        base_bias_boost = self.boosting_results['base_bias']
        base_var_boost = self.boosting_results['base_variance']
        boosting_bias = self.boosting_results['boosting_bias']
        boosting_var = self.boosting_results['boosting_variance']
        
        print(f"\nBAGGING ANALYSIS:")
        print(f"Base Learner - Bias: {base_bias_bag:.4f}, Variance: {base_var_bag:.6f}")
        print(f"Bagging Ensemble - Bias: {bagging_bias:.4f}, Variance: {bagging_var:.6f}")
        print(f"Bias change: {bagging_bias - base_bias_bag:+.4f}")
        print(f"Variance change: {bagging_var - base_var_bag:+.6f}")
        
        print(f"\nBOOSTING ANALYSIS:")
        print(f"Base Learner - Bias: {base_bias_boost:.4f}, Variance: {base_var_boost:.6f}")
        print(f"Boosting Ensemble - Bias: {boosting_bias:.4f}, Variance: {boosting_var:.6f}")
        print(f"Bias change: {boosting_bias - base_bias_boost:+.4f}")
        print(f"Variance change: {boosting_var - base_var_boost:+.6f}")
        
        # Mathematical analysis
        print(f"\nMATHEMATICAL ANALYSIS:")
        print("-" * 30)
        
        # Total error approximation (Bias² + Variance)
        base_total_error_bag = base_bias_bag**2 + base_var_bag
        bagging_total_error = bagging_bias**2 + bagging_var
        
        base_total_error_boost = base_bias_boost**2 + base_var_boost
        boosting_total_error = boosting_bias**2 + boosting_var
        
        print(f"Bagging Total Error (Bias² + Variance):")
        print(f"  Base Learner: {base_bias_bag:.4f}² + {base_var_bag:.6f} = {base_total_error_bag:.6f}")
        print(f"  Ensemble: {bagging_bias:.4f}² + {bagging_var:.6f} = {bagging_total_error:.6f}")
        print(f"  Improvement: {base_total_error_bag - bagging_total_error:.6f}")
        
        print(f"\nBoosting Total Error (Bias² + Variance):")
        print(f"  Base Learner: {base_bias_boost:.4f}² + {base_var_boost:.6f} = {base_total_error_boost:.6f}")
        print(f"  Ensemble: {boosting_bias:.4f}² + {boosting_var:.6f} = {boosting_total_error:.6f}")
        print(f"  Improvement: {base_total_error_boost - boosting_total_error:.6f}")
        
        # Store for visualization
        self.bias_variance_data = {
            'bagging': {
                'base': {'bias': base_bias_bag, 'variance': base_var_bag, 'total_error': base_total_error_bag},
                'ensemble': {'bias': bagging_bias, 'variance': bagging_var, 'total_error': bagging_total_error}
            },
            'boosting': {
                'base': {'bias': base_bias_boost, 'variance': base_var_boost, 'total_error': base_total_error_boost},
                'ensemble': {'bias': boosting_bias, 'variance': boosting_var, 'total_error': boosting_total_error}
            }
        }
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n6. CREATING VISUALIZATIONS")
        print("-" * 50)
        
        # 1. Cross-validation score comparison
        self.plot_cv_comparison()
        
        # 2. Bias-variance trade-off visualization
        self.plot_bias_variance_tradeoff()
        
        # 3. Learning curves
        self.plot_learning_curves()
        
        # 4. Model complexity analysis
        self.plot_complexity_analysis()
        
        # 5. Total error analysis
        self.plot_total_error_analysis()
        
    def plot_cv_comparison(self):
        """Plot cross-validation score comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bagging CV scores
        ax1.boxplot([self.bagging_results['base_scores'], 
                     self.bagging_results['bagging_scores']], 
                    labels=['Base Learner\n(Decision Tree)', 'Bagging\nEnsemble'])
        ax1.set_title('Bagging: Cross-Validation Score Distribution')
        ax1.set_ylabel('Accuracy Score')
        ax1.grid(True, alpha=0.3)
        
        # Add statistical annotations
        base_mean = self.bagging_results['base_scores'].mean()
        base_std = self.bagging_results['base_scores'].std()
        bagging_mean = self.bagging_results['bagging_scores'].mean()
        bagging_std = self.bagging_results['bagging_scores'].std()
        
        ax1.annotate(f'Mean: {base_mean:.3f}\nStd: {base_std:.3f}', 
                    xy=(0.7, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        ax1.annotate(f'Mean: {bagging_mean:.3f}\nStd: {bagging_std:.3f}', 
                    xy=(0.7, 0.6), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Boosting CV scores
        ax2.boxplot([self.boosting_results['base_scores'], 
                     self.boosting_results['boosting_scores']], 
                    labels=['Base Learner\n(Decision Stump)', 'Boosting\nEnsemble'])
        ax2.set_title('Boosting: Cross-Validation Score Distribution')
        ax2.set_ylabel('Accuracy Score')
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotations
        base_mean = self.boosting_results['base_scores'].mean()
        base_std = self.boosting_results['base_scores'].std()
        boosting_mean = self.boosting_results['boosting_scores'].mean()
        boosting_std = self.boosting_results['boosting_scores'].std()
        
        ax2.annotate(f'Mean: {base_mean:.3f}\nStd: {base_std:.3f}', 
                    xy=(0.7, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        ax2.annotate(f'Mean: {boosting_mean:.3f}\nStd: {boosting_std:.3f}', 
                    xy=(0.7, 0.6), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Accuracy comparison
        methods = ['Base Learner', 'Bagging', 'Base Learner', 'Boosting']
        accuracies = [self.bagging_results['base_acc'], self.bagging_results['bagging_acc'],
                     self.boosting_results['base_acc'], self.boosting_results['boosting_acc']]
        colors = ['lightcoral', 'lightblue', 'lightcoral', 'lightgreen']
        
        bars = ax3.bar(methods, accuracies, color=colors, alpha=0.7)
        ax3.set_title('Test Set Accuracy Comparison')
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Variance comparison
        variances = [self.bagging_results['base_variance'], 
                     self.bagging_results['bagging_variance'],
                     self.boosting_results['base_variance'], 
                     self.boosting_results['boosting_variance']]
        
        bars = ax4.bar(methods, variances, color=colors, alpha=0.7)
        ax4.set_title('Cross-Validation Score Variance')
        ax4.set_ylabel('Variance')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, var in zip(bars, variances):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{var:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cv_comparison.png'), dpi=300, bbox_inches='tight')
        print("Saved: cv_comparison.png")
        
    def plot_bias_variance_tradeoff(self):
        """Plot bias-variance trade-off visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bagging bias-variance
        bagging_data = self.bias_variance_data['bagging']
        x_pos = np.arange(2)
        width = 0.35
        
        base_bias = [bagging_data['base']['bias'], bagging_data['base']['variance']]
        ensemble_bias = [bagging_data['ensemble']['bias'], bagging_data['ensemble']['variance']]
        
        ax1.bar(x_pos - width/2, base_bias, width, label='Base Learner', 
                color='lightcoral', alpha=0.7)
        ax1.bar(x_pos + width/2, ensemble_bias, width, label='Bagging Ensemble', 
                color='lightblue', alpha=0.7)
        
        ax1.set_xlabel('Metric')
        ax1.set_ylabel('Value')
        ax1.set_title('Bagging: Bias-Variance Trade-off')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Bias', 'Variance'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (base, ensemble) in enumerate(zip(base_bias, ensemble_bias)):
            ax1.text(i - width/2, base + 0.001, f'{base:.4f}', ha='center', va='bottom')
            ax1.text(i + width/2, ensemble + 0.001, f'{ensemble:.4f}', ha='center', va='bottom')
        
        # Boosting bias-variance
        boosting_data = self.bias_variance_data['boosting']
        base_bias = [boosting_data['base']['bias'], boosting_data['base']['variance']]
        ensemble_bias = [boosting_data['ensemble']['bias'], boosting_data['ensemble']['variance']]
        
        ax2.bar(x_pos - width/2, base_bias, width, label='Base Learner', 
                color='lightcoral', alpha=0.7)
        ax2.bar(x_pos + width/2, ensemble_bias, width, label='Boosting Ensemble', 
                color='lightgreen', alpha=0.7)
        
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Value')
        ax2.set_title('Boosting: Bias-Variance Trade-off')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Bias', 'Variance'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (base, ensemble) in enumerate(zip(base_bias, ensemble_bias)):
            ax2.text(i - width/2, base + 0.001, f'{base:.4f}', ha='center', va='bottom')
            ax2.text(i + width/2, ensemble + 0.001, f'{ensemble:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'bias_variance_tradeoff.png'), dpi=300, bbox_inches='tight')
        print("Saved: bias_variance_tradeoff.png")
        
    def plot_total_error_analysis(self):
        """Plot total error analysis (Bias² + Variance)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bagging total error
        bagging_data = self.bias_variance_data['bagging']
        x_pos = np.arange(3)
        width = 0.35
        
        base_metrics = [bagging_data['base']['bias']**2, bagging_data['base']['variance'], 
                       bagging_data['base']['total_error']]
        ensemble_metrics = [bagging_data['ensemble']['bias']**2, bagging_data['ensemble']['variance'], 
                           bagging_data['ensemble']['total_error']]
        
        ax1.bar(x_pos - width/2, base_metrics, width, label='Base Learner', 
                color='lightcoral', alpha=0.7)
        ax1.bar(x_pos + width/2, ensemble_metrics, width, label='Bagging Ensemble', 
                color='lightblue', alpha=0.7)
        
        ax1.set_xlabel('Error Component')
        ax1.set_ylabel('Value')
        ax1.set_title('Bagging: Total Error Decomposition')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Bias²', 'Variance', 'Total Error'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (base, ensemble) in enumerate(zip(base_metrics, ensemble_metrics)):
            ax1.text(i - width/2, base + 0.0001, f'{base:.6f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width/2, ensemble + 0.0001, f'{ensemble:.6f}', ha='center', va='bottom', fontsize=8)
        
        # Boosting total error
        boosting_data = self.bias_variance_data['boosting']
        base_metrics = [boosting_data['base']['bias']**2, boosting_data['base']['variance'], 
                       boosting_data['base']['total_error']]
        ensemble_metrics = [boosting_data['ensemble']['bias']**2, boosting_data['ensemble']['variance'], 
                           boosting_data['ensemble']['total_error']]
        
        ax2.bar(x_pos - width/2, base_metrics, width, label='Base Learner', 
                color='lightcoral', alpha=0.7)
        ax2.bar(x_pos + width/2, ensemble_metrics, width, label='Boosting Ensemble', 
                color='lightgreen', alpha=0.7)
        
        ax2.set_xlabel('Error Component')
        ax2.set_ylabel('Value')
        ax2.set_title('Boosting: Total Error Decomposition')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Bias²', 'Variance', 'Total Error'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (base, ensemble) in enumerate(zip(base_metrics, ensemble_metrics)):
            ax2.text(i - width/2, base + 0.0001, f'{base:.6f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i + width/2, ensemble + 0.0001, f'{ensemble:.6f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'total_error_analysis.png'), dpi=300, bbox_inches='tight')
        print("Saved: total_error_analysis.png")
        
    def plot_learning_curves(self):
        """Plot learning curves to show how ensemble size affects performance"""
        print("\nGenerating learning curves...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.3, random_state=42
        )
        
        # Test different ensemble sizes
        ensemble_sizes = [1, 5, 10, 25, 50, 100]
        
        bagging_scores = []
        boosting_scores = []
        
        for n_estimators in ensemble_sizes:
            # Bagging
            bagging = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, 
                                           random_state=42, bootstrap=True)
            bagging.fit(X_train, y_train)
            bagging_scores.append(accuracy_score(y_test, bagging.predict(X_test)))
            
            # Boosting
            base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
            boosting = AdaBoostClassifier(estimator=base_learner, 
                                        n_estimators=n_estimators, random_state=42)
            boosting.fit(X_train, y_train)
            boosting_scores.append(accuracy_score(y_test, boosting.predict(X_test)))
        
        # Plot learning curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bagging learning curve
        ax1.plot(ensemble_sizes, bagging_scores, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Estimators')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Bagging: Learning Curve')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(ensemble_sizes))
        
        # Add value labels
        for i, score in enumerate(bagging_scores):
            ax1.annotate(f'{score:.3f}', (ensemble_sizes[i], score), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Boosting learning curve
        ax2.plot(ensemble_sizes, boosting_scores, 's-', color='green', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Estimators')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Boosting: Learning Curve')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(ensemble_sizes))
        
        # Add value labels
        for i, score in enumerate(boosting_scores):
            ax2.annotate(f'{score:.3f}', (ensemble_sizes[i], score), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        print("Saved: learning_curves.png")
        
    def plot_complexity_analysis(self):
        """Plot how model complexity affects bias-variance trade-off"""
        print("\nAnalyzing model complexity effects...")
        
        # Test different tree depths
        depths = [1, 2, 3, 5, 7, 10, 15, 20]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_clf, self.y_clf, test_size=0.3, random_state=42
        )
        
        base_scores = []
        bagging_scores = []
        boosting_scores = []
        
        for depth in depths:
            # Base learner
            base = DecisionTreeClassifier(max_depth=depth, random_state=42)
            base.fit(X_train, y_train)
            base_scores.append(accuracy_score(y_test, base.predict(X_test)))
            
            # Bagging
            bagging = RandomForestClassifier(n_estimators=50, max_depth=depth, 
                                           random_state=42, bootstrap=True)
            bagging.fit(X_train, y_train)
            bagging_scores.append(accuracy_score(y_test, bagging.predict(X_test)))
            
            # Boosting
            base_learner = DecisionTreeClassifier(max_depth=depth, random_state=42)
            boosting = AdaBoostClassifier(estimator=base_learner, 
                                        n_estimators=50, random_state=42)
            boosting.fit(X_train, y_train)
            boosting_scores.append(accuracy_score(y_test, boosting.predict(X_test)))
        
        # Plot complexity analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy vs complexity
        ax1.plot(depths, base_scores, 'o-', color='red', linewidth=2, markersize=8, 
                label='Base Learner')
        ax1.plot(depths, bagging_scores, 's-', color='blue', linewidth=2, markersize=8, 
                label='Bagging')
        ax1.plot(depths, boosting_scores, '^-', color='green', linewidth=2, markersize=8, 
                label='Boosting')
        ax1.set_xlabel('Tree Depth (Complexity)')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Model Complexity vs Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bias-Variance trade-off visualization
        # For simplicity, we'll use depth as a proxy for complexity
        # and show the general trend
        complexity_levels = ['Low', 'Medium', 'High']
        bias_trend = [0.3, 0.2, 0.1]  # Bias decreases with complexity
        variance_trend = [0.1, 0.2, 0.4]  # Variance increases with complexity
        
        x_pos = np.arange(len(complexity_levels))
        width = 0.35
        
        ax2.bar(x_pos - width/2, bias_trend, width, label='Bias', 
                color='lightcoral', alpha=0.7)
        ax2.bar(x_pos + width/2, variance_trend, width, label='Variance', 
                color='lightblue', alpha=0.7)
        
        ax2.set_xlabel('Model Complexity')
        ax2.set_ylabel('Value')
        ax2.set_title('General Bias-Variance Trade-off')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(complexity_levels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'complexity_analysis.png'), dpi=300, bbox_inches='tight')
        print("Saved: complexity_analysis.png")
        
    def provide_detailed_explanations(self):
        """Provide detailed explanations for each question"""
        print("\n7. DETAILED EXPLANATIONS FOR EACH QUESTION")
        print("=" * 80)
        
        print("\nQuestion 1: What is the primary effect of Bagging on the bias-variance trade-off?")
        print("-" * 70)
        print("ANSWER: Bagging primarily REDUCES VARIANCE without significantly affecting bias.")
        print("\nDetailed Explanation:")
        print("• Bagging works by training multiple models on bootstrap samples of the data")
        print("• Each model sees a different subset of the training data")
        print("• The final prediction is the average (regression) or majority vote (classification)")
        print("• This averaging process reduces the variance of predictions")
        print("• The bias remains approximately the same as the base learner")
        print(f"• In our demonstration: Base learner variance: {self.bagging_results['base_variance']:.6f}")
        print(f"  Bagging ensemble variance: {self.bagging_results['bagging_variance']:.6f}")
        print(f"  Variance reduction: {self.bagging_results['base_variance'] - self.bagging_results['bagging_variance']:.6f}")
        print(f"  Relative variance reduction: {((self.bagging_results['base_variance'] - self.bagging_results['bagging_variance'])/self.bagging_results['base_variance'])*100:.1f}%")
        
        print("\nQuestion 2: What is the primary effect of Boosting on the bias-variance trade-off?")
        print("-" * 70)
        print("ANSWER: Boosting primarily REDUCES BIAS but can increase variance.")
        print("\nDetailed Explanation:")
        print("• Boosting trains models sequentially, each focusing on previous errors")
        print("• Each new model is trained to correct the mistakes of previous models")
        print("• This iterative correction process reduces the overall bias")
        print("• However, the ensemble can become more sensitive to noise, increasing variance")
        print(f"• In our demonstration: Base learner bias: {self.boosting_results['base_bias']:.4f}")
        print(f"  Boosting ensemble bias: {self.boosting_results['boosting_bias']:.4f}")
        print(f"  Bias reduction: {self.boosting_results['base_bias'] - self.boosting_results['boosting_bias']:.4f}")
        print(f"  Relative bias reduction: {((self.boosting_results['base_bias'] - self.boosting_results['boosting_bias'])/self.boosting_results['base_bias'])*100:.1f}%")
        
        print("\nQuestion 3: Why are high-variance, low-bias models like deep decision trees good base learners for Bagging?")
        print("-" * 70)
        print("ANSWER: High-variance, low-bias models benefit most from variance reduction.")
        print("\nDetailed Explanation:")
        print("• High-variance models are sensitive to the specific training data")
        print("• When trained on different bootstrap samples, they produce diverse predictions")
        print("• Bagging's averaging process reduces this variance effectively")
        print("• The low bias ensures the ensemble maintains good accuracy")
        print("• In our demonstration, the deep decision tree (max_depth=10) shows high variance")
        print(f"  Base variance: {self.bagging_results['base_variance']:.6f}")
        print(f"  which is effectively reduced by the bagging ensemble to: {self.bagging_results['bagging_variance']:.6f}")
        
        print("\nQuestion 4: Why are high-bias, low-variance models (weak learners) like decision stumps suitable for Boosting?")
        print("-" * 70)
        print("ANSWER: Weak learners allow Boosting to focus on specific error patterns.")
        print("\nDetailed Explanation:")
        print("• Weak learners have high bias but low variance")
        print("• They can focus on specific subsets of the data")
        print("• Boosting can combine multiple weak learners to reduce overall bias")
        print("• Each weak learner corrects specific mistakes of previous models")
        print("• In our demonstration, decision stumps (max_depth=1) are weak learners")
        print(f"  Base bias: {self.boosting_results['base_bias']:.4f}")
        print(f"  which Boosting combines effectively to reduce bias to: {self.boosting_results['boosting_bias']:.4f}")
        
        print("\nQuestion 5: If your model suffers from high bias, would you choose Bagging or Boosting? Justify your choice.")
        print("-" * 70)
        print("ANSWER: Choose BOOSTING when suffering from high bias.")
        print("\nDetailed Justification:")
        print("• High bias means the model is underfitting the data")
        print("• Bagging primarily reduces variance, not bias")
        print("• Boosting specifically targets bias reduction by focusing on errors")
        print("• Boosting can transform weak learners into a strong ensemble")
        print("• In our demonstration:")
        print(f"  - Base learner bias: {self.boosting_results['base_bias']:.4f}")
        print(f"  - Boosting reduced bias to: {self.boosting_results['boosting_bias']:.4f}")
        print(f"  - Total error improvement: {self.bias_variance_data['boosting']['base']['total_error']:.6f} → {self.bias_variance_data['boosting']['ensemble']['total_error']:.6f}")
        print("• However, be aware that Boosting may increase variance as a trade-off")
        
    def run_complete_analysis(self):
        """Run the complete analysis"""
        print("\n" + "="*80)
        print("RUNNING COMPLETE BIAS-VARIANCE ANALYSIS")
        print("="*80)
        
        # Run all demonstrations
        self.demonstrate_bagging()
        self.demonstrate_boosting()
        self.analyze_bias_variance_tradeoff()
        
        # Create visualizations
        self.create_visualizations()
        
        # Provide detailed explanations
        self.provide_detailed_explanations()
        
        print(f"\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print(f"All visualizations saved to: {save_dir}")
        print("="*80)

# Run the analysis
if __name__ == "__main__":
    analyzer = BiasVarianceAnalysis()
    analyzer.run_complete_analysis()
