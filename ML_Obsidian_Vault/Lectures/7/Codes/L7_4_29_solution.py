import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_29")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid Unicode issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostFeatureEngineering:
    def __init__(self):
        # Dataset: 8 students with 3 features for predicting pass/fail
        self.data = {
            'Student': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'Study_Hours': [2, 4, 6, 8, 3, 7, 5, 9],
            'Sleep_Hours': [6, 7, 8, 7, 5, 9, 6, 8],
            'Exercise_Score': [3, 5, 7, 8, 4, 6, 5, 9],
            'Pass': [0, 0, 1, 1, 0, 1, 0, 1]
        }
        
        self.df = pd.DataFrame(self.data)
        self.X = self.df[['Study_Hours', 'Sleep_Hours', 'Exercise_Score']].values
        self.y = np.array(self.data['Pass'])
        
        # Feature names for easier reference
        self.feature_names = ['Study_Hours', 'Sleep_Hours', 'Exercise_Score']
        
        # Initial weights (equal for all samples)
        self.N = len(self.X)
        self.initial_weights = np.ones(self.N) / self.N
        
        print("=== AdaBoost Feature Engineering Challenge ===")
        print("Dataset:")
        print(self.df.to_string(index=False))
        print(f"\nInitial weights: {self.initial_weights}")
        print("=" * 60)
        
    def find_optimal_thresholds(self):
        """Find optimal thresholds for each feature that minimize classification error"""
        print("\n1. FINDING OPTIMAL THRESHOLDS FOR EACH FEATURE")
        print("-" * 50)
        
        optimal_thresholds = {}
        optimal_errors = {}
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            print(f"\n{feature_name}:")
            print(f"Values: {sorted(self.X[:, feature_idx])}")
            
            # Get unique values for potential thresholds
            unique_values = sorted(set(self.X[:, feature_idx]))
            thresholds = []
            for i in range(len(unique_values) - 1):
                thresholds.append((unique_values[i] + unique_values[i + 1]) / 2)
            
            # Add boundary thresholds
            thresholds = [unique_values[0] - 0.5] + thresholds + [unique_values[-1] + 0.5]
            
            print(f"Potential thresholds: {[f'{t:.1f}' for t in thresholds]}")
            
            best_threshold = None
            best_error = float('inf')
            best_predictions = None
            
            for threshold in thresholds:
                # Test threshold: x <= threshold -> predict 0, x > threshold -> predict 1
                predictions = (self.X[:, feature_idx] > threshold).astype(int)
                errors = (predictions != self.y).astype(int)
                error_rate = np.mean(errors)
                
                print(f"  Threshold {threshold:.1f}: Predictions {predictions}, Errors {errors}, Error Rate {error_rate:.3f}")
                
                if error_rate < best_error:
                    best_error = error_rate
                    best_threshold = threshold
                    best_predictions = predictions
            
            # Also test the reverse direction: x <= threshold -> predict 1, x > threshold -> predict 0
            for threshold in thresholds:
                predictions = (self.X[:, feature_idx] <= threshold).astype(int)
                errors = (predictions != self.y).astype(int)
                error_rate = np.mean(errors)
                
                print(f"  Threshold {threshold:.1f} (reverse): Predictions {predictions}, Errors {errors}, Error Rate {error_rate:.3f}")
                
                if error_rate < best_error:
                    best_error = error_rate
                    best_threshold = threshold
                    best_predictions = predictions
                    reverse = True
                else:
                    reverse = False
            
            optimal_thresholds[feature_name] = (best_threshold, reverse)
            optimal_errors[feature_name] = best_error
            
            print(f"  → Best threshold: {best_threshold:.1f} (reverse: {reverse})")
            print(f"  → Best error rate: {best_error:.3f}")
        
        return optimal_thresholds, optimal_errors
    
    def create_decision_stumps(self, optimal_thresholds):
        """Create decision stump weak learners using optimal thresholds"""
        print("\n2. CREATING DECISION STUMP WEAK LEARNERS")
        print("-" * 50)
        
        weak_learners = []
        
        for feature_name, (threshold, reverse) in optimal_thresholds.items():
            feature_idx = self.feature_names.index(feature_name)
            
            if reverse:
                # x <= threshold -> predict 1, x > threshold -> predict 0
                def stump(x, t=threshold, idx=feature_idx):
                    return 1 if x[idx] <= t else 0
                description = f"{feature_name} <= {threshold:.1f} → 1, > {threshold:.1f} → 0"
            else:
                # x <= threshold -> predict 0, x > threshold -> predict 1
                def stump(x, t=threshold, idx=feature_idx):
                    return 1 if x[idx] > t else 0
                description = f"{feature_name} <= {threshold:.1f} → 0, > {threshold:.1f} → 1"
            
            weak_learners.append((description, stump, feature_name))
            print(f"Stump {len(weak_learners)}: {description}")
        
        return weak_learners
    
    def evaluate_weak_learners(self, weak_learners, weights):
        """Evaluate each weak learner and calculate weighted errors"""
        print("\n3. EVALUATING WEAK LEARNERS WITH WEIGHTED ERRORS")
        print("-" * 50)
        
        results = []
        
        for i, (description, stump, feature_name) in enumerate(weak_learners):
            # Get predictions
            predictions = np.array([stump(x) for x in self.X])
            
            # Calculate errors
            errors = (predictions != self.y).astype(int)
            
            # Calculate weighted error
            weighted_error = np.sum(weights * errors)
            
            # Calculate unweighted error rate
            unweighted_error = np.mean(errors)
            
            results.append({
                'index': i,
                'description': description,
                'feature': feature_name,
                'predictions': predictions,
                'errors': errors,
                'unweighted_error': unweighted_error,
                'weighted_error': weighted_error
            })
            
            print(f"\nStump {i+1} ({description}):")
            print(f"  Predictions: {predictions}")
            print(f"  Errors: {errors}")
            print(f"  Unweighted error rate: {unweighted_error:.3f}")
            print(f"  Weighted error: {weighted_error:.3f}")
            
            # Show detailed breakdown
            for j, (pred, true_label, weight) in enumerate(zip(predictions, self.y, weights)):
                student = self.data['Student'][j]
                error = errors[j]
                contribution = weight * error
                print(f"    Student {student}: Pred={pred}, True={true_label}, Weight={weight:.3f}, Error={error}, Contribution={contribution:.3f}")
        
        return results
    
    def find_best_weak_learner(self, results):
        """Find the best weak learner based on weighted error"""
        print("\n4. FINDING THE BEST WEAK LEARNER")
        print("-" * 50)
        
        best_result = min(results, key=lambda x: x['weighted_error'])
        
        print(f"Best weak learner: Stump {best_result['index']+1}")
        print(f"Description: {best_result['description']}")
        print(f"Feature: {best_result['feature']}")
        print(f"Weighted error: {best_result['weighted_error']:.3f}")
        print(f"Unweighted error rate: {best_result['unweighted_error']:.3f}")
        
        return best_result
    
    def analyze_feature_importance(self, results):
        """Analyze which feature is most important for predicting student success"""
        print("\n5. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 50)
        
        # Sort features by their best error rate
        feature_performance = {}
        for result in results:
            feature = result['feature']
            if feature not in feature_performance or result['unweighted_error'] < feature_performance[feature]['error']:
                feature_performance[feature] = {
                    'error': result['unweighted_error'],
                    'weighted_error': result['weighted_error'],
                    'description': result['description']
                }
        
        # Sort by unweighted error rate (lower is better)
        sorted_features = sorted(feature_performance.items(), key=lambda x: x[1]['error'])
        
        print("Feature ranking by classification performance:")
        for i, (feature, perf) in enumerate(sorted_features):
            print(f"{i+1}. {feature}: Error rate = {perf['error']:.3f}, Weighted error = {perf['weighted_error']:.3f}")
            print(f"   Best stump: {perf['description']}")
        
        most_important = sorted_features[0][0]
        print(f"\nMost important feature: {most_important}")
        print(f"Reason: Lowest classification error rate ({feature_performance[most_important]['error']:.3f})")
        
        return sorted_features
    
    def visualize_results(self, results, optimal_thresholds):
        """Create visualizations for the analysis"""
        print("\n6. CREATING VISUALIZATIONS")
        print("-" * 50)
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AdaBoost Feature Engineering Challenge Analysis', fontsize=16, fontweight='bold')
        
        # 1. Dataset visualization
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=100, cmap='RdYlBu', alpha=0.7)
        ax1.set_xlabel('Study Hours')
        ax1.set_ylabel('Sleep Hours')
        ax1.set_title('Study vs Sleep Hours\n(Pass: Blue, Fail: Red)')
        ax1.grid(True, alpha=0.3)
        
        # Add student labels
        for i, student in enumerate(self.data['Student']):
            ax1.annotate(student, (self.X[i, 0], self.X[i, 1]), xytext=(5, 5), textcoords='offset points')
        
        # 2. Feature distributions
        ax2 = axes[0, 1]
        for i, feature_name in enumerate(self.feature_names):
            ax2.hist([self.X[self.y == 0, i], self.X[self.y == 1, i]], 
                    label=[f'{feature_name} (Fail)', f'{feature_name} (Pass)'], 
                    alpha=0.7, bins=5)
        ax2.set_xlabel('Feature Values')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Feature Distributions by Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error rates comparison
        ax3 = axes[0, 2]
        features = [r['feature'] for r in results]
        unweighted_errors = [r['unweighted_error'] for r in results]
        weighted_errors = [r['weighted_error'] for r in results]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax3.bar(x - width/2, unweighted_errors, width, label='Unweighted Error', alpha=0.8)
        ax3.bar(x + width/2, weighted_errors, width, label='Weighted Error', alpha=0.8)
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Error Rate')
        ax3.set_title('Error Rates Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(features, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Decision boundaries visualization
        ax4 = axes[1, 0]
        # Create mesh grid
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        # Plot decision boundary for best feature (Study Hours)
        best_feature_idx = 0  # Study Hours
        best_threshold, reverse = optimal_thresholds['Study_Hours']
        
        if reverse:
            Z = (xx <= best_threshold).astype(int)
        else:
            Z = (xx > best_threshold).astype(int)
        
        ax4.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax4.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=100, cmap='RdYlBu', alpha=0.7)
        ax4.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold: {best_threshold:.1f}')
        ax4.set_xlabel('Study Hours')
        ax4.set_ylabel('Sleep Hours')
        ax4.set_title('Decision Boundary: Study Hours')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Weighted error breakdown
        ax5 = axes[1, 1]
        student_names = self.data['Student']
        weights = self.initial_weights
        
        # Show weights and errors for best weak learner
        best_result = min(results, key=lambda x: x['weighted_error'])
        errors = best_result['errors']
        
        colors = ['red' if e == 1 else 'green' for e in errors]
        ax5.bar(student_names, weights, color=colors, alpha=0.7)
        ax5.set_xlabel('Students')
        ax5.set_ylabel('Initial Weights')
        ax5.set_title('Initial Weights and Errors\n(Red: Misclassified, Green: Correct)')
        ax5.grid(True, alpha=0.3)
        
        # Add error labels
        for i, (student, weight, error) in enumerate(zip(student_names, weights, errors)):
            ax5.annotate(f'E={error}', (i, weight), xytext=(0, 5), textcoords='offset points', 
                        ha='center', fontsize=8)
        
        # 6. Feature importance ranking
        ax6 = axes[1, 2]
        feature_names = [r['feature'] for r in results]
        error_rates = [r['unweighted_error'] for r in results]
        
        # Sort by error rate (ascending)
        sorted_indices = np.argsort(error_rates)
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_errors = [error_rates[i] for i in sorted_indices]
        
        bars = ax6.barh(sorted_features, sorted_errors, color='skyblue', alpha=0.7)
        ax6.set_xlabel('Error Rate (Lower is Better)')
        ax6.set_title('Feature Importance Ranking')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, sorted_errors)):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{error:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'adaboost_feature_engineering_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        
        # Create detailed decision stump visualization
        self.visualize_decision_stumps(results, optimal_thresholds)
        
        print(f"Visualizations saved to: {save_dir}")
    
    def visualize_decision_stumps(self, results, optimal_thresholds):
        """Create detailed visualization of decision stumps"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Decision Stump Weak Learners Analysis', fontsize=16, fontweight='bold')
        
        for i, (result, (feature_name, (threshold, reverse))) in enumerate(zip(results, optimal_thresholds.items())):
            ax = axes[i]
            feature_idx = self.feature_names.index(feature_name)
            feature_values = self.X[:, feature_idx]
            
            # Plot feature values
            colors = ['red' if y == 0 else 'blue' for y in self.y]
            ax.scatter(feature_values, [0]*len(feature_values), c=colors, s=100, alpha=0.7)
            
            # Add threshold line
            ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                      label=f'Threshold: {threshold:.1f}')
            
            # Add decision regions
            if reverse:
                ax.axvspan(ax.get_xlim()[0], threshold, alpha=0.2, color='blue', label='Predict 1')
                ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.2, color='red', label='Predict 0')
            else:
                ax.axvspan(ax.get_xlim()[0], threshold, alpha=0.2, color='red', label='Predict 0')
                ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.2, color='blue', label='Predict 1')
            
            # Add student labels
            for j, student in enumerate(self.data['Student']):
                ax.annotate(student, (feature_values[j], 0), xytext=(0, 10), textcoords='offset points', 
                           ha='center', fontsize=10)
            
            ax.set_xlabel(feature_name)
            ax.set_title(f'Decision Stump {i+1}: {result["description"]}\nError Rate: {result["unweighted_error"]:.3f}')
            ax.set_yticks([])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'decision_stumps_analysis.png'), 
                   dpi=300, bbox_inches='tight')
    
    def run_complete_analysis(self):
        """Run the complete AdaBoost feature engineering analysis"""
        print("=== COMPLETE ANALYSIS ===")
        
        # Step 1: Find optimal thresholds
        optimal_thresholds, optimal_errors = self.find_optimal_thresholds()
        
        # Step 2: Create decision stumps
        weak_learners = self.create_decision_stumps(optimal_thresholds)
        
        # Step 3: Evaluate weak learners
        results = self.evaluate_weak_learners(weak_learners, self.initial_weights)
        
        # Step 4: Find best weak learner
        best_learner = self.find_best_weak_learner(results)
        
        # Step 5: Analyze feature importance
        feature_ranking = self.analyze_feature_importance(results)
        
        # Step 6: Create visualizations
        self.visualize_results(results, optimal_thresholds)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY OF FINDINGS")
        print("=" * 60)
        print(f"1. Best weak learner: {best_learner['description']}")
        print(f"2. Most important feature: {feature_ranking[0][0]}")
        print(f"3. Feature ranking: {[f[0] for f in feature_ranking]}")
        print(f"4. All visualizations saved to: {save_dir}")
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'weak_learners': weak_learners,
            'results': results,
            'best_learner': best_learner,
            'feature_ranking': feature_ranking
        }

# Run the analysis
if __name__ == "__main__":
    adaboost_analysis = AdaBoostFeatureEngineering()
    results = adaboost_analysis.run_complete_analysis()
    
    print("\nAnalysis complete! Check the generated visualizations for detailed insights.")
