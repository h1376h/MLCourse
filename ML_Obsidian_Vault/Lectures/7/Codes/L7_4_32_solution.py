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
save_dir = os.path.join(images_dir, "L7_4_Quiz_32")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostErrorAnalysis:
    def __init__(self):
        # Given data from the problem
        self.sample_weights = np.array([0.15, 0.12, 0.08, 0.20, 0.10, 0.05, 0.18, 0.06, 0.14, 0.12])
        self.weak_learner_1 = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1, -1])
        self.weak_learner_2 = np.array([1, -1, -1, -1, 1, -1, 1, -1, 1, -1])
        self.weak_learner_3 = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1, -1])
        self.true_labels = np.array([1, 1, -1, -1, 1, -1, 1, -1, 1, -1])
        
        # Sample indices for labeling
        self.sample_indices = np.arange(1, 11)
        
        print("=== AdaBoost Error Analysis Game ===")
        print(f"Dataset: {len(self.sample_weights)} samples with binary labels")
        print(f"Sample weights: {self.sample_weights}")
        print(f"True labels: {self.true_labels}")
        print("=" * 60)
        
    def task_1_misclassified_samples(self):
        """Task 1: Identify misclassified samples for each weak learner"""
        print("\n1. MISCLASSIFIED SAMPLES ANALYSIS")
        print("-" * 60)
        
        # Calculate misclassifications for each weak learner
        misclassified_1 = self.weak_learner_1 != self.true_labels
        misclassified_2 = self.weak_learner_2 != self.true_labels
        misclassified_3 = self.weak_learner_3 != self.true_labels
        
        print("Weak Learner 1 misclassifications:")
        for i, (pred, true, is_misclassified) in enumerate(zip(self.weak_learner_1, self.true_labels, misclassified_1)):
            status = "MISCLASSIFIED" if is_misclassified else "Correct"
            print(f"  Sample {i+1}: Predicted {pred}, True {true} -> {status}")
        
        print("\nWeak Learner 2 misclassifications:")
        for i, (pred, true, is_misclassified) in enumerate(zip(self.weak_learner_2, self.true_labels, misclassified_2)):
            status = "MISCLASSIFIED" if is_misclassified else "Correct"
            print(f"  Sample {i+1}: Predicted {pred}, True {true} -> {status}")
        
        print("\nWeak Learner 3 misclassifications:")
        for i, (pred, true, is_misclassified) in enumerate(zip(self.weak_learner_3, self.true_labels, misclassified_3)):
            status = "MISCLASSIFIED" if is_misclassified else "Correct"
            print(f"  Sample {i+1}: Predicted {pred}, True {true} -> {status}")
        
        # Create visualization
        self.plot_misclassifications(misclassified_1, misclassified_2, misclassified_3)
        
        return misclassified_1, misclassified_2, misclassified_3
    
    def task_2_weighted_errors(self):
        """Task 2: Calculate weighted error for each weak learner"""
        print("\n2. WEIGHTED ERROR CALCULATIONS")
        print("-" * 60)
        
        # Calculate misclassifications
        misclassified_1 = self.weak_learner_1 != self.true_labels
        misclassified_2 = self.weak_learner_2 != self.true_labels
        misclassified_3 = self.weak_learner_3 != self.true_labels
        
        # Calculate weighted errors
        weighted_error_1 = np.sum(self.sample_weights * misclassified_1)
        weighted_error_2 = np.sum(self.sample_weights * misclassified_2)
        weighted_error_3 = np.sum(self.sample_weights * misclassified_3)
        
        print("Weak Learner 1:")
        print(f"  Misclassified samples: {np.where(misclassified_1)[0] + 1}")
        print(f"  Weights of misclassified: {self.sample_weights[misclassified_1]}")
        print(f"  Weighted error: ε₁ = {weighted_error_1:.4f}")
        
        print("\nWeak Learner 2:")
        print(f"  Misclassified samples: {np.where(misclassified_2)[0] + 1}")
        print(f"  Weights of misclassified: {self.sample_weights[misclassified_2]}")
        print(f"  Weighted error: ε₂ = {weighted_error_2:.4f}")
        
        print("\nWeak Learner 3:")
        print(f"  Misclassified samples: {np.where(misclassified_3)[0] + 1}")
        print(f"  Weights of misclassified: {self.sample_weights[misclassified_3]}")
        print(f"  Weighted error: ε₃ = {weighted_error_3:.4f}")
        
        # Create visualization
        self.plot_weighted_errors([weighted_error_1, weighted_error_2, weighted_error_3])
        
        return weighted_error_1, weighted_error_2, weighted_error_3
    
    def task_3_performance_ranking(self):
        """Task 3: Rank weak learners by performance"""
        print("\n3. WEAK LEARNER PERFORMANCE RANKING")
        print("-" * 60)
        
        # Calculate weighted errors
        weighted_errors = self.task_2_weighted_errors()
        
        # Create ranking
        learners = ['Weak Learner 1', 'Weak Learner 2', 'Weak Learner 3']
        ranking = sorted(zip(learners, weighted_errors), key=lambda x: x[1])
        
        print("Performance Ranking (Best to Worst):")
        for i, (learner, error) in enumerate(ranking):
            rank = i + 1
            print(f"  {rank}. {learner}: ε = {error:.4f}")
        
        best_learner = ranking[0][0]
        worst_learner = ranking[-1][0]
        
        print(f"\nBest performer: {best_learner}")
        print(f"Worst performer: {worst_learner}")
        
        # Create visualization
        self.plot_performance_ranking(learners, weighted_errors)
        
        return ranking, best_learner, worst_learner
    
    def task_4_remove_weak_learner(self):
        """Task 4: Analyze removing one weak learner"""
        print("\n4. REMOVING WEAK LEARNER ANALYSIS")
        print("-" * 60)
        
        # Calculate weighted errors
        weighted_errors = self.task_2_weighted_errors()
        learners = ['Weak Learner 1', 'Weak Learner 2', 'Weak Learner 3']
        
        print("Analysis of removing each weak learner:")
        for i, (learner, error) in enumerate(zip(learners, weighted_errors)):
            remaining_learners = [j for j in range(3) if j != i]
            remaining_errors = [weighted_errors[j] for j in remaining_learners]
            avg_remaining_error = np.mean(remaining_errors)
            
            print(f"\n{learner} (ε = {error:.4f}):")
            print(f"  Remaining learners: {[learners[j] for j in remaining_learners]}")
            print(f"  Average error of remaining: {avg_remaining_error:.4f}")
            print(f"  Impact: {'Positive' if avg_remaining_error < error else 'Negative'}")
        
        # Create visualization
        self.plot_removal_impact(learners, weighted_errors)
        
        return learners, weighted_errors
    
    def task_5_ensemble_performance_change(self):
        """Task 5: Analyze ensemble performance change after removing worst learner"""
        print("\n5. ENSEMBLE PERFORMANCE CHANGE ANALYSIS")
        print("-" * 60)
        
        # Get ranking
        ranking, best_learner, worst_learner = self.task_3_performance_ranking()
        
        # Calculate ensemble performance with all learners
        all_predictions = np.array([self.weak_learner_1, self.weak_learner_2, self.weak_learner_3])
        ensemble_predictions_all = np.sign(np.sum(all_predictions, axis=0))
        ensemble_accuracy_all = np.mean(ensemble_predictions_all == self.true_labels)
        
        # Calculate ensemble performance without worst learner
        learners = ['Weak Learner 1', 'Weak Learner 2', 'Weak Learner 3']
        worst_idx = learners.index(worst_learner)
        remaining_predictions = np.array([all_predictions[i] for i in range(3) if i != worst_idx])
        ensemble_predictions_reduced = np.sign(np.sum(remaining_predictions, axis=0))
        ensemble_accuracy_reduced = np.mean(ensemble_predictions_reduced == self.true_labels)
        
        print(f"Original ensemble (all 3 learners):")
        print(f"  Accuracy: {ensemble_accuracy_all:.1%}")
        print(f"  Error rate: {1 - ensemble_accuracy_all:.1%}")
        
        print(f"\nReduced ensemble (without {worst_learner}):")
        print(f"  Accuracy: {ensemble_accuracy_reduced:.1%}")
        print(f"  Error rate: {1 - ensemble_accuracy_reduced:.1%}")
        
        print(f"\nPerformance change:")
        accuracy_change = ensemble_accuracy_reduced - ensemble_accuracy_all
        print(f"  Accuracy change: {accuracy_change:+.1%}")
        
        if accuracy_change > 0:
            print("  Removing the worst learner IMPROVES ensemble performance!")
        elif accuracy_change < 0:
            print("  Removing the worst learner REDUCES ensemble performance!")
        else:
            print("  Removing the worst learner has NO EFFECT on ensemble performance!")
        
        # Create visualization
        self.plot_ensemble_comparison(ensemble_predictions_all, ensemble_predictions_reduced, 
                                    ensemble_accuracy_all, ensemble_accuracy_reduced)
        
        return ensemble_accuracy_all, ensemble_accuracy_reduced, accuracy_change
    
    def plot_misclassifications(self, misclassified_1, misclassified_2, misclassified_3):
        """Plot misclassification patterns for all weak learners"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample weights visualization
        ax1 = axes[0, 0]
        bars1 = ax1.bar(self.sample_indices, self.sample_weights, 
                        color=['red' if m else 'green' for m in misclassified_1],
                        alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Sample Weight')
        ax1.set_title('Weak Learner 1: Sample Weights\n(Red=Misclassified, Green=Correct)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars1, self.sample_weights):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Weak Learner 2
        ax2 = axes[0, 1]
        bars2 = ax2.bar(self.sample_indices, self.sample_weights,
                        color=['red' if m else 'green' for m in misclassified_2],
                        alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Sample Weight')
        ax2.set_title('Weak Learner 2: Sample Weights\n(Red=Misclassified, Green=Correct)')
        ax2.grid(True, alpha=0.3)
        
        for bar, weight in zip(bars2, self.sample_weights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Weak Learner 3
        ax3 = axes[1, 0]
        bars3 = ax3.bar(self.sample_indices, self.sample_weights,
                        color=['red' if m else 'green' for m in misclassified_3],
                        alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Sample Weight')
        ax3.set_title('Weak Learner 3: Sample Weights\n(Red=Misclassified, Green=Correct)')
        ax3.grid(True, alpha=0.3)
        
        for bar, weight in zip(bars3, self.sample_weights):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary data
        summary_data = [
            ['Weak Learner 1', f'{np.sum(misclassified_1)}', f'{np.sum(self.sample_weights * misclassified_1):.4f}'],
            ['Weak Learner 2', f'{np.sum(misclassified_2)}', f'{np.sum(self.sample_weights * misclassified_2):.4f}'],
            ['Weak Learner 3', f'{np.sum(misclassified_3)}', f'{np.sum(self.sample_weights * misclassified_3):.4f}']
        ]
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Learner', 'Misclassified\nCount', 'Weighted\nError'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color the header
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on performance
        errors = [np.sum(self.sample_weights * m) for m in [misclassified_1, misclassified_2, misclassified_3]]
        for i, error in enumerate(errors):
            if error == min(errors):
                color = '#4CAF50'  # Green for best
            elif error == max(errors):
                color = '#F44336'   # Red for worst
            else:
                color = '#FF9800'   # Orange for middle
            for j in range(3):
                table[(i+1, j)].set_facecolor(color)
        
        ax4.set_title('Summary of Misclassifications', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'misclassifications_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weighted_errors(self, weighted_errors):
        """Plot weighted errors for all weak learners"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of weighted errors
        learners = ['Weak Learner 1', 'Weak Learner 2', 'Weak Learner 3']
        colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
        
        bars = ax1.bar(learners, weighted_errors, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weak Learners')
        ax1.set_ylabel('Weighted Error (ε)')
        ax1.set_title('Weighted Error Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, error in zip(bars, weighted_errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{error:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Horizontal line at 0.5 (random classifier threshold)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Classifier (ε = 0.5)')
        ax1.legend()
        
        # Pie chart showing error distribution
        ax2.pie(weighted_errors, labels=learners, autopct='%1.1f%%', startangle=90,
                colors=colors, explode=(0.05, 0.05, 0.05))
        ax2.set_title('Proportion of Total Weighted Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weighted_errors_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_ranking(self, learners, weighted_errors):
        """Plot performance ranking of weak learners"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create ranking
        ranking = sorted(zip(learners, weighted_errors), key=lambda x: x[1])
        ranked_learners = [learner for learner, _ in ranking]
        ranked_errors = [error for _, error in ranking]
        
        # Horizontal bar chart (best to worst)
        colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
        bars = ax1.barh(ranked_learners, ranked_errors, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weighted Error (ε)')
        ax1.set_title('Performance Ranking (Best to Worst)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, error in zip(bars, ranked_errors):
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
                    f'{error:.4f}', ha='left', va='center', fontweight='bold')
        
        # Performance improvement visualization
        best_error = ranked_errors[0]
        improvements = [(error - best_error) / best_error * 100 for error in ranked_errors]
        
        ax2.bar(ranked_learners, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Weak Learners')
        ax2.set_ylabel('Error Increase (%)')
        ax2.set_title('Performance Degradation Relative to Best Learner')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, improvement in enumerate(improvements):
            ax2.text(i, improvement + 1, f'{improvement:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_ranking.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_removal_impact(self, learners, weighted_errors):
        """Plot impact of removing each weak learner"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate impact of removing each learner
        impacts = []
        remaining_errors = []
        
        for i in range(3):
            remaining = [weighted_errors[j] for j in range(3) if j != i]
            avg_remaining = np.mean(remaining)
            impact = avg_remaining - weighted_errors[i]
            impacts.append(impact)
            remaining_errors.append(avg_remaining)
        
        # Impact visualization
        colors = ['green' if impact < 0 else 'red' for impact in impacts]
        bars = ax1.bar(learners, impacts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weak Learners')
        ax1.set_ylabel('Impact on Average Error')
        ax1.set_title('Impact of Removing Each Weak Learner')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if impact > 0 else -0.01),
                    f'{impact:+.4f}', ha='center', 
                    va='bottom' if impact > 0 else 'top', fontweight='bold')
        
        # Before vs After comparison
        x = np.arange(len(learners))
        width = 0.35
        
        ax2.bar(x - width/2, weighted_errors, width, label='Original Error', alpha=0.7, edgecolor='black')
        ax2.bar(x + width/2, remaining_errors, width, label='Average Error After Removal', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Weak Learners')
        ax2.set_ylabel('Weighted Error (ε)')
        ax2.set_title('Error Comparison: Before vs After Removal')
        ax2.set_xticks(x)
        ax2.set_xticklabels(learners)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'removal_impact_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ensemble_comparison(self, ensemble_predictions_all, ensemble_predictions_reduced, 
                                ensemble_accuracy_all, ensemble_accuracy_reduced):
        """Plot ensemble performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original ensemble predictions
        correct_all = ensemble_predictions_all == self.true_labels
        colors_all = ['green' if c else 'red' for c in correct_all]
        
        bars1 = ax1.bar(self.sample_indices, self.sample_weights, 
                        color=colors_all, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Sample Weight')
        ax1.set_title(f'Original Ensemble (All 3 Learners)\nAccuracy: {ensemble_accuracy_all:.1%}')
        ax1.grid(True, alpha=0.3)
        
        # Reduced ensemble predictions
        correct_reduced = ensemble_predictions_reduced == self.true_labels
        colors_reduced = ['green' if c else 'red' for c in correct_reduced]
        
        bars2 = ax2.bar(self.sample_indices, self.sample_weights,
                        color=colors_reduced, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Sample Weight')
        ax2.set_title(f'Reduced Ensemble (Without Worst Learner)\nAccuracy: {ensemble_accuracy_reduced:.1%}')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar, weight in zip(bars, self.sample_weights):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{weight:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Accuracy comparison
        accuracies = [ensemble_accuracy_all, ensemble_accuracy_reduced]
        labels = ['All 3 Learners', 'Without Worst Learner']
        colors = ['#2196F3', '#FF9800']
        
        bars3 = ax3.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Ensemble Accuracy Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars3, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Error rate comparison
        error_rates = [1 - ensemble_accuracy_all, 1 - ensemble_accuracy_reduced]
        bars4 = ax4.bar(labels, error_rates, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Ensemble Error Rate Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, err in zip(bars4, error_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{err:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_analysis(self):
        """Run all tasks and generate comprehensive analysis"""
        print("Starting comprehensive AdaBoost Error Analysis...")
        print("=" * 80)
        
        # Run all tasks
        task1_results = self.task_1_misclassified_samples()
        task2_results = self.task_2_weighted_errors()
        task3_results = self.task_3_performance_ranking()
        task4_results = self.task_4_remove_weak_learner()
        task5_results = self.task_5_ensemble_performance_change()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        
        # Summary
        print("\nSUMMARY OF FINDINGS:")
        print("-" * 40)
        print(f"• Best weak learner: {task3_results[1]}")
        print(f"• Worst weak learner: {task3_results[2]}")
        print(f"• Original ensemble accuracy: {task5_results[0]:.1%}")
        print(f"• Reduced ensemble accuracy: {task5_results[1]:.1%}")
        print(f"• Performance change: {task5_results[2]:+.1%}")
        
        print(f"\nAll visualizations saved to: {save_dir}")
        
        return {
            'task1': task1_results,
            'task2': task2_results,
            'task3': task3_results,
            'task4': task4_results,
            'task5': task5_results
        }

if __name__ == "__main__":
    # Create and run the analysis
    analyzer = AdaBoostErrorAnalysis()
    results = analyzer.run_complete_analysis()
