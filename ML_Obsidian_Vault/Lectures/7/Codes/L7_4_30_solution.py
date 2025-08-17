import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
import math

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_30")
os.makedirs(save_dir, exist_ok=True)

# Disable LaTeX to avoid Unicode issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdaBoostVsRandomClassifier:
    def __init__(self):
        # Problem parameters
        self.n_samples = 1000
        self.n_classes = 2
        self.samples_per_class = 500
        self.random_accuracy = 0.5
        self.random_error = 0.5
        self.weak_learner_error = 0.45
        self.max_iterations = 100
        self.training_time_per_learner = 1  # seconds
        
        print("=== AdaBoost vs Random Classifier Battle ===")
        print(f"Dataset: {self.n_samples} samples ({self.samples_per_class} per class)")
        print(f"Random classifier accuracy: {self.random_accuracy:.1%}")
        print(f"Random classifier error: ε = {self.random_error:.2f}")
        print(f"Weak learner error: ε = {self.weak_learner_error:.2f}")
        print(f"Maximum weak learners: {self.max_iterations}")
        print(f"Training time per learner: {self.training_time_per_learner} second")
        print("=" * 60)
        
    def task_1_random_classifier_accuracy(self):
        """Task 1: Expected accuracy of random classifier after 1000 predictions"""
        print("\n1. RANDOM CLASSIFIER ACCURACY AFTER 1000 PREDICTIONS")
        print("-" * 60)
        
        n_predictions = 1000
        p_success = 0.5
        
        expected_correct = n_predictions * p_success
        expected_accuracy = p_success
        std_correct = np.sqrt(n_predictions * p_success * (1 - p_success))
        confidence_interval = 1.96 * std_correct / n_predictions
        
        print(f"Expected accuracy: {expected_accuracy:.1%}")
        print(f"95% confidence interval: {expected_accuracy:.1%} ± {confidence_interval:.1%}")
        
        # Simulate multiple runs
        n_simulations = 10000
        simulation_results = np.random.binomial(n_predictions, p_success, n_simulations) / n_predictions
        
        # Plot distribution
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(simulation_results, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(expected_accuracy, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected_accuracy:.1%}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Random Classifier Accuracies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, 101), simulation_results[:100], 'b-', alpha=0.7)
        plt.axhline(y=expected_accuracy, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Simulation Run')
        plt.ylabel('Accuracy')
        plt.title('First 100 Simulation Runs')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_classifier_analysis.png'), dpi=300, bbox_inches='tight')
        
        return expected_accuracy, confidence_interval
    
    def task_2_adaboost_training_error_bound(self):
        """Task 2: Calculate theoretical training error bound for AdaBoost after 100 iterations"""
        print("\n2. ADABOOST TRAINING ERROR BOUND AFTER 100 ITERATIONS")
        print("-" * 60)
        
        gamma = 0.5 - self.weak_learner_error
        print(f"Margin: γ = {gamma:.3f}")
        
        iterations = np.arange(1, self.max_iterations + 1)
        error_bounds = np.exp(-2 * iterations * gamma**2)
        
        print(f"Error bound after 100 iterations: {error_bounds[99]:.6f}")
        
        # Plot error bounds
        plt.figure(figsize=(10, 6))
        plt.semilogy(iterations, error_bounds, 'b-', linewidth=2)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Error = 0.1')
        plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Error = 0.05')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Error Bound')
        plt.title('AdaBoost Training Error Bound vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'adaboost_error_bounds.png'), dpi=300, bbox_inches='tight')
        
        return error_bounds[99]
    
    def task_3_iterations_to_beat_random(self):
        """Task 3: How many iterations does AdaBoost need to achieve better performance than random guessing"""
        print("\n3. ITERATIONS NEEDED TO BEAT RANDOM GUESSING")
        print("-" * 60)
        
        target_error = 0.5
        gamma = 0.5 - self.weak_learner_error
        
        if gamma > 0:
            iterations_needed = int(np.ceil(-np.log(target_error) / (2 * gamma**2)))
            print(f"Iterations needed: {iterations_needed}")
        else:
            iterations_needed = float('inf')
            print("Cannot beat random guessing with γ ≤ 0")
        
        # Plot iterations vs target error
        plt.figure(figsize=(10, 6))
        target_errors = np.logspace(-3, 0, 100)
        iterations_needed_array = []
        
        for target_err in target_errors:
            if gamma > 0:
                iters = int(np.ceil(-np.log(target_err) / (2 * gamma**2)))
                iterations_needed_array.append(iters)
            else:
                iterations_needed_array.append(float('inf'))
        
        valid_indices = [i for i, iters in enumerate(iterations_needed_array) if iters != float('inf')]
        valid_targets = [target_errors[i] for i in valid_indices]
        valid_iterations = [iterations_needed_array[i] for i in valid_indices]
        
        plt.loglog(valid_targets, valid_iterations, 'b-', linewidth=2)
        plt.axhline(y=iterations_needed, color='red', linestyle='--', alpha=0.7, 
                   label=f'Target: {target_error:.3f} → {iterations_needed} iterations')
        plt.xlabel('Target Error')
        plt.ylabel('Iterations Needed')
        plt.title('Iterations vs Target Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'iterations_to_beat_random.png'), dpi=300, bbox_inches='tight')
        
        return iterations_needed
    
    def task_4_maximum_training_time(self):
        """Task 4: Maximum training time if each weak learner takes 1 second"""
        print("\n4. MAXIMUM TRAINING TIME")
        print("-" * 60)
        
        max_time = self.max_iterations * self.training_time_per_learner
        print(f"Maximum training time: {max_time} seconds")
        print(f"Maximum training time: {max_time/60:.1f} minutes")
        
        # Plot training time vs iterations
        plt.figure(figsize=(10, 6))
        iterations = np.arange(1, self.max_iterations + 1)
        training_times = iterations * self.training_time_per_learner
        
        plt.plot(iterations, training_times, 'b-', linewidth=2)
        plt.axhline(y=max_time, color='red', linestyle='--', alpha=0.7, 
                   label=f'Maximum: {max_time} seconds')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time vs Iterations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'training_time_analysis.png'), dpi=300, bbox_inches='tight')
        
        return max_time
    
    def task_5_weak_learner_comparison(self):
        """Task 5: Compare 50 weak learners with ε=0.4 vs 100 weak learners with ε=0.45"""
        print("\n5. WEAK LEARNER COMPARISON")
        print("-" * 60)
        
        # Scenario A: 50 weak learners with ε = 0.4
        scenario_a_iters = 50
        scenario_a_error = 0.4
        scenario_a_gamma = 0.5 - scenario_a_error
        scenario_a_error_bound = np.exp(-2 * scenario_a_iters * scenario_a_gamma**2)
        scenario_a_accuracy = 1 - scenario_a_error_bound
        scenario_a_time = scenario_a_iters * self.training_time_per_learner
        
        # Scenario B: 100 weak learners with ε = 0.45
        scenario_b_iters = 100
        scenario_b_error = 0.45
        scenario_b_gamma = 0.5 - scenario_b_error
        scenario_b_error_bound = np.exp(-2 * scenario_b_iters * scenario_b_gamma**2)
        scenario_b_accuracy = 1 - scenario_b_error_bound
        scenario_b_time = scenario_b_iters * self.training_time_per_learner
        
        print(f"Scenario A: Accuracy = {scenario_a_accuracy:.4f}, Time = {scenario_a_time}s")
        print(f"Scenario B: Accuracy = {scenario_b_accuracy:.4f}, Time = {scenario_b_time}s")
        
        # Efficiency comparison
        efficiency_a = scenario_a_accuracy / scenario_a_time
        efficiency_b = scenario_b_accuracy / scenario_b_time
        
        print(f"Efficiency A: {efficiency_a:.6f}, Efficiency B: {efficiency_b:.6f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        scenarios = ['Scenario A\n(50 learners, ε=0.4)', 'Scenario B\n(100 learners, ε=0.45)']
        accuracies = [scenario_a_accuracy, scenario_b_accuracy]
        times = [scenario_a_time, scenario_b_time]
        efficiencies = [efficiency_a, efficiency_b]
        
        plt.subplot(1, 3, 1)
        plt.bar(scenarios, accuracies, color=['lightblue', 'lightcoral'], alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.bar(scenarios, times, color=['lightblue', 'lightcoral'], alpha=0.7)
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.bar(scenarios, efficiencies, color=['lightblue', 'lightcoral'], alpha=0.7)
        plt.ylabel('Efficiency (Accuracy/Second)')
        plt.title('Efficiency Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'weak_learner_comparison.png'), dpi=300, bbox_inches='tight')
        
        return scenario_a_accuracy, scenario_b_accuracy, efficiency_a, efficiency_b
    
    def run_complete_analysis(self):
        """Run all tasks and generate comprehensive analysis"""
        print("=" * 80)
        print("COMPLETE ANALYSIS: ADABOOST VS RANDOM CLASSIFIER BATTLE")
        print("=" * 80)
        
        # Task 1: Random classifier accuracy
        random_acc, random_ci = self.task_1_random_classifier_accuracy()
        
        # Task 2: AdaBoost training error bound
        adaboost_error_bound = self.task_2_adaboost_training_error_bound()
        
        # Task 3: Iterations to beat random
        iterations_to_beat_random = self.task_3_iterations_to_beat_random()
        
        # Task 4: Maximum training time
        max_training_time = self.task_4_maximum_training_time()
        
        # Task 5: Weak learner comparison
        acc_a, acc_b, eff_a, eff_b = self.task_5_weak_learner_comparison()
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        print(f"1. Random Classifier Accuracy: {random_acc:.1%} ± {random_ci:.1%}")
        print(f"2. AdaBoost Error Bound (100 iterations): {adaboost_error_bound:.6f}")
        print(f"3. Iterations to Beat Random: {iterations_to_beat_random}")
        print(f"4. Maximum Training Time: {max_training_time} seconds")
        print(f"5. Best Weak Learner Strategy: {'Scenario A' if eff_a > eff_b else 'Scenario B'}")
        
        print(f"\nPlots saved to: {save_dir}")
        print("=" * 80)

if __name__ == "__main__":
    # Create and run the analysis
    analyzer = AdaBoostVsRandomClassifier()
    analyzer.run_complete_analysis()
