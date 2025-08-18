import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import minimize_scalar
import seaborn as sns

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_4_Quiz_21")
os.makedirs(save_dir, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

def analyze_convergence_rate_theory():
    """Analyze the theoretical relationship between weak learner error and convergence speed."""
    print_step_header(1, "Theoretical Convergence Rate Analysis")
    
    print("AdaBoost Convergence Theory:")
    print("-" * 40)
    print("Training error bound: E ≤ ∏(t=1 to T) 2√(εt(1-εt))")
    print("Where εt is the error rate of weak learner t")
    print()
    
    # Define the convergence function
    def convergence_factor(epsilon):
        """Calculate the convergence factor 2√(ε(1-ε)) for a given error rate."""
        return 2 * np.sqrt(epsilon * (1 - epsilon))
    
    # Analyze different error rates
    epsilon_values = np.linspace(0.01, 0.49, 100)
    convergence_factors = [convergence_factor(eps) for eps in epsilon_values]
    
    # Find optimal error rate (minimum convergence factor)
    optimal_idx = np.argmin(convergence_factors)
    optimal_epsilon = epsilon_values[optimal_idx]
    optimal_factor = convergence_factors[optimal_idx]
    
    print(f"Optimal weak learner error rate: {optimal_epsilon:.3f}")
    print(f"Minimum convergence factor: {optimal_factor:.3f}")
    print(f"This occurs at ε = 0 (perfect weak learner)")
    print()
    
    # Practical analysis for realistic error rates
    practical_epsilons = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
    practical_factors = [convergence_factor(eps) for eps in practical_epsilons]
    
    print("Practical Error Rates and Convergence Factors:")
    for eps, factor in zip(practical_epsilons, practical_factors):
        print(f"ε = {eps:.2f} → factor = {factor:.3f}")
    
    # Visualize convergence rate analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Convergence factor vs error rate
    axes[0, 0].plot(epsilon_values, convergence_factors, 'b-', linewidth=2)
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random classifier (ε=0.5)')
    axes[0, 0].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='No improvement (factor=1)')
    axes[0, 0].set_xlabel('Weak Learner Error Rate (ε)')
    axes[0, 0].set_ylabel('Convergence Factor 2√(ε(1-ε))')
    axes[0, 0].set_title('Convergence Factor vs Error Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_xlim(0, 0.5)
    
    # Plot 2: Training error bound after T iterations
    T_values = range(1, 51)
    error_rates = [0.1, 0.2, 0.3, 0.4, 0.45]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for eps, color in zip(error_rates, colors):
        factor = convergence_factor(eps)
        bounds = [factor**t for t in T_values]
        axes[0, 1].plot(T_values, bounds, color=color, linewidth=2, label=f'ε = {eps}')
    
    axes[0, 1].set_xlabel('Number of Iterations (T)')
    axes[0, 1].set_ylabel('Training Error Bound')
    axes[0, 1].set_title('Training Error Bound vs Iterations')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Iterations needed for different target errors
    target_errors = [0.1, 0.01, 0.001]
    iterations_needed = []
    
    for target in target_errors:
        iterations_for_target = []
        for eps in error_rates:
            factor = convergence_factor(eps)
            if factor < 1:
                # Solve factor^T = target for T
                T_needed = np.log(target) / np.log(factor)
                iterations_for_target.append(T_needed)
            else:
                iterations_for_target.append(np.inf)
        iterations_needed.append(iterations_for_target)
    
    x = np.arange(len(error_rates))
    width = 0.25
    
    for i, (target, iterations) in enumerate(zip(target_errors, iterations_needed)):
        finite_iterations = [it if it != np.inf else 1000 for it in iterations]
        axes[1, 0].bar(x + i*width, finite_iterations, width, 
                      label=f'Target: {target}', alpha=0.7)
    
    axes[1, 0].set_xlabel('Weak Learner Error Rate')
    axes[1, 0].set_ylabel('Iterations Needed')
    axes[1, 0].set_title('Iterations Needed for Different Target Errors')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels([f'{eps}' for eps in error_rates])
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Convergence speed comparison
    convergence_speeds = [1/factor if factor < 1 else 0 for factor in practical_factors]
    
    axes[1, 1].bar(range(len(practical_epsilons)), convergence_speeds, 
                   color='lightblue', alpha=0.7)
    axes[1, 1].set_xlabel('Weak Learner Error Rate')
    axes[1, 1].set_ylabel('Convergence Speed (1/factor)')
    axes[1, 1].set_title('Convergence Speed Comparison')
    axes[1, 1].set_xticks(range(len(practical_epsilons)))
    axes[1, 1].set_xticklabels([f'{eps}' for eps in practical_epsilons])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_rate_theory.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_epsilon, practical_epsilons, practical_factors

def analyze_iterations_vs_error_bound():
    """Analyze how the number of iterations affects the training error bound."""
    print_step_header(2, "Iterations vs Training Error Bound Analysis")
    
    def calculate_error_bound(epsilon, T):
        """Calculate training error bound for given error rate and iterations."""
        factor = 2 * np.sqrt(epsilon * (1 - epsilon))
        return factor ** T
    
    # Different scenarios
    scenarios = {
        'Excellent Weak Learners': {'epsilon': 0.1, 'color': 'blue'},
        'Good Weak Learners': {'epsilon': 0.2, 'color': 'green'},
        'Average Weak Learners': {'epsilon': 0.3, 'color': 'orange'},
        'Poor Weak Learners': {'epsilon': 0.4, 'color': 'red'},
        'Barely Better than Random': {'epsilon': 0.45, 'color': 'purple'}
    }
    
    T_range = range(1, 101)
    
    print("Error Bound Analysis:")
    print("-" * 30)
    
    # Calculate and display key results
    for name, params in scenarios.items():
        eps = params['epsilon']
        bound_10 = calculate_error_bound(eps, 10)
        bound_50 = calculate_error_bound(eps, 50)
        bound_100 = calculate_error_bound(eps, 100)
        
        print(f"{name} (ε={eps}):")
        print(f"  After 10 iterations: {bound_10:.6f}")
        print(f"  After 50 iterations: {bound_50:.6f}")
        print(f"  After 100 iterations: {bound_100:.6f}")
        print()
    
    # Visualize the analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Error bounds vs iterations (log scale)
    for name, params in scenarios.items():
        eps = params['epsilon']
        color = params['color']
        bounds = [calculate_error_bound(eps, t) for t in T_range]
        axes[0, 0].plot(T_range, bounds, color=color, linewidth=2, label=name)
    
    axes[0, 0].set_xlabel('Number of Iterations (T)')
    axes[0, 0].set_ylabel('Training Error Bound')
    axes[0, 0].set_title('Training Error Bound vs Iterations (Log Scale)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Linear scale for first 20 iterations
    T_short = range(1, 21)
    for name, params in scenarios.items():
        eps = params['epsilon']
        color = params['color']
        bounds = [calculate_error_bound(eps, t) for t in T_short]
        axes[0, 1].plot(T_short, bounds, color=color, linewidth=2, label=name, marker='o')
    
    axes[0, 1].set_xlabel('Number of Iterations (T)')
    axes[0, 1].set_ylabel('Training Error Bound')
    axes[0, 1].set_title('Training Error Bound vs Iterations (First 20)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Iterations needed to reach specific error targets
    error_targets = [0.1, 0.01, 0.001, 0.0001]
    iterations_matrix = []
    
    for target in error_targets:
        iterations_row = []
        for name, params in scenarios.items():
            eps = params['epsilon']
            factor = 2 * np.sqrt(eps * (1 - eps))
            if factor < 1:
                T_needed = np.log(target) / np.log(factor)
                iterations_row.append(min(T_needed, 1000))  # Cap at 1000
            else:
                iterations_row.append(1000)  # Won't converge
        iterations_matrix.append(iterations_row)
    
    # Create heatmap
    scenario_names = list(scenarios.keys())
    im = axes[1, 0].imshow(iterations_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_xticks(range(len(scenario_names)))
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in scenario_names], rotation=45)
    axes[1, 0].set_yticks(range(len(error_targets)))
    axes[1, 0].set_yticklabels([f'{target}' for target in error_targets])
    axes[1, 0].set_xlabel('Weak Learner Quality')
    axes[1, 0].set_ylabel('Target Error')
    axes[1, 0].set_title('Iterations Needed to Reach Target Error')
    
    # Add text annotations
    for i in range(len(error_targets)):
        for j in range(len(scenario_names)):
            text = f'{int(iterations_matrix[i][j])}'
            if iterations_matrix[i][j] >= 1000:
                text = '∞'
            axes[1, 0].text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Convergence rate comparison
    convergence_rates = []
    for name, params in scenarios.items():
        eps = params['epsilon']
        factor = 2 * np.sqrt(eps * (1 - eps))
        rate = -np.log(factor) if factor < 1 else 0
        convergence_rates.append(rate)
    
    bars = axes[1, 1].bar(range(len(scenario_names)), convergence_rates, 
                         color=[params['color'] for params in scenarios.values()], alpha=0.7)
    axes[1, 1].set_xlabel('Weak Learner Quality')
    axes[1, 1].set_ylabel('Convergence Rate (-log(factor))')
    axes[1, 1].set_title('Convergence Rate Comparison')
    axes[1, 1].set_xticks(range(len(scenario_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in scenario_names], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, convergence_rates):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iterations_vs_error_bound.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return scenarios, iterations_matrix

def find_optimal_error_rate():
    """Find the optimal weak learner error rate for fastest convergence."""
    print_step_header(3, "Finding Optimal Error Rate")
    
    print("Theoretical Analysis:")
    print("-" * 25)
    print("The convergence factor is: f(ε) = 2√(ε(1-ε))")
    print("To minimize this, we take the derivative and set it to zero:")
    print("f'(ε) = 2 * (1-2ε) / (2√(ε(1-ε))) = (1-2ε) / √(ε(1-ε))")
    print("Setting f'(ε) = 0: 1-2ε = 0 → ε = 0.5")
    print("But this is a maximum, not minimum!")
    print()
    print("Since f(ε) is minimized at the boundaries:")
    print("- As ε → 0: f(ε) → 0 (perfect weak learner)")
    print("- As ε → 0.5: f(ε) → 1 (random classifier)")
    print()
    print("Therefore, the optimal error rate is ε = 0 (impossible in practice)")
    print("In practice, we want the smallest possible ε > 0")
    
    # Numerical analysis
    epsilon_range = np.linspace(0.001, 0.499, 1000)
    convergence_factors = 2 * np.sqrt(epsilon_range * (1 - epsilon_range))
    
    # Find minimum (should be at ε ≈ 0)
    min_idx = np.argmin(convergence_factors)
    optimal_epsilon = epsilon_range[min_idx]
    min_factor = convergence_factors[min_idx]
    
    print(f"\nNumerical verification:")
    print(f"Minimum factor: {min_factor:.6f} at ε = {optimal_epsilon:.6f}")
    
    # Practical recommendations
    practical_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    practical_factors = [2 * np.sqrt(eps * (1 - eps)) for eps in practical_thresholds]
    
    print(f"\nPractical Error Rates and Convergence:")
    for eps, factor in zip(practical_thresholds, practical_factors):
        iterations_for_01 = np.log(0.01) / np.log(factor) if factor < 1 else np.inf
        print(f"ε = {eps:.2f} → factor = {factor:.3f} → {iterations_for_01:.0f} iterations for 1% error")
    
    return optimal_epsilon, practical_thresholds, practical_factors

def estimate_iterations_for_target_error():
    """Estimate iterations needed for given error targets with different weak learner qualities."""
    print_step_header(4, "Estimating Iterations for Target Errors")

    def iterations_needed(epsilon, target_error):
        """Calculate iterations needed to reach target error."""
        if epsilon >= 0.5:
            return np.inf  # Won't converge
        factor = 2 * np.sqrt(epsilon * (1 - epsilon))
        if factor >= 1:
            return np.inf
        return np.log(target_error) / np.log(factor)

    # Different target errors
    targets = [0.1, 0.05, 0.01, 0.005, 0.001]
    error_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    print("Iterations Needed for Different Target Errors:")
    print("-" * 50)
    print(f"{'Error Rate':<12} {'10%':<8} {'5%':<8} {'1%':<8} {'0.5%':<8} {'0.1%':<8}")
    print("-" * 50)

    results_matrix = []
    for eps in error_rates:
        row = []
        row_str = f"{eps:<12.2f}"
        for target in targets:
            iterations = iterations_needed(eps, target)
            if iterations == np.inf:
                row_str += f"{'∞':<8}"
                row.append(1000)  # For plotting
            else:
                row_str += f"{int(iterations):<8}"
                row.append(iterations)
        results_matrix.append(row)
        print(row_str)

    # Visualize the results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Heatmap of iterations needed
    im = axes[0, 0].imshow(results_matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_xticks(range(len(targets)))
    axes[0, 0].set_xticklabels([f'{t*100}%' for t in targets])
    axes[0, 0].set_yticks(range(len(error_rates)))
    axes[0, 0].set_yticklabels([f'{eps}' for eps in error_rates])
    axes[0, 0].set_xlabel('Target Error')
    axes[0, 0].set_ylabel('Weak Learner Error Rate')
    axes[0, 0].set_title('Iterations Needed (Heatmap)')
    plt.colorbar(im, ax=axes[0, 0])

    # Line plot for specific target errors
    target_indices = [0, 2, 4]  # 10%, 1%, 0.1%
    colors = ['blue', 'red', 'green']

    for i, color in zip(target_indices, colors):
        target = targets[i]
        iterations = [row[i] for row in results_matrix]
        # Cap at 500 for visualization
        iterations_capped = [min(it, 500) for it in iterations]
        axes[0, 1].plot(error_rates, iterations_capped, color=color,
                       linewidth=2, marker='o', label=f'Target: {target*100}%')

    axes[0, 1].set_xlabel('Weak Learner Error Rate')
    axes[0, 1].set_ylabel('Iterations Needed (capped at 500)')
    axes[0, 1].set_title('Iterations vs Error Rate for Different Targets')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Convergence efficiency (1/iterations)
    efficiency_matrix = []
    for row in results_matrix:
        efficiency_row = [1/it if it < 1000 else 0 for it in row]
        efficiency_matrix.append(efficiency_row)

    im2 = axes[1, 0].imshow(efficiency_matrix, cmap='plasma', aspect='auto')
    axes[1, 0].set_xticks(range(len(targets)))
    axes[1, 0].set_xticklabels([f'{t*100}%' for t in targets])
    axes[1, 0].set_yticks(range(len(error_rates)))
    axes[1, 0].set_yticklabels([f'{eps}' for eps in error_rates])
    axes[1, 0].set_xlabel('Target Error')
    axes[1, 0].set_ylabel('Weak Learner Error Rate')
    axes[1, 0].set_title('Convergence Efficiency (1/iterations)')
    plt.colorbar(im2, ax=axes[1, 0])

    # Practical recommendations
    practical_scenarios = {
        'Quick Prototyping': {'target': 0.1, 'max_iterations': 20},
        'Production System': {'target': 0.01, 'max_iterations': 100},
        'High Precision': {'target': 0.001, 'max_iterations': 500}
    }

    recommendations = []
    for scenario, params in practical_scenarios.items():
        target = params['target']
        max_iter = params['max_iterations']

        # Find best error rate within iteration budget
        best_eps = None
        for eps in error_rates:
            needed = iterations_needed(eps, target)
            if needed <= max_iter:
                best_eps = eps
                break

        recommendations.append({
            'scenario': scenario,
            'target': target,
            'max_iter': max_iter,
            'recommended_eps': best_eps,
            'actual_iterations': iterations_needed(best_eps, target) if best_eps else None
        })

    # Plot recommendations
    scenario_names = [r['scenario'] for r in recommendations]
    recommended_eps = [r['recommended_eps'] if r['recommended_eps'] else 0 for r in recommendations]

    bars = axes[1, 1].bar(range(len(scenario_names)), recommended_eps,
                         color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    axes[1, 1].set_xlabel('Use Case Scenario')
    axes[1, 1].set_ylabel('Recommended Max Error Rate')
    axes[1, 1].set_title('Recommended Error Rates for Different Scenarios')
    axes[1, 1].set_xticks(range(len(scenario_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in scenario_names])
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels
    for bar, eps in zip(bars, recommended_eps):
        if eps > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                           f'{eps:.2f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., 0.01,
                           'N/A', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'iterations_estimation.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPractical Recommendations:")
    print("-" * 30)
    for rec in recommendations:
        print(f"{rec['scenario']}:")
        print(f"  Target error: {rec['target']*100}%")
        print(f"  Max iterations: {rec['max_iter']}")
        if rec['recommended_eps']:
            print(f"  Recommended max ε: {rec['recommended_eps']:.2f}")
            print(f"  Actual iterations needed: {int(rec['actual_iterations'])}")
        else:
            print(f"  Recommendation: Target not achievable within iteration budget")
        print()

    return results_matrix, recommendations

def analyze_geometric_progression_convergence():
    """Analyze convergence when weak learner errors follow geometric progression."""
    print_step_header(5, "Geometric Progression Error Rates")

    print("Analyzing convergence when error rates follow geometric progression:")
    print("εt = ε0 * r^(t-1), where r is the common ratio")
    print()

    # Different geometric progressions
    progressions = {
        'Improving (r=0.9)': {'epsilon_0': 0.4, 'ratio': 0.9, 'color': 'blue'},
        'Improving (r=0.8)': {'epsilon_0': 0.4, 'ratio': 0.8, 'color': 'green'},
        'Constant (r=1.0)': {'epsilon_0': 0.3, 'ratio': 1.0, 'color': 'orange'},
        'Degrading (r=1.1)': {'epsilon_0': 0.2, 'ratio': 1.1, 'color': 'red'}
    }

    T_max = 50
    iterations = range(1, T_max + 1)

    # Calculate error bounds for each progression
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    results = {}
    for name, params in progressions.items():
        eps_0 = params['epsilon_0']
        r = params['ratio']
        color = params['color']

        # Calculate error rates over time
        error_rates = [eps_0 * (r ** (t-1)) for t in iterations]
        # Ensure error rates don't exceed 0.5
        error_rates = [min(eps, 0.49) for eps in error_rates]

        # Calculate convergence factors
        factors = [2 * np.sqrt(eps * (1 - eps)) for eps in error_rates]

        # Calculate cumulative error bound
        cumulative_bound = 1.0
        error_bounds = []
        for factor in factors:
            cumulative_bound *= factor
            error_bounds.append(cumulative_bound)

        results[name] = {
            'error_rates': error_rates,
            'factors': factors,
            'bounds': error_bounds
        }

        # Plot error rates over time
        axes[0, 0].plot(iterations, error_rates, color=color, linewidth=2,
                       marker='o', markersize=3, label=name)

        # Plot error bounds over time
        axes[0, 1].plot(iterations, error_bounds, color=color, linewidth=2, label=name)

    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Weak Learner Error Rate')
    axes[0, 0].set_title('Error Rates Over Time (Geometric Progression)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 0.5)

    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Training Error Bound')
    axes[0, 1].set_title('Training Error Bound Over Time')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Compare final bounds after 50 iterations
    final_bounds = [results[name]['bounds'][-1] for name in progressions.keys()]
    progression_names = list(progressions.keys())

    bars = axes[1, 0].bar(range(len(progression_names)), final_bounds,
                         color=[params['color'] for params in progressions.values()], alpha=0.7)
    axes[1, 0].set_xlabel('Progression Type')
    axes[1, 0].set_ylabel('Final Error Bound (after 50 iterations)')
    axes[1, 0].set_title('Final Error Bounds Comparison')
    axes[1, 0].set_xticks(range(len(progression_names)))
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in progression_names])
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels
    for bar, bound in zip(bars, final_bounds):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                       f'{bound:.2e}', ha='center', va='bottom', rotation=45)

    # Convergence rate analysis
    convergence_rates = []
    for name, data in results.items():
        # Calculate average convergence rate
        bounds = data['bounds']
        if len(bounds) > 1:
            # Fit exponential decay: bound = exp(-rate * t)
            log_bounds = np.log(bounds)
            rate = -(log_bounds[-1] - log_bounds[0]) / len(bounds)
            convergence_rates.append(max(rate, 0))
        else:
            convergence_rates.append(0)

    bars2 = axes[1, 1].bar(range(len(progression_names)), convergence_rates,
                          color=[params['color'] for params in progressions.values()], alpha=0.7)
    axes[1, 1].set_xlabel('Progression Type')
    axes[1, 1].set_ylabel('Average Convergence Rate')
    axes[1, 1].set_title('Average Convergence Rate Comparison')
    axes[1, 1].set_xticks(range(len(progression_names)))
    axes[1, 1].set_xticklabels([name.replace(' ', '\n') for name in progression_names])
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars2, convergence_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                       f'{rate:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometric_progression_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Geometric Progression Analysis Results:")
    print("-" * 40)
    for name, data in results.items():
        final_bound = data['bounds'][-1]
        initial_eps = data['error_rates'][0]
        final_eps = data['error_rates'][-1]
        print(f"{name}:")
        print(f"  Initial ε: {initial_eps:.3f}")
        print(f"  Final ε: {final_eps:.3f}")
        print(f"  Final error bound: {final_bound:.2e}")
        print()

    return results

def main():
    """Main function to run the complete convergence analysis."""
    print("Question 21: AdaBoost Convergence Rate Analysis")
    print("=" * 60)

    # Theoretical analysis
    optimal_eps, practical_eps, practical_factors = analyze_convergence_rate_theory()

    # Iterations vs error bound
    scenarios, iterations_matrix = analyze_iterations_vs_error_bound()

    # Optimal error rate
    theoretical_optimal, thresholds, factors = find_optimal_error_rate()

    # Iterations estimation
    results_matrix, recommendations = estimate_iterations_for_target_error()

    # Geometric progression analysis
    geometric_results = analyze_geometric_progression_convergence()

    print(f"\nAnalysis complete! Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
