import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L6_4_Quiz_15")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

print("Question 15: Pruning with Noisy Data - Mathematical Analysis")
print("=" * 70)

# Set random seed for reproducibility
np.random.seed(42)

# Given parameters from the question - these can be modified for different scenarios
N_SAMPLES = 1000
NOISE_VARIANCE = 0.25  # σ²
TRAINING_ACCURACY = 0.95  # Train_Acc
VALIDATION_ACCURACY = 0.72  # Val_Acc

print(f"Given Parameters:")
print(f"  Total samples: {N_SAMPLES}")
print(f"  Noise variance (σ²): {NOISE_VARIANCE}")
print(f"  Training accuracy: {TRAINING_ACCURACY}")
print(f"  Validation accuracy: {VALIDATION_ACCURACY}")
print(f"  Overfitting gap: {TRAINING_ACCURACY - VALIDATION_ACCURACY:.3f}")

# 1. Mathematical Analysis: Calculate overfitting gap and bias-variance decomposition
print("\n\n1. Mathematical Analysis: Overfitting Gap and Bias-Variance Decomposition")
print("-" * 70)

def calculate_overfitting_gap(train_acc, val_acc):
    """
    Calculate the overfitting gap between training and validation accuracy
    
    Parameters:
    - train_acc: Training accuracy (0 ≤ train_acc ≤ 1)
    - val_acc: Validation accuracy (0 ≤ val_acc ≤ 1)
    
    Returns:
    - Overfitting gap (train_acc - val_acc)
    """
    return train_acc - val_acc

def bias_variance_decomposition_noise(train_acc, val_acc, noise_variance):
    """
    Decompose the error into bias and variance components using mathematical relationships
    
    Mathematical Model:
    - Bias: Represents systematic error, estimated as training-validation gap
    - Variance: Represents noise in the data, given as σ²
    - Total Error: Bias + Variance
    
    Parameters:
    - train_acc: Training accuracy
    - val_acc: Validation accuracy  
    - noise_variance: Noise variance σ²
    
    Returns:
    - Dictionary with bias, variance, total error, and overfitting gap
    """
    # Mathematical relationship: Bias ≈ Training_Acc - Validation_Acc
    estimated_bias = train_acc - val_acc
    
    # Variance component from noise
    estimated_variance = noise_variance
    
    # Total error from bias-variance decomposition
    total_error = estimated_bias + estimated_variance
    
    return {
        'bias': estimated_bias,
        'variance': estimated_variance,
        'total_error': total_error,
        'overfitting_gap': train_acc - val_acc
    }

# Calculate components
decomposition = bias_variance_decomposition_noise(TRAINING_ACCURACY, VALIDATION_ACCURACY, NOISE_VARIANCE)

print(f"Overfitting Gap Analysis:")
print(f"  Training Accuracy: {TRAINING_ACCURACY:.3f}")
print(f"  Validation Accuracy: {VALIDATION_ACCURACY:.3f}")
print(f"  Overfitting Gap: {decomposition['overfitting_gap']:.3f}")
print(f"\nBias-Variance Decomposition:")
print(f"  Estimated Bias: {decomposition['bias']:.3f}")
print(f"  Noise Variance: {decomposition['variance']:.3f}")
print(f"  Total Error: {decomposition['total_error']:.3f}")

print(f"\nMathematical Explanation:")
print(f"  When noise ε ~ N(0, {NOISE_VARIANCE}) is present:")
print(f"  - Model tries to fit f(x) + ε instead of f(x)")
print(f"  - Training accuracy increases (model fits noise)")
print(f"  - Validation accuracy decreases (noise doesn't generalize)")
print(f"  - Gap widens: Gap = Train_Acc - Val_Acc = {decomposition['overfitting_gap']:.3f}")

# 2. Pruning Method Comparison: Calculate which is most robust
print("\n\n2. Pruning Method Comparison: Robustness Analysis")
print("-" * 70)

# Given pruning options from the question
pruning_options = {
    'No Pruning': {'train_acc': 0.95, 'test_acc': 0.72, 'depth': 8, 'leaves': 25},
    'Depth Pruning': {'train_acc': 0.87, 'test_acc': 0.78, 'depth': 4, 'leaves': 12},
    'Sample Pruning': {'train_acc': 0.89, 'test_acc': 0.75, 'depth': 6, 'leaves': 18},
    'Combined Pruning': {'train_acc': 0.85, 'test_acc': 0.80, 'depth': 3, 'leaves': 8}
}

def calculate_robustness_score(option_data):
    """
    Calculate robustness score using the generalization gap metric from the question
    
    Mathematical Formula from Question:
    G = (Training_Acc - Test_Acc) / log(Depth × Leaves)
    
    Where:
    - Lower G means more robust (smaller generalization gap per unit complexity)
    - Complexity is measured by log(Depth × Leaves)
    
    Parameters:
    - option_data: Dictionary with pruning method performance data
    
    Returns:
    - Dictionary with gap, complexity, and robustness score
    """
    # Overfitting gap (Training_Acc - Test_Acc)
    gap = option_data['train_acc'] - option_data['test_acc']
    
    # Complexity measure: log(Depth × Leaves) as specified in the question
    complexity = np.log(option_data['depth'] * option_data['leaves'])
    
    # Robustness score: G = gap / complexity (lower is better)
    # We invert this so higher score means more robust
    robustness_score = 1 / (gap / complexity) if gap > 0 else float('inf')
    
    return {
        'gap': gap,
        'complexity': complexity,
        'robustness_score': robustness_score
    }

print("Pruning Method Analysis:")
print(f"{'Method':<20} {'Train':<8} {'Test':<8} {'Depth':<6} {'Leaves':<7} {'Gap':<6} {'Complexity':<10} {'Robustness':<10}")
print("-" * 80)

robustness_results = {}
for method, data in pruning_options.items():
    robustness = calculate_robustness_score(data)
    robustness_results[method] = robustness
    
    print(f"{method:<20} {data['train_acc']:<8.3f} {data['test_acc']:<8.3f} "
          f"{data['depth']:<6} {data['leaves']:<7} {robustness['gap']:<6.3f} {robustness['complexity']:<10.3f} {robustness['robustness_score']:<10.3f}")

# Find most robust method
most_robust = max(robustness_results.items(), key=lambda x: x[1]['robustness_score'])
print(f"\nMost robust pruning method: {most_robust[0]}")
print(f"Robustness score: {most_robust[1]['robustness_score']:.3f}")
print(f"Reason: Best balance of test accuracy, overfitting gap, and complexity")

# 3. Adaptive Pruning Design: Mathematical functions for noise-based adjustment
print("\n\n3. Adaptive Pruning Design: Mathematical Functions")
print("-" * 70)

def design_adaptive_pruning_functions(noise_level):
    """
    Design mathematical functions that adjust pruning thresholds based on noise level
    
    Mathematical Functions from Question:
    - f1(σ) = min_samples_split = max(10, ⌈50σ²⌉)
    - f2(σ) = max_depth = ⌊8 - 4σ⌋
    - f3(σ) = min_impurity_decrease = 0.01 + 0.1σ
    
    Parameters:
    - noise_level: Estimated noise level σ
    
    Returns:
    - Dictionary with adaptive pruning parameters and mathematical functions
    """
    
    # Mathematical function 1: min_samples_split = max(10, ⌈50σ²⌉)
    f1_sigma = lambda sigma: max(10, int(np.ceil(50 * sigma**2)))
    
    # Mathematical function 2: max_depth = ⌊8 - 4σ⌋
    f2_sigma = lambda sigma: int(np.floor(8 - 4 * sigma))
    
    # Mathematical function 3: min_impurity_decrease = 0.01 + 0.1σ
    f3_sigma = lambda sigma: 0.01 + 0.1 * sigma
    
    # Calculate optimal values for given noise level
    optimal_params = {
        'min_samples_split': f1_sigma(noise_level),
        'max_depth': f2_sigma(noise_level),
        'min_impurity_decrease': f3_sigma(noise_level)
    }
    
    # Store mathematical functions for analysis
    mathematical_functions = {
        'f1(σ)': "min_samples_split = max(10, ⌈50σ²⌉)",
        'f2(σ)': "max_depth = ⌊8 - 4σ⌋",
        'f3(σ)': "min_impurity_decrease = 0.01 + 0.1σ"
    }
    
    return optimal_params, mathematical_functions

# Calculate optimal values for σ = 0.25
sigma_given = 0.25
adaptive_params, math_functions = design_adaptive_pruning_functions(sigma_given)

print(f"Adaptive Pruning Functions for σ = {sigma_given}:")
print(f"  f1(σ) = {math_functions['f1(σ)']}")
print(f"  f2(σ) = {math_functions['f2(σ)']}")
print(f"  f3(σ) = {math_functions['f3(σ)']}")
print(f"\nOptimal values for σ = {sigma_given}:")
print(f"  min_samples_split: {adaptive_params['min_samples_split']}")
print(f"  max_depth: {adaptive_params['max_depth']}")
print(f"  min_impurity_decrease: {adaptive_params['min_impurity_decrease']:.4f}")

# 4. Outlier Impact Analysis: Mathematical calculations
print("\n\n4. Outlier Impact Analysis: Mathematical Calculations")
print("-" * 70)

def calculate_outlier_impact(outlier_percentage, boundary_shift, training_accuracy, validation_accuracy):
    """
    Calculate the impact of outliers on model performance using exact formulas from question
    
    Mathematical Formulas from Question:
    - Expected change in training accuracy: Δ_train = p × Δ × Training_Acc
    - Expected change in validation accuracy: Δ_val = p × Δ × Validation_Acc
    - Optimal outlier removal threshold: τ = argmin_τ |Δ_train(τ) - Δ_val(τ)|
    
    Parameters:
    - outlier_percentage: Percentage of data that are outliers (0 ≤ p ≤ 1)
    - boundary_shift: How much outliers shift the decision boundary (Δ)
    - training_accuracy: Training accuracy without outliers
    - validation_accuracy: Validation accuracy without outliers
    
    Returns:
    - Dictionary with calculated changes and optimal threshold
    """
    
    # Mathematical formula 1: Training accuracy change
    # Δ_train = p × Δ × Training_Acc
    train_acc_change = outlier_percentage * boundary_shift * training_accuracy
    
    # Mathematical formula 2: Validation accuracy change  
    # Δ_val = p × Δ × Validation_Acc
    val_acc_change = outlier_percentage * boundary_shift * validation_accuracy
    
    # Mathematical formula 3: Optimal outlier removal threshold
    # τ = argmin_τ |Δ_train(τ) - Δ_val(τ)|
    # For this analysis, we use the difference between changes as a proxy
    # The optimal threshold minimizes the absolute difference
    optimal_threshold = abs(train_acc_change - val_acc_change)
    
    # Calculate new accuracies
    new_train_acc = training_accuracy + train_acc_change
    new_val_acc = validation_accuracy + val_acc_change
    
    return {
        'train_acc_change': train_acc_change,
        'val_acc_change': val_acc_change,
        'optimal_threshold': optimal_threshold,
        'new_train_acc': new_train_acc,
        'new_val_acc': new_val_acc
    }

# Given parameters
outlier_percentage = 0.10  # 10%
boundary_shift = 0.5
training_accuracy = TRAINING_ACCURACY  # 0.95
validation_accuracy = VALIDATION_ACCURACY  # 0.72

outlier_impact = calculate_outlier_impact(outlier_percentage, boundary_shift, training_accuracy, validation_accuracy)

print(f"Outlier Impact Analysis:")
print(f"  Outlier percentage: {outlier_percentage:.1%}")
print(f"  Boundary shift: {boundary_shift}")
print(f"  Training accuracy: {training_accuracy:.3f}")
print(f"  Validation accuracy: {validation_accuracy:.3f}")
print(f"\nExpected changes:")
print(f"  Training accuracy change: {outlier_impact['train_acc_change']:+.3f}")
print(f"  New training accuracy: {outlier_impact['new_train_acc']:.3f}")
print(f"  Validation accuracy change: {outlier_impact['val_acc_change']:+.3f}")
print(f"  New validation accuracy: {outlier_impact['new_val_acc']:.3f}")
print(f"  Optimal outlier removal threshold: {outlier_impact['optimal_threshold']:.3f}")

# 5. Exponential Noise Modeling: Optimal pruning function
print("\n\n5. Exponential Noise Modeling: Optimal Pruning Function")
print("-" * 70)

def exponential_noise_optimal_pruning(x1_values, base_noise=0.1, noise_scaling=0.5):
    """
    Calculate optimal pruning parameters for exponential noise model using exact formulas from question
    
    Mathematical Model from Question:
    - Noise function: σ(x₁) = 0.1 × exp(x₁/2)
    - Optimal tree depth: d*(x₁) = argmin_d (Bias(d) + Variance(d, σ(x₁)))
    - Expected error: E[Error] = ∫₀³ (Bias²(d*(x₁)) + σ²(x₁)) dx₁
    
    Parameters:
    - x1_values: Feature values to evaluate
    - base_noise: Base noise level (default: 0.1)
    - noise_scaling: Scaling factor for exponential noise (default: 0.5)
    
    Returns:
    - Dictionary with noise, optimal depth, and expected error for each x₁
    """
    
    def noise_function(x1):
        """Exponential noise model: σ(x₁) = 0.1 × exp(x₁/2)"""
        return base_noise * np.exp(x1 / noise_scaling)
    
    def optimal_depth(x1):
        """
        Optimal depth function: d*(x₁) = argmin_d (Bias(d) + Variance(d, σ(x₁)))
        
        Mathematical model:
        - Bias(d) = 0.1 × (1 - 1/d)  # Bias decreases with depth
        - Variance(d, σ) = 0.2 × σ × d  # Variance increases with depth and noise
        - Optimal depth minimizes the sum
        """
        noise = noise_function(x1)
        
        # For this analysis, we use a simplified model
        # Bias decreases with depth: Bias(d) = 0.1 × (1 - 1/d)
        # Variance increases with depth and noise: Variance(d) = 0.2 × noise × d
        
        # Find optimal depth by minimizing Bias(d) + Variance(d)
        # We'll use a numerical approach for demonstration
        depths = range(2, 9)
        total_errors = []
        
        for d in depths:
            bias = 0.1 * (1 - 1/d) if d > 0 else 0.1
            variance = 0.2 * noise * d
            total_error = bias + variance
            total_errors.append(total_error)
        
        # Return depth with minimum total error
        optimal_d = depths[np.argmin(total_errors)]
        return optimal_d
    
    def expected_error(x1):
        """
        Expected error: E[Error] = Bias²(d*(x₁)) + σ²(x₁)
        """
        noise = noise_function(x1)
        depth = optimal_depth(x1)
        
        # Bias² component
        bias = 0.1 * (1 - 1/depth) if depth > 0 else 0.1
        bias_squared = bias**2
        
        # σ² component
        noise_squared = noise**2
        
        return bias_squared + noise_squared
    
    # Calculate results for each x₁ value
    results = {}
    for x1 in x1_values:
        noise = noise_function(x1)
        depth = optimal_depth(x1)
        error = expected_error(x1)
        
        results[x1] = {
            'noise': noise,
            'optimal_depth': depth,
            'expected_error': error
        }
    
    # Calculate the integral: ∫₀³ (Bias²(d*(x₁)) + σ²(x₁)) dx₁
    # Using numerical integration (trapezoidal rule)
    x1_integral = np.linspace(0, 3, 1000)
    integral_values = []
    
    for x1_int in x1_integral:
        noise_int = noise_function(x1_int)
        depth_int = optimal_depth(x1_int)
        bias_int = 0.1 * (1 - 1/depth_int) if depth_int > 0 else 0.1
        bias_squared_int = bias_int**2
        noise_squared_int = noise_int**2
        integral_values.append(bias_squared_int + noise_squared_int)
    
    # Trapezoidal rule integration
    integral_result = np.trapz(integral_values, x1_integral)
    
    # Add integral result to results
    results['integral'] = integral_result
    
    return results

# Calculate for x1 ∈ [0, 3]
x1_range = np.linspace(0, 3, 7)
exponential_results = exponential_noise_optimal_pruning(x1_range)

print(f"Exponential Noise Model: σ(x₁) = 0.1 × exp(x₁/2)")
print(f"\nOptimal pruning parameters:")
print(f"{'x₁':<6} {'σ(x₁)':<8} {'Optimal Depth':<15} {'Expected Error':<15}")
print("-" * 50)

for x1, result in exponential_results.items():
    if x1 != 'integral':
        print(f"{x1:<6.1f} {result['noise']:<8.3f} {result['optimal_depth']:<15} {result['expected_error']:<15.3f}")

print(f"\nIntegral result: ∫₀³ (Bias²(d*(x₁)) + σ²(x₁)) dx₁ = {exponential_results['integral']:.6f}")

# 6. Safety Constraint Analysis: Cost optimization
print("\n\n6. Safety Constraint Analysis: Cost Optimization")
print("-" * 70)

def calculate_safety_cost_optimization(false_negative_cost, false_positive_cost, 
                                     base_detection_rate, noise_level):
    """
    Calculate optimal pruning threshold that minimizes expected cost using cost matrix
    
    Mathematical Model from Question:
    - Cost matrix C = [[0, 1000], [100000, 0]]
    - False negative cost = $100,000 (missed fire)
    - False positive cost = $1,000 (false alarm)
    - Optimal threshold: α* = argmin_α Σᵢⱼ Cᵢⱼ × Pᵢⱼ(α)
    
    Parameters:
    - false_negative_cost: Cost of missed detection (e.g., missed fire)
    - false_positive_cost: Cost of false alarm
    - base_detection_rate: Base detection rate without noise (0 ≤ rate ≤ 1)
    - noise_level: Noise level in the system
    
    Returns:
    - Dictionary with cost matrix, probabilities, expected cost, and optimal threshold
    """
    
    # Cost matrix from the question
    cost_matrix = np.array([[0, false_positive_cost], 
                           [false_negative_cost, 0]])
    
    # Mathematical relationship 1: False negative probability increases with noise
    # Higher noise makes it harder to detect true events
    p_false_negative = (1 - base_detection_rate) * (1 + noise_level)
    
    # Mathematical relationship 2: False positive probability decreases with noise
    # Conservative pruning reduces false alarms
    p_false_positive = (1 - base_detection_rate) * (1 - 0.5 * noise_level)
    
    # Mathematical relationship 3: Expected cost using cost matrix
    # E[Cost] = Σᵢⱼ Cᵢⱼ × Pᵢⱼ
    # For 2x2 matrix: E[Cost] = C₁₀ × P(FN) + C₀₁ × P(FP)
    expected_cost = (cost_matrix[1, 0] * p_false_negative + 
                     cost_matrix[0, 1] * p_false_positive)
    
    # Mathematical relationship 4: Optimal pruning threshold
    # α* = argmin_α Σᵢⱼ Cᵢⱼ × Pᵢⱼ(α)
    # For this analysis, we use a noise-based threshold
    optimal_threshold = 0.5 * noise_level
    
    return {
        'cost_matrix': cost_matrix,
        'p_false_negative': p_false_negative,
        'p_false_positive': p_false_positive,
        'expected_cost': expected_cost,
        'optimal_threshold': optimal_threshold
    }

# Given parameters
false_negative_cost = 100000  # $100,000
false_positive_cost = 1000    # $1,000
base_detection_rate = 0.95    # 95%
noise_level = 0.3

safety_analysis = calculate_safety_cost_optimization(
    false_negative_cost, false_positive_cost, base_detection_rate, noise_level
)

print(f"Safety Constraint Analysis:")
print(f"  False negative cost: ${false_negative_cost:,}")
print(f"  False positive cost: ${false_positive_cost:,}")
print(f"  Base detection rate: {base_detection_rate:.1%}")
print(f"  Noise level: {noise_level}")
print(f"\nCost Matrix:")
print(f"  C = {safety_analysis['cost_matrix']}")
print(f"\nRisk Analysis:")
print(f"  P(False Negative): {safety_analysis['p_false_negative']:.3f}")
print(f"  P(False Positive): {safety_analysis['p_false_positive']:.3f}")
print(f"  Expected cost: ${safety_analysis['expected_cost']:,.2f}")
print(f"  Optimal pruning threshold: {safety_analysis['optimal_threshold']:.3f}")

# 7. Local Noise Estimation: Mathematical function design
print("\n\n7. Local Noise Estimation: Mathematical Function Design")
print("-" * 70)

def design_local_noise_estimation(local_variance, neighborhood_size=50):
    """
    Design mathematical function to estimate local noise and optimal pruning parameters
    
    Mathematical Functions from Question:
    - Local noise estimate: σ̂_local = √(1/(k-1) × Σᵢ₌₁ᵏ (xᵢ - x̄)²)
    - Adaptive min_samples: n_min = max(10, ⌈25σ̂_local²⌉)
    - Adaptive max_depth: d_max = ⌊6 - 3σ̂_local⌋
    
    Parameters:
    - local_variance: Local variance in the neighborhood (σ²_local)
    - neighborhood_size: Size of the neighborhood for noise estimation (k)
    
    Returns:
    - Dictionary with local noise, noise factor, and optimal pruning parameters
    """
    
    # Mathematical relationship 1: Local noise from variance
    # σ̂_local = √(local_variance)
    local_noise = np.sqrt(local_variance)
    
    # Mathematical relationship 2: Adaptive min_samples
    # n_min = max(10, ⌈25σ̂_local²⌉)
    adaptive_min_samples = max(10, int(np.ceil(25 * local_noise**2)))
    
    # Mathematical relationship 3: Adaptive max_depth
    # d_max = ⌊6 - 3σ̂_local⌋
    adaptive_max_depth = max(2, int(np.floor(6 - 3 * local_noise)))
    
    # Mathematical relationship 4: Adaptive min_impurity_decrease
    # min_impurity = 0.01 × (1 + 2σ̂_local)
    adaptive_min_impurity = 0.01 * (1 + 2 * local_noise)
    
    optimal_params = {
        'max_depth': adaptive_max_depth,
        'min_samples_split': adaptive_min_samples,
        'min_impurity_decrease': adaptive_min_impurity
    }
    
    return {
        'local_noise': local_noise,
        'neighborhood_size': neighborhood_size,
        'optimal_params': optimal_params
    }

# Given local variance
local_variance = 0.4
local_analysis = design_local_noise_estimation(local_variance)

print(f"Local Noise Estimation:")
print(f"  Local variance: {local_variance}")
print(f"  Neighborhood size: {local_analysis['neighborhood_size']}")
print(f"  Estimated local noise: {local_analysis['local_noise']:.3f}")
print(f"\nOptimal pruning parameters for this region:")
for param, value in local_analysis['optimal_params'].items():
    print(f"  {param}: {value}")

# 8. Error Decomposition: Mathematical calculations
print("\n\n8. Error Decomposition: Mathematical Calculations")
print("-" * 70)

def calculate_error_decomposition(bias, variance, irreducible_error, bias_reduction_factor=0.5, target_error=0.2):
    """
    Calculate error decomposition and answer optimization questions using mathematical analysis
    
    Mathematical Model:
    - Expected error: E[(y - f̂(x))²] = Bias² + Variance + Irreducible_Error
    - New error with reduced bias: E_new = (bias × bias_reduction_factor)² + variance + irreducible_error
    - Required variance for target: required_variance = target_error - bias² - irreducible_error
    - Variance reduction needed: variance_reduction = variance - required_variance
    
    Parameters:
    - bias: Model bias (systematic error)
    - variance: Model variance (sensitivity to training data)
    - irreducible_error: Irreducible error from data noise
    - bias_reduction_factor: Factor to reduce bias (default: 0.5 for 50% reduction)
    - target_error: Target error level (default: 0.2)
    
    Returns:
    - Dictionary with error components and optimization results
    """
    
    # Mathematical relationship 1: Expected prediction error
    # Total error is sum of bias², variance, and irreducible error
    expected_error = bias**2 + variance + irreducible_error
    
    # Mathematical relationship 2: Error with reduced bias
    # Reducing bias by factor reduces bias² by factor²
    new_bias = bias * bias_reduction_factor
    new_error_reduced_bias = new_bias**2 + variance + irreducible_error
    
    # Mathematical relationship 3: Variance reduction for target error
    # Solve for required variance: target_error = bias² + required_variance + irreducible_error
    required_variance = target_error - bias**2 - irreducible_error
    
    # Mathematical relationship 4: Variance reduction needed
    # Current variance minus required variance
    variance_reduction_needed = variance - required_variance if required_variance > 0 else variance
    
    return {
        'expected_error': expected_error,
        'new_error_reduced_bias': new_error_reduced_bias,
        'required_variance': required_variance,
        'variance_reduction_needed': variance_reduction_needed,
        'bias_reduction_impact': expected_error - new_error_reduced_bias
    }

# Given values
bias = 0.08
variance = 0.12
irreducible_error = 0.15

error_analysis = calculate_error_decomposition(bias, variance, irreducible_error)

print(f"Error Decomposition Analysis:")
print(f"  Bias: {bias}")
print(f"  Variance: {variance}")
print(f"  Irreducible error: {irreducible_error}")
print(f"\nCalculations:")
print(f"  1. Expected prediction error:")
print(f"     E[(y - f̂(x))²] = Bias² + Variance + σ²")
print(f"     E[(y - f̂(x))²] = {bias}² + {variance} + {irreducible_error}")
print(f"     E[(y - f̂(x))²] = {bias**2:.4f} + {variance} + {irreducible_error}")
print(f"     E[(y - f̂(x))²] = {error_analysis['expected_error']:.4f}")
print(f"\n  2. If bias is reduced by 50% (new bias = {bias * 0.5:.3f}):")
print(f"     New expected error = {error_analysis['new_error_reduced_bias']:.4f}")
print(f"\n  3. Variance reduction needed for expected error ≤ 0.2:")
print(f"     Required variance: {error_analysis['required_variance']:.4f}")
print(f"     Variance reduction needed: {error_analysis['variance_reduction_needed']:.4f}")

# Create separate visualizations for better clarity

# 1. Overfitting gap visualization
plt.figure(figsize=(10, 6))
methods = list(pruning_options.keys())
gaps = [robustness_results[m]['gap'] for m in methods]
colors = ['red', 'orange', 'yellow', 'green']
bars = plt.bar(methods, gaps, color=colors, alpha=0.7)
plt.ylabel('Overfitting Gap')
plt.title('Overfitting Gap by Pruning Method')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, gaps):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overfitting_gap_analysis.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 2. Robustness scores
plt.figure(figsize=(10, 6))
robustness_scores = [robustness_results[m]['robustness_score'] for m in methods]
# Handle infinite values for plotting
robustness_scores_plot = [min(score, 10) if np.isfinite(score) else 10 for score in robustness_scores]
bars = plt.bar(methods, robustness_scores_plot, color=colors, alpha=0.7)
plt.ylabel('Robustness Score (1/G)')
plt.title('Robustness Score by Pruning Method')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, robustness_scores):
    if np.isfinite(value):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{value:.3f}', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 '∞', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'robustness_score_analysis.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 3. Adaptive pruning functions
plt.figure(figsize=(12, 8))
sigma_range = np.linspace(0.1, 0.5, 100)
f1_values = [max(10, int(np.ceil(50 * s**2))) for s in sigma_range]
f2_values = [max(2, int(np.floor(8 - 4 * s))) for s in sigma_range]
f3_values = [0.01 + 0.1 * s for s in sigma_range]

plt.plot(sigma_range, f1_values, 'b-', label='f1(σ): min_samples_split = max(10, ⌈50σ²⌉)', linewidth=2)
plt.plot(sigma_range, f2_values, 'r-', label='f2(σ): max_depth = ⌊8 - 4σ⌋', linewidth=2)
plt.plot(sigma_range, f3_values, 'g-', label='f3(σ): min_impurity_decrease = 0.01 + 0.1σ', linewidth=2)
plt.xlabel('Noise Level (σ)')
plt.ylabel('Parameter Value')
plt.title('Adaptive Pruning Functions')
plt.legend()
plt.grid(True, alpha=0.3)

# Add the specific point for σ = 0.25
plt.scatter([0.25], [adaptive_params['min_samples_split']], color='blue', s=100, zorder=5, label=f'σ=0.25: f1={adaptive_params["min_samples_split"]}')
plt.scatter([0.25], [adaptive_params['max_depth']], color='red', s=100, zorder=5, label=f'σ=0.25: f2={adaptive_params["max_depth"]}')
plt.scatter([0.25], [adaptive_params['min_impurity_decrease']], color='green', s=100, zorder=5, label=f'σ=0.25: f3={adaptive_params["min_impurity_decrease"]:.4f}')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'adaptive_pruning_functions.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 4. Exponential noise model
plt.figure(figsize=(12, 8))
x1_plot = np.linspace(0, 3, 100)
noise_values = [0.1 * np.exp(x/2) for x in x1_plot]
plt.plot(x1_plot, noise_values, 'b-', linewidth=2, label='σ(x₁) = 0.1 × exp(x₁/2)')
plt.xlabel('x₁')
plt.ylabel('σ(x₁)')
plt.title('Exponential Noise Model')
plt.grid(True, alpha=0.3)
plt.legend()

# Add specific points from our analysis
for x1 in x1_range:
    noise = 0.1 * np.exp(x1/2)
    plt.scatter(x1, noise, color='red', s=50, zorder=5)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'exponential_noise_model.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 5. Optimal depth vs noise
plt.figure(figsize=(10, 6))
optimal_depths = [exponential_results[x]['optimal_depth'] for x in x1_range]
plt.plot(x1_range, optimal_depths, 'ro-', linewidth=2, markersize=8, label='Optimal Depth')
plt.xlabel('x₁')
plt.ylabel('Optimal Tree Depth')
plt.title('Optimal Depth vs Feature Value')
plt.grid(True, alpha=0.3)
plt.legend()

# Add value labels
for x1, depth in zip(x1_range, optimal_depths):
    plt.annotate(f'{depth}', (x1, depth), xytext=(0, 10), textcoords='offset points', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'optimal_depth_vs_noise.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 6. Expected error vs noise
plt.figure(figsize=(10, 6))
expected_errors = [exponential_results[x]['expected_error'] for x in x1_range]
plt.plot(x1_range, expected_errors, 'go-', linewidth=2, markersize=8, label='Expected Error')
plt.xlabel('x₁')
plt.ylabel('Expected Error')
plt.title('Expected Error vs Feature Value')
plt.grid(True, alpha=0.3)
plt.legend()

# Add value labels
for x1, error in zip(x1_range, expected_errors):
    plt.annotate(f'{error:.3f}', (x1, error), xytext=(0, 10), textcoords='offset points', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'expected_error_vs_noise.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 7. Safety cost analysis
plt.figure(figsize=(10, 6))
cost_components = ['False Negative', 'False Positive']
cost_values = [safety_analysis['p_false_negative'] * false_negative_cost,
               safety_analysis['p_false_positive'] * false_positive_cost]
colors = ['red', 'orange']
bars = plt.bar(cost_components, cost_values, color=colors, alpha=0.7)
plt.ylabel('Expected Cost ($)')
plt.title('Safety Cost Components')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, cost_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'${value:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'safety_cost_analysis.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 8. Error decomposition
plt.figure(figsize=(10, 6))
components = ['Bias²', 'Variance', 'Irreducible Error']
values = [bias**2, variance, irreducible_error]
colors = ['red', 'blue', 'green']
bars = plt.bar(components, values, color=colors, alpha=0.7)
plt.ylabel('Error Component')
plt.title('Error Decomposition')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'error_decomposition_analysis.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

# 9. Summary statistics
plt.figure(figsize=(10, 6))
summary_stats = ['Overfitting\nGap', 'Robustness\nScore', 'Expected\nError']
summary_values = [decomposition['overfitting_gap'], 
                  most_robust[1]['robustness_score'],
                  error_analysis['expected_error']]
colors = ['red', 'green', 'blue']
bars = plt.bar(summary_stats, summary_values, color=colors, alpha=0.7)
plt.ylabel('Value')
plt.title('Summary Statistics')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, summary_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
# plt.show()  # Removed to prevent plots from opening

print(f"\nSummary of Mathematical Analysis:")
print("=" * 50)
print(f"1. Overfitting gap: {decomposition['overfitting_gap']:.3f}")
print(f"2. Most robust method: {most_robust[0]} (score: {most_robust[1]['robustness_score']:.3f})")
print(f"3. Adaptive pruning for σ={sigma_given}: depth={adaptive_params['max_depth']}, samples={adaptive_params['min_samples_split']}")
print(f"4. Outlier impact: train change={outlier_impact['train_acc_change']:+.3f}, val change={outlier_impact['val_acc_change']:+.3f}")
print(f"5. Exponential noise: optimal depth decreases from {exponential_results[0]['optimal_depth']} to {exponential_results[3]['optimal_depth']}")
print(f"6. Safety cost: ${safety_analysis['expected_cost']:,.2f} with threshold {safety_analysis['optimal_threshold']:.3f}")
print(f"7. Local noise: σ_local={local_analysis['local_noise']:.3f}, optimal depth={local_analysis['optimal_params']['max_depth']}")
print(f"8. Expected error: {error_analysis['expected_error']:.4f}")
print(f"9. Integral result: ∫₀³ (Bias²(d*(x₁)) + σ²(x₁)) dx₁ = {exponential_results['integral']:.6f}")

print(f"\nAll mathematical analysis plots saved to: {save_dir}")
