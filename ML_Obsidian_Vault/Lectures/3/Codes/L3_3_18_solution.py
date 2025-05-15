import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images") # Goes up one level from Codes to Lectures/3, then to Images
save_dir = os.path.join(images_dir, "L3_3_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Clinical trial data
x_data = np.array([10, 20, 30, 40, 50])  # Dosage (mg)
y_data = np.array([5, 8, 13, 15, 21])    # Blood Pressure Reduction (mmHg)
n = len(x_data)

print("Question 18: Blood Pressure Response to Medication Dosage\n")
print("Data:")
print(f"Dosage (x): {x_data}")
print(f"Blood Pressure Reduction (y): {y_data}")
print(f"Number of data points (n): {n}\n")

# Task 1: Calculate the maximum likelihood estimates for w0 and w1
def calculate_mle_coefficients(x, y):
    print("Task 1: Calculate MLE for w0 and w1")
    
    # Step 1.1: Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    print(f"  Step 1.1: Calculate means of x and y")
    print(f"    x̄ = (x₁ + x₂ + ... + xₙ)/n")
    print(f"      = ({' + '.join(map(str, x))})/5")
    print(f"      = {sum(x)}/5")
    print(f"      = {x_mean:.2f}")
    print()
    
    print(f"    ȳ = (y₁ + y₂ + ... + yₙ)/n")
    print(f"      = ({' + '.join(map(str, y))})/5")
    print(f"      = {sum(y)}/5")
    print(f"      = {y_mean:.2f}")
    print()
    
    # Step 1.2: Calculate deviations from means
    x_dev = x - x_mean
    y_dev = y - y_mean
    
    print(f"  Step 1.2: Calculate deviations from means")
    print(f"    xᵢ - x̄ = {x_dev}")
    print(f"    yᵢ - ȳ = {y_dev}")
    print()
    
    # Step 1.3: Calculate cross-products and squares
    xy_dev = x_dev * y_dev
    x_dev_sq = x_dev**2
    
    print(f"  Step 1.3: Calculate cross-products and squares")
    print(f"    (xᵢ - x̄)(yᵢ - ȳ) = {xy_dev}")
    print(f"    (xᵢ - x̄)² = {x_dev_sq}")
    print()
    
    # Step 1.4: Calculate Sxy and Sxx
    sxy = np.sum(xy_dev)
    sxx = np.sum(x_dev_sq)
    
    print(f"  Step 1.4: Calculate sums")
    print(f"    Sxy = Σ(xᵢ - x̄)(yᵢ - ȳ)")
    print(f"        = {' + '.join(map(str, xy_dev))}")
    print(f"        = {sxy:.2f}")
    print()
    
    print(f"    Sxx = Σ(xᵢ - x̄)²")
    print(f"        = {' + '.join(map(str, x_dev_sq))}")
    print(f"        = {sxx:.2f}")
    print()
    
    # Step 1.5: Calculate MLE for w1 (slope) and w0 (intercept)
    w1_mle = sxy / sxx
    w0_mle = y_mean - w1_mle * x_mean
    
    print(f"  Step 1.5: Calculate MLE for w1 (slope) and w0 (intercept)")
    print(f"    ŵ₁ = Sxy / Sxx")
    print(f"       = {sxy:.2f} / {sxx:.2f}")
    print(f"       = {w1_mle:.4f}")
    print()
    
    print(f"    ŵ₀ = ȳ - ŵ₁x̄")
    print(f"       = {y_mean:.2f} - {w1_mle:.4f} × {x_mean:.2f}")
    print(f"       = {y_mean:.2f} - {w1_mle * x_mean:.4f}")
    print(f"       = {w0_mle:.4f}")
    print()
    
    return w0_mle, w1_mle, sxx, x_mean, x_dev, y_dev

w0_mle, w1_mle, sxx, x_mean, x_dev, y_dev = calculate_mle_coefficients(x_data, y_data)

# Task 2: Estimate the noise variance sigma^2
def estimate_noise_variance_mle(x, y, w0, w1, n_points):
    print("Task 2: Estimate the noise variance σ² (MLE)")
    
    # Step 2.1: Calculate predicted values
    y_pred = w0 + w1 * x
    
    print(f"  Step 2.1: Calculate predicted values")
    print(f"    ŷᵢ = ŵ₀ + ŵ₁xᵢ")
    for i in range(n_points):
        print(f"    ŷ{i+1} = {w0:.4f} + {w1:.4f} × {x[i]}")
        print(f"         = {w0:.4f} + {w1 * x[i]:.4f}")
        print(f"         = {y_pred[i]:.4f}")
    print()
    
    # Step 2.2: Calculate residuals
    residuals = y - y_pred
    
    print(f"  Step 2.2: Calculate residuals")
    print(f"    eᵢ = yᵢ - ŷᵢ")
    for i in range(n_points):
        print(f"    e{i+1} = {y[i]} - {y_pred[i]:.4f}")
        print(f"        = {residuals[i]:.4f}")
    print()
    
    # Step 2.3: Calculate squared residuals
    squared_residuals = residuals**2
    
    print(f"  Step 2.3: Calculate squared residuals")
    print(f"    eᵢ² = (yᵢ - ŷᵢ)²")
    for i in range(n_points):
        print(f"    e{i+1}² = ({residuals[i]:.4f})²")
        print(f"         = {squared_residuals[i]:.4f}")
    print()
    
    # Step 2.4: Calculate RSS and MLE for sigma^2
    rss = np.sum(squared_residuals)
    sigma_sq_mle = rss / n_points  # MLE for variance uses n in the denominator
    
    print(f"  Step 2.4: Calculate Residual Sum of Squares (RSS)")
    print(f"    RSS = Σ(yᵢ - ŷᵢ)²")
    print(f"        = {' + '.join([f'{sr:.4f}' for sr in squared_residuals])}")
    print(f"        = {rss:.4f}")
    print()
    
    print(f"  Step 2.5: Calculate MLE for σ² (noise variance)")
    print(f"    σ̂²_MLE = RSS / n")
    print(f"           = {rss:.4f} / {n_points}")
    print(f"           = {sigma_sq_mle:.4f}")
    print()
    
    return sigma_sq_mle, y_pred, residuals, squared_residuals

sigma_sq_mle, y_pred, residuals, squared_residuals = estimate_noise_variance_mle(x_data, y_data, w0_mle, w1_mle, n)

# Task 3: Write the complete predictive distribution for a new patient receiving a 35mg dose
def get_predictive_distribution(x_new, w0, w1, sigma_sq, n_points, sxx_val, x_mean_val):
    print("Task 3: Predictive distribution for a new patient (x_new = 35mg)")
    
    # Step 3.1: Calculate mean of the predictive distribution
    y_pred_new_mean = w0 + w1 * x_new
    
    print(f"  Step 3.1: Calculate mean of the predictive distribution")
    print(f"    E[y_new|x_new, D] = ŵ₀ + ŵ₁x_new")
    print(f"                      = {w0:.4f} + {w1:.4f} × {x_new}")
    print(f"                      = {w0:.4f} + {w1 * x_new:.4f}")
    print(f"                      = {y_pred_new_mean:.4f}")
    print()
    
    # Step 3.2: Calculate variance of the predictive distribution
    # Var(y_new|x_new, D) = σ̂²_mle * (1 + 1/n + (x_new - x̄)² / Sxx)
    var_term1 = 1
    var_term2 = 1/n_points
    var_term3 = ((x_new - x_mean_val)**2) / sxx_val
    var_factor = 1 + var_term2 + var_term3
    var_pred_new = sigma_sq * var_factor
    std_pred_new = np.sqrt(var_pred_new)
    
    print(f"  Step 3.2: Calculate variance of the predictive distribution")
    print(f"    Var[y_new|x_new, D] = σ̂²_MLE × (1 + 1/n + (x_new - x̄)² / Sxx)")
    print(f"    Term 1 = 1")
    print(f"    Term 2 = 1/n = 1/{n_points} = {var_term2:.4f}")
    print(f"    Term 3 = (x_new - x̄)² / Sxx")
    print(f"            = ({x_new} - {x_mean_val:.2f})² / {sxx_val:.2f}")
    print(f"            = {(x_new - x_mean_val):.2f}² / {sxx_val:.2f}")
    print(f"            = {(x_new - x_mean_val)**2:.2f} / {sxx_val:.2f}")
    print(f"            = {var_term3:.4f}")
    print()
    
    print(f"    Var_factor = 1 + Term 2 + Term 3")
    print(f"               = 1 + {var_term2:.4f} + {var_term3:.4f}")
    print(f"               = {var_factor:.4f}")
    print()
    
    print(f"    Var[y_new|x_new, D] = σ̂²_MLE × Var_factor")
    print(f"                         = {sigma_sq:.4f} × {var_factor:.4f}")
    print(f"                         = {var_pred_new:.4f}")
    print()
    
    print(f"    Standard deviation = √Var[y_new|x_new, D]")
    print(f"                        = √{var_pred_new:.4f}")
    print(f"                        = {std_pred_new:.4f}")
    print()
    
    # Step 3.3: Write the complete predictive distribution
    print(f"  Step 3.3: Complete predictive distribution")
    print(f"    y_new|x_new = {x_new}mg, D ~ N({y_pred_new_mean:.4f}, {var_pred_new:.4f})")
    print()
    
    return y_pred_new_mean, var_pred_new, std_pred_new, var_term1, var_term2, var_term3, var_factor

x_new = 35
y_pred_new_mean, y_pred_new_var, y_pred_new_std, var_term1, var_term2, var_term3, var_factor = get_predictive_distribution(
    x_new, w0_mle, w1_mle, sigma_sq_mle, n, sxx, x_mean
)

# Task 4: FDA requirement: P(y_new >= 12mmHg | x_new = 35mg) >= 0.80
def check_fda_requirement(mean_pred, std_pred, threshold_reduction, required_prob):
    print("Task 4: FDA Requirement Check (P(y_new >= 12mmHg at x_new = 35mg) >= 0.80)")
    
    # Step 4.1: Calculate Z-score for the threshold
    z_score = (threshold_reduction - mean_pred) / std_pred
    
    print(f"  Step 4.1: Calculate Z-score for the threshold")
    print(f"    Z = (threshold - μ) / σ")
    print(f"      = ({threshold_reduction} - {mean_pred:.4f}) / {std_pred:.4f}")
    print(f"      = {(threshold_reduction - mean_pred):.4f} / {std_pred:.4f}")
    print(f"      = {z_score:.4f}")
    print()
    
    # Step 4.2: Calculate the probability using the standard normal CDF
    prob_le_threshold = stats.norm.cdf(z_score)
    prob_ge_threshold = 1 - prob_le_threshold
    
    print(f"  Step 4.2: Calculate the probability")
    print(f"    P(y_new ≥ threshold) = P(Z ≥ z_score)")
    print(f"                          = 1 - P(Z ≤ z_score)")
    print(f"                          = 1 - Φ({z_score:.4f})")
    print(f"                          = 1 - {prob_le_threshold:.4f}")
    print(f"                          = {prob_ge_threshold:.4f}")
    print()
    
    # Step 4.3: Check if the requirement is met
    meets_requirement = prob_ge_threshold >= required_prob
    
    print(f"  Step 4.3: Check if the requirement is met")
    print(f"    FDA requirement: P(y_new ≥ 12mmHg) ≥ 0.80")
    print(f"    Calculated: P(y_new ≥ 12mmHg) = {prob_ge_threshold:.4f}")
    print(f"    Is {prob_ge_threshold:.4f} ≥ {required_prob:.2f}? {'Yes' if meets_requirement else 'No'}")
    print()
    
    return prob_ge_threshold, meets_requirement

threshold_reduction_fda = 12
required_probability_fda = 0.80
prob_ge_12, meets_fda_requirement = check_fda_requirement(
    y_pred_new_mean, y_pred_new_std, threshold_reduction_fda, required_probability_fda
)

# Task 5: Fit quadratic model for comparison
def fit_quadratic_model(x, y):
    # Create design matrix X with columns [1, x, x^2]
    X = np.column_stack((np.ones(len(x)), x, x**2))
    # Calculate OLS/MLE estimates: w = (X^T X)^(-1) X^T y
    w_quad = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    # Calculate predictions
    y_pred_quad = X.dot(w_quad)
    # Calculate residuals
    residuals_quad = y - y_pred_quad
    # Calculate RSS and sigma^2
    rss_quad = np.sum(residuals_quad**2)
    sigma_sq_quad = rss_quad / len(x)
    
    print("Task 5: Fit quadratic model for comparison")
    print(f"  Quadratic model: y = w₀ + w₁x + w₂x²")
    print(f"  MLE estimates: w₀ = {w_quad[0]:.4f}, w₁ = {w_quad[1]:.4f}, w₂ = {w_quad[2]:.4f}")
    print(f"  RSS = {rss_quad:.4f}")
    print(f"  σ² (MLE) = {sigma_sq_quad:.4f}")
    print()
    
    return w_quad, y_pred_quad, residuals_quad, rss_quad, sigma_sq_quad

w_quad, y_pred_quad, residuals_quad, rss_quad, sigma_sq_quad = fit_quadratic_model(x_data, y_data)

# Visualizations
def create_visualizations(x_d, y_d, w0, w1, w_quad, y_pred, y_pred_quad, residuals, x_n, 
                          y_pred_n_mean, y_pred_n_std, threshold, prob_ge_thresh, save_location):
    print("Creating Visualizations...")
    saved_files = []

    # Plot 1: Data, Regression Line, and Prediction Point
    plt.figure(figsize=(10, 6))
    plt.scatter(x_d, y_d, color='blue', label='Clinical Trial Data', s=50)
    
    # Regression line
    x_line = np.linspace(min(x_d)-5, max(x_d)+5, 100)
    y_line = w0 + w1 * x_line
    plt.plot(x_line, y_line, color='red', label=f'Regression Line: ŷ = {w0:.2f} + {w1:.2f}x')
    
    # Prediction point
    plt.scatter([x_n], [y_pred_n_mean], color='green', marker='X', s=100, label=f'Prediction for x={x_n}mg (Mean={y_pred_n_mean:.2f}mmHg)')
    
    plt.xlabel('Dosage (x, mg)', fontsize=12)
    plt.ylabel('Blood Pressure Reduction (y, mmHg)', fontsize=12)
    plt.title('Blood Pressure Response to Medication Dosage', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    file_path1 = os.path.join(save_location, "plot_regression_and_data.png")
    plt.savefig(file_path1, dpi=300, bbox_inches='tight')
    saved_files.append(file_path1)
    print(f"  Saved: {file_path1}")
    plt.close()

    # Plot 2: Predictive Distribution for x_new = 35mg
    plt.figure(figsize=(10, 6))
    x_dist = np.linspace(y_pred_n_mean - 4*y_pred_n_std, y_pred_n_mean + 4*y_pred_n_std, 500)
    y_dist_pdf = stats.norm.pdf(x_dist, loc=y_pred_n_mean, scale=y_pred_n_std)
    
    plt.plot(x_dist, y_dist_pdf, 'b-', linewidth=2, label=f'N({y_pred_n_mean:.2f}, {y_pred_n_std**2:.2f})')
    
    # Shade area for P(y_new >= threshold)
    x_fill = np.linspace(threshold, y_pred_n_mean + 4*y_pred_n_std, 200)
    y_fill_pdf = stats.norm.pdf(x_fill, loc=y_pred_n_mean, scale=y_pred_n_std)
    plt.fill_between(x_fill, y_fill_pdf, color='skyblue', alpha=0.5, label=f'P(y ≥ {threshold}mmHg) = {prob_ge_thresh:.2f}')
    
    plt.axvline(y_pred_n_mean, color='green', linestyle='--', label=f'Mean Prediction = {y_pred_n_mean:.2f}mmHg')
    plt.axvline(threshold, color='red', linestyle=':', label=f'FDA Threshold = {threshold}mmHg')
    
    plt.xlabel('Predicted Blood Pressure Reduction (y_new, mmHg)', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Predictive Distribution for Dosage x = {x_n}mg', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    file_path2 = os.path.join(save_location, "plot_predictive_distribution.png")
    plt.savefig(file_path2, dpi=300, bbox_inches='tight')
    saved_files.append(file_path2)
    print(f"  Saved: {file_path2}")
    plt.close()
    
    # NEW PLOT 3: Residual Analysis
    plt.figure(figsize=(12, 10))
    
    # Residuals vs Fitted Values
    plt.subplot(2, 2, 1)
    plt.scatter(y_pred, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.grid(True)
    
    # Residuals vs X
    plt.subplot(2, 2, 2)
    plt.scatter(x_d, residuals, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Dosage (x)')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Dosage')
    plt.grid(True)
    
    # Normal Q-Q Plot
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, plot=plt)
    plt.title('Normal Q-Q Plot of Residuals')
    plt.grid(True)
    
    # Histogram of Residuals
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=5, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    file_path3 = os.path.join(save_location, "plot_residual_analysis.png")
    plt.savefig(file_path3, dpi=300, bbox_inches='tight')
    saved_files.append(file_path3)
    print(f"  Saved: {file_path3}")
    plt.close()
    
    # NEW PLOT 4: Compare Linear and Quadratic Models
    plt.figure(figsize=(10, 8))
    
    # Data and Models
    plt.subplot(2, 1, 1)
    plt.scatter(x_d, y_d, color='blue', label='Clinical Trial Data', s=50)
    
    # Generate smoother x values for curves
    x_smooth = np.linspace(min(x_d) - 5, max(x_d) + 5, 100)
    
    # Linear model
    y_linear = w0 + w1 * x_smooth
    plt.plot(x_smooth, y_linear, color='red', linewidth=2, 
             label=f'Linear: ŷ = {w0:.2f} + {w1:.2f}x')
    
    # Quadratic model
    y_quad_smooth = w_quad[0] + w_quad[1] * x_smooth + w_quad[2] * x_smooth**2
    plt.plot(x_smooth, y_quad_smooth, color='green', linewidth=2, 
             label=f'Quadratic: ŷ = {w_quad[0]:.2f} + {w_quad[1]:.2f}x + {w_quad[2]:.4f}x²')
    
    plt.xlabel('Dosage (x, mg)')
    plt.ylabel('Blood Pressure Reduction (y, mmHg)')
    plt.title('Comparison of Linear vs Quadratic Models')
    plt.legend()
    plt.grid(True)
    
    # Compare Residuals
    plt.subplot(2, 1, 2)
    
    # Create indices for bar positions
    indices = np.arange(len(x_d))
    bar_width = 0.35
    
    # Plot residuals side by side
    plt.bar(indices - bar_width/2, residuals, bar_width, label='Linear Model Residuals', color='red', alpha=0.7)
    plt.bar(indices + bar_width/2, residuals_quad, bar_width, label='Quadratic Model Residuals', color='green', alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Data Point Index')
    plt.ylabel('Residual Value')
    plt.title('Residuals Comparison')
    plt.xticks(indices, [f'Point {i+1}\n(x={x})' for i, x in enumerate(x_d)])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    file_path4 = os.path.join(save_location, "plot_model_comparison.png")
    plt.savefig(file_path4, dpi=300, bbox_inches='tight')
    saved_files.append(file_path4)
    print(f"  Saved: {file_path4}")
    plt.close()
    
    return saved_files

visualization_paths = create_visualizations(
    x_data, y_data, w0_mle, w1_mle, w_quad, y_pred, y_pred_quad, residuals,
    x_new, y_pred_new_mean, y_pred_new_std, 
    threshold_reduction_fda, prob_ge_12,
    save_dir
)
print(f"Visualizations saved to: {', '.join(visualization_paths)}\n")

print("Task 5: Quadratic Relationship (Conceptual Explanation)")
print("If the true relationship is quadratic (e.g., y = β₀ + β₁x + β₂x² + ε) but a linear model is used:")
print("1. Structural Error (Model Misspecification):")
print("   - The linear model would be misspecified. This means the functional form of the model does not match the true underlying process.")
print("   - The error term ε' = y - (w₀ + w₁x) in the linear model would not just capture random noise. It would also include the systematic error (β₂x²).")
print("   - As a result, the assumption that errors are independent and identically distributed with mean zero (E[ε'] = 0) would be violated. The expected value of the error would depend on x.")
print("   - This leads to biased estimates of w₀ and w₁ (they try to compensate for the missing quadratic term).")
print("   - The estimate of σ² would likely be inflated because it absorbs both true noise and model misspecification error.")
print()
print("   Quantitative comparison from our analysis:")
print(f"   - Linear model RSS: {np.sum(residuals**2):.4f}, σ² (MLE): {sigma_sq_mle:.4f}")
print(f"   - Quadratic model RSS: {rss_quad:.4f}, σ² (MLE): {sigma_sq_quad:.4f}")
print(f"   - Reduction in RSS: {(np.sum(residuals**2) - rss_quad):.4f} ({100 * (1 - rss_quad/np.sum(residuals**2)):.2f}%)")
print()
print("2. Appropriateness of MLE:")
print("   - MLE finds the parameters that maximize the likelihood of observing the data *given the assumed model*.")
print("   - If the assumed model (linear) is incorrect, MLE will still provide the 'best' parameters for that linear model according to the likelihood principle (e.g., minimizing sum of squared errors if Gaussian noise is assumed).")
print("   - However, these parameters (w₀_mle, w₁_mle) will not be good estimates of the true underlying process parameters if the true process is quadratic.")
print("   - Inferences based on these parameters (e.g., confidence intervals, hypothesis tests) would be misleading.")
print("   - To appropriately estimate parameters for a quadratic relationship, one should specify a quadratic model (y = w₀ + w₁x + w₂x² + ε) and then apply MLE (or OLS, which is equivalent for Gaussian noise) to this correct model. This would yield estimates for w₀, w₁, and w₂.")

print("\nEnd of Solution for Question 18.") 