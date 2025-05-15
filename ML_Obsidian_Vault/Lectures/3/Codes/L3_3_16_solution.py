import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_3_Quiz_16")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# True model parameters
true_w0 = 2  # true intercept
true_w1 = 3  # true slope
true_sigma_squared = 4  # true noise variance

# Given data points
x = np.array([1, 2, 3, 4])
y = np.array([6, 9, 11, 16])

print("Treasure Hunter on Probability Island\n")
print("Given information:")
print(f"True model: y = {true_w1}x + {true_w0} + ε, where ε ~ N(0, {true_sigma_squared})")
print("\nObserved treasure locations:")
for i in range(len(x)):
    print(f"Treasure {i+1}: x = {x[i]}, y = {y[i]}")
print("\n")

# Step 1: Write the likelihood function
def step1_likelihood_function():
    """Write and explain the likelihood function."""
    print("Step 1: Writing the likelihood function\n")
    
    print("For a linear regression model y = w₁x + w₀ + ε where ε ~ N(0, σ²),")
    print("the likelihood function is the product of probabilities of observing each data point:")
    print("L(w₀, w₁, σ² | data) = ∏ᵢ₌₁ⁿ P(yᵢ | xᵢ, w₀, w₁, σ²)")
    print("\nSince y follows a normal distribution with mean w₁x + w₀ and variance σ²:")
    print("P(yᵢ | xᵢ, w₀, w₁, σ²) = (1/√(2πσ²)) * exp(-(yᵢ - (w₁xᵢ + w₀))²/(2σ²))")
    print("\nThe likelihood function becomes:")
    print("L(w₀, w₁, σ² | data) = ∏ᵢ₌₁ⁿ (1/√(2πσ²)) * exp(-(yᵢ - (w₁xᵢ + w₀))²/(2σ²))")
    print("\nTaking the natural logarithm (log-likelihood):")
    print("ln L(w₀, w₁, σ² | data) = -n/2 * ln(2πσ²) - (1/2σ²) * ∑ᵢ₌₁ⁿ (yᵢ - (w₁xᵢ + w₀))²")
    
    # Implement the likelihood function
    def likelihood(w0, w1, sigma_squared, x, y):
        n = len(x)
        residuals = y - (w1 * x + w0)
        exponent = -np.sum(residuals**2) / (2 * sigma_squared)
        coefficient = (2 * np.pi * sigma_squared) ** (-n/2)
        return coefficient * np.exp(exponent)
    
    def log_likelihood(w0, w1, sigma_squared, x, y):
        n = len(x)
        residuals = y - (w1 * x + w0)
        return -n/2 * np.log(2 * np.pi * sigma_squared) - np.sum(residuals**2) / (2 * sigma_squared)
    
    print("\nFor our data with 4 treasure locations, the log-likelihood function is:")
    log_likelihood_value = log_likelihood(true_w0, true_w1, true_sigma_squared, x, y)
    print(f"ln L({true_w0}, {true_w1}, {true_sigma_squared} | data) = {log_likelihood_value:.4f}")
    
    # Create a 3D plot of the likelihood function for different w0, w1 values
    w0_range = np.linspace(0, 4, 50)
    w1_range = np.linspace(1, 5, 50)
    w0_grid, w1_grid = np.meshgrid(w0_range, w1_range)
    
    log_likelihoods = np.zeros_like(w0_grid)
    for i in range(len(w0_range)):
        for j in range(len(w1_range)):
            log_likelihoods[j, i] = log_likelihood(w0_range[i], w1_range[j], true_sigma_squared, x, y)
    
    # Convert to likelihood (unnormalized)
    likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods))
    
    # Create 3D surface plot of likelihood
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(w0_grid, w1_grid, likelihoods, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Intercept (w₀)')
    ax1.set_ylabel('Slope (w₁)')
    ax1.set_zlabel('Likelihood (unnormalized)')
    ax1.set_title('Likelihood Function L(w₀, w₁ | data)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Create contour plot of log-likelihood
    ax2 = fig.add_subplot(gs[0, 1])
    contour = ax2.contourf(w0_grid, w1_grid, log_likelihoods, 20, cmap='viridis')
    ax2.set_xlabel('Intercept (w₀)')
    ax2.set_ylabel('Slope (w₁)')
    ax2.set_title('Log-Likelihood Function ln L(w₀, w₁ | data)')
    fig.colorbar(contour, ax=ax2)
    
    # Mark the true parameter values
    ax2.plot(true_w0, true_w1, 'rx', markersize=10, label='True parameters')
    
    # Create a plot of the model with data points
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(x, y, color='blue', s=100, label='Observed treasures')
    
    # Plot the true model line
    x_line = np.linspace(0, 5, 100)
    y_line = true_w1 * x_line + true_w0
    ax3.plot(x_line, y_line, 'r-', label=f'True model: y = {true_w1}x + {true_w0}')
    
    # Add confidence bands for the true model
    y_upper = y_line + 2 * np.sqrt(true_sigma_squared)
    y_lower = y_line - 2 * np.sqrt(true_sigma_squared)
    ax3.fill_between(x_line, y_lower, y_upper, color='red', alpha=0.2, 
                    label='95% confidence band')
    
    ax3.set_xlabel('x (distance from shore)')
    ax3.set_ylabel('y (steps along coast)')
    ax3.set_title('Treasure Locations and True Model')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    likelihood_plot_path = os.path.join(save_dir, "plot1_likelihood_function.png")
    plt.savefig(likelihood_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return likelihood_plot_path

# Step 2: Calculate the maximum likelihood estimates
def step2_mle_parameters():
    """Calculate the MLE estimates for w0 and w1."""
    print("\nStep 2: Calculating the Maximum Likelihood Estimates for w₀ and w₁\n")
    
    print("For linear regression, the MLE estimates for w₀ and w₁ can be derived by")
    print("maximizing the log-likelihood function:")
    print("ln L(w₀, w₁, σ² | data) = -n/2 * ln(2πσ²) - (1/2σ²) * ∑ᵢ₌₁ⁿ (yᵢ - (w₁xᵢ + w₀))²")
    
    print("\nTaking derivatives with respect to w₀ and w₁ and setting to zero gives the normal equations:")
    print("∂ln L/∂w₀ = 0 ⟹ ∑ᵢ₌₁ⁿ (yᵢ - (w₁xᵢ + w₀)) = 0")
    print("∂ln L/∂w₁ = 0 ⟹ ∑ᵢ₌₁ⁿ xᵢ(yᵢ - (w₁xᵢ + w₀)) = 0")
    
    print("\nThese equations can be solved to yield the MLE estimates:")
    
    # Calculate means
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate MLE for w1 (slope)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    w1_mle = numerator / denominator
    
    # Calculate MLE for w0 (intercept)
    w0_mle = y_mean - w1_mle * x_mean
    
    print("\nCalculating w₁ (slope):")
    print(f"w₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
    print(f"x̄ = {x_mean}, ȳ = {y_mean}")
    print(f"Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator}")
    print(f"Σ(xᵢ - x̄)² = {denominator}")
    print(f"w₁ = {numerator} / {denominator} = {w1_mle}")
    
    print("\nCalculating w₀ (intercept):")
    print(f"w₀ = ȳ - w₁ * x̄")
    print(f"w₀ = {y_mean} - {w1_mle} * {x_mean} = {w0_mle}")
    
    print("\nThe MLE estimates are:")
    print(f"w₀ = {w0_mle}")
    print(f"w₁ = {w1_mle}")
    
    # Create a plot to show the MLE fit
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', s=100, label='Observed treasures')
    
    # Plot the true model
    x_line = np.linspace(0, 5, 100)
    y_true = true_w1 * x_line + true_w0
    plt.plot(x_line, y_true, 'r--', label=f'True model: y = {true_w1}x + {true_w0}')
    
    # Plot the MLE model
    y_mle = w1_mle * x_line + w0_mle
    plt.plot(x_line, y_mle, 'g-', linewidth=2, 
             label=f'MLE model: y = {w1_mle:.4f}x + {w0_mle:.4f}')
    
    # Add labels and title
    plt.xlabel('x (distance from shore)')
    plt.ylabel('y (steps along coast)')
    plt.title('Treasure Locations with True and MLE Models')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    mle_plot_path = os.path.join(save_dir, "plot2_mle_fit.png")
    plt.savefig(mle_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return w0_mle, w1_mle, mle_plot_path

# Step 3: Estimate the noise variance
def step3_variance_estimation(w0_mle, w1_mle):
    """Estimate the noise variance using MLE."""
    print("\nStep 3: Estimating the Noise Variance σ² using MLE\n")
    
    print("For linear regression, the MLE of the error variance σ² is given by:")
    print("σ² = (1/n) * ∑ᵢ₌₁ⁿ (yᵢ - (ŵ₁xᵢ + ŵ₀))²")
    print("where ŵ₀ and ŵ₁ are the MLE estimates for w₀ and w₁")
    
    # Calculate the residuals
    y_pred = w1_mle * x + w0_mle
    residuals = y - y_pred
    
    # Calculate the MLE for variance
    n = len(x)
    sigma_squared_mle = np.sum(residuals**2) / n
    
    print("\nCalculating residuals (prediction errors):")
    for i in range(n):
        print(f"Residual {i+1}: y_{i+1} - (ŵ₁x_{i+1} + ŵ₀) = {y[i]} - ({w1_mle} * {x[i]} + {w0_mle}) = {residuals[i]}")
    
    print(f"\nSum of squared residuals: Σ residuals² = {np.sum(residuals**2)}")
    print(f"Number of data points: n = {n}")
    print(f"MLE estimate of variance: σ² = {np.sum(residuals**2)} / {n} = {sigma_squared_mle}")
    
    # Note: The unbiased estimator (dividing by n-2 instead of n)
    sigma_squared_unbiased = np.sum(residuals**2) / (n - 2)
    print(f"\nNote: The unbiased estimator of variance is: s² = {np.sum(residuals**2)} / {n-2} = {sigma_squared_unbiased}")
    print("(This divides by n-2 instead of n to account for estimating 2 parameters: w₀ and w₁)")
    
    # Create a visualization of the residuals
    plt.figure(figsize=(10, 8))
    
    # Create a GridSpec layout
    gs = GridSpec(2, 1, height_ratios=[2, 1])
    
    # Top plot: Data and model fit
    ax1 = plt.subplot(gs[0])
    
    # Plot data points
    ax1.scatter(x, y, color='blue', s=100, label='Observed treasures')
    
    # Plot the MLE model
    x_line = np.linspace(0, 5, 100)
    y_mle = w1_mle * x_line + w0_mle
    ax1.plot(x_line, y_mle, 'g-', linewidth=2, 
             label=f'MLE model: y = {w1_mle:.4f}x + {w0_mle:.4f}')
    
    # Add residual lines
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y[i], y_pred[i]], 'r-', linewidth=1.5)
    
    # Add labels and title
    ax1.set_xlabel('x (distance from shore)')
    ax1.set_ylabel('y (steps along coast)')
    ax1.set_title('Treasure Locations with MLE Model and Residuals')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom plot: Residual distribution
    ax2 = plt.subplot(gs[1])
    
    # Create a histogram of the residuals
    bins = np.linspace(-4, 4, 20)
    ax2.hist(residuals, bins=bins, alpha=0.7, density=True, 
             label=f'Residuals\nσ²_MLE = {sigma_squared_mle:.4f}')
    
    # Add the theoretical normal distribution
    x_norm = np.linspace(-4, 4, 1000)
    y_norm = stats.norm.pdf(x_norm, 0, np.sqrt(sigma_squared_mle))
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2, 
             label=f'N(0, σ²_MLE = {sigma_squared_mle:.4f})')
    
    # Add the true normal distribution for comparison
    y_true_norm = stats.norm.pdf(x_norm, 0, np.sqrt(true_sigma_squared))
    ax2.plot(x_norm, y_true_norm, 'g--', linewidth=2, 
             label=f'True N(0, σ² = {true_sigma_squared})')
    
    # Add labels and title
    ax2.set_xlabel('Residual value')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Residuals')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    variance_plot_path = os.path.join(save_dir, "plot3_variance_estimation.png")
    plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return sigma_squared_mle, variance_plot_path

# Step 4: Calculate probability of treasure beyond y > 12 at x = 2.5
def step4_probability_calculation(w0_mle, w1_mle, sigma_squared_mle):
    """Calculate probability of treasure beyond y > 12 at x = 2.5."""
    print("\nStep 4: Calculating the Probability of Treasure Beyond y > 12 at x = 2.5\n")
    
    # New location of interest
    x_new = 2.5
    y_threshold = 12
    
    # Calculate the predicted y at x = 2.5
    y_pred = w1_mle * x_new + w0_mle
    
    print(f"For x = {x_new}, the predicted position is:")
    print(f"ŷ = ŵ₁x + ŵ₀ = {w1_mle} * {x_new} + {w0_mle} = {y_pred}")
    
    print("\nTo find P(y > 12 | x = 2.5), we need to consider the distribution of y given x:")
    print(f"y | x=2.5 ~ N(ŷ, σ²) = N({y_pred}, {sigma_squared_mle})")
    
    # Calculate the z-score
    sigma = np.sqrt(sigma_squared_mle)
    z_score = (y_threshold - y_pred) / sigma
    
    # Calculate the probability
    prob = 1 - stats.norm.cdf(z_score)
    
    print("\nCalculating the standardized z-score:")
    print(f"z = (y_threshold - ŷ) / σ = ({y_threshold} - {y_pred}) / {sigma} = {z_score}")
    
    print("\nThe probability is:")
    print(f"P(y > {y_threshold} | x = {x_new}) = 1 - Φ(z) = 1 - Φ({z_score}) = {prob}")
    print(f"There is a {prob*100:.2f}% chance the treasure is beyond y > {y_threshold}.")
    
    # Create a visualization of the probability
    plt.figure(figsize=(10, 6))
    
    # Range of y values for plotting
    y_range = np.linspace(y_pred - 4*sigma, y_pred + 4*sigma, 1000)
    
    # Calculate the normal PDF
    y_pdf = stats.norm.pdf(y_range, y_pred, sigma)
    
    # Plot the normal distribution
    plt.plot(y_range, y_pdf, 'b-', linewidth=2, 
             label=f'Distribution of y at x={x_new}')
    
    # Fill the area representing P(y > 12)
    y_fill = y_range[y_range >= y_threshold]
    y_fill_pdf = stats.norm.pdf(y_fill, y_pred, sigma)
    plt.fill_between(y_fill, 0, y_fill_pdf, color='red', alpha=0.3, 
                    label=f'P(y > {y_threshold}) = {prob:.4f}')
    
    # Add vertical lines
    plt.axvline(x=y_pred, color='green', linestyle='-', linewidth=2, 
               label=f'Predicted y = {y_pred:.4f}')
    plt.axvline(x=y_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold y = {y_threshold}')
    
    # Add text annotation for the probability
    plt.text(y_threshold + 0.5, max(y_pdf)/2, 
             f'P(y > {y_threshold}) = {prob:.4f}\n= {prob*100:.2f}%', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('y (steps along coast)')
    plt.ylabel('Probability Density')
    plt.title(f'Probability of Finding Treasure Beyond y > {y_threshold} at x = {x_new}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    probability_plot_path = os.path.join(save_dir, "plot4_probability_calculation.png")
    plt.savefig(probability_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return prob, probability_plot_path

# Step 5: Calculate the 90% prediction interval
def step5_prediction_interval(w0_mle, w1_mle, sigma_squared_mle):
    """Calculate the 90% prediction interval for where to dig at x = 2.5."""
    print("\nStep 5: Calculating the 90% Prediction Interval at x = 2.5\n")
    
    # New location of interest
    x_new = 2.5
    
    # Calculate the predicted y at x = 2.5
    y_pred = w1_mle * x_new + w0_mle
    
    # Standard deviation of the prediction
    sigma = np.sqrt(sigma_squared_mle)
    
    # Calculate the prediction interval
    n = len(x)
    x_mean = np.mean(x)
    sum_squared_diff = np.sum((x - x_mean) ** 2)
    
    # Prediction variance (for a new observation)
    # Standard error of prediction = σ * sqrt(1 + 1/n + (x_new - x_mean)²/sum_squared_diff)
    prediction_variance = sigma_squared_mle * (1 + 1/n + (x_new - x_mean)**2/sum_squared_diff)
    prediction_std = np.sqrt(prediction_variance)
    
    # For 90% prediction interval, we need the t-statistic with n-2 degrees of freedom
    # For a two-sided interval, we need the 95th percentile (because 90% in the middle means 5% in each tail)
    t_critical = stats.t.ppf(0.95, n-2)
    
    # Calculate the prediction interval
    lower_bound = y_pred - t_critical * prediction_std
    upper_bound = y_pred + t_critical * prediction_std
    
    print("A prediction interval accounts for both the uncertainty in estimating the mean")
    print("and the randomness of the new observation.")
    
    print(f"\nFor x = {x_new}, the predicted position is:")
    print(f"ŷ = ŵ₁x + ŵ₀ = {w1_mle} * {x_new} + {w0_mle} = {y_pred}")
    
    print("\nThe standard error of prediction is:")
    print(f"SE_pred = σ * √(1 + 1/n + (x_new - x̄)²/Σ(xᵢ - x̄)²)")
    print(f"= {sigma} * √(1 + 1/{n} + ({x_new} - {x_mean})²/{sum_squared_diff})")
    print(f"= {prediction_std}")
    
    print(f"\nFor a 90% prediction interval, we need the t-critical value with {n-2} degrees of freedom:")
    print(f"t_critical = {t_critical}")
    
    print("\nThe 90% prediction interval is:")
    print(f"ŷ ± t_critical * SE_pred = {y_pred} ± {t_critical} * {prediction_std}")
    print(f"= [{lower_bound}, {upper_bound}]")
    
    print(f"\nWith 90% confidence, the treasure at x = {x_new} is located")
    print(f"between y = {lower_bound} and y = {upper_bound} steps along the coast.")
    
    # Create a visualization of the prediction interval
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, color='blue', s=100, label='Observed treasures')
    
    # Plot the MLE model
    x_line = np.linspace(0, 5, 100)
    y_mle = w1_mle * x_line + w0_mle
    plt.plot(x_line, y_mle, 'g-', linewidth=2, 
             label=f'MLE model: y = {w1_mle:.4f}x + {w0_mle:.4f}')
    
    # Calculate prediction intervals for the whole range
    prediction_stds = []
    for x_val in x_line:
        pred_var = sigma_squared_mle * (1 + 1/n + (x_val - x_mean)**2/sum_squared_diff)
        prediction_stds.append(np.sqrt(pred_var))
    
    prediction_stds = np.array(prediction_stds)
    
    # Add prediction interval bands
    plt.fill_between(x_line, 
                    y_mle - t_critical * prediction_stds, 
                    y_mle + t_critical * prediction_stds, 
                    color='green', alpha=0.2, label='90% Prediction Interval')
    
    # Highlight the point at x = 2.5
    plt.scatter([x_new], [y_pred], color='red', s=150, marker='*',
               label=f'Prediction at x = {x_new}')
    
    # Add vertical lines at x = 2.5 to highlight the prediction interval
    plt.vlines(x=x_new, ymin=lower_bound, ymax=upper_bound, 
              color='red', linestyle='--', linewidth=2)
    
    # Add text annotation for the prediction interval
    plt.text(x_new + 0.1, (lower_bound + upper_bound)/2, 
             f'90% PI: [{lower_bound:.2f}, {upper_bound:.2f}]', 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('x (distance from shore)')
    plt.ylabel('y (steps along coast)')
    plt.title('Treasure Locations with MLE Model and 90% Prediction Interval')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    pi_plot_path = os.path.join(save_dir, "plot5_prediction_interval.png")
    plt.savefig(pi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create an additional plot that shows the detailed situation at x = 2.5
    plt.figure(figsize=(10, 6))
    
    # Range of y values for plotting
    y_range = np.linspace(y_pred - 4*prediction_std, y_pred + 4*prediction_std, 1000)
    
    # Calculate the t distribution PDF
    y_pdf = stats.t.pdf((y_range - y_pred)/prediction_std, n-2) / prediction_std
    
    # Plot the t distribution
    plt.plot(y_range, y_pdf, 'b-', linewidth=2, 
             label=f't-distribution with {n-2} df')
    
    # Fill the area representing 90% prediction interval
    mask_90 = (y_range >= lower_bound) & (y_range <= upper_bound)
    plt.fill_between(y_range, 0, y_pdf, where=mask_90, color='green', alpha=0.3, 
                    label='90% Prediction Interval')
    
    # Add vertical lines
    plt.axvline(x=y_pred, color='green', linestyle='-', linewidth=2, 
               label=f'Predicted y = {y_pred:.4f}')
    plt.axvline(x=lower_bound, color='red', linestyle='--', linewidth=2, 
               label=f'Lower bound = {lower_bound:.4f}')
    plt.axvline(x=upper_bound, color='red', linestyle='--', linewidth=2, 
               label=f'Upper bound = {upper_bound:.4f}')
    
    # Add text annotation for the prediction interval
    plt.text((lower_bound + upper_bound)/2, max(y_pdf)*0.8, 
             f'90% Prediction Interval\n[{lower_bound:.2f}, {upper_bound:.2f}]', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8),
             horizontalalignment='center')
    
    # Add labels and title
    plt.xlabel('y (steps along coast)')
    plt.ylabel('Probability Density')
    plt.title(f'90% Prediction Interval for Treasure at x = {x_new}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    pi_detail_plot_path = os.path.join(save_dir, "plot6_prediction_interval_detail.png")
    plt.savefig(pi_detail_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return lower_bound, upper_bound, pi_plot_path, pi_detail_plot_path

# Run all steps
likelihood_plot_path = step1_likelihood_function()
w0_mle, w1_mle, mle_plot_path = step2_mle_parameters()
sigma_squared_mle, variance_plot_path = step3_variance_estimation(w0_mle, w1_mle)
prob, probability_plot_path = step4_probability_calculation(w0_mle, w1_mle, sigma_squared_mle)
lower_bound, upper_bound, pi_plot_path, pi_detail_plot_path = step5_prediction_interval(w0_mle, w1_mle, sigma_squared_mle)

# Summary of results
print("\n\nSummary of Results:")
print("-----------------")
print(f"1. Likelihood Function: Successfully derived and visualized")
print(f"2. MLE Parameters: w₀ = {w0_mle:.4f}, w₁ = {w1_mle:.4f}")
print(f"3. MLE Noise Variance: σ² = {sigma_squared_mle:.4f}")
print(f"4. Probability of y > 12 at x = 2.5: {prob:.4f} ({prob*100:.2f}%)")
print(f"5. 90% Prediction Interval at x = 2.5: [{lower_bound:.4f}, {upper_bound:.4f}]")

print("\nAll visualizations saved to:", save_dir)
print(f"- Likelihood function plot: {os.path.basename(likelihood_plot_path)}")
print(f"- MLE fit plot: {os.path.basename(mle_plot_path)}")
print(f"- Variance estimation plot: {os.path.basename(variance_plot_path)}")
print(f"- Probability calculation plot: {os.path.basename(probability_plot_path)}")
print(f"- Prediction interval plot: {os.path.basename(pi_plot_path)}")
print(f"- Prediction interval detail plot: {os.path.basename(pi_detail_plot_path)}") 