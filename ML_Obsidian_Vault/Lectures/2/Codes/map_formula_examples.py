import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta
import os

def normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq):
    """
    Calculate the MAP estimate for a normal distribution with known variance.
    
    Parameters:
    -----------
    mu0 : float
        Prior mean
    sigma0_sq : float
        Prior variance
    new_data : array-like
        Observed data points
    sigma_sq : float
        Variance of the data
        
    Returns:
    --------
    map_estimate : float
        MAP estimate of the mean
    """
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    return numerator / denominator

def visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, title="MAP Estimation Example", save_path=None):
    """
    Visualize the prior, likelihood, and posterior distributions for normal MAP estimation.
    
    Parameters:
    -----------
    mu0 : float
        Prior mean
    sigma0_sq : float
        Prior variance
    new_data : array-like
        Observed data points
    sigma_sq : float
        Variance of the data
    title : str
        Plot title
    save_path : str, optional
        Path to save the visualization
    """
    # Calculate MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    
    # Calculate posterior parameters analytically (for comparison)
    N = len(new_data)
    posterior_var = 1 / (1/sigma0_sq + N/sigma_sq)
    posterior_mean = posterior_var * (mu0/sigma0_sq + sum(new_data)/sigma_sq)
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    x = np.linspace(min(min(new_data), mu0) - 3*np.sqrt(max(sigma0_sq, sigma_sq)),
                   max(max(new_data), mu0) + 3*np.sqrt(max(sigma0_sq, sigma_sq)), 
                   1000)
    
    # Plot prior distribution
    prior = norm.pdf(x, mu0, np.sqrt(sigma0_sq))
    plt.plot(x, prior, 'b-', label=f'Prior (μ₀={mu0:.1f})', alpha=0.6)
    
    # Plot likelihood (based on data)
    data_mean = np.mean(new_data)
    likelihood = norm.pdf(x, data_mean, np.sqrt(sigma_sq/N))  # Using SEM for likelihood
    plt.plot(x, likelihood, 'g-', label=f'Likelihood (mean={data_mean:.1f})', alpha=0.6)
    
    # Plot posterior distribution
    posterior = norm.pdf(x, posterior_mean, np.sqrt(posterior_var))
    plt.plot(x, posterior, 'r-', label=f'Posterior', alpha=0.6)
    
    # Add MAP estimate
    plt.axvline(x=map_estimate, color='k', linestyle='--', label=f'MAP={map_estimate:.2f}')
    
    # Add data points
    plt.plot(new_data, np.zeros_like(new_data), 'go', label='Data points')
    
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return map_estimate

def example_student_height(save_dir):
    """Example: Estimating average student height in a class."""
    # Prior knowledge: Average height of students is around 170 cm with variance 25
    mu0 = 170  # cm
    sigma0_sq = 25  # cm²
    
    # Observed data: Heights of 5 randomly selected students
    new_data = [165, 173, 168, 180, 172]  # cm
    
    # Known variance in the population
    sigma_sq = 20  # cm²
    
    # Print problem setup
    print("=" * 50)
    print("EXAMPLE: STUDENT HEIGHT ESTIMATION")
    print("=" * 50)
    print(f"Prior belief: Average student height is {mu0} cm (variance: {sigma0_sq} cm²)")
    print(f"Observed data: Heights of 5 students: {new_data}")
    print(f"Known population variance: {sigma_sq} cm²")
    
    # Calculate and print MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    data_mean = np.mean(new_data)
    
    # Step-by-step calculation for educational purposes
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    print("\nStep-by-step MAP calculation:")
    print(f"1. Calculate variance ratio: σ₀²/σ² = {sigma0_sq}/{sigma_sq} = {ratio:.4f}")
    print(f"2. Calculate numerator: μ₀ + (σ₀²/σ²)∑x = {mu0} + {ratio:.4f} × {sum(new_data)} = {numerator:.4f}")
    print(f"3. Calculate denominator: 1 + (σ₀²/σ²)N = 1 + {ratio:.4f} × {N} = {denominator:.4f}")
    print(f"4. Final MAP estimate: {numerator:.4f}/{denominator:.4f} = {map_estimate:.2f} cm")
    
    print(f"\nComparison:")
    print(f"- MAP estimate: {map_estimate:.2f} cm")
    print(f"- Sample mean: {data_mean:.2f} cm")
    print(f"- Prior mean: {mu0:.2f} cm")
    
    # Visualize the result
    save_path = os.path.join(save_dir, "student_height_map.png")
    visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, 
                        title="Student Height Estimation Using MAP",
                        save_path=save_path)
    
    return map_estimate

def example_online_learning_score(save_dir):
    """Example: Estimating a student's true skill level in an online learning platform."""
    # Prior knowledge: Average score for this difficulty level is 70/100 with variance 100
    mu0 = 70  # score out of 100
    sigma0_sq = 100  # variance in prior belief
    
    # Observed data: Student's recent quiz scores
    new_data = [85, 82, 90, 88]  # recent scores
    
    # Known variance in quiz scores due to question variation
    sigma_sq = 64  # variance in quiz scores
    
    # Print problem setup
    print("\n" + "=" * 50)
    print("EXAMPLE: ONLINE LEARNING SCORE ESTIMATION")
    print("=" * 50)
    print(f"Prior belief: Average score at this difficulty is {mu0}/100 (variance: {sigma0_sq})")
    print(f"Observed data: Student's recent scores: {new_data}")
    print(f"Known score variance: {sigma_sq}")
    
    # Calculate and print MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    data_mean = np.mean(new_data)
    
    # Step-by-step calculation for educational purposes
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    print("\nStep-by-step MAP calculation:")
    print(f"1. Calculate variance ratio: σ₀²/σ² = {sigma0_sq}/{sigma_sq} = {ratio:.4f}")
    print(f"2. Calculate numerator: μ₀ + (σ₀²/σ²)∑x = {mu0} + {ratio:.4f} × {sum(new_data)} = {numerator:.4f}")
    print(f"3. Calculate denominator: 1 + (σ₀²/σ²)N = 1 + {ratio:.4f} × {N} = {denominator:.4f}")
    print(f"4. Final MAP estimate: {numerator:.4f}/{denominator:.4f} = {map_estimate:.2f}")
    
    print(f"\nComparison:")
    print(f"- MAP estimate: {map_estimate:.2f}")
    print(f"- Sample mean: {data_mean:.2f}")
    print(f"- Prior mean: {mu0:.2f}")
    
    # Interpretation
    print("\nInterpretation:")
    if map_estimate > mu0 + 10:
        print("The student is significantly above average at this difficulty level.")
    elif map_estimate > mu0:
        print("The student is above average at this difficulty level.")
    else:
        print("The student is performing at or below the average for this difficulty level.")
    
    # Visualize the result
    save_path = os.path.join(save_dir, "online_learning_map.png")
    visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, 
                        title="Online Learning Score Estimation Using MAP",
                        save_path=save_path)
    
    return map_estimate

def example_manufacturing_process(save_dir):
    """Example: Estimating true component dimension in a manufacturing process."""
    # Prior knowledge: Design specification is 50 mm with variance 0.04
    mu0 = 50  # mm
    sigma0_sq = 0.04  # mm²
    
    # Observed data: Measurements of produced components
    new_data = [50.2, 50.3, 50.1, 50.25, 50.15]  # mm
    
    # Known measurement error variance
    sigma_sq = 0.01  # mm²
    
    # Print problem setup
    print("\n" + "=" * 50)
    print("EXAMPLE: MANUFACTURING PROCESS QUALITY CONTROL")
    print("=" * 50)
    print(f"Prior belief: Component should be {mu0} mm (variance: {sigma0_sq} mm²)")
    print(f"Observed data: Measured dimensions: {new_data}")
    print(f"Known measurement error variance: {sigma_sq} mm²")
    
    # Calculate and print MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    data_mean = np.mean(new_data)
    
    # Step-by-step calculation for educational purposes
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    print("\nStep-by-step MAP calculation:")
    print(f"1. Calculate variance ratio: σ₀²/σ² = {sigma0_sq}/{sigma_sq} = {ratio:.4f}")
    print(f"2. Calculate numerator: μ₀ + (σ₀²/σ²)∑x = {mu0} + {ratio:.4f} × {sum(new_data)} = {numerator:.4f}")
    print(f"3. Calculate denominator: 1 + (σ₀²/σ²)N = 1 + {ratio:.4f} × {N} = {denominator:.4f}")
    print(f"4. Final MAP estimate: {numerator:.4f}/{denominator:.4f} = {map_estimate:.3f} mm")
    
    print(f"\nComparison:")
    print(f"- MAP estimate: {map_estimate:.3f} mm")
    print(f"- Sample mean: {data_mean:.3f} mm")
    print(f"- Design specification: {mu0:.3f} mm")
    
    # Quality control assessment
    tolerance = 0.3  # mm
    print("\nQuality Control Assessment:")
    if abs(map_estimate - mu0) > tolerance:
        print(f"WARNING: True component dimension likely outside tolerance ({tolerance} mm).")
        print(f"Process adjustment recommended.")
    else:
        print(f"Process appears within tolerance ({tolerance} mm).")
    
    # Visualize the result
    save_path = os.path.join(save_dir, "manufacturing_map.png")
    visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, 
                        title="Manufacturing Process Quality Control Using MAP",
                        save_path=save_path)
    
    return map_estimate

def example_sensor_measurement(save_dir):
    """Example: Estimating the true temperature using a sensor with known error characteristics."""
    # Prior knowledge: Expected temperature based on weather forecast and building conditions
    mu0 = 25  # °C (expected temperature)
    sigma0_sq = 4  # °C² (uncertainty in our prior belief)
    
    # Observed data: Readings from the temperature sensor
    new_data = [23, 24, 26]  # °C (sensor readings)
    
    # Known sensor error variance
    sigma_sq = 1  # °C² (known sensor accuracy)
    
    # Print problem setup
    print("\n" + "=" * 50)
    print("EXAMPLE: SENSOR TEMPERATURE MEASUREMENT")
    print("=" * 50)
    print(f"Prior belief: Expected temperature is {mu0}°C (variance: {sigma0_sq}°C²)")
    print(f"Observed data: Sensor readings: {new_data}")
    print(f"Known sensor error variance: {sigma_sq}°C²")
    
    # Calculate relevant statistics
    N = len(new_data)
    data_mean = np.mean(new_data)
    data_sum = sum(new_data)
    ratio = sigma0_sq / sigma_sq
    
    # Calculate MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    
    # Calculate MLE estimate (for comparison)
    mle_estimate = data_mean
    
    # Step-by-step calculation for educational purposes
    numerator = mu0 + ratio * data_sum
    denominator = 1 + ratio * N
    
    print("\nStep-by-step MAP calculation:")
    print(f"1. Calculate sample mean: {data_sum}/{N} = {data_mean:.2f}°C")
    print(f"2. Calculate variance ratio: σ₀²/σ² = {sigma0_sq}/{sigma_sq} = {ratio:.2f}")
    print(f"3. Calculate numerator: μ₀ + (σ₀²/σ²)∑x = {mu0} + {ratio:.2f} × {data_sum} = {numerator:.2f}")
    print(f"4. Calculate denominator: 1 + (σ₀²/σ²)N = 1 + {ratio:.2f} × {N} = {denominator:.2f}")
    print(f"5. Final MAP estimate: {numerator:.2f}/{denominator:.2f} = {map_estimate:.2f}°C")
    
    print(f"\nComparison:")
    print(f"- Prior belief (expected temperature): {mu0:.2f}°C")
    print(f"- Sample mean (MLE estimate): {data_mean:.2f}°C")
    print(f"- MAP estimate: {map_estimate:.2f}°C")
    
    # Analysis of result
    print("\nAnalysis:")
    if abs(map_estimate - mu0) < 0.2:
        print("The MAP estimate is very close to our prior expectation.")
    elif abs(map_estimate - data_mean) < 0.2:
        print("The MAP estimate is very close to our observed readings.")
    else:
        diff_from_prior = abs(map_estimate - mu0)
        diff_from_data = abs(map_estimate - data_mean)
        if diff_from_prior < diff_from_data:
            print(f"The MAP estimate ({map_estimate:.2f}°C) is closer to our prior expectation ({mu0:.2f}°C) than to the sample mean ({data_mean:.2f}°C).")
        else:
            print(f"The MAP estimate ({map_estimate:.2f}°C) is closer to the sample mean ({data_mean:.2f}°C) than to our prior expectation ({mu0:.2f}°C).")
    
    # Temperature control application
    comfort_lower = 22
    comfort_upper = 26
    print("\nTemperature Control Application:")
    if map_estimate < comfort_lower:
        print(f"The true temperature is likely below the comfort zone ({comfort_lower}°C). Consider increasing heating.")
    elif map_estimate > comfort_upper:
        print(f"The true temperature is likely above the comfort zone ({comfort_upper}°C). Consider cooling.")
    else:
        print(f"The true temperature is likely within the comfort zone ({comfort_lower}-{comfort_upper}°C). No action needed.")
    
    # Create enhanced visualization
    save_path = os.path.join(save_dir, "sensor_map.png")
    
    # Use the standard visualization function
    visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, 
                        title="Sensor Temperature Measurement Using MAP",
                        save_path=save_path)
    
    # Create additional visualization showing both MLE and MAP estimates
    plt.figure(figsize=(10, 6))
    x = np.linspace(min(min(new_data), mu0) - 3, max(max(new_data), mu0) + 3, 1000)
    
    # Plot prior distribution
    prior = norm.pdf(x, mu0, np.sqrt(sigma0_sq))
    plt.plot(x, prior, 'b-', label=f'Prior (μ₀={mu0}°C)', alpha=0.6)
    
    # Plot likelihood (based on data)
    likelihood = norm.pdf(x, data_mean, np.sqrt(sigma_sq/N))
    plt.plot(x, likelihood, 'g-', label=f'Likelihood (mean={data_mean}°C)', alpha=0.6)
    
    # Calculate posterior parameters analytically
    posterior_var = 1 / (1/sigma0_sq + N/sigma_sq)
    posterior_mean = posterior_var * (mu0/sigma0_sq + data_sum/sigma_sq)
    
    # Plot posterior distribution
    posterior = norm.pdf(x, posterior_mean, np.sqrt(posterior_var))
    plt.plot(x, posterior, 'r-', label=f'Posterior', alpha=0.6)
    
    # Add MAP estimate
    plt.axvline(x=map_estimate, color='k', linestyle='--', label=f'MAP={map_estimate:.2f}°C')
    
    # Add MLE estimate
    plt.axvline(x=mle_estimate, color='g', linestyle='--', label=f'MLE={mle_estimate:.2f}°C')
    
    # Add data points
    plt.plot(new_data, np.zeros_like(new_data), 'go', label='Sensor readings')
    
    # Add comfort zone
    plt.axvspan(comfort_lower, comfort_upper, alpha=0.2, color='gray', label=f'Comfort zone ({comfort_lower}-{comfort_upper}°C)')
    
    plt.title("Sensor Temperature Measurement: Prior, Likelihood, and Posterior")
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the enhanced visualization
    enhanced_save_path = os.path.join(save_dir, "sensor_map_enhanced.png")
    plt.savefig(enhanced_save_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced figure saved to {enhanced_save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return map_estimate

def example_stock_prediction(save_dir):
    """Example: Estimating true stock returns based on historical and recent data."""
    # Prior knowledge: Historical average return of a stock
    mu0 = 5.0  # % return (historical average)
    sigma0_sq = 4  # variance in prior belief
    
    # Observed data: Recent daily returns
    new_data = [8.2, 7.5, 9.1, 7.8, 8.4, 7.9]  # recent returns in %
    
    # Known variance in daily returns
    sigma_sq = 10  # variance in daily returns
    
    # Print problem setup
    print("\n" + "=" * 50)
    print("EXAMPLE: STOCK RETURN PREDICTION")
    print("=" * 50)
    print(f"Prior belief: Historical average return is {mu0}% (variance: {sigma0_sq})")
    print(f"Observed data: Recent daily returns: {new_data}%")
    print(f"Known return variance: {sigma_sq}")
    
    # Calculate and print MAP estimate
    map_estimate = normal_map_estimate(mu0, sigma0_sq, new_data, sigma_sq)
    data_mean = np.mean(new_data)
    
    # Step-by-step calculation for educational purposes
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    
    print("\nStep-by-step MAP calculation:")
    print(f"1. Calculate variance ratio: σ₀²/σ² = {sigma0_sq}/{sigma_sq} = {ratio:.4f}")
    print(f"2. Calculate numerator: μ₀ + (σ₀²/σ²)∑x = {mu0} + {ratio:.4f} × {sum(new_data)} = {numerator:.4f}")
    print(f"3. Calculate denominator: 1 + (σ₀²/σ²)N = 1 + {ratio:.4f} × {N} = {denominator:.4f}")
    print(f"4. Final MAP estimate: {numerator:.4f}/{denominator:.4f} = {map_estimate:.2f}%")
    
    print(f"\nComparison:")
    print(f"- MAP estimate: {map_estimate:.2f}%")
    print(f"- Sample mean (recent average): {data_mean:.2f}%")
    print(f"- Prior mean (historical average): {mu0:.2f}%")
    
    # Analysis of the result
    print("\nAnalysis:")
    if map_estimate > mu0:
        print(f"The MAP estimate ({map_estimate:.2f}%) is higher than the historical average ({mu0:.2f}%),")
        print(f"suggesting recent performance is better than the historical trend.")
    else:
        print(f"The MAP estimate ({map_estimate:.2f}%) is lower than the historical average ({mu0:.2f}%),")
        print(f"suggesting recent performance is worse than the historical trend.")
    
    print(f"\nSince the variance ratio ({ratio:.4f}) is less than 1, we trust our prior more than the new data.")
    print(f"This means our MAP estimate is closer to the historical average than to the recent average.")
    
    # Visualize the result
    save_path = os.path.join(save_dir, "stock_returns_map.png")
    visualize_normal_map(mu0, sigma0_sq, new_data, sigma_sq, 
                        title="Stock Return Prediction Using MAP",
                        save_path=save_path)
    
    return map_estimate

def run_all_examples(save_dir):
    """Run all MAP estimation examples."""
    example_student_height(save_dir)
    example_online_learning_score(save_dir)
    example_manufacturing_process(save_dir)
    example_sensor_measurement(save_dir)
    example_stock_prediction(save_dir)  # New example with ratio < 1

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    # Make sure images directory exists
    os.makedirs(save_dir, exist_ok=True)
    run_all_examples(save_dir) 