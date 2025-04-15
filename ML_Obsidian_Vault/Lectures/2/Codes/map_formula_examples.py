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
    
    plt.show()
    
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

def run_all_examples(save_dir):
    """Run all MAP estimation examples."""
    example_student_height(save_dir)
    example_online_learning_score(save_dir)
    example_manufacturing_process(save_dir)

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    # Make sure images directory exists
    os.makedirs(save_dir, exist_ok=True)
    run_all_examples(save_dir) 