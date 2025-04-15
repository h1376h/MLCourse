import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def plot_distributions(name, mu0, sigma0_sq, new_data, sigma_sq, map_estimate, save_path=None):
    plt.figure(figsize=(12, 6))
    
    # Generate points for plotting
    x = np.linspace(min(min(new_data), mu0) - 3*np.sqrt(max(sigma0_sq, sigma_sq)),
                    max(max(new_data), mu0) + 3*np.sqrt(max(sigma0_sq, sigma_sq)), 
                    1000)
    
    # Plot prior distribution (historical)
    prior = norm.pdf(x, mu0, np.sqrt(sigma0_sq))
    plt.plot(x, prior, 'b-', label=f'Prior (μ₀={mu0:.1f})', alpha=0.6)
    
    # Plot likelihood (new data)
    likelihood = norm.pdf(x, np.mean(new_data), np.sqrt(sigma_sq))
    plt.plot(x, likelihood, 'g-', label=f'Likelihood (μ={np.mean(new_data):.1f})', alpha=0.6)
    
    # Calculate and plot posterior distribution
    N = len(new_data)
    posterior_var = 1 / (1/sigma0_sq + N/sigma_sq)
    posterior_mean = posterior_var * (mu0/sigma0_sq + sum(new_data)/sigma_sq)
    posterior = norm.pdf(x, posterior_mean, np.sqrt(posterior_var))
    plt.plot(x, posterior, 'r-', label=f'Posterior (MAP={map_estimate:.1f})', alpha=0.6)
    
    # Plot MAP estimate as vertical line
    plt.axvline(x=map_estimate, color='k', linestyle='--', 
                label=f'MAP Estimate ({map_estimate:.1f})')
    
    # Add data points
    plt.plot(new_data, np.zeros_like(new_data), 'g^', label='New Data Points')
    
    plt.title(f'{name}: Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks
    
    return posterior_mean, posterior_var

def calculate_map(mu0, sigma0_sq, sigma_sq, new_data):
    N = len(new_data)
    ratio = sigma0_sq / sigma_sq
    
    numerator = mu0 + ratio * sum(new_data)
    denominator = 1 + ratio * N
    map_estimate = numerator / denominator
    
    # Also calculate simple average for comparison
    simple_average = np.mean(new_data)
    
    return map_estimate, simple_average

def print_analysis(name, mu0, new_data, historical_data=None, save_dir=None):
    print(f"\n{'='*50}")
    print(f"{name} Example:")
    print(f"{'='*50}")
    
    scenarios = {
        "Basketball": "Let's analyze your basketball shooting percentage. We'll look at your historical performance and today's shots.",
        "Video Game": "Let's analyze your gaming performance. We'll compare your historical scores with recent games.",
        "Test Scores": "Let's analyze your test performance. We'll look at your previous semester and recent tests.",
        "Daily Steps": "Let's analyze your daily activity. We'll compare your typical steps with recent days."
    }
    print(scenarios.get(name, ""))
    
    # Calculate historical statistics
    if historical_data is None:
        # If no historical data provided, use simplified variance
        historical_data = [mu0] * 30  # Assume 30 past observations
    
    # Calculate variances dynamically
    sigma0_sq = np.var(historical_data)
    sigma_sq = np.var(new_data)
    
    print(f"\nHistorical Data Analysis:")
    print(f"- Historical data points: {historical_data}")
    print(f"- Historical mean (μ0): {mu0:.2f}")
    print(f"- Historical variance (σ0²): {sigma0_sq:.2f}")
    print(f"  (This represents how spread out your historical performance was)")
    
    print(f"\nNew Data Analysis:")
    print(f"- New data points: {new_data}")
    print(f"- New data variance (σ²): {sigma_sq:.2f}")
    print(f"  (This represents how consistent your recent performance is)")
    
    # Calculate intermediate values
    ratio = sigma0_sq/sigma_sq
    data_sum = sum(new_data)
    data_mean = np.mean(new_data)
    N = len(new_data)
    
    print(f"\nDetailed MAP Calculation Steps:")
    print(f"1. Variance Ratio (σ0²/σ²):")
    print(f"   {sigma0_sq:.2f} / {sigma_sq:.2f} = {ratio:.4f}")
    print(f"   - If > 1: We trust new data more")
    print(f"   - If < 1: We trust historical data more")
    
    print(f"\n2. New Data Summary:")
    print(f"   - Number of new observations (N): {N}")
    print(f"   - Sum of new values: {data_sum:.2f}")
    print(f"   - Average of new values: {data_mean:.2f}")
    
    numerator = mu0 + ratio * data_sum
    denominator = 1 + ratio * N
    map_result = numerator / denominator
    
    print(f"\n3. MAP Formula Components:")
    print(f"   Numerator = μ0 + (σ0²/σ²)∑x")
    print(f"   = {mu0:.2f} + {ratio:.4f} × {data_sum:.2f}")
    print(f"   = {numerator:.2f}")
    
    print(f"\n4. Denominator = 1 + (σ0²/σ²)N")
    print(f"   = 1 + {ratio:.4f} × {N}")
    print(f"   = {denominator:.2f}")
    
    print(f"\n5. Final MAP Estimate:")
    print(f"   = {numerator:.2f} / {denominator:.2f}")
    print(f"   = {map_result:.2f}")
    
    # Create save path if directory is provided
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"normal_map_{name.lower().replace(' ', '_')}.png"
        save_path = os.path.join(save_dir, filename)
    
    # Add visualization
    plot_distributions(name, mu0, sigma0_sq, new_data, sigma_sq, map_result, save_path)
    
    print(f"\nResults Comparison:")
    print(f"- MAP Estimate: {map_result:.2f}")
    print(f"- Simple Average of new data: {data_mean:.2f}")
    print(f"- Historical mean (μ0): {mu0:.2f}")
    
    print(f"\nInterpretation:")
    if abs(map_result - mu0) < abs(map_result - data_mean):
        print("The MAP estimate is closer to your historical average.")
        print("This suggests we should trust your past performance more.")
    else:
        print("The MAP estimate is closer to your recent performance.")
        print("This suggests your recent data is more reliable.")
    
    return map_result, save_path

def generate_examples(save_dir=None):
    results = {}
    
    # Example 1: Basketball Shot Accuracy
    basketball_history = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 3  # 70% historical accuracy
    basketball_new = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # Recent performance: made 8 out of 10 shots
    print("""
    Basketball Shooting Analysis
    - Historical data represents 30 shots over past month: 21 makes (70% accuracy)
    - New data is from today's practice: 8 makes in 10 attempts (80% accuracy)
    - We'll see if this improvement is significant or just random variation
    """)
    map_result, path = print_analysis("Basketball", mu0=0.70, new_data=basketball_new, 
                              historical_data=basketball_history, save_dir=save_dir)
    results["Basketball"] = {"map": map_result, "path": path}

    # Example 2: Video Game Score
    game_history = [820, 870, 840, 860, 880, 830, 850, 840, 860, 850]  # Consistent scores around 850
    game_new = [950, 900, 1000]  # Recent breakthrough in performance
    print("""
    Video Game Performance Analysis
    - Historical data shows 10 games with scores clustering around 850
    - Recent 3 games show significant improvement (950, 900, 1000)
    - Let's see if this improvement indicates real skill increase
    """)
    map_result, path = print_analysis("Video Game", mu0=850, new_data=game_new, 
                              historical_data=game_history, save_dir=save_dir)
    results["Video Game"] = {"map": map_result, "path": path}

    # Example 3: Test Scores
    test_history = [83, 86, 84, 87, 85, 83, 86, 85, 84, 87]  # Previous semester scores
    test_new = [92, 88, 90]  # Recent test scores after new study method
    print("""
    Math Test Score Analysis
    - Historical data: 10 tests from previous semester (average ~85%)
    - New data: 3 recent tests after adopting a new study method
    - We'll analyze if the new study method is making a real difference
    """)
    map_result, path = print_analysis("Test Scores", mu0=85, new_data=test_new, 
                              historical_data=test_history, save_dir=save_dir)
    results["Test Scores"] = {"map": map_result, "path": path}

    # Example 4: Daily Steps
    steps_history = [7800, 8200, 7900, 8100, 8000, 7900, 8100, 8000, 8100, 7900]  # Regular routine
    steps_new = [10000, 9500, 9800]  # After joining sports team
    print("""
    Daily Steps Analysis
    - Historical data: 10 days of regular routine (averaging 8000 steps)
    - New data: 3 days after joining the school sports team
    - Let's see if this activity increase is sustainable
    """)
    map_result, path = print_analysis("Daily Steps", mu0=8000, new_data=steps_new, 
                              historical_data=steps_history, save_dir=save_dir)
    results["Daily Steps"] = {"map": map_result, "path": path}
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    generate_examples(save_dir) 