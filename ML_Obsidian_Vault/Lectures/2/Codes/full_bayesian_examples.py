import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def plot_full_posterior(name, prior_params, observed_data, posterior_params, prediction_params=None, save_path=None):
    """
    Plot prior, likelihood, and posterior distributions for Bayesian inference.
    Also shows predictive distribution if prediction_params provided.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top plot: Prior, Likelihood, Posterior
    ax = axes[0]
    
    # For Beta-Binomial model
    if isinstance(prior_params, tuple) and len(prior_params) == 2:
        # Generate points for plotting
        x = np.linspace(0, 1, 1000)
        
        # Prior (Beta distribution)
        prior_alpha, prior_beta = prior_params
        prior = stats.beta.pdf(x, prior_alpha, prior_beta)
        ax.plot(x, prior, 'b-', label=f'Prior Beta({prior_alpha},{prior_beta})', alpha=0.6)
        
        # Likelihood (based on observed data)
        n = len(observed_data) if isinstance(observed_data, list) else observed_data[1]
        k = sum(observed_data) if isinstance(observed_data, list) else observed_data[0]
        likelihood = stats.binom.pmf(k, n, x) * n * 10  # Scaled for visibility
        ax.plot(x, likelihood, 'g-', label=f'Likelihood: {k}/{n} successes', alpha=0.6)
        
        # Posterior (Beta distribution)
        post_alpha, post_beta = posterior_params
        posterior = stats.beta.pdf(x, post_alpha, post_beta)
        
        # Maximum a posteriori (MAP) for comparison
        map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else (post_alpha / (post_alpha + post_beta))
        
        ax.plot(x, posterior, 'r-', label=f'Posterior Beta({post_alpha:.1f},{post_beta:.1f})', alpha=0.8)
        ax.axvline(x=map_estimate, color='k', linestyle='--', label=f'MAP Estimate: {map_estimate:.3f}')
        
        # Credible interval
        ci_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
        ci_upper = stats.beta.ppf(0.975, post_alpha, post_beta)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray', label=f'95% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        # If we have prediction params for Beta-Binomial
        if prediction_params and len(prediction_params) == 3:
            post_alpha, post_beta, future_trials = prediction_params
            
            # Ensure future_trials is an integer
            future_trials = int(future_trials)
            
            # Bottom plot: Predictive Distribution
            ax = axes[1]
            
            # Generate predictive distribution using Beta-Binomial
            x_pred = np.arange(future_trials + 1)
            pred_probs = np.zeros_like(x_pred, dtype=float)
            
            # Beta-Binomial predictive distribution
            for k in range(future_trials + 1):
                pred_probs[k] = stats.betabinom.pmf(k, future_trials, post_alpha, post_beta)
            
            ax.bar(x_pred, pred_probs, alpha=0.7, color='purple')
            ax.set_xlabel(f'Number of Successes in {future_trials} Future Trials')
            ax.set_ylabel('Probability')
            ax.set_title('Predictive Distribution')
            
            # Expected value and quantiles for prediction
            expected_value = future_trials * post_alpha / (post_alpha + post_beta)
            q25 = stats.betabinom.ppf(0.25, future_trials, post_alpha, post_beta)
            q75 = stats.betabinom.ppf(0.75, future_trials, post_alpha, post_beta)
            
            ax.axvline(x=expected_value, color='r', linestyle='--', 
                      label=f'Expected: {expected_value:.1f}')
            ax.axvspan(q25, q75, alpha=0.2, color='orange', 
                      label=f'50% Prediction Interval: [{q25}, {q75}]')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
        
    # For Normal model
    elif isinstance(prior_params, tuple) and len(prior_params) == 4:
        prior_mean, prior_var, like_mean, like_var = prior_params
        
        # Generate points for plotting
        x = np.linspace(min(prior_mean, like_mean) - 3*max(np.sqrt(prior_var), np.sqrt(like_var)),
                        max(prior_mean, like_mean) + 3*max(np.sqrt(prior_var), np.sqrt(like_var)), 
                        1000)
        
        # Prior (Normal distribution)
        prior = stats.norm.pdf(x, prior_mean, np.sqrt(prior_var))
        ax.plot(x, prior, 'b-', label=f'Prior N({prior_mean:.1f}, {np.sqrt(prior_var):.1f})', alpha=0.6)
        
        # Likelihood (Normal distribution)
        likelihood = stats.norm.pdf(x, like_mean, np.sqrt(like_var))
        ax.plot(x, likelihood, 'g-', label=f'Likelihood N({like_mean:.1f}, {np.sqrt(like_var):.1f})', alpha=0.6)
        
        # Posterior (Normal distribution)
        post_var = 1 / (1/prior_var + 1/like_var)
        post_mean = post_var * (prior_mean/prior_var + like_mean/like_var)
        posterior = stats.norm.pdf(x, post_mean, np.sqrt(post_var))
        
        ax.plot(x, posterior, 'r-', label=f'Posterior N({post_mean:.1f}, {np.sqrt(post_var):.1f})', alpha=0.8)
        ax.axvline(x=post_mean, color='k', linestyle='--', label=f'MAP/Mean: {post_mean:.3f}')
        
        # Credible interval
        ci_lower = stats.norm.ppf(0.025, post_mean, np.sqrt(post_var))
        ci_upper = stats.norm.ppf(0.975, post_mean, np.sqrt(post_var))
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray', label=f'95% Credible Interval: [{ci_lower:.1f}, {ci_upper:.1f}]')
        
        # If we have prediction params for Normal model
        if prediction_params and len(prediction_params) == 3:
            post_mean, post_var, pred_var = prediction_params
            
            # Bottom plot: Predictive Distribution
            ax = axes[1]
            
            # Generate predictive distribution which is Normal with increased variance
            x_pred = np.linspace(post_mean - 3*np.sqrt(post_var + pred_var),
                              post_mean + 3*np.sqrt(post_var + pred_var),
                              1000)
            
            # For normal-normal model, predictive is also normal but with combined variance
            pred_dist = stats.norm.pdf(x_pred, post_mean, np.sqrt(post_var + pred_var))
            
            ax.plot(x_pred, pred_dist, color='purple', alpha=0.7)
            ax.set_xlabel('Predicted Value')
            ax.set_ylabel('Probability Density')
            ax.set_title('Predictive Distribution')
            
            # Prediction interval
            pi_lower = stats.norm.ppf(0.025, post_mean, np.sqrt(post_var + pred_var))
            pi_upper = stats.norm.ppf(0.975, post_mean, np.sqrt(post_var + pred_var))
            
            ax.axvline(x=post_mean, color='r', linestyle='--', 
                      label=f'Expected: {post_mean:.1f}')
            ax.axvspan(pi_lower, pi_upper, alpha=0.2, color='orange', 
                      label=f'95% Prediction Interval: [{pi_lower:.1f}, {pi_upper:.1f}]')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Set title and labels for top plot
    ax = axes[0]
    ax.set_title(f'{name}: Full Bayesian Analysis - Posterior Distribution')
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Close the figure to avoid displaying in notebooks

def beta_binomial_example(name, prior_params, observed_data, future_trials, description=None, save_path=None):
    """Run a full Beta-Binomial Bayesian analysis example"""
    print(f"\n{'='*50}")
    print(f"{name} Example:")
    print(f"{'='*50}")
    
    if description:
        print(description)
    
    # Setup parameters
    prior_alpha, prior_beta = prior_params
    
    # Process observed data
    if isinstance(observed_data, list):
        n = len(observed_data)
        k = sum(observed_data)
    else:
        k, n = observed_data
    
    print(f"\nPrior Distribution:")
    print(f"- Beta({prior_alpha}, {prior_beta})")
    if prior_alpha == prior_beta:
        print(f"- This represents a prior belief centered at p=0.5 (balanced)")
    elif prior_alpha > prior_beta:
        prior_mode = (prior_alpha - 1) / (prior_alpha + prior_beta - 2) if prior_alpha > 1 and prior_beta > 1 else None
        if prior_mode:
            print(f"- This represents a prior belief favoring success (mode at p={prior_mode:.2f})")
        else:
            print(f"- This represents a prior belief favoring success")
    else:
        prior_mode = (prior_alpha - 1) / (prior_alpha + prior_beta - 2) if prior_alpha > 1 and prior_beta > 1 else None
        if prior_mode:
            print(f"- This represents a prior belief favoring failure (mode at p={prior_mode:.2f})")
        else:
            print(f"- This represents a prior belief favoring failure")
    
    print(f"\nObserved Data:")
    print(f"- {k} successes out of {n} trials ({k/n:.1%} success rate)")
    
    # Calculate posterior parameters
    post_alpha = prior_alpha + k
    post_beta = prior_beta + (n - k)
    
    print(f"\nPosterior Distribution:")
    print(f"- Beta({post_alpha}, {post_beta})")
    
    # Posterior statistics
    post_mean = post_alpha / (post_alpha + post_beta)
    post_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))
    post_mode = (post_alpha - 1) / (post_alpha + post_beta - 2) if post_alpha > 1 and post_beta > 1 else None
    
    print(f"- Posterior mean: {post_mean:.4f}")
    if post_mode:
        print(f"- Posterior mode (MAP estimate): {post_mode:.4f}")
    else:
        print(f"- Posterior mode is at boundary (0 or 1)")
    print(f"- Posterior standard deviation: {np.sqrt(post_var):.4f}")
    
    # Credible interval
    ci_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
    ci_upper = stats.beta.ppf(0.975, post_alpha, post_beta)
    print(f"- 95% credible interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Probability statements
    threshold = 0.5  # Example threshold
    prob_above = 1 - stats.beta.cdf(threshold, post_alpha, post_beta)
    print(f"\nProbability Statements:")
    print(f"- Probability that true parameter > {threshold}: {prob_above:.2%}")
    
    # Predictive distribution
    print(f"\nPredictive Distribution for {future_trials} future trials:")
    ev = future_trials * post_mean
    var = future_trials * post_mean * (1-post_mean) * (post_alpha + post_beta + future_trials) / (post_alpha + post_beta + 1)
    
    print(f"- Expected number of successes: {ev:.2f}")
    print(f"- Predictive standard deviation: {np.sqrt(var):.2f}")
    
    # High probability scenario
    high_threshold = int(future_trials * 0.7)  # 70% success rate in future trials
    prob_high_success = 1 - stats.betabinom.cdf(high_threshold-1, future_trials, post_alpha, post_beta)
    print(f"- Probability of at least {high_threshold} successes in {future_trials} trials: {prob_high_success:.2%}")
    
    # Create visualization
    posterior_params = (post_alpha, post_beta)
    prediction_params = (post_alpha, post_beta, future_trials)
    
    plot_full_posterior(name, prior_params, observed_data, posterior_params, prediction_params, save_path)
    
    return post_mean, post_mode, (ci_lower, ci_upper)

def normal_normal_example(name, prior_params, observed_data, pred_var=None, description=None, save_path=None):
    """Run a full Normal-Normal Bayesian analysis example"""
    print(f"\n{'='*50}")
    print(f"{name} Example:")
    print(f"{'='*50}")
    
    if description:
        print(description)
    
    # Setup parameters
    prior_mean, prior_var = prior_params
    
    # Process observed data
    if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray):
        data = np.array(observed_data)
        data_mean = np.mean(data)
        data_var = np.var(data)  # using MLE variance
        n = len(data)
        like_var = data_var / n  # variance of the sample mean
    else:
        data_mean, like_var, n = observed_data  # If already summarized
    
    print(f"\nPrior Distribution:")
    print(f"- Normal({prior_mean:.2f}, {np.sqrt(prior_var):.2f})")
    
    print(f"\nObserved Data:")
    if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray):
        print(f"- Data points: {data.tolist()}")
        print(f"- Sample mean: {data_mean:.2f}")
        print(f"- Sample standard deviation: {np.sqrt(data_var):.2f}")
    else:
        print(f"- Sample mean: {data_mean:.2f} (based on {n} observations)")
    
    # Calculate posterior parameters
    post_precision = 1/prior_var + n/data_var if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray) else 1/prior_var + 1/like_var
    post_var = 1 / post_precision
    post_mean = post_var * (prior_mean/prior_var + n*data_mean/data_var if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray) else prior_mean/prior_var + data_mean/like_var)
    
    print(f"\nPosterior Distribution:")
    print(f"- Normal({post_mean:.2f}, {np.sqrt(post_var):.2f})")
    
    # Credible interval
    ci_lower = stats.norm.ppf(0.025, post_mean, np.sqrt(post_var))
    ci_upper = stats.norm.ppf(0.975, post_mean, np.sqrt(post_var))
    print(f"- 95% credible interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Probability statements
    threshold = prior_mean  # Example threshold using prior mean
    prob_above = 1 - stats.norm.cdf(threshold, post_mean, np.sqrt(post_var))
    print(f"\nProbability Statements:")
    print(f"- Probability that true parameter > {threshold:.2f}: {prob_above:.2%}")
    
    # Predictive distribution if prediction variance provided
    if pred_var is not None:
        # For normal model, the predictive variance is posterior variance + new observation variance
        pred_total_var = post_var + pred_var
        
        print(f"\nPredictive Distribution for future observations:")
        print(f"- Predictive mean: {post_mean:.2f}")
        print(f"- Predictive standard deviation: {np.sqrt(pred_total_var):.2f}")
        
        # Prediction interval
        pi_lower = stats.norm.ppf(0.025, post_mean, np.sqrt(pred_total_var))
        pi_upper = stats.norm.ppf(0.975, post_mean, np.sqrt(pred_total_var))
        print(f"- 95% prediction interval: [{pi_lower:.2f}, {pi_upper:.2f}]")
        
        # Create visualization
        if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray):
            vis_prior_params = (prior_mean, prior_var, data_mean, data_var/n)
        else:
            vis_prior_params = (prior_mean, prior_var, data_mean, like_var)
        
        posterior_params = (post_mean, post_var)
        prediction_params = (post_mean, post_var, pred_var)
        
        plot_full_posterior(name, vis_prior_params, observed_data, posterior_params, prediction_params, save_path)
    else:
        # Create visualization without predictive distribution
        if isinstance(observed_data, list) or isinstance(observed_data, np.ndarray):
            vis_prior_params = (prior_mean, prior_var, data_mean, data_var/n)
        else:
            vis_prior_params = (prior_mean, prior_var, data_mean, like_var)
        
        posterior_params = (post_mean, post_var)
        
        plot_full_posterior(name, vis_prior_params, observed_data, posterior_params, None, save_path)
    
    return post_mean, (ci_lower, ci_upper)

def generate_examples(save_dir=None):
    results = {}
    
    # Create save directory if provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Example 1: Basketball Shot Success (Beta-Binomial)
    basketball_desc = """
    Basketball Free Throw Analysis
    - Historical data suggests a shooting accuracy of around 70%
    - New session: Made 8 out of 10 free throws (80%)
    - We'll perform full Bayesian analysis to:
        1. Update our belief about true shooting percentage
        2. Quantify uncertainty with credible intervals
        3. Make predictions about future performance
    """
    
    save_path = os.path.join(save_dir, "full_bayesian_basketball.png") if save_dir else None
    basketball_results = beta_binomial_example(
        name="Basketball Free Throw",
        prior_params=(7, 3),  # Prior: Beta(7,3) centered at 0.7
        observed_data=(8, 10),  # 8 successes out of 10 shots
        future_trials=10,  # Predict next 10 shots
        description=basketball_desc,
        save_path=save_path
    )
    results["Basketball"] = {"results": basketball_results, "path": save_path}
    
    # Example 2: Video Game A/B Testing (Beta-Binomial)
    game_desc = """
    Video Game A/B Testing Analysis
    - Testing two versions of a game feature
    - Version A (control): 30 plays with 15 conversions (50%)
    - Version B (new): 20 plays with 14 conversions (70%)
    - We'll perform full Bayesian analysis to:
        1. Determine probability that B is better than A
        2. Quantify uncertainty for each version
        3. Make predictions about future conversion rates
    """
    
    # First analyze version A
    save_path_A = os.path.join(save_dir, "full_bayesian_game_A.png") if save_dir else None
    game_A_results = beta_binomial_example(
        name="Video Game A/B Test - Version A",
        prior_params=(1, 1),  # Uninformative prior: Beta(1,1)
        observed_data=(15, 30),  # 15 conversions out of 30 plays
        future_trials=100,  # Predict rate for next 100 plays
        description="Version A (control) analysis:",
        save_path=save_path_A
    )
    
    # Then analyze version B
    save_path_B = os.path.join(save_dir, "full_bayesian_game_B.png") if save_dir else None
    game_B_results = beta_binomial_example(
        name="Video Game A/B Test - Version B",
        prior_params=(1, 1),  # Uninformative prior: Beta(1,1)
        observed_data=(14, 20),  # 14 conversions out of 20 plays
        future_trials=100,  # Predict rate for next 100 plays
        description="Version B (new) analysis:",
        save_path=save_path_B
    )
    
    # Compare the two versions
    print("\nA/B Test Comparison:")
    # Simulate samples from both posteriors to calculate P(B > A)
    np.random.seed(42)  # For reproducibility
    samples_A = np.random.beta(1 + 15, 1 + 30 - 15, 10000)  # Posterior for A: Beta(16,16)
    samples_B = np.random.beta(1 + 14, 1 + 20 - 14, 10000)  # Posterior for B: Beta(15,7)
    prob_B_better = np.mean(samples_B > samples_A)
    print(f"- Probability that Version B is better than Version A: {prob_B_better:.2%}")
    
    # Expected improvement
    improvement = np.mean(samples_B - samples_A)
    print(f"- Expected improvement: {improvement:.2%}")
    
    # Calculate 95% credible interval for the difference
    diff_samples = samples_B - samples_A
    ci_diff_lower = np.percentile(diff_samples, 2.5)
    ci_diff_upper = np.percentile(diff_samples, 97.5)
    print(f"- 95% credible interval for difference: [{ci_diff_lower:.2%}, {ci_diff_upper:.2%}]")
    
    results["Video Game A/B Test"] = {
        "results_A": game_A_results, 
        "results_B": game_B_results,
        "prob_B_better": prob_B_better,
        "path_A": save_path_A,
        "path_B": save_path_B
    }
    
    # Example 3: Test Score Prediction (Normal-Normal)
    test_desc = """
    Test Score Prediction Analysis
    - Historical average on similar tests: 85 points
    - Recent performance: scores of 92, 88, and 90
    - We'll perform full Bayesian analysis to:
        1. Update our belief about student's true ability
        2. Quantify uncertainty with credible intervals
        3. Predict performance on future tests
    """
    
    # Note: Normal-Normal conjugate prior assumes known variance
    # For simplicity, we're using the sample variance of the data
    test_scores = [92, 88, 90]
    test_var = np.var(test_scores)  # Sample variance
    
    save_path = os.path.join(save_dir, "full_bayesian_test_scores.png") if save_dir else None
    test_results = normal_normal_example(
        name="Test Score Prediction",
        prior_params=(85, 25),  # Prior: Normal(85, 5²)
        observed_data=test_scores,
        pred_var=test_var,  # For predictive distribution
        description=test_desc,
        save_path=save_path
    )
    results["Test Scores"] = {"results": test_results, "path": save_path}
    
    return results

if __name__ == "__main__":
    # Use relative path for save directory
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level and then to Images
    save_dir = os.path.join(os.path.dirname(script_dir), "Images")
    results = generate_examples(save_dir)
    print("\nExamples completed and saved to:", save_dir) 