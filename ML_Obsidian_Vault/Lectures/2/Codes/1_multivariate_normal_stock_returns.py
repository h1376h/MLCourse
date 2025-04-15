import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
import seaborn as sns

print("\n=== EXAMPLE 2: STOCK RETURNS MODELING ===\n")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the Lectures/2 directory
parent_dir = os.path.dirname(current_dir)
# Use Images relative to the parent directory
images_dir = os.path.join(parent_dir, "Images", "Multivariate_Normal")

# Make sure images directory exists
os.makedirs(images_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate daily returns data for three stocks
print("\nStep 1: Analyze the data")

# Let's simulate daily returns for three stocks with correlations typical of tech stocks
# We'll use realistic parameters: daily returns are typically small with standard deviations of about 1-2%

# Parameters
n_days = 500  # Number of trading days (about 2 years)
stock_means = np.array([0.0005, 0.0007, 0.0006])  # Mean daily returns (AAPL, MSFT, GOOG)
stock_stds = np.array([0.015, 0.014, 0.016])  # Standard deviations

# Correlation matrix (AAPL, MSFT, GOOG)
stock_corr = np.array([
    [1.00, 0.72, 0.63],
    [0.72, 1.00, 0.58],
    [0.63, 0.58, 1.00]
])

# Convert correlation matrix to covariance matrix
stock_cov = np.zeros_like(stock_corr)
for i in range(3):
    for j in range(3):
        stock_cov[i, j] = stock_corr[i, j] * stock_stds[i] * stock_stds[j]

# Generate the data
stock_returns = np.random.multivariate_normal(stock_means, stock_cov, n_days)

# Create a pandas DataFrame for easier handling
dates = pd.date_range(start='2021-01-01', periods=n_days, freq='B')
returns_df = pd.DataFrame(stock_returns, index=dates, columns=['AAPL', 'MSFT', 'GOOG'])

print("Generated daily returns for 3 stocks over", n_days, "trading days")
print("\nFirst 5 days of returns:")
print(returns_df.head())

print("\nSummary statistics:")
print(returns_df.describe())

print("\nTrue parameters used to generate the data:")
print(f"Mean vector = {stock_means}")
print(f"Covariance matrix = \n{stock_cov}")

# Step 2: Calculate sample statistics and fit multivariate normal distribution
print("\nStep 2: Fit a multivariate normal distribution")

# Calculate sample mean and covariance
sample_mean = returns_df.mean().values
sample_cov = returns_df.cov().values

print("\nSample mean vector:")
print(sample_mean)
print("\nSample covariance matrix:")
print(sample_cov)

# Create the multivariate normal model
mvn_model = stats.multivariate_normal(sample_mean, sample_cov)

# Step 3: Assess the fit
print("\nStep 3: Assess the fit")

# Generate Quantile-Quantile plots to check multivariate normality
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, stock in enumerate(returns_df.columns):
    # Calculate theoretical quantiles
    sorted_data = np.sort(returns_df[stock])
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n_days + 1) / (n_days + 1), 
                                         loc=sample_mean[i], 
                                         scale=np.sqrt(sample_cov[i, i]))
    
    # Plot Q-Q plot
    axes[i].scatter(theoretical_quantiles, sorted_data, alpha=0.5)
    
    # Add a reference line
    min_val = min(np.min(sorted_data), np.min(theoretical_quantiles))
    max_val = max(np.max(sorted_data), np.max(theoretical_quantiles))
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Labels and title
    axes[i].set_xlabel('Theoretical Quantiles')
    axes[i].set_ylabel('Sample Quantiles')
    axes[i].set_title(f'Q-Q Plot for {stock}')
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'stock_returns_qq_plots.png'), dpi=100)
plt.close()

# Create pairplot to visualize relationships
plt.figure(figsize=(12, 10))
sns.pairplot(returns_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 15})
plt.suptitle('Pairwise Relationships between Stock Returns', y=1.02)
plt.savefig(os.path.join(images_dir, 'stock_returns_pairplot.png'), dpi=100)
plt.close()

# Step 4: Calculate portfolio risk
print("\nStep 4: Calculate portfolio risk")

# Let's define a portfolio with equal weights
weights = np.array([1/3, 1/3, 1/3])

# Calculate portfolio expected return
portfolio_return = np.dot(weights, sample_mean)

# Calculate portfolio variance
portfolio_variance = np.dot(weights.T, np.dot(sample_cov, weights))
portfolio_std = np.sqrt(portfolio_variance)

print(f"\nPortfolio with equal weights: {weights}")
print(f"Expected daily return: {portfolio_return:.6f} ({portfolio_return*252:.4f} annualized)")
print(f"Portfolio variance: {portfolio_variance:.8f}")
print(f"Portfolio volatility (standard deviation): {portfolio_std:.4f} ({portfolio_std*np.sqrt(252):.4f} annualized)")

# Efficient frontier simulation
print("\nCalculating efficient frontier...")

# Generate different weight combinations
n_portfolios = 10000
results = np.zeros((n_portfolios, 2))  # To store return and risk

for i in range(n_portfolios):
    # Generate random weights
    w = np.random.random(3)
    w = w / np.sum(w)  # Normalize
    
    # Calculate portfolio return and variance
    r = np.dot(w, sample_mean)
    v = np.dot(w.T, np.dot(sample_cov, w))
    results[i, 0] = r * 252  # Annualized return
    results[i, 1] = np.sqrt(v) * np.sqrt(252)  # Annualized volatility

# Create plot for portfolio optimization
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(results[:, 1], results[:, 0], c=results[:, 0]/results[:, 1], 
           cmap='viridis', marker='o', s=10, alpha=0.3)

# Highlight the equal-weight portfolio
eq_weight_return = portfolio_return * 252
eq_weight_risk = portfolio_std * np.sqrt(252)
ax.scatter(eq_weight_risk, eq_weight_return, c='red', marker='*', s=200, label='Equal Weight Portfolio')

# Labels
ax.set_xlabel('Annualized Volatility (Risk)')
ax.set_ylabel('Annualized Return')
ax.set_title('Efficient Frontier with Random Portfolios')
ax.grid(alpha=0.3)
ax.legend()

# Add text annotation explaining the concept
ax.text(0.98, 0.02, 
       ("The multivariate normal distribution models the joint behavior of returns.\n"
        "Portfolios on the upper edge of this plot form the efficient frontier,\n"
        "offering the best return/risk trade-off under the model's assumptions."),
       transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig(os.path.join(images_dir, 'stock_returns_modeling.png'), dpi=100)
plt.close()

print(f"Generated visualizations for the stock returns example")

# Combine all plots into a single figure for the summary
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Correlation heatmap
ax1 = axes[0, 0]
sns.heatmap(stock_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
ax1.set_title('Stock Returns Correlation Matrix')
ax1.set_xticklabels(['AAPL', 'MSFT', 'GOOG'])
ax1.set_yticklabels(['AAPL', 'MSFT', 'GOOG'])

# Plot 2: Time series of returns
ax2 = axes[0, 1]
returns_df.cumsum().plot(ax=ax2)
ax2.set_title('Cumulative Returns')
ax2.grid(alpha=0.3)
ax2.set_ylabel('Cumulative Return')

# Plot 3: 3D scatter plot
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.scatter(returns_df['AAPL'], returns_df['MSFT'], returns_df['GOOG'], 
            alpha=0.5, c=np.arange(n_days), cmap='viridis', s=15)
ax3.set_xlabel('AAPL Returns')
ax3.set_ylabel('MSFT Returns')
ax3.set_zlabel('GOOG Returns')
ax3.set_title('3D Visualization of Joint Returns')

# Plot 4: Density plot of portfolio returns
ax4 = axes[1, 1]
portfolio_returns = np.dot(returns_df, weights)
sns.histplot(portfolio_returns, kde=True, ax=ax4)
ax4.axvline(portfolio_return, color='r', linestyle='--', label=f'Expected Return: {portfolio_return:.4f}')
ax4.set_title('Equal-Weight Portfolio Returns Distribution')
ax4.grid(alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'stock_returns_summary.png'), dpi=100)
plt.close()

print(f"Generated summary visualization for the stock returns example")

print("\nKey insights from the Stock Returns example:")
print("1. The multivariate normal distribution is a standard model for stock returns in portfolio theory.")
print("2. Correlation between stock returns is crucial for understanding diversification benefits.")
print("3. Portfolio variance depends not just on individual asset variances but also on their covariances.")
print("4. The formula for portfolio variance with weights w is: σ²ₚ = w^T Σ w")
print(f"5. In this example, the equal-weight portfolio has an expected return of {portfolio_return:.6f} with standard deviation {portfolio_std:.6f}.")
print("6. Modern Portfolio Theory uses the multivariate normal model to find optimal risk-return portfolios.")
print("7. While the normal distribution is widely used, real stock returns often show fatter tails than the normal distribution predicts.")

# Display plots if running in interactive mode
plt.close()  # Close any remaining figures

# Create a summary figure to be shown when plt.show() is called
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5, 
         "Stock Returns Modeling Example\n\n"
         "All visualizations have been saved to:\n"
         f"{images_dir}\n\n"
         "Key files:\n"
         "- stock_returns_qq_plots.png\n"
         "- stock_returns_pairplot.png\n"
         "- stock_returns_modeling.png\n"
         "- stock_returns_summary.png",
         ha='center', va='center', fontsize=12)
plt.axis('off')
plt.tight_layout()

# This will show the summary figure
plt.show() 