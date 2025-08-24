import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L7_3_Quiz_22")
os.makedirs(save_dir, exist_ok=True)

# Enable LaTeX style plotting
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

print("Question 22: Random Forest Stock Performance Prediction")
print("=" * 60)

# Given data: Stock performance predictions from 12 trees
tech_stock = np.array([0.8, 0.7, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8])
energy_stock = np.array([0.4, 0.5, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4])
healthcare_stock = np.array([0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5])

stocks = {
    'Tech': tech_stock,
    'Energy': energy_stock,
    'Healthcare': healthcare_stock
}

print("\nGiven Stock Performance Predictions (12 trees):")
print(f"Tech Stock: {tech_stock}")
print(f"Energy Stock: {energy_stock}")
print(f"Healthcare Stock: {healthcare_stock}")

# Task 1: Calculate ensemble performance score and consistency for each stock
print("\n" + "="*60)
print("TASK 1: Calculate ensemble performance score and consistency for each stock")
print("="*60)

ensemble_scores = {}
consistencies = {}
variances = {}
std_deviations = {}

for stock_name, predictions in stocks.items():
    print(f"\n{stock_name} Stock Analysis:")
    print("-" * 30)
    
    # Calculate ensemble score (mean) - Step by step
    print("Step 1: Calculate Ensemble Score (Mean)")
    print(f"  Formula: μ = (Σx_i) / n")
    print(f"  Where n = {len(predictions)} (number of trees)")
    print(f"  Sum of predictions: {np.sum(predictions):.1f}")
    print(f"  Mean = {np.sum(predictions):.1f} / {len(predictions)} = {np.mean(predictions):.4f}")
    
    ensemble_score = np.mean(predictions)
    ensemble_scores[stock_name] = ensemble_score
    print(f"  Ensemble Score: {ensemble_score:.4f}")
    
    # Calculate variance - Step by step
    print("\nStep 2: Calculate Variance")
    print(f"  Formula: σ² = Σ(x_i - μ)² / (n-1)")
    print(f"  Using sample variance (n-1 denominator)")
    
    # Show individual deviations
    deviations = predictions - ensemble_score
    print(f"  Deviations from mean: {deviations}")
    print(f"  Squared deviations: {deviations**2}")
    print(f"  Sum of squared deviations: {np.sum(deviations**2):.6f}")
    
    variance = np.var(predictions, ddof=1)  # Sample variance
    print(f"  Variance = {np.sum(deviations**2):.6f} / ({len(predictions)}-1) = {variance:.6f}")
    variances[stock_name] = variance
    
    # Calculate standard deviation - Step by step
    print("\nStep 3: Calculate Standard Deviation")
    print(f"  Formula: σ = √σ²")
    std_dev = np.std(predictions, ddof=1)   # Sample standard deviation
    print(f"  Standard Deviation = √{variance:.6f} = {std_dev:.6f}")
    std_deviations[stock_name] = std_dev
    
    # Calculate coefficient of variation - Step by step
    print("\nStep 4: Calculate Coefficient of Variation")
    print(f"  Formula: CV = σ / μ")
    coefficient_of_variation = std_dev / ensemble_score if ensemble_score != 0 else float('inf')
    print(f"  Coefficient of Variation = {std_dev:.6f} / {ensemble_score:.4f} = {coefficient_of_variation:.6f}")
    
    # Calculate consistency - Step by step
    print("\nStep 5: Calculate Consistency")
    print(f"  Formula: Consistency = 1 - CV")
    consistency = 1 - coefficient_of_variation
    consistencies[stock_name] = consistency
    print(f"  Consistency = 1 - {coefficient_of_variation:.6f} = {consistency:.6f}")
    
    # Show individual tree predictions
    print(f"\nSummary:")
    print(f"  Individual Tree Predictions: {predictions}")
    print(f"  Sorted Predictions: {np.sort(predictions)}")
    print(f"  Range: {np.max(predictions) - np.min(predictions):.1f}")
    print(f"  Min: {np.min(predictions):.1f}, Max: {np.max(predictions):.1f}")

# Task 2: Choose 2 stocks to minimize risk
print("\n" + "="*60)
print("TASK 2: Choose 2 stocks to minimize risk")
print("="*60)

print("Step 1: Rank stocks by variance (lower variance = lower risk)")
print("  Risk Ranking (Lowest to Highest):")

# Sort stocks by variance (lower variance = lower risk)
risk_ranking = sorted(variances.items(), key=lambda x: x[1])
for i, (stock, var) in enumerate(risk_ranking, 1):
    print(f"  {i}. {stock}: Variance = {var:.6f}")

print(f"\nStep 2: Select lowest risk stocks")
print(f"  To minimize risk, choose: {risk_ranking[0][0]} and {risk_ranking[1][0]}")
print(f"  Both have variance = {risk_ranking[0][1]:.6f}")

# Task 3: Stock with highest potential return
print("\n" + "="*60)
print("TASK 3: Stock with highest potential return (highest ensemble score)")
print("="*60)

print("Step 1: Compare ensemble scores")
for stock_name, score in ensemble_scores.items():
    print(f"  {stock_name}: {score:.4f}")

best_return_stock = max(ensemble_scores.items(), key=lambda x: x[1])
print(f"\nStep 2: Identify highest return")
print(f"  Highest Potential Return: {best_return_stock[0]} Stock")
print(f"  Ensemble Score: {best_return_stock[1]:.4f}")

# Task 4: Risk-adjusted scoring system
print("\n" + "="*60)
print("TASK 4: Risk-adjusted scoring system: Score = Ensemble_Score × (1 - Variance)")
print("="*60)

risk_adjusted_scores = {}
print("Step 1: Apply risk-adjusted formula for each stock")
print("  Formula: Risk-Adjusted Score = Ensemble_Score × (1 - Variance)")

for stock_name in stocks.keys():
    print(f"\n  {stock_name} Stock:")
    print(f"    Ensemble Score = {ensemble_scores[stock_name]:.4f}")
    print(f"    Variance = {variances[stock_name]:.6f}")
    print(f"    Risk-Adjusted Score = {ensemble_scores[stock_name]:.4f} × (1 - {variances[stock_name]:.6f})")
    
    risk_adjusted_score = ensemble_scores[stock_name] * (1 - variances[stock_name])
    risk_adjusted_scores[stock_name] = risk_adjusted_score
    
    print(f"    Risk-Adjusted Score = {ensemble_scores[stock_name]:.4f} × {1 - variances[stock_name]:.6f}")
    print(f"    Risk-Adjusted Score = {risk_adjusted_score:.6f}")

print("\nStep 2: Rank stocks by risk-adjusted score")
risk_adj_ranking = sorted(risk_adjusted_scores.items(), key=lambda x: x[1], reverse=True)
for i, (stock, score) in enumerate(risk_adj_ranking, 1):
    print(f"  {i}. {stock}: {score:.6f}")

# Task 5: Value at Risk (VaR) at 95% confidence level
print("\n" + "="*60)
print("TASK 5: Value at Risk (VaR) at 95% confidence level")
print("="*60)

print("Step 1: Understand VaR calculation")
print("  VaR at 95% confidence = 5th percentile of predictions")
print("  This means 95% of predictions are above this value")
print("  Formula: Sort predictions and find the value at 5% position")

var_95 = {}
for stock_name, predictions in stocks.items():
    print(f"\n  {stock_name} Stock:")
    print(f"    Step 1: Sort predictions in ascending order")
    sorted_preds = np.sort(predictions)
    print(f"      Sorted: {sorted_preds}")
    
    print(f"    Step 2: Calculate 5th percentile position")
    n = len(predictions)
    position = 0.05 * n
    print(f"      Position = 0.05 × {n} = {position}")
    
    # Calculate 5th percentile
    var_95_value = np.percentile(predictions, 5)
    var_95[stock_name] = var_95_value
    
    print(f"    Step 3: Find 5th percentile value")
    print(f"      5th Percentile (VaR at 95% confidence): {var_95_value:.4f}")
    print(f"      This means 95% of predictions are above {var_95_value:.4f}")
    
    # Show which predictions are below VaR
    below_var = predictions[predictions <= var_95_value]
    print(f"      Predictions below VaR: {below_var}")

# Create comprehensive summary table
print("\n" + "="*60)
print("COMPREHENSIVE SUMMARY TABLE")
print("="*60)

summary_data = []
for stock_name in stocks.keys():
    summary_data.append({
        'Stock': stock_name,
        'Ensemble Score': f"{ensemble_scores[stock_name]:.4f}",
        'Variance': f"{variances[stock_name]:.6f}",
        'Std Dev': f"{std_deviations[stock_name]:.6f}",
        'Consistency': f"{consistencies[stock_name]:.6f}",
        'Risk-Adjusted Score': f"{risk_adjusted_scores[stock_name]:.6f}",
        'VaR (95%)': f"{var_95[stock_name]:.4f}"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Create visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 1. Bar plot of ensemble scores
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
stock_names = list(ensemble_scores.keys())
scores = list(ensemble_scores.values())
bars = plt.bar(stock_names, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Ensemble Performance Scores', fontsize=14)
plt.ylabel('Score')
plt.ylim(0, 1)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

# 2. Bar plot of variances (risk)
plt.subplot(2, 3, 2)
var_values = list(variances.values())
bars = plt.bar(stock_names, var_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Risk (Variance)', fontsize=14)
plt.ylabel('Variance')
for bar, var in zip(bars, var_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{var:.4f}', ha='center', va='bottom')

# 3. Bar plot of consistencies
plt.subplot(2, 3, 3)
cons_values = list(consistencies.values())
bars = plt.bar(stock_names, cons_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Consistency', fontsize=14)
plt.ylabel('Consistency')
plt.ylim(0, 1)
for bar, cons in zip(bars, cons_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{cons:.3f}', ha='center', va='bottom')

# 4. Risk-adjusted scores
plt.subplot(2, 3, 4)
risk_adj_values = list(risk_adjusted_scores.values())
bars = plt.bar(stock_names, risk_adj_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Risk-Adjusted Scores', fontsize=14)
plt.ylabel('Score')
for bar, score in zip(bars, risk_adj_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

# 5. VaR comparison
plt.subplot(2, 3, 5)
var_values = list(var_95.values())
bars = plt.bar(stock_names, var_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('VaR at 95% Confidence', fontsize=14)
plt.ylabel('VaR')
for bar, var_val in zip(bars, var_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{var_val:.3f}', ha='center', va='bottom')

# 6. Scatter plot: Risk vs Return
plt.subplot(2, 3, 6)
plt.scatter(var_values, scores, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1'])
for i, stock in enumerate(stock_names):
    plt.annotate(stock, (var_values[i], scores[i]), xytext=(5, 5), 
                 textcoords='offset points', fontsize=10)
plt.xlabel('Risk (Variance)')
plt.ylabel('Return (Ensemble Score)')
plt.title('Risk vs Return')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stock_analysis_overview.png'), dpi=300, bbox_inches='tight')

# Create detailed individual stock plots
plt.figure(figsize=(15, 10))

for i, (stock_name, predictions) in enumerate(stocks.items(), 1):
    plt.subplot(2, 3, i)
    
    # Histogram of predictions
    plt.hist(predictions, bins=6, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i-1])
    plt.axvline(ensemble_scores[stock_name], color='red', linestyle='--', 
                label=f'Mean: {ensemble_scores[stock_name]:.3f}')
    plt.axvline(var_95[stock_name], color='orange', linestyle=':', 
                label=f'VaR (95%): {var_95[stock_name]:.3f}')
    
    plt.title(f'{stock_name} Stock: Prediction Distribution')
    plt.xlabel('Performance Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Add summary statistics
plt.subplot(2, 3, 4)
plt.axis('off')
summary_text = "Summary Statistics:\n\n"
for stock_name in stocks.keys():
    summary_text += f"{stock_name}:\n"
    summary_text += f"  Mean: {ensemble_scores[stock_name]:.3f}\n"
    summary_text += f"  Std: {std_deviations[stock_name]:.3f}\n"
    summary_text += f"  Var: {variances[stock_name]:.4f}\n"
    summary_text += f"  VaR: {var_95[stock_name]:.3f}\n\n"

plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace')

# Risk-return scatter plot
plt.subplot(2, 3, 5)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, stock in enumerate(stock_names):
    plt.scatter(variances[stock], ensemble_scores[stock], 
               s=200, c=colors[i], label=stock, alpha=0.8)
    plt.annotate(stock, (variances[stock], ensemble_scores[stock]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=12)

plt.xlabel('Risk (Variance)')
plt.ylabel('Return (Ensemble Score)')
plt.title('Risk-Return Profile')
plt.legend()
plt.grid(True, alpha=0.3)

# Consistency vs Performance
plt.subplot(2, 3, 6)
for i, stock in enumerate(stock_names):
    plt.scatter(consistencies[stock], ensemble_scores[stock], 
               s=200, c=colors[i], label=stock, alpha=0.8)
    plt.annotate(stock, (consistencies[stock], ensemble_scores[stock]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=12)

plt.xlabel('Consistency')
plt.ylabel('Return (Ensemble Score)')
plt.title('Consistency vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detailed_stock_analysis.png'), dpi=300, bbox_inches='tight')

# Create box plots for comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
data_to_plot = [tech_stock, energy_stock, healthcare_stock]
bp = plt.boxplot(data_to_plot, labels=stock_names, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
plt.title('Prediction Distributions (Box Plots)')
plt.ylabel('Performance Score')
plt.grid(True, alpha=0.3)

# Violin plot
plt.subplot(2, 2, 2)
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Create violin plot manually since we have discrete data
for i, (stock_name, predictions) in enumerate(stocks.items()):
    x_pos = i + 1
    # Create histogram-like representation
    hist, bin_edges = np.histogram(predictions, bins=8)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_height = max(hist) if len(hist) > 0 else 1
    
    for j, (center, height) in enumerate(zip(bin_centers, hist)):
        if height > 0:
            width = height / max_height * 0.3
            rect = Rectangle((x_pos - width/2, center - 0.05), width, 0.1, 
                           facecolor=colors[i], alpha=0.7)
            plt.gca().add_patch(rect)
    
    plt.scatter([x_pos] * len(predictions), predictions, 
               c=colors[i], alpha=0.6, s=50)

plt.xticks(range(1, len(stock_names) + 1), stock_names)
plt.title('Prediction Distributions (Violin-like)')
plt.ylabel('Performance Score')
plt.grid(True, alpha=0.3)

# Performance over trees
plt.subplot(2, 2, 3)
for i, (stock_name, predictions) in enumerate(stocks.items()):
    plt.plot(range(1, 13), predictions, 'o-', label=stock_name, 
             color=colors[i], linewidth=2, markersize=6)
plt.xlabel('Tree Number')
plt.ylabel('Performance Score')
plt.title('Performance Across Trees')
plt.legend()
plt.grid(True, alpha=0.3)

# Cumulative performance
plt.subplot(2, 2, 4)
for i, (stock_name, predictions) in enumerate(stocks.items()):
    cumulative = np.cumsum(predictions)
    plt.plot(range(1, 13), cumulative, 'o-', label=stock_name, 
             color=colors[i], linewidth=2, markersize=6)
plt.xlabel('Number of Trees')
plt.ylabel('Cumulative Performance Score')
plt.title('Cumulative Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'stock_comparison_plots.png'), dpi=300, bbox_inches='tight')

print(f"\nAll visualizations saved to: {save_dir}")
print("\nAnalysis Complete!")
