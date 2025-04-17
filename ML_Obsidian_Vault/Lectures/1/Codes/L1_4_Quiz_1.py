import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime, timedelta

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L1_4_Quiz_1")
os.makedirs(save_dir, exist_ok=True)

def print_step_header(step_number, step_title):
    """Print a formatted step header."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}: {step_title}")
    print(f"{'=' * 80}\n")

# Step 1: Stock Price Prediction Analysis
print_step_header(1, "Stock Price Prediction Analysis")

# Generate synthetic stock price data
np.random.seed(42)
n_days = 100
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
base_price = 100
daily_returns = np.random.normal(0.001, 0.02, n_days)  # Mean return of 0.1%, std of 2%
prices = base_price * np.cumprod(1 + daily_returns)

# Create DataFrame
stock_data = pd.DataFrame({
    'Date': dates,
    'Price': prices,
    'Previous_Price': np.roll(prices, 1)
})
stock_data = stock_data.iloc[1:]  # Remove first row with NaN

# Plot the stock prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Date'], stock_data['Price'], label='Stock Price')
plt.title('Synthetic Stock Price Data', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "stock_prices.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 2: Medical Image Classification Analysis
print_step_header(2, "Medical Image Classification Analysis")

# Generate synthetic medical image data
n_samples = 1000
features = np.random.normal(0, 1, (n_samples, 2))
labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive cases

# Plot the synthetic data
plt.figure(figsize=(10, 6))
plt.scatter(features[labels == 0, 0], features[labels == 0, 1], 
            color='blue', label='No Tumor', alpha=0.5)
plt.scatter(features[labels == 1, 0], features[labels == 1, 1], 
            color='red', label='Tumor', alpha=0.5)
plt.title('Synthetic Medical Image Data Distribution', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "medical_data.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 3: Poetry Understanding Analysis
print_step_header(3, "Poetry Understanding Analysis")

# Generate synthetic poetry data
poems = [
    "The sun sets in the west",
    "A bird flies in the sky",
    "The river flows gently",
    "Stars twinkle at night"
]

# Plot word frequency
words = ' '.join(poems).lower().split()
unique_words = set(words)
word_counts = {word: words.count(word) for word in unique_words}

plt.figure(figsize=(10, 6))
plt.bar(word_counts.keys(), word_counts.values())
plt.title('Word Frequency in Sample Poems', fontsize=14)
plt.xlabel('Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "poetry_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# Step 4: Customer Churn Prediction Analysis
print_step_header(4, "Customer Churn Prediction Analysis")

# Generate synthetic customer data
n_customers = 1000
customer_data = pd.DataFrame({
    'Age': np.random.normal(35, 10, n_customers),
    'Usage_Hours': np.random.normal(20, 5, n_customers),
    'Support_Calls': np.random.poisson(2, n_customers),
    'Churn': np.random.binomial(1, 0.2, n_customers)  # 20% churn rate
})

# Plot the relationships
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Age vs Churn
axes[0].scatter(customer_data['Age'], customer_data['Churn'], alpha=0.5)
axes[0].set_title('Age vs Churn', fontsize=12)
axes[0].set_xlabel('Age', fontsize=10)
axes[0].set_ylabel('Churn', fontsize=10)

# Usage Hours vs Churn
axes[1].scatter(customer_data['Usage_Hours'], customer_data['Churn'], alpha=0.5)
axes[1].set_title('Usage Hours vs Churn', fontsize=12)
axes[1].set_xlabel('Usage Hours', fontsize=10)
axes[1].set_ylabel('Churn', fontsize=10)

# Support Calls vs Churn
axes[2].scatter(customer_data['Support_Calls'], customer_data['Churn'], alpha=0.5)
axes[2].set_title('Support Calls vs Churn', fontsize=12)
axes[2].set_xlabel('Support Calls', fontsize=10)
axes[2].set_ylabel('Churn', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "customer_churn.png"), dpi=300, bbox_inches='tight')
plt.close()

# Print analysis results
print("\nAnalysis Results:")
print("1. Stock Price Prediction:")
print("   - Current approach: Using only previous day's price")
print("   - Issues: Missing important factors like market trends, news, etc.")
print("   - Suggested improvement: Include multiple features (technical indicators, news sentiment)")

print("\n2. Medical Image Classification:")
print("   - Current approach: Binary classification with labeled images")
print("   - Strengths: Clear task, labeled data, measurable performance")
print("   - Well-posed: Yes, meets all criteria for a well-posed problem")

print("\n3. Poetry Understanding:")
print("   - Current approach: Attempting to understand meaning")
print("   - Issues: Subjective nature, no clear performance measure")
print("   - Suggested improvement: Focus on specific tasks (sentiment analysis, theme detection)")

print("\n4. Customer Churn Prediction:")
print("   - Current approach: Multiple features for prediction")
print("   - Strengths: Clear features, measurable outcome, practical application")
print("   - Well-posed: Yes, meets all criteria for a well-posed problem") 