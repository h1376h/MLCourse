import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_2")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Step 1: Identify features that might cause multicollinearity
def identify_multicollinearity():
    """Identify features that might cause multicollinearity and explain why."""
    print("Step 1: Identifying features that might cause multicollinearity")
    print("\nFeatures that might cause multicollinearity:")
    print("1. x1: Size of the house (in square meters) and x3: Size of the house (in square feet)")
    print("   These features are directly related through a constant conversion factor.")
    print("   Since 1 square meter = 10.764 square feet, we have x3 ≈ 10.764 * x1.")
    print("   This creates a perfect linear relationship between the two variables.")
    print("\n2. x2: Number of bedrooms and x4: Number of bathrooms")
    print("   These features might be moderately correlated, as larger houses tend to have more")
    print("   of both bedrooms and bathrooms. While not a perfect linear relationship,")
    print("   this could still contribute to multicollinearity.")
    print()
    
    # Create a simulated dataset to illustrate multicollinearity
    np.random.seed(42)
    n = 100  # number of samples
    
    # Generate house sizes in square meters
    x1 = np.random.uniform(50, 300, n)
    
    # Convert to square feet (adding some minor measurement noise)
    x3 = x1 * 10.764 + np.random.normal(0, 0.5, n)
    
    # Generate number of bedrooms based on house size with some randomness
    x2 = np.clip(np.round(x1 / 50 + np.random.normal(0, 0.5, n)), 1, 10)
    
    # Generate number of bathrooms correlated with bedrooms
    x4 = np.clip(np.round(x2 * 0.7 + np.random.normal(0, 0.3, n)), 1, 7)
    
    # Generate year built (not directly correlated with others)
    x5 = np.random.randint(1950, 2023, n)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Size_sqm': x1,
        'Bedrooms': x2,
        'Size_sqft': x3,
        'Bathrooms': x4,
        'Year_built': x5
    })
    
    print("Simulated housing dataset (first few rows):")
    print(data.head())
    print()
    
    # Calculate and display correlation matrix
    corr_matrix = data.corr()
    print("Correlation matrix:")
    print(corr_matrix)
    print()
    
    # Visualize the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Matrix of Housing Features')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # Create scatter plot of square meters vs square feet
    plt.figure(figsize=(7, 5))
    plt.scatter(x1, x3, alpha=0.6, edgecolor='k')
    plt.plot([min(x1), max(x1)], [min(x1) * 10.764, max(x1) * 10.764], 'r--', label='Perfect Conversion')
    plt.xlabel('Size (square meters)')
    plt.ylabel('Size (square feet)')
    plt.title('Relationship Between Square Meters and Square Feet')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sqm_vs_sqft.png"), dpi=300)
    plt.close()
    
    # Create scatter plot of bedrooms vs bathrooms
    plt.figure(figsize=(7, 5))
    # Add jitter to better visualize discrete data points
    jitter_x = np.random.normal(0, 0.1, n)
    jitter_y = np.random.normal(0, 0.1, n)
    plt.scatter(x2 + jitter_x, x4 + jitter_y, alpha=0.6, edgecolor='k')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Number of Bathrooms')
    plt.title('Relationship Between Bedrooms and Bathrooms')
    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bedrooms_vs_bathrooms.png"), dpi=300)
    plt.close()
    
    # Create a pairplot for all features to visualize relationships
    plt.figure(figsize=(10, 8))
    sns.pairplot(data, diag_kind='kde')
    plt.suptitle('Pairwise Relationships Between Features', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_pairplot.png"), dpi=300)
    plt.close()
    
    return data

data = identify_multicollinearity()

# Step 2: Describe methods to detect multicollinearity
def detect_multicollinearity(data):
    """Describe and demonstrate methods to detect multicollinearity."""
    print("Step 2: Methods to detect multicollinearity")
    
    print("\nMethod 1: Correlation Matrix")
    print("- Examine the correlation coefficients between predictor variables.")
    print("- High correlation coefficients (close to 1 or -1) indicate potential multicollinearity.")
    print("- Simple and intuitive but may not detect multicollinearity involving more than two variables.")
    print("- Example: The correlation coefficient between Size_sqm and Size_sqft is very high, indicating multicollinearity.")
    
    print("\nMethod 2: Variance Inflation Factor (VIF)")
    print("- VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity.")
    print("- VIF = 1/(1-R²), where R² is from regressing one predictor on all other predictors.")
    print("- VIF > 10 is often considered indicative of problematic multicollinearity.")
    print("- VIF = 1 means no multicollinearity; higher values indicate increasing multicollinearity.")
    
    # Let's calculate VIF for each feature
    
    # Prepare the features
    X = data.values
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    print("\nVIF values:")
    print(vif_data)
    print("\nInterpretation:")
    print("- Very high VIF for Size_sqm and Size_sqft confirms strong multicollinearity.")
    print("- Moderate VIF for Bedrooms and Bathrooms suggests some multicollinearity.")
    print("- Low VIF for Year_built indicates it's not strongly correlated with other predictors.")
    
    # Create VIF visualization - simplified horizontal bar chart
    plt.figure(figsize=(9, 5))
    colors = ['skyblue' if vif < 10 else 'red' for vif in vif_data["VIF"]]
    
    # Check if we need log scale due to very high VIFs
    if max(vif_data["VIF"]) > 100:
        plt.barh(vif_data["Feature"], np.log10(vif_data["VIF"]), color=colors)
        plt.axvline(x=1, color='r', linestyle='--', alpha=0.7, label='Log10(VIF) = 1 (VIF = 10)')
        plt.xlabel('Log10(VIF) - Log Scale')
        plt.title('Log-Scaled Variance Inflation Factors')
    else:
        plt.barh(vif_data["Feature"], vif_data["VIF"], color=colors)
        plt.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='VIF = 10 threshold')
        plt.xlabel('VIF Value')
        plt.title('Variance Inflation Factors')
    
    plt.ylabel('Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vif_values.png"), dpi=300)
    plt.close()
    
    print("\nMethod 3: Eigenvalue Analysis/Condition Number")
    print("- Analyze eigenvalues of the correlation matrix of predictors.")
    print("- The condition number is the ratio of the largest to smallest eigenvalue.")
    print("- High condition numbers (e.g., > 30) indicate multicollinearity.")
    print("- Can detect multicollinearity involving multiple variables.")
    
    # Calculate eigenvalues of the correlation matrix
    corr_matrix = data.corr()
    eigenvalues = np.linalg.eigvals(corr_matrix)
    condition_number = max(eigenvalues) / min(eigenvalues)
    
    print("\nEigenvalues of the correlation matrix:")
    print(eigenvalues)
    print(f"Condition number: {condition_number:.2f}")
    print("Interpretation: A high condition number confirms the presence of multicollinearity.")
    
    # Create eigenvalue visualization
    plt.figure(figsize=(8, 5))
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    plt.bar(range(1, len(eigenvalues) + 1), sorted_eigenvalues, color='lightgreen')
    plt.xlabel('Principal Component Number')
    plt.ylabel('Eigenvalue')
    plt.title(f'Eigenvalues of Correlation Matrix (Condition Number: {condition_number:.1f})')
    plt.xticks(range(1, len(eigenvalues) + 1))
    
    # Add a line showing the "elbow" or scree plot concept
    plt.plot(range(1, len(eigenvalues) + 1), sorted_eigenvalues, 'ro-')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eigenvalues.png"), dpi=300)
    plt.close()
    
    # Create a visualization showing how multicollinearity affects coefficient stability
    # Let's use a simple simulation
    n_simulations = 20
    beta_sqm_values = []
    beta_sqft_values = []
    
    # Target variable (create a synthetic house price)
    y = data['Size_sqm'] * 5000 + data['Bedrooms'] * 25000 + np.random.normal(0, 50000, len(data))
    
    for i in range(n_simulations):
        # Add some random noise to the data
        noise = np.random.normal(0, 0.1, len(data))
        data_noise = data.copy()
        data_noise['Size_sqm'] = data['Size_sqm'] + noise
        
        # Model with both square meters and square feet
        X_multi = data_noise[['Size_sqm', 'Size_sqft']].values
        model_multi = LinearRegression().fit(X_multi, y)
        
        beta_sqm_values.append(model_multi.coef_[0])
        beta_sqft_values.append(model_multi.coef_[1])
    
    # Visualize the coefficient stability
    plt.figure(figsize=(8, 6))
    plt.scatter(beta_sqm_values, beta_sqft_values, c=range(n_simulations), cmap='viridis', s=100, alpha=0.7)
    
    plt.xlabel('Coefficient for Size_sqm')
    plt.ylabel('Coefficient for Size_sqft')
    plt.title('Coefficient Instability Due to Multicollinearity')
    plt.colorbar(label='Simulation Number')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_instability.png"), dpi=300)
    plt.close()
    
    return vif_data, eigenvalues, condition_number

vif_data, eigenvalues, condition_number = detect_multicollinearity(data)

# Step 3: Propose approaches to address multicollinearity
def address_multicollinearity(data):
    """Propose and demonstrate approaches to address multicollinearity."""
    print("\nStep 3: Approaches to address multicollinearity")
    
    print("\nApproach 1: Feature Selection/Elimination")
    print("- Remove one or more of the correlated features.")
    print("- Keep the feature that is more theoretically relevant or easier to interpret.")
    print("- Example: Remove Size_sqft and keep Size_sqm (or vice versa).")
    print("- Simple solution that directly eliminates the multicollinearity problem.")
    
    # Demonstrate feature elimination
    data_reduced = data.drop('Size_sqft', axis=1)
    
    print("\nReduced dataset after removing Size_sqft:")
    print(data_reduced.head())
    
    # Calculate new correlation matrix
    corr_matrix_reduced = data_reduced.corr()
    print("\nNew correlation matrix after feature elimination:")
    print(corr_matrix_reduced)
    
    # Visualize the new correlation matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix_reduced, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Matrix After Removing Size_sqft')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_after_elimination.png"), dpi=300)
    plt.close()
    
    # Calculate new VIF values
    X_reduced = data_reduced.values
    vif_reduced = pd.DataFrame()
    vif_reduced["Feature"] = data_reduced.columns
    vif_reduced["VIF"] = [variance_inflation_factor(X_reduced, i) for i in range(X_reduced.shape[1])]
    
    print("\nVIF values after feature elimination:")
    print(vif_reduced)
    
    # Visualize the new VIF values in a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(vif_reduced["Feature"], vif_reduced["VIF"], color='skyblue')
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='VIF = 10 threshold')
    plt.xticks(rotation=45)
    plt.ylabel('VIF Value')
    plt.title('VIF Values After Feature Elimination')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vif_after_elimination.png"), dpi=300)
    plt.close()
    
    # Create a before/after VIF comparison
    plt.figure(figsize=(10, 6))
    
    # Create a dataframe for comparison
    vif_comparison = pd.DataFrame({
        'Feature': vif_reduced["Feature"],
        'Before': [vif_data.loc[vif_data['Feature'] == feat, 'VIF'].values[0] 
                  if feat in vif_data['Feature'].values else 0 
                  for feat in vif_reduced["Feature"]],
        'After': vif_reduced["VIF"]
    })
    
    # Use log scale if VIF values are very high
    if vif_comparison['Before'].max() > 100:
        ax = vif_comparison.plot(x='Feature', y=['Before', 'After'], kind='bar', logy=True, figsize=(10, 6))
        plt.ylabel('VIF (log scale)')
    else:
        ax = vif_comparison.plot(x='Feature', y=['Before', 'After'], kind='bar', figsize=(10, 6))
        plt.ylabel('VIF Value')
    
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.7, label='VIF = 10 threshold')
    plt.title('VIF Values Before and After Feature Elimination')
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vif_comparison.png"), dpi=300)
    plt.close()
    
    print("\nApproach 2: Feature Transformation/Regularization")
    print("- Principal Component Analysis (PCA): Transform correlated features into uncorrelated components.")
    print("- Ridge Regression: Use L2 regularization to reduce coefficient magnitudes without eliminating features.")
    print("- Create new composite features: Combine correlated features into a single new feature.")
    print("- Example: Use only the squared meters and create a bedroom-to-bathroom ratio feature.")
    
    # Demonstrate feature transformation - create a new feature
    data_transformed = data_reduced.copy()
    data_transformed['Bedroom_to_Bathroom_Ratio'] = data['Bedrooms'] / data['Bathrooms']
    data_transformed = data_transformed.drop(['Bedrooms', 'Bathrooms'], axis=1)
    
    print("\nTransformed dataset with new composite feature:")
    print(data_transformed.head())
    
    # Visualize the transformed dataset correlation
    corr_transformed = data_transformed.corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_transformed, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Matrix After Feature Transformation')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_after_transformation.png"), dpi=300)
    plt.close()
    
    # Ridge Regression Demonstration
    from sklearn.linear_model import Ridge
    
    # Simulating a target variable (house price) for demonstration
    np.random.seed(42)
    house_price = 100000 + 2000 * data['Size_sqm'] + 10000 * data['Bedrooms'] + \
                 5000 * data['Bathrooms'] + 100 * (data['Year_built'] - 1950) + \
                 np.random.normal(0, 50000, len(data))
    
    # Prepare the data
    X_orig = data.drop(['Size_sqm', 'Size_sqft'], axis=1)  # For simplicity, using only discrete features
    X_orig = np.column_stack([X_orig, data['Size_sqm']])  # Add back size_sqm
    
    # Standardize for regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    
    # Compare OLS and Ridge
    ols = LinearRegression()
    ridge = Ridge(alpha=10.0)  # alpha is the regularization strength
    
    ols.fit(X_scaled, house_price)
    ridge.fit(X_scaled, house_price)
    
    # Plot coefficient comparison
    feature_names = list(data.drop(['Size_sqm', 'Size_sqft'], axis=1).columns) + ['Size_sqm']
    
    plt.figure(figsize=(9, 5))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, ols.coef_, width, label='OLS Coefficients')
    plt.bar(x + width/2, ridge.coef_, width, label='Ridge Coefficients')
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value (Scaled)')
    plt.title('Comparison of OLS vs Ridge Regression Coefficients')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ridge_vs_ols.png"), dpi=300)
    plt.close()
    
    # Create a coefficient stability comparison - OLS vs Ridge
    n_simulations = 20
    ols_coefs = []
    ridge_coefs = []
    
    for i in range(n_simulations):
        # Add some random noise to the data
        noise = np.random.normal(0, 0.1, len(data))
        X_noise = X_scaled.copy() 
        X_noise[:, -1] += noise  # Add noise to Size_sqm
        
        # Fit models
        ols_model = LinearRegression().fit(X_noise, house_price)
        ridge_model = Ridge(alpha=10.0).fit(X_noise, house_price)
        
        ols_coefs.append(ols_model.coef_)
        ridge_coefs.append(ridge_model.coef_)
    
    # Convert to arrays for easier plotting
    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)
    
    # Plot coefficient stability comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate coefficient of variation for each feature (std/mean)
    ols_cv = np.std(ols_coefs, axis=0) / np.abs(np.mean(ols_coefs, axis=0))
    ridge_cv = np.std(ridge_coefs, axis=0) / np.abs(np.mean(ridge_coefs, axis=0))
    
    # Create a bar chart comparing stability
    plt.bar(x - width/2, ols_cv, width, label='OLS Coefficient Variation')
    plt.bar(x + width/2, ridge_cv, width, label='Ridge Coefficient Variation')
    
    plt.xlabel('Features')
    plt.ylabel('Coefficient of Variation (std/|mean|)')
    plt.title('OLS vs Ridge Coefficient Stability')
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_stability_comparison.png"), dpi=300)
    plt.close()
    
    print("\nComparison of OLS and Ridge coefficients:")
    coef_comparison = pd.DataFrame({
        'Feature': feature_names,
        'OLS_Coef': ols.coef_,
        'Ridge_Coef': ridge.coef_
    })
    print(coef_comparison)
    print("\nRidge regression reduces the magnitude of coefficients, which helps address multicollinearity.")
    
    return data_reduced, data_transformed, coef_comparison

data_reduced, data_transformed, coef_comparison = address_multicollinearity(data)

# Step 4: Explain effects of ignoring multicollinearity
def explain_effects_of_ignoring():
    """Explain what would happen if multicollinearity is ignored."""
    print("\nStep 4: Effects of ignoring multicollinearity")
    
    effects = [
        "1. Inflated Standard Errors: The standard errors of the coefficient estimates become larger, "
        + "making it harder to detect significant relationships.",
        
        "2. Unstable Coefficients: The coefficient estimates become highly sensitive to small changes "
        + "in the data or model specification.",
        
        "3. Incorrect Sign of Coefficients: Coefficients may have the wrong sign (opposite of what "
        + "theory or common sense would suggest).",
        
        "4. Difficulty in Determining Individual Feature Importance: It becomes hard to isolate the "
        + "effect of each predictor on the response variable.",
        
        "5. Reduced Statistical Power: The increased standard errors lead to wider confidence intervals "
        + "and reduced ability to reject null hypotheses."
    ]
    
    for effect in effects:
        print(effect)
    
    # Demonstrate unstable coefficients with a simulation
    print("\nDemonstration of coefficient instability due to multicollinearity:")
    
    # Generate multiple samples with small variations
    np.random.seed(42)
    n_samples = 50
    n_obs = 100
    coefs_with_multicollinearity = []
    coefs_without_multicollinearity = []
    
    for i in range(n_samples):
        # Create a new sample with small variations
        x1 = np.random.uniform(50, 300, n_obs)
        x3 = x1 * 10.764 + np.random.normal(0, 1, n_obs)  # Almost perfect collinearity
        
        # Target with noise
        y = 200 + 100 * x1 + 5 * x3 + np.random.normal(0, 5000, n_obs)
        
        # Model with multicollinearity
        X_multi = np.column_stack([np.ones(n_obs), x1, x3])
        try:
            # Using np.linalg.lstsq for numerical stability
            coefs_multi, _, _, _ = np.linalg.lstsq(X_multi, y, rcond=None)
            coefs_with_multicollinearity.append(coefs_multi)
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            continue
        
        # Model without multicollinearity
        X_single = np.column_stack([np.ones(n_obs), x1])
        coefs_single, _, _, _ = np.linalg.lstsq(X_single, y, rcond=None)
        coefs_without_multicollinearity.append(coefs_single[:2])  # Only take intercept and x1 coef
    
    coefs_with_multicollinearity = np.array(coefs_with_multicollinearity)
    coefs_without_multicollinearity = np.array(coefs_without_multicollinearity)
    
    # Calculate coefficient variance
    var_with_multi = np.var(coefs_with_multicollinearity, axis=0)
    var_without_multi = np.var(coefs_without_multicollinearity, axis=0)
    
    print(f"Variance of coefficients with multicollinearity: {var_with_multi}")
    print(f"Variance of coefficients without multicollinearity: {var_without_multi[:2]}")
    print(f"Ratio of variances (with/without) for intercept: {var_with_multi[0]/var_without_multi[0]:.2f}")
    print(f"Ratio of variances (with/without) for x1: {var_with_multi[1]/var_without_multi[1]:.2f}")
    
    # Visualize coefficient stability - for interpretability use boxplots
    plt.figure(figsize=(12, 6))
    
    # Create a dataframe for easier plotting
    df_coefs = pd.DataFrame()
    
    # For models with multicollinearity (add both coefficients)
    df_multi_x1 = pd.DataFrame({
        'Coefficient Value': coefs_with_multicollinearity[:, 1],
        'Variable': 'Size_sqm (with multicollinearity)',
        'Group': 'With Multicollinearity'
    })
    
    df_multi_x3 = pd.DataFrame({
        'Coefficient Value': coefs_with_multicollinearity[:, 2],
        'Variable': 'Size_sqft (with multicollinearity)',
        'Group': 'With Multicollinearity'
    })
    
    # For models without multicollinearity (only x1 coefficient)
    df_single_x1 = pd.DataFrame({
        'Coefficient Value': coefs_without_multicollinearity[:, 1],
        'Variable': 'Size_sqm (without multicollinearity)',
        'Group': 'Without Multicollinearity'
    })
    
    # Combine all data
    df_coefs = pd.concat([df_multi_x1, df_multi_x3, df_single_x1])
    
    # Plot using seaborn boxplot
    sns.boxplot(x='Variable', y='Coefficient Value', data=df_coefs)
    plt.title('Coefficient Stability Comparison')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_stability_boxplot.png"), dpi=300)
    plt.close()
    
    # Create a visual to demonstrate the incorrect sign effect
    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.hist(coefs_with_multicollinearity[:, 1], bins=20, alpha=0.7, color='blue', 
             label='Size_sqm coefficient')
    plt.hist(coefs_with_multicollinearity[:, 2], bins=20, alpha=0.7, color='red', 
             label='Size_sqft coefficient')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title('Coefficients with Multicollinearity')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(coefs_without_multicollinearity[:, 1], bins=20, alpha=0.7, color='green',
             label='Size_sqm coefficient (no multicollinearity)')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title('Coefficients without Multicollinearity')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_sign_problem.png"), dpi=300)
    plt.close()
    
    # Create a visualization for the standard error effect
    plt.figure(figsize=(10, 6))
    
    # Create synthetic confidence intervals
    true_coef = 100  # True coefficient value
    x_range = np.linspace(0, 10, 11)
    
    # Standard errors for different scenarios
    se_nomulti = 10
    se_multi = 40
    
    # Calculate confidence intervals
    y_nomulti_upper = true_coef + 1.96 * se_nomulti
    y_nomulti_lower = true_coef - 1.96 * se_nomulti
    
    y_multi_upper = true_coef + 1.96 * se_multi
    y_multi_lower = true_coef - 1.96 * se_multi
    
    # Create plots
    plt.plot([0, 10], [true_coef, true_coef], 'k-', label='True coefficient value')
    
    # Confidence intervals without multicollinearity
    plt.fill_between(x_range, y_nomulti_lower, y_nomulti_upper, 
                    color='green', alpha=0.3, label='95% CI without multicollinearity')
    
    # Confidence intervals with multicollinearity
    plt.fill_between(x_range, y_multi_lower, y_multi_upper, 
                    color='red', alpha=0.3, label='95% CI with multicollinearity')
    
    plt.xlabel('Hypothetical Studies')
    plt.ylabel('Coefficient Value')
    plt.title('Effect of Multicollinearity on Confidence Intervals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_interval_effect.png"), dpi=300)
    plt.close()

explain_effects_of_ignoring()

# Summary of the solution
print("\nQuestion 2 Solution Summary:")
print("1. Multicollinearity exists between size in square meters and square feet (perfect correlation)")
print("   and potentially between number of bedrooms and bathrooms (moderate correlation).")
print("2. Methods to detect multicollinearity include correlation matrix analysis, VIF calculation,")
print("   and eigenvalue/condition number analysis.")
print("3. Approaches to address multicollinearity include feature elimination and feature transformation/regularization.")
print("4. Ignoring multicollinearity leads to inflated standard errors, unstable coefficients,")
print("   potentially incorrect coefficient signs, and difficulty determining feature importance.")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- correlation_matrix.png: Heatmap showing feature correlations")
print("- sqm_vs_sqft.png: Scatter plot showing the relationship between square meters and square feet")
print("- bedrooms_vs_bathrooms.png: Scatter plot showing the relationship between bedrooms and bathrooms")
print("- feature_pairplot.png: Pairwise relationships between all features")
print("- vif_values.png: Bar chart of Variance Inflation Factors")
print("- eigenvalues.png: Bar chart of eigenvalues with scree plot")
print("- coefficient_instability.png: Scatter plot showing coefficient instability")
print("- correlation_after_elimination.png: Correlation matrix after removing collinear feature")
print("- vif_after_elimination.png: VIF values after feature elimination")
print("- vif_comparison.png: Comparison of VIF values before and after feature elimination")
print("- correlation_after_transformation.png: Correlation matrix after feature transformation")
print("- ridge_vs_ols.png: Comparison of OLS and Ridge regression coefficients")
print("- coefficient_stability_comparison.png: Stability comparison between OLS and Ridge")
print("- coefficient_stability_boxplot.png: Boxplot comparing coefficient distributions")
print("- coefficient_sign_problem.png: Histograms showing coefficient sign instability")
print("- confidence_interval_effect.png: Visualization of inflated confidence intervals") 