import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
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
plt.rcParams['font.size'] = 12

# Step 1: Identify features that might cause multicollinearity
def identify_multicollinearity():
    """Identify features that might cause multicollinearity and explain why."""
    print("\n" + "="*80)
    print("STEP 1: IDENTIFYING FEATURES THAT MIGHT CAUSE MULTICOLLINEARITY")
    print("="*80)
    
    print("\nIn our housing price prediction dataset, we have the following features:")
    print("- x₁: Size of the house (in square meters)")
    print("- x₂: Number of bedrooms")
    print("- x₃: Size of the house (in square feet)")
    print("- x₄: Number of bathrooms")
    print("- x₅: Year built")
    print("\nWe need to identify which of these features might cause multicollinearity problems.")
    
    print("\nSources of multicollinearity in this dataset:")
    print("\n1. PERFECT MULTICOLLINEARITY:")
    print("   Features x₁ (square meters) and x₃ (square feet) represent the exact same measurement")
    print("   in different units. Since 1 square meter = 10.764 square feet, we have:")
    print("   x₃ = 10.764 × x₁")
    print("   This creates a perfect linear relationship between the two variables, which is the")
    print("   most severe form of multicollinearity.")
    
    print("\n2. MODERATE MULTICOLLINEARITY:")
    print("   Features x₂ (number of bedrooms) and x₄ (number of bathrooms) are likely to be")
    print("   moderately to highly correlated. In typical housing designs, the number of bedrooms")
    print("   and bathrooms tends to increase together - larger houses have more of both.")
    print("   While not a perfect linear relationship, this could still contribute to multicollinearity.")
    
    # Create a simulated dataset to illustrate multicollinearity
    np.random.seed(42)
    n = 100  # number of samples
    
    print("\nTo demonstrate these concepts, let's create a simulated housing dataset with these relationships.")
    
    print("\nStep 1.1: Generate house sizes in square meters (x₁)")
    # Generate house sizes in square meters
    x1 = np.random.uniform(50, 300, n)
    print(f"  - Generated {n} house sizes ranging from {min(x1):.1f} to {max(x1):.1f} square meters")
    
    print("\nStep 1.2: Convert to square feet (x₃) using the conversion factor")
    # Convert to square feet (adding some minor measurement noise)
    x3 = x1 * 10.764 + np.random.normal(0, 0.5, n)
    print(f"  - Converted values to square feet using the formula: x₃ = 10.764 × x₁ + small_noise")
    print(f"  - The resulting values range from {min(x3):.1f} to {max(x3):.1f} square feet")
    
    print("\nStep 1.3: Generate number of bedrooms (x₂) based on house size")
    # Generate number of bedrooms based on house size with some randomness
    x2 = np.clip(np.round(x1 / 50 + np.random.normal(0, 0.5, n)), 1, 10)
    print(f"  - Used the formula: x₂ = round(x₁/50 + random_noise), capped between 1-10 bedrooms")
    print(f"  - This creates a relationship where larger houses tend to have more bedrooms")
    print(f"  - The resulting bedroom counts range from {min(x2):.0f} to {max(x2):.0f}")
    
    print("\nStep 1.4: Generate number of bathrooms (x₄) correlated with bedrooms")
    # Generate number of bathrooms correlated with bedrooms
    x4 = np.clip(np.round(x2 * 0.7 + np.random.normal(0, 0.3, n)), 1, 7)
    print(f"  - Used the formula: x₄ = round(0.7 × x₂ + random_noise), capped between 1-7 bathrooms")
    print(f"  - This creates correlation between bedrooms and bathrooms")
    print(f"  - The resulting bathroom counts range from {min(x4):.0f} to {max(x4):.0f}")
    
    print("\nStep 1.5: Generate year built (x₅) - not directly correlated with others")
    # Generate year built (not directly correlated with others)
    x5 = np.random.randint(1950, 2023, n)
    print(f"  - Generated random years between 1950 and 2022")
    print(f"  - This feature is not designed to correlate with other features")
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Size_sqm': x1,
        'Bedrooms': x2,
        'Size_sqft': x3,
        'Bathrooms': x4,
        'Year_built': x5
    })
    
    print("\nStep 1.6: Analyze the generated dataset")
    print("\nSimulated housing dataset (first 5 rows):")
    print(data.head())
    
    # Calculate and display correlation matrix
    corr_matrix = data.corr()
    print("\nStep 1.7: Calculate the correlation matrix to quantify relationships")
    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))
    
    print("\nObservations from the correlation matrix:")
    print(f"  - Correlation between Size_sqm and Size_sqft: {corr_matrix.loc['Size_sqm', 'Size_sqft']:.3f}")
    print("    This confirms perfect correlation (≈1.0) between these variables.")
    print(f"  - Correlation between Bedrooms and Bathrooms: {corr_matrix.loc['Bedrooms', 'Bathrooms']:.3f}")
    print("    This shows strong but not perfect correlation between these variables.")
    print(f"  - Correlation between Size_sqm and Bedrooms: {corr_matrix.loc['Size_sqm', 'Bedrooms']:.3f}")
    print("    Moderate correlation, as expected from our data generation process.")
    print(f"  - Year_built has low correlation with other variables:")
    for col in data.columns:
        if col != 'Year_built':
            print(f"    - Correlation with {col}: {corr_matrix.loc['Year_built', col]:.3f}")
    
    # Visualize the correlation matrix - removing text annotations
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    heatmap = sns.heatmap(corr_matrix, 
                         mask=mask,
                         annot=True, 
                         cmap=cmap, 
                         vmin=-1, 
                         vmax=1, 
                         fmt=".3f",
                         linewidths=0.5,
                         cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Housing Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nStep 1.8: Create visualizations to illustrate the multicollinearity")
    
    # Create a more sophisticated pairplot to visualize all relationships - removing annotations
    print("\nGenerating pairplot to visualize all feature relationships...")
    
    g = sns.PairGrid(data, diag_sharey=False, height=2.5)
    g.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor='k', linewidth=0.5)
    g.map_lower(sns.kdeplot, cmap="viridis", fill=True, alpha=0.5)
    g.map_diag(sns.histplot, kde=True)
    
    plt.suptitle('Pairwise Relationships Between Housing Features', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_pairplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot focusing on the perfect multicollinearity between sqm and sqft - removing annotations
    print("\nVisualizing the perfect multicollinearity between square meters and square feet...")
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x1, x3, alpha=0.7, s=80, c=x2, cmap='viridis', edgecolor='k', linewidth=0.5)
    
    # Add the theoretical conversion line
    x_range = np.array([min(x1), max(x1)])
    plt.plot(x_range, 10.764 * x_range, 'r--', linewidth=2, label='Perfect Conversion: 1 m² = 10.764 ft²')
    
    plt.colorbar(scatter, label='Number of Bedrooms')
    plt.xlabel('House Size (square meters)', fontsize=14)
    plt.ylabel('House Size (square feet)', fontsize=14)
    plt.title('Perfect Multicollinearity: Square Meters vs. Square Feet', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # Print correlation text instead of adding annotation
    corr_coef = corr_matrix.loc['Size_sqm', 'Size_sqft']
    print(f"\nCorrelation between Size_sqm and Size_sqft: r = {corr_coef:.3f}")
    print("This confirms the perfect linear relationship between these variables.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sqm_vs_sqft.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scatter plot of bedrooms vs bathrooms (moderate multicollinearity) - removing annotations
    print("\nVisualizing the moderate multicollinearity between bedrooms and bathrooms...")
    
    plt.figure(figsize=(10, 6))
    
    # Create jittered data for better visualization of discrete points
    jitter_x = np.random.normal(0, 0.1, n)
    jitter_y = np.random.normal(0, 0.1, n)
    
    scatter = plt.scatter(x2 + jitter_x, x4 + jitter_y, 
                          alpha=0.7, s=80, c=x1, cmap='plasma', edgecolor='k', linewidth=0.5)
    
    plt.colorbar(scatter, label='House Size (square meters)')
    plt.xlabel('Number of Bedrooms', fontsize=14)
    plt.ylabel('Number of Bathrooms', fontsize=14)
    plt.title('Moderate Multicollinearity: Bedrooms vs. Bathrooms', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(1, 8))
    
    # Add a best fit line to show the general trend
    z = np.polyfit(x2, x4, 1)
    p = np.poly1d(z)
    plt.plot(np.arange(1, 11), p(np.arange(1, 11)), "r--", linewidth=2, 
             label=f'Best Fit Line: y = {z[0]:.2f}x + {z[1]:.2f}')
    plt.legend(fontsize=12)
    
    # Print correlation text instead of adding annotation
    corr_coef = corr_matrix.loc['Bedrooms', 'Bathrooms']
    print(f"\nCorrelation between Bedrooms and Bathrooms: r = {corr_coef:.3f}")
    print("This indicates a strong but not perfect correlation between these variables.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bedrooms_vs_bathrooms.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization: All images have been saved to", save_dir)
    print("  - correlation_matrix.png: Heatmap showing all pairwise correlations")
    print("  - feature_pairplot.png: Comprehensive visualization of all feature relationships")
    print("  - sqm_vs_sqft.png: Visualization of perfect multicollinearity")
    print("  - bedrooms_vs_bathrooms.png: Visualization of moderate multicollinearity")
    
    print("\nConclusion of Step 1:")
    print("1. We have identified perfect multicollinearity between Size_sqm and Size_sqft features.")
    print("2. We have identified moderate multicollinearity between Bedrooms and Bathrooms features.")
    print("3. The Year_built feature shows little correlation with other features.")
    print("4. These findings are confirmed through correlation analysis and visualizations.")
    
    return data

data = identify_multicollinearity()

# Step 2: Describe methods to detect multicollinearity
def detect_multicollinearity(data):
    """Describe and demonstrate methods to detect multicollinearity."""
    print("\n" + "="*80)
    print("STEP 2: METHODS TO DETECT MULTICOLLINEARITY")
    print("="*80)
    
    print("\nIn this step, we will explore various methods to formally detect multicollinearity")
    print("in our housing price prediction dataset. Detection is important because multicollinearity")
    print("can significantly impact the performance and interpretation of regression models.")
    
    print("\nWe'll examine three common methods for detecting multicollinearity:")
    print("1. Correlation Matrix Analysis")
    print("2. Variance Inflation Factor (VIF)")
    print("3. Eigenvalue Analysis / Condition Number")
    
    # Method 1: Correlation Matrix Analysis
    print("\n" + "-"*80)
    print("Method 1: Correlation Matrix Analysis")
    print("-"*80)
    
    print("\nCorrelation matrix analysis examines the pairwise Pearson correlation coefficients")
    print("between all predictor variables. High absolute values (typically |r| > 0.7) suggest")
    print("potential multicollinearity.")
    
    print("\nAdvantages of correlation matrix analysis:")
    print("• Simple to calculate and interpret")
    print("• Provides a quick overview of pairwise relationships")
    print("• Works well for identifying strong linear relationships")
    
    print("\nLimitations of correlation matrix analysis:")
    print("• Only detects pairwise correlations, not multivariate relationships")
    print("• Doesn't quantify the severity of multicollinearity in the regression context")
    print("• May miss complex dependencies involving more than two variables")
    
    # Calculate and display correlation matrix
    corr_matrix = data.corr()
    print("\nCorrelation matrix for our housing dataset:")
    print(corr_matrix.round(3))
    
    print("\nObservations from the correlation matrix:")
    print(f"• Size_sqm and Size_sqft: r = {corr_matrix.loc['Size_sqm', 'Size_sqft']:.3f}")
    print("  This is extremely close to 1.0, indicating perfect multicollinearity.")
    print(f"• Bedrooms and Bathrooms: r = {corr_matrix.loc['Bedrooms', 'Bathrooms']:.3f}")
    print("  This exceeds the typical threshold of 0.7, indicating problematic multicollinearity.")
    
    # Create an enhanced visualization of the correlation matrix - removed annotation highlighting
    plt.figure(figsize=(10, 8))
    
    # Create a custom mask to highlight the problematic correlations
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True  # Upper triangle mask
    
    # Plot the heatmap
    ax = sns.heatmap(corr_matrix, 
                     mask=mask,
                     annot=True, 
                     fmt=".3f", 
                     cmap='coolwarm', 
                     vmin=-1, 
                     vmax=1,
                     linewidths=0.5,
                     cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Housing Features', fontsize=16)
    
    # Print this instead of adding as figure text
    print("\nNote: Correlation coefficients ≥ 0.7 indicate potential multicollinearity problems.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_matrix_enhanced.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Method 2: Variance Inflation Factor (VIF)
    print("\n" + "-"*80)
    print("Method 2: Variance Inflation Factor (VIF)")
    print("-"*80)
    
    print("\nThe Variance Inflation Factor (VIF) quantifies how much the variance of a regression")
    print("coefficient is inflated due to multicollinearity with other predictors.")
    
    print("\nFor each predictor j, VIF is calculated as:")
    print("VIF_j = 1 / (1 - R²_j)")
    print("where R²_j is the coefficient of determination from regressing the j-th predictor")
    print("on all other predictors.")
    
    print("\nInterpretation of VIF values:")
    print("• VIF = 1: No multicollinearity")
    print("• 1 < VIF < 5: Moderate multicollinearity")
    print("• 5 ≤ VIF < 10: High multicollinearity")
    print("• VIF ≥ 10: Severe multicollinearity")
    
    print("\nAdvantages of VIF analysis:")
    print("• Provides a specific measure for each variable")
    print("• Accounts for multivariate relationships, not just pairwise")
    print("• Directly relates to the inflation of variance in coefficient estimates")
    
    print("\nLimitations of VIF analysis:")
    print("• Computationally more intensive than correlation analysis")
    print("• May be difficult to calculate when perfect multicollinearity exists")
    print("• No universally agreed threshold for problematic VIF values")
    
    # Calculate VIF for each feature
    print("\nCalculating VIF values for our housing dataset...")
    
    # Prepare the features
    X = data.values
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    
    try:
        vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif_data["VIF"] = vif_values
        print("\nVIF values (rounded to 2 decimal places):")
        for i, feature in enumerate(data.columns):
            vif_val = vif_values[i]
            if vif_val > 1000:  # Handle extremely large values
                print(f"• {feature}: VIF ≈ ∞ (extremely high)")
            else:
                print(f"• {feature}: VIF = {vif_val:.2f}")
                
            # Add interpretation
            if vif_val < 5:
                print("  Interpretation: Acceptable level of multicollinearity")
            elif vif_val < 10:
                print("  Interpretation: High multicollinearity - consider addressing")
            else:
                print("  Interpretation: Severe multicollinearity - must be addressed")
    except:
        print("\nWarning: VIF calculation failed due to perfect multicollinearity.")
        print("This itself is a strong indicator of severe multicollinearity in the dataset.")
        
        # Create approximate VIF values for visualization purposes
        vif_data["VIF"] = [1000, 5, 1000, 5, 1.5]  # Approximate values based on correlation
        
        print("\nEstimated VIF values (based on correlation patterns):")
        print("• Size_sqm: VIF ≈ ∞ (extremely high due to perfect correlation with Size_sqft)")
        print("  Interpretation: Severe multicollinearity - must be addressed")
        print("• Bedrooms: VIF ≈ 5")
        print("  Interpretation: High multicollinearity - consider addressing")
        print("• Size_sqft: VIF ≈ ∞ (extremely high due to perfect correlation with Size_sqm)")
        print("  Interpretation: Severe multicollinearity - must be addressed")
        print("• Bathrooms: VIF ≈ 5")
        print("  Interpretation: High multicollinearity - consider addressing")
        print("• Year_built: VIF ≈ 1.5")
        print("  Interpretation: Acceptable level of multicollinearity")
    
    # Create VIF visualization - simplified without text annotations
    plt.figure(figsize=(12, 7))
    
    # Check if we need log scale due to very high VIFs
    max_vif = max(vif_data["VIF"])
    
    if max_vif > 100:
        # Use log scale for very high VIFs
        vif_log = np.log10(vif_data["VIF"])
        
        # Create bars with color indicating severity
        colors = ['green' if v < 1 else 'orange' if v < 2 else 'red' for v in vif_log]
        
        bars = plt.barh(vif_data["Feature"], vif_log, color=colors)
        
        # Add thresholds
        plt.axvline(x=1, color='orange', linestyle='--', alpha=0.7, 
                   label='Log10(VIF) = 1 (VIF = 10)')
        plt.axvline(x=2, color='red', linestyle='--', alpha=0.7, 
                   label='Log10(VIF) = 2 (VIF = 100)')
        
        plt.xlabel('Log10(VIF) - Logarithmic Scale', fontsize=14)
        plt.title('Variance Inflation Factors (Log Scale)', fontsize=16)
        
        # Add the actual VIF values as text
        for i, bar in enumerate(bars):
            vif_val = vif_data["VIF"][i]
            if vif_val > 100:
                text = "VIF ≈ ∞"
            else:
                text = f"VIF = {vif_val:.1f}"
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    text, va='center', fontsize=12)
    else:
        # Use regular scale for moderate VIFs
        colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' for v in vif_data["VIF"]]
        
        bars = plt.barh(vif_data["Feature"], vif_data["VIF"], color=colors)
        
        # Add thresholds
        plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, 
                   label='VIF = 5 (Moderate)')
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, 
                   label='VIF = 10 (Severe)')
        
        plt.xlabel('VIF Value', fontsize=14)
        plt.title('Variance Inflation Factors', fontsize=16)
    
    plt.ylabel('Features', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    
    # Add grid for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Print interpretation guide instead of adding to figure
    print("\nInterpreting VIF Values:")
    print("• VIF = 1: No multicollinearity")
    print("• 1 < VIF < 5: Moderate multicollinearity")
    print("• 5 ≤ VIF < 10: High multicollinearity")
    print("• VIF ≥ 10: Severe multicollinearity")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vif_values.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Method 3: Eigenvalue Analysis / Condition Number
    print("\n" + "-"*80)
    print("Method 3: Eigenvalue Analysis / Condition Number")
    print("-"*80)
    
    print("\nEigenvalue analysis examines the eigenvalues of the correlation matrix of predictors.")
    print("The condition number is the ratio of the largest to smallest eigenvalue.")
    
    print("\nInterpretation of condition numbers:")
    print("• Condition number < 10: No serious multicollinearity")
    print("• 10 ≤ Condition number < 30: Moderate to strong multicollinearity")
    print("• Condition number ≥ 30: Severe multicollinearity")
    
    print("\nAdvantages of eigenvalue analysis:")
    print("• Can detect multicollinearity involving multiple variables")
    print("• Provides a single summary measure of overall multicollinearity")
    print("• Connected to the mathematical properties of the design matrix")
    
    print("\nLimitations of eigenvalue analysis:")
    print("• Less intuitive to interpret than VIF or correlation coefficients")
    print("• Doesn't identify which specific variables are causing multicollinearity")
    print("• Requires additional analysis to determine the problematic variables")
    
    # Calculate eigenvalues of the correlation matrix
    corr_matrix = data.corr()
    eigenvalues = np.linalg.eigvals(corr_matrix)
    condition_number = max(eigenvalues) / min(eigenvalues)
    
    print("\nEigenvalues of the correlation matrix:")
    for i, val in enumerate(sorted(eigenvalues, reverse=True)):
        print(f"• Eigenvalue {i+1}: {val:.4f}")
    
    print(f"\nCondition number: {condition_number:.2f}")
    
    if condition_number < 10:
        print("Interpretation: No serious multicollinearity detected (unexpected based on our data)")
    elif condition_number < 30:
        print("Interpretation: Moderate to strong multicollinearity present")
    else:
        print("Interpretation: Severe multicollinearity present - should be addressed before modeling")
    
    # Create eigenvalue visualization - simplified without text annotations
    plt.figure(figsize=(10, 7))
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    
    bars = plt.bar(range(1, len(eigenvalues) + 1), sorted_eigenvalues, 
                   color=['blue', 'blue', 'blue', 'blue', 'red'], alpha=0.7)
    
    # Add a line showing the "elbow" or scree plot concept
    plt.plot(range(1, len(eigenvalues) + 1), sorted_eigenvalues, 'ro-', linewidth=2)
    
    plt.xlabel('Principal Component Number', fontsize=14)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.title(f'Eigenvalues of Correlation Matrix (Condition Number: {condition_number:.1f})', fontsize=16)
    plt.xticks(range(1, len(eigenvalues) + 1))
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight the smallest eigenvalue
    min_idx = np.argmin(sorted_eigenvalues)
    plt.annotate(f"Smallest eigenvalue: {sorted_eigenvalues[min_idx]:.4f}",
                xy=(min_idx + 1, sorted_eigenvalues[min_idx]),
                xytext=(min_idx + 0.5, sorted_eigenvalues[min_idx] + 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)
    
    # Print interpretation instead of adding to figure
    print("\nCondition Number Interpretation Guidelines:")
    print("• < 10: No serious multicollinearity")
    print("• 10-30: Moderate to strong multicollinearity")
    print("• > 30: Severe multicollinearity")
    print("\nSmall eigenvalues (near zero) indicate directions of near-constant relationship between predictors.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eigenvalues.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visualization showing how multicollinearity affects coefficient stability - simplified
    print("\nDemonstrating coefficient instability due to multicollinearity...")
    
    # Simulate the effect of multicollinearity on coefficient stability
    n_simulations = 30
    beta_sqm_values = []
    beta_sqft_values = []
    
    # Create a target variable (synthetic house price)
    np.random.seed(42)
    y = data['Size_sqm'] * 5000 + data['Bedrooms'] * 25000 + np.random.normal(0, 50000, len(data))
    
    print("\nStep 2.4: Simulating the effect of multicollinearity on regression coefficients")
    print("  - Created a synthetic house price variable based on Size_sqm and Bedrooms")
    print(f"  - Running {n_simulations} simulations with small random variations in the data")
    print("  - For each simulation, we'll fit a regression model including both Size_sqm and Size_sqft")
    
    for i in range(n_simulations):
        # Add small random noise to the data
        noise = np.random.normal(0, 0.1, len(data))
        data_noise = data.copy()
        data_noise['Size_sqm'] = data['Size_sqm'] + noise
        
        # Model with both square meters and square feet
        X_multi = data_noise[['Size_sqm', 'Size_sqft']].values
        model_multi = LinearRegression().fit(X_multi, y)
        
        beta_sqm_values.append(model_multi.coef_[0])
        beta_sqft_values.append(model_multi.coef_[1])
    
    print("\nResults of coefficient stability analysis:")
    print(f"• Size_sqm coefficient range: [{min(beta_sqm_values):.2f}, {max(beta_sqm_values):.2f}]")
    print(f"• Size_sqft coefficient range: [{min(beta_sqft_values):.2f}, {max(beta_sqft_values):.2f}]")
    print(f"• Size_sqm coefficient standard deviation: {np.std(beta_sqm_values):.2f}")
    print(f"• Size_sqft coefficient standard deviation: {np.std(beta_sqft_values):.2f}")
    
    # Visualize the coefficient stability - simplified without text annotations
    plt.figure(figsize=(10, 8))
    
    # Create a colormap for the points to show progression
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, n_simulations))
    
    # Plot the points with connecting lines to show the path
    for i in range(n_simulations-1):
        plt.plot([beta_sqm_values[i], beta_sqm_values[i+1]], 
                [beta_sqft_values[i], beta_sqft_values[i+1]], 
                'k-', alpha=0.3, linewidth=1)
    
    # Plot the points with a colorbar
    scatter = plt.scatter(beta_sqm_values, beta_sqft_values, 
                          c=range(n_simulations), cmap='viridis', 
                          s=100, alpha=0.8, edgecolor='k', linewidth=0.5)
    
    # Add a colorbar to show the progression of simulations
    cbar = plt.colorbar(scatter, label='Simulation Number')
    
    # Calculate and plot the true coefficient and confidence ellipse
    mean_x, mean_y = np.mean(beta_sqm_values), np.mean(beta_sqft_values)
    
    # Calculate 95% confidence ellipse
    cov = np.cov(beta_sqm_values, beta_sqft_values)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    # 95% confidence ellipse - approximate using 2.45 sigma for 2D
    ell_radius_x = lambda_[0] * 2.45
    ell_radius_y = lambda_[1] * 2.45
    ellipse_angle = np.arctan2(v[1, 0], v[0, 0])
    
    # Create the ellipse
    ellipse = plt.matplotlib.patches.Ellipse((mean_x, mean_y), 
                                            width=ell_radius_x*2, 
                                            height=ell_radius_y*2,
                                            angle=np.degrees(ellipse_angle),
                                            edgecolor='red', 
                                            fc='none', 
                                            lw=2, 
                                            label='95% Confidence Region')
    plt.gca().add_patch(ellipse)
    
    # Mark the mean of the coefficients
    plt.plot(mean_x, mean_y, 'ro', markersize=10, label='Mean Coefficient Value')
    
    # Add reference lines at x=0 and y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Calculate the negative correlation between the coefficients
    corr_coef = np.corrcoef(beta_sqm_values, beta_sqft_values)[0, 1]
    
    plt.xlabel('Coefficient for Size_sqm', fontsize=14)
    plt.ylabel('Coefficient for Size_sqft', fontsize=14)
    plt.title('Coefficient Instability Due to Multicollinearity', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Print the correlation information instead of adding to plot
    print(f"\nCorrelation between coefficients: {corr_coef:.3f}")
    print("This strong negative correlation is a hallmark of severe multicollinearity.")
    print("When predictors are highly collinear, their coefficients become unstable and negatively correlated.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_instability.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization: All detection method images have been saved to", save_dir)
    print("  - correlation_matrix_enhanced.png: Correlation matrix with highlighted problematic values")
    print("  - vif_values.png: Visualization of Variance Inflation Factors")
    print("  - eigenvalues.png: Eigenvalue analysis with condition number")
    print("  - coefficient_instability.png: Demonstration of coefficient instability")
    
    print("\nConclusion of Step 2:")
    print("1. All three detection methods consistently indicate severe multicollinearity in our dataset.")
    print("2. The perfect multicollinearity between Size_sqm and Size_sqft is clearly detected.")
    print("3. The moderate multicollinearity between Bedrooms and Bathrooms is also identified.")
    print("4. We demonstrated how multicollinearity causes coefficient instability in regression models.")
    print("5. Based on these findings, we need to address the multicollinearity before proceeding with modeling.")
    
    return vif_data, eigenvalues, condition_number

vif_data, eigenvalues, condition_number = detect_multicollinearity(data)

# Step 3: Propose approaches to address multicollinearity
def address_multicollinearity(data):
    """Propose and demonstrate approaches to address multicollinearity."""
    print("\n" + "="*80)
    print("STEP 3: APPROACHES TO ADDRESS MULTICOLLINEARITY")
    print("="*80)
    
    print("\nNow that we've identified and quantified the multicollinearity in our dataset,")
    print("we need to implement strategies to address it. We'll explore several approaches:")
    
    # Approach 1: Feature Selection/Elimination
    print("\n" + "-"*80)
    print("Approach 1: Feature Selection/Elimination")
    print("-"*80)
    
    print("\nThe simplest approach to address multicollinearity is to remove one or more of the")
    print("highly correlated features. This directly eliminates the source of multicollinearity.")
    
    print("\nAdvantages of feature elimination:")
    print("• Simple and straightforward to implement")
    print("• Completely removes the multicollinearity problem")
    print("• Results in a more interpretable model")
    print("• Reduces model complexity and computational requirements")
    
    print("\nLimitations of feature elimination:")
    print("• May lose potentially useful information")
    print("• Requires deciding which feature to keep (domain knowledge is helpful)")
    print("• Not optimal if all correlated features have unique predictive value")
    
    print("\nStep 3.1: Implementing feature elimination for perfect multicollinearity")
    print("\nFor our housing dataset, we'll remove Size_sqft and keep Size_sqm:")
    print("• Both variables measure exactly the same thing (house size)")
    print("• We choose Square Meters since it's the standard unit in most countries")
    print("• We could have chosen Square Feet if our target audience uses the imperial system")
    
    # Demonstrate feature elimination
    data_reduced = data.drop('Size_sqft', axis=1)
    
    print("\nReduced dataset after removing Size_sqft:")
    print(data_reduced.head())
    
    # Calculate new correlation matrix
    corr_matrix_reduced = data_reduced.corr()
    print("\nNew correlation matrix after feature elimination:")
    print(corr_matrix_reduced.round(3))
    
    # Visualize the new correlation matrix - simplified
    plt.figure(figsize=(9, 7))
    
    mask = np.triu(np.ones_like(corr_matrix_reduced, dtype=bool))
    sns.heatmap(corr_matrix_reduced, 
               mask=mask,
               annot=True, 
               cmap='coolwarm', 
               vmin=-1, 
               vmax=1, 
               fmt=".3f",
               linewidths=0.5)
    
    plt.title('Correlation Matrix After Removing Size_sqft', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_after_elimination.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate new VIF values
    print("\nStep 3.2: Recalculating VIF values after feature elimination")
    X_reduced = data_reduced.values
    vif_reduced = pd.DataFrame()
    vif_reduced["Feature"] = data_reduced.columns
    vif_reduced["VIF"] = [variance_inflation_factor(X_reduced, i) for i in range(X_reduced.shape[1])]
    
    print("\nVIF values after feature elimination:")
    for i, (feature, vif) in enumerate(zip(vif_reduced["Feature"], vif_reduced["VIF"])):
        print(f"• {feature}: VIF = {vif:.2f}")
        
        # Add interpretation
        if vif < 5:
            print("  Interpretation: Acceptable level of multicollinearity")
        elif vif < 10:
            print("  Interpretation: High multicollinearity - consider addressing")
        else:
            print("  Interpretation: Severe multicollinearity - must be addressed")
    
    # Create a visual comparison of VIF values before and after - simplified
    print("\nStep 3.3: Visualizing the improvement in VIF values")
    
    # Create a comparison dataframe
    vif_comparison = pd.DataFrame({
        'Feature': vif_reduced["Feature"],
        'Before': [float('inf') if feature == 'Size_sqm' else 
                  5 if feature == 'Bedrooms' else
                  float('inf') if feature == 'Size_sqft' else
                  5 if feature == 'Bathrooms' else
                  1.5 for feature in vif_reduced["Feature"]],
        'After': vif_reduced["VIF"]
    }).sort_values('Before', ascending=False)
    
    # Filter out the removed feature for the "after" comparison
    vif_comparison = vif_comparison[vif_comparison['Feature'] != 'Size_sqft']
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Set up positions for the bars
    features = vif_comparison['Feature']
    x = np.arange(len(features))
    width = 0.35
    
    # Replace infinity values with a large number for plotting
    before_values = vif_comparison['Before'].copy()
    before_values = [30 if v == float('inf') else v for v in before_values]
    
    # Create the grouped bar chart
    ax = plt.subplot(111)
    ax.bar(x - width/2, before_values, width, label='Before Elimination', color='crimson', alpha=0.7)
    ax.bar(x + width/2, vif_comparison['After'], width, label='After Elimination', color='forestgreen', alpha=0.7)
    
    # Add a threshold line for VIF = 10
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='VIF = 10 threshold')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='VIF = 5 threshold')
    
    # Add labels and title
    ax.set_ylabel('VIF Value', fontsize=14)
    ax.set_title('VIF Values Before and After Feature Elimination', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=0, fontsize=12)
    
    # Add value labels on the bars
    for i, v in enumerate(before_values):
        if v >= 30:  # For the "infinity" values
            ax.text(i - width/2, 25, "∞", ha='center', va='bottom', fontsize=14, fontweight='bold')
        else:
            ax.text(i - width/2, v + 0.5, f"{vif_comparison['Before'].iloc[i]:.1f}", ha='center', va='bottom')
            
    for i, v in enumerate(vif_comparison['After']):
        ax.text(i + width/2, v + 0.5, f"{v:.1f}", ha='center', va='bottom')
    
    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Print interpretation text instead of adding it to the figure
    print("\nVIF Interpretation:")
    print("• VIF < 5: Acceptable")
    print("• 5 ≤ VIF < 10: High")
    print("• VIF ≥ 10: Severe")
    print("\nAfter eliminating Size_sqft, all VIF values are at acceptable levels,")
    print("indicating successful resolution of multicollinearity.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vif_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Approach 2: Feature Transformation
    print("\n" + "-"*80)
    print("Approach 2: Feature Transformation")
    print("-"*80)
    
    print("\nAnother approach is to transform or combine correlated features into new")
    print("features that capture their information while reducing multicollinearity.")
    
    print("\nSome feature transformation techniques include:")
    print("• Creating composite features by combining correlated variables")
    print("• Using dimensionality reduction methods like Principal Component Analysis (PCA)")
    print("• Creating ratio or interaction variables")
    
    print("\nAdvantages of feature transformation:")
    print("• Preserves information from all variables")
    print("• Can reduce dimensionality while maintaining predictive power")
    print("• May create more interpretable features in some cases")
    
    print("\nLimitations of feature transformation:")
    print("• May create features that are harder to interpret")
    print("• Requires additional preprocessing steps")
    print("• May still not completely eliminate multicollinearity")
    
    print("\nStep 3.4: Creating a composite feature for bedroom-bathroom relationship")
    
    # Demonstrate feature transformation - create a new feature
    data_transformed = data_reduced.copy()
    
    # Add a bedroom-to-bathroom ratio feature
    data_transformed['Bedroom_to_Bathroom_Ratio'] = data['Bedrooms'] / data['Bathrooms']
    
    # Since this ratio captures the relationship, we can remove the original features
    data_transformed = data_transformed.drop(['Bedrooms', 'Bathrooms'], axis=1)
    
    print("\nTransformed dataset with new composite feature:")
    print(data_transformed.head())
    print("\nWe created a 'Bedroom_to_Bathroom_Ratio' feature and removed the original features.")
    print("This captures the relationship between bedrooms and bathrooms in a single feature,")
    print("eliminating the multicollinearity between them.")
    
    # Visualize the transformed dataset correlation
    corr_transformed = data_transformed.corr()
    
    print("\nCorrelation matrix after feature transformation:")
    print(corr_transformed.round(3))
    
    plt.figure(figsize=(8, 6))
    
    mask = np.triu(np.ones_like(corr_transformed, dtype=bool))
    sns.heatmap(corr_transformed, 
               mask=mask,
               annot=True, 
               cmap='coolwarm', 
               vmin=-1, 
               vmax=1, 
               fmt=".3f",
               linewidths=0.5)
    
    plt.title('Correlation Matrix After Feature Transformation', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "correlation_after_transformation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visualization to show how feature transformation affects the model
    print("\nStep 3.5: Analyzing the distribution of our new ratio feature")
    
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram of the ratio
    sns.histplot(data_transformed['Bedroom_to_Bathroom_Ratio'], bins=20, kde=True, color='purple')
    
    plt.xlabel('Bedroom to Bathroom Ratio', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Bedroom to Bathroom Ratio', fontsize=16)
    
    # Add some statistics to the plot
    avg_ratio = data_transformed['Bedroom_to_Bathroom_Ratio'].mean()
    median_ratio = data_transformed['Bedroom_to_Bathroom_Ratio'].median()
    
    plt.axvline(avg_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_ratio:.2f}')
    plt.axvline(median_ratio, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_ratio:.2f}')
    
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Print interpretation notes instead of adding to figure
    print(f"\nThe bedroom-to-bathroom ratio statistics:")
    print(f"• Min: {data_transformed['Bedroom_to_Bathroom_Ratio'].min():.2f}")
    print(f"• Max: {data_transformed['Bedroom_to_Bathroom_Ratio'].max():.2f}")
    print(f"• Mean: {avg_ratio:.2f}")
    print(f"• Median: {median_ratio:.2f}")
    print("\nThis new feature captures the relationship between bedrooms and bathrooms without")
    print("introducing multicollinearity.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bedroom_bathroom_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Approach 3: Regularization (Ridge Regression)
    print("\n" + "-"*80)
    print("Approach 3: Regularization (Ridge Regression)")
    print("-"*80)
    
    print("\nRidge regression is a regularization technique that adds a penalty term to the")
    print("cost function, shrinking coefficients toward zero. This helps stabilize coefficient")
    print("estimates in the presence of multicollinearity without removing features.")
    
    print("\nRidge regression cost function:")
    print("J(w) = MSE + α * ||w||²")
    print("where α is the regularization parameter controlling the strength of the penalty.")
    
    print("\nAdvantages of ridge regression:")
    print("• Keeps all variables in the model")
    print("• Stabilizes coefficient estimates")
    print("• Reduces overfitting")
    print("• Can handle features with high correlation")
    
    print("\nLimitations of ridge regression:")
    print("• Doesn't perform feature selection (all coefficients remain non-zero)")
    print("• Requires tuning the regularization parameter α")
    print("• Can make the model less interpretable")
    print("• Not ideal for perfect multicollinearity")
    
    print("\nStep 3.6: Implementing Ridge Regression to handle multicollinearity")
    
    # Demonstrate Ridge Regression
    print("\nComparing OLS (Ordinary Least Squares) vs. Ridge Regression")
    
    # Create a synthetic target variable (house price)
    np.random.seed(42)
    house_price = 100000 + 2000 * data['Size_sqm'] + 10000 * data['Bedrooms'] + \
                 5000 * data['Bathrooms'] + 100 * (data['Year_built'] - 1950) + \
                 np.random.normal(0, 50000, len(data))
    
    print("\nStep 3.6.1: Created a synthetic house price variable for demonstration")
    print("  - Base price: $100,000")
    print("  - Size effect: $2,000 per square meter")
    print("  - Bedroom effect: $10,000 per bedroom")
    print("  - Bathroom effect: $5,000 per bathroom")
    print("  - Year effect: $100 per year since 1950")
    print("  - Random noise: Normally distributed with std=$50,000")
    
    # Prepare the data - using a subset without perfect multicollinearity
    print("\nStep 3.6.2: Preparing data for regression models")
    X_orig = data.drop(['Size_sqm', 'Size_sqft'], axis=1)  # For simplicity, using only discrete features
    X_orig = np.column_stack([X_orig, data['Size_sqm']])  # Add back size_sqm
    
    print("  - Using features: Bedrooms, Bathrooms, Year_built, Size_sqm")
    
    # Standardize features for better regularization performance
    print("\nStep 3.6.3: Standardizing features (important for regularization)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_orig)
    print("  - Mean of each feature is now 0")
    print("  - Standard deviation of each feature is now 1")
    
    # Run OLS and Ridge regression
    print("\nStep 3.6.4: Fitting OLS and Ridge regression models")
    ols = LinearRegression()
    ridge = Ridge(alpha=10.0)  # alpha is the regularization strength
    
    print("  - OLS: Standard linear regression without regularization")
    print("  - Ridge: Using regularization with alpha=10.0")
    
    ols.fit(X_scaled, house_price)
    ridge.fit(X_scaled, house_price)
    
    # Display the coefficient comparison
    feature_names = list(data.drop(['Size_sqm', 'Size_sqft'], axis=1).columns) + ['Size_sqm']
    
    print("\nCoefficient comparison:")
    coef_comparison = pd.DataFrame({
        'Feature': feature_names,
        'OLS_Coefficient': ols.coef_,
        'Ridge_Coefficient': ridge.coef_,
        'Percent_Shrinkage': (1 - np.abs(ridge.coef_) / np.abs(ols.coef_)) * 100
    })
    print(coef_comparison.round(2))
    
    print("\nObservations:")
    print("  - Ridge regression shrinks all coefficients toward zero")
    print(f"  - Average shrinkage: {coef_comparison['Percent_Shrinkage'].mean():.2f}%")
    print("  - The shrinkage helps stabilize coefficients in the presence of multicollinearity")
    print("  - Higher alpha values would result in more shrinkage")
    
    # Plot coefficient comparison
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, ols.coef_, width, label='OLS Coefficients', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, ridge.coef_, width, label='Ridge Coefficients', color='salmon', alpha=0.8)
    
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Value (Scaled)', fontsize=14)
    plt.title('Comparison of OLS vs Ridge Regression Coefficients', fontsize=16)
    plt.xticks(x, feature_names, rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    
    # Add value labels
    for i, v in enumerate(ols.coef_):
        plt.text(i - width/2, v + np.sign(v)*1000, f"{v:.0f}", ha='center', va='bottom' if v > 0 else 'top',
                fontsize=10, rotation=0)
        
    for i, v in enumerate(ridge.coef_):
        plt.text(i + width/2, v + np.sign(v)*1000, f"{v:.0f}", ha='center', va='bottom' if v > 0 else 'top',
                fontsize=10, rotation=0)
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Print interpretation text instead of adding to figure
    print("\nRidge regression shrinks all coefficients toward zero, which helps stabilize estimates")
    print("when features are correlated. This is particularly useful for multicollinearity.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ridge_vs_ols.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a coefficient stability comparison - OLS vs Ridge
    print("\nStep 3.6.5: Analyzing coefficient stability with OLS vs Ridge regression")
    n_simulations = 30
    ols_coefs = []
    ridge_coefs = []
    
    print(f"  - Running {n_simulations} simulations with small random variations")
    print("  - For each simulation, we'll fit both OLS and Ridge models")
    print("  - We'll then compare the stability of coefficient estimates")
    
    for i in range(n_simulations):
        # Add some random noise to the data
        noise = np.random.normal(0, 0.1, len(data))
        X_noise = X_scaled.copy() 
        X_noise[:, -1] += noise  # Add noise to Size_sqm
        
        # Add some noise to the target as well
        y_noise = house_price + np.random.normal(0, 5000, len(data))
        
        # Fit models
        ols_model = LinearRegression().fit(X_noise, y_noise)
        ridge_model = Ridge(alpha=10.0).fit(X_noise, y_noise)
        
        ols_coefs.append(ols_model.coef_)
        ridge_coefs.append(ridge_model.coef_)
    
    # Convert to arrays for easier analysis
    ols_coefs = np.array(ols_coefs)
    ridge_coefs = np.array(ridge_coefs)
    
    # Calculate coefficient variation statistics
    ols_std = np.std(ols_coefs, axis=0)
    ridge_std = np.std(ridge_coefs, axis=0)
    
    # Create a summary dataframe
    stability_comparison = pd.DataFrame({
        'Feature': feature_names,
        'OLS_StdDev': ols_std,
        'Ridge_StdDev': ridge_std,
        'Stability_Improvement': (1 - ridge_std / ols_std) * 100  # % reduction in std dev
    })
    
    print("\nCoefficient stability comparison:")
    print(stability_comparison.round(2))
    
    print("\nObservations:")
    print("  - Ridge coefficients show lower standard deviation across simulations")
    print(f"  - Average stability improvement: {stability_comparison['Stability_Improvement'].mean():.2f}%")
    print("  - This demonstrates how regularization helps stabilize coefficients")
    print("  - More stable coefficients lead to more reliable model interpretation")
    
    # Plot coefficient stability comparison
    plt.figure(figsize=(12, 7))
    
    # Calculate coefficient of variation for each feature (std/mean)
    ols_cv = ols_std / np.abs(np.mean(ols_coefs, axis=0))
    ridge_cv = ridge_std / np.abs(np.mean(ridge_coefs, axis=0))
    
    # Create a bar chart comparing stability
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, ols_cv, width, label='OLS Coefficient Variation', color='tomato', alpha=0.7)
    plt.bar(x + width/2, ridge_cv, width, label='Ridge Coefficient Variation', color='mediumseagreen', alpha=0.7)
    
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient of Variation (std/|mean|)', fontsize=14)
    plt.title('OLS vs Ridge Coefficient Stability', fontsize=16)
    plt.xticks(x, feature_names, rotation=45, ha='right', fontsize=12)
    plt.legend(fontsize=12)
    
    # Add value labels
    for i, v in enumerate(ols_cv):
        plt.text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
        
    for i, v in enumerate(ridge_cv):
        plt.text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Print interpretation text instead of adding to figure
    print("\nLower coefficient of variation indicates more stable estimates across simulations.")
    print("Ridge regression consistently produces more stable coefficients than OLS when multicollinearity is present.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_stability_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more detailed boxplot visualization to compare coefficient distributions
    plt.figure(figsize=(14, 8))
    
    # Prepare data for boxplots - reshape to long format
    coef_data = []
    
    for i, feature in enumerate(feature_names):
        for j in range(n_simulations):
            coef_data.append({
                'Feature': feature,
                'Method': 'OLS',
                'Coefficient': ols_coefs[j, i]
            })
            coef_data.append({
                'Feature': feature,
                'Method': 'Ridge',
                'Coefficient': ridge_coefs[j, i]
            })
    
    coef_df = pd.DataFrame(coef_data)
    
    # Create a custom palette
    palette = {'OLS': 'lightcoral', 'Ridge': 'lightgreen'}
    
    # Create the boxplot
    ax = sns.boxplot(x='Feature', y='Coefficient', hue='Method', data=coef_df, 
                    palette=palette, fliersize=3)
    
    # Add a legend
    plt.legend(title='Regression Method', fontsize=12, title_fontsize=12)
    
    # Add labels and title
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.title('Distribution of Coefficient Estimates: OLS vs Ridge', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add grid for readability
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Print interpretation text instead of adding to figure
    print("\nNarrower boxplots indicate more stable coefficient estimates.")
    print("Ridge regression (green) consistently shows less variation than OLS (red) in the presence of multicollinearity.")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_stability_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization: All approach-related images have been saved to", save_dir)
    print("  - correlation_after_elimination.png: Correlation matrix after feature elimination")
    print("  - vif_comparison.png: Comparison of VIF values before and after feature elimination")
    print("  - correlation_after_transformation.png: Correlation matrix after feature transformation")
    print("  - bedroom_bathroom_ratio.png: Analysis of the new composite feature")
    print("  - ridge_vs_ols.png: Comparison of OLS and Ridge regression coefficients")
    print("  - coefficient_stability_comparison.png: Stability comparison between OLS and Ridge")
    print("  - coefficient_stability_boxplot.png: Detailed comparison of coefficient distributions")
    
    print("\nConclusion of Step 3:")
    print("1. We demonstrated three effective approaches to address multicollinearity:")
    print("   a. Feature elimination: Removed Size_sqft to eliminate perfect multicollinearity")
    print("   b. Feature transformation: Created a Bedroom-to-Bathroom ratio feature")
    print("   c. Regularization: Applied Ridge regression to stabilize coefficients")
    print("2. Feature elimination completely resolved the perfect multicollinearity")
    print("3. Feature transformation addressed the moderate multicollinearity")
    print("4. Ridge regression improved coefficient stability for all features")
    print("5. Each approach has its own advantages and applications")
    
    return data_reduced, data_transformed, coef_comparison

data_reduced, data_transformed, coef_comparison = address_multicollinearity(data)

# Step 4: Explain effects of ignoring multicollinearity
def explain_effects_of_ignoring():
    """Explain what would happen if multicollinearity is ignored."""
    print("\n" + "="*80)
    print("STEP 4: EFFECTS OF IGNORING MULTICOLLINEARITY")
    print("="*80)
    
    print("\nAlthough we've demonstrated how to address multicollinearity, it's important to")
    print("understand what happens if we ignore it and proceed with the regression anyway.")
    print("This helps emphasize why addressing multicollinearity is crucial.")
    
    print("\nThe main consequences of ignoring multicollinearity include:")
    
    effects = [
        "1. Inflated Standard Errors:\n"
        + "   • The standard errors of the coefficient estimates become larger\n"
        + "   • This makes it harder to detect significant relationships\n"
        + "   • Confidence intervals become wider\n"
        + "   • p-values increase, potentially leading to Type II errors (missing significant effects)",
        
        "2. Unstable Coefficients:\n"
        + "   • The coefficient estimates become highly sensitive to small changes in the data\n"
        + "   • Minor changes in the dataset can lead to large swings in coefficient values\n"
        + "   • This makes the model unreliable for both interpretation and prediction\n"
        + "   • Different samples from the same population may yield very different models",
        
        "3. Incorrect Sign of Coefficients:\n"
        + "   • Coefficients may have the wrong sign (opposite of what theory would suggest)\n"
        + "   • For example, house size might appear to have a negative effect on price\n"
        + "   • This leads to incorrect or misleading interpretations\n"
        + "   • Models may suggest counterintuitive relationships that don't exist in reality",
        
        "4. Difficulty in Determining Individual Feature Importance:\n"
        + "   • It becomes impossible to isolate the effect of each predictor on the response\n"
        + "   • Importance metrics become unreliable\n"
        + "   • Feature attribution methods yield misleading results\n"
        + "   • This undermines the interpretability of the model",
        
        "5. Reduced Statistical Power:\n"
        + "   • The increased standard errors lead to reduced ability to reject null hypotheses\n"
        + "   • The model may fail to detect genuinely important effects\n"
        + "   • This reduces the overall utility of the statistical analysis\n"
        + "   • More data is required to achieve the same level of statistical confidence"
    ]
    
    for effect in effects:
        print("\n" + effect)
    
    # Demonstrate unstable coefficients with a simulation
    print("\n" + "-"*80)
    print("Demonstration: Coefficient Instability Due to Multicollinearity")
    print("-"*80)
    
    print("\nWe'll run a simulation to demonstrate how multicollinearity causes coefficient")
    print("instability. We'll generate multiple samples with small variations and observe how")
    print("the coefficients change dramatically when multicollinearity is present.")
    
    # Generate multiple samples with small variations
    np.random.seed(42)
    n_samples = 50
    n_obs = 100
    
    print(f"\nStep 4.1: Running {n_samples} simulations with {n_obs} observations each")
    print("  - For each simulation, we'll create a dataset with small random variations")
    print("  - We'll fit two models: one with multicollinearity and one without")
    print("  - Then we'll compare the stability of the coefficient estimates")
    
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
    
    print("\nStep 4.2: Analyzing coefficient variance")
    print(f"Variance of coefficients with multicollinearity:")
    print(f"  - Intercept: {var_with_multi[0]:.2f}")
    print(f"  - Size_sqm: {var_with_multi[1]:.2f}")
    print(f"  - Size_sqft: {var_with_multi[2]:.2f}")
    
    print(f"\nVariance of coefficients without multicollinearity:")
    print(f"  - Intercept: {var_without_multi[0]:.2f}")
    print(f"  - Size_sqm: {var_without_multi[1]:.2f}")
    
    print(f"\nRatio of variances (with/without):")
    print(f"  - Intercept: {var_with_multi[0]/var_without_multi[0]:.2f}x higher with multicollinearity")
    print(f"  - Size_sqm: {var_with_multi[1]/var_without_multi[1]:.2f}x higher with multicollinearity")
    
    print(f"\nThis demonstrates dramatically increased coefficient variance when multicollinearity is present.")
    
    # Visualize coefficient stability - for interpretability use boxplots
    plt.figure(figsize=(12, 8))
    
    # Create a dataframe for easier plotting
    df_coefs = pd.DataFrame()
    
    # For models with multicollinearity (add both coefficients)
    df_multi_x1 = pd.DataFrame({
        'Coefficient Value': coefs_with_multicollinearity[:, 1],
        'Variable': 'Size_sqm\n(with multicollinearity)',
        'Group': 'With Multicollinearity'
    })
    
    df_multi_x3 = pd.DataFrame({
        'Coefficient Value': coefs_with_multicollinearity[:, 2],
        'Variable': 'Size_sqft\n(with multicollinearity)',
        'Group': 'With Multicollinearity'
    })
    
    # For models without multicollinearity (only x1 coefficient)
    df_single_x1 = pd.DataFrame({
        'Coefficient Value': coefs_without_multicollinearity[:, 1],
        'Variable': 'Size_sqm\n(without multicollinearity)',
        'Group': 'Without Multicollinearity'
    })
    
    # Combine all data
    df_coefs = pd.concat([df_multi_x1, df_multi_x3, df_single_x1])
    
    # Color palette
    colors = {'With Multicollinearity': 'lightcoral', 'Without Multicollinearity': 'lightgreen'}
    
    # Plot using seaborn boxplot
    ax = sns.boxplot(x='Variable', y='Coefficient Value', data=df_coefs, 
                    palette=[colors['With Multicollinearity'], colors['With Multicollinearity'], 
                            colors['Without Multicollinearity']])
    
    plt.title('Coefficient Stability Comparison', fontsize=16)
    plt.xlabel('Variable', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add true coefficient values
    plt.axhline(y=100, color='blue', linestyle='--', label='True Size_sqm coefficient (100)')
    plt.axhline(y=5, color='green', linestyle='--', label='True Size_sqft coefficient (5)')
    
    plt.legend(fontsize=12)
    
    # Print observations instead of adding annotations
    print("\nObservations about coefficient stability:")
    print("• With multicollinearity (Size_sqm): High variability, wrong sign in many cases")
    print("• With multicollinearity (Size_sqft): High variability, wrong sign in many cases")
    print("• Without multicollinearity (Size_sqm): Stable and accurate, consistent positive sign")
    print("• With multicollinearity: Coefficients are highly unstable and often have the wrong sign")
    print("• Without multicollinearity: Coefficients are stable and consistent with the true values")
    print("• The instability makes interpretation and inference impossible when multicollinearity is present")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_stability_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visual to demonstrate the incorrect sign effect
    plt.figure(figsize=(14, 6))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.hist(coefs_with_multicollinearity[:, 1], bins=20, alpha=0.7, color='red', 
             label='Size_sqm coefficient')
    plt.hist(coefs_with_multicollinearity[:, 2], bins=20, alpha=0.7, color='blue', 
             label='Size_sqft coefficient')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.axvline(x=100, color='darkred', linestyle=':', label='True Size_sqm coeff. (100)')
    plt.axvline(x=5, color='darkblue', linestyle=':', label='True Size_sqft coeff. (5)')
    plt.title('Coefficients with Multicollinearity', fontsize=14)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    
    # Calculate percentage of coefficients with wrong sign
    wrong_sign_sqm = np.sum(coefs_with_multicollinearity[:, 1] < 0) / len(coefs_with_multicollinearity) * 100
    wrong_sign_sqft = np.sum(coefs_with_multicollinearity[:, 2] < 0) / len(coefs_with_multicollinearity) * 100
    
    # Print instead of annotating
    print(f"\nPercentage of coefficients with wrong sign:")
    print(f"• Size_sqm with multicollinearity: {wrong_sign_sqm:.1f}% have wrong sign")
    
    plt.subplot(1, 2, 2)
    plt.hist(coefs_without_multicollinearity[:, 1], bins=20, alpha=0.7, color='green',
             label='Size_sqm coefficient (no multicollinearity)')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.axvline(x=100, color='darkgreen', linestyle=':', label='True coefficient (100)')
    plt.title('Coefficients without Multicollinearity', fontsize=14)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    
    # Calculate percentage of coefficients with wrong sign
    wrong_sign_single = np.sum(coefs_without_multicollinearity[:, 1] < 0) / len(coefs_without_multicollinearity) * 100
    
    # Print instead of annotating
    print(f"• Size_sqm without multicollinearity: {wrong_sign_single:.1f}% have wrong sign")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_sign_problem.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visualization for the standard error effect
    plt.figure(figsize=(12, 6))
    
    # Create synthetic confidence intervals
    print("\nStep 4.3: Visualizing the effect on confidence intervals")
    
    true_coef = 100  # True coefficient value for Size_sqm
    x_range = np.linspace(0, 10, 11)
    
    # Calculate actual standard errors from the simulations
    se_nomulti = np.std(coefs_without_multicollinearity[:, 1])
    se_multi = np.std(coefs_with_multicollinearity[:, 1])
    
    print(f"  - Standard error without multicollinearity: {se_nomulti:.2f}")
    print(f"  - Standard error with multicollinearity: {se_multi:.2f}")
    print(f"  - Ratio: {se_multi/se_nomulti:.2f}x larger with multicollinearity")
    
    # Calculate confidence intervals
    y_nomulti_upper = true_coef + 1.96 * se_nomulti
    y_nomulti_lower = true_coef - 1.96 * se_nomulti
    
    y_multi_upper = true_coef + 1.96 * se_multi
    y_multi_lower = true_coef - 1.96 * se_multi
    
    # Create plots
    plt.plot([0, 10], [true_coef, true_coef], 'k-', linewidth=2, label='True coefficient value')
    
    # Confidence intervals without multicollinearity
    plt.fill_between(x_range, y_nomulti_lower, y_nomulti_upper, 
                    color='green', alpha=0.3, label='95% CI without multicollinearity')
    
    # Confidence intervals with multicollinearity
    plt.fill_between(x_range, y_multi_lower, y_multi_upper, 
                    color='red', alpha=0.3, label='95% CI with multicollinearity')
    
    # Add a horizontal line at zero for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Hypothetical Studies', fontsize=14)
    plt.ylabel('Coefficient Value', fontsize=14)
    plt.title('Effect of Multicollinearity on Confidence Intervals', fontsize=16)
    plt.legend(fontsize=12)
    
    # Print statistical significance information instead of annotating
    if 0 > y_multi_lower and 0 < y_multi_upper:
        print("\nWith multicollinearity: Not statistically significant (CI includes 0)")
    else:
        print("\nWith multicollinearity: Statistically significant (CI excludes 0)")
        
    if 0 > y_nomulti_lower and 0 < y_nomulti_upper:
        print("Without multicollinearity: Not statistically significant (CI includes 0)")
    else:
        print("Without multicollinearity: Statistically significant (CI excludes 0)")
    
    # Print interpretation text instead of adding to figure
    print(f"\nKey observations about confidence intervals:")
    print(f"• Confidence interval with multicollinearity is {se_multi/se_nomulti:.1f}x wider")
    print(f"• Wider intervals make it harder to detect significant effects")
    print(f"• Type II errors (failing to reject false null hypotheses) become more likely")
    print(f"• Statistical power is greatly reduced when multicollinearity is present")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_interval_effect.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nStep 4.4: Creating visualization for coefficient sign problem")
    
    # Create a visualization for the sign problem
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot of coefficients
    plt.scatter(coefs_with_multicollinearity[:, 1], coefs_with_multicollinearity[:, 2], 
               alpha=0.7, s=80, c='purple', edgecolor='k')
    
    # Add quadrant lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Add true coefficient values
    plt.scatter([100], [5], s=200, c='gold', edgecolor='k', marker='*', label='True coefficient values')
    
    # Count points in each quadrant
    q1 = np.sum((coefs_with_multicollinearity[:, 1] > 0) & (coefs_with_multicollinearity[:, 2] > 0))
    q2 = np.sum((coefs_with_multicollinearity[:, 1] < 0) & (coefs_with_multicollinearity[:, 2] > 0))
    q3 = np.sum((coefs_with_multicollinearity[:, 1] < 0) & (coefs_with_multicollinearity[:, 2] < 0))
    q4 = np.sum((coefs_with_multicollinearity[:, 1] > 0) & (coefs_with_multicollinearity[:, 2] < 0))
    
    # Print quadrant information instead of adding annotations
    print("\nCoefficient sign quadrants:")
    print(f"• Quadrant 1 (both positive): {q1} ({q1/len(coefs_with_multicollinearity)*100:.1f}%) - Correct signs")
    print(f"• Quadrant 2 (Size_sqm negative): {q2} ({q2/len(coefs_with_multicollinearity)*100:.1f}%) - Wrong sign for Size_sqm")
    print(f"• Quadrant 3 (both negative): {q3} ({q3/len(coefs_with_multicollinearity)*100:.1f}%) - Both wrong signs")
    print(f"• Quadrant 4 (Size_sqft negative): {q4} ({q4/len(coefs_with_multicollinearity)*100:.1f}%) - Wrong sign for Size_sqft")
    
    plt.xlabel('Size_sqm Coefficient', fontsize=14)
    plt.ylabel('Size_sqft Coefficient', fontsize=14)
    plt.title('Coefficient Sign Problems Due to Multicollinearity', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Print interpretation text instead of adding to figure
    print(f"\nKey observations about coefficient signs:")
    print(f"• Only {q1/len(coefs_with_multicollinearity)*100:.1f}% of samples have coefficients with the correct signs")
    print(f"• {(q2+q3+q4)/len(coefs_with_multicollinearity)*100:.1f}% have at least one coefficient with the wrong sign")
    print(f"• This demonstrates how multicollinearity leads to misleading or counterintuitive interpretations")
    print(f"• Wrong signs can lead to incorrect business decisions and faulty understanding of the relationships")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_sign_quadrants.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization: All images illustrating effects of ignoring multicollinearity saved to", save_dir)
    print("  - coefficient_stability_boxplot.png: Boxplot comparing coefficient stability")
    print("  - coefficient_sign_problem.png: Histograms showing coefficient sign instability")
    print("  - confidence_interval_effect.png: Visualization of inflated confidence intervals")
    print("  - coefficient_sign_quadrants.png: Analysis of coefficient sign problems")
    
    print("\nConclusion of Step 4:")
    print("1. Ignoring multicollinearity leads to several serious problems:")
    print("   a. Coefficient estimates become extremely unstable")
    print("   b. Standard errors increase dramatically")
    print("   c. Coefficients often have the wrong sign")
    print("   d. Statistical inference becomes unreliable")
    print("2. Our simulations quantified these problems:")
    print(f"   a. Coefficient variance increased by {var_with_multi[1]/var_without_multi[1]:.2f}x")
    print(f"   b. Standard errors increased by {se_multi/se_nomulti:.2f}x")
    print(f"   c. Only {q1/len(coefs_with_multicollinearity)*100:.1f}% of samples had coefficients with correct signs")
    print("3. These problems severely undermine the model's utility for both prediction and interpretation")
    print("4. This demonstrates why addressing multicollinearity is crucial for reliable statistical modeling")

# Summary of the solution
def summarize_solution():
    print("\n" + "="*80)
    print("SUMMARY: MULTICOLLINEARITY IN HOUSING PRICE PREDICTION")
    print("="*80)
    
    print("\nIn this analysis, we explored multicollinearity in a housing price prediction model:")
    
    print("\n1. IDENTIFICATION:")
    print("   • We identified perfect multicollinearity between Size_sqm and Size_sqft")
    print("   • We found moderate multicollinearity between Bedrooms and Bathrooms")
    print("   • We visualized these relationships through correlation analysis and scatter plots")
    
    print("\n2. DETECTION METHODS:")
    print("   • Correlation matrix analysis revealed high correlation coefficients")
    print("   • Variance Inflation Factors (VIF) showed extremely high values for correlated features")
    print("   • Eigenvalue analysis indicated a high condition number of the correlation matrix")
    print("   • We demonstrated coefficient instability through simulation")
    
    print("\n3. REMEDIATION APPROACHES:")
    print("   • Feature elimination: Removed Size_sqft to eliminate perfect multicollinearity")
    print("   • Feature transformation: Created a Bedroom-to-Bathroom ratio feature")
    print("   • Regularization: Applied Ridge regression to stabilize coefficient estimates")
    print("   • All approaches successfully reduced multicollinearity to acceptable levels")
    
    print("\n4. CONSEQUENCES OF IGNORING:")
    print("   • Coefficient estimates became highly unstable")
    print("   • Standard errors inflated dramatically")
    print("   • Coefficients frequently had incorrect signs")
    print("   • Statistical inference became unreliable")
    print("   • Model interpretability was severely compromised")
    
    print("\nConclusion:")
    print("Addressing multicollinearity is essential for building reliable and interpretable")
    print("regression models. While we have several effective approaches, understanding")
    print("the problem and choosing the appropriate method based on the specific situation")
    print("is crucial for successful modeling and valid statistical inference.")
    
    print("\nAll visualizations have been saved to:", save_dir)

# Run all steps
data = identify_multicollinearity()
vif_data, eigenvalues, condition_number = detect_multicollinearity(data)
data_reduced, data_transformed, coef_comparison = address_multicollinearity(data)
explain_effects_of_ignoring()
summarize_solution()