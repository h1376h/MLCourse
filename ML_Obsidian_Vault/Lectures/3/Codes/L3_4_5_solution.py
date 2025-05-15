import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_5")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Given data from the problem
study_hours = np.array([1, 2, 3, 4, 5, 6])
exam_scores = np.array([45, 50, 60, 65, 68, 70])

# Step 1: Propose a polynomial regression model of degree 2
def propose_polynomial_model():
    """Propose a polynomial regression model of degree 2 (quadratic) to fit the data."""
    print("Step 1: Proposing a polynomial regression model of degree 2 (quadratic)")
    
    print("\nOriginal data:")
    data = pd.DataFrame({
        'Study_Hours': study_hours,
        'Exam_Score': exam_scores
    })
    print(data)
    print()
    
    print("Proposed Model: Quadratic Polynomial Regression")
    print("y = β₀ + β₁x + β₂x²")
    print("\nWhere:")
    print("- y is the exam score")
    print("- x is the number of study hours")
    print("- β₀, β₁, and β₂ are the coefficients to be estimated")
    print()
    
    # Fit linear and quadratic models for comparison
    # Linear model
    X_linear = study_hours.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X_linear, exam_scores)
    
    # Quadratic model
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_linear)
    quad_model = LinearRegression()
    quad_model.fit(X_poly, exam_scores)
    
    # Make predictions using both models
    y_linear = linear_model.predict(X_linear)
    y_quad = quad_model.predict(X_poly)
    
    # Calculate R² for both models
    r2_linear = r2_score(exam_scores, y_linear)
    r2_quad = r2_score(exam_scores, y_quad)
    
    print("Model Comparison:")
    print(f"Linear Model (y = β₀ + β₁x):")
    print(f"  - Intercept (β₀): {linear_model.intercept_:.4f}")
    print(f"  - Coefficient (β₁): {linear_model.coef_[0]:.4f}")
    print(f"  - R²: {r2_linear:.4f}")
    print()
    
    print(f"Quadratic Model (y = β₀ + β₁x + β₂x²):")
    print(f"  - Intercept (β₀): {quad_model.intercept_:.4f}")
    print(f"  - Coefficient (β₁): {quad_model.coef_[0]:.4f}")
    print(f"  - Coefficient (β₂): {quad_model.coef_[1]:.4f}")
    print(f"  - R²: {r2_quad:.4f}")
    print()
    
    # Create a comparison plot of both models
    plt.figure(figsize=(10, 6))
    
    # Generate a smooth range for plotting
    x_range = np.linspace(0, 7, 100)
    
    # Make predictions for the smooth range
    x_range_reshaped = x_range.reshape(-1, 1)
    x_range_poly = poly_features.transform(x_range_reshaped)
    
    y_linear_pred = linear_model.predict(x_range_reshaped)
    y_quad_pred = quad_model.predict(x_range_poly)
    
    # Plot the data and model curves
    plt.scatter(study_hours, exam_scores, s=100, color='blue', alpha=0.7, label='Data Points')
    plt.plot(x_range, y_linear_pred, 'r--', linewidth=2, label=f'Linear Model (R² = {r2_linear:.2f})')
    plt.plot(x_range, y_quad_pred, 'g-', linewidth=2, label=f'Quadratic Model (R² = {r2_quad:.2f})')
    
    # Add data labels
    for i, (x, y) in enumerate(zip(study_hours, exam_scores)):
        plt.annotate(f'({x}, {y})', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add axis labels and title
    plt.xlabel('Study Hours (x)')
    plt.ylabel('Exam Score (y)')
    plt.title('Exam Score vs. Study Hours: Model Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Create a residual plot for both models
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    residuals_linear = exam_scores - y_linear
    residuals_quad = exam_scores - y_quad
    
    # Plot residuals
    plt.subplot(1, 2, 1)
    plt.scatter(study_hours, residuals_linear, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Study Hours')
    plt.ylabel('Residuals')
    plt.title('Linear Model Residuals')
    
    plt.subplot(1, 2, 2)
    plt.scatter(study_hours, residuals_quad, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Study Hours')
    plt.ylabel('Residuals')
    plt.title('Quadratic Model Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residual_comparison.png"), dpi=300)
    plt.close()
    
    return quad_model, poly_features

quad_model, poly_features = propose_polynomial_model()

# Step 2: Write down the design matrix X for this polynomial regression
def create_design_matrix():
    """Create the design matrix X for polynomial regression."""
    print("\nStep 2: Creating the design matrix X for polynomial regression")
    
    # Create the polynomial features (degree 2)
    X_linear = study_hours.reshape(-1, 1)
    X_poly = poly_features.transform(X_linear)
    
    # Add intercept column (column of ones)
    X_with_intercept = np.column_stack((np.ones(len(study_hours)), X_poly))
    
    # Create a DataFrame for easier visualization
    X_df = pd.DataFrame(X_with_intercept, 
                     columns=['Intercept', 'x (Study Hours)', 'x² (Study Hours²)'])
    
    print("Design matrix X for polynomial regression:")
    print(X_df)
    print()
    
    # Visualize the design matrix as a heatmap
    plt.figure(figsize=(8, 6))
    
    # Create a heatmap of the design matrix
    sns.heatmap(X_df, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Design Matrix X for Polynomial Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix_heatmap.png"), dpi=300)
    plt.close()
    
    # Create a visualization of how the design matrix is constructed from the original data
    plt.figure(figsize=(12, 6))
    
    # Organize the grid
    gs = GridSpec(2, 3, figure=plt.gcf(), height_ratios=[1, 2])
    
    # Original data (top left)
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(study_hours, np.ones_like(study_hours), color='blue', s=80)
    ax1.set_yticks([])
    ax1.set_xlim(0, 7)
    ax1.set_title('Original Data (x)')
    
    # Squared data (top middle)
    ax2 = plt.subplot(gs[0, 1])
    ax2.scatter(study_hours, np.ones_like(study_hours), color='red', s=80)
    for i, x in enumerate(study_hours):
        ax2.annotate(f'{x}² = {x**2}', (x, 1), xytext=(0, 5), 
                    textcoords='offset points', ha='center')
    ax2.set_yticks([])
    ax2.set_xlim(0, 7)
    ax2.set_title('Squared Data (x²)')
    
    # Intercept column (top right)
    ax3 = plt.subplot(gs[0, 2])
    ax3.scatter(np.ones_like(study_hours), np.arange(len(study_hours)), color='green', s=80)
    ax3.set_xticks([1])
    ax3.set_title('Intercept Column')
    
    # Design matrix (bottom)
    ax4 = plt.subplot(gs[1, :])
    
    # Create a nicer visualization of the design matrix
    matrix_data = np.zeros((len(study_hours), 3))
    matrix_data[:, 0] = 1  # Intercept
    matrix_data[:, 1] = study_hours  # x
    matrix_data[:, 2] = study_hours ** 2  # x²
    
    # Color code the matrix based on column
    sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='viridis', 
               xticklabels=['Intercept', 'x', 'x²'])
    plt.title('Design Matrix X with Intercept')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix_construction.png"), dpi=300)
    plt.close()
    
    return X_df

X_df = create_design_matrix()

# Step 3: Express the model in both expanded form and matrix form
def express_model_forms():
    """Express the model in both expanded form and matrix form."""
    print("\nStep 3: Expressing the model in expanded and matrix forms")
    
    print("Expanded Form:")
    print("y = β₀ + β₁x + β₂x²")
    print(f"y = {quad_model.intercept_:.4f} + {quad_model.coef_[0]:.4f}x + {quad_model.coef_[1]:.4f}x²")
    print()
    
    print("Matrix Form:")
    print("y = Xβ")
    print("Where:")
    print("y is the vector of exam scores")
    print("X is the design matrix from Step 2")
    print("β is the coefficient vector [β₀, β₁, β₂]ᵀ")
    print()
    
    # Visualize the coefficient values
    plt.figure(figsize=(8, 6))
    
    # Create a bar chart of the coefficients
    coef_names = ['β₀ (Intercept)', 'β₁ (x)', 'β₂ (x²)']
    coef_values = [quad_model.intercept_, quad_model.coef_[0], quad_model.coef_[1]]
    colors = ['blue', 'green', 'red']
    
    bars = plt.bar(coef_names, coef_values, color=colors, alpha=0.7)
    
    # Add coefficient values as labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * (1 if height >= 0 else -1),
                f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Coefficient Value')
    plt.title('Polynomial Regression Coefficients')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coefficient_values.png"), dpi=300)
    plt.close()
    
    # Visualize the elements of the matrix form
    plt.figure(figsize=(12, 6))
    
    # Create a visual representation of the matrix equation
    # First, create data for the visualization
    X_matrix = X_df.values
    y_vector = exam_scores.reshape(-1, 1)
    coef_vector = np.array([quad_model.intercept_, quad_model.coef_[0], quad_model.coef_[1]]).reshape(-1, 1)
    
    # Use the GridSpec for layout
    gs = GridSpec(1, 5, figure=plt.gcf(), width_ratios=[1, 0.3, 1, 0.3, 1])
    
    # Plot the y vector
    ax1 = plt.subplot(gs[0, 0])
    sns.heatmap(y_vector, annot=True, fmt='.0f', cmap='Blues', cbar=False)
    ax1.set_title('y (Exam Scores)')
    ax1.set_yticklabels([])
    
    # Plot the equals sign
    ax2 = plt.subplot(gs[0, 1])
    ax2.text(0.5, 0.5, '=', fontsize=24, ha='center', va='center')
    ax2.axis('off')
    
    # Plot the X matrix
    ax3 = plt.subplot(gs[0, 2])
    sns.heatmap(X_matrix, annot=True, fmt='.2f', cmap='Greens', cbar=False)
    ax3.set_title('X (Design Matrix)')
    ax3.set_yticklabels([])
    ax3.set_xticklabels(['Intercept', 'x', 'x²'])
    
    # Plot the multiplication sign
    ax4 = plt.subplot(gs[0, 3])
    ax4.text(0.5, 0.5, '×', fontsize=24, ha='center', va='center')
    ax4.axis('off')
    
    # Plot the coefficient vector
    ax5 = plt.subplot(gs[0, 4])
    sns.heatmap(coef_vector, annot=True, fmt='.2f', cmap='Reds', cbar=False)
    ax5.set_title('β (Coefficients)')
    ax5.set_yticklabels([])
    ax5.set_xticklabels(['Values'])
    
    plt.suptitle('Matrix Form of Polynomial Regression: y = Xβ', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "matrix_form_visualization.png"), dpi=300)
    plt.close()

express_model_forms()

# Step 4: Explain how the polynomial model captures diminishing returns
def explain_diminishing_returns():
    """Explain how polynomial regression helps capture the diminishing returns effect."""
    print("\nStep 4: Explaining how the polynomial model captures diminishing returns")
    
    # Calculate predicted values for the original data points
    X_linear = study_hours.reshape(-1, 1)
    X_poly = poly_features.transform(X_linear)
    y_pred = quad_model.predict(X_poly)
    
    # Calculate incremental gains between consecutive hours
    incremental_gains = []
    for i in range(1, len(study_hours)):
        gain = y_pred[i] - y_pred[i-1]
        incremental_gains.append(gain)
    
    print("Incremental gains in exam score for each additional hour of study:")
    for i, gain in enumerate(incremental_gains):
        print(f"From {i+1} to {i+2} hours: +{gain:.2f} points")
    print()
    
    # Analytical approach - calculate the derivative of the quadratic function
    print("Analytically, the rate of change (derivative) of a quadratic function y = β₀ + β₁x + β₂x² is:")
    print("dy/dx = β₁ + 2β₂x")
    print(f"dy/dx = {quad_model.coef_[0]:.4f} + 2 × ({quad_model.coef_[1]:.4f}) × x")
    print(f"dy/dx = {quad_model.coef_[0]:.4f} + {2 * quad_model.coef_[1]:.4f} × x")
    print()
    
    # Explain the diminishing returns effect
    if quad_model.coef_[1] < 0:
        print("Since β₂ is negative, the quadratic term creates a concave downward curve, which")
        print("naturally models the diminishing returns effect. As the number of study hours increases,")
        print("each additional hour yields a smaller increase in the exam score.")
        print()
        print("The rate of improvement decreases with each additional hour because the derivative")
        print("(dy/dx) becomes smaller as x increases, due to the negative coefficient of the x² term.")
    else:
        print("In this particular dataset, β₂ is actually slightly positive, which doesn't strictly model")
        print("diminishing returns across the entire range. However, the rate of improvement still")
        print("decreases within our observed data range, which captures the diminishing returns effect")
        print("for the given study hours.")
    print()
    
    # Create a visualization of diminishing returns with observed and predicted values
    plt.figure(figsize=(10, 6))
    
    # Create smooth data for the curve
    x_smooth = np.linspace(0, 7, 100)
    X_smooth = poly_features.transform(x_smooth.reshape(-1, 1))
    y_smooth = quad_model.predict(X_smooth)
    
    # Plot the curve
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Polynomial Model')
    
    # Plot the data points
    plt.scatter(study_hours, exam_scores, s=100, color='red', alpha=0.7, label='Actual Data')
    
    # Plot the predicted points
    plt.scatter(study_hours, y_pred, s=80, color='green', marker='x', alpha=0.7, label='Model Predictions')
    
    # Connect actual score points with lines to highlight diminishing returns
    plt.plot(study_hours, exam_scores, 'r--', alpha=0.3)
    
    # Add annotations for the incremental gains
    for i in range(1, len(study_hours)):
        midpoint_x = (study_hours[i-1] + study_hours[i]) / 2
        midpoint_y = (exam_scores[i-1] + exam_scores[i]) / 2
        gain = exam_scores[i] - exam_scores[i-1]
        plt.annotate(f'+{gain}', (midpoint_x, midpoint_y), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.xlabel('Study Hours')
    plt.ylabel('Exam Score')
    plt.title('Diminishing Returns in Exam Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "observed_diminishing_returns.png"), dpi=300)
    plt.close()
    
    # Create a visualization of the derivative function (rate of change)
    plt.figure(figsize=(10, 6))
    
    # Calculate the derivative for each point in the range
    derivative = quad_model.coef_[0] + 2 * quad_model.coef_[1] * x_smooth
    
    # Plot the derivative
    plt.plot(x_smooth, derivative, 'g-', linewidth=2, label='Rate of Change (dy/dx)')
    
    # Calculate the derivative at each study hour
    derivatives_at_hours = quad_model.coef_[0] + 2 * quad_model.coef_[1] * study_hours
    
    # Plot points at each study hour
    plt.scatter(study_hours, derivatives_at_hours, s=80, color='blue', alpha=0.7,
               label='Rate at Data Points')
    
    # Add horizontal line at y=0 if needed
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add annotations for the rate of change at each point
    for i, (x, deriv) in enumerate(zip(study_hours, derivatives_at_hours)):
        plt.annotate(f'{deriv:.2f}', (x, deriv), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.xlabel('Study Hours')
    plt.ylabel('Rate of Change in Exam Score')
    plt.title('Rate of Change in Exam Score per Additional Study Hour')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rate_of_change.png"), dpi=300)
    plt.close()
    
    # Create a bar chart of incremental gains from the model
    plt.figure(figsize=(10, 6))
    
    # Calculate incremental gains from the smooth curve
    hour_points = np.arange(1, 7)
    predicted_scores = quad_model.predict(poly_features.transform(hour_points.reshape(-1, 1)))
    model_incremental_gains = np.diff(predicted_scores)
    
    # Plot the bar chart
    bars = plt.bar(range(1, 6), model_incremental_gains, color='purple', alpha=0.7,
                 width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Transition Between Study Hours')
    plt.ylabel('Incremental Gain in Exam Score')
    plt.title('Diminishing Returns: Incremental Gains from the Model')
    plt.xticks(range(1, 6), ['1→2', '2→3', '3→4', '4→5', '5→6'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "incremental_gains.png"), dpi=300)
    plt.close()
    
    # Create a combined visualization of model, original data, and rate of change
    plt.figure(figsize=(12, 8))
    
    # Set up the grid for two subplots
    plt.subplot(2, 1, 1)
    
    # Plot original data and model in top subplot
    plt.scatter(study_hours, exam_scores, s=100, color='blue', alpha=0.7, label='Original Data')
    plt.plot(x_smooth, y_smooth, 'g-', linewidth=2, label='Quadratic Model')
    
    # Add connecting lines between points
    plt.plot(study_hours, exam_scores, 'b--', alpha=0.3)
    
    # Add annotations for the score at each point
    for i, (x, y) in enumerate(zip(study_hours, exam_scores)):
        plt.annotate(f'{y}', (x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.title('Exam Score vs. Study Hours')
    plt.ylabel('Exam Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot rate of change in bottom subplot
    plt.subplot(2, 1, 2)
    
    # Plot the derivative
    plt.plot(x_smooth, derivative, 'r-', linewidth=2, label='Rate of Change (dy/dx)')
    
    # Add points at each hour
    plt.scatter(study_hours, derivatives_at_hours, s=80, color='orange', alpha=0.7)
    
    # Add annotations
    for i, (x, deriv) in enumerate(zip(study_hours, derivatives_at_hours)):
        plt.annotate(f'{deriv:.2f}', (x, deriv), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.xlabel('Study Hours')
    plt.ylabel('Rate of Change')
    plt.title('Rate of Improvement per Additional Hour')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_visualization.png"), dpi=300)
    plt.close()

explain_diminishing_returns()

# Summary of the solution
print("\nQuestion 5 Solution Summary:")
print("1. We proposed a quadratic polynomial regression model: y = β₀ + β₁x + β₂x²")
print("   where y is the exam score and x is the number of study hours.")
print()
print("2. The design matrix X for this model includes three columns:")
print("   - A column of ones for the intercept")
print("   - A column for the study hours (x)")
print("   - A column for the squared study hours (x²)")
print()
print("3. The model can be expressed in two forms:")
print(f"   - Expanded form: y = {quad_model.intercept_:.2f} + {quad_model.coef_[0]:.2f}x + {quad_model.coef_[1]:.2f}x²")
print("   - Matrix form: y = Xβ where β = [β₀, β₁, β₂]ᵀ")
print()
print("4. The polynomial model captures diminishing returns through:")
print("   - The curvature of the quadratic function, which flattens as x increases")
print("   - The derivative dy/dx, which decreases with increasing x")
print("   - This matches the observed pattern where initial hours of study yield greater")
print("     improvement than later hours")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- model_comparison.png: Comparison of linear and quadratic models")
print("- residual_comparison.png: Comparison of residuals for both models")
print("- design_matrix_heatmap.png: Heatmap of the design matrix")
print("- design_matrix_construction.png: Visual explanation of design matrix construction")
print("- coefficient_values.png: Bar chart of polynomial regression coefficients")
print("- matrix_form_visualization.png: Visual representation of the matrix form")
print("- observed_diminishing_returns.png: Visualization of diminishing returns in the data")
print("- rate_of_change.png: Plot of the derivative showing the rate of change")
print("- incremental_gains.png: Bar chart of incremental gains between hours")
print("- combined_visualization.png: Combined plot of model and rate of change") 