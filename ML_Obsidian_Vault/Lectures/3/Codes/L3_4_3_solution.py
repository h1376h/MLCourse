import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_3")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Step 1: Propose a model with interaction terms between fertilizer and water
def propose_interaction_model():
    """Propose a model with interaction terms between fertilizer and water."""
    print("Step 1: Proposing a multiple regression model with interaction terms")
    
    print("\nModel with interaction between fertilizer and water:")
    print("y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + β₄x₃ + ε")
    print("\nWhere:")
    print("- y is the crop yield (tons/hectare)")
    print("- x₁ is the amount of fertilizer (kg/hectare)")
    print("- x₂ is the amount of water (liters/day)")
    print("- x₃ is the average daily temperature (°C)")
    print("- x₁ × x₂ is the interaction term between fertilizer and water")
    print("- ε is the error term")
    print()
    
    print("Explanation of the interaction term:")
    print("- The interaction term β₃(x₁ × x₂) captures how the effect of fertilizer depends on the water level.")
    print("- A positive β₃ would indicate that the positive effect of fertilizer is amplified with more water.")
    print("- A negative β₃ would indicate that the effect of fertilizer decreases with more water.")
    print()
    
    # Create simulated data to demonstrate the interaction effect
    np.random.seed(42)
    n = 100
    
    # Generate predictor variables
    fertilizer = np.random.uniform(50, 200, n)  # kg/hectare
    water = np.random.uniform(100, 500, n)  # liters/day
    
    # Create interaction effect
    interaction_coef = 0.0001  # small positive effect
    
    # Generate yield with interaction effect (ignoring temperature for now)
    base_yield = 5  # base yield in tons/hectare
    fertilizer_effect = 0.01  # positive effect of fertilizer
    water_effect = 0.005  # positive effect of water
    
    # Calculate yield
    yield_tons = base_yield + fertilizer_effect * fertilizer + water_effect * water + \
                 interaction_coef * fertilizer * water + np.random.normal(0, 0.5, n)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Fertilizer': fertilizer,
        'Water': water,
        'Yield': yield_tons
    })
    
    # Fit a linear model with interaction
    X = np.column_stack((fertilizer, water, fertilizer * water))
    model = LinearRegression()
    model.fit(X, yield_tons)
    
    print("Simulated coefficients from interaction model:")
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    print(f"Fertilizer coefficient (β₁): {model.coef_[0]:.4f}")
    print(f"Water coefficient (β₂): {model.coef_[1]:.4f}")
    print(f"Interaction coefficient (β₃): {model.coef_[2]:.6f}")
    print()
    
    # Visualize the interaction effect with a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up a colormap for better visibility
    norm = mpl.colors.Normalize(vmin=min(yield_tons), vmax=max(yield_tons))
    cmap = plt.cm.viridis
    
    # Create a scatter plot with color based on yield
    sc = ax.scatter(fertilizer, water, yield_tons, c=yield_tons, cmap=cmap, 
                  alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
    
    # Add a colorbar
    fig.colorbar(sc, ax=ax, label='Yield (tons/hectare)')
    
    # Set labels and title
    ax.set_xlabel('Fertilizer (kg/hectare)')
    ax.set_ylabel('Water (liters/day)')
    ax.set_zlabel('Yield (tons/hectare)')
    ax.set_title('3D Visualization of Interaction Effect')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_3d.png"), dpi=300)
    plt.close()
    
    # Create a 2D visualization to show how the effect of fertilizer changes with water
    plt.figure(figsize=(10, 6))
    
    # Select low, medium, and high water levels
    water_levels = [min(water), np.percentile(water, 50), max(water)]
    colors = ['blue', 'green', 'red']
    labels = ['Low Water', 'Medium Water', 'High Water']
    
    fertilizer_range = np.linspace(min(fertilizer), max(fertilizer), 100)
    
    for i, water_level in enumerate(water_levels):
        # Predict yield at this water level for different fertilizer amounts
        predicted_yield = model.intercept_ + model.coef_[0] * fertilizer_range + \
                          model.coef_[1] * water_level + model.coef_[2] * fertilizer_range * water_level
        
        plt.plot(fertilizer_range, predicted_yield, color=colors[i], linewidth=2, 
                label=f'{labels[i]} ({water_level:.0f} L/day)')
    
    # Add data points with transparency and color based on water level
    sc = plt.scatter(fertilizer, yield_tons, c=water, cmap='viridis', alpha=0.4, s=50, edgecolor='k')
    
    plt.colorbar(sc, label='Water Level (liters/day)')
    plt.xlabel('Fertilizer (kg/hectare)')
    plt.ylabel('Predicted Yield (tons/hectare)')
    plt.title('Effect of Fertilizer on Yield at Different Water Levels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_lines.png"), dpi=300)
    plt.close()
    
    # Add a contour plot to visualize interaction in 2D
    plt.figure(figsize=(10, 7))
    
    # Create a grid of points
    fert_grid = np.linspace(min(fertilizer), max(fertilizer), 100)
    water_grid = np.linspace(min(water), max(water), 100)
    fert_mesh, water_mesh = np.meshgrid(fert_grid, water_grid)
    
    # Predict yield for each combination
    yield_pred = (model.intercept_ + 
                 model.coef_[0] * fert_mesh + 
                 model.coef_[1] * water_mesh + 
                 model.coef_[2] * fert_mesh * water_mesh)
    
    # Create contour plot
    contour = plt.contourf(fert_mesh, water_mesh, yield_pred, 20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Predicted Yield (tons/hectare)')
    
    # Add scatter points of actual data
    plt.scatter(fertilizer, water, c=yield_tons, cmap='viridis', 
               edgecolor='k', s=50, alpha=0.7)
    
    plt.xlabel('Fertilizer (kg/hectare)')
    plt.ylabel('Water (liters/day)')
    plt.title('Contour Plot of Predicted Yield with Interaction Effect')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_contour.png"), dpi=300)
    plt.close()
    
    return data

data = propose_interaction_model()

# Step 2: Suggest a feature transformation for temperature
def transform_temperature():
    """Suggest a feature transformation for temperature to model diminishing returns and eventual negative impact."""
    print("\nStep 2: Suggesting a feature transformation for temperature")
    
    print("\nProposed transformation for temperature:")
    print("Add a quadratic term for temperature: x₃² (temperature squared)")
    print("\nUpdated model:")
    print("y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + β₄x₃ + β₅x₃² + ε")
    print()
    
    print("Explanation of the quadratic temperature term:")
    print("- The linear term β₄x₃ captures the initial positive effect of temperature.")
    print("- The quadratic term β₅x₃² (with expected negative β₅) captures the diminishing returns")
    print("  and eventual negative impact of high temperatures.")
    print("- This creates an inverted U-shape relationship between temperature and yield.")
    print("- The optimal temperature can be calculated as -β₄/(2β₅).")
    
    # Create simulated data to demonstrate the quadratic effect of temperature
    np.random.seed(42)
    n = 200
    
    # Generate temperature data
    temperature = np.random.uniform(5, 40, n)  # °C
    
    # Generate yield with quadratic temperature effect
    # Setting coefficients to create a peak around 25°C
    temp_linear_coef = 0.5  # positive effect initially
    temp_quad_coef = -0.01  # negative effect for quadratic term
    
    # Optimal temperature: -temp_linear_coef / (2 * temp_quad_coef) ≈ 25°C
    optimal_temp = -temp_linear_coef / (2 * temp_quad_coef)
    
    # Calculate yield based only on temperature effect
    yield_tons = 5 + temp_linear_coef * temperature + temp_quad_coef * temperature**2 + \
                np.random.normal(0, 0.5, n)
    
    # Create DataFrame
    temp_data = pd.DataFrame({
        'Temperature': temperature,
        'Yield': yield_tons
    })
    
    # Fit a quadratic model
    X = np.column_stack((temperature, temperature**2))
    model = LinearRegression()
    model.fit(X, yield_tons)
    
    print("\nSimulated coefficients from quadratic temperature model:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Temperature linear coefficient (β₄): {model.coef_[0]:.4f}")
    print(f"Temperature quadratic coefficient (β₅): {model.coef_[1]:.4f}")
    
    # Calculate optimal temperature from model
    optimal_temp_model = -model.coef_[0] / (2 * model.coef_[1])
    print(f"Optimal temperature calculated from model: {optimal_temp_model:.2f}°C")
    print()
    
    # Visualize the quadratic relationship with a scatter plot and fitted curve
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the data points
    plt.scatter(temperature, yield_tons, alpha=0.5, color='cornflowerblue', edgecolor='navy')
    
    # Sort temperature values for a smooth line plot
    temp_range = np.linspace(min(temperature), max(temperature), 100)
    predicted_yield = model.intercept_ + model.coef_[0] * temp_range + model.coef_[1] * temp_range**2
    
    # Add the fitted curve
    plt.plot(temp_range, predicted_yield, 'r-', linewidth=2, label='Quadratic Model')
    
    # Mark the optimal temperature
    optimal_yield = model.intercept_ + model.coef_[0] * optimal_temp_model + model.coef_[1] * optimal_temp_model**2
    plt.scatter([optimal_temp_model], [optimal_yield], s=100, c='green', marker='*', 
              edgecolor='k', label=f'Optimal Temperature: {optimal_temp_model:.1f}°C')
    
    # Add a vertical line at the optimal temperature
    plt.axvline(x=optimal_temp_model, color='green', linestyle='--', alpha=0.5)
    
    # Linear regression for comparison
    linear_model = LinearRegression().fit(temperature.reshape(-1, 1), yield_tons)
    linear_predicted = linear_model.intercept_ + linear_model.coef_[0] * temp_range
    plt.plot(temp_range, linear_predicted, 'b--', linewidth=2, label='Linear Model')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Quadratic Relationship Between Temperature and Crop Yield')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_quadratic.png"), dpi=300)
    plt.close()
    
    # Create a residual plot to show how quadratic model fits better
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    linear_pred = linear_model.predict(temperature.reshape(-1, 1))
    linear_residuals = yield_tons - linear_pred
    
    quad_pred = model.predict(X)
    quad_residuals = yield_tons - quad_pred
    
    plt.subplot(1, 2, 1)
    plt.scatter(temperature, linear_residuals, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Residuals')
    plt.title('Linear Model Residuals')
    
    plt.subplot(1, 2, 2)
    plt.scatter(temperature, quad_residuals, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Residuals')
    plt.title('Quadratic Model Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_residuals.png"), dpi=300)
    plt.close()
    
    return temp_data, optimal_temp_model

temp_data, optimal_temp = transform_temperature()

# Step 3: Propose a feature transformation for water to capture diminishing returns
def transform_water():
    """Propose a feature transformation for water to capture the diminishing returns effect."""
    print("\nStep 3: Proposing a feature transformation for water")
    
    print("\nProposed transformation for water:")
    print("Use a logarithmic transformation: log(x₂)")
    print("\nUpdated model:")
    print("y = β₀ + β₁x₁ + β₂log(x₂) + β₃(x₁ × log(x₂)) + β₄x₃ + β₅x₃² + ε")
    print()
    
    print("Explanation of the logarithmic water transformation:")
    print("- The logarithmic transformation log(x₂) captures the diminishing returns of water.")
    print("- Each additional unit of water provides less benefit than the previous unit.")
    print("- The effect follows the law of diminishing returns, common in agricultural contexts.")
    print("- This transformation is often used when the marginal effect decreases as the variable increases.")
    print()
    
    # Create simulated data to demonstrate the logarithmic effect of water
    np.random.seed(42)
    n = 150
    
    # Generate water data
    water = np.random.uniform(50, 800, n)  # liters/day
    
    # Generate yield with logarithmic water effect
    log_water = np.log(water)
    water_coef = 2.0  # coefficient for log(water)
    
    # Calculate yield based only on water effect
    yield_tons = 3 + water_coef * log_water + np.random.normal(0, 0.3, n)
    
    # Create DataFrame
    water_data = pd.DataFrame({
        'Water': water,
        'LogWater': log_water,
        'Yield': yield_tons
    })
    
    # Fit models: linear and logarithmic
    X_linear = water.reshape(-1, 1)
    X_log = log_water.reshape(-1, 1)
    
    model_linear = LinearRegression().fit(X_linear, yield_tons)
    model_log = LinearRegression().fit(X_log, yield_tons)
    
    print("Model comparison for water effect:")
    print(f"Linear model R² = {model_linear.score(X_linear, yield_tons):.4f}")
    print(f"Logarithmic model R² = {model_log.score(X_log, yield_tons):.4f}")
    print()
    
    print("Coefficients:")
    print(f"Linear model - Water coefficient: {model_linear.coef_[0]:.6f}")
    print(f"Logarithmic model - Log(Water) coefficient: {model_log.coef_[0]:.4f}")
    print()
    
    # Visualize the transformation with both models plotted
    plt.figure(figsize=(10, 6))
    
    # Sort for better visualization
    water_sorted = np.sort(water)
    log_water_sorted = np.log(water_sorted)
    
    # Predictions
    linear_pred = model_linear.predict(water_sorted.reshape(-1, 1))
    log_pred = model_log.predict(log_water_sorted.reshape(-1, 1))
    
    # Plot data and models
    plt.scatter(water, yield_tons, alpha=0.5, label='Data Points', color='gray', edgecolor='k')
    plt.plot(water_sorted, linear_pred, 'r-', linewidth=2, label=f'Linear Model (R² = {model_linear.score(X_linear, yield_tons):.3f})')
    plt.plot(water_sorted, log_pred, 'g-', linewidth=2, label=f'Log Model (R² = {model_log.score(X_log, yield_tons):.3f})')
    
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Comparison of Linear vs. Logarithmic Models for Water Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_transformation.png"), dpi=300)
    plt.close()
    
    # Visualize the marginal effects
    plt.figure(figsize=(10, 6))
    
    # Calculate marginal effects for display
    water_range = np.linspace(50, 800, 100)
    
    # For linear model - constant effect
    linear_effect = np.ones_like(water_range) * model_linear.coef_[0]
    
    # For log model - diminishing effect (derivative of β * log(x) is β/x)
    log_effect = model_log.coef_[0] / water_range
    
    plt.plot(water_range, linear_effect, 'r-', linewidth=2, label='Linear Model (Constant Effect)')
    plt.plot(water_range, log_effect, 'g-', linewidth=2, label='Log Model (Diminishing Effect)')
    
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Marginal Effect on Yield')
    plt.title('Marginal Effect of Water on Yield')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_marginal_effect.png"), dpi=300)
    plt.close()
    
    # Create an interaction plot with log-transformed water
    # We'll use fertilizer from the first step
    np.random.seed(42)
    n = 100
    
    # Generate predictor variables
    fertilizer = np.random.uniform(50, 200, n)  # kg/hectare
    water_new = np.random.uniform(50, 800, n)  # liters/day
    log_water_new = np.log(water_new)
    
    # Create interaction effect with log-transformed water
    fert_coef = 0.01
    log_water_coef = 2.0
    interaction_coef = 0.003
    
    # Calculate yield
    yield_tons = 3 + fert_coef * fertilizer + log_water_coef * log_water_new + \
                interaction_coef * fertilizer * log_water_new + np.random.normal(0, 0.3, n)
    
    # Create a DataFrame
    interaction_data = pd.DataFrame({
        'Fertilizer': fertilizer,
        'Water': water_new,
        'LogWater': log_water_new,
        'Yield': yield_tons
    })
    
    # Fit the model with interaction
    X = np.column_stack((fertilizer, log_water_new, fertilizer * log_water_new))
    log_interaction_model = LinearRegression().fit(X, yield_tons)
    
    print("Log-transformed water interaction model coefficients:")
    print(f"Intercept (β₀): {log_interaction_model.intercept_:.4f}")
    print(f"Fertilizer coefficient (β₁): {log_interaction_model.coef_[0]:.4f}")
    print(f"Log(Water) coefficient (β₂): {log_interaction_model.coef_[1]:.4f}")
    print(f"Interaction coefficient (β₃): {log_interaction_model.coef_[2]:.4f}")
    
    # Create a 2D visualization with contour plot
    plt.figure(figsize=(10, 7))
    
    # Create a grid for predictions
    fert_grid = np.linspace(min(fertilizer), max(fertilizer), 100)
    water_grid = np.linspace(min(water_new), max(water_new), 100)
    fert_mesh, water_mesh = np.meshgrid(fert_grid, water_grid)
    
    # Transform water to log for prediction
    log_water_mesh = np.log(water_mesh)
    
    # Predict yield
    yield_pred = (log_interaction_model.intercept_ + 
                 log_interaction_model.coef_[0] * fert_mesh + 
                 log_interaction_model.coef_[1] * log_water_mesh + 
                 log_interaction_model.coef_[2] * fert_mesh * log_water_mesh)
    
    # Create the contour plot
    contour = plt.contourf(fert_mesh, water_mesh, yield_pred, 20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Predicted Yield (tons/hectare)')
    
    # Add points
    plt.scatter(fertilizer, water_new, c=yield_tons, cmap='viridis', 
               edgecolor='k', s=50, alpha=0.7)
    
    plt.xlabel('Fertilizer (kg/hectare)')
    plt.ylabel('Water (liters/day)')
    plt.title('Log-Transformed Water and Fertilizer Interaction Effect')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "log_water_interaction.png"), dpi=300)
    plt.close()
    
    # Show how the model with log-transformed water makes more biological sense
    plt.figure(figsize=(12, 6))
    
    # Generate predictor values
    water_vals = np.linspace(50, 800, 100)
    
    # At different fixed fertilizer levels
    fert_levels = [60, 120, 180]
    colors = ['blue', 'green', 'red']
    
    for i, fert in enumerate(fert_levels):
        # Linear model predictions
        y_linear = model_linear.intercept_ + model_linear.coef_[0] * water_vals
        
        # Log model predictions
        y_log = model_log.intercept_ + model_log.coef_[0] * np.log(water_vals)
        
        plt.subplot(1, 2, 1)
        plt.plot(water_vals, y_linear, color=colors[i], linestyle='-')
        
        plt.subplot(1, 2, 2)
        plt.plot(water_vals, y_log, color=colors[i], linestyle='-', 
                label=f'Fertilizer = {fert} kg/ha')
    
    plt.subplot(1, 2, 1)
    plt.scatter(water_data['Water'], water_data['Yield'], alpha=0.3, color='gray', edgecolor='k')
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Linear Water Effect\n(Unlikely: Unlimited Growth)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(water_data['Water'], water_data['Yield'], alpha=0.3, color='gray', edgecolor='k')
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Logarithmic Water Effect\n(Realistic: Diminishing Returns)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_biological_comparison.png"), dpi=300)
    plt.close()
    
    return water_data, interaction_data

water_data, interaction_data = transform_water()

# Step 4: Write the complete equation for the proposed model
def write_complete_model():
    """Write the complete equation for the proposed model."""
    print("\nStep 4: Complete equation for the proposed model")
    
    print("\nFinal proposed model:")
    print("y = β₀ + β₁x₁ + β₂log(x₂) + β₃(x₁ × log(x₂)) + β₄x₃ + β₅x₃² + ε")
    print()
    
    print("Where:")
    print("- y is the crop yield (tons/hectare)")
    print("- x₁ is the amount of fertilizer (kg/hectare)")
    print("- x₂ is the amount of water (liters/day)")
    print("- x₃ is the average daily temperature (°C)")
    print("- β₀ is the intercept")
    print("- β₁ is the coefficient for fertilizer")
    print("- β₂ is the coefficient for log-transformed water")
    print("- β₃ is the coefficient for the interaction between fertilizer and log-transformed water")
    print("- β₄ is the coefficient for temperature")
    print("- β₅ is the coefficient for temperature squared")
    print("- ε is the error term")
    print()
    
    print("Explanation of the complete model:")
    print("1. The model captures the effect of fertilizer through β₁x₁.")
    print("2. It captures the diminishing returns of water through β₂log(x₂).")
    print("3. It accounts for the interaction between fertilizer and water through β₃(x₁ × log(x₂)).")
    print("4. It models the inverted U-shape relationship of temperature through β₄x₃ + β₅x₃².")
    print("5. This model addresses all three observed effects from the initial analysis.")
    
    # Create synthetic data to visualize the complete model
    np.random.seed(42)
    n = 200
    
    # Generate predictors
    fertilizer = np.random.uniform(50, 200, n)
    water = np.random.uniform(100, 800, n)
    temperature = np.random.uniform(10, 35, n)
    
    # Transform variables
    log_water = np.log(water)
    temp_squared = temperature**2
    fert_water_interaction = fertilizer * log_water
    
    # Set coefficients
    intercept = 2.0
    fert_coef = 0.01
    log_water_coef = 1.5
    interaction_coef = 0.002
    temp_coef = 0.3
    temp_sq_coef = -0.006
    
    # Generate yield
    yield_tons = (intercept + 
                 fert_coef * fertilizer + 
                 log_water_coef * log_water + 
                 interaction_coef * fert_water_interaction + 
                 temp_coef * temperature + 
                 temp_sq_coef * temp_squared + 
                 np.random.normal(0, 0.5, n))
    
    # Create DataFrame
    full_data = pd.DataFrame({
        'Fertilizer': fertilizer,
        'Water': water,
        'LogWater': log_water,
        'Temperature': temperature,
        'TempSquared': temp_squared,
        'FertWaterInteraction': fert_water_interaction,
        'Yield': yield_tons
    })
    
    # Fit the complete model
    X = np.column_stack((fertilizer, log_water, fert_water_interaction, 
                         temperature, temp_squared))
    
    full_model = LinearRegression().fit(X, yield_tons)
    
    print("\nSimulated coefficients from complete model:")
    print(f"Intercept (β₀): {full_model.intercept_:.4f}")
    print(f"Fertilizer coefficient (β₁): {full_model.coef_[0]:.4f}")
    print(f"Log(Water) coefficient (β₂): {full_model.coef_[1]:.4f}")
    print(f"Interaction coefficient (β₃): {full_model.coef_[2]:.4f}")
    print(f"Temperature coefficient (β₄): {full_model.coef_[3]:.4f}")
    print(f"Temperature² coefficient (β₅): {full_model.coef_[4]:.4f}")
    print()
    
    # Calculate optimal temperature
    optimal_temp = -full_model.coef_[3] / (2 * full_model.coef_[4])
    print(f"Optimal temperature from complete model: {optimal_temp:.2f}°C")
    
    # Create a correlation matrix visualization
    plt.figure(figsize=(10, 8))
    corr_matrix = full_data.drop('FertWaterInteraction', axis=1).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Model Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_correlation_matrix.png"), dpi=300)
    plt.close()
    
    # Create a coefficient visualization
    plt.figure(figsize=(10, 6))
    
    # Define custom names for readability
    coef_names = ['Fertilizer', 'Log(Water)', 'Fert×Log(Water)', 'Temperature', 'Temperature²']
    coef_values = full_model.coef_
    
    # Create bar chart of coefficients
    colors = ['blue', 'green', 'purple', 'orange', 'red']
    plt.bar(coef_names, coef_values, color=colors)
    
    # Add labels
    for i, v in enumerate(coef_values):
        plt.text(i, v + 0.01 * np.sign(v), f'{v:.3f}', ha='center')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of the Complete Crop Yield Model')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_coefficients.png"), dpi=300)
    plt.close()
    
    # Create partial dependence plots for each variable
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Fertilizer effect (at median water and optimal temperature)
    median_water = np.median(water)
    log_median_water = np.log(median_water)
    fert_range = np.linspace(min(fertilizer), max(fertilizer), 100)
    
    fert_effect = (full_model.intercept_ +
                  full_model.coef_[0] * fert_range +
                  full_model.coef_[1] * log_median_water +
                  full_model.coef_[2] * fert_range * log_median_water +
                  full_model.coef_[3] * optimal_temp +
                  full_model.coef_[4] * optimal_temp**2)
    
    axes[0].plot(fert_range, fert_effect, 'b-', linewidth=2)
    axes[0].set_xlabel('Fertilizer (kg/hectare)')
    axes[0].set_ylabel('Predicted Yield (tons/hectare)')
    axes[0].set_title(f'Fertilizer Effect\n(Water={median_water:.0f}L, Temp={optimal_temp:.1f}°C)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Water effect (at median fertilizer and optimal temperature)
    median_fert = np.median(fertilizer)
    water_range = np.linspace(min(water), max(water), 100)
    log_water_range = np.log(water_range)
    
    water_effect = (full_model.intercept_ +
                   full_model.coef_[0] * median_fert +
                   full_model.coef_[1] * log_water_range +
                   full_model.coef_[2] * median_fert * log_water_range +
                   full_model.coef_[3] * optimal_temp +
                   full_model.coef_[4] * optimal_temp**2)
    
    axes[1].plot(water_range, water_effect, 'g-', linewidth=2)
    axes[1].set_xlabel('Water (liters/day)')
    axes[1].set_ylabel('Predicted Yield (tons/hectare)')
    axes[1].set_title(f'Water Effect\n(Fertilizer={median_fert:.0f}kg, Temp={optimal_temp:.1f}°C)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Temperature effect (at median fertilizer and water)
    temp_range = np.linspace(min(temperature), max(temperature), 100)
    
    temp_effect = (full_model.intercept_ +
                  full_model.coef_[0] * median_fert +
                  full_model.coef_[1] * log_median_water +
                  full_model.coef_[2] * median_fert * log_median_water +
                  full_model.coef_[3] * temp_range +
                  full_model.coef_[4] * temp_range**2)
    
    axes[2].plot(temp_range, temp_effect, 'r-', linewidth=2)
    axes[2].axvline(x=optimal_temp, color='k', linestyle='--', 
                   label=f'Optimal: {optimal_temp:.1f}°C')
    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_ylabel('Predicted Yield (tons/hectare)')
    axes[2].set_title(f'Temperature Effect\n(Fertilizer={median_fert:.0f}kg, Water={median_water:.0f}L)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "partial_dependence_plots.png"), dpi=300)
    plt.close()
    
    # Create a 3D partial dependence plot for fertilizer and water interaction
    # (at optimal temperature)
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # Create meshgrid
    fert_grid = np.linspace(min(fertilizer), max(fertilizer), 30)
    water_grid = np.linspace(min(water), max(water), 30)
    fert_mesh, water_mesh = np.meshgrid(fert_grid, water_grid)
    log_water_mesh = np.log(water_mesh)
    
    # Calculate predictions
    z = (full_model.intercept_ +
        full_model.coef_[0] * fert_mesh +
        full_model.coef_[1] * log_water_mesh +
        full_model.coef_[2] * fert_mesh * log_water_mesh +
        full_model.coef_[3] * optimal_temp +
        full_model.coef_[4] * optimal_temp**2)
    
    # Create the surface plot
    surf = ax.plot_surface(fert_mesh, water_mesh, z, cmap='viridis', alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add colorbar
    fig = plt.gcf()
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Predicted Yield')
    
    # Set labels
    ax.set_xlabel('Fertilizer (kg/hectare)')
    ax.set_ylabel('Water (liters/day)')
    ax.set_zlabel('Predicted Yield (tons/hectare)')
    ax.set_title(f'Interaction Between Fertilizer and Water\n(at Optimal Temperature: {optimal_temp:.1f}°C)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_3d_final.png"), dpi=300)
    plt.close()
    
    return full_data, full_model

full_data, full_model = write_complete_model()

# Summary of the solution
print("\nQuestion 3 Solution Summary:")
print("1. To model the joint effect of fertilizer and water, we added an interaction term: β₃(x₁ × x₂)")
print("2. For temperature's diminishing returns and eventual negative impact, we added a quadratic term: β₅x₃²")
print("3. For water's diminishing returns, we applied a logarithmic transformation: β₂log(x₂)")
print("4. The complete model is: y = β₀ + β₁x₁ + β₂log(x₂) + β₃(x₁ × log(x₂)) + β₄x₃ + β₅x₃² + ε")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- interaction_3d.png: 3D visualization of fertilizer-water interaction")
print("- interaction_lines.png: Fertilizer effect at different water levels")
print("- interaction_contour.png: Contour plot of fertilizer-water interaction")
print("- temperature_quadratic.png: Visualization of quadratic temperature effect")
print("- temperature_residuals.png: Residual plots comparing linear vs quadratic models")
print("- water_transformation.png: Linear vs logarithmic water transformation")
print("- water_marginal_effect.png: Diminishing marginal effect of water")
print("- log_water_interaction.png: Log-water and fertilizer interaction")
print("- water_biological_comparison.png: Biological comparison of linear vs log models")
print("- model_correlation_matrix.png: Correlation matrix of all model variables")
print("- model_coefficients.png: Bar chart of model coefficients")
print("- partial_dependence_plots.png: Partial dependence plots for each variable")
print("- interaction_3d_final.png: 3D surface plot of final fertilizer-water interaction") 