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
    print("="*80)
    print("STEP 1: MODELING INTERACTION BETWEEN FERTILIZER AND WATER")
    print("="*80)
    
    print("\nPROBLEM INSIGHT:")
    print("More fertilizer generally increases yield, but the effect depends on water amount.")
    print("This indicates an interaction between fertilizer and water - the effect of")
    print("one variable depends on the level of the other variable.")
    
    print("\nPROPOSED MODEL:")
    print("y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + β₄x₃ + ε")
    print("\nWhere:")
    print("- y is the crop yield (tons/hectare)")
    print("- x₁ is the amount of fertilizer (kg/hectare)")
    print("- x₂ is the amount of water (liters/day)")
    print("- x₃ is the average daily temperature (°C)")
    print("- x₁ × x₂ is the interaction term between fertilizer and water")
    print("- ε is the error term")
    print()
    
    print("DETAILED EXPLANATION OF THE INTERACTION TERM:")
    print("1. In a standard linear model without interaction (y = β₀ + β₁x₁ + β₂x₂),")
    print("   the effect of fertilizer on yield would be CONSTANT (β₁) regardless of water level.")
    print("2. With an interaction term (y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂)):")
    print("   - The effect of fertilizer on yield is now: β₁ + β₃x₂")
    print("   - The effect of water on yield is now: β₂ + β₃x₁")
    print("3. Interpretation of interaction coefficient (β₃):")
    print("   - A positive β₃ means fertilizer effect increases with more water")
    print("   - A negative β₃ means fertilizer effect decreases with more water")
    print("   - This captures the agricultural reality that fertilizer effectiveness")
    print("     typically depends on water availability to dissolve and transport nutrients")
    print()
    
    print("MATHEMATICAL DERIVATION OF FERTILIZER EFFECT WITH INTERACTION:")
    print("1. Basic model with interaction: y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂)")
    print("2. To find fertilizer effect, take partial derivative with respect to x₁:")
    print("   ∂y/∂x₁ = β₁ + β₃x₂")
    print("3. This shows that the effect of adding 1 unit of fertilizer")
    print("   depends on the current water level (x₂)")
    print()
    
    # Create simulated data to demonstrate the interaction effect
    print("SIMULATION: Creating synthetic data to demonstrate interaction effects")
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
    print(f"Generating yields based on: y = {base_yield} + {fertilizer_effect}*x₁ + {water_effect}*x₂ + {interaction_coef}*(x₁×x₂) + noise")
    yield_tons = base_yield + fertilizer_effect * fertilizer + water_effect * water + \
                 interaction_coef * fertilizer * water + np.random.normal(0, 0.5, n)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Fertilizer': fertilizer,
        'Water': water,
        'Yield': yield_tons
    })
    
    # Display basic statistics
    print("\nSimulated data summary statistics:")
    print(f"Number of observations: {n}")
    print(f"Fertilizer range: {min(fertilizer):.1f} to {max(fertilizer):.1f} kg/hectare")
    print(f"Water range: {min(water):.1f} to {max(water):.1f} liters/day")
    print(f"Yield range: {min(yield_tons):.2f} to {max(yield_tons):.2f} tons/hectare")
    
    # Fit a linear model with interaction
    print("\nFitting multiple linear regression model with interaction...")
    X = np.column_stack((fertilizer, water, fertilizer * water))
    model = LinearRegression()
    model.fit(X, yield_tons)
    
    print("\nESTIMATED MODEL COEFFICIENTS:")
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    print(f"Fertilizer coefficient (β₁): {model.coef_[0]:.4f}")
    print(f"Water coefficient (β₂): {model.coef_[1]:.4f}")
    print(f"Interaction coefficient (β₃): {model.coef_[2]:.6f}")
    
    # Calculate model performance metrics
    y_pred = model.predict(X)
    mse = np.mean((yield_tons - y_pred)**2)
    r2 = model.score(X, yield_tons)
    print(f"\nModel performance: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # Calculate fertilizer effects at different water levels
    low_water = min(water)
    med_water = np.median(water)
    high_water = max(water)
    
    print("\nFERTILIZER EFFECT AT DIFFERENT WATER LEVELS:")
    print(f"At low water ({low_water:.1f} L/day): Effect of 1 kg fertilizer = {model.coef_[0] + model.coef_[2]*low_water:.5f} tons/ha")
    print(f"At medium water ({med_water:.1f} L/day): Effect of 1 kg fertilizer = {model.coef_[0] + model.coef_[2]*med_water:.5f} tons/ha")
    print(f"At high water ({high_water:.1f} L/day): Effect of 1 kg fertilizer = {model.coef_[0] + model.coef_[2]*high_water:.5f} tons/ha")
    print(f"→ The fertilizer effect increases by {model.coef_[2]*100:.6f} for each 100 L/day increase in water")
    
    print("\nVISUALIZATION: Creating visualizations to illustrate interaction effect...")
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
    print(f"Saved 3D visualization to: {os.path.join(save_dir, 'interaction_3d.png')}")
    plt.close()
    
    # Create a 2D visualization to show how the effect of fertilizer changes with water
    print("\nCreating 2D visualization of fertilizer effect at different water levels...")
    plt.figure(figsize=(10, 6))
    
    # Select low, medium, and high water levels
    water_levels = [min(water), np.percentile(water, 50), max(water)]
    colors = ['blue', 'green', 'red']
    labels = ['Low Water', 'Medium Water', 'High Water']
    
    fertilizer_range = np.linspace(min(fertilizer), max(fertilizer), 100)
    
    # Plot yield vs. fertilizer at different water levels
    for i, water_level in enumerate(water_levels):
        # Predict yield at this water level for different fertilizer amounts
        predicted_yield = model.intercept_ + model.coef_[0] * fertilizer_range + \
                          model.coef_[1] * water_level + model.coef_[2] * fertilizer_range * water_level
        
        # Calculate the slope (effect of fertilizer) at this water level
        fert_effect = model.coef_[0] + model.coef_[2] * water_level
        
        plt.plot(fertilizer_range, predicted_yield, color=colors[i], linewidth=2, 
                label=f'{labels[i]} ({water_level:.0f} L/day) - Fert Effect: {fert_effect:.5f}')
    
    # Add data points with transparency and color based on water level
    sc = plt.scatter(fertilizer, yield_tons, c=water, cmap='viridis', alpha=0.4, s=50, edgecolor='k')
    
    plt.colorbar(sc, label='Water Level (liters/day)')
    plt.xlabel('Fertilizer (kg/hectare)')
    plt.ylabel('Predicted Yield (tons/hectare)')
    plt.title('Effect of Fertilizer on Yield at Different Water Levels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interaction_lines.png"), dpi=300)
    print(f"Saved 2D interaction visualization to: {os.path.join(save_dir, 'interaction_lines.png')}")
    plt.close()
    
    # Add a contour plot to visualize interaction in 2D
    print("\nCreating contour plot to visualize combined effect of water and fertilizer...")
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
    print(f"Saved contour plot to: {os.path.join(save_dir, 'interaction_contour.png')}")
    plt.close()
    
    print("\nKEY FINDINGS FROM STEP 1:")
    print("1. The interaction coefficient (β₃) is positive, confirming that fertilizer's effect")
    print("   increases with higher water levels.")
    print("2. This aligns with agricultural knowledge: plants can better utilize fertilizer")
    print("   nutrients when adequate water is available.")
    print("3. The contour plot shows how yield is maximized at high levels of both fertilizer and water.")
    print("4. Without the interaction term, we would incorrectly assume that fertilizer has the same")
    print("   effect regardless of water availability.")
    print()
    
    return data

data = propose_interaction_model()

# Step 2: Suggest a feature transformation for temperature
def transform_temperature():
    """Suggest a feature transformation for temperature to model diminishing returns and eventual negative impact."""
    print("="*80)
    print("STEP 2: MODELING NON-LINEAR TEMPERATURE EFFECTS")
    print("="*80)
    
    print("\nPROBLEM INSIGHT:")
    print("Higher temperatures improve yield up to a point, after which they become harmful.")
    print("This suggests a non-linear, inverted U-shaped relationship that can't be")
    print("captured by a standard linear model.")
    
    print("\nPROPOSED TRANSFORMATION:")
    print("Add a quadratic term for temperature: x₃² (temperature squared)")
    
    print("\nUPDATED MODEL:")
    print("y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁ × x₂) + β₄x₃ + β₅x₃² + ε")
    print()
    
    print("DETAILED EXPLANATION OF THE QUADRATIC TEMPERATURE TERM:")
    print("1. The linear term β₄x₃ captures the initial positive effect of temperature.")
    print("2. The quadratic term β₅x₃² (with expected negative β₅) captures the diminishing returns")
    print("   and eventual negative impact of high temperatures.")
    print("3. This creates an inverted U-shape relationship between temperature and yield.")
    print("4. Mathematical properties of this relationship:")
    print("   - The first derivative (∂y/∂x₃ = β₄ + 2β₅x₃) represents the marginal effect")
    print("     of temperature, which decreases as temperature increases")
    print("   - When set to zero (β₄ + 2β₅x₃ = 0), we can find the optimal temperature: x₃ = -β₄/(2β₅)")
    print("   - For temperatures below this optimum, the marginal effect is positive")
    print("   - For temperatures above this optimum, the marginal effect becomes negative")
    print()
    
    # Create simulated data to demonstrate the quadratic effect of temperature
    print("SIMULATION: Creating synthetic data to demonstrate quadratic temperature effects")
    np.random.seed(42)
    n = 200
    
    # Generate temperature data
    temperature = np.random.uniform(5, 40, n)  # °C
    
    # Generate yield with quadratic temperature effect
    # Setting coefficients to create a peak around 25°C
    temp_linear_coef = 0.5  # positive effect initially
    temp_quad_coef = -0.01  # negative effect for quadratic term
    
    # Optimal temperature: -temp_linear_coef / (2 * temp_quad_coef) ≈ 25°C
    optimal_temp_true = -temp_linear_coef / (2 * temp_quad_coef)
    
    print(f"True model parameters used for simulation:")
    print(f"- Temperature linear coefficient: {temp_linear_coef}")
    print(f"- Temperature quadratic coefficient: {temp_quad_coef}")
    print(f"- This creates an optimal temperature at: {optimal_temp_true:.2f}°C")
    
    # Calculate yield based only on temperature effect
    yield_tons = 5 + temp_linear_coef * temperature + temp_quad_coef * temperature**2 + \
                np.random.normal(0, 0.5, n)
    
    # Create DataFrame
    temp_data = pd.DataFrame({
        'Temperature': temperature,
        'Yield': yield_tons
    })
    
    # Display basic statistics
    print("\nSimulated data summary statistics:")
    print(f"Number of observations: {n}")
    print(f"Temperature range: {min(temperature):.1f} to {max(temperature):.1f}°C")
    print(f"Yield range: {min(yield_tons):.2f} to {max(yield_tons):.2f} tons/hectare")
    
    # Fit a quadratic model
    print("\nFitting quadratic model for temperature...")
    X = np.column_stack((temperature, temperature**2))
    model = LinearRegression()
    model.fit(X, yield_tons)
    
    print("\nESTIMATED MODEL COEFFICIENTS:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Temperature linear coefficient (β₄): {model.coef_[0]:.4f}")
    print(f"Temperature quadratic coefficient (β₅): {model.coef_[1]:.4f}")
    
    # Calculate optimal temperature from model
    optimal_temp_model = -model.coef_[0] / (2 * model.coef_[1])
    print(f"\nOptimal temperature calculated from model: {optimal_temp_model:.2f}°C")
    print(f"(True optimal temperature was: {optimal_temp_true:.2f}°C)")
    
    # Calculate model performance metrics
    y_pred_quad = model.predict(X)
    mse_quad = np.mean((yield_tons - y_pred_quad)**2)
    r2_quad = model.score(X, yield_tons)
    
    # Compare with linear model
    linear_model = LinearRegression().fit(temperature.reshape(-1, 1), yield_tons)
    y_pred_linear = linear_model.predict(temperature.reshape(-1, 1))
    mse_linear = np.mean((yield_tons - y_pred_linear)**2)
    r2_linear = linear_model.score(temperature.reshape(-1, 1), yield_tons)
    
    print("\nMODEL COMPARISON:")
    print(f"Linear model:   R² = {r2_linear:.4f}, MSE = {mse_linear:.4f}")
    print(f"Quadratic model: R² = {r2_quad:.4f}, MSE = {mse_quad:.4f}")
    print(f"Improvement: {(r2_quad-r2_linear)*100:.2f}% in R², {(1-mse_quad/mse_linear)*100:.2f}% reduction in MSE")
    
    # Calculate temperature effects at different points
    temp_low = 10
    temp_optimal = optimal_temp_model
    temp_high = 35
    
    print("\nMARGINAL EFFECT OF TEMPERATURE AT DIFFERENT POINTS:")
    marginal_low = model.coef_[0] + 2 * model.coef_[1] * temp_low
    marginal_opt = model.coef_[0] + 2 * model.coef_[1] * temp_optimal
    marginal_high = model.coef_[0] + 2 * model.coef_[1] * temp_high
    
    print(f"At low temperature ({temp_low}°C): Effect = {marginal_low:.4f}")
    print(f"At optimal temperature ({temp_optimal:.1f}°C): Effect ≈ {marginal_opt:.4f} (should be near zero)")
    print(f"At high temperature ({temp_high}°C): Effect = {marginal_high:.4f}")
    
    print("\nVISUALIZATION: Creating visualizations to illustrate temperature effects...")
    # Visualize the quadratic relationship with a scatter plot and fitted curve
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of the data points
    plt.scatter(temperature, yield_tons, alpha=0.5, color='cornflowerblue', edgecolor='navy')
    
    # Sort temperature values for a smooth line plot
    temp_range = np.linspace(min(temperature), max(temperature), 100)
    predicted_yield_quad = model.intercept_ + model.coef_[0] * temp_range + model.coef_[1] * temp_range**2
    
    # Add the fitted curve
    plt.plot(temp_range, predicted_yield_quad, 'r-', linewidth=2, label='Quadratic Model')
    
    # Mark the optimal temperature
    optimal_yield = model.intercept_ + model.coef_[0] * optimal_temp_model + model.coef_[1] * optimal_temp_model**2
    plt.scatter([optimal_temp_model], [optimal_yield], s=100, c='green', marker='*', 
              edgecolor='k', label=f'Optimal Temperature: {optimal_temp_model:.1f}°C')
    
    # Add a vertical line at the optimal temperature
    plt.axvline(x=optimal_temp_model, color='green', linestyle='--', alpha=0.5)
    
    # Linear regression for comparison
    linear_predicted = linear_model.intercept_ + linear_model.coef_[0] * temp_range
    plt.plot(temp_range, linear_predicted, 'b--', linewidth=2, label='Linear Model')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Quadratic Relationship Between Temperature and Crop Yield')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_quadratic.png"), dpi=300)
    print(f"Saved temperature effect visualization to: {os.path.join(save_dir, 'temperature_quadratic.png')}")
    plt.close()
    
    # Create a residual plot to show how quadratic model fits better
    print("\nCreating residual plots to compare model fits...")
    plt.figure(figsize=(10, 6))
    
    # Calculate residuals
    linear_residuals = yield_tons - y_pred_linear
    quad_residuals = yield_tons - y_pred_quad
    
    plt.subplot(1, 2, 1)
    plt.scatter(temperature, linear_residuals, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Residuals')
    plt.title('Linear Model Residuals\n(Pattern indicates poor fit)')
    
    plt.subplot(1, 2, 2)
    plt.scatter(temperature, quad_residuals, alpha=0.5, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Residuals')
    plt.title('Quadratic Model Residuals\n(More random pattern indicates better fit)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_residuals.png"), dpi=300)
    print(f"Saved residual plots to: {os.path.join(save_dir, 'temperature_residuals.png')}")
    plt.close()
    
    # Plot the marginal effect of temperature (first derivative)
    print("\nCreating plot of marginal temperature effect...")
    plt.figure(figsize=(10, 6))
    
    # Calculate marginal effect across temperature range
    marginal_effect = model.coef_[0] + 2 * model.coef_[1] * temp_range
    
    plt.plot(temp_range, marginal_effect, 'g-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=optimal_temp_model, color='b', linestyle='--', 
               label=f'Optimal temperature: {optimal_temp_model:.1f}°C')
    
    # Add points at specific temperatures
    plt.scatter([temp_low, optimal_temp_model, temp_high], 
               [marginal_low, marginal_opt, marginal_high],
               s=100, c=['blue', 'green', 'red'], edgecolor='k',
               label='Reference points')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Marginal Effect on Yield (∂y/∂x₃)')
    plt.title('Marginal Effect of Temperature on Crop Yield')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temperature_marginal_effect.png"), dpi=300)
    print(f"Saved marginal effect plot to: {os.path.join(save_dir, 'temperature_marginal_effect.png')}")
    plt.close()
    
    print("\nKEY FINDINGS FROM STEP 2:")
    print("1. The quadratic model significantly outperforms the linear model, confirming")
    print("   the non-linear relationship between temperature and crop yield.")
    print("2. The optimal temperature is estimated at approximately {:.1f}°C, which aligns".format(optimal_temp_model))
    print("   with typical optimal growing temperatures for many crops.")
    print("3. At temperatures below {:.1f}°C, increasing temperature has a positive effect".format(optimal_temp_model))
    print("   on yield, while above this temperature, the effect becomes negative.")
    print("4. The residual plots confirm that the quadratic model captures the")
    print("   true relationship better than a linear model.")
    print()
    
    return temp_data, optimal_temp_model

temp_data, optimal_temp = transform_temperature()

# Step 3: Propose a feature transformation for water to capture diminishing returns
def transform_water():
    """Propose a feature transformation for water to capture the diminishing returns effect."""
    print("="*80)
    print("STEP 3: MODELING DIMINISHING RETURNS FOR WATER")
    print("="*80)
    
    print("\nPROBLEM INSIGHT:")
    print("The effect of water on yield diminishes as more water is added.")
    print("This follows the law of diminishing returns, common in agricultural contexts.")
    
    print("\nPROPOSED TRANSFORMATION:")
    print("Use a logarithmic transformation: log(x₂)")
    
    print("\nUPDATED MODEL:")
    print("y = β₀ + β₁x₁ + β₂log(x₂) + β₃(x₁ × log(x₂)) + β₄x₃ + β₅x₃² + ε")
    print()
    
    print("DETAILED EXPLANATION OF THE LOGARITHMIC WATER TRANSFORMATION:")
    print("1. The logarithmic transformation log(x₂) captures the diminishing returns of water:")
    print("   - For a linear term (β₂x₂), each additional unit of water adds a constant β₂ to yield")
    print("   - For a logarithmic term (β₂log(x₂)), the effect of additional water diminishes")
    print("2. Mathematical properties:")
    print("   - The marginal effect is: ∂y/∂x₂ = β₂/x₂")
    print("   - As x₂ increases, the effect of adding more water decreases proportionally")
    print("   - This follows the principle that plants have limited capacity to utilize water")
    print("3. Note that we've also updated the interaction term to x₁ × log(x₂) to maintain")
    print("   consistency in how water is represented throughout the model")
    print()
    
    # Create simulated data to demonstrate the logarithmic effect of water
    print("SIMULATION: Creating synthetic data to demonstrate logarithmic water effects")
    np.random.seed(42)
    n = 150
    
    # Generate water data
    water = np.random.uniform(50, 800, n)  # liters/day
    
    # Generate yield with logarithmic water effect
    log_water = np.log(water)
    water_coef = 2.0  # coefficient for log(water)
    
    print(f"True model parameters used for simulation:")
    print(f"- Base yield: 3.0 tons/hectare")
    print(f"- Log(Water) coefficient: {water_coef}")
    print(f"- Model: y = 3.0 + {water_coef} × log(water) + noise")
    
    # Calculate yield based only on water effect
    yield_tons = 3 + water_coef * log_water + np.random.normal(0, 0.3, n)
    
    # Create DataFrame
    water_data = pd.DataFrame({
        'Water': water,
        'LogWater': log_water,
        'Yield': yield_tons
    })
    
    # Display basic statistics
    print("\nSimulated data summary statistics:")
    print(f"Number of observations: {n}")
    print(f"Water range: {min(water):.1f} to {max(water):.1f} liters/day")
    print(f"Log(Water) range: {min(log_water):.2f} to {max(log_water):.2f}")
    print(f"Yield range: {min(yield_tons):.2f} to {max(yield_tons):.2f} tons/hectare")
    
    # Fit models: linear and logarithmic
    print("\nComparing linear vs. logarithmic water transformation...")
    X_linear = water.reshape(-1, 1)
    X_log = log_water.reshape(-1, 1)
    
    model_linear = LinearRegression().fit(X_linear, yield_tons)
    model_log = LinearRegression().fit(X_log, yield_tons)
    
    # Calculate metrics
    y_pred_linear = model_linear.predict(X_linear)
    y_pred_log = model_log.predict(X_log)
    
    mse_linear = np.mean((yield_tons - y_pred_linear)**2)
    mse_log = np.mean((yield_tons - y_pred_log)**2)
    
    print("\nMODEL COMPARISON:")
    print(f"Linear model:   R² = {model_linear.score(X_linear, yield_tons):.4f}, MSE = {mse_linear:.4f}")
    print(f"Logarithmic model: R² = {model_log.score(X_log, yield_tons):.4f}, MSE = {mse_log:.4f}")
    print(f"Improvement: {(model_log.score(X_log, yield_tons)-model_linear.score(X_linear, yield_tons))*100:.2f}% in R²")
    print(f"             {(1-mse_log/mse_linear)*100:.2f}% reduction in MSE")
    
    print("\nESTIMATED MODEL COEFFICIENTS:")
    print(f"Linear model - Intercept: {model_linear.intercept_:.4f}")
    print(f"Linear model - Water coefficient: {model_linear.coef_[0]:.6f}")
    print(f"Logarithmic model - Intercept: {model_log.intercept_:.4f}")
    print(f"Logarithmic model - Log(Water) coefficient: {model_log.coef_[0]:.4f}")
    
    # Calculate water effects at different points
    water_low = 100
    water_med = 300
    water_high = 600
    
    print("\nMARGINAL EFFECT OF WATER AT DIFFERENT LEVELS:")
    # Linear model has constant effect
    linear_effect = model_linear.coef_[0]
    
    # Log model has diminishing effect (derivative of β * log(x) is β/x)
    log_effect_low = model_log.coef_[0] / water_low
    log_effect_med = model_log.coef_[0] / water_med
    log_effect_high = model_log.coef_[0] / water_high
    
    print("Linear model (constant effect):")
    print(f"- Effect of +1 L/day at any water level: {linear_effect:.6f} tons/ha")
    print("\nLogarithmic model (diminishing effect):")
    print(f"- Effect of +1 L/day at {water_low} L/day: {log_effect_low:.6f} tons/ha")
    print(f"- Effect of +1 L/day at {water_med} L/day: {log_effect_med:.6f} tons/ha")
    print(f"- Effect of +1 L/day at {water_high} L/day: {log_effect_high:.6f} tons/ha")
    print(f"- At {water_high} L/day, the effect is {log_effect_high/log_effect_low*100:.1f}% of what it was at {water_low} L/day")
    
    print("\nVISUALIZATION: Creating visualizations to illustrate water transformation effects...")
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
    
    # Highlight specific points
    for w in [water_low, water_med, water_high]:
        linear_y = model_linear.intercept_ + model_linear.coef_[0] * w
        log_y = model_log.intercept_ + model_log.coef_[0] * np.log(w)
        plt.scatter([w, w], [linear_y, log_y], c=['red', 'green'], s=50, edgecolor='k')
    
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Comparison of Linear vs. Logarithmic Models for Water Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_transformation.png"), dpi=300)
    print(f"Saved water transformation visualization to: {os.path.join(save_dir, 'water_transformation.png')}")
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
    
    # Highlight specific points
    plt.scatter([water_low, water_med, water_high], 
               [linear_effect[0], linear_effect[0], linear_effect[0]], 
               c='red', s=50, edgecolor='k')
    plt.scatter([water_low, water_med, water_high], 
               [log_effect_low, log_effect_med, log_effect_high], 
               c='green', s=50, edgecolor='k')
    
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Marginal Effect on Yield (tons/ha per L/day)')
    plt.title('Marginal Effect of Water on Yield')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_marginal_effect.png"), dpi=300)
    print(f"Saved marginal effect plot to: {os.path.join(save_dir, 'water_marginal_effect.png')}")
    plt.close()
    
    # Create an interaction plot with log-transformed water
    print("\nSimulating interaction between fertilizer and log-transformed water...")
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
    
    print(f"Simulation parameters:")
    print(f"- Fertilizer coefficient: {fert_coef}")
    print(f"- Log(Water) coefficient: {log_water_coef}")
    print(f"- Interaction coefficient: {interaction_coef}")
    print(f"- Model: y = 3.0 + {fert_coef}*fert + {log_water_coef}*log(water) + {interaction_coef}*(fert*log(water)) + noise")
    
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
    print("\nFitting model with log-transformed water and interaction...")
    X = np.column_stack((fertilizer, log_water_new, fertilizer * log_water_new))
    log_interaction_model = LinearRegression().fit(X, yield_tons)
    
    print("\nESTIMATED MODEL COEFFICIENTS:")
    print(f"Intercept (β₀): {log_interaction_model.intercept_:.4f}")
    print(f"Fertilizer coefficient (β₁): {log_interaction_model.coef_[0]:.4f}")
    print(f"Log(Water) coefficient (β₂): {log_interaction_model.coef_[1]:.4f}")
    print(f"Interaction coefficient (β₃): {log_interaction_model.coef_[2]:.4f}")
    
    # Calculate model performance
    y_pred = log_interaction_model.predict(X)
    mse = np.mean((yield_tons - y_pred)**2)
    r2 = log_interaction_model.score(X, yield_tons)
    print(f"\nModel performance: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # Calculate fertilizer effect at different water levels with log transformation
    water_level_low = 100
    water_level_high = 600
    log_water_low = np.log(water_level_low)
    log_water_high = np.log(water_level_high)
    
    print("\nFERTILIZER EFFECT WITH LOG-TRANSFORMED WATER:")
    effect_low = log_interaction_model.coef_[0] + log_interaction_model.coef_[2] * log_water_low
    effect_high = log_interaction_model.coef_[0] + log_interaction_model.coef_[2] * log_water_high
    
    print(f"At {water_level_low} L/day water: Effect of +1 kg fertilizer = {effect_low:.5f} tons/ha")
    print(f"At {water_level_high} L/day water: Effect of +1 kg fertilizer = {effect_high:.5f} tons/ha")
    print(f"Ratio of effects: {effect_high/effect_low:.2f}")
    
    print("\nCreating 2D contour plot of log-water and fertilizer interaction...")
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
    print(f"Saved contour plot to: {os.path.join(save_dir, 'log_water_interaction.png')}")
    plt.close()
    
    # Show how the model with log-transformed water makes more biological sense
    print("\nCreating comparison of linear vs. logarithmic water models (biological plausibility)...")
    plt.figure(figsize=(12, 6))
    
    # Generate predictor values
    water_vals = np.linspace(50, 800, 100)
    
    # At different fixed fertilizer levels
    fert_levels = [60, 120, 180]
    colors = ['blue', 'green', 'red']
    
    for i, fert in enumerate(fert_levels):
        # Linear model predictions (assuming model_linear is simple water effect)
        y_linear = model_linear.intercept_ + model_linear.coef_[0] * water_vals
        
        # Log model predictions (assuming model_log is log water effect)
        y_log = model_log.intercept_ + model_log.coef_[0] * np.log(water_vals)
        
        plt.subplot(1, 2, 1)
        plt.plot(water_vals, y_linear, color=colors[i], linestyle='-',
                label=f'Fert = {fert} kg/ha' if i == 0 else None)
        
        plt.subplot(1, 2, 2)
        plt.plot(water_vals, y_log, color=colors[i], linestyle='-', 
                label=f'Fertilizer = {fert} kg/ha')
    
    plt.subplot(1, 2, 1)
    plt.scatter(water_data['Water'], water_data['Yield'], alpha=0.3, color='gray', edgecolor='k')
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Linear Water Effect\n(Unlikely: Unlimited Growth)')
    if len(fert_levels) > 0:
        plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(water_data['Water'], water_data['Yield'], alpha=0.3, color='gray', edgecolor='k')
    plt.xlabel('Water (liters/day)')
    plt.ylabel('Yield (tons/hectare)')
    plt.title('Logarithmic Water Effect\n(Realistic: Diminishing Returns)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "water_biological_comparison.png"), dpi=300)
    print(f"Saved biological comparison to: {os.path.join(save_dir, 'water_biological_comparison.png')}")
    plt.close()
    
    print("\nKEY FINDINGS FROM STEP 3:")
    print("1. The logarithmic model significantly outperforms the linear model for water effects,")
    print(f"   with an R² improvement of {(model_log.score(X_log, yield_tons)-model_linear.score(X_linear, yield_tons))*100:.2f}%.")
    print("2. The marginal effect of water decreases as water quantity increases, aligning with")
    print("   agricultural principles of diminishing returns.")
    print("3. The logarithmic transformation creates a more biologically plausible model, as plants")
    print("   have limited capacity to utilize additional water beyond certain thresholds.")
    print("4. The interaction between fertilizer and log-transformed water maintains the appropriate")
    print("   diminishing returns property while capturing the synergistic relationship.")
    print()
    
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