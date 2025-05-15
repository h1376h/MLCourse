import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_4_Quiz_4")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Given data from the problem
car_data = pd.DataFrame({
    'Car_Model': ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F'],
    'Engine_Type': ['Hybrid', 'Gasoline', 'Diesel', 'Gasoline', 'Hybrid', 'Diesel'],
    'Transmission': ['Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual', 'Manual'],
    'Weight_kg': [1200, 1500, 1800, 1400, 1300, 1700],
    'MPG': [45, 32, 28, 30, 42, 30]
})

# Step 1: Create appropriate dummy variables for categorical predictors
def create_dummy_variables():
    """Create appropriate dummy variables for the categorical predictors."""
    print("Step 1: Creating appropriate dummy variables for categorical predictors")
    
    print("\nOriginal data:")
    print(car_data)
    print()
    
    print("For Engine Type, we have three categories: Hybrid, Gasoline, and Diesel.")
    print("We need (3-1) = 2 dummy variables. Let's choose Hybrid as the reference category.")
    print()
    
    print("For Transmission, we have two categories: Automatic and Manual.")
    print("We need (2-1) = 1 dummy variable. Let's choose Automatic as the reference category.")
    print()
    
    # Create the dummy variables manually for clarity
    # For Engine Type (reference: Hybrid)
    car_data['Engine_Gasoline'] = (car_data['Engine_Type'] == 'Gasoline').astype(int)
    car_data['Engine_Diesel'] = (car_data['Engine_Type'] == 'Diesel').astype(int)
    
    # For Transmission (reference: Automatic)
    car_data['Trans_Manual'] = (car_data['Transmission'] == 'Manual').astype(int)
    
    print("Data with dummy variables:")
    print(car_data)
    print()
    
    # Visualize the categorical data
    plt.figure(figsize=(10, 5))
    
    # Count cars by engine type and transmission
    engine_counts = car_data['Engine_Type'].value_counts().reset_index()
    engine_counts.columns = ['Engine_Type', 'Count']
    
    transmission_counts = car_data['Transmission'].value_counts().reset_index()
    transmission_counts.columns = ['Transmission', 'Count']
    
    # Create subplots
    plt.subplot(1, 2, 1)
    sns.barplot(x='Engine_Type', y='Count', data=engine_counts, palette='Set2')
    plt.title('Cars by Engine Type')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Transmission', y='Count', data=transmission_counts, palette='Set2')
    plt.title('Cars by Transmission')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "categorical_counts.png"), dpi=300)
    plt.close()
    
    # Create a heatmap of the dummy variable encoding
    plt.figure(figsize=(10, 6))
    
    # Create a binary display of the dummy variables
    dummy_vars = car_data[['Engine_Gasoline', 'Engine_Diesel', 'Trans_Manual']]
    
    # Add car models as index for clarity
    dummy_vars_with_labels = dummy_vars.copy()
    dummy_vars_with_labels.index = car_data['Car_Model']
    
    # Create a heatmap
    sns.heatmap(dummy_vars_with_labels, cmap='Blues', annot=True, fmt='d', cbar=False)
    plt.title('Dummy Variable Encoding')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dummy_encoding.png"), dpi=300)
    plt.close()
    
    # Visualize the relationship between categorical variables and MPG
    plt.figure(figsize=(12, 6))
    
    # Create subplots for MPG by engine type and transmission
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Engine_Type', y='MPG', data=car_data, palette='Set2')
    plt.title('MPG by Engine Type')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Transmission', y='MPG', data=car_data, palette='Set2')
    plt.title('MPG by Transmission')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mpg_by_category.png"), dpi=300)
    plt.close()
    
    # Create a more detailed visualization with points
    plt.figure(figsize=(10, 6))
    
    # Plot MPG by weight with colors for engine type and markers for transmission
    markers = {'Automatic': 'o', 'Manual': '^'}
    
    # Create a scatter plot
    for engine in car_data['Engine_Type'].unique():
        for trans in car_data['Transmission'].unique():
            subset = car_data[(car_data['Engine_Type'] == engine) & 
                            (car_data['Transmission'] == trans)]
            plt.scatter(subset['Weight_kg'], subset['MPG'], 
                       label=f'{engine}, {trans}',
                       marker=markers[trans], s=100, alpha=0.7)
    
    plt.xlabel('Weight (kg)')
    plt.ylabel('MPG')
    plt.title('MPG vs Weight by Engine Type and Transmission')
    plt.legend(title='Engine, Transmission')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mpg_by_weight_categories.png"), dpi=300)
    plt.close()
    
    return car_data

car_data = create_dummy_variables()

# Step 2: Write down the design matrix X
def create_design_matrix():
    """Create the design matrix X using the dummy variables and the weight feature."""
    print("\nStep 2: Creating the design matrix X")
    
    # Extract the features needed for the design matrix
    X_features = car_data[['Weight_kg', 'Engine_Gasoline', 'Engine_Diesel', 'Trans_Manual']]
    
    # Add a column of ones for the intercept
    X_with_intercept = np.column_stack((np.ones(len(X_features)), X_features))
    
    # Create a DataFrame for better visualization
    X_df = pd.DataFrame(X_with_intercept, 
                      columns=['Intercept', 'Weight_kg', 'Engine_Gasoline', 'Engine_Diesel', 'Trans_Manual'])
    
    print("Design matrix X:")
    print(X_df)
    print()
    
    # Visualize the design matrix
    plt.figure(figsize=(10, 6))
    
    # Add car model as index for clarity
    X_df_labeled = X_df.copy()
    X_df_labeled.index = car_data['Car_Model']
    
    # Create a heatmap
    sns.heatmap(X_df_labeled, annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title('Design Matrix X')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix.png"), dpi=300)
    plt.close()
    
    # Create a visualization showing the structure of the design matrix
    plt.figure(figsize=(8, 6))
    
    # Create a structurally similar matrix but with colors indicating the type of feature
    # 0: Intercept, 1: Continuous feature, 2: Dummy variable
    structure_matrix = np.zeros((len(X_features), X_with_intercept.shape[1]))
    structure_matrix[:, 0] = 0  # Intercept
    structure_matrix[:, 1] = 1  # Weight (continuous)
    structure_matrix[:, 2:] = 2  # Dummy variables
    
    # Create a DataFrame for the structure matrix
    structure_df = pd.DataFrame(structure_matrix, 
                              columns=['Intercept', 'Weight_kg', 'Engine_Gasoline', 'Engine_Diesel', 'Trans_Manual'])
    structure_df.index = car_data['Car_Model']
    
    # Create a custom colormap for the three types of features
    cmap = plt.cm.colors.ListedColormap(['lightblue', 'green', 'orange'])
    
    # Create the heatmap
    ax = sns.heatmap(structure_df, cmap=cmap, cbar=False, linewidths=0.5, linecolor='white')
    
    # Create a custom legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='lightblue', label='Intercept')
    green_patch = mpatches.Patch(color='green', label='Continuous Feature')
    orange_patch = mpatches.Patch(color='orange', label='Dummy Variable')
    
    plt.legend(handles=[blue_patch, green_patch, orange_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title('Design Matrix Structure')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "design_matrix_structure.png"), dpi=300)
    plt.close()
    
    return X_df

X_df = create_design_matrix()

# Step 3: Write the full regression equation with coefficients
def write_regression_equation():
    """Write the full regression equation with coefficients for all variables."""
    print("\nStep 3: Writing the full regression equation")
    
    print("The full regression equation is:")
    print("MPG = w₀ + w₁ × Weight_kg + w₂ × Engine_Gasoline + w₃ × Engine_Diesel + w₄ × Trans_Manual + ε")
    print()
    
    print("Where:")
    print("- MPG is the fuel efficiency in miles per gallon")
    print("- Weight_kg is the car weight in kilograms")
    print("- Engine_Gasoline is 1 if the engine is Gasoline, 0 otherwise")
    print("- Engine_Diesel is 1 if the engine is Diesel, 0 otherwise")
    print("- Trans_Manual is 1 if the transmission is Manual, 0 otherwise")
    print("- w₀, w₁, w₂, w₃, and w₄ are the regression coefficients")
    print("- ε is the error term")
    print()
    
    # Let's fit the model to get some actual coefficients
    y = car_data['MPG'].values
    X = X_df.values
    
    # Fit the model
    model = LinearRegression(fit_intercept=False)  # Intercept is already in the design matrix
    model.fit(X, y)
    
    print("Fitted model coefficients:")
    print(f"w₀ (Intercept): {model.coef_[0]:.4f}")
    print(f"w₁ (Weight_kg): {model.coef_[1]:.4f}")
    print(f"w₂ (Engine_Gasoline): {model.coef_[2]:.4f}")
    print(f"w₃ (Engine_Diesel): {model.coef_[3]:.4f}")
    print(f"w₄ (Trans_Manual): {model.coef_[4]:.4f}")
    print()
    
    # Calculate predicted values
    y_pred = model.predict(X)
    
    # Add to DataFrame for visualization
    results_df = pd.DataFrame({
        'Car_Model': car_data['Car_Model'],
        'Actual_MPG': y,
        'Predicted_MPG': y_pred,
        'Residual': y - y_pred
    })
    
    print("Model predictions:")
    print(results_df)
    print()
    
    # Visualize the coefficients
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot of the coefficients
    coef_names = ['Intercept', 'Weight_kg', 'Engine_Gasoline', 'Engine_Diesel', 'Trans_Manual']
    colors = ['gray', 'blue', 'green', 'red', 'purple']
    
    plt.bar(coef_names, model.coef_, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add coefficient values as text
    for i, v in enumerate(model.coef_):
        plt.text(i, v + np.sign(v)*0.5, f'{v:.2f}', ha='center')
    
    plt.ylabel('Coefficient Value')
    plt.title('Regression Coefficients')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "regression_coefficients.png"), dpi=300)
    plt.close()
    
    # Visualize the actual vs. predicted values
    plt.figure(figsize=(10, 6))
    
    # Create a scatter plot
    plt.scatter(y, y_pred, s=100, alpha=0.7)
    
    # Add a perfect prediction line
    min_val = min(min(y), min(y_pred)) - 1
    max_val = max(max(y), max(y_pred)) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    # Add car model labels
    for i, model_name in enumerate(car_data['Car_Model']):
        plt.annotate(model_name, (y[i], y_pred[i]), fontsize=10, 
                     xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Actual MPG')
    plt.ylabel('Predicted MPG')
    plt.title('Actual vs. Predicted MPG')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # Visualize the residuals
    plt.figure(figsize=(10, 6))
    
    # Create a residual plot
    plt.bar(car_data['Car_Model'], y - y_pred, color='teal', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='-', linewidth=2)
    
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Model Residuals by Car')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "residuals.png"), dpi=300)
    plt.close()
    
    return model.coef_

coefficients = write_regression_equation()

# Step 4: Explain how to interpret the coefficient of one of the dummy variables
def interpret_coefficient():
    """Explain how to interpret the coefficient of one of the dummy variables."""
    print("\nStep 4: Interpreting the coefficient of a dummy variable")
    
    print("Let's interpret the coefficient of Engine_Diesel (w₃):")
    print(f"w₃ = {coefficients[3]:.4f}")
    print()
    
    explanation = f"""The coefficient w₃ = {coefficients[3]:.4f} for Engine_Diesel can be interpreted as follows:

All else being equal (i.e., holding weight and transmission type constant),
cars with Diesel engines have an MPG that is {coefficients[3]:.2f} units different
from cars with Hybrid engines (the reference category).

Since the coefficient is negative, Diesel engine cars have, on average, 
{abs(coefficients[3]):.2f} MPG less than Hybrid engines cars, 
controlling for other variables in the model.

This is a "ceteris paribus" or "all else held constant" interpretation, 
which is the standard way to interpret coefficients in multiple regression.
"""
    
    print(explanation)
    print()
    
    # Visualize the engine effect holding other variables constant
    plt.figure(figsize=(12, 6))
    
    # Calculate the effect of each engine type across weight range
    weight_range = np.linspace(1100, 1900, 100)
    
    # Set dummy variables for each engine type (keeping transmission constant as Automatic)
    # Hybrid (reference): Engine_Gasoline=0, Engine_Diesel=0, Trans_Manual=0
    # Gasoline: Engine_Gasoline=1, Engine_Diesel=0, Trans_Manual=0
    # Diesel: Engine_Gasoline=0, Engine_Diesel=1, Trans_Manual=0
    
    # Calculate MPG for each engine type across weight range
    mpg_hybrid = coefficients[0] + coefficients[1] * weight_range  # Reference
    mpg_gasoline = mpg_hybrid + coefficients[2]  # Add Gasoline effect
    mpg_diesel = mpg_hybrid + coefficients[3]    # Add Diesel effect
    
    # Plot the lines
    plt.plot(weight_range, mpg_hybrid, 'g-', linewidth=3, label='Hybrid')
    plt.plot(weight_range, mpg_gasoline, 'b-', linewidth=3, label='Gasoline')
    plt.plot(weight_range, mpg_diesel, 'r-', linewidth=3, label='Diesel')
    
    # Add arrows showing the diesel effect at a specific weight
    specific_weight = 1500
    hybrid_mpg_at_weight = coefficients[0] + coefficients[1] * specific_weight
    diesel_mpg_at_weight = hybrid_mpg_at_weight + coefficients[3]
    
    plt.annotate('', xy=(specific_weight, diesel_mpg_at_weight), 
                xytext=(specific_weight, hybrid_mpg_at_weight),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    
    plt.text(specific_weight + 20, (hybrid_mpg_at_weight + diesel_mpg_at_weight)/2, 
            f'Diesel effect:\n{coefficients[3]:.2f} MPG', 
            va='center')
    
    # Plot actual data points
    for i, row in car_data.iterrows():
        if row['Transmission'] == 'Automatic':  # Only plot Automatic for clarity
            if row['Engine_Type'] == 'Hybrid':
                color = 'green'
            elif row['Engine_Type'] == 'Gasoline':
                color = 'blue'
            else:  # Diesel
                color = 'red'
            
            plt.scatter(row['Weight_kg'], row['MPG'], s=100, color=color, 
                       edgecolor='black', zorder=5,
                       label=f"{row['Car_Model']} ({row['Engine_Type']})")
    
    plt.xlabel('Weight (kg)')
    plt.ylabel('MPG')
    plt.title('Effect of Engine Type on MPG, Controlling for Weight\n(for Automatic Transmission)')
    
    # Create a custom legend to avoid duplicates
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', lw=3, label='Hybrid'),
        Line2D([0], [0], color='b', lw=3, label='Gasoline'),
        Line2D([0], [0], color='r', lw=3, label='Diesel'),
    ]
    
    # Add a label for each car model in the plot
    for i, row in car_data.iterrows():
        if row['Transmission'] == 'Automatic':
            plt.text(row['Weight_kg'] + 20, row['MPG'], row['Car_Model'], fontsize=9)
    
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "engine_effect.png"), dpi=300)
    plt.close()
    
    # Create a comparison of all categorical effects
    plt.figure(figsize=(12, 6))
    
    # Set a reference weight (e.g., the mean weight)
    ref_weight = np.mean(car_data['Weight_kg'])
    
    # Calculate the base value (Hybrid, Automatic, at reference weight)
    base_mpg = coefficients[0] + coefficients[1] * ref_weight
    
    # Define all combinations of engine and transmission
    combinations = [
        ('Hybrid', 'Automatic', 0, 0, 0),
        ('Hybrid', 'Manual', 0, 0, 1),
        ('Gasoline', 'Automatic', 1, 0, 0),
        ('Gasoline', 'Manual', 1, 0, 1),
        ('Diesel', 'Automatic', 0, 1, 0),
        ('Diesel', 'Manual', 0, 1, 1)
    ]
    
    # Calculate MPG for each combination
    combination_mpgs = []
    for engine, trans, gas_dummy, diesel_dummy, manual_dummy in combinations:
        mpg = base_mpg + coefficients[2] * gas_dummy + coefficients[3] * diesel_dummy + coefficients[4] * manual_dummy
        combination_mpgs.append((f"{engine}, {trans}", mpg))
    
    # Sort by MPG for better visualization
    combination_mpgs.sort(key=lambda x: x[1], reverse=True)
    
    # Create bar colors based on engine type
    colors = []
    for combo, _ in combination_mpgs:
        if 'Hybrid' in combo:
            colors.append('green')
        elif 'Gasoline' in combo:
            colors.append('blue')
        else:  # Diesel
            colors.append('red')
    
    # Create the bar chart
    bars = plt.bar([x[0] for x in combination_mpgs], [x[1] for x in combination_mpgs], color=colors, alpha=0.7)
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', 
                ha='center', va='bottom')
    
    plt.axhline(y=base_mpg, color='gray', linestyle='--', 
               label=f'Reference (Hybrid, Auto) at {ref_weight:.0f}kg: {base_mpg:.1f} MPG')
    
    plt.ylabel('Predicted MPG')
    plt.title(f'Predicted MPG by Engine Type and Transmission\n(at Weight = {ref_weight:.0f}kg)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_categorical_effects.png"), dpi=300)
    plt.close()

interpret_coefficient()

# Summary of the solution
print("\nQuestion 4 Solution Summary:")
print("1. We created dummy variables for the categorical predictors:")
print("   - Engine_Gasoline, Engine_Diesel (reference: Hybrid)")
print("   - Trans_Manual (reference: Automatic)")
print()
print("2. The design matrix X includes the intercept, weight, and the dummy variables:")
print("   X = [1, Weight_kg, Engine_Gasoline, Engine_Diesel, Trans_Manual]")
print()
print("3. The full regression equation is:")
print("   MPG = w₀ + w₁ × Weight_kg + w₂ × Engine_Gasoline + w₃ × Engine_Diesel + w₄ × Trans_Manual + ε")
print()
print("4. The coefficient w₃ for Engine_Diesel can be interpreted as:")
print(f"   The effect on MPG of having a Diesel engine compared to a Hybrid engine (reference),")
print(f"   holding weight and transmission type constant. Since w₃ = {coefficients[3]:.2f},")
print(f"   Diesel engines reduce MPG by {abs(coefficients[3]):.2f} units compared to Hybrid engines,")
print("   all else being equal.")

print("\nSaved visualizations to:", save_dir)
print("Generated images:")
print("- categorical_counts.png: Bar chart showing distribution of categorical variables")
print("- dummy_encoding.png: Heatmap visualizing the dummy variable encoding")
print("- mpg_by_category.png: Box plots showing MPG by engine type and transmission")
print("- mpg_by_weight_categories.png: Scatter plot of MPG vs weight with categories")
print("- design_matrix.png: Heatmap of the design matrix X")
print("- design_matrix_structure.png: Visual representation of the design matrix structure")
print("- regression_coefficients.png: Bar chart showing regression coefficients")
print("- actual_vs_predicted.png: Scatter plot of actual vs. predicted MPG")
print("- residuals.png: Bar chart of model residuals")
print("- engine_effect.png: Line plot showing the effect of engine type on MPG")
print("- all_categorical_effects.png: Bar chart comparing all categorical combinations") 