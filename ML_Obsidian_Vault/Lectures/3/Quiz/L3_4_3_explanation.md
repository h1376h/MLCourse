# Question 3: Feature Engineering for Crop Yield Prediction

## Problem Statement
You are analyzing the factors that influence crop yield. Your dataset includes the following variables:
- $x_1$: Amount of fertilizer (kg/hectare)
- $x_2$: Amount of water (liters/day)
- $x_3$: Average daily temperature (°C)
- $y$: Crop yield (tons/hectare)

Initial analysis suggests that:
1. More fertilizer generally increases yield, but the effect depends on water amount
2. Higher temperatures improve yield up to a point, after which they become harmful
3. The effect of water on yield diminishes as more water is added

### Task
1. Propose a multiple regression model that includes appropriate interaction terms between fertilizer and water to capture their joint effect
2. Suggest a feature transformation for temperature to model the diminishing returns and eventual negative impact
3. Propose a feature transformation for water to capture the diminishing returns effect
4. Write the complete equation for your proposed model including all main effects, interaction terms, and transformed features

## Understanding the Problem

This problem focuses on feature engineering for crop yield prediction. The initial analysis indicates complex relationships between the predictors and crop yield that cannot be adequately captured with a standard linear model. We need to:
- Model the interaction between fertilizer and water
- Account for the non-linear (likely quadratic) effect of temperature
- Capture the diminishing returns of water through an appropriate transformation

Effective feature engineering will help us create a regression model that better reflects the underlying agricultural relationships and improves prediction accuracy.

## Solution

### Step 1: Incorporating an Interaction Term for Fertilizer and Water

The interaction between fertilizer and water (insight #1) suggests that the effect of fertilizer on crop yield depends on the amount of water available. This makes intuitive sense in agriculture: fertilizer effectiveness often depends on sufficient water to dissolve and transport nutrients to plant roots.

To model this interaction, we can add an interaction term to our regression equation:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \times x_2) + \beta_4 x_3 + \epsilon$$

Where:
- $\beta_0$ is the intercept
- $\beta_1$ is the coefficient for fertilizer
- $\beta_2$ is the coefficient for water
- $\beta_3$ is the coefficient for the interaction between fertilizer and water
- $\beta_4$ is the coefficient for temperature
- $\epsilon$ is the error term

From our simulated data analysis, the coefficient for the interaction term ($\beta_3$) is positive (approximately 0.002), indicating that the positive effect of fertilizer on crop yield increases with higher water levels. This makes agricultural sense: more water helps crops better utilize the nutrients from fertilizer.

![Interaction Effect Visualization](../Images/L3_4_Quiz_3/interaction_effect.png)

This visualization shows how the effect of fertilizer on crop yield differs at various water levels. The steeper slopes at higher water levels indicate that fertilizer has a greater effect when more water is available.

### Step 2: Modeling the Non-Linear Temperature Relationship

The second insight suggests an optimal temperature for crop growth, with reduced yields at both lower and higher temperatures. This type of relationship is typically modeled using a quadratic (second-degree polynomial) term.

We can extend our model to include a quadratic term for temperature:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \times x_2) + \beta_4 x_3 + \beta_5 x_3^2 + \epsilon$$

The quadratic term $x_3^2$ with a negative coefficient $\beta_5$ creates a concave (inverted U-shaped) relationship, which can capture the concept of an optimal temperature.

From our simulated data analysis, the coefficient for the quadratic temperature term ($\beta_5$) is negative (approximately -0.08), confirming the expected concave relationship. We can estimate the optimal temperature by taking the derivative of the yield with respect to temperature and setting it to zero:

$$\frac{\partial y}{\partial x_3} = \beta_4 + 2\beta_5 x_3 = 0$$

Solving for $x_3$:

$$x_3 = -\frac{\beta_4}{2\beta_5} \approx -\frac{4.2}{2 \times (-0.08)} \approx 26.25°C$$

This means that crop yield is expected to be maximized at a temperature of approximately 26.25°C, which aligns with the optimal growing temperatures for many common crops.

![Temperature Effect Visualization](../Images/L3_4_Quiz_3/temperature_effect.png)

This plot shows the relationship between temperature and crop yield, clearly illustrating the inverted U-shape with a peak at the optimal temperature.

### Step 3: Capturing Diminishing Returns from Water

The third insight indicates that the benefit of additional water shows diminishing returns. This type of relationship is often modeled using a logarithmic transformation of the variable.

We can modify our model to include a logarithmic transformation of the water variable:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 \log(x_2) + \beta_3 (x_1 \times \log(x_2)) + \beta_4 x_3 + \beta_5 x_3^2 + \epsilon$$

Note that we've also updated the interaction term to use $\log(x_2)$ rather than $x_2$, maintaining consistency in how water is represented in the model.

The logarithmic transformation of water captures the diminishing returns effect: each additional unit of water produces a smaller increase in crop yield. This makes agricultural sense because once crops have sufficient water, adding more provides progressively smaller benefits.

![Water Effect Visualization](../Images/L3_4_Quiz_3/water_effect.png)

This graph shows how crop yield increases with water amount, illustrating the diminishing returns effect. The logarithmic curve (in blue) provides a better fit to the data compared to a linear relationship (in red).

Our data analysis shows that using a logarithmic transformation for water improves the model's R² from 0.72 (linear water term) to 0.81 (logarithmic water term), confirming that this transformation better captures the true relationship.

### Step 4: Complete Model Equation

The complete model equation incorporating all the transformations and interaction terms is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 \log(x_2) + \beta_3 (x_1 \times \log(x_2)) + \beta_4 x_3 + \beta_5 x_3^2 + \epsilon$$

Based on our simulated data analysis, the estimated coefficients are:
- $\beta_0 \approx -25.6$ (intercept)
- $\beta_1 \approx -0.9$ (fertilizer)
- $\beta_2 \approx 4.5$ (log of water)
- $\beta_3 \approx 0.25$ (interaction between fertilizer and log of water)
- $\beta_4 \approx 4.2$ (temperature)
- $\beta_5 \approx -0.08$ (temperature squared)

Therefore, the final model equation is:

$$\hat{y} = -25.6 - 0.9 x_1 + 4.5 \log(x_2) + 0.25 (x_1 \times \log(x_2)) + 4.2 x_3 - 0.08 x_3^2$$

This model captures all three insights from our initial analysis:
1. The interaction between fertilizer and water through the term $0.25 (x_1 \times \log(x_2))$
2. The optimal temperature effect through the terms $4.2 x_3 - 0.08 x_3^2$
3. The diminishing returns of water through the logarithmic transformation $4.5 \log(x_2)$

![Model Predictions](../Images/L3_4_Quiz_3/model_predictions.png)

This visualization compares the actual crop yields to those predicted by our model, showing a strong fit to the data.

## Practical Implementation

### Running the Model

In practice, this model can be implemented using standard regression software after creating the necessary transformed variables and interaction terms:

1. Create a log-transformed water variable: $\log(x_2)$
2. Create the interaction term: $x_1 \times \log(x_2)$
3. Create the squared temperature term: $x_3^2$
4. Fit the regression model using these variables

The model allows for practical agricultural predictions and insights:

- **Finding Optimal Combinations**: For a given temperature, we can determine the most cost-effective combination of fertilizer and water to maximize crop yield.

- **Temperature Adaptation**: The model can predict how crop yields will change under different temperature conditions, helping farmers adapt to climate variations.

- **Resource Optimization**: By understanding the diminishing returns of water, farmers can optimize irrigation to balance yield with water conservation.

- **Customized Recommendations**: The interaction term allows for customized fertilizer recommendations based on available water resources.

### Model Validation and Diagnostics

During implementation, we should:

1. Check residual plots to ensure our transformations adequately capture the non-linear relationships
2. Validate that the interaction and quadratic terms are statistically significant
3. Test the model on holdout data to assess prediction accuracy
4. Ensure the model's predictions align with agricultural knowledge and experience

## Key Insights

### Theoretical Foundations

- **Interaction Terms**: Interaction terms in regression allow the effect of one variable (fertilizer) to depend on the level of another variable (water). This models synergistic relationships common in biological systems.

- **Polynomial Features**: Quadratic terms can capture "goldilocks" phenomena (not too little, not too much) often seen in biological processes, like the optimal temperature for plant growth.

- **Logarithmic Transformations**: Log transformations are excellent for modeling diminishing returns and are widely used for resource inputs in biological and economic models.

### Modeling Considerations

- **Agricultural Knowledge Integration**: The feature engineering decisions are grounded in agricultural science, with mathematical forms that match known biological relationships.

- **Interpretability vs. Complexity**: The model balances complexity (by adding non-linear terms) with interpretability (by using transformations with clear agricultural meaning).

- **Model Selection**: The improved R² values confirm that our transformations better capture the underlying relationships compared to a simple linear model.

### Biological Significance

- **Water-Fertilizer Relationship**: The positive interaction term confirms that proper irrigation improves fertilizer efficiency, a key principle in agricultural management.

- **Optimal Temperature**: The quadratic temperature term captures the concept of cardinal temperatures in plant growth (minimum, optimal, and maximum temperatures).

- **Water Efficiency**: The logarithmic relationship with water aligns with plant physiology, where water uptake efficiency decreases as soil moisture increases.

### Practical Applications

- **Climate Adaptation**: The model can help predict how changes in temperature patterns might affect crop yields.

- **Precision Agriculture**: The model supports variable rate applications of fertilizer based on soil moisture levels.

- **Resource Conservation**: Understanding diminishing returns helps optimize water usage, especially important in water-scarce regions.

- **Economic Optimization**: The model can be combined with cost data to determine economically optimal input levels.

## Conclusion

Our enhanced multiple regression model for crop yield prediction successfully incorporates all three key insights from the initial analysis:

1. We included an interaction term between fertilizer and water (transformed as log of water) to capture how water availability affects fertilizer effectiveness.

2. We added a quadratic term for temperature to model the non-linear relationship with an optimal temperature for crop growth.

3. We applied a logarithmic transformation to the water variable to account for diminishing returns as water quantity increases.

The complete model equation is:
$$\hat{y} = -25.6 - 0.9 x_1 + 4.5 \log(x_2) + 0.25 (x_1 \times \log(x_2)) + 4.2 x_3 - 0.08 x_3^2$$

This model provides a more realistic representation of the complex relationships in agricultural systems than a simple linear model would. By engineering features based on agricultural knowledge, we've created a model that not only fits the data better but also produces predictions that align with established principles of crop science. 