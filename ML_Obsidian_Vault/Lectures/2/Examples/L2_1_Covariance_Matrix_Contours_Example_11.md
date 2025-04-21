### Example 11: Geometric Area Interpretation of Covariance

#### Problem Statement
How can we visualize covariance as a geometric area to provide an intuitive understanding?

#### Solution

##### Step 1: Calculating Means and Centering Data
We center the data by subtracting the means from each point:
- Mean calculation: μₓ = average of all x values, μᵧ = average of all y values
- Centering: x' = x - μₓ, y' = y - μᵧ

##### Step 2: Visualizing Covariance as Area
Covariance can be understood as the average "signed area" of rectangles formed by:
- Width = deviation in x from mean (x - μₓ)
- Height = deviation in y from mean (y - μᵧ)

For positive correlation:
- Most areas are positive (1st and 3rd quadrants)
- Covariance = 1.06
- Correlation = 0.87

For zero correlation:
- Positive and negative areas cancel out
- Covariance = -0.36
- Correlation = -0.19

For negative correlation:
- Most areas are negative (2nd and 4th quadrants)
- Covariance = -0.48
- Correlation = -0.85

![Geometric Area](../Images/Contour_Plots/ex11_correlation_geometric.png)

## Running the Examples

You can run the code that generates these examples and visualizations using:

```bash
python3 ML_Obsidian_Vault/Lectures/2/Codes/L2_1_CMC_example_11_geometric_area_covariance.py
```