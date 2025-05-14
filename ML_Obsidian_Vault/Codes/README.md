# Linear Regression Visualization Scripts

This directory contains scripts to generate visualizations for the Linear Regression notes.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the plots directory exists:
```bash
python ensure_plots_directory.py
```

## Running the Visualizations

You can run all visualizations at once:
```bash
python run_all_visualizations.py
```

Or run individual scripts:
```bash
python learning_algorithm_diagram.py
python error_measurement.py
python cost_function_visualization.py
python geometric_interpretation.py
python regularization_comparison.py
python gradient_descent.py
python mse_vs_training_size.py
python linear_regression_training_examples.py
```

## Output

All generated plots will be saved in the `plots/` directory. These plots are referenced in the main Markdown file (`../linear_regression_notes.md`).

## Script Descriptions

- `learning_algorithm_diagram.py`: Generates a flowchart visualization of the learning algorithm process
- `error_measurement.py`: Visualizes how error is measured in linear regression
- `cost_function_visualization.py`: Creates 3D surface and contour plots of the cost function
- `geometric_interpretation.py`: Shows the geometric interpretation of linear regression as projection
- `regularization_comparison.py`: Compares Ridge and Lasso regularization techniques
- `gradient_descent.py`: Visualizes the gradient descent optimization process 
- `mse_vs_training_size.py`: Shows how MSE decreases as the number of training examples increases
- `linear_regression_training_examples.py`: Demonstrates how model fit improves with more training data
- `ensure_plots_directory.py`: Creates the plots directory if it doesn't exist 