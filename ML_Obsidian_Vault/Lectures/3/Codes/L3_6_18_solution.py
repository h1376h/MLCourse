import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Create directory to save figures
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), "Images")
save_dir = os.path.join(images_dir, "L3_6_Quiz_18")
os.makedirs(save_dir, exist_ok=True)

# Set a nice style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Generate synthetic house price data with non-linear relationship
np.random.seed(42)
n_samples = 200

# Generate features that would be relevant for house prices
# X1: size in square feet (normalized)
# X2: number of bedrooms (normalized)
# X3: age of house in years (normalized)
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
X3 = np.random.normal(0, 1, n_samples)

# The true relationship is non-linear
# Price = f(size, bedrooms, age) + noise
# We'll use a non-linear function with interactions
y_true = 3*X1**2 + 2*X1*X2 - 1.5*X3 + 0.5*X1*X3**2 + 10
noise = np.random.normal(0, 2, n_samples)  # Noise component
y = y_true + noise

# Combine features
X = np.column_stack((X1, X2, X3))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Problem 1: Overfitting with a high-degree polynomial model
# This will create the scenario described in the question
poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train a linear regression model on the polynomial features
model_overfit = LinearRegression()
model_overfit.fit(X_train_poly, y_train)

# Predictions
y_train_pred_overfit = model_overfit.predict(X_train_poly)
y_test_pred_overfit = model_overfit.predict(X_test_poly)

# Calculate errors
train_mse_overfit = mean_squared_error(y_train, y_train_pred_overfit)
test_mse_overfit = mean_squared_error(y_test, y_test_pred_overfit)
train_r2_overfit = r2_score(y_train, y_train_pred_overfit)
test_r2_overfit = r2_score(y_test, y_test_pred_overfit)

print("\nOverfit Model Results:")
print(f"Training MSE: {train_mse_overfit:.2f}")
print(f"Test MSE: {test_mse_overfit:.2f}")
print(f"Training R²: {train_r2_overfit:.2f}")
print(f"Test R²: {test_r2_overfit:.2f}")
print(f"Error ratio (Test/Train): {test_mse_overfit/train_mse_overfit:.2f}x")

# Visualization 1: Actual vs. Predicted with Overfitting
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred_overfit, alpha=0.5, label='Training data')
plt.scatter(y_test, y_test_pred_overfit, alpha=0.5, label='Test data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual House Prices', fontsize=12)
plt.ylabel('Predicted House Prices', fontsize=12)
plt.title('Actual vs. Predicted House Prices (Overfit Model)', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, "actual_vs_predicted_overfit.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 2: Residuals vs. Predicted (showing the pattern)
plt.figure(figsize=(10, 6))
residuals_train = y_train - y_train_pred_overfit
residuals_test = y_test - y_test_pred_overfit

plt.scatter(y_train_pred_overfit, residuals_train, alpha=0.5, label='Training residuals')
plt.scatter(y_test_pred_overfit, residuals_test, alpha=0.5, label='Test residuals')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# Add a trend line to highlight the pattern
z = np.polyfit(np.concatenate([y_train_pred_overfit, y_test_pred_overfit]), 
               np.concatenate([residuals_train, residuals_test]), 3)
p = np.poly1d(z)
x_trend = np.linspace(min(np.concatenate([y_train_pred_overfit, y_test_pred_overfit])),
                      max(np.concatenate([y_train_pred_overfit, y_test_pred_overfit])), 100)
plt.plot(x_trend, p(x_trend), "r--", linewidth=2)

plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs. Predicted Values (Overfit Model)', fontsize=14)
plt.legend()
plt.savefig(os.path.join(save_dir, "residuals_vs_predicted.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 3: Learning Curve
def plot_learning_curve(estimator, X, y, title, filename, ylim=None, cv=5,
                       train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
        scoring='neg_mean_squared_error')
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return train_scores_mean, test_scores_mean

# Generate learning curve for the overfit model
estimator = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), LinearRegression())
train_scores, test_scores = plot_learning_curve(
    estimator, X, y, 
    "Learning Curve (Overfit Model)", 
    "learning_curve_overfit.png"
)

# Solution 1: Regularization (Ridge Regression)
from sklearn.linear_model import Ridge

# Create the pipeline with regularization
ridge_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    Ridge(alpha=10.0)  # Alpha controls the regularization strength
)

# Fit the model
ridge_model.fit(X_train, y_train)

# Predictions
y_train_pred_ridge = ridge_model.predict(X_train)
y_test_pred_ridge = ridge_model.predict(X_test)

# Calculate errors
train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

print("\nRegularized Model (Ridge) Results:")
print(f"Training MSE: {train_mse_ridge:.2f}")
print(f"Test MSE: {test_mse_ridge:.2f}")
print(f"Training R²: {train_r2_ridge:.2f}")
print(f"Test R²: {test_r2_ridge:.2f}")
print(f"Error ratio (Test/Train): {test_mse_ridge/train_mse_ridge:.2f}x")

# Visualization 4: Comparing residuals before and after regularization
plt.figure(figsize=(12, 5))

# Before regularization
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred_overfit, residuals_test, alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Predicted Values', fontsize=10)
plt.ylabel('Residuals', fontsize=10)
plt.title('Residuals Before Regularization', fontsize=12)

# After regularization
plt.subplot(1, 2, 2)
residuals_test_ridge = y_test - y_test_pred_ridge
plt.scatter(y_test_pred_ridge, residuals_test_ridge, alpha=0.7, color='green')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Predicted Values', fontsize=10)
plt.ylabel('Residuals', fontsize=10)
plt.title('Residuals After Regularization', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "residuals_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Solution 2: Feature Engineering (add non-linear transformations)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create features that better model the underlying relationship
X_train_engineered = np.column_stack((
    X_train,  # Original features
    X_train[:, 0]**2,  # Size squared
    X_train[:, 0] * X_train[:, 1],  # Size x Bedrooms interaction
    X_train[:, 0] * X_train[:, 2]**2  # Size x Age^2 interaction
))

X_test_engineered = np.column_stack((
    X_test,  # Original features
    X_test[:, 0]**2,  # Size squared
    X_test[:, 0] * X_test[:, 1],  # Size x Bedrooms interaction
    X_test[:, 0] * X_test[:, 2]**2  # Size x Age^2 interaction
))

# Create a pipeline with scaling and linear regression
feature_engineering_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# Fit the model
feature_engineering_model.fit(X_train_engineered, y_train)

# Predictions
y_train_pred_fe = feature_engineering_model.predict(X_train_engineered)
y_test_pred_fe = feature_engineering_model.predict(X_test_engineered)

# Calculate errors
train_mse_fe = mean_squared_error(y_train, y_train_pred_fe)
test_mse_fe = mean_squared_error(y_test, y_test_pred_fe)
train_r2_fe = r2_score(y_train, y_train_pred_fe)
test_r2_fe = r2_score(y_test, y_test_pred_fe)

print("\nFeature Engineering Model Results:")
print(f"Training MSE: {train_mse_fe:.2f}")
print(f"Test MSE: {test_mse_fe:.2f}")
print(f"Training R²: {train_r2_fe:.2f}")
print(f"Test R²: {test_r2_fe:.2f}")
print(f"Error ratio (Test/Train): {test_mse_fe/train_mse_fe:.2f}x")

# Visualization 5: Learning Curve for Feature Engineering Model
feature_eng_pipeline = Pipeline([
    ('features', 'passthrough'),
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

def custom_feature_transformer(X):
    return np.column_stack((
        X,
        X[:, 0]**2,
        X[:, 0] * X[:, 1],
        X[:, 0] * X[:, 2]**2
    ))

# Plot learning curve for feature engineered model
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.column_stack((
            X,
            X[:, 0]**2,
            X[:, 0] * X[:, 1],
            X[:, 0] * X[:, 2]**2
        ))

fe_pipeline = Pipeline([
    ('features', CustomFeatureTransformer()),
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

train_scores_fe, test_scores_fe = plot_learning_curve(
    fe_pipeline, X, y, 
    "Learning Curve (Feature Engineering Model)", 
    "learning_curve_feature_engineering.png"
)

# Visualization 6: Compare all models' performance
plt.figure(figsize=(12, 8))

models = ['Overfit Model', 'Regularized (Ridge)', 'Feature Engineering']
train_mse = [train_mse_overfit, train_mse_ridge, train_mse_fe]
test_mse = [test_mse_overfit, test_mse_ridge, test_mse_fe]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_mse, width, label='Training MSE')
plt.bar(x + width/2, test_mse, width, label='Test MSE')

plt.axhline(y=np.var(y), color='r', linestyle='--', label='Baseline (Variance)')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14)
plt.xticks(x, models)
plt.legend()

# Add error ratio as text
for i, (train, test) in enumerate(zip(train_mse, test_mse)):
    ratio = test/train
    plt.text(i, max(train, test) + 1, f'Ratio: {ratio:.2f}x', ha='center')

plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# Visualization 7: Cross-validation with different amounts of training data
from sklearn.model_selection import cross_val_score

cv_sizes = np.linspace(0.1, 1.0, 10)
cv_scores_ridge = []
cv_scores_fe = []

for size in cv_sizes:
    n_samples_cv = int(size * len(X_train))
    X_train_subset = X_train[:n_samples_cv]
    y_train_subset = y_train[:n_samples_cv]
    
    # Ridge model
    ridge_cv = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=10.0)
    )
    scores_ridge = cross_val_score(ridge_cv, X_train_subset, y_train_subset, 
                                   cv=5, scoring='neg_mean_squared_error')
    cv_scores_ridge.append(-np.mean(scores_ridge))
    
    # Feature Engineering model
    X_train_subset_fe = np.column_stack((
        X_train_subset,
        X_train_subset[:, 0]**2,
        X_train_subset[:, 0] * X_train_subset[:, 1],
        X_train_subset[:, 0] * X_train_subset[:, 2]**2
    ))
    fe_cv = Pipeline([
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])
    scores_fe = cross_val_score(fe_cv, X_train_subset_fe, y_train_subset, 
                               cv=5, scoring='neg_mean_squared_error')
    cv_scores_fe.append(-np.mean(scores_fe))

plt.figure(figsize=(10, 6))
plt.plot(cv_sizes * len(X_train), cv_scores_ridge, 'o-', label='Regularization (Ridge)')
plt.plot(cv_sizes * len(X_train), cv_scores_fe, 'o-', label='Feature Engineering')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Mean Squared Error (Cross-Validation)', fontsize=12)
plt.title('Effect of Training Set Size on Model Performance', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "training_size_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to: {save_dir}")

# Summary of findings
print("\nSummary of Findings:")
print("1. The original model showed classic signs of overfitting:")
print(f"   - Low training error (MSE: {train_mse_overfit:.2f})")
print(f"   - High test error (MSE: {test_mse_overfit:.2f})")
print(f"   - High test/train error ratio ({test_mse_overfit/train_mse_overfit:.2f}x)")
print("   - Clear patterns in the residuals")
print("   - Learning curve showing validation error plateauing")

print("\n2. Solution 1 - Regularization (Ridge):")
print(f"   - Training MSE: {train_mse_ridge:.2f}")
print(f"   - Test MSE: {test_mse_ridge:.2f}")
print(f"   - Test/Train ratio: {test_mse_ridge/train_mse_ridge:.2f}x")
print("   - Reduced overfitting by penalizing large coefficients")

print("\n3. Solution 2 - Feature Engineering:")
print(f"   - Training MSE: {train_mse_fe:.2f}")
print(f"   - Test MSE: {test_mse_fe:.2f}")
print(f"   - Test/Train ratio: {test_mse_fe/train_mse_fe:.2f}x")
print("   - Addressed model misspecification by adding appropriate non-linear terms")

print("\n4. Evaluation technique:")
print("   - Cross-validation provides the most reliable assessment")
print("   - Learning curves show how model performance varies with training size")
print("   - Residual plots help identify patterns and model misspecification")

print("\n5. More data vs. More features:")
print("   - In this case, adding the right features (addressing model specification)")
print("   - was more effective than just adding more training examples")
print("   - This is evident from the learning curves, where validation error plateaus") 