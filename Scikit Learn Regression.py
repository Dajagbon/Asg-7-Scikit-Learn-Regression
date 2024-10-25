# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
linear_reg = LinearRegression()
svr = SVR()  # Consider tuning parameters like kernel, C, epsilon
rf = RandomForestRegressor(random_state=42)
knn = KNeighborsRegressor()
gb = GradientBoostingRegressor(random_state=42)

# Train the models
linear_reg.fit(X_train, y_train)
svr.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Make predictions
y_pred_linear_reg = linear_reg.predict(X_test)
y_pred_svr = svr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Evaluate the models
models = {
    "Linear Regression": y_pred_linear_reg,
    "Support Vector Regression": y_pred_svr,
    "Random Forest Regressor": y_pred_rf,
    "K-Nearest Neighbors Regressor": y_pred_knn,
    "Gradient Boosting Regressor": y_pred_gb
}

for model_name, y_pred in models.items():
    print(f"Model: {model_name}")
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("\n")

# Plot the regression results
plt.figure(figsize=(10, 8))
for model_name, y_pred in models.items():
    plt.subplot(3, 2, list(models.keys()).index(model_name) + 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
    plt.title(f'Regression Results for {model_name}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()
