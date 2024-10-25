# Asg-7-Scikit-Learn-Regression Diabetes Regression Models

## Project Overview

The gradient-boosting regressor model performs best among the models in my script. It has the best mean squared error and r-squared score. The gradient boosting regressor has a mean squared error of 2898.4. For mean squared error, a score close to 0 is considered good. A perfect model would have an MSE of 0, which is rarely achievable in real-world applications. The gradient-boosting regressor model has an r-squared score of 0.4529. A weak performance r2 score is Below 0.5. It might be useful for rough predictions, but the model isn't capturing much of the underlying pattern.

This project aims to explore and compare the performance of various machine learning regression models on the diabetes dataset provided by Scikit Learn. The goal is to predict the progression of diabetes based on various medical features. The models used in this project include:

- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regressor
- K-Nearest Neighbors Regressor (KNN)
- Gradient Boosting Regressor

## Dataset

The dataset used in this project is the diabetes dataset from Scikit Learn. It contains 10 baseline variables (features) and a target variable that measures the progression of diabetes one year after baseline.

## Project Structure

### `Scikit Learn Regression.py`

This script contains the implementation of the regression models and their evaluation. Below is a detailed explanation of the class design, attributes, methods, and any limitations.

### Class Design and Implementation

The script does not use custom classes but follows a procedural approach. Here is a breakdown of the key components:

#### Libraries Imported

- [`pandas`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition") and [`numpy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For data manipulation and numerical operations.
- [`sklearn.datasets`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): To load the diabetes dataset.
- [`sklearn.model_selection`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For splitting the dataset into training and testing sets.
- [`sklearn.linear_model`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For Linear Regression.
- [`sklearn.svm`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For Support Vector Regression.
- [`sklearn.ensemble`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For Random Forest and Gradient Boosting Regressors.
- [`sklearn.neighbors`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For K-Nearest Neighbors Regressor.
- [`sklearn.metrics`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A5%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For evaluating the models using Mean Squared Error and R-squared metrics.
- [`seaborn`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition") and [`matplotlib.pyplot`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"): For plotting the regression results.

#### Data Loading and Splitting

- **[`load_diabetes()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A3%2C%22character%22%3A29%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Loads the diabetes dataset.
- **[`train_test_split()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A4%2C%22character%22%3A36%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Splits the dataset into training and testing sets with an 80-20 ratio.

#### Model Initialization

- **[`LinearRegression()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A5%2C%22character%22%3A33%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Initializes the Linear Regression model.
- **[`SVR()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A6%2C%22character%22%3A24%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Initializes the Support Vector Regression model.
- **[`RandomForestRegressor(random_state=42)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A29%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Initializes the Random Forest Regressor with a fixed random state for reproducibility.
- **[`KNeighborsRegressor()`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A8%2C%22character%22%3A30%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Initializes the K-Nearest Neighbors Regressor.
- **[`GradientBoostingRegressor(random_state=42)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A7%2C%22character%22%3A52%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Initializes the Gradient Boosting Regressor with a fixed random state for reproducibility.

#### Model Training

- **`fit(X_train, y_train)`**: Trains each model using the training data.

#### Model Prediction

- **[`predict(X_test)`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A36%2C%22character%22%3A31%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")**: Makes predictions using the testing data for each model.

#### Model Evaluation

- **`mean_squared_error(y_test, y_pred)`**: Calculates the Mean Squared Error for each model.
- **`r2_score(y_test, y_pred)`**: Calculates the R-squared value for each model.

#### Plotting

- **`sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)`**: Creates scatter plots to visualize the true vs. predicted values for each model.
- **`plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)`**: Adds a reference line to the plots.

### Limitations

1. **Parameter Tuning**: The models are used with their default parameters. Tuning parameters like [`kernel`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A47%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"), [`C`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A55%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"), and [`epsilon`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A23%2C%22character%22%3A58%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition") for SVR, or the number of trees for Random Forest, could improve performance.
2. **Standardization**: The SVR model might perform better with standardized data, which is not done in this script.
3. **Model Selection**: Only a few regression models are explored. Other models like Ridge, Lasso, or ElasticNet could also be considered.
4. **Cross-Validation**: The script uses a simple train-test split. Cross-validation could provide a more robust evaluation of model performance.

## How to Run

1. Ensure you have Python installed along with the necessary libraries ([`pandas`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A1%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"), [`numpy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A2%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"), `scikit-learn`, [`seaborn`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A10%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition"), [`matplotlib`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FC%3A%2FUsers%2Fdanie%2FDownloads%2FPythonclasstwo%2FAsg%207%2FAsg%2FScikit%20Learn%20Regression.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A11%2C%22character%22%3A7%7D%7D%5D%2C%22fc8a19de-f4d1-4ef5-b1bd-ffa08c8b01d7%22%5D "Go to definition")).
2. Run the `Scikit Learn Regression.py` script.
3. The script will print the Mean Squared Error and R-squared values for each model and display scatter plots of the true vs. predicted values.

## Conclusion

This project provides a comparative analysis of different regression models on the diabetes dataset. It serves as a starting point for further exploration and optimization of machine learning models for predicting diabetes progression.
