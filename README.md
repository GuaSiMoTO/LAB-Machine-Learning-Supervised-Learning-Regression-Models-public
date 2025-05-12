# LAB-Machine-Learning-Supervised-Learning-Regression-Models

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)


# Bike Sharing Dataset Analysis

## Introduction

In this project, we will walk through a comprehensive Machine Learning workflow using the Bike Sharing dataset. The dataset contains daily records of bike rentals in a specific city. We will cover data collection, preprocessing, feature engineering, model training, evaluation, and comparison of different regression techniques. The goal is to build and compare Linear and Polynomial Regression models, along with regularization techniques like Lasso and Ridge, to predict the total number of bike rentals (`cnt`). Additionally, we will explore hyperparameter tuning using GridSearchCV to optimize the model performance.

## Setup Instructions

To run the notebook and reproduce the results, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/codeskills-ai/LAB-Supervised-Learning-Regression-Models.git
    cd LAB-Supervised-Learning-Regression-Models
    ```

2. **Create a virtual environment and install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

## Steps in the Workflow

### Step 1: Data Load

Load the Bike Sharing Dataset, which contains daily records of bike rentals in a specific city. Use Python libraries to download, unzip, and load the dataset into a pandas DataFrame for further analysis.

- Import the necessary Python libraries: `pandas`, `zipfile`, `io`, and `requests`.
- Download the dataset using the provided URL and unzip it.
- Load the `day.csv` file, which contains the daily records, into a pandas DataFrame.
- Display the first few rows of the dataset to understand its structure and the type of data it contains.

### Step 2: Explore Data

In this section, we will explore the dataset visually to understand the distribution of the target variable (`cnt`) and its relationship with other features. Visualization helps in identifying patterns, trends, and potential correlations between variables.

- Import the necessary visualization libraries: `seaborn` and `matplotlib.pyplot`.
- Visualize the distribution of the target variable (`cnt`) using a histogram.
- Use a pairplot to generate scatter plots for some features (e.g., `temp`, `atemp`, `hum`, `windspeed`) against the target variable (`cnt`).

### Step 3: Feature Selection

#### 3.0 Remove the 'dteday' column

Remove the `dteday` column from the dataset as it represents the date and is not needed for our analysis.

## 3.1 Explore the correlation with the target variable and select the best features

In this step, we will explore the correlation between the features and the target variable (`cnt`). This helps in identifying which features are most strongly associated with the number of bike rentals.

- Generate a correlation matrix to examine the relationships between all features and the target variable.
- Visualize the correlation matrix using a heatmap to easily identify the strength of the relationships.
- Select features that show a high correlation with the target variable (cnt) for further analysis. For this, we will manually select features with a correlation coefficient greater than 0.5. Later, we will use regularization techniques (like Lasso or Ridge) to refine the selection and further reduce multicollinearity.

## 3.2 Explore multicollinearity among features

Multicollinearity occurs when features are highly correlated with each other, which can negatively affect the performance of regression models. When features are multicollinear, it becomes difficult for the model to determine the individual effect of each feature, as they are providing overlapping information. To handle this issue and improve the model's accuracy, we need to check for multicollinearity among the selected features.

### Steps to follow:

1. **Calculate the Variance Inflation Factor (VIF)** for each feature. The VIF measures how much the variance of the estimated regression coefficients increases when your predictors are correlated. A high VIF indicates that the feature is highly correlated with other features in the dataset. Generally, a VIF value above **5 or 10** suggests problematic multicollinearity.

2. **Interpret the VIF values**:
   - Features with a **low VIF** (close to 1) are **not highly correlated** with others and are safe to keep in the model.
   - Features with **high VIF** values are **highly correlated** with other features and should be considered for removal. Common practice is to remove features with a VIF higher than **10**, as they contribute to multicollinearity and may distort the results.

3. **Decision Making**:
   - **Remove features with excessively high VIF** (e.g., above 10). For example, if `temp` and `atemp` have high VIFs and are very correlated, consider keeping only one of them (e.g., `temp`).
   - If some features, like `instant` or `registered`, have a high VIF but are essentially **redundant** (e.g., `registered` and `casual`), remove the redundant variables to avoid overfitting.
   - Keep the features that provide unique information and have reasonable VIFs (e.g., around 1 to 5).

4. **Example Features to Keep**:
   After analyzing the VIFs, **you may decide to keep the features that show the lowest multicollinearity**. For instance, if `season`, `yr`, and `casual` have VIFs below 5 or 10, they might remain in your model. However, features like `instant`, `registered`, or `atemp` might need to be removed due to their high VIF.

### Step 4: Generate a Linear Regression Model

Create a linear regression model to predict the total number of bike rentals (`cnt`) using the selected features.

- Import the necessary libraries: `train_test_split`, `LinearRegression`, `mean_squared_error`, and `r2_score`.
- Split the data into training and testing sets.
- Train the linear regression model on the training data.

#### 4.1 Plot the Result

Visualize the predictions against the actual values to assess model performance.

#### 4.2 Evaluate the Model

Evaluate the model's performance using Mean Squared Error (MSE) and R-squared (R²) metrics.

### Step 5: Generate a Polynomial Regression Model

Extend the linear regression model by generating a polynomial regression model of degree 2 to capture non-linear relationships.

- Create polynomial features from the selected features and train the polynomial regression model.

#### 5.1 Plot the Results

Visualize the predictions against the actual values to assess how well the model fits the data.

#### 5.2 Evaluate the Model

Evaluate the polynomial regression model's performance using MSE and R² metrics.

### Step 6: Compare Linear and Polynomial Regression Models

Compare the performance of the Linear Regression (LR) and Polynomial Regression (PR) models.

- Compare MSE and R² scores to determine which model better fits the data.

### Step 7: Use Lasso and Ridge to Test the Features

Apply Lasso and Ridge regression to both linear and polynomial regression models to reduce overfitting and improve generalization.

#### 7.1 Lasso Regression

- Apply Lasso Regression to linear and polynomial features and evaluate the model performance.

#### 7.2 Ridge Regression

- Apply Ridge Regression to linear and polynomial features and evaluate the model performance.

### Step 8: Compare the Results of All Models

Compare the performance of Linear, Polynomial, Ridge, and Lasso Regression models using MSE and R² metrics.

- Discuss the results and determine the best model based on predictive accuracy and generalization.

### Step 9: Use GridSearchCV to Find the Best Model

## 9.1 Perform Grid Search for Ridge Regression

In this step, we will perform a **Grid Search** to tune the **Ridge Regression model** with polynomial features. The goal is to find the best hyperparameter (`alpha`) for the model using cross-validation.

1. **Import the necessary libraries**:
   - Import `GridSearchCV` from `sklearn.model_selection` for performing the grid search.
   - Import `PolynomialFeatures` from `sklearn.preprocessing` to create polynomial features.
   - Import `Ridge` from `sklearn.linear_model` for the Ridge Regression model.
   - Use `Pipeline` from `sklearn.pipeline` to chain the polynomial transformation and Ridge regression together.

2. **Define the parameter grid** for Ridge Regression:
   - We will tune the `alpha` parameter, which controls the strength of the regularization. You can define a grid of possible `alpha` values to try, such as `[0.1, 1.0, 10.0, 100.0]`.

3. **Use `GridSearchCV`** to find the best `alpha` value:
   - `GridSearchCV` will perform cross-validation to search for the best `alpha` by training and evaluating the model on the training data.
   
4. **Train the Ridge Regression model**:
   - After performing the grid search, train the model using the best `alpha` parameter found by `GridSearchCV`.

5. **Evaluate the model**:
   - Use **Mean Squared Error (MSE)** and **R-squared (R²)** metrics to evaluate the performance of the model on the test data.

### Step 10: Compare the MSE and R² Results for All Models

After obtaining the optimized Ridge Regression model, compare its performance with the Linear, Polynomial, Lasso, and initial Ridge Regression models.

- Print and compare the MSE and R² scores for all models.
