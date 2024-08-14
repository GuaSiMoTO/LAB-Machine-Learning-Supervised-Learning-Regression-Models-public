# LAB-Supervised-Learning-Regression-Models

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

#### 3.1 Explore Correlation

Explore the correlation between the features and the target variable (`cnt`) to identify which features are most strongly associated with bike rentals.

- Generate and visualize a correlation matrix.
- Select features that show a high correlation with the target variable for further analysis.

#### 3.2 Explore Multicollinearity

Check for multicollinearity among the selected features using the Variance Inflation Factor (VIF).

- Calculate VIF for each feature to assess multicollinearity.

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

Optimize the Polynomial Ridge Regression model by searching for the best hyperparameters using GridSearchCV.

- Perform Grid Search to tune the polynomial degree and `alpha` parameter for Ridge Regression.
- Evaluate the model using MSE and R² metrics.

### Step 10: Compare the MSE and R² Results for All Models

After obtaining the optimized Ridge Regression model, compare its performance with the Linear, Polynomial, Lasso, and initial Ridge Regression models.

- Print and compare the MSE and R² scores for all models.