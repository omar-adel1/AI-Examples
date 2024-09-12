# Canada Per Capita Income Prediction

This project implements a linear regression model to predict the per capita income in Canada based on historical data. The model uses gradient descent for optimization and includes functionality for normalization of data and error rate calculation.

## Dataset

The dataset used is `canada_per_capita_income.csv`, which contains the following columns:

- **year**: The year of the data point.
- **income**: The per capita income in that year.

The dataset is used to build a linear regression model to predict per capita income for future years.

## Project Description

1. **Data Preparation**:
   - Read the dataset and normalize the 'year' and 'income' columns.

2. **Model Training**:
   - Implement gradient descent to optimize the parameters (slope `m` and intercept `b`) of the linear regression model.
   - Train the model over a specified number of epochs.

3. **Prediction**:
   - Predict the per capita income for the year 2020 using the trained model.

4. **Evaluation**:
   - Calculate the average error rate and model accuracy.

5. **Visualization**:
   - Plot the normalized data points and the regression line.

## Code Description

- **Data Normalization**:
  The 'year' and 'income' columns are normalized to fit within the range [0, 1].

- **Gradient Descent**:
  Updates the parameters `m` and `b` to minimize the error between the predicted and actual values.

- **Error Calculation**:
  Computes the average error rate of the model predictions.

- **Visualization**:
  Uses Matplotlib to display a scatter plot of the normalized data points and the fitted regression line.
