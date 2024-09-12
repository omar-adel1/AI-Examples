import pandas as pd
import numpy as np

df = pd.read_csv('./Datasets/Social_Network_Ads.csv')

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(old_m1, old_m2, old_b, points, learning_rate):
    gradient_m1 = 0
    gradient_m2 = 0
    gradient_b = 0
    n = len(points)
    for i in range(n):
        x1 = points.iloc[i].Age
        x2 = points.iloc[i].EstimatedSalary
        y = points.iloc[i].Purchased
        linear_model = old_m1 * x1 + old_m2 * x2 + old_b
        y_predicted = _sigmoid(linear_model)
        gradient_m1 += (1/n) * x1 * (y_predicted - y)
        gradient_m2 += (1/n) * x2 * (y_predicted - y)
        gradient_b += (1/n) * (y_predicted - y)
    m1 = old_m1 - learning_rate * gradient_m1
    m2 = old_m2 - learning_rate * gradient_m2
    b = old_b - learning_rate * gradient_b
    return m1, m2, b

def predict(X):
    linear_model = m1 * X[:, 0] + m2 * X[:, 1] + b
    y_predicted = _sigmoid(linear_model)
    return [1 if i > 0.5 else 0 for i in y_predicted]

# Initialize parameters
m1 = 0
m2 = 0
b = 0
iterations = 1000
learning_rate = 0.0001

for i in range(iterations):
    m1, m2, b = gradient_descent(m1, m2, b, df, learning_rate)


training_data = df[['Age', 'EstimatedSalary']].values

train_predictions = predict(training_data)

accuracy = np.mean(train_predictions == df['Purchased'].values)
print(f'Training Accuracy: {accuracy * 100:.2f}%')

new_data = np.array([
    [47, 25000],  
    [40, 120000],
    [50, 150000] 
])

predictions = predict(new_data)
print(f'Predictions for new data: {predictions}')
