import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./Datasets/canada_per_capita_income.csv')

df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
df['income_normalized'] = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())

def calculate_error_rate(m, b, points):
    total_error_rate = 0
    for i in range(len(points)):
        x = points.iloc[i].year_normalized
        actual_y = points.iloc[i].income_normalized
        predicted_y = m * x + b
        error_rate = abs(predicted_y - actual_y) / actual_y * 100
        total_error_rate += error_rate
    return total_error_rate / len(points)

def gradient_descent(old_m, old_b, points, learning_rate):
    gradient_m = 0
    gradient_b = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].year_normalized
        y = points.iloc[i].income_normalized
        gradient_m += -2/n * x * (y - (old_m * x + old_b))
        gradient_b += -2/n * (y - (old_m * x + old_b))
    m = old_m - learning_rate * gradient_m
    b = old_b - learning_rate * gradient_b
    return m, b

# Initialize parameters
m = 0
b = 0
L = 0.0001 
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, df, L)
print(f"Final values: m = {m}, b = {b}")

plt.figure(figsize=(10, 6))
plt.scatter(df.year_normalized, df.income_normalized, color='red', label='Data Points')
plt.plot(df.year_normalized, [m * x + b for x in df.year_normalized], label='Regression Line', color='blue')
plt.xlabel('Year (Normalized)')
plt.ylabel('Per Capita Income (Normalized)')
plt.title('Normalized Per Capita Income in Canada vs Year')
plt.legend()
plt.show()

year_to_predict = 2020
year_normalized = (year_to_predict - df['year'].min()) / (df['year'].max() - df['year'].min())
predicted_income_normalized = m * year_normalized + b
predicted_income = predicted_income_normalized * (df['income'].max() - df['income'].min()) + df['income'].min()
print(f"Predicted per capita income for the year {year_to_predict} is: {predicted_income}")

average_error_rate = calculate_error_rate(m, b, df)
accuracy = 100 - average_error_rate

print(f"Average Error Rate: {average_error_rate}%")
print(f"Model Accuracy: {accuracy}%")