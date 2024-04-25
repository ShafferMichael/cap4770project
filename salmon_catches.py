import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Adjust the paths to where you've saved your files
salmon_path = 'SalmonandSeaTroutNets1952-2022.csv'
temperature_path = 'ohc_levitus_climdash_seasonal.csv'

salmon_data = pd.read_csv(salmon_path)
temperature_data = pd.read_csv(temperature_path)

# Preprocessing
# Assuming the temperature data needs to be averaged over each year and the salmon data is already annual
# temperature_data['Year'] = pd.to_datetime(temperature_data['Date']).dt.year
# average_yearly_temp = temperature_data.groupby('Year').mean().reset_index()
average_yearly_temp = temperature_data.groupby('Year')['heat content anomaly (10^22  Joules)'].mean().reset_index()
average_yearly_temp.columns = ['Year', 'Temperature']  # Renaming for clarity


# Merge datasets on 'Year'
merged_data = pd.merge(salmon_data, average_yearly_temp, on='Year', how='inner')

# Handle missing values - example: fill with mean or drop
# This step highly depends on your dataset specifics
merged_data.fillna(merged_data.mean(), inplace=True)

# You may need to adjust the column names based on your dataset's specific columns

# Plotting the trends
plt.figure(figsize=(12, 6))
plt.plot(merged_data['Year'], merged_data['Temperature'], label='Average Ocean Temperature')
plt.plot(merged_data['Year'], merged_data['SalmonCatch'], label='Salmon Catch', secondary_y=True)
plt.title('Yearly Trends in Ocean Temperature and Salmon Catches')
plt.xlabel('Year')
plt.legend()
plt.show()

# Correlation matrix
correlation_matrix = merged_data[['Temperature', 'SalmonCatch']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Calculate Pearson correlation
correlation_coef, p_value = pearsonr(merged_data['Temperature'], merged_data['SalmonCatch'])
print(f"Pearson correlation coefficient: {correlation_coef:.3f}")
print(f"P-value: {p_value:.3f}")

# Split data
X = merged_data[['Temperature']]  # Predictor variable(s)
y = merged_data['SalmonCatch']  # Response variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 ): {r2:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, predictions, color='red', linewidth=2, label='Predicted')
plt.title('Temperature vs. Salmon Catch')
plt.xlabel('Ocean Temperature')
plt.ylabel('Salmon Catch')
plt.legend()
plt.show()
