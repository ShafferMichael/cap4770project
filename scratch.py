import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
temperature_path = "data/ohc_levitus_climdash_seasonal.csv"
temperature_data = pd.read_csv(temperature_path)

salmon_path = 'data/SalmonandSeaTroutNets1952-2022.csv'
salmon_data = pd.read_csv(salmon_path)

# Create a list to store key-value pairs for yearly temperature anomalies 
yearly_anomaly_pairs = []

# Iterate over rows and populate the list
for index, row in temperature_data.iterrows():
    year = row['Year']
    anomaly = row['heat content anomaly (10^22 Joules)']
    pair = {'Year': year, 'Anomaly': anomaly}  # Create a dictionary for each pair
    yearly_anomaly_pairs.append(pair)

# Convert the list of dictionaries into a DataFrame
yearly_anomaly_df = pd.DataFrame(yearly_anomaly_pairs)

# Merge datasets on 'Year'
merged_data = pd.merge(salmon_data, yearly_anomaly_df, on='Year', how='inner')

# Plotting
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=merged_data, x='Year', y='Wild MSW number', hue='Anomaly', palette='coolwarm', size='Anomaly', sizes=(50, 200))
# plt.title('Wild MSW Number vs. Heat Anomaly')
# plt.xlabel('Year')
# plt.ylabel('Wild MSW Number')
# plt.legend(title='Heat Anomaly')
# plt.grid(True)
# plt.show()

# # Calculate correlation matrix
correlation_matrix = merged_data.corr()
# # Plot correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
# plt.title('Correlation Matrix')
# plt.show()


# Calculate Pearson correlation coefficients
pearson_coefficients = merged_data.corr(method='pearson')
# Plot correlation matrix with Pearson coefficients
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, 
            cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title('Correlation Matrix with Pearson Coefficients')
plt.show()