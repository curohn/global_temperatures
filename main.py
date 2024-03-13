# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read in the data
df = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
us_temps = df[df['Country'] == 'United States']

# Clean and prepare the data
us_temps['dt'] = pd.to_datetime(us_temps['dt'])
us_temps.set_index('dt', inplace=True)
# Fill missing values using forward fill
us_temps['AverageTemperature'].fillna(method='ffill', inplace=True)
# Data is inaccurate before 1820, so we will filter it out
us_temps = us_temps[us_temps.index.year > 1820]
yearly_average_temps = us_temps['AverageTemperature'].resample('A').mean()
# Calculate the lower and upper bounds for the uncertainty
lower_bound = yearly_average_temps - us_temps['AverageTemperatureUncertainty'].resample('A').mean()
upper_bound = yearly_average_temps + us_temps['AverageTemperatureUncertainty'].resample('A').mean()


# Analyze the data
X = yearly_average_temps.index.year.values.reshape(-1, 1)  # Input: Year
y = yearly_average_temps.values.reshape(-1, 1)  # Output: Yearly Average Temperature
# Create a LinearRegression object
model = LinearRegression()
# Fit the model to your data
model.fit(X, y)

# Make predictions
predictions = model.predict(X)


# Plot the data
plt.figure(figsize=(10,5))
#plt.plot(yearly_average_temps, label='Yearly Average Temperature')
plt.plot(X, y, label='Yearly Average Temperature')
plt.plot(X, predictions, color='red', label='Linear Regression')
plt.fill_between(yearly_average_temps.index, lower_bound, upper_bound, color='b', alpha=.1)
plt.title('US Average Temperatures Over Time (Yearly Average with Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Average Yearly Temperature')
plt.legend()
plt.show()

# TODO:
# - Plot all years on the same graph