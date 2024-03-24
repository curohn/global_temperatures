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
us_temps['AverageTemperature'].fillna(method='ffill', inplace=True) # Fill missing values using forward fill
us_temps = us_temps[us_temps.index.year > 1820] # Data is inaccurate before 1820, so we will filter it out
yearly_average_temps = us_temps['AverageTemperature'].resample('A').mean()
# Calculate the lower and upper bounds for the uncertainty
lower_bound = yearly_average_temps - us_temps['AverageTemperatureUncertainty'].resample('A').mean()
upper_bound = yearly_average_temps + us_temps['AverageTemperatureUncertainty'].resample('A').mean()

# Fit a linear regression model to the data
future_periods = 50
X = np.arange(len(yearly_average_temps)).reshape(-1, 1)
y = yearly_average_temps.values
model = LinearRegression()
model.fit(X, y) # Train the model
y_pred = model.predict(X) # Fit line to existing data
X_future = np.arange(len(yearly_average_temps), len(yearly_average_temps) + future_periods).reshape(-1, 1)
y_future_pred = model.predict(X_future) # Predict for future periods

# Plot the data
plt.figure(figsize=(10,5))
plt.grid(True)
plt.xlim([yearly_average_temps.index[0], pd.date_range(yearly_average_temps.index[-1], periods=future_periods+10, freq='A')[-1]])
plt.plot(yearly_average_temps, label='Yearly Average Temperature') # Plot the data
plt.plot(yearly_average_temps.index, y_pred, label='Linear Regression', linestyle='--', color='r') # Plot the linear regression line
plt.plot(pd.date_range(yearly_average_temps.index[-1], periods=future_periods, freq='A'), y_future_pred, label='Future Predictions', linestyle='--', color='g') # Plot the future predictions
plt.fill_between(yearly_average_temps.index, lower_bound, upper_bound, color='b', alpha=.1, label='Uncertainty') # Plot the uncertainty
plt.title('US Average Temperatures Over Time (Yearly Average with Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Average Yearly Temperature')
plt.legend()
plt.show()

# TODO:
# 1. Check for Linearity
# 2. Check for Independence
# 3. Check for Homoscedasticity
# 4. Check for Normality