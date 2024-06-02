"""
This script performs seasonal decomposition of a stock's closing price.
It downloads the stock data, performs seasonal decomposition, and visualizes the components.

Author: kangwijen

Parameters: None
Returns: None
Example: python decomposition.py
"""

import yfinance as yf
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# Define the stock symbol
STOCK = input("Enter the stock symbol: ")

# Define the period for seasonal decomposition (in days)
PERIOD = input("Enter the period for seasonal decomposition (in days): ")

# Convert inputs to uppercase and integer
STOCK = str(STOCK).upper()
PERIOD = int(PERIOD)

# Download the stock data
data = yf.download(STOCK, progress=False)

# Use the 'Close' price and drop missing values
data = data['Close']

# Fill missing values using the previous value
data = data.ffill().bfill()

# Perform seasonal decomposition
result = seasonal_decompose(data, model='multiplicative', period=PERIOD)

# Calculate the Q1 and Q3 for the residual component
residual = result.resid.ffill().bfill()
q1 = np.percentile(residual, 25)
q3 = np.percentile(residual, 75)
iqr = q3 - q1

# Calculate bounds for significant residuals
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Create a plot
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Original Data', 'Trend', 'Seasonal', 'Residual')
)

# Add the original data
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data,
        name='Original Data'),
    row=1, col=1
)

# Add the trend component
fig.add_trace(
    go.Scatter(
        x=result.trend.index,
        y=result.trend,
        name='Trend'),
    row=2, col=1
)

# Add the seasonal component
fig.add_trace(
    go.Scatter(
        x=result.seasonal.index,
        y=result.seasonal,
        name='Seasonal'),
    row=3, col=1
)

# Add the residual component
fig.add_trace(
    go.Scatter(
        x=result.resid.index,
        y=result.resid,
        name='Residual'
    ),
    row=4, col=1
)

# Add lower and upper bounds to the residual plot
fig.add_trace(
    go.Scatter(
        x=result.resid.index,
        y=[lower_bound]*len(result.resid.index),
        name='Lower Bound',
        line={"color": 'green', "dash": 'dash'}
    ),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=result.resid.index,
        y=[upper_bound]*len(result.resid.index),
         name='Upper Bound',
         line={"color": 'red', "dash": 'dash'}
    ),
    row=4, col=1
)

# Update the layout
fig.update_layout(
    title=f'Seasonal Decomposition of {STOCK} with {PERIOD}-day Period',
    showlegend=False
)

# Show the plot
fig.show()
