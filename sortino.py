"""
This script performs rolling sortino on the log returns of a stock.
It downloads the stock data, calculates the log returns, and performs sortino ratio analysis.

Author: kangwijen

Parameters: None
Returns: None
Example: python sortino.py
"""


import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define the stock symbol
STOCK = input("Enter the stock symbol: ")

# Define the period (in days)
PERIOD = input("Enter the period (in days) (default is 252): ") or 252

# Define the window size (in days)
WINDOW = input("Enter the window size (in days) (default is 21): ") or 21

# Define the risk-free rate
RISK_FREE_RATE = input("Enter the risk-free rate (default is 0.05): ") or 0.05

# Convert inputs to uppercase and integers
STOCK = str(STOCK).upper()
PERIOD = int(PERIOD)
WINDOW = int(WINDOW)
RISK_FREE_RATE = float(RISK_FREE_RATE)

# Download the stock data
data = yf.download(STOCK, progress=False)

# Use the 'Close' price and fill missing values
data = data['Close'].ffill().bfill()

# Calculate the log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Calculate the risk-adjusted returns
log_returns = log_returns - RISK_FREE_RATE / PERIOD

# Calculate the rolling standard deviation of downside returns
downside_returns = log_returns[log_returns < 0]
rolling_downside_std = downside_returns.rolling(WINDOW).std()

# Calculate the rolling Sortino ratio
rolling_sortino = log_returns.rolling(WINDOW).mean() / rolling_downside_std

# Fill missing values
rolling_sortino = rolling_sortino.ffill().bfill()

# Calculate the Q1 and Q3 quantiles
q1 = rolling_sortino.quantile(0.25)
q3 = rolling_sortino.quantile(0.75)

# Calculate the IQR
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Plot the rolling Sortino ratio
fig = go.Figure()

# Add the rolling Sortino ratio
fig.add_trace(
    go.Scatter(
        x=rolling_sortino.index,
        y=rolling_sortino,
        name='Rolling Sortino Ratio',
        mode='lines'
    )
)

# Add the lower bound
lower_bound = pd.Series(lower_bound, index=rolling_sortino.index)
upper_bound = pd.Series(upper_bound, index=rolling_sortino.index)

fig.add_trace(
    go.Scatter(
        x=lower_bound.index,
        y=lower_bound,
        name='Lower Bound',
        mode='lines',
        line={"color": 'red', "dash": 'dash'}
    )
)

# Add the upper bound
fig.add_trace(
    go.Scatter(
        x=upper_bound.index,
        y=upper_bound,
        name='Upper Bound',
        mode='lines',
        line={"color": 'green', "dash": 'dash'}
    )
)

# Add the outliers
outliers = rolling_sortino[(rolling_sortino < lower_bound) | (rolling_sortino > upper_bound)]
fig.add_trace(
    go.Scatter(
        x=outliers.index,
        y=outliers,
        name='Outliers',
        mode='markers',
        marker={"color": 'red', "size": 8}
    )
)

# Add the mean
mean = rolling_sortino.mean()
fig.add_trace(
    go.Scatter(
        x=rolling_sortino.index,
        y=pd.Series(mean, index=rolling_sortino.index),
        name='Mean',
        mode='lines'
    )
)

# Update the layout
fig.update_layout(
    title=f'Rolling Sortino Ratio for {STOCK}',
    xaxis_title='Date',
    yaxis_title='Sortino Ratio',
    showlegend=False
)

# Show the plot
fig.show()
