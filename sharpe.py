"""
This script performs rolling sharpe on the log returns of a stock.
It downloads the stock data, calculates the log returns, and performs sharpe ratio analysis.

Author: kangwijen

Parameters: None
Returns: None
Example: python normality.py
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

# Calculate the rolling Sharpe ratio
rolling_sharpe = log_returns.rolling(window=WINDOW).mean() / log_returns.rolling(window=WINDOW).std()

# Calculate the Q1 and Q3 quantiles
q1 = rolling_sharpe.quantile(0.25)
q3 = rolling_sharpe.quantile(0.75)

# Calculate the IQR
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Plot the rolling Sharpe ratio
fig = go.Figure()

# Add the rolling Sharpe ratio
fig.add_trace(
    go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe,
        name='Rolling Sharpe Ratio',
        mode='lines'
    )
)

# Add the lower bound
lower_bound = pd.Series(lower_bound, index=rolling_sharpe.index)
upper_bound = pd.Series(upper_bound, index=rolling_sharpe.index)

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
outliers = rolling_sharpe[(rolling_sharpe < lower_bound) | (rolling_sharpe > upper_bound)]
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
mean = rolling_sharpe.mean()
fig.add_trace(
    go.Scatter(
        x=rolling_sharpe.index,
        y=pd.Series(mean, index=rolling_sharpe.index),
        name='Mean',
        mode='lines'
    )
)

# Update the layout
fig.update_layout(
    title=f'Rolling Sharpe Ratio for {STOCK}',
    xaxis_title='Date',
    yaxis_title='Sharpe Ratio',
    showlegend=False
)

# Show the plot
fig.show()
