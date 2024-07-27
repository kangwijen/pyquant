"""
This script performs rolling treynor on the log returns of a stock.
It downloads the stock data, calculates the log returns, and performs treynor ratio analysis.

Author: kangwijen

Parameters: None
Returns: None
Example: python treynor.py
"""

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define the stock symbol
STOCK = input("Enter the stock symbol: ")

# Define the benchmark symbol
BENCHMARK = input("Enter the benchmark symbol (default is ^JKSE): ") or '^JKSE'

# Define the period (in days)
PERIOD = input("Enter the period (in days) (default is 252): ") or 252

# Define the window size (in days)
WINDOW = input("Enter the window size (in days) (default is 21): ") or 21

# Define the risk-free rate
RISK_FREE_RATE = input("Enter the risk-free rate (default is 0.05): ") or 0.05

# Convert inputs to uppercase and integers
STOCK = str(STOCK).upper()
BENCHMARK = str(BENCHMARK).upper()
PERIOD = int(PERIOD)
WINDOW = int(WINDOW)
RISK_FREE_RATE = float(RISK_FREE_RATE)

# Download the datas
data = yf.download(STOCK, progress=False)
benchmark_data = yf.download(BENCHMARK, progress=False)

# Use the 'Close' price and fill missing values
data = data['Close'].ffill().bfill()
benchmark_data = benchmark_data['Close'].ffill().bfill()

# Calculate the log returns
log_returns = np.log(data / data.shift(1)).dropna()
benchmark_log_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()

# Calculate the risk-adjusted returns
log_returns = log_returns - RISK_FREE_RATE / PERIOD

# Calculate the rolling beta of the stock
rolling_covariance = log_returns.rolling(WINDOW).cov(benchmark_log_returns)
rolling_variance = benchmark_log_returns.rolling(WINDOW).var()
rolling_beta = rolling_covariance / rolling_variance

# Cut the beta to the same length as the log returns
rolling_beta = rolling_beta.loc[log_returns.index]

# Calculate the rolling Treynor ratio
rolling_treynor = log_returns.rolling(WINDOW).mean() / rolling_beta

# Fill missing values
rolling_treynor = rolling_treynor.ffill().bfill()

# Calculate the Q1 and Q3 quantiles
q1 = rolling_treynor.quantile(0.25)
q3 = rolling_treynor.quantile(0.75)

# Calculate the IQR
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Plot the rolling Treynor ratio
fig = go.Figure()

# Add the rolling Treynor ratio
fig.add_trace(
    go.Scatter(
        x=rolling_treynor.index,
        y=rolling_treynor,
        name='Rolling Treynor Ratio',
        mode='lines'
    )
)

# Add the lower bound
lower_bound = pd.Series(lower_bound, index=rolling_treynor.index)
upper_bound = pd.Series(upper_bound, index=rolling_treynor.index)

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
outliers = rolling_treynor[(rolling_treynor < lower_bound) | (rolling_treynor > upper_bound)]
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
mean = rolling_treynor.mean()
fig.add_trace(
    go.Scatter(
        x=rolling_treynor.index,
        y=pd.Series(mean, index=rolling_treynor.index),
        name='Mean',
        mode='lines'
    )
)

# Update the layout
fig.update_layout(
    title=f'Rolling Treynor Ratio for {STOCK}',
    xaxis_title='Date',
    yaxis_title='Treynor Ratio',
    showlegend=False
)

# Show the plot
fig.show()
