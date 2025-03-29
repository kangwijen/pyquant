"""
This script performs rolling Alpha on the log returns of a stock.
It downloads the stock data, calculates the log returns, and performs Alpha ratio analysis.

Author: kangwijen

Parameters: None
Returns: None
Example: python alpha.py
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

# Download the data
data = yf.download(STOCK, progress=False)
benchmark_data = yf.download(BENCHMARK, progress=False)

# Use the 'Close' price and fill missing values
data = data['Close'].ffill().bfill()
benchmark_data = benchmark_data['Close'].ffill().bfill()

# Calculate the log returns
log_returns = np.log(data / data.shift(1)).dropna()
benchmark_log_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()

# Align the data by index (date)
aligned_data = log_returns.align(benchmark_log_returns, join='inner')[0]
aligned_benchmark = log_returns.align(benchmark_log_returns, join='inner')[1]

# Calculate the risk-adjusted returns
risk_adjusted_log_returns = aligned_data - RISK_FREE_RATE / PERIOD

# Calculate the rolling beta of the stock
rolling_covariance = risk_adjusted_log_returns.rolling(WINDOW).cov(aligned_benchmark)
rolling_variance = aligned_benchmark.rolling(WINDOW).var()
rolling_beta = rolling_covariance / rolling_variance

# Calculate the rolling mean of the benchmark
rolling_mean = aligned_benchmark.rolling(WINDOW).mean()

# Calculate the expected return of the stock
expected_return = RISK_FREE_RATE / PERIOD + rolling_beta * (rolling_mean - RISK_FREE_RATE / PERIOD)

# Calculate the rolling Alpha of the stock
rolling_alpha = risk_adjusted_log_returns - expected_return

# Fill missing values
rolling_alpha = rolling_alpha.ffill().bfill()

# Add smoothing to the rolling Alpha
rolling_alpha = rolling_alpha.ewm(span=WINDOW).mean()

# Calculate the Q1 and Q3 quantiles
q1 = rolling_alpha.quantile(0.25)
q3 = rolling_alpha.quantile(0.75)

# Calculate the IQR
iqr = q3 - q1

# Calculate the lower and upper bounds
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Plot the rolling Alpha ratio
fig = go.Figure()

# Add the rolling Alpha ratio
fig.add_trace(
    go.Scatter(
        x=rolling_alpha.index,
        y=rolling_alpha,
        name='Rolling Alpha Ratio',
        mode='lines'
    )
)

# Add the lower bound
lower_bound = pd.Series(lower_bound, index=rolling_alpha.index)
upper_bound = pd.Series(upper_bound, index=rolling_alpha.index)

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
outliers = rolling_alpha[(rolling_alpha < lower_bound) | (rolling_alpha > upper_bound)]
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
mean = rolling_alpha.mean()
fig.add_trace(
    go.Scatter(
        x=rolling_alpha.index,
        y=pd.Series(mean, index=rolling_alpha.index),
        name='Mean',
        mode='lines'
    )
)

# Update the layout
fig.update_layout(
    title=f'Rolling Alpha Ratio for {STOCK}',
    xaxis_title='Date',
    yaxis_title='Alpha Ratio',
    showlegend=False
)

# Show the plot
fig.show()
