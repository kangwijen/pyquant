"""
This script performs autocorrelation tests on the log returns of a stock using various statistical tests.
It downloads the stock data, calculates the log returns, and performs autocorrelation tests.

Author: kangwijen

Parameters: None
Returns: None
Example: python autocorrelation.py
"""

import yfinance as yf
import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from prettytable import PrettyTable
from colorama import Fore
import plotly.graph_objs as go

# Define the stock symbol
STOCK = input("Enter the stock symbol: ")

# Convert inputs to uppercase
STOCK = str(STOCK).upper()

# Download the stock data
data = yf.download(STOCK, progress=False)

# Use the 'Close' price and fill missing values
data = data['Close'].ffill().bfill()

# Calculate the log returns
log_returns = np.log(data / data.shift(1)).dropna()

# Perform the Ljung-Box test
ljung_box_result = acorr_ljungbox(log_returns, lags=10, return_df=True)
ljung_box_stat = ljung_box_result['lb_stat']
ljung_box_p_value = ljung_box_result['lb_pvalue']

# Perform the Durbin-Watson test
durbin_watson_result = durbin_watson(log_returns)

# Make a table to display the results of Ljung-Box test
table = PrettyTable()
table.field_names = ['Lag', 'Ljung-Box Statistic', 'P-Value', 'Significance']
for i in range(10):
    table.add_row([
        i + 1,
        f'{ljung_box_stat[i+1]:.4f}',
        f'{ljung_box_p_value[i+1]:.4f}',
        Fore.GREEN + 'Yes' + Fore.RESET if ljung_box_p_value[i+1] < 0.05 else Fore.RED + 'No' + Fore.RESET
    ])

# Print the results of the Ljung-Box test
print('Results of the Ljung-Box Test:')
print(table)

# Interpret the results of the Ljung-Box test
if any(ljung_box_p_value < 0.05):
    print(Fore.RED + 'The log returns are not independently distributed.' + Fore.RESET)
else:
    print(Fore.GREEN + 'The log returns are independently distributed.' + Fore.RESET)

# Print the results of the Durbin-Watson test
print(f'\nDurbin-Watson Statistic: {durbin_watson_result:.2f}')

# Interpret the results of the Durbin-Watson test
if durbin_watson_result < 1.5:
    print(Fore.GREEN + 'Positive autocorrelation.' + Fore.RESET)
elif durbin_watson_result > 2.5:
    print(Fore.RED + 'Negative autocorrelation.' + Fore.RESET)
else:
    print(Fore.YELLOW +'No autocorrelation.' + Fore.RESET)

# Create a QQ plot
res = stats.probplot(log_returns, dist="norm")

# Extract the quantiles and the least-squares fit line
osm = res[0][0]
osr = res[0][1]
slope = res[1][0]
intercept = res[1][1]
r = res[1][2]

# Create a trace for the sample data
sample_trace = go.Scatter(
    x=osm,
    y=osr,
    mode='markers',
    name='Sample Data'
)

# Create a trace for the theoretical quantile-quantile line
line_trace = go.Scatter(
    x=osm,
    y=slope * osm + intercept,
    mode='lines',
    name=f'Fit Line (r={r:.2f})'
)

# Combine the traces into a figure
fig = go.Figure(data=[sample_trace, line_trace])

# Add titles and labels
fig.update_layout(
    title=f'QQ Plot of Log Returns for {STOCK}',
    xaxis_title='Theoretical Quantiles',
    yaxis_title='Sample Quantiles',
    showlegend=False
)

# Show the plot
fig.show()
