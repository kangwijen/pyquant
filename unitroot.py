"""
This script performs unitroot tests on the log returns of a stock using various statistical tests.
It downloads the stock data, calculates the log returns, and performs unitroot tests.

Author: kangwijen

Parameters: None
Returns: None
Example: python unitroot.py
"""

import yfinance as yf
import numpy as np
from arch.unitroot import ADF, PhillipsPerron, KPSS
from prettytable import PrettyTable
from colorama import Fore

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

# Perform the augmented Dickey-Fuller test
adf_result = ADF(log_returns)
adf_stat = adf_result.stat
adf_p_value = adf_result.pvalue

# Perform the Phillips-Perron test
pp_result = PhillipsPerron(log_returns)
pp_stat = pp_result.stat
pp_p_value = pp_result.pvalue

# Perform the Kwiatkowski-Phillips-Schmidt-Shin test
kpss_result = KPSS(log_returns)
kpss_stat = kpss_result.stat
kpss_p_value = kpss_result.pvalue

# Function to generate conclusion string based on p-value
def get_adf_pp_conclusion(p_value):
    """
    Generate conclusion string based on the p-value of the ADF and PP tests.
    """
    return (
        Fore.RED + 'The returns are not stationary' + Fore.RESET
        if p_value > 0.05
        else Fore.GREEN + 'The returns are stationary' + Fore.RESET
    )

def get_kpss_conclusion(p_value):
    """
    Generate conclusion string based on the p-value of the KPSS test.
    """
    return (
        Fore.GREEN + 'The returns are stationary' + Fore.RESET
        if p_value > 0.05
        else Fore.RED + 'The returns are not stationary' + Fore.RESET
    )

# Make unit root test table
unit_root_table = PrettyTable()
unit_root_table.field_names = ['Test', 'Statistic', 'P-Value', 'Conclusion']
unit_root_table.add_row([
    'Augmented Dickey-Fuller',
    f'{adf_stat:.4f}', f'{adf_p_value:.4f}',
    get_adf_pp_conclusion(adf_p_value)
])
unit_root_table.add_row([
    'Phillips-Perron',
    f'{pp_stat:.4f}', f'{pp_p_value:.4f}',
    get_adf_pp_conclusion(pp_p_value)
])
unit_root_table.add_row([
    'Kwiatkowski-Phillips-Schmidt-Shin',
    f'{kpss_stat:.4f}', f'{kpss_p_value:.4f}',
    get_kpss_conclusion(kpss_p_value)
])

print(unit_root_table)
