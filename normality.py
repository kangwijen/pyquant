"""
This script performs normality tests on the log returns of a stock using various statistical tests.
It downloads the stock data, calculates the log returns, and performs normality tests.

Author: kangwijen

Parameters: None
Returns: None
Example: python normality.py
"""

import yfinance as yf
import numpy as np
from scipy.stats import shapiro, anderson, kstest, jarque_bera
from prettytable import PrettyTable
from colorama import Fore

# Define the stock symbol
STOCK = input("Enter the stock symbol: ")

# Define the period for seasonal decomposition
PERIOD = input("Enter the period for seasonal decomposition: ")

# Convert inputs to uppercase and integer
STOCK = str(STOCK).upper()
PERIOD = int(PERIOD)

# Download the stock data
data = yf.download(STOCK, progress=False)

# Use the 'Close' price and drop missing values
data = data['Close']

# Fill missing values using the previous value
data = data.ffill().bfill()

# Calculate the log returns
log_returns = np.log(data / data.shift(1))

# Calculate the mean and standard deviation of the log returns
mean = log_returns.mean()
std_dev = log_returns.std()

# Calculate the skewness and kurtosis of the log returns
skewness = log_returns.skew()
kurtosis = log_returns.kurtosis()

# Perform the Jarque-Bera test
jarque_bera_stat, jarque_p_value = jarque_bera(log_returns.dropna())

# Perform the Shapiro-Wilk test
shapiro_wilk_stat, shapiro_p_value = shapiro(log_returns.dropna())

# Perform the Anderson-Darling test
anderson_result = anderson(log_returns.dropna())

# Perform the Kolmogorov-Smirnov test
kolmogorov_smirnov_stat, kolmogorov_p_value = kstest(log_returns.dropna(), 'norm')

# Function to generate conclusion string based on p-value
def get_normality_conclusion(p_value):
    """
    Generate conclusion string based on the p-value of the normality test.
    """
    return (
        Fore.RED + 'The returns are not normally distributed' + Fore.RESET
        if p_value < 0.05
        else Fore.GREEN + 'The returns are normally distributed' + Fore.RESET
    )

# Make summary statistics table
summary_table = PrettyTable()
summary_table.field_names = ["Statistic", "Value"]
summary_table.add_row(["Mean", f"{mean:.4f}"])
summary_table.add_row(["Standard Deviation", f"{std_dev:.4f}"])
summary_table.add_row(["Skewness", f"{skewness:.4f}"])
summary_table.add_row(["Kurtosis", f"{kurtosis:.4f}"])

# Make normality tests table
normality_table = PrettyTable()
normality_table.field_names = ["Test", "Statistic", "P-Value", "Conclusion"]
normality_table.add_row([
    "Jarque-Bera",
    f"{jarque_bera_stat:.4f}", f"{jarque_p_value:.4f}",
    get_normality_conclusion(jarque_p_value)
])
normality_table.add_row([
    "Shapiro-Wilk",
    f"{shapiro_wilk_stat:.4f}", f"{shapiro_p_value:.4f}",
    get_normality_conclusion(shapiro_p_value)
])
normality_table.add_row([
    "Kolmogorov-Smirnov", 
    f"{kolmogorov_smirnov_stat:.4f}", f"{kolmogorov_p_value:.4f}",
    get_normality_conclusion(kolmogorov_p_value)
])

# Make Anderson-Darling test table
anderson_table = PrettyTable()
anderson_table.field_names = ["Significance Level", "Critical Value"]
for i, level in enumerate(anderson_result.significance_level):
    anderson_table.add_row([f"{level:.0f}%", f"{anderson_result.critical_values[i]:.4f}"])
anderson_conclusion = (
    Fore.RED + 'The returns are not normally distributed' + Fore.RESET
    if anderson_result.statistic > anderson_result.critical_values[2]
    else Fore.GREEN + 'The returns are normally distributed' + Fore.RESET
)

# Print results
print(f'Summary Statistics for {STOCK}')
print(summary_table)
print("\nNormality Tests")
print(normality_table)
print("\nAnderson-Darling Test")
print(f"Statistic: {anderson_result.statistic:.4f}")
print(anderson_table)
print(f"Conclusion: {anderson_conclusion}")
