# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import yfinance as yf

# %%
# Set the working directory
os.chdir(r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data")

# %%
# Load the CSV file
file_path = r"YMAX ETF Stock Price History.csv"
YMAX = pd.read_csv(file_path)

# Display the first few rows
YMAX.head()

# %%
# Load the CSV file
file_path = r"YMAG ETF Stock Price History.csv"
YMAG = pd.read_csv(file_path)

# Display the first few rows
YMAG.head()

# %%
#Convert dates to their right format in both YMAX and YMAG DataFrames
YMAG['Date'] = pd.to_datetime(YMAG['Date'])
YMAX['Date'] = pd.to_datetime(YMAX['Date'])
YMAX = YMAX.set_index('Date')
YMAG = YMAG.set_index('Date')

# %%
# Drop columns Vol and Change in both YMAX and YMAG DataFrames
YMAX = YMAX.drop(columns=['Vol.', 'Change %'])
YMAG = YMAG.drop(columns=['Vol.', 'Change %'])

# %%
#View the first few rows of the YMAX DataFrame
YMAX.head()

# %%
#View the first few rows of the YMAG DataFrame
YMAG.head()

# %%
# Convert Date index to datetime format
YMAX.index = pd.to_datetime(YMAX.index)

# Convert dates to numerical format for matplotlib
YMAX['Date'] = YMAX.index.map(mdates.date2num)

# Select required columns for candlestick_ohlc
ohlc = YMAX[['Date', 'Open', 'High', 'Low', 'Price']].copy()

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Format the x-axis dates to avoid clutter
ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto-adjust date intervals
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format as YYYY-MM-DD

# Plot the candlestick chart
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red')

# Labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('YMAX ETF Candlestick Chart')

# Rotate x-axis labels for better readability
fig.autofmt_xdate()

# Show the plot
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# Assume YMAG is your DataFrame with columns 'Open', 'High', 'Low', and 'Price'
# and its index contains date information.

# Convert index to datetime if it's not already
YMAG.index = pd.to_datetime(YMAG.index)

# Reset the index so that the dates become a column,
# and rename the index column to "Date" for clarity.
data = YMAG.reset_index().rename(columns={'index': 'Date'})

# Convert dates to Matplotlib’s numeric format
data['Date'] = data['Date'].map(mdates.date2num)

# Prepare the quotes array: each row is [date, open, high, low, close]
quotes = data[['Date', 'Open', 'High', 'Low', 'Price']].values

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the candlestick chart
candlestick_ohlc(ax, quotes, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Format the x-axis with date labels
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Set titles and labels
plt.title("YMAG ETF Candlestick Chart")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()

# Display the chart
plt.show()


# %%
# Drop Open, High, Low columns and rename Price to YMAX in YMAX DataFrame
YMAX = YMAX.drop(columns=['Open', 'High', 'Low']).rename(columns={'Price': 'YMAX'})

# Drop Open, High, Low columns and rename Price to YMAG in YMAG DataFrame
YMAG = YMAG.drop(columns=['Open', 'High', 'Low']).rename(columns={'Price': 'YMAG'})

# Ensure the Date index is set properly and drop any residual "Date" columns if present
if 'Date' in YMAX.columns:
    YMAX = YMAX.drop(columns=['Date'])

if 'Date' in YMAG.columns:
    YMAG = YMAG.drop(columns=['Date'])

# Merge both dataframes on the Date index
YMAX_YMAG_df = pd.merge(YMAX, YMAG, left_index=True, right_index=True, how='inner')
YMAX_YMAG_merged = YMAX_YMAG_df.copy()

# Display the corrected merged DataFrame
YMAX_YMAG_df.head()

# %%
#Compute the sum of Missing values in the YMAX_YMAG DataFrame, if zero then there are no missing values
YMAX_YMAG_df.isnull().sum()

# %%

# Optional: Set a dark theme (similar to Plotly's dark mode)
sns.set_theme(style="darkgrid")  # You can remove this if you prefer the default style

# Ensure the DataFrame index is in datetime format
YMAX_YMAG_df.index = pd.to_datetime(YMAX_YMAG_df.index)

# Create the figure and axis
plt.figure(figsize=(12, 6))

# Plot the YMAX line (blue)
plt.plot(YMAX_YMAG_df.index, YMAX_YMAG_df["YMAX"], color="blue", label="YMAX", linewidth=2)

# Plot the YMAG line (green)
plt.plot(YMAX_YMAG_df.index, YMAX_YMAG_df["YMAG"], color="green", label="YMAG", linewidth=2)

# Set title and axis labels
plt.title("YMAX vs YMAG Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")

# Rotate x-axis labels for better readability
plt.xticks(rotation=-45)

# Place the legend at the top-left
plt.legend(loc="upper left")

# Adjust layout for a neat appearance
plt.tight_layout()

# Show the plot
plt.show()


# %%
# Define rolling window size (e.g., 20 days)
window_size = 21

# Compute daily returns
returns = YMAX_YMAG_df.pct_change().dropna()

# Compute rolling volatilities (standard deviation of returns)
YMAX_YMAG_df["YMAX Volatility"] = returns["YMAX"].rolling(window=window_size).std()
YMAX_YMAG_df["YMAG Volatility"] = returns["YMAG"].rolling(window=window_size).std()

# Compute rolling correlation
YMAX_YMAG_df["Rolling Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["YMAG"])

# Drop NaN values resulting from rolling calculations
YMAX_YMAG_df = YMAX_YMAG_df.dropna()

YMAX_YMAG_df

# %%
# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")

# Ensure the DataFrame index is in datetime format
YMAX_YMAG_df.index = pd.to_datetime(YMAX_YMAG_df.index)

# Create the figure and axis
plt.figure(figsize=(12, 6))

# Plot YMAX Volatility (blue)
plt.plot(YMAX_YMAG_df.index, YMAX_YMAG_df["YMAX Volatility"], color="blue", label="YMAX Volatility", linewidth=2)

# Plot YMAG Volatility (green)
plt.plot(YMAX_YMAG_df.index, YMAX_YMAG_df["YMAG Volatility"], color="green", label="YMAG Volatility", linewidth=2)

# Set title and axis labels
plt.title("Rolling Volatilities of YMAX and YMAG")
plt.xlabel("Date")
plt.ylabel("Volatility")

# Format the x-axis dates
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=-45)

# Position the legend at the top-left
plt.legend(loc="upper left")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# %%
# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")  # You can remove or change this style if you prefer

# Ensure the DataFrame index is in datetime format
YMAX_YMAG_df.index = pd.to_datetime(YMAX_YMAG_df.index)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Rolling Correlation (red)
ax.plot(YMAX_YMAG_df.index, YMAX_YMAG_df["Rolling Correlation"], color="red", label="Rolling Correlation", linewidth=2)

# Set title and axis labels
ax.set_title("Rolling Correlation between YMAX and YMAG")
ax.set_xlabel("Date")
ax.set_ylabel("Correlation")

# Set the y-axis range to [0, 1.5]
ax.set_ylim([0, 1.5])

# Format the x-axis dates: auto locator and formatter
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=-45)

# Position the legend at the top-left
ax.legend(loc="upper left")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# %%
# Download the data of VIX, VVIX, and QQQ ETF
VIX = yf.download('^VIX', start='2024-01-01', end='2025-01-30')
VVIX = yf.download('^VVIX', start='2024-01-01', end='2025-01-30')
QQQ = yf.download('QQQ', start='2024-01-01', end='2025-01-30')

# Rename the columns
VIX.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
VVIX.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
QQQ.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# Function to create a candlestick chart using Matplotlib
def create_candlestick_chart(df, title):
    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index)
    
    # Create a new 'Date' column with numerical values for Matplotlib
    df['Date'] = df.index.map(mdates.date2num)
    
    # Prepare the OHLC data: Date, Open, High, Low, Close
    ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the candlestick chart
    candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red')
    
    # Format the x-axis to display dates properly
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Set the title and axis labels
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    
    plt.tight_layout()
    plt.show()

# Create charts for each asset (Assuming VIX, VVIX, and QQQ DataFrames contain the columns: Open, High, Low, Close)
create_candlestick_chart(VIX, "VIX Candlestick Chart")
create_candlestick_chart(VVIX, "VVIX Candlestick Chart")
create_candlestick_chart(QQQ, "QQQ Candlestick Chart")


# %%
# Rename "Close" column and drop other columns
VIX = VIX[['Close']].rename(columns={'Close': 'VIX'})
VVIX = VVIX[['Close']].rename(columns={'Close': 'VVIX'})
QQQ = QQQ[['Close']].rename(columns={'Close': 'QQQ'})

# Merge VIX, VVIX, and QQQ DataFrames on the Date index
merged_vix_vvix_qqq_df = VIX \
    .merge(VVIX, left_index=True, right_index=True, how='outer') \
    .merge(QQQ, left_index=True, right_index=True, how='outer')

# Merge the VIX, VVIX, and QQQ DataFrames with the YMAX_YMAG_merged DataFrame
merged_df = YMAX_YMAG_merged.merge(
    merged_vix_vvix_qqq_df, left_index=True, right_index=True, how='outer'
)

#Drop the NaN values
merged_df = merged_df.dropna()
merged_df

# %%
# Exporting the prices to a csv
merged_df.to_csv('All assets Prices.csv', index=False)

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Define the colors for each asset
colors = {
    "YMAX": "blue",
    "YMAG": "green",
    "VIX": "red",
    "VVIX": "purple",
    "QQQ": "orange"
}

# Create a figure using GridSpec with 3 rows and 2 columns.
# The third row (for QQQ) will span both columns and be slightly taller.
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# First row: YMAX and YMAG side by side
ax_ymax = fig.add_subplot(gs[0, 0])
ax_ymag = fig.add_subplot(gs[0, 1])

# Second row: VIX and VVIX side by side
ax_vix = fig.add_subplot(gs[1, 0])
ax_vvix = fig.add_subplot(gs[1, 1])

# Third row: QQQ spanning both columns
ax_qqq = fig.add_subplot(gs[2, :])

# Plot YMAX Price
ax_ymax.plot(merged_df.index, merged_df["YMAX"], color=colors["YMAX"], label="YMAX")
ax_ymax.set_title("YMAX Price")
ax_ymax.set_xlabel("Date")
ax_ymax.set_ylabel("Price")
ax_ymax.legend()
ax_ymax.tick_params(axis='x', labelrotation=-45)

# Plot YMAG Price
ax_ymag.plot(merged_df.index, merged_df["YMAG"], color=colors["YMAG"], label="YMAG")
ax_ymag.set_title("YMAG Price")
ax_ymag.set_xlabel("Date")
ax_ymag.set_ylabel("Price")
ax_ymag.legend()
ax_ymag.tick_params(axis='x', labelrotation=-45)

# Plot VIX Price
ax_vix.plot(merged_df.index, merged_df["VIX"], color=colors["VIX"], label="VIX")
ax_vix.set_title("VIX Price")
ax_vix.set_xlabel("Date")
ax_vix.set_ylabel("Price")
ax_vix.legend()
ax_vix.tick_params(axis='x', labelrotation=-45)

# Plot VVIX Price
ax_vvix.plot(merged_df.index, merged_df["VVIX"], color=colors["VVIX"], label="VVIX")
ax_vvix.set_title("VVIX Price")
ax_vvix.set_xlabel("Date")
ax_vvix.set_ylabel("Price")
ax_vvix.legend()
ax_vvix.tick_params(axis='x', labelrotation=-45)

# Plot QQQ Price on the larger, spanning subplot
ax_qqq.plot(merged_df.index, merged_df["QQQ"], color=colors["QQQ"], label="QQQ")
ax_qqq.set_title("QQQ Price")
ax_qqq.set_xlabel("Date")
ax_qqq.set_ylabel("Price")
ax_qqq.legend()
ax_qqq.tick_params(axis='x', labelrotation=-45)

# Set an overall figure title and adjust layout to prevent overlaps
fig.suptitle("YMAX, YMAG, VIX, VVIX, and QQQ Price Trends", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
# Define rolling window size (e.g., 21 days)
window_size = 21

# Compute daily returns for each asset in merged_df
returns = merged_df.pct_change().dropna()

# Initialize stats_df with rolling volatilities
stats_df = pd.DataFrame(index=returns.index)

# Compute rolling volatilities (standard deviation of returns) for each asset
for column in merged_df.columns:
    stats_df[f"{column} Volatility"] = returns[column].rolling(window=window_size).std()

# Compute rolling correlations
stats_df["YMAX-VIX Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["VIX"])
stats_df["YMAX-VVIX Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["VVIX"])
stats_df["YMAG-VIX Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["VIX"])
stats_df["YMAG-VVIX Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["VVIX"])
stats_df["YMAX-QQQ Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["QQQ"])
stats_df["YMAG-QQQ Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["QQQ"])


# Drop NaN values resulting from rolling calculations
stats_df = stats_df.dropna()
stats_df

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Define the colors for each asset
colors = {
    "YMAX": "blue",
    "YMAG": "green",
    "VIX": "red",
    "VVIX": "purple",
    "QQQ": "orange"
}

# Create a figure with GridSpec: 3 rows and 2 columns, where the third row will span both columns.
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

# First row: YMAX and YMAG
ax_ymax = fig.add_subplot(gs[0, 0])
ax_ymag = fig.add_subplot(gs[0, 1])

# Second row: VIX and VVIX
ax_vix = fig.add_subplot(gs[1, 0])
ax_vvix = fig.add_subplot(gs[1, 1])

# Third row: QQQ spanning both columns
ax_qqq = fig.add_subplot(gs[2, :])

# Plot YMAX Volatility
ax_ymax.plot(stats_df.index, stats_df["YMAX Volatility"], color=colors["YMAX"], label="YMAX Volatility")
ax_ymax.set_title("YMAX Volatility")
ax_ymax.set_xlabel("Date")
ax_ymax.set_ylabel("Volatility")
ax_ymax.legend()

# Plot YMAG Volatility
ax_ymag.plot(stats_df.index, stats_df["YMAG Volatility"], color=colors["YMAG"], label="YMAG Volatility")
ax_ymag.set_title("YMAG Volatility")
ax_ymag.set_xlabel("Date")
ax_ymag.set_ylabel("Volatility")
ax_ymag.legend()

# Plot VIX Volatility
ax_vix.plot(stats_df.index, stats_df["VIX Volatility"], color=colors["VIX"], label="VIX Volatility")
ax_vix.set_title("VIX Volatility")
ax_vix.set_xlabel("Date")
ax_vix.set_ylabel("Volatility")
ax_vix.legend()

# Plot VVIX Volatility
ax_vvix.plot(stats_df.index, stats_df["VVIX Volatility"], color=colors["VVIX"], label="VVIX Volatility")
ax_vvix.set_title("VVIX Volatility")
ax_vvix.set_xlabel("Date")
ax_vvix.set_ylabel("Volatility")
ax_vvix.legend()

# Plot QQQ Volatility on a larger, spanning subplot
ax_qqq.plot(stats_df.index, stats_df["QQQ Volatility"], color=colors["QQQ"], label="QQQ Volatility")
ax_qqq.set_title("QQQ Volatility")
ax_qqq.set_xlabel("Date")
ax_qqq.set_ylabel("Volatility")
ax_qqq.legend()

# Set an overall title and adjust layout so that titles and labels don't overlap
fig.suptitle("Rolling Volatilities of YMAX, YMAG, VIX, VVIX, and QQQ", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: for styling
import matplotlib.dates as mdates

# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")

# Ensure the DataFrame index is in datetime format
stats_df.index = pd.to_datetime(stats_df.index)

# -----------------------------
# Figure for YMAX correlations
# -----------------------------
fig_ymax, ax_ymax = plt.subplots(figsize=(12, 6))

# Plot YMAX-VIX Correlation (Blue)
ax_ymax.plot(stats_df.index, stats_df["YMAX-VIX Correlation"],
             color="blue", label="YMAX-VIX Correlation", linewidth=2)

# Plot YMAX-VVIX Correlation (Red)
ax_ymax.plot(stats_df.index, stats_df["YMAX-VVIX Correlation"],
             color="red", label="YMAX-VVIX Correlation", linewidth=2)

# Set title and axis labels
ax_ymax.set_title("Rolling Correlations of YMAX with VIX and VVIX")
ax_ymax.set_xlabel("Date")
ax_ymax.set_ylabel("Correlation")

# Format the x-axis with auto date locators and formatters
ax_ymax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_ymax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)

# Position the legend at the top-left
ax_ymax.legend(loc="upper left")

plt.tight_layout()
plt.show()

# -----------------------------
# Figure for YMAG correlations
# -----------------------------
fig_ymag, ax_ymag = plt.subplots(figsize=(12, 6))

# Plot YMAG-VIX Correlation (Green)
ax_ymag.plot(stats_df.index, stats_df["YMAG-VIX Correlation"],
             color="green", label="YMAG-VIX Correlation", linewidth=2)

# Plot YMAG-VVIX Correlation (Purple)
ax_ymag.plot(stats_df.index, stats_df["YMAG-VVIX Correlation"],
             color="purple", label="YMAG-VVIX Correlation", linewidth=2)

# Set title and axis labels
ax_ymag.set_title("Rolling Correlations of YMAG with VIX and VVIX")
ax_ymag.set_xlabel("Date")
ax_ymag.set_ylabel("Correlation")

# Format the x-axis with auto date locators and formatters
ax_ymag.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_ymag.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)

# Position the legend at the top-left
ax_ymag.legend(loc="upper left")

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")

# Ensure the DataFrame index is in datetime format
stats_df.index = pd.to_datetime(stats_df.index)

# -----------------------------
# Plot for YMAX-QQQ Correlation
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(stats_df.index, stats_df["YMAX-QQQ Correlation"],
         color="orange", linewidth=2, label="YMAX-QQQ Correlation")
plt.title("Rolling YMAX-QQQ Correlation")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot for YMAG-QQQ Correlation
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(stats_df.index, stats_df["YMAG-QQQ Correlation"],
         color="magenta", linewidth=2, label="YMAG-QQQ Correlation")
plt.title("Rolling YMAG-QQQ Correlation")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")

# Ensure the DataFrame index is in datetime format
stats_df.index = pd.to_datetime(stats_df.index)

# Create the base figure and first axis (left y-axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot YMAX-QQQ Correlation on the left y-axis (in orange)
ax1.plot(stats_df.index, stats_df["YMAX-QQQ Correlation"],
         color="orange", linewidth=2, label="YMAX-QQQ Correlation")
ax1.set_xlabel("Date")
ax1.set_ylabel("YMAX-QQQ Correlation", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

# Format the x-axis for dates
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)

# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot YMAX-VIX Correlation (in blue) and YMAX-VVIX Correlation (in red) on the right y-axis
ax2.plot(stats_df.index, stats_df["YMAX-VIX Correlation"],
         color="blue", linewidth=2, label="YMAX-VIX Correlation")
ax2.plot(stats_df.index, stats_df["YMAX-VVIX Correlation"],
         color="red", linewidth=2, label="YMAX-VVIX Correlation")
ax2.set_ylabel("YMAX-VIX/VVIX Correlation", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Set the plot title and adjust layout
plt.title("Rolling Correlations: YMAX-QQQ (Left) vs YMAX-VIX/VVIX (Right)")
plt.tight_layout()
plt.show()


# %%
# Optional: Set a dark-themed style similar to Plotly's dark mode
sns.set_theme(style="darkgrid")

# Ensure the DataFrame index is in datetime format
stats_df.index = pd.to_datetime(stats_df.index)

# Create the base figure and first axis (left y-axis)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot YMAG-QQQ Correlation on the left y-axis (magenta)
ax1.plot(stats_df.index, stats_df["YMAG-QQQ Correlation"],
         color="magenta", linewidth=2, label="YMAG-QQQ Correlation")
ax1.set_xlabel("Date")
ax1.set_ylabel("YMAG-QQQ Correlation", color="magenta")
ax1.tick_params(axis="y", labelcolor="magenta")

# Format the x-axis for dates
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=-45)

# Create the second y-axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot YMAG-VIX Correlation (green) and YMAG-VVIX Correlation (purple) on the right y-axis
ax2.plot(stats_df.index, stats_df["YMAG-VIX Correlation"],
         color="green", linewidth=2, label="YMAG-VIX Correlation")
ax2.plot(stats_df.index, stats_df["YMAG-VVIX Correlation"],
         color="purple", linewidth=2, label="YMAG-VVIX Correlation")
ax2.set_ylabel("YMAG-VIX-VVIX Correlation", color="black")
ax2.tick_params(axis="y", labelcolor="black")

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Set the plot title and adjust layout
plt.title("Rolling Correlations: YMAG-QQQ (Left) vs YMAG-VIX/VVIX (Right)")
plt.tight_layout()
plt.show()


# %%
# correlation of them all against each other
merged_df.corr()

# %% [markdown]
# The above analysis presents visualizations of the volatilities and correlations of the assets. We have both the overall correlations and the rolling correlations, the rolling correlations help track how asset relationships change over time, revealing shifts in market dynamics and risk exposure. Unlike static correlations, they adapt to different market conditions, aiding in diversification and hedging strategies. A sudden rise in correlation may signal market stress, while a drop can indicate idiosyncratic movements.

# %% [markdown]
# ## Monthly Data Analysis

# %%
# Import YMAX ETF Stock Price History (Monthly).csv
ymax_monthly_df = pd.read_csv("YMAX ETF Stock Price History (Monthly).csv", 
                      index_col="Date",    
                      parse_dates=True)     # Parse the index as datetime

# Import YMAG ETF Stock Price History (Monthly).csv
ymag_monthly_df = pd.read_csv("YMAG ETF Stock Price History (Monthly).csv", 
                      index_col="Date", 
                      parse_dates=True)

# Display information about the YMAX ETF Stock Price History DataFrame,
# including column names, data types, and non-null counts.
print("YMAX ETF Stock Price History Info:")
ymax_monthly_df.info()

# Display information about the YMAG ETF Stock Price History DataFrame,
# including column names, data types, and non-null counts.
print("\nYMAG ETF Stock Price History Info:")
ymag_monthly_df.info()


# %%
# Drop the 'Vol.' and 'Change %' columns from YMAX and YMAG DataFrames
ymax_monthly_df = ymax_monthly_df.drop(columns=['Vol.', 'Change %'])
ymag_monthly_df = ymag_monthly_df.drop(columns=['Vol.', 'Change %'])


# %%
# Drop Open, High, Low columns and rename Price to YMAX in the YMAX monthly DataFrame
ymax = ymax_monthly_df.drop(columns=['Open', 'High', 'Low']).rename(columns={'Price': 'YMAX'})

# Drop Open, High, Low columns and rename Price to YMAG in the YMAG monthly DataFrame
ymag = ymag_monthly_df.drop(columns=['Open', 'High', 'Low']).rename(columns={'Price': 'YMAG'})

# Ensure the Date index is set properly and remove any residual "Date" columns if present
if 'Date' in ymax.columns:
    ymax = ymax.drop(columns=['Date'])
if 'Date' in ymag.columns:
    ymag = ymag.drop(columns=['Date'])

# Merge both DataFrames on the Date index
ymax_ymag_monthly_merged = pd.merge(ymax, ymag, left_index=True, right_index=True, how='inner')
ymax_ymag_monthly_merged = ymax_ymag_monthly_merged.copy()

# Display the merged DataFrame
print(ymax_ymag_monthly_merged.head())


# %%
# Download monthly data of VIX, VVIX, and QQQ ETF
VVIX_monthly = yf.download('^VVIX', start='2024-01-01', end='2025-01-30', interval='1mo')
QQQ_monthly = yf.download('QQQ', start='2024-01-01', end='2025-01-30', interval='1mo')

# Rename the columns for consistency
VVIX_monthly.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
QQQ_monthly.columns = ['Close', 'High', 'Low', 'Open', 'Volume']


# %%
# Rename "Close" column for VVIX and QQQ, keeping only the "Close" column
VVIX_monthly = VVIX_monthly[['Close']].rename(columns={'Close': 'VVIX'})
QQQ_monthly = QQQ_monthly[['Close']].rename(columns={'Close': 'QQQ'})

# Merge the VVIX and QQQ DataFrames on the Date index (VIX is omitted)
merged_monthly_vvix_qqq_df = VVIX_monthly.merge(
    QQQ_monthly, left_index=True, right_index=True, how='outer'
)

# Merge the resulting VVIX and QQQ DataFrame with the ymax_ymag_monthly_merged DataFrame
merged_monthly_vvix_qqq_df = ymax_ymag_monthly_merged.merge(
    merged_monthly_vvix_qqq_df, left_index=True, right_index=True, how='outer'
)

# Drop any rows with missing values
merged_monthly_vvix_qqq_df = merged_monthly_vvix_qqq_df.dropna()

# Display the merged DataFrame
merged_monthly_vvix_qqq_df


# %%
# Define rolling window size (e.g., 2 periods for rolling volatility)
window_size = 5

# Compute daily returns for each asset in merged_monthly_vvix_qqq_df
returns = merged_monthly_vvix_qqq_df.pct_change().dropna()

# Initialize stats_df with rolling volatilities
stats_df = pd.DataFrame(index=returns.index)

# Compute rolling volatilities (standard deviation of returns) for each asset
for column in merged_monthly_vvix_qqq_df.columns:
    stats_df[f"{column} Volatility"] = returns[column].rolling(window=window_size).std()

# Compute rolling correlations
stats_df["YMAX-VVIX Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["VVIX"])
stats_df["YMAG-VVIX Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["VVIX"])
stats_df["YMAX-QQQ Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["QQQ"])
stats_df["YMAG-QQQ Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["QQQ"])

# Drop NaN values resulting from rolling calculations
stats_df = stats_df.dropna()
stats_df


# %%


# %%


# %%


# %%


# %%


# %%



