# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import yfinance as yf
from datetime import datetime

# %%
# Suppress all warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# %%
# Set the working directory
os.chdir(r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data")

# %%
# Function to view all rows and columns
def view_all():
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    print("Display set to show all rows and columns.")

# Function to reset display options to default
def reset_display():
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    print("Display options reset to default.")

# Example Usage
view_all()  # Set to view all rows and columns
reset_display()  # Reset to default display settings


# %%
# Load the excel data file
file_path = r"YMAG ETF Price & Dividends.xlsx"
YMAG = pd.read_excel(file_path)

# Convert 'Date' column to datetime
YMAG["Date"] = pd.to_datetime(YMAG["Date"])

# Drop unnecessary columns
YMAG = YMAG.drop(columns=['Vol.', 'Change %'])

# Display the first few rows
YMAG

# %%
# check for missing values in YMAX
YMAG.isnull().sum()

# %%
# Load the Excel data file
file_path = r"YMAX ETF Price & Dividends.xlsx"
YMAX = pd.read_excel(file_path)

# Convert 'Date' column to datetime
YMAX["Date"] = pd.to_datetime(YMAX["Date"])

# Drop unnecessary columns
YMAX = YMAX.drop(columns=['Vol.', 'Change %'])

# Display the first few rows
YMAX

# %%
# Check for missing values in YMAX
YMAX.isnull().sum()

# %%
#Extracting the Dividends data
# Create a new DataFrame with only rows where 'Open' is NaN
YMAX_Dividends = YMAX[YMAX['Open'].isna()].copy()

# Drop unnecessary columns
YMAX_Dividends = YMAX_Dividends.drop(columns=['Open', 'High', 'Low'])

# Rename 'Price' column to 'Dividends'
YMAX_Dividends = YMAX_Dividends.rename(columns={'Price': 'YMAX Dividends'})

# Display the new DataFrame
YMAX_Dividends

# %%
#Extracting the Dividends data
# Create a new DataFrame with only rows where 'Open' is NaN
YMAG_Dividends = YMAG[YMAG['Open'].isna()].copy()

# Drop unnecessary columns
YMAG_Dividends = YMAG_Dividends.drop(columns=['Open', 'High', 'Low'])

# Rename 'Price' column to 'Dividends'
YMAG_Dividends = YMAG_Dividends.rename(columns={'Price': 'YMAG Dividends'})

# Display the new DataFrame
YMAG_Dividends

# %%
#check for missing values in YMAG_Dividends
print(YMAG_Dividends.isnull().sum())
print(YMAX_Dividends.isnull().sum())

# %%
# Drop all rows with NaN values in YMAX and YMAG, to remove the duplicate rows where dividends were
YMAX = YMAX.dropna()
YMAG = YMAG.dropna()

# %%
# Merge dividends with YMAX price data
YMAX = YMAX.merge(YMAX_Dividends, on="Date", how="left")
YMAX["YMAX Dividends"].fillna(0, inplace=True)  # Fill missing dividends with 0

# Merge dividends with YMAG price data
YMAG = YMAG.merge(YMAG_Dividends, on="Date", how="left")
YMAG["YMAG Dividends"].fillna(0, inplace=True)  # Fill missing dividends with 0

# Rename price in YMAX to YMAX
YMAX = YMAX.rename(columns={'Price': 'YMAX'})

# Rename price in YMAG to YMAG
YMAG = YMAG.rename(columns={'Price': 'YMAG'})

# Display merged data
YMAX.head()


# %%
# Display merged data
YMAG.head()

# %%
# check and print sum of missing values in YMAX
print(YMAX.isnull().sum())

# check and print sum of missing values in YMAG
print(YMAG.isnull().sum())

# %%
# Merge dividends with YMAX price data
YMAG_YMAX_Divs_n_prices = YMAX[["Date", "YMAX", "YMAX Dividends"]].merge(YMAG[["Date", "YMAG", "YMAG Dividends"]],
                                                                         on="Date", how="left")

#check and print the sum of missing values
print(YMAG_YMAX_Divs_n_prices.isna().sum())

# Drop all rows with NaN values in YMAG_YMAX_Divs_n_prices
YMAG_YMAX_Divs_n_prices = YMAG_YMAX_Divs_n_prices.dropna()
print("sum of missing values after dropping them")
print(YMAG_YMAX_Divs_n_prices.isna().sum())


# %%
# Convert 'Date' column to datetime format
YMAG_YMAX_Divs_n_prices['Date'] = pd.to_datetime(YMAG_YMAX_Divs_n_prices['Date'])

# Set 'Date' as the index
YMAG_YMAX_Divs_n_prices.set_index('Date', inplace=True)

YMAG_YMAX_Divs_n_prices 


# %%
# Download the data of VIX, VVIX, and QQQ ETF
VIX = yf.download('^VIX', start='2024-01-01', end='2025-02-14')
VVIX = yf.download('^VVIX', start='2024-01-01', end='2025-02-14')
QQQ = yf.download('QQQ', start='2024-01-01', end='2025-02-14')

# Rename the columns
VIX.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
VVIX.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
QQQ.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Rename "Close" column and drop other columns
VIX = VIX[['Close']].rename(columns={'Close': 'VIX'})
VVIX = VVIX[['Close']].rename(columns={'Close': 'VVIX'})
QQQ = QQQ[['Close']].rename(columns={'Close': 'QQQ'})

# %%
VIX

# %%
# Merge VIX, VVIX, and QQQ DataFrames on the Date index
merged_vix_vvix_qqq_df = VIX \
    .merge(VVIX, left_index=True, right_index=True, how='outer') \
    .merge(QQQ, left_index=True, right_index=True, how='outer')

# Merge the VIX, VVIX, and QQQ DataFrames with the YMAG_YMAX_Divs_n_prices DataFrame
All_Assets = YMAG_YMAX_Divs_n_prices.merge(merged_vix_vvix_qqq_df, left_index=True, right_index=True, how='outer')

#Drop the NaN values
All_Assets = All_Assets.dropna()

# Sort the DataFrame by 'Date' in descending order
All_Assets = All_Assets.sort_values(by='Date', ascending=True)

# Print the sum of missing values in each column
print("The sum of missing values in each column:")
print(All_Assets.isnull().sum())

# %%
# print rhe info of the merged DataFrame
All_Assets.info()

# %%
# Display the first few rows of the merged DataFrame
All_Assets

# %%
# Define rolling window size (e.g., 21 days)
window_size = 21

# Compute daily returns for each asset in All_Assets, excluding dividends columns
returns = All_Assets.loc[:, ~All_Assets.columns.str.contains('Dividends')].pct_change().dropna()

# Initialize stats_df with rolling volatilities
Prices_and_stats_df = pd.DataFrame(index=returns.index)

# Compute rolling correlations
Prices_and_stats_df["YMAX-VIX Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["VIX"])
Prices_and_stats_df["YMAX-VVIX Correlation"] = returns["YMAX"].rolling(window=window_size).corr(returns["VVIX"])
Prices_and_stats_df["YMAG-VIX Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["VIX"])
Prices_and_stats_df["YMAG-VVIX Correlation"] = returns["YMAG"].rolling(window=window_size).corr(returns["VVIX"])

#Merge the prices and stats data
Prices_and_stats_df = All_Assets.merge(Prices_and_stats_df, left_index=True, right_index=True)

# Drop NaN values resulting from rolling calculations
Prices_and_stats_df = Prices_and_stats_df.dropna()

# Rest the prices and stats data index
Prices_and_stats_df.reset_index(inplace=True)

# Display the first and last few rows of Prices_and_stats_df
Prices_and_stats_df

# %%
# Export the data to excel
# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('Strategy 2 Prices and Statistics.xlsx', engine='xlsxwriter') as writer:
    # Write Prices_and_stats_df to the first sheet
    Prices_and_stats_df.to_excel(writer, sheet_name='Prices and Correlations', index=False)
    
print(f"✅ Performance DataFrames successfully saved to 'Prices_and_stats_df.xlsx'")

# %% [markdown]
# ## BACKTESTING THE INVESTMENT RULES
# 
# ### **📌 Investment Rules for Strategy 2**
# 
# 1. **Invest Only If:**
#    - **VIX is between 15 and 20** (inclusive), **AND**
#    - **VVIX is between 90 and 100** (i.e., VVIX is below 100 but not less than 90).
# 
#    > **Interpretation**: Go long on YMAX/YMAG (without a hedge) **only** when both conditions are met.
# 
# 2. **Exit the Market If:**
#    - **VIX drops below 15** or **rises above 20**, **OR**
#    - **VVIX goes above or equal to 100, or falls below 90**.
# 
#    > **Interpretation**: If either VIX or VVIX is outside the desired range, close all positions and remain in cash.
# 
# 3. **Re-Enter the Market When:**
#    - **VIX is again within 15–20**, **AND**
#    - **VVIX has stabilized in a tighter range – specifically, between 90 and 95 (inclusive).**
# 
#    > **Interpretation**: Re-enter only when both VIX is back in the 15–20 range and VVIX has come back to a “safe zone” (90–95).
# 
# ---
# 
# ### **🚀 Summary of Logic**
# 
# 1. **In-Market Condition**:  
#    $15 \leq \text{VIX} \leq 20$ **AND** $90 \leq \text{VVIX} < 100$.
# 
# 2. **Exit Condition**:  
#    $\text{VIX} < 15$ **OR** $\text{VIX} > 20$ **OR** $\text{VVIX} < 90$ **OR** $\text{VVIX} \geq 100$.
# 
# 3. **Re-Entry Condition**:  
#    $\text{VIX}$ returns to $[15,20]$ **AND** $\text{VVIX}$ is in the range $[90,95]$.
# 

# %% [markdown]
# #### YMAX BACKTEST:

# %%
# ------------------------------------------------------------
# 1) PREPARE THE DATA
# ------------------------------------------------------------
# DataFrame "Prices_and_stats_df" with Columns: ["Date", "YMAX", "YMAX Dividends", "VIX", "VVIX", ...]
# We copy it for this strategy's backtest on YMAX.
ymax_df = Prices_and_stats_df.copy()

# Define initial portfolio value
initial_investment = 10000.0  # $10,000 starting capital

# Sort by date to ensure chronological order
ymax_df.sort_values("Date", inplace=True)
ymax_df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------
# 2) DEFINE HELPER FUNCTIONS FOR CONDITIONS
# ------------------------------------------------------------
# function to check if it is time to enter the market or not
def in_market_condition(vix, vvix):
    """
    Condition for staying in the market once we're already invested:
    15 <= VIX <= 20, 90 <= VVIX < 100
    """
    return (15 <= vix <= 20) and (90 <= vvix < 100)

#function to check if it is time to re-enter the market or not
def reentry_condition(vix, vvix):
    """
    Stricter condition for re-entering the market after an exit:
    15 <= VIX <= 20, 90 <= VVIX <= 95
    """
    return (15 <= vix <= 20) and (90 <= vvix <= 95)

# function to check the exit condition if it is time to exit the market or not to exit
def exit_condition(vix, vvix):
    """
    If VIX < 15 or VIX > 20, or VVIX < 90 or VVIX >= 100 -> EXIT
    """
    return (vix < 15) or (vix > 20) or (vvix < 90) or (vvix >= 100)

# ------------------------------------------------------------
# 3) SET UP BACKTEST COLUMNS AND INITIAL VALUES
# ------------------------------------------------------------
ymax_df["Portfolio_Value"] = np.nan
ymax_df.loc[0, "Portfolio_Value"] = initial_investment

# Boolean: are we currently invested?
ymax_df["In_Market"] = False
ymax_df.loc[0, "In_Market"] = False

# Track how many shares of YMAX we hold
ymax_df["Shares_Held"] = 0.0
ymax_df.loc[0, "Shares_Held"] = 0.0

# Strategy label for each day
ymax_df["Strategy"] = "No Investment"

# ------------------------------------------------------------
# 4) BACKTEST LOOP
# ------------------------------------------------------------
for i in range(1, len(ymax_df)):
    # Copy forward previous day's portfolio value, in_market status, and shares
    ymax_df.loc[i, "Portfolio_Value"] = ymax_df.loc[i-1, "Portfolio_Value"]
    ymax_df.loc[i, "In_Market"] = ymax_df.loc[i-1, "In_Market"]
    ymax_df.loc[i, "Shares_Held"] = ymax_df.loc[i-1, "Shares_Held"]

    # Current day data
    vix_today = ymax_df.loc[i, "VIX"]
    vvix_today = ymax_df.loc[i, "VVIX"]
    ymax_price_today = ymax_df.loc[i, "YMAX"]
    ymax_div_today = ymax_df.loc[i, "YMAX Dividends"]

    # Are we in the market at the start of today?
    currently_in_market = ymax_df.loc[i-1, "In_Market"]

    if currently_in_market:
        # Check if we remain in the market or exit
        if exit_condition(vix_today, vvix_today):
            # EXIT: close position -> shares = 0, remain in cash
            ymax_df.loc[i, "In_Market"] = False
            ymax_df.loc[i, "Shares_Held"] = 0.0
            ymax_df.loc[i, "Strategy"] = "No Investment"
            # Portfolio_Value stays the same as yesterday's (carried forward)
        else:
            # REMAIN IN MARKET: update portfolio value based on today's price + dividend
            shares_held = ymax_df.loc[i, "Shares_Held"]
            new_portfolio_value = shares_held * (ymax_price_today + ymax_div_today)
            ymax_df.loc[i, "Portfolio_Value"] = new_portfolio_value
            ymax_df.loc[i, "Strategy"] = "Long YMAX"
    else:
        # currently out of market, check re-entry condition
        if reentry_condition(vix_today, vvix_today):
            # ENTER: buy shares with all capital
            cash_available = ymax_df.loc[i, "Portfolio_Value"]
            if ymax_price_today > 0:
                shares_bought = cash_available / ymax_price_today
                ymax_df.loc[i, "Shares_Held"] = shares_bought
                ymax_df.loc[i, "In_Market"] = True
                ymax_df.loc[i, "Strategy"] = "Long YMAX"
                # Immediately compute today's new portfolio value
                new_portfolio_value = shares_bought * (ymax_price_today + ymax_div_today)
                ymax_df.loc[i, "Portfolio_Value"] = new_portfolio_value
            else:
                # If price is 0 or invalid, skip
                ymax_df.loc[i, "Strategy"] = "No Investment"
        else:
            # Stay in cash
            ymax_df.loc[i, "Strategy"] = "No Investment"

# ------------------------------------------------------------
# 5) COMPUTE DAILY RETURNS
# ------------------------------------------------------------
ymax_df["Portfolio_Return"] = ymax_df["Portfolio_Value"].pct_change()

# ------------------------------------------------------------
# 6) VISUALIZE RESULTS
# ------------------------------------------------------------
sns.set_style("whitegrid")

# (A) Portfolio Performance Plot
plt.figure(figsize=(12, 6))
ax = sns.lineplot(
    x=ymax_df["Date"],
    y=ymax_df["Portfolio_Value"],
    color="blue",
    linewidth=2.5,
    label="Portfolio Value (YMAX)"
)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.xlabel("Date", fontsize=14, fontweight="bold")
plt.ylabel("Portfolio Value ($)", fontsize=14, fontweight="bold")
plt.title("📈 Portfolio Performance for YMAX (Strategy 2)", fontsize=16, fontweight="bold", color="darkblue")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")
plt.legend(fontsize=12, loc="upper left", frameon=True, shadow=True, edgecolor="black")
plt.show()

# (B) Strategy Usage Over Time
plt.figure(figsize=(12, 4))
ax = sns.barplot(
    x=ymax_df["Strategy"].value_counts().index,
    y=ymax_df["Strategy"].value_counts().values,
    palette=["royalblue", "red", "green"]
)
plt.xlabel("Strategy", fontsize=14, fontweight="bold")
plt.ylabel("Number of Days", fontsize=14, fontweight="bold")
plt.title("📊 Strategy 2 Distribution Over Time (YMAX)", fontsize=16, fontweight="bold", color="navy")
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis="y", linestyle="--", linewidth=1.2, color="gray")
plt.show()


# %%
# ---------------------------------------------
# STEP A: Identify Entry and Exit Days
# ---------------------------------------------
# Entry Day = Yesterday was out of market, today is in market
ymax_df["Entry"] = (ymax_df["In_Market"].shift(1) == False) & (ymax_df["In_Market"] == True)

# Exit Day = Yesterday was in market, today is out of market
ymax_df["Exit"] = (ymax_df["In_Market"].shift(1) == True) & (ymax_df["In_Market"] == False)

# ---------------------------------------------
# STEP B: Prepare Data for Plotting
# ---------------------------------------------
entry_days = ymax_df[ymax_df["Entry"] == True]
exit_days  = ymax_df[ymax_df["Exit"] == True]

# Sort data by date (just to be safe)
ymax_df.sort_values("Date", inplace=True)
ymax_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# STEP C: Create the Dual-Axis Plot
# ---------------------------------------------
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(12, 6))

# LEFT Y-AXIS: Plot YMAX Price
ax1.set_xlabel("Date", fontsize=14, fontweight="bold")
ax1.set_ylabel("YMAX Price ($)", fontsize=14, fontweight="bold", color="blue")

# Plot YMAX Price line
line1, = ax1.plot(ymax_df["Date"], ymax_df["YMAX"], color="blue", linewidth=2, label="YMAX Price")

# Add entry (^) and exit (v) markers on YMAX
ax1.scatter(entry_days["Date"], entry_days["YMAX"], marker="^", color="blue", s=100, label="Entry")
ax1.scatter(exit_days["Date"], exit_days["YMAX"], marker="v", color="red", s=100, label="Exit")

# Make the left y-axis text blue for clarity
ax1.tick_params(axis='y', labelcolor="blue")

# Format the date axis
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)

# ---------------------------------------------
# RIGHT Y-AXIS: Plot Portfolio Value
# ---------------------------------------------
ax2 = ax1.twinx()  # share the same x-axis
ax2.set_ylabel("Portfolio Value ($)", fontsize=14, fontweight="bold", color="red")

line2, = ax2.plot(
    ymax_df["Date"],
    ymax_df["Portfolio_Value"],
    color="red",
    linewidth=2,
    label="Portfolio Value"
)
ax2.tick_params(axis='y', labelcolor="red")
plt.yticks(fontsize=11)

# ---------------------------------------------
# STEP D: Enhance the Plot
# ---------------------------------------------
plt.title("YMAX Price (Left Axis) vs. Portfolio Value (Right Axis)", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", frameon=True, shadow=True)

plt.tight_layout()
plt.show()


# %%
import plotly.graph_objects as go

# ---------------------------------------------
# STEP A: Identify Entry and Exit Days
# ---------------------------------------------
ymax_df["Entry"] = (ymax_df["In_Market"].shift(1) == False) & (ymax_df["In_Market"] == True)
ymax_df["Exit"]  = (ymax_df["In_Market"].shift(1) == True) & (ymax_df["In_Market"] == False)

# ---------------------------------------------
# STEP B: Prepare Data for Plotting
# ---------------------------------------------
entry_days = ymax_df[ymax_df["Entry"] == True]
exit_days  = ymax_df[ymax_df["Exit"] == True]

# Sort data by date
ymax_df.sort_values("Date", inplace=True)
ymax_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# STEP C: Create Plotly Figure
# ---------------------------------------------
fig = go.Figure()

# 1) YMAX Price (Left Axis)
fig.add_trace(
    go.Scatter(
        x=ymax_df["Date"],
        y=ymax_df["YMAX"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="YMAX Price",
        yaxis="y1"
    )
)

# 2) Entry Markers (triangle-up)
fig.add_trace(
    go.Scatter(
        x=entry_days["Date"],
        y=entry_days["YMAX"],
        mode="markers",
        marker=dict(symbol="triangle-up", color="blue", size=12),
        name="Entry",
        yaxis="y1"
    )
)

# 3) Exit Markers (triangle-down)
fig.add_trace(
    go.Scatter(
        x=exit_days["Date"],
        y=exit_days["YMAX"],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=12),
        name="Exit",
        yaxis="y1"
    )
)

# 4) Portfolio Value (Right Axis)
fig.add_trace(
    go.Scatter(
        x=ymax_df["Date"],
        y=ymax_df["Portfolio_Value"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Portfolio Value",
        yaxis="y2"
    )
)

# ---------------------------------------------
# STEP D: Configure Layout for Dual Axis
# ---------------------------------------------
fig.update_layout(
    title="YMAX Price (Left Axis) vs. Portfolio Value (Right Axis) - Interactive Plotly",
    xaxis=dict(
        title="Date",
        type="date",
        tickformat="%Y-%m",
        tickangle=45
    ),
    yaxis=dict(
        title="YMAX Price ($)",
        side="left",
        showgrid=False,
        color="blue"
    ),
    yaxis2=dict(
        title="Portfolio Value ($)",
        side="right",
        overlaying="y",
        position=1.0,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        color="red"
    ),
    legend=dict(
        x=1.0,
        y=1.0,
        xanchor='center',
        yanchor='top',
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1
    ),
    hovermode="x unified"
)

# ---------------------------------------------
# STEP E: Show Interactive Figure
# ---------------------------------------------
fig.show()


# %%


# %% [markdown]
# #### Performance Metrics for Strategy 2 on YMAX

# %%
# ---------------------------------------------
# STEP 1: Compute Performance Metrics for Strategy 2 on YMAX
# ---------------------------------------------
# Compute daily returns from portfolio value (if not already computed)
ymax_df["Portfolio_Return"] = ymax_df["Portfolio_Value"].pct_change()

performance_metrics = {}

# Total Return (%)
performance_metrics["Total Return (%)"] = (ymax_df["Portfolio_Value"].iloc[-1] / ymax_df["Portfolio_Value"].iloc[0] - 1) * 100

# CAGR (Compounded Annual Growth Rate) (%)
num_days = (ymax_df["Date"].iloc[-1] - ymax_df["Date"].iloc[0]).days
years = num_days / 365
performance_metrics["CAGR (%)"] = ((ymax_df["Portfolio_Value"].iloc[-1] / ymax_df["Portfolio_Value"].iloc[0]) ** (1 / years) - 1) * 100

# Annualized Volatility (%)
performance_metrics["Annualized Volatility (%)"] = ymax_df["Portfolio_Return"].std() * np.sqrt(252) * 100

# Sharpe Ratio (assuming risk-free rate = 2%)
risk_free_rate = 0.02
sharpe_ratio = (performance_metrics["CAGR (%)"] / 100 - risk_free_rate) / (performance_metrics["Annualized Volatility (%)"] / 100)
performance_metrics["Sharpe Ratio"] = sharpe_ratio

# Max Drawdown (%) Calculation
rolling_max = ymax_df["Portfolio_Value"].cummax()
drawdown = (ymax_df["Portfolio_Value"] / rolling_max) - 1  # Decimal values (e.g., -0.10 for -10%)
drawdown_percentage = drawdown * 100  # Convert to percentage
performance_metrics["Max Drawdown (%)"] = drawdown_percentage.min()  # Most negative value

# Calmar Ratio (CAGR / |Max Drawdown|)
max_drawdown_abs = abs(performance_metrics["Max Drawdown (%)"])
performance_metrics["Calmar Ratio"] = performance_metrics["CAGR (%)"] / max_drawdown_abs if max_drawdown_abs != 0 else np.nan

# Convert metrics to a DataFrame for display
performance_df = pd.DataFrame(performance_metrics, index=["YMAX Strategy 2"])

# %%


# ---------------------------------------------
# STEP 2: Plot Drawdown Over Time
# ---------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

ax = sns.lineplot(
    x=ymax_df["Date"],
    y=drawdown_percentage,
    color="red",
    linewidth=2,
    label="Drawdown (%)"
)

# Format X-axis to display months
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Format Y-axis labels as percentages (e.g., -1.00%, -2.00%)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:.2f}%"))

plt.xlabel("Date", fontsize=14, fontweight="bold")
plt.ylabel("Drawdown (%)", fontsize=14, fontweight="bold")
plt.title("📉 Drawdowns of YMAX Strategy 2 Over Time", fontsize=16, fontweight="bold", color="darkred")

# Add thick, colored grid lines
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")

plt.legend(fontsize=12, loc="upper left", frameon=True, shadow=True)
plt.tight_layout()
plt.show()


# %%
# Display the performance metrics for YMAX
performance_df_ymax = performance_df
performance_df_ymax = performance_df_ymax.round(2)
performance_df_ymax

# %% [markdown]
# ## YMAG BACKTEST

# %%
# ------------------------------------------------------------
# 1) PREPARE THE DATA
# ------------------------------------------------------------
# Copy the cleaned DataFrame for YMAG backtesting.
ymag_df = Prices_and_stats_df.copy()

# Define initial portfolio value
initial_investment = 10000.0  # $10,000 starting capital

# Sort by date to ensure chronological order
ymag_df.sort_values("Date", inplace=True)
ymag_df.reset_index(drop=True, inplace=True)

# ------------------------------------------------------------
# 2) DEFINE HELPER FUNCTIONS FOR CONDITIONS
# ------------------------------------------------------------
def in_market_condition(vix, vvix):
    """
    Condition for staying in the market once we're already invested:
    15 <= VIX <= 20, and 90 <= VVIX < 100.
    """
    return (15 <= vix <= 20) and (90 <= vvix < 100)

def reentry_condition(vix, vvix):
    """
    Stricter condition for re-entering the market after an exit:
    15 <= VIX <= 20, and 90 <= VVIX <= 95.
    """
    return (15 <= vix <= 20) and (90 <= vvix <= 95)

def exit_condition(vix, vvix):
    """
    Exit the market if VIX < 15 or VIX > 20, or if VVIX < 90 or VVIX >= 100.
    """
    return (vix < 15) or (vix > 20) or (vvix < 90) or (vvix >= 100)

# ------------------------------------------------------------
# 3) SET UP BACKTEST COLUMNS AND INITIAL VALUES
# ------------------------------------------------------------
ymag_df["Portfolio_Value"] = np.nan
ymag_df.loc[0, "Portfolio_Value"] = initial_investment

# Boolean: are we currently invested?
ymag_df["In_Market"] = False
ymag_df.loc[0, "In_Market"] = False

# Track how many shares of YMAG we hold
ymag_df["Shares_Held"] = 0.0
ymag_df.loc[0, "Shares_Held"] = 0.0

# Strategy label for each day
ymag_df["Strategy"] = "No Investment"

# ------------------------------------------------------------
# 4) BACKTEST LOOP
# ------------------------------------------------------------
for i in range(1, len(ymag_df)):
    # Carry forward previous day's portfolio value, in_market status, and shares
    ymag_df.loc[i, "Portfolio_Value"] = ymag_df.loc[i-1, "Portfolio_Value"]
    ymag_df.loc[i, "In_Market"] = ymag_df.loc[i-1, "In_Market"]
    ymag_df.loc[i, "Shares_Held"] = ymag_df.loc[i-1, "Shares_Held"]
    
    # Current day data
    vix_today = ymag_df.loc[i, "VIX"]
    vvix_today = ymag_df.loc[i, "VVIX"]
    ymag_price_today = ymag_df.loc[i, "YMAG"]
    ymag_div_today = ymag_df.loc[i, "YMAG Dividends"]
    
    # Check if we are currently in the market
    currently_in_market = ymag_df.loc[i-1, "In_Market"]
    
    if currently_in_market:
        # If in the market, check if exit condition is met
        if exit_condition(vix_today, vvix_today):
            # EXIT: close position -> set shares to 0, mark as not in market
            ymag_df.loc[i, "In_Market"] = False
            ymag_df.loc[i, "Shares_Held"] = 0.0
            ymag_df.loc[i, "Strategy"] = "No Investment"
            # Portfolio value is carried forward (no update)
        else:
            # REMAIN IN MARKET: update portfolio value based on today's YMAG price and dividend
            shares_held = ymag_df.loc[i, "Shares_Held"]
            new_portfolio_value = shares_held * (ymag_price_today + ymag_div_today)
            ymag_df.loc[i, "Portfolio_Value"] = new_portfolio_value
            ymag_df.loc[i, "Strategy"] = "Long YMAG"
    else:
        # If not in the market, check re-entry condition
        if reentry_condition(vix_today, vvix_today):
            # ENTER: invest all available cash
            cash_available = ymag_df.loc[i, "Portfolio_Value"]
            if ymag_price_today > 0:
                shares_bought = cash_available / ymag_price_today
                ymag_df.loc[i, "Shares_Held"] = shares_bought
                ymag_df.loc[i, "In_Market"] = True
                ymag_df.loc[i, "Strategy"] = "Long YMAG"
                # Update portfolio value immediately with today's price and dividend
                new_portfolio_value = shares_bought * (ymag_price_today + ymag_div_today)
                ymag_df.loc[i, "Portfolio_Value"] = new_portfolio_value
            else:
                ymag_df.loc[i, "Strategy"] = "No Investment"
        else:
            # Otherwise, remain out of the market
            ymag_df.loc[i, "Strategy"] = "No Investment"

# ------------------------------------------------------------
# 5) COMPUTE DAILY RETURNS
# ------------------------------------------------------------
ymag_df["Portfolio_Return"] = ymag_df["Portfolio_Value"].pct_change()

# ------------------------------------------------------------
# 6) VISUALIZE RESULTS
# ------------------------------------------------------------
sns.set_style("whitegrid")

# (A) Portfolio Performance Plot for YMAG
plt.figure(figsize=(12, 6))
ax = sns.lineplot(
    x=ymag_df["Date"],
    y=ymag_df["Portfolio_Value"],
    color="green",
    linewidth=2.5,
    label="Portfolio Value (YMAG)"
)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xlabel("Date", fontsize=14, fontweight="bold")
plt.ylabel("Portfolio Value ($)", fontsize=14, fontweight="bold")
plt.title("📈 Portfolio Performance for YMAG (Strategy 2)", fontsize=16, fontweight="bold", color="darkgreen")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")
plt.legend(fontsize=12, loc="upper left", frameon=True, shadow=True, edgecolor="black")
plt.show()

# (B) Strategy Usage Over Time for YMAG
plt.figure(figsize=(12, 4))
ax = sns.barplot(
    x=ymag_df["Strategy"].value_counts().index,
    y=ymag_df["Strategy"].value_counts().values,
    palette=["green", "red", "blue"]
)
plt.xlabel("Strategy", fontsize=14, fontweight="bold")
plt.ylabel("Number of Days", fontsize=14, fontweight="bold")
plt.title("📊 Strategy 2 Distribution Over Time (YMAG)", fontsize=16, fontweight="bold", color="navy")
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis="y", linestyle="--", linewidth=1.2, color="gray")
plt.show()


# %%
# ---------------------------------------------
# STEP A: Identify Entry and Exit Days for YMAG
# ---------------------------------------------
# Entry Day = Yesterday was out of market, today is in market
ymag_df["Entry"] = (ymag_df["In_Market"].shift(1) == False) & (ymag_df["In_Market"] == True)

# Exit Day = Yesterday was in market, today is out of market
ymag_df["Exit"] = (ymag_df["In_Market"].shift(1) == True) & (ymag_df["In_Market"] == False)

# ---------------------------------------------
# STEP B: Prepare Data for Plotting
# ---------------------------------------------
entry_days = ymag_df[ymag_df["Entry"] == True]
exit_days  = ymag_df[ymag_df["Exit"] == True]

# Sort data by date (just to be safe)
ymag_df.sort_values("Date", inplace=True)
ymag_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# STEP C: Create the Dual-Axis Plot for YMAG
# ---------------------------------------------
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(12, 6))

# LEFT Y-AXIS: Plot YMAG Price
ax1.set_xlabel("Date", fontsize=14, fontweight="bold")
ax1.set_ylabel("YMAG Price ($)", fontsize=14, fontweight="bold", color="blue")

# Plot YMAG Price line
line1, = ax1.plot(
    ymag_df["Date"],
    ymag_df["YMAG"],
    color="blue",
    linewidth=2,
    label="YMAG Price"
)

# Add entry (^) and exit (v) markers on YMAG Price
ax1.scatter(
    entry_days["Date"], 
    entry_days["YMAG"], 
    marker="^", 
    color="blue", 
    s=100, 
    label="Entry"
)
ax1.scatter(
    exit_days["Date"], 
    exit_days["YMAG"], 
    marker="v", 
    color="red",
    s=100, 
    label="Exit"
)

# Make the left y-axis text blue for clarity
ax1.tick_params(axis='y', labelcolor="blue")

# Format the date axis on x-axis
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)

# ---------------------------------------------
# RIGHT Y-AXIS: Plot Portfolio Value
# ---------------------------------------------
ax2 = ax1.twinx()  # share the same x-axis
ax2.set_ylabel("Portfolio Value ($)", fontsize=14, fontweight="bold", color="red")

line2, = ax2.plot(
    ymag_df["Date"],
    ymag_df["Portfolio_Value"],
    color="red",
    linewidth=2,
    label="Portfolio Value"
)
ax2.tick_params(axis='y', labelcolor="red")
plt.yticks(fontsize=11)

# ---------------------------------------------
# STEP D: Enhance the Plot
# ---------------------------------------------
plt.title("YMAG Price (Left Axis) vs. Portfolio Value (Right Axis)", fontsize=16, fontweight="bold")
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", frameon=True, shadow=True)

plt.tight_layout()
plt.show()


# %%
# ---------------------------------------------
# STEP A: Identify Entry and Exit Days for YMAG
# ---------------------------------------------
ymag_df["Entry"] = (ymag_df["In_Market"].shift(1) == False) & (ymag_df["In_Market"] == True)
ymag_df["Exit"]  = (ymag_df["In_Market"].shift(1) == True) & (ymag_df["In_Market"] == False)

# ---------------------------------------------
# STEP B: Prepare Data for Plotting
# ---------------------------------------------
entry_days = ymag_df[ymag_df["Entry"] == True]
exit_days  = ymag_df[ymag_df["Exit"] == True]

# Sort data by date (just to be safe)
ymag_df.sort_values("Date", inplace=True)
ymag_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# STEP C: Create the Plotly Figure for YMAG
# ---------------------------------------------
fig = go.Figure()

# 1) YMAG Price (Left Axis)
fig.add_trace(
    go.Scatter(
        x=ymag_df["Date"],
        y=ymag_df["YMAG"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="YMAG Price",
        yaxis="y1"
    )
)

# 2) Entry Markers (triangle-up)
fig.add_trace(
    go.Scatter(
        x=entry_days["Date"],
        y=entry_days["YMAG"],
        mode="markers",
        marker=dict(symbol="triangle-up", color="blue", size=12),
        name="Entry",
        yaxis="y1"
    )
)

# 3) Exit Markers (triangle-down)
fig.add_trace(
    go.Scatter(
        x=exit_days["Date"],
        y=exit_days["YMAG"],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=12),
        name="Exit",
        yaxis="y1"
    )
)

# 4) Portfolio Value (Right Axis)
fig.add_trace(
    go.Scatter(
        x=ymag_df["Date"],
        y=ymag_df["Portfolio_Value"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Portfolio Value",
        yaxis="y2"
    )
)

# ---------------------------------------------
# STEP D: Configure Layout for Dual Axis and Legend Position
# ---------------------------------------------
fig.update_layout(
    title="YMAG Price (Left Axis) vs. Portfolio Value (Right Axis) - Interactive Plotly",
    xaxis=dict(
        title="Date",
        type="date",
        tickformat="%Y-%m",
        tickangle=45
    ),
    yaxis=dict(
        title="YMAG Price ($)",
        side="left",
        showgrid=False,
        color="blue"
    ),
    yaxis2=dict(
        title="Portfolio Value ($)",
        side="right",
        overlaying="y",
        position=1.0,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        color="red"
    ),
    legend=dict(
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="black",
        borderwidth=1
    ),
    hovermode="x unified"
)

# ---------------------------------------------
# STEP E: Show Interactive Figure
# ---------------------------------------------
fig.show()


# %% [markdown]
# #####  PERFORMANCE METRICS FOR YMAG

# %%
# ------------------------------------------------------------
# 1) COMPUTE PERFORMANCE METRICS FOR YMAG
# ------------------------------------------------------------

# Compute daily returns from portfolio value for YMAG
ymag_df["Portfolio_Return"] = ymag_df["Portfolio_Value"].pct_change()

performance_metrics_ymag = {}

# Total Return (%) for YMAG
performance_metrics_ymag["Total Return (%)"] = (ymag_df["Portfolio_Value"].iloc[-1] / ymag_df["Portfolio_Value"].iloc[0] - 1) * 100

# CAGR (Compounded Annual Growth Rate) for YMAG
num_days = (ymag_df["Date"].iloc[-1] - ymag_df["Date"].iloc[0]).days
years = num_days / 365
performance_metrics_ymag["CAGR (%)"] = ((ymag_df["Portfolio_Value"].iloc[-1] / ymag_df["Portfolio_Value"].iloc[0]) ** (1 / years) - 1) * 100

# Annualized Volatility (%) for YMAG
performance_metrics_ymag["Annualized Volatility (%)"] = ymag_df["Portfolio_Return"].std() * np.sqrt(252) * 100

# Sharpe Ratio for YMAG (assuming risk-free rate = 2%)
risk_free_rate = 0.02
sharpe_ratio_ymag = (performance_metrics_ymag["CAGR (%)"] / 100 - risk_free_rate) / (performance_metrics_ymag["Annualized Volatility (%)"] / 100)
performance_metrics_ymag["Sharpe Ratio"] = sharpe_ratio_ymag

# Max Drawdown (%) Calculation for YMAG
rolling_max_ymag = ymag_df["Portfolio_Value"].cummax()
drawdown_ymag = (ymag_df["Portfolio_Value"] / rolling_max_ymag) - 1  # Decimal values
drawdown_percentage_ymag = drawdown_ymag * 100  # Convert to percentage
performance_metrics_ymag["Max Drawdown (%)"] = drawdown_percentage_ymag.min()  # Most negative value

# Calmar Ratio for YMAG (CAGR divided by absolute Max Drawdown)
max_drawdown_abs_ymag = abs(performance_metrics_ymag["Max Drawdown (%)"])
performance_metrics_ymag["Calmar Ratio"] = performance_metrics_ymag["CAGR (%)"] / max_drawdown_abs_ymag if max_drawdown_abs_ymag != 0 else np.nan

# Convert performance metrics to a DataFrame for display
performance_df_ymag = pd.DataFrame(performance_metrics_ymag, index=["YMAG Strategy 2"])


# %%
# ------------------------------------------------------------
# 2) PLOT DRAWDOWNS OVER TIME FOR YMAG
# ------------------------------------------------------------
sns.set_style("darkgrid")
plt.figure(figsize=(12, 6))

ax = sns.lineplot(
    x=ymag_df["Date"],
    y=drawdown_percentage_ymag,
    color="red",
    linewidth=2,
    label="Drawdown (%)"
)

# Format X-axis to display one tick per month
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Format Y-axis labels to show percentages with two decimal places
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y:.2f}%"))

plt.xlabel("Date", fontsize=14, fontweight="bold")
plt.ylabel("Drawdown (%)", fontsize=14, fontweight="bold")
plt.title("📉 Drawdowns of YMAG Strategy 2 Over Time", fontsize=16, fontweight="bold", color="darkred")

# Add thicker, colored grid lines for better visibility
plt.grid(True, linestyle="--", linewidth=1.2, color="gray")
plt.legend(fontsize=12, loc="upper left", frameon=True, shadow=True)
plt.tight_layout()
plt.show()


# %%
# Display the performance metrics for YMAX
performance_df_ymag = performance_df_ymag.round(2)
performance_df_ymag

# %%
# 1️⃣ Convert portfolio returns to 2-decimal percentage
ymax_df["Portfolio_Return"] = (ymax_df["Portfolio_Return"] * 100).round(2)
ymag_df["Portfolio_Return"] = (ymag_df["Portfolio_Return"] * 100).round(2)

# 2️⃣ Rename the 'Portfolio_Return' column to 'Portfolio_Return (%)'
ymax_df.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
ymag_df.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)

# 3️⃣ Export the data to Excel
output_filename = "Strategy 2 Performance.xlsx"

with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
    # Sheet 1: Raw data (Prices_and_stats_df)
    Prices_and_stats_df.to_excel(writer, sheet_name="Sheet1", index=False)
    
    # Sheet 2: YMAX trading results
    ymax_df.to_excel(writer, sheet_name="Ymax Trading Results", index=False)
    
    # Sheet 3: YMAG trading results
    ymag_df.to_excel(writer, sheet_name="YMAG Trading Results", index=False)
    
    # Sheet 4: YMAX Performance Metrics
    performance_df_ymax.to_excel(writer, sheet_name="YMAX Performance", index=True)

    # Sheet 5: YMAG Performance Metrics
    performance_df_ymag.to_excel(writer, sheet_name="YMAG Performance", index=True)

print(f"✅ All DataFrames successfully saved to '{output_filename}'")


# %%


# %%


# %% [markdown]
# ### **📌 Investment Rules for Strategy 3:**
# 1. **Rule 2**: If **VIX is above 20**, **Long YMAX/YMAG (no hedge)** and remain fully invested.
# 2. **Rule 4**: If **VVIX goes above 100**, **exit the market** (close all positions).
# 3. **Rule 5**: Re-enter the market when **VVIX drops back below 95 or 90** (to avoid losses and whipsaws).
# 
# ##### **📌 Summary**
# - **Exit conditions** are based on **VVIX spikes above 100**.
# - **Re-entry conditions** trigger only when **VVIX stabilizes under 95-90**.
# 

# %% [markdown]
# ### **📌 Investment Rules for Strategy 4:**
# 1. **Rule 1**: If **VIX is under 15**, **Long YMAX/YMAG (no hedge)** and backtest separately to analyze performance.
# 2. **Rule 2**: If **VVIX goes above 100**, **exit the market** (close all positions).
# 3. **Rule 3**: Re-enter the market when **VVIX drops back below 95 or 90** (to avoid losses and whipsaws).
# 
# ##### **📌 Summary**
# - **Exit conditions** are based on **VVIX spikes above 100**.
# - **Re-entry conditions** trigger only when **VVIX stabilizes under 95-90**.
# 

# %% [markdown]
# 

# %% [markdown]
# enter when vix is less than 20 (range is 0 to 20) and VVIX is less than 95 (range is 95 to 0)
# 
# 
# exit if VVIX goes above 100, or VIX goes above 20
# 
# 

# %% [markdown]
# 
# # Design of the web app. 
# Have a header, header 1.
# Have a list of strategies, strategy 1 is the correlation strategy, 2 is the indicator ranges strategy and 3 is the option to combine strategies
# In the option to combine strategies, we need options of a user to decide how to combine them, eg begin with strategy 2, and if there is no investment in strategy 2, and there is investment in strategy 1, go to strategy 1, trade until: 
# 1. it closes by itself, that is it reaches a day/period of no investmnent, then checks if strategy 1 allows to investment under that condition and switches to it.
# 2. it continues running, and as it runs conditions of strategy 1 reappear, invest in it too, not sure how funds allocation will come into play here though, but give suggestions.
# - Under header 1, what we have is just description of the strategies, not the option to select the strategies yet.
# 
# Have a Header 2 called Backtester
# - First option have the option to specify which asset/assets to backtest. User will select YMAX or YMAG, or Both. If user selects both, there will be side by side tabs to view their plots, and performace measures will be displayed together, eg YMAX in row 1 YMAG in row 2, etc
# - Next the user will specify which strategy to backtest, either strategy 1, 2 or both. If both, have been specified, have the option of which strategy will be prioritized as number 1, and which as 2, and secondly the combination strategies mentioned above, of closing by itself or continues running.
# 
# 
# after selecting the Assets, 
# In strategy 1 of correlation, I need the following:
# - A slider to set the correlation window from 1 to 30, increments of 1,2,3, and so on.
# - plotly plot of Portfolio performance over time
# - bar chart of strategy distribution over time
# - Plotly plot of Drawdwon over time
# - diplay performance summary results: Total Return (%), CAGR (%), Annualized Volatility (%), Sharpe Ratio, Max Drawdown (%), Calmar Ratio
# 
# In strategy 2 and all the others, the visuals will be as above.

# %% [markdown]
# ---
# the data being supplied is the prices of YMAX and YMAG and their dividends, then VIX, VVIX and QQQ Prices.
# 
# Another feature of the web app is to have a different markdown page explaining updating of dividends and portfolio values, detailed description of the strategies like strategy 2, and much more other descriptions.
# 
# Or maybe it can be a page with table of contents and headings.


