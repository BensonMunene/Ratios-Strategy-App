import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os

# -----------------------------------------------------------------------------
# Optionally, set the current working directory to where this script is stored.
# os.chdir(r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Codes")
# -----------------------------------------------------------------------------

# Define the base directory for the data files
data_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data\TradingView Data"

# -----------------------------------------------------------------------------
# Main Title and Description
# -----------------------------------------------------------------------------
st.title("Multi-Asset Price Analysis by VVIX/VIX Ratio")
st.markdown("""
This app analyzes price data for multiple assets by computing the **VVIX/VIX** ratio.  
The ratio is floored to create integer segments, and the plot below shows the selected asset's price with color-coded segments.  
The bottom subplot displays a color-coded bar indicating the corresponding floored ratio over time.
""")

# -----------------------------------------------------------------------------
# Sidebar Selections
# -----------------------------------------------------------------------------
st.sidebar.header("Data Options")

# Add QQQ to the dataset options
dataset_option = st.sidebar.selectbox("Select Dataset", ["YMAX", "YMAG", "QQQ"])
timeframe_option = st.sidebar.selectbox("Select Timeframe", ["Daily", "4H", "1H", "30M"])

# -----------------------------------------------------------------------------
# Build the Filename Based on User Selections
# (Adjust these if your actual filenames differ.)
# -----------------------------------------------------------------------------
if timeframe_option == "Daily":
    if dataset_option == "YMAX":
        filename = "YMAX_VIX_VVIX_QQQ_Daily.csv"
    elif dataset_option == "YMAG":
        filename = "YMAG_VIX_VVIX_QQQ_Daily.csv"
    else:  # dataset_option == "QQQ"
        filename = "QQQ_VIX_VVIX_QQQ_Daily.csv"

elif timeframe_option == "4H":
    if dataset_option == "YMAX":
        filename = "YMAX_VIX_VVIX_QQQ_4H.csv"
    elif dataset_option == "YMAG":
        filename = "YMAG_VIX_VVIX_QQQ_4H.csv"
    else:  # QQQ
        filename = "QQQ_VIX_VVIX_QQQ_4H.csv"

elif timeframe_option == "1H":
    if dataset_option == "YMAX":
        filename = "YMAX_VIX_VVIX_QQQ_1H.csv"
    elif dataset_option == "YMAG":
        filename = "YMAG_VIX_VVIX_QQQ_1H.csv"
    else:  # QQQ
        filename = "QQQ_VIX_VVIX_QQQ_1H.csv"

elif timeframe_option == "30M":
    if dataset_option == "YMAX":
        filename = "YMAX_VIX_VVIX_QQQ_30M.csv"
    elif dataset_option == "YMAG":
        filename = "YMAG_VIX_VVIX_QQQ_30M.csv"
    else:  # QQQ
        filename = "QQQ_VIX_VVIX_QQQ_30M.csv"

file_path = os.path.join(data_dir, filename)

# -----------------------------------------------------------------------------
# Load Data Function with Streamlit Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df_loaded = pd.read_csv(path)
    return df_loaded

# -----------------------------------------------------------------------------
# Try to Load the Data
# -----------------------------------------------------------------------------
try:
    df = load_data(file_path)
except Exception as e:
    st.error(f"Error loading data from {file_path}: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values('Date', inplace=True)

# Compute the VVIX/VIX ratio
df['ratio'] = df['VVIX'] / df['VIX']

# Replace inf/-inf with NaN, then fill with 0 to avoid IntCastingNaNError
df['ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['ratio'].fillna(0, inplace=True)

# Floor the ratio to int
df['ratio_int'] = np.floor(df['ratio']).astype(int)

# Identify segments (change_id) where the floored ratio changes
df['change_id'] = (df['ratio_int'] != df['ratio_int'].shift(1)).cumsum()

# -----------------------------------------------------------------------------
# Define a Color Map (Updated ratio=7 from gray to pink, ratio=8 from pink to hotpink)
# Also include 0 in case any ratio floors to 0
# -----------------------------------------------------------------------------
color_map = {
    0: 'black',
    1: 'green',
    2: 'red',
    3: 'blue',
    4: 'orange',
    5: 'purple',
    6: 'brown',
    7: 'pink',     # Changed from gray to pink
    8: 'hotpink',  # Changed from pink to hotpink to avoid duplication
    9: 'olive',
    10: 'cyan'
}
unique_ratios = sorted(df['ratio_int'].unique())

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
fig, (ax_line, ax_bar) = plt.subplots(
    2, 1, sharex=True, figsize=(12, 8),
    gridspec_kw={'height_ratios': [2, 0.4]}
)

# Set white backgrounds
fig.patch.set_facecolor('white')
ax_line.set_facecolor('white')
ax_bar.set_facecolor('white')

# Use Seaborn whitegrid style for a clean look
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Top Subplot: Price Line with Colored Segments
# -----------------------------------------------------------------------------
# Add dummy lines for the legend
for ratio_val in unique_ratios:
    ax_line.plot([], [], color=color_map.get(ratio_val, 'black'),
                 label=f'Ratio = {ratio_val}')

# Plot the chosen price column ("QQQ" in your CSV) with segment-wise coloring
for i in range(len(df) - 1):
    x1 = df.iloc[i]['Date']
    y1 = df.iloc[i]['QQQ']   # Assumes each CSV has a column named "QQQ"
    ratio1 = df.iloc[i]['ratio_int']
    x2 = df.iloc[i+1]['Date']
    y2 = df.iloc[i+1]['QQQ']
    color = color_map.get(ratio1, 'black')
    ax_line.plot([x1, x2], [y1, y2], color=color, linewidth=2)

# Add semi-transparent rectangles for each ratio segment
for _, grp in df.groupby('change_id'):
    ratio_val = grp['ratio_int'].iloc[0]
    color = color_map.get(ratio_val, 'black')
    start_date = grp['Date'].iloc[0]
    end_date = grp['Date'].iloc[-1]
    ax_line.axvspan(start_date, end_date, color=color, alpha=0.2)

# Dynamically name the plot title based on user selections
plot_title = f"{dataset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio"
ax_line.set_title(plot_title, fontsize=14)
ax_line.set_ylabel('Price', fontsize=12)
ax_line.legend(title='Floored Ratio')

# -----------------------------------------------------------------------------
# Bottom Subplot: Color-Coded Bar Indicator
# -----------------------------------------------------------------------------
for ratio_val in unique_ratios:
    ax_bar.plot([], [], color=color_map.get(ratio_val, 'black'),
                label=f'Ratio = {ratio_val}')

for _, grp in df.groupby('change_id'):
    ratio_val = grp['ratio_int'].iloc[0]
    color = color_map.get(ratio_val, 'black')
    start_date = grp['Date'].iloc[0]
    end_date = grp['Date'].iloc[-1]
    ax_bar.axvspan(start_date, end_date, facecolor=color, alpha=1.0)

ax_bar.set_ylim(0, 1)
ax_bar.set_yticks([])
ax_bar.set_ylabel('')
ax_bar.set_title('Floored Ratio Over Time (Color-Coded Bar)')
ax_bar.legend(title='Floored Ratio', loc='upper left')

# -----------------------------------------------------------------------------
# X-Axis Formatting
# -----------------------------------------------------------------------------
month_locator = mdates.MonthLocator()  # Tick every month
month_formatter = mdates.DateFormatter('%b %Y')  # Format ticks as "Jan 2020"
ax_line.xaxis.set_major_locator(month_locator)
ax_line.xaxis.set_major_formatter(month_formatter)
ax_bar.xaxis.set_major_locator(month_locator)
ax_bar.xaxis.set_major_formatter(month_formatter)

plt.setp(ax_line.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax_line.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
ax_bar.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')

plt.tight_layout()

# -----------------------------------------------------------------------------
# Display the Plot
# -----------------------------------------------------------------------------
st.pyplot(fig)

# -----------------------------------------------------------------------------
# Display Processed Data
# -----------------------------------------------------------------------------
st.markdown("## Processed Data")
st.markdown("""
Below is the processed DataFrame used to generate the above plots.  
It includes the original columns (Date, VIX, VVIX, QQQ) along with the computed **ratio**,  
the floored ratio (**ratio_int**), and the **change_id** (segment index).
""")
# Display about 10 rows at a time (approx. height=300)
st.dataframe(df, height=300)
