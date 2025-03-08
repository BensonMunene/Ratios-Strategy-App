import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import os

# -----------------------------------------------------------------------------
# 1. Set Streamlit page configuration to wide layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Asset Price Analysis by VVIX/VIX Ratio",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Main Title & Brief Description
# -----------------------------------------------------------------------------
st.title("Multi-Asset Price Analysis by VVIX/VIX Ratio")
st.markdown("""
This app analyzes price data for multiple assets by computing the **VVIX/VIX** ratio.  
The ratio is **floored** to create integer segments, and the plot below shows the selected asset's price over time,  
color‐coded by those segments. The bottom subplot displays a color-coded bar indicating when each floored ratio changes.
""")

# -----------------------------------------------------------------------------
# 3. Dataset and Timeframe Selection (in main body)
# -----------------------------------------------------------------------------
st.markdown("#### Select Dataset")
st.markdown("Choose from **YMAX**, **YMAG**, or **QQQ**.")
dataset_option = st.selectbox("Dataset", ["YMAX", "YMAG", "QQQ"])

st.markdown("#### Select Timeframe")
st.markdown("""Choose the desired time interval: 
- **Daily** Frequency Data
- **4H** (4 hours)
- **1H** (1 hour)
- **30M** (30 minutes)
""")
timeframe_option = st.selectbox("Timeframe", ["Daily", "4H", "1H", "30M"])

# -----------------------------------------------------------------------------
# 4. Define your data directory and CSV-filename logic
# -----------------------------------------------------------------------------
data_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data\TradingView Data"

def get_filename(dataset, timeframe):
    """
    Return the CSV filename for the chosen dataset & timeframe.
    For QQQ, reuse the YMAX CSV by default.
    """
    if timeframe == "Daily":
        if dataset == "YMAX":
            return "YMAX_VIX_VVIX_QQQ_Daily.csv"
        elif dataset == "YMAG":
            return "YMAG_VIX_VVIX_QQQ_Daily.csv"
        else:  # "QQQ" -> reuse YMAX daily
            return "YMAX_VIX_VVIX_QQQ_Daily.csv"

    elif timeframe == "4H":
        if dataset == "YMAX":
            return "YMAX_VIX_VVIX_QQQ_4H.csv"
        elif dataset == "YMAG":
            return "YMAG_VIX_VVIX_QQQ_4H.csv"
        else:  # "QQQ" -> reuse YMAX 4H
            return "YMAX_VIX_VVIX_QQQ_4H.csv"

    elif timeframe == "1H":
        if dataset == "YMAX":
            return "YMAX_VIX_VVIX_QQQ_1H.csv"
        elif dataset == "YMAG":
            return "YMAG_VIX_VVIX_QQQ_1H.csv"
        else:  # "QQQ" -> reuse YMAX 1H
            return "YMAX_VIX_VVIX_QQQ_1H.csv"

    elif timeframe == "30M":
        if dataset == "YMAX":
            return "YMAX_VIX_VVIX_QQQ_30M.csv"
        elif dataset == "YMAG":
            return "YMAG_VIX_VVIX_QQQ_30M.csv"
        else:  # "QQQ" -> reuse YMAX 30M
            return "YMAX_VIX_VVIX_QQQ_30M.csv"

filename = get_filename(dataset_option, timeframe_option)
file_path = os.path.join(data_dir, filename)

# -----------------------------------------------------------------------------
# 5. Load Data (with caching)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(file_path)
except Exception as e:
    st.error(f"Error loading data from {file_path}: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Data Preparation
# -----------------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values('Date', inplace=True)

# Compute ratio, replace inf with NaN, fill NaN with 0
ratio_series = df['VVIX'] / df['VIX']
ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan)
ratio_series = ratio_series.fillna(0)

df['ratio'] = ratio_series
df['ratio_int'] = np.floor(df['ratio']).astype(int)

# Identify segments (change_id) where the floored ratio changes
df['change_id'] = (df['ratio_int'] != df['ratio_int'].shift(1)).cumsum()

# -----------------------------------------------------------------------------
# 7. Define Color Map (ratio=7 => black)
# -----------------------------------------------------------------------------
color_map = {
    0: 'grey',
    1: 'green',
    2: 'red',
    3: 'blue',
    4: 'orange',
    5: 'purple',
    6: 'brown',
    7: 'black',
    8: 'hotpink',
    9: 'olive',
    10: 'cyan'
}
unique_ratios = sorted(df['ratio_int'].unique())

# -----------------------------------------------------------------------------
# 8. Plotting (increase figure size)
# -----------------------------------------------------------------------------
fig, (ax_line, ax_bar) = plt.subplots(
    2, 1, sharex=True, figsize=(16, 10),
    gridspec_kw={'height_ratios': [2, 0.4]}
)

fig.patch.set_facecolor('white')
ax_line.set_facecolor('white')
ax_bar.set_facecolor('white')
sns.set_style("whitegrid")

# --- Top Subplot: Price Line with Colored Segments ---
for ratio_val in unique_ratios:
    ax_line.plot([], [], color=color_map.get(ratio_val, 'black'),
                 label=f'Ratio = {ratio_val}')

for i in range(len(df) - 1):
    x1 = df.iloc[i]['Date']
    y1 = df.iloc[i]['QQQ']
    ratio1 = df.iloc[i]['ratio_int']
    x2 = df.iloc[i+1]['Date']
    y2 = df.iloc[i+1]['QQQ']
    color = color_map.get(ratio1, 'black')
    ax_line.plot([x1, x2], [y1, y2], color=color, linewidth=2)

for _, grp in df.groupby('change_id'):
    ratio_val = grp['ratio_int'].iloc[0]
    color = color_map.get(ratio_val, 'black')
    start_date = grp['Date'].iloc[0]
    end_date = grp['Date'].iloc[-1]
    ax_line.axvspan(start_date, end_date, color=color, alpha=0.2)

plot_title = f"{dataset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio"
ax_line.set_title(plot_title, fontsize=16)
ax_line.set_ylabel('Price', fontsize=12)
ax_line.legend(title='Floored Ratio')

# --- Bottom Subplot: Color-Coded Bar Indicator ---
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
ax_bar.set_title('Floored Ratio Over Time (Color-Coded Bar)', fontsize=12)
ax_bar.legend(title='Floored Ratio', loc='upper left')

# X-axis formatting
month_locator = mdates.MonthLocator()
month_formatter = mdates.DateFormatter('%b %Y')
ax_line.xaxis.set_major_locator(month_locator)
ax_line.xaxis.set_major_formatter(month_formatter)
ax_bar.xaxis.set_major_locator(month_locator)
ax_bar.xaxis.set_major_formatter(month_formatter)
plt.setp(ax_line.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=45, ha='right')

ax_line.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
ax_bar.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')

plt.tight_layout()
st.pyplot(fig)

# -----------------------------------------------------------------------------
# 9. Display Processed Data (Wide)
# -----------------------------------------------------------------------------
st.markdown("## Processed Data")
st.markdown("""
Below is the processed DataFrame used to generate the above plots.
It includes columns (`Date`, `VIX`, `VVIX`, `QQQ`), the computed **ratio**,  
the floored ratio (**ratio_int**), and the **change_id** (segment index).
""")
st.dataframe(df, height=300, use_container_width=True)
