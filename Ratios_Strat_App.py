import warnings
import logging

# Suppress specific Streamlit warnings regarding missing ScriptRunContext and session state
warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
warnings.filterwarnings("ignore", message="Session state does not function when running a script without `streamlit run`")

# Set the logging level for Streamlit's scriptrunner to ERROR to reduce warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.dates import DateFormatter

# -----------------------------------------------------------------------------
# 1. Set Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Asset Price Analysis by VVIX/VIX Ratio",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Seaborn Theme for Aesthetics
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

# -----------------------------------------------------------------------------
# 3. Main Title & Brief Description
# -----------------------------------------------------------------------------
st.title("Multi-Asset Price Analysis by VVIX/VIX Ratio")
st.markdown("""
This app analyzes price data for multiple assets by computing the **VVIX/VIX** ratio.  
The ratio is **floored** to create integer segments.
""")

# -----------------------------------------------------------------------------
# 4. Asset and Timeframe Selection
# -----------------------------------------------------------------------------
st.markdown("#### Select Asset")
st.markdown("Choose from **YMAX**, **YMAG**, **QQQ**, or **GLD**.")
asset_option = st.selectbox("Select Asset", ["YMAX", "YMAG", "QQQ", "GLD"])

st.markdown("#### Select Time Frequency of the Asset Data")
st.markdown("""
Choose the desired frequency of the data: 
- **Daily** Frequency Data
- **4H** (4 hours frequency)
- **1H** (1 hour frequency)
- **30M** (30 minutes frequency)
""")
timeframe_option = st.selectbox("Select Timeframe", ["Daily", "4H", "1H", "30M"])

# -----------------------------------------------------------------------------
# 5. Filename Logic (Assuming CSV files are in the same directory as the script)
# -----------------------------------------------------------------------------
def get_filename(asset, timeframe):
    """
    Return the CSV filename for the chosen asset & timeframe.
    
    - YMAX / YMAG: old naming convention => {asset}_VIX_VVIX_QQQ_{timeframe}.csv
    - QQQ: new naming => QQQ_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
    - GLD: new naming => GLD_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
    """
    if asset in ["YMAX", "YMAG"]:
        return f"{asset}_VIX_VVIX_QQQ_{timeframe}.csv"
    elif asset == "QQQ":
        if timeframe == "Daily":
            return "QQQ_VIX_VVIX_Daily.csv"
        elif timeframe == "4H":
            return "QQQ_VIX_VVIX_4H.csv"
        elif timeframe == "1H":
            return "QQQ_VIX_VVIX_Hourly.csv"
        elif timeframe == "30M":
            return "QQQ_VIX_VVIX_30Mins.csv"
    elif asset == "GLD":
        if timeframe == "Daily":
            return "GLD_VIX_VVIX_Daily.csv"
        elif timeframe == "4H":
            return "GLD_VIX_VVIX_4H.csv"
        elif timeframe == "1H":
            return "GLD_VIX_VVIX_Hourly.csv"
        elif timeframe == "30M":
            return "GLD_VIX_VVIX_30Mins.csv"

# -----------------------------------------------------------------------------
# 6. Load Data Immediately Once Asset/Timeframe Are Selected
# -----------------------------------------------------------------------------
filename = get_filename(asset_option, timeframe_option)
file_path = os.path.join(os.getcwd(), filename)

@st.cache_data
def load_data(path):
    df_temp = pd.read_csv(path)
    # If the CSV has "Dates" instead of "Date", rename it
    if 'Dates' in df_temp.columns and 'Date' not in df_temp.columns:
        df_temp.rename(columns={'Dates': 'Date'}, inplace=True)
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
    df_temp.sort_values('Date', inplace=True)
    return df_temp

try:
    df = load_data(file_path)
except Exception as e:
    st.error(f"Error loading data from {file_path}: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 7. Determine Default Date Range Based on Data
# -----------------------------------------------------------------------------
if df.empty:
    st.warning("No data available for this asset/timeframe combination.")
    st.stop()

price_column = asset_option
if price_column not in df.columns:
    st.error(f"Column '{price_column}' not found in the dataset. Columns found: {list(df.columns)}")
    st.stop()

earliest_date = df['Date'].iloc[0].date()
latest_date   = df['Date'].iloc[-1].date()

# -----------------------------------------------------------------------------
# 8. Date Range Inputs
# -----------------------------------------------------------------------------
st.markdown("### Specify a Date Range for the Zones Indicator")
col1, col2 = st.columns(2)

with col1:
    from_date = st.date_input(
        "From Date",
        value=earliest_date,
        min_value=earliest_date,
        max_value=latest_date,
        help="Select the start date (automatically set to the earliest date in the dataset)."
    )
with col2:
    to_date = st.date_input(
        "To Date",
        value=latest_date,
        min_value=earliest_date,
        max_value=latest_date,
        help="Select the end date (automatically set to the latest date in the dataset)."
    )

# -----------------------------------------------------------------------------
# 9. Button to Generate Zones Indicator
# -----------------------------------------------------------------------------
st.markdown("""
You have successfully specified your asset, timeframe, and date range.  
Please click the button below to generate the zones indicator plot **using your chosen date range**.
""")
generate_button = st.button("Generate the zones indicator")

if generate_button:
    # Filter data to the chosen date range
    mask = (df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))
    df_filtered = df.loc[mask].copy()

    if df_filtered.empty:
        st.warning("No data available in the specified date range.")
        st.stop()

    # Compute ratio (VVIX / VIX), handle infinities and NaNs
    df_filtered['ratio'] = df_filtered['VVIX'] / df_filtered['VIX']
    df_filtered['ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered['ratio'].fillna(0, inplace=True)

    # Create floored ratio column and identify segments where the ratio changes
    df_filtered['ratio_int'] = np.floor(df_filtered['ratio']).astype(int)
    df_filtered['change_id'] = (df_filtered['ratio_int'] != df_filtered['ratio_int'].shift(1)).cumsum()

    # -----------------------------------------------------------------------------
    # 10. Define Color Map
    # -----------------------------------------------------------------------------
    color_map = {
        0: 'grey',
        1: 'limegreen',
        2: 'crimson',
        3: 'dodgerblue',
        4: 'gold',
        5: 'mediumpurple',
        6: 'sienna',
        7: 'black',
        8: 'deeppink',
        9: 'darkolivegreen',
        10: 'cyan'
    }

    min_price = df_filtered[price_column].min()
    max_price = df_filtered[price_column].max()

    # -----------------------------------------------------------------------------
    # 11. Plotly Date/Time Formatting
    # -----------------------------------------------------------------------------
    if timeframe_option == "Daily":
        xaxis_dtick = "M1"
        xaxis_tickformat = "%b %Y"
        xaxis_hoverformat = "%b %d, %Y"
    else:
        xaxis_dtick = None
        xaxis_tickformat = "%b %d %H:%M"
        xaxis_hoverformat = "%b %d, %Y %H:%M"

    # -----------------------------------------------------------------------------
    # 12. Create the Plotly Figure
    # -----------------------------------------------------------------------------
    def create_plotly_figure(df_data):
        plotly_title = f"{asset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio"
        fig = make_subplots(rows=1, cols=1)
        groups = list(df_data.groupby('change_id'))
        added_legends = set()

        for i, (change_id, group_original) in enumerate(groups):
            group_original = group_original.copy()
            start_date = group_original['Date'].iloc[0]
            end_date   = group_original['Date'].iloc[-1]
            
            # Skip the first row for boundary duplication (except for the first group)
            if i == 0:
                line_data = group_original
            else:
                line_data = group_original.iloc[1:]
            
            ratio_val = group_original['ratio_int'].iloc[0]
            color = color_map.get(ratio_val, 'black')

            # Background rectangle for the segment
            fig.add_shape(
                type='rect',
                x0=start_date,
                x1=end_date,
                y0=min_price,
                y1=max_price,
                xref='x',
                yref='y',
                fillcolor=color,
                opacity=0.3,
                line_width=0,
                layer='below'
            )
            
            # Only show legend once per ratio value
            show_legend = ratio_val not in added_legends
            if show_legend:
                added_legends.add(ratio_val)

            # Add the price line for this segment
            fig.add_trace(
                go.Scatter(
                    x=line_data['Date'],
                    y=line_data[price_column],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'Ratio = {ratio_val}',
                    showlegend=show_legend
                )
            )

        fig.update_xaxes(
            dtick=xaxis_dtick,
            tickformat=xaxis_tickformat,
            hoverformat=xaxis_hoverformat,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikethickness=1,
            spikecolor='grey'
        )
        fig.update_yaxes(
            title_text='Price',
            showspikes=True,
            spikemode='across',
            spikethickness=1,
            spikecolor='grey'
        )
        fig.update_layout(
            title={
                'text': plotly_title,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            hovermode="closest",
            showlegend=True,
            legend_title='Floored Ratio',
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig

    # -----------------------------------------------------------------------------
    # 13. Create the Matplotlib/Seaborn Figure (Continuous Plot)
    # -----------------------------------------------------------------------------
    def create_matplotlib_figure(df_data):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare legend placeholders for each unique ratio
        unique_ratios = sorted(df_data['ratio_int'].unique())
        for ratio_val in unique_ratios:
            ax.plot([], [], color=color_map.get(ratio_val, 'black'),
                    label=f'Ratio = {ratio_val}')

        # Plot line segments for each consecutive pair of data points
        for i in range(len(df_data) - 1):
            x1 = df_data.iloc[i]['Date']
            y1 = df_data.iloc[i][price_column]
            ratio1 = df_data.iloc[i]['ratio_int']
            x2 = df_data.iloc[i+1]['Date']
            y2 = df_data.iloc[i+1][price_column]
            color = color_map.get(ratio1, 'black')
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

        # Add semi-transparent background rectangles for each ratio segment
        for _, grp in df_data.groupby('change_id'):
            ratio_val = grp['ratio_int'].iloc[0]
            color = color_map.get(ratio_val, 'black')
            start_date = grp['Date'].iloc[0]
            end_date = grp['Date'].iloc[-1]
            ax.axvspan(start_date, end_date, facecolor=color, alpha=0.3)

        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price", fontsize=10)
        ax.set_ylim(min_price, max_price)

        # Format the date labels (e.g., "Jan 01, 2022")
        date_formatter = DateFormatter("%b %d, %Y")
        ax.xaxis.set_major_formatter(date_formatter)

        # Rotate the x-axis labels 90 degrees and adjust font size
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        ax.grid(True, which='major', axis='both', alpha=0.5)
        ax.legend(
            title="Floored Ratio",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            title_fontsize=10
        )
        ax.set_title(
            f"{asset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio",
            fontsize=12
        )

        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------------------
    # 14. Create Tabs for Plotly vs. Matplotlib
    # -----------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["Plotly Plot", "Matplotlib Plot"])

    with tab1:
        plotly_fig = create_plotly_figure(df_filtered)
        st.plotly_chart(plotly_fig, use_container_width=True)

    with tab2:
        matplotlib_fig = create_matplotlib_figure(df_filtered)
        st.pyplot(matplotlib_fig)

    # -----------------------------------------------------------------------------
    # 15. Display Processed Data
    # -----------------------------------------------------------------------------
    st.markdown("## Processed Data")
    st.markdown("""
    Below is the processed DataFrame used to generate the above plots.
    It includes columns (`Date`, `VIX`, `VVIX`, and the asset price column),
    the computed **ratio**, the floored ratio (**ratio_int**), and the **change_id** (segment index).
    """)
    st.dataframe(df_filtered, height=300, use_container_width=True)
