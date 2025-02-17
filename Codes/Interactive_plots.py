import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------
# STEP A: Load Data (Replace with Your Data)
# ---------------------------------------------
# You should replace this with actual data loading
data = {
    "Date": pd.date_range(start="2024-01-01", periods=50, freq="D"),
    "YMAG": [20 + i * 0.05 + (i % 5) * 0.2 for i in range(50)],
    "Portfolio_Value": [10000 + (i % 10) * 50 for i in range(50)],
    "In_Market": [True if i % 7 != 0 else False for i in range(50)]
}

# Convert to DataFrame
ymag_df = pd.DataFrame(data)

# ---------------------------------------------
# STEP B: Identify Entry and Exit Days for YMAG
# ---------------------------------------------
ymag_df["Entry"] = (ymag_df["In_Market"].shift(1) == False) & (ymag_df["In_Market"] == True)
ymag_df["Exit"]  = (ymag_df["In_Market"].shift(1) == True) & (ymag_df["In_Market"] == False)

# Prepare Entry & Exit Data
entry_days = ymag_df[ymag_df["Entry"] == True]
exit_days  = ymag_df[ymag_df["Exit"] == True]

# Ensure sorted data
ymag_df.sort_values("Date", inplace=True)
ymag_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------
# STEP C: Create the Plotly Figure for YMAG
# ---------------------------------------------
fig = go.Figure()

# YMAG Price (Left Axis)
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

# Entry Markers
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

# Exit Markers
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

# Portfolio Value (Right Axis)
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

# Configure Layout for Dual Axis
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
# STEP D: Streamlit Web App
# ---------------------------------------------
st.set_page_config(page_title="YMAG Trading Visualization", layout="wide")

st.title("📈 YMAG Trading Data Dashboard")
st.write("This dashboard visualizes the trading data of YMAG with entry/exit points.")

# Display Plotly Figure
st.plotly_chart(fig, use_container_width=True)

# Display Data Preview
st.subheader("📊 Data Preview")
st.dataframe(ymag_df)

st.markdown("Developed with **Streamlit & Plotly** 🚀")
