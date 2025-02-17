import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Dual Plot Dashboard: YMAG and YMAX")

# ==============================================================
# Create Sample Data for Both YMAG and YMAX
# ==============================================================

# Common date range
dates = pd.date_range(start="2021-01-01", periods=100, freq="D")

# --- Sample Data for YMAG ---
np.random.seed(42)  # seed for reproducibility
ymag_price = np.cumsum(np.random.normal(loc=0.5, scale=2, size=100)) + 100
portfolio_value_ymag = np.cumsum(np.random.normal(loc=0.3, scale=1.5, size=100)) + 50
in_market_ymag = np.random.choice([True, False], size=100)

ymag_df = pd.DataFrame({
    "Date": dates,
    "YMAG": ymag_price,
    "Portfolio_Value": portfolio_value_ymag,
    "In_Market": in_market_ymag
})

# --- Sample Data for YMAX ---
np.random.seed(101)  # different seed for YMAX
ymax_price = np.cumsum(np.random.normal(loc=0.4, scale=2, size=100)) + 200
portfolio_value_ymax = np.cumsum(np.random.normal(loc=0.2, scale=1.5, size=100)) + 100
in_market_ymax = np.random.choice([True, False], size=100)

ymax_df = pd.DataFrame({
    "Date": dates,
    "YMAX": ymax_price,
    "Portfolio_Value": portfolio_value_ymax,
    "In_Market": in_market_ymax
})

# ==============================================================
# Plot 1: YMAG Price vs. Portfolio Value
# ==============================================================

# --- Identify Entry and Exit Days for YMAG ---
ymag_df["Entry"] = (ymag_df["In_Market"].shift(1) == False) & (ymag_df["In_Market"] == True)
ymag_df["Exit"]  = (ymag_df["In_Market"].shift(1) == True) & (ymag_df["In_Market"] == False)

entry_days_ymag = ymag_df[ymag_df["Entry"]]
exit_days_ymag  = ymag_df[ymag_df["Exit"]]

# Sort data by Date
ymag_df.sort_values("Date", inplace=True)
ymag_df.reset_index(drop=True, inplace=True)

# --- Create Plotly Figure for YMAG ---
fig1 = go.Figure()

# YMAG Price (Left Axis)
fig1.add_trace(
    go.Scatter(
        x=ymag_df["Date"],
        y=ymag_df["YMAG"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="YMAG Price",
        yaxis="y1"
    )
)

# Entry Markers (triangle-up)
fig1.add_trace(
    go.Scatter(
        x=entry_days_ymag["Date"],
        y=entry_days_ymag["YMAG"],
        mode="markers",
        marker=dict(symbol="triangle-up", color="blue", size=12),
        name="Entry",
        yaxis="y1"
    )
)

# Exit Markers (triangle-down)
fig1.add_trace(
    go.Scatter(
        x=exit_days_ymag["Date"],
        y=exit_days_ymag["YMAG"],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=12),
        name="Exit",
        yaxis="y1"
    )
)

# Portfolio Value (Right Axis)
fig1.add_trace(
    go.Scatter(
        x=ymag_df["Date"],
        y=ymag_df["Portfolio_Value"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Portfolio Value",
        yaxis="y2"
    )
)

# Layout configuration for YMAG plot
fig1.update_layout(
    title="YMAG Price (Left Axis) vs. Portfolio Value (Right Axis)",
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

# ==============================================================
# Plot 2: YMAX Price vs. Portfolio Value
# ==============================================================

# --- Identify Entry and Exit Days for YMAX ---
ymax_df["Entry"] = (ymax_df["In_Market"].shift(1) == False) & (ymax_df["In_Market"] == True)
ymax_df["Exit"]  = (ymax_df["In_Market"].shift(1) == True) & (ymax_df["In_Market"] == False)

entry_days_ymax = ymax_df[ymax_df["Entry"]]
exit_days_ymax  = ymax_df[ymax_df["Exit"]]

# Sort data by Date
ymax_df.sort_values("Date", inplace=True)
ymax_df.reset_index(drop=True, inplace=True)

# --- Create Plotly Figure for YMAX ---
fig2 = go.Figure()

# YMAX Price (Left Axis)
fig2.add_trace(
    go.Scatter(
        x=ymax_df["Date"],
        y=ymax_df["YMAX"],
        mode="lines",
        line=dict(color="blue", width=2),
        name="YMAX Price",
        yaxis="y1"
    )
)

# Entry Markers (triangle-up)
fig2.add_trace(
    go.Scatter(
        x=entry_days_ymax["Date"],
        y=entry_days_ymax["YMAX"],
        mode="markers",
        marker=dict(symbol="triangle-up", color="blue", size=12),
        name="Entry",
        yaxis="y1"
    )
)

# Exit Markers (triangle-down)
fig2.add_trace(
    go.Scatter(
        x=exit_days_ymax["Date"],
        y=exit_days_ymax["YMAX"],
        mode="markers",
        marker=dict(symbol="triangle-down", color="red", size=12),
        name="Exit",
        yaxis="y1"
    )
)

# Portfolio Value (Right Axis)
fig2.add_trace(
    go.Scatter(
        x=ymax_df["Date"],
        y=ymax_df["Portfolio_Value"],
        mode="lines",
        line=dict(color="red", width=2),
        name="Portfolio Value",
        yaxis="y2"
    )
)

# Layout configuration for YMAX plot
fig2.update_layout(
    title="YMAX Price (Left Axis) vs. Portfolio Value (Right Axis)",
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

# ==============================================================
# Display the Two Plots in Separate Tabs
# ==============================================================

tab1, tab2 = st.tabs(["Plot 1: YMAG vs Portfolio", "Plot 2: YMAX vs Portfolio"])

with tab1:
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.plotly_chart(fig2, use_container_width=True)
