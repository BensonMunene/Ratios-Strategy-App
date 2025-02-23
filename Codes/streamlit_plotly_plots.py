import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("YMAX & YMAG Trading Results Dashboard")

# Load the uploaded Excel file
file_path = "D:/Benson/aUpWork/Douglas Backtester Algo/Backtester Algorithm/Data/Strategy 2 Performance.xlsx"                                          

xls = pd.ExcelFile(file_path)

# Read YMAX and YMAG trading results
ymax_df = pd.read_excel(xls, sheet_name="Ymax Trading Results")
ymag_df = pd.read_excel(xls, sheet_name="YMAG Trading Results")

# Function to create Plotly chart
def create_plot(df, asset_name):
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    entry_days = df[df["Entry"] == True]
    exit_days = df[df["Exit"] == True]
    
    fig = go.Figure()
    
    # Asset Price (Left Axis)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df[asset_name],
            mode="lines",
            line=dict(color="blue", width=2),
            name=f"{asset_name} Price",
            yaxis="y1"
        )
    )
    
    # Entry Markers
    fig.add_trace(
        go.Scatter(
            x=entry_days["Date"],
            y=entry_days[asset_name],
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
            y=exit_days[asset_name],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12),
            name="Exit",
            yaxis="y1"
        )
    )
    
    # Portfolio Value (Right Axis)
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Portfolio_Value"],
            mode="lines",
            line=dict(color="red", width=2),
            name="Portfolio Value",
            yaxis="y2"
        )
    )
    
    # Layout Configuration
    fig.update_layout(
        title=f"{asset_name} Price vs. Portfolio Value",
        xaxis=dict(title="Date", type="date", tickformat="%Y-%m", tickangle=45),
        yaxis=dict(title=f"{asset_name} Price ($)", side="left", showgrid=False, color="blue"),
        yaxis2=dict(title="Portfolio Value ($)", side="right", overlaying="y", position=1.0,
                    showgrid=True, gridwidth=1, gridcolor="lightgray", color="red"),
        legend=dict(x=0.5, y=1.0, xanchor='center', yanchor='top',
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1),
        hovermode="x unified"
    )
    return fig

# Create plots
ymax_plot = create_plot(ymax_df, "YMAX")
ymag_plot = create_plot(ymag_df, "YMAG")

# Display in Streamlit tabs
tab1, tab2 = st.tabs(["YMAX Trading Results", "YMAG Trading Results"])

with tab1:
    st.plotly_chart(ymax_plot, use_container_width=True)

with tab2:
    st.plotly_chart(ymag_plot, use_container_width=True)




