import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==============================================
# Page Configuration
# ==============================================
st.set_page_config(page_title="YMAX YMAG Backtester", layout="wide")

# ==============================================
# Helper Functions
# ==============================================
def compute_rolling_correlations(df, window):
    """
    Compute rolling correlations for:
      - YMAX-VIX
      - YMAX-VVIX
      - YMAG-VIX
      - YMAG-VVIX
    using a specified rolling window.
    """
    # Calculate daily returns for correlation calculations
    returns = df.loc[:, ~df.columns.str.contains("Dividends")].pct_change()
    corr_df = pd.DataFrame(index=df.index)

    corr_df["YMAX-VIX Correlation"] = (
        returns["YMAX"].rolling(window=window).corr(returns["VIX"])
    )
    corr_df["YMAX-VVIX Correlation"] = (
        returns["YMAX"].rolling(window=window).corr(returns["VVIX"])
    )
    corr_df["YMAG-VIX Correlation"] = (
        returns["YMAG"].rolling(window=window).corr(returns["VIX"])
    )
    corr_df["YMAG-VVIX Correlation"] = (
        returns["YMAG"].rolling(window=window).corr(returns["VVIX"])
    )

    merged = df.join(corr_df)
    merged.dropna(inplace=True)
    return merged


def backtest_strategy_1(df, asset="YMAX", initial_investment=10_000):
    """
    Implements Strategy 1 on the given DataFrame for the chosen asset ("YMAX" or "YMAG").
    Strategy 1 rules:
      1) If VIX < 20 and VVIX < 100 → Long (No Hedge)
      2) If VIX >= 20 or VVIX >= 100 → Long + Short QQQ
      3) If VIX >= 20 or VVIX >= 100 and correlation < -0.3 → No Investment
    """
    temp_df = df.copy()

    if asset == "YMAX":
        corr_vix_col = "YMAX-VIX Correlation"
        corr_vvix_col = "YMAX-VVIX Correlation"
        price_col = "YMAX"
        div_col = "YMAX Dividends"
    else:
        corr_vix_col = "YMAG-VIX Correlation"
        corr_vvix_col = "YMAG-VVIX Correlation"
        price_col = "YMAG"
        div_col = "YMAG Dividends"

    def strategy_rule(row):
        # Rule 1
        if row["VIX"] < 20 and row["VVIX"] < 100:
            return "Long (No Hedge)"
        # Rule 2 or Rule 3
        elif row["VIX"] >= 20 or row["VVIX"] >= 100:
            # Check correlation for "No Investment" scenario
            if row[corr_vix_col] < -0.3 or row[corr_vvix_col] < -0.3:
                return "No Investment"
            else:
                return "Long + Short QQQ"
        else:
            return "No Investment"

    temp_df["Strategy"] = temp_df.apply(strategy_rule, axis=1)
    temp_df["Portfolio_Value"] = initial_investment
    temp_df["QQQ_Shares_Short"] = 0
    temp_df["QQQ_Short_Loss"] = 0

    for i in range(1, len(temp_df)):
        prev_val = temp_df.iloc[i - 1]["Portfolio_Value"]
        today_strategy = temp_df.iloc[i]["Strategy"]
        y_price_yest = temp_df.iloc[i - 1][price_col]
        q_price_yest = temp_df.iloc[i - 1]["QQQ"]
        y_price_today = temp_df.iloc[i][price_col]
        y_div_today = temp_df.iloc[i][div_col]

        # Carry forward QQQ shares short by default
        temp_df.at[temp_df.index[i], "QQQ_Shares_Short"] = temp_df.iloc[i - 1]["QQQ_Shares_Short"]

        if today_strategy == "Long (No Hedge)":
            shares_held = prev_val / y_price_yest
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = shares_held * (y_price_today + y_div_today)

        elif today_strategy == "Long + Short QQQ":
            shares_held = prev_val / y_price_yest
            # If switching to hedge from a different strategy, recalc QQQ shares
            if temp_df.iloc[i - 1]["Strategy"] != "Long + Short QQQ":
                temp_df.at[temp_df.index[i], "QQQ_Shares_Short"] = prev_val / q_price_yest

            qqq_shares_short = temp_df.iloc[i]["QQQ_Shares_Short"]
            q_price_today = temp_df.iloc[i]["QQQ"]
            hedge_pnl = qqq_shares_short * (q_price_yest - q_price_today)

            # Record hedge profit/loss
            temp_df.at[temp_df.index[i], "QQQ_Short_Loss"] = -hedge_pnl
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = (
                shares_held * (y_price_today + y_div_today)
            ) + hedge_pnl

        else:  # "No Investment"
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df


def calculate_performance_metrics(portfolio_df):
    """
    Given a portfolio DataFrame with 'Portfolio_Value' and 'Date' columns,
    calculate key performance metrics:
      - Total Return
      - CAGR
      - Ann. Volatility
      - Sharpe Ratio
      - Max Drawdown
      - Calmar Ratio
    Returns a dict with these metrics.
    """
    df = portfolio_df.dropna(subset=["Portfolio_Value"]).copy()
    if len(df) < 2:
        return {}

    start_val = df.iloc[0]["Portfolio_Value"]
    end_val = df.iloc[-1]["Portfolio_Value"]
    total_return = (end_val / start_val - 1) * 100

    num_days = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days
    years = num_days / 365 if num_days > 0 else 1
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    # Annualized Volatility => "Ann. Volatility (%)"
    ann_vol = df["Portfolio_Return"].std() * np.sqrt(252) * 100

    # Sharpe Ratio (assuming 2% risk-free)
    risk_free_rate = 0.02
    sharpe = (cagr / 100 - risk_free_rate) / (ann_vol / 100) if ann_vol != 0 else np.nan

    # Max Drawdown
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown = df["Portfolio_Value"] / rolling_max - 1
    max_dd = drawdown.min() * 100  # negative value

    # Calmar Ratio
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Ann. Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Calmar Ratio": calmar,
    }


def plot_portfolio_value(df, asset_label="Portfolio Value"):
    """
    Returns a Plotly line chart of 'Portfolio_Value' vs. 'Date'.
    """
    fig = px.line(
        df,
        x="Date",
        y="Portfolio_Value",
        title=f"{asset_label} Over Time",
        labels={"Portfolio_Value": "Portfolio Value ($)"},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(xaxis_title="Date", yaxis_title="Value ($)")
    return fig


def plot_strategy_distribution(df):
    """
    Returns a Plotly bar chart for distribution of Strategy usage.
    """
    strat_counts = df["Strategy"].value_counts()
    fig = px.bar(
        x=strat_counts.index,
        y=strat_counts.values,
        labels={"x": "Strategy", "y": "Number of Days"},
        title="Strategy Distribution Over Time",
    )
    return fig


def plot_drawdown(df):
    """
    Returns a Plotly area chart of drawdown (%) over time.
    """
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown_series = (df["Portfolio_Value"] / rolling_max - 1) * 100
    dd_df = df[["Date"]].copy()
    dd_df["Drawdown"] = drawdown_series

    fig = px.area(
        dd_df,
        x="Date",
        y="Drawdown",
        title="Drawdown Over Time",
        labels={"Drawdown": "Drawdown (%)"},
    )
    fig.update_traces(line=dict(width=2), fill="tozeroy")
    return fig


# ==============================================
# Sidebar Navigation
# ==============================================
page = st.sidebar.radio("Navigation", ["Backtester", "Strategy Overview", "About"])

# ==============================================
# 1) BACKTESTER PAGE
# ==============================================
if page == "Backtester":
    st.title("YMAX YMAG Backtester")

    # ~~~~~~~~~~~~~~~~ HEADERS ~~~~~~~~~~~~~~~~~
    st.header("Strategy Summaries")
    st.markdown(
        """
**Strategy 1:** Uses VIX/VVIX thresholds and correlation with market volatility to determine 
long positions in YMAX/YMAG—with hedging via QQQ when volatility is high.

**Strategy 2:** Enters positions only when VIX and VVIX are within a narrow “safe” range, 
exiting if conditions stray and re-entering once stability returns.
"""
    )

    # ~~~~~~~~~~~~~~~~ SELECTIONS ~~~~~~~~~~~~~~~
    col_sel1, col_sel2 = st.columns([1, 1])  # Adjust widths if desired

    with col_sel1:
        st.subheader("Asset Selection")
        asset_choice = st.radio(
            "Select the asset(s) to backtest:",
            options=["YMAX", "YMAG", "Both"],
            index=2  # default to "Both" if you like
        )

    with col_sel2:
        st.subheader("Strategy Selection")
        strategy_choice = st.radio(
            "Select strategy to backtest:",
            options=["Strategy 1 (Investment Rules)", "Strategy 2 (Investment Rules)", "Both Strategies"]
        )

        if strategy_choice == "Both Strategies":
            st.markdown("**Specify Strategy Priority**")
            primary_strategy = st.radio(
                "Choose the primary strategy:",
                options=["Strategy 1 (Investment Rules)", "Strategy 2 (Investment Rules)"]
            )
            st.markdown("**Combination Mode**")
            combination_mode = st.radio(
                "Select the combination mode:",
                options=["Close by itself", "Continue running"]
            )

    st.markdown("---")

    # ~~~~~~~~~~~~~~~~ PARAMETERS & RUN ~~~~~~~~~~~~~~~
    st.subheader("Strategy 1 Parameters")
    corr_window = st.slider("Select correlation window (days):", min_value=1, max_value=30, value=14)

    run_backtest = st.button("Run Backtest for Selected Strategy(ies)")

    # ~~~~~~~~~~~~~~~~ BACKTEST LOGIC ~~~~~~~~~~~~~~~
    if run_backtest:
        # 1) Load CSV
        try:
            all_assets = pd.read_csv("All Assets and Dividends.csv")
        except FileNotFoundError:
            st.error("Could not find 'All Assets and Dividends.csv'. Make sure it's in the same folder.")
            st.stop()

        # 2) Prepare data
        all_assets["Date"] = pd.to_datetime(all_assets["Date"])
        all_assets.set_index("Date", inplace=True)
        all_assets.sort_index(inplace=True)

        # 3) Compute rolling correlations
        prices_and_stats = compute_rolling_correlations(all_assets, corr_window)
        prices_and_stats.reset_index(inplace=True)

        # 4) Strategy 1 Backtest (or part of "Both Strategies")
        run_strat_1 = (strategy_choice in ["Strategy 1 (Investment Rules)", "Both Strategies"])

        if run_strat_1:
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # A) BOTH Assets
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if asset_choice == "Both":
                st.markdown("## Strategy 1 Backtest - YMAX")
                ymax_res = backtest_strategy_1(prices_and_stats, asset="YMAX")

                # YMAX Charts
                col_ymax1, col_ymax2 = st.columns(2)
                with col_ymax1:
                    fig_val_ymax = plot_portfolio_value(ymax_res, asset_label="YMAX Portfolio")
                    st.plotly_chart(fig_val_ymax, use_container_width=True)
                with col_ymax2:
                    fig_strat_ymax = plot_strategy_distribution(ymax_res)
                    st.plotly_chart(fig_strat_ymax, use_container_width=True)

                # Drawdown (full width)
                fig_dd_ymax = plot_drawdown(ymax_res)
                st.plotly_chart(fig_dd_ymax, use_container_width=True)

                # Performance Table
                ymax_metrics = calculate_performance_metrics(ymax_res)
                if ymax_metrics:
                    df_ymax_perf = pd.DataFrame([ymax_metrics], index=["YMAX Strategy"]).round(2)
                    st.dataframe(df_ymax_perf)
                else:
                    st.info("Not enough data points for YMAX metrics.")

                st.markdown("## Strategy 1 Backtest - YMAG")
                ymag_res = backtest_strategy_1(prices_and_stats, asset="YMAG")

                # YMAG Charts
                col_ymag1, col_ymag2 = st.columns(2)
                with col_ymag1:
                    fig_val_ymag = plot_portfolio_value(ymag_res, asset_label="YMAG Portfolio")
                    st.plotly_chart(fig_val_ymag, use_container_width=True)
                with col_ymag2:
                    fig_strat_ymag = plot_strategy_distribution(ymag_res)
                    st.plotly_chart(fig_strat_ymag, use_container_width=True)

                # Drawdown (full width)
                fig_dd_ymag = plot_drawdown(ymag_res)
                st.plotly_chart(fig_dd_ymag, use_container_width=True)

                # Performance Table
                ymag_metrics = calculate_performance_metrics(ymag_res)
                if ymag_metrics:
                    df_ymag_perf = pd.DataFrame([ymag_metrics], index=["YMAG Strategy"]).round(2)
                    st.dataframe(df_ymag_perf)
                else:
                    st.info("Not enough data points for YMAG metrics.")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # B) SINGLE Asset (YMAX or YMAG)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                st.markdown(f"## Strategy 1 Backtest - {asset_choice}")
                res = backtest_strategy_1(prices_and_stats, asset=asset_choice)

                # Charts
                col1, col2 = st.columns(2)
                with col1:
                    fig_val = plot_portfolio_value(res, asset_label=f"{asset_choice} Portfolio")
                    st.plotly_chart(fig_val, use_container_width=True)
                with col2:
                    fig_strat = plot_strategy_distribution(res)
                    st.plotly_chart(fig_strat, use_container_width=True)

                # Drawdown
                fig_dd = plot_drawdown(res)
                st.plotly_chart(fig_dd, use_container_width=True)

                # Performance
                metrics_res = calculate_performance_metrics(res)
                if metrics_res:
                    perf_df = pd.DataFrame([metrics_res], index=[f"{asset_choice} Strategy"]).round(2)
                    st.dataframe(perf_df)
                else:
                    st.info("Not enough data points for metrics.")
        else:
            st.warning("Strategy 1 is not selected. (Strategy 2 is still a placeholder.)")

    else:
        st.info("Click 'Run Backtest for Selected Strategy(ies)' to see results.")


# ==============================================
# 2) STRATEGY OVERVIEW PAGE
# ==============================================
elif page == "Strategy Overview":
    st.title("Strategy Overview")
    st.markdown("## Table of Contents")
    st.markdown("- [Strategy 1 Detailed Explanation](#strategy-1-detailed-explanation)")
    st.markdown("- [Strategy 2 Detailed Explanation](#strategy-2-detailed-explanation)")
    st.markdown("---")

    st.markdown("### Strategy 1 Detailed Explanation")
    st.markdown(
        """
**Investment Rules for Strategy 1:**
1. **Rule 1:** If VIX < 20 and VVIX < 100 → Long YMAX/YMAG (no hedge).
2. **Rule 2:** If VIX ≥ 20 or VVIX ≥ 100 → Long YMAX/YMAG and short an equal dollar amount of QQQ.
3. **Rule 3:** If VIX ≥ 20 or VVIX ≥ 100 and correlation of YMAX/YMAG with VIX/VVIX < -0.3 → No investment.
"""
    )

    st.markdown("### Strategy 2 Detailed Explanation")
    st.markdown(
        """
**Investment Rules for Strategy 2:**
1. **Invest Only If:**  
   - VIX is between 15 and 20 (inclusive) **AND**  
   - VVIX is between 90 and 100.

2. **Exit the Market If:**  
   - VIX drops below 15 or rises above 20, **OR**  
   - VVIX goes above or equal to 100, or falls below 90.

3. **Re-Enter the Market When:**  
   - VIX is again within 15–20, **AND**  
   - VVIX is between 90 and 95 (inclusive).

**Summary of Logic:**
- **In-Market Condition:** VIX ∈ [15, 20] and VVIX ∈ [90, 100)
- **Exit Condition:** VIX < 15 or VIX > 20 or VVIX < 90 or VVIX ≥ 100
- **Re-Entry Condition:** VIX ∈ [15, 20] and VVIX ∈ [90, 95]
"""
    )

# ==============================================
# 3) ABOUT PAGE
# ==============================================
elif page == "About":
    st.title("About")
    st.write("This page is under construction.")
