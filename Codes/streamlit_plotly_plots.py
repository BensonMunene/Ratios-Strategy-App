import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime

# ==============================================
# Page Configuration
# ==============================================
st.set_page_config(page_title="YMAX YMAG Backtester", layout="wide")

# ==============================================
# Strategy 1 Helper (Already Implemented)
# ==============================================
def backtest_strategy_1(df, asset="YMAX", initial_investment=10_000):
    """
    Implements Strategy 1 for the chosen asset ("YMAX" or "YMAG"):
      1) If VIX < 20 and VVIX < 100 => Long (No Hedge)
      2) If VIX >= 20 or VVIX >= 100 => Long + Short QQQ
      3) If VIX >= 20 or VVIX >= 100 and correlation < -0.3 => No Investment
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
        if row["VIX"] < 20 and row["VVIX"] < 100:
            return "Long (No Hedge)"
        elif row["VIX"] >= 20 or row["VVIX"] >= 100:
            # Check correlation for "No Investment"
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
            if temp_df.iloc[i - 1]["Strategy"] != "Long + Short QQQ":
                temp_df.at[temp_df.index[i], "QQQ_Shares_Short"] = prev_val / q_price_yest

            qqq_shares_short = temp_df.iloc[i]["QQQ_Shares_Short"]
            q_price_today = temp_df.iloc[i]["QQQ"]
            hedge_pnl = qqq_shares_short * (q_price_yest - q_price_today)

            temp_df.at[temp_df.index[i], "QQQ_Short_Loss"] = -hedge_pnl
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = (
                shares_held * (y_price_today + y_div_today)
            ) + hedge_pnl

        else:  # "No Investment"
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df

# ==============================================
# Strategy 2 Helper (Indicator Range-Based)
# ==============================================
def backtest_strategy_2(df, asset="YMAX", initial_investment=10_000):
    """
    Implements Strategy 2 for the chosen asset ("YMAX" or "YMAG"):
    1) Invest Only If:
       - VIX ∈ [15,20]
       - VVIX ∈ [90,100)
    2) Exit If:
       - VIX ∉ [15,20]
       - VVIX ∉ [90,100)
    3) Re-Enter If:
       - Already exited at least once
       - VIX ∈ [15,20]
       - VVIX ∈ [90,95]
    """
    temp_df = df.copy()

    # Identify columns for asset + dividends
    if asset == "YMAX":
        price_col = "YMAX"
        div_col = "YMAX Dividends"
    else:
        price_col = "YMAG"
        div_col = "YMAG Dividends"

    # We'll track a boolean 'in_market' plus 'entered_once'
    in_market = False
    entered_once = False

    temp_df["Strategy"] = "No Investment"
    temp_df["Portfolio_Value"] = initial_investment
    temp_df["Shares_Held"] = 0.0

    for i in range(1, len(temp_df)):
        prev_val = temp_df.iloc[i - 1]["Portfolio_Value"]
        prev_shares = temp_df.iloc[i - 1]["Shares_Held"]
        vix = temp_df.iloc[i]["VIX"]
        vvix = temp_df.iloc[i]["VVIX"]
        price_yest = temp_df.iloc[i - 1][price_col]
        price_today = temp_df.iloc[i][price_col]
        div_today = temp_df.iloc[i][div_col]

        # Carry forward shares by default
        temp_df.at[temp_df.index[i], "Shares_Held"] = prev_shares

        if in_market:
            # Check exit condition => if VIX out of [15,20] or VVIX out of [90,100)
            if (vix < 15 or vix > 20) or (vvix < 90 or vvix >= 100):
                # Exit
                in_market = False
                temp_df.at[temp_df.index[i], "Strategy"] = "Exit"
                # Portfolio stays same as previous day => no new shares
                temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val
                temp_df.at[temp_df.index[i], "Shares_Held"] = 0.0
            else:
                # Stay invested
                temp_df.at[temp_df.index[i], "Strategy"] = "Long"
                shares_held = prev_shares
                new_val = shares_held * (price_today + div_today)
                temp_df.at[temp_df.index[i], "Portfolio_Value"] = new_val
        else:
            # Not in market
            if not entered_once:
                # We have never entered => use the "Invest Only If" condition
                if (15 <= vix <= 20) and (90 <= vvix < 100):
                    # Enter
                    in_market = True
                    entered_once = True
                    temp_df.at[temp_df.index[i], "Strategy"] = "Long"
                    shares_held = prev_val / price_yest
                    temp_df.at[temp_df.index[i], "Shares_Held"] = shares_held
                    new_val = shares_held * (price_today + div_today)
                    temp_df.at[temp_df.index[i], "Portfolio_Value"] = new_val
                else:
                    # Remain out
                    temp_df.at[temp_df.index[i], "Strategy"] = "No Investment"
                    temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val
            else:
                # We have entered once before => check re-entry condition
                # Re-enter if VIX ∈ [15,20], VVIX ∈ [90,95]
                if (15 <= vix <= 20) and (90 <= vvix <= 95):
                    in_market = True
                    temp_df.at[temp_df.index[i], "Strategy"] = "Long"
                    shares_held = prev_val / price_yest
                    temp_df.at[temp_df.index[i], "Shares_Held"] = shares_held
                    new_val = shares_held * (price_today + div_today)
                    temp_df.at[temp_df.index[i], "Portfolio_Value"] = new_val
                else:
                    temp_df.at[temp_df.index[i], "Strategy"] = "No Investment"
                    temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val

    # Calculate daily returns
    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df

# ==============================================
# Combination Logic
# ==============================================
def backtest_combined_strategies(df, asset="YMAX", initial_investment=10_000,
                                 primary="Strategy 1", mode="Close by itself"):
    """
    Combines Strategy 1 & Strategy 2 for a single asset:
      - If mode == "Close by itself", once we enter a strategy, we stay until it exits, then we can switch.
      - If mode == "Continue running", we can run both in parallel (split capital 50/50 if both say invest).
    Returns a DataFrame with columns:
      [Portfolio_Value, Strategy, ...].
    """
    # We run each strategy's logic in parallel behind the scenes
    s1_df = backtest_strategy_1(df, asset=asset, initial_investment=initial_investment)
    s2_df = backtest_strategy_2(df, asset=asset, initial_investment=initial_investment)

    # We'll create a new DataFrame that picks daily from S1 or S2 or both
    combo_df = df.copy()
    combo_df["Strategy"] = "None"
    combo_df["Portfolio_Value"] = initial_investment
    combo_df["Portfolio_Return"] = 0.0

    # For "Close by itself" we track which strategy is active
    active_strategy = None
    capital = initial_investment

    # For "Continue running", we track daily partial allocation
    # We'll store daily capital for S1, S2
    s1_cap = initial_investment
    s2_cap = initial_investment

    for i in range(len(combo_df)):
        if i == 0:
            # initial row
            combo_df.at[combo_df.index[i], "Portfolio_Value"] = initial_investment
            continue

        # Grab today's signals from s1_df and s2_df
        s1_strat = s1_df.iloc[i]["Strategy"]
        s2_strat = s2_df.iloc[i]["Strategy"]
        # Yesterday's combo portfolio
        prev_val = combo_df.iloc[i - 1]["Portfolio_Value"]

        if mode == "Close by itself":
            # If we currently have an active strategy
            if active_strategy is None:
                # We are not in the market => check primary first
                if primary == "Strategy 1":
                    if s1_strat != "No Investment" and s1_strat != "Exit":
                        # Use strategy 1
                        active_strategy = "S1"
                        combo_df.at[combo_df.index[i], "Strategy"] = "S1"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = s1_df.iloc[i]["Portfolio_Value"]
                    else:
                        # check strategy 2
                        if s2_strat not in ["No Investment", "Exit"]:
                            active_strategy = "S2"
                            combo_df.at[combo_df.index[i], "Strategy"] = "S2"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = s2_df.iloc[i]["Portfolio_Value"]
                        else:
                            # no investment
                            combo_df.at[combo_df.index[i], "Strategy"] = "None"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = prev_val
                else:
                    # primary == "Strategy 2"
                    if s2_strat not in ["No Investment", "Exit"]:
                        active_strategy = "S2"
                        combo_df.at[combo_df.index[i], "Strategy"] = "S2"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = s2_df.iloc[i]["Portfolio_Value"]
                    else:
                        # check s1
                        if s1_strat not in ["No Investment", "Exit"]:
                            active_strategy = "S1"
                            combo_df.at[combo_df.index[i], "Strategy"] = "S1"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = s1_df.iloc[i]["Portfolio_Value"]
                        else:
                            combo_df.at[combo_df.index[i], "Strategy"] = "None"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = prev_val
            else:
                # We have an active strategy => see if that strategy says "No Investment" or "Exit"
                if active_strategy == "S1":
                    if s1_strat in ["No Investment", "Exit"]:
                        # close out => check if S2 invests now
                        active_strategy = None
                        # portfolio_value becomes previous day's value => we do not take s1's new day
                        combo_df.at[combo_df.index[i], "Strategy"] = "None"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = prev_val
                        # then we see if S2 invests => if yes, switch to S2
                        if s2_strat not in ["No Investment", "Exit"]:
                            active_strategy = "S2"
                            combo_df.at[combo_df.index[i], "Strategy"] = "S2"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = s2_df.iloc[i]["Portfolio_Value"]
                    else:
                        # stay in S1
                        combo_df.at[combo_df.index[i], "Strategy"] = "S1"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = s1_df.iloc[i]["Portfolio_Value"]

                else:  # active_strategy == "S2"
                    if s2_strat in ["No Investment", "Exit"]:
                        # close out => check if S1 invests
                        active_strategy = None
                        combo_df.at[combo_df.index[i], "Strategy"] = "None"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = prev_val
                        # see if S1 invests
                        if s1_strat not in ["No Investment", "Exit"]:
                            active_strategy = "S1"
                            combo_df.at[combo_df.index[i], "Strategy"] = "S1"
                            combo_df.at[combo_df.index[i], "Portfolio_Value"] = s1_df.iloc[i]["Portfolio_Value"]
                    else:
                        # stay in S2
                        combo_df.at[combo_df.index[i], "Strategy"] = "S2"
                        combo_df.at[combo_df.index[i], "Portfolio_Value"] = s2_df.iloc[i]["Portfolio_Value"]

        else:
            # mode == "Continue running"
            # If S1 invests => portion allocated. If S2 invests => portion allocated.
            # 50% each if both invest, else 100% if only one invests.
            invests_s1 = s1_strat not in ["No Investment", "Exit"]
            invests_s2 = s2_strat not in ["No Investment", "Exit"]
            if invests_s1 and invests_s2:
                # half capital in each
                val_s1 = s1_df.iloc[i]["Portfolio_Value"]
                val_s2 = s2_df.iloc[i]["Portfolio_Value"]
                # But each strategy was originally run with initial_investment. We can see how it grows from day 0.
                # We'll approximate the ratio of today's value to day 0 => apply half capital
                ratio_s1 = val_s1 / initial_investment
                ratio_s2 = val_s2 / initial_investment
                new_val = (0.5 * initial_investment * ratio_s1) + (0.5 * initial_investment * ratio_s2)
                # But we need to scale from prev_val? 
                # For simplicity, we can assume the portfolio_value is: 0.5*s1_value + 0.5*s2_value each day
                new_val = 0.5 * s1_df.iloc[i]["Portfolio_Value"] + 0.5 * s2_df.iloc[i]["Portfolio_Value"]
                combo_df.at[combo_df.index[i], "Strategy"] = "S1+S2"
                combo_df.at[combo_df.index[i], "Portfolio_Value"] = new_val
            elif invests_s1:
                combo_df.at[combo_df.index[i], "Strategy"] = "S1"
                combo_df.at[combo_df.index[i], "Portfolio_Value"] = s1_df.iloc[i]["Portfolio_Value"]
            elif invests_s2:
                combo_df.at[combo_df.index[i], "Strategy"] = "S2"
                combo_df.at[combo_df.index[i], "Portfolio_Value"] = s2_df.iloc[i]["Portfolio_Value"]
            else:
                combo_df.at[combo_df.index[i], "Strategy"] = "None"
                combo_df.at[combo_df.index[i], "Portfolio_Value"] = prev_val

    combo_df["Portfolio_Return"] = combo_df["Portfolio_Value"].pct_change()
    return combo_df

# ==============================================
# Rolling Correlation + Other Common Functions
# ==============================================
def compute_rolling_correlations(df, window):
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

def calculate_performance_metrics(portfolio_df):
    df = portfolio_df.dropna(subset=["Portfolio_Value"]).copy()
    if len(df) < 2:
        return {}

    start_val = df.iloc[0]["Portfolio_Value"]
    end_val = df.iloc[-1]["Portfolio_Value"]
    total_return = (end_val / start_val - 1) * 100

    num_days = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days
    years = num_days / 365 if num_days > 0 else 1
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else np.nan

    ann_vol = df["Portfolio_Return"].std() * np.sqrt(252) * 100
    risk_free_rate = 0.02
    sharpe = (cagr / 100 - risk_free_rate) / (ann_vol / 100) if ann_vol != 0 else np.nan

    rolling_max = df["Portfolio_Value"].cummax()
    drawdown = df["Portfolio_Value"] / rolling_max - 1
    max_dd = drawdown.min() * 100

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Ann. Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Calmar Ratio": calmar,
    }

def plot_portfolio_value(df, asset_label="Portfolio Value", key_prefix=""):
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

def plot_strategy_distribution(df, key_prefix=""):
    strat_counts = df["Strategy"].value_counts()
    fig = px.bar(
        x=strat_counts.index,
        y=strat_counts.values,
        labels={"x": "Strategy", "y": "Number of Days"},
        title="Strategy Distribution Over Time",
    )
    return fig

def plot_drawdown(df, key_prefix=""):
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
# Global placeholders for final data (for export)
# ==============================================
prices_and_stats_df = None
ymax_df_final = None
ymag_df_final = None
perf_df_ymax_final = None
perf_df_ymag_final = None

# ==============================================
# Sidebar Navigation
# ==============================================
page = st.sidebar.radio("Navigation", ["Backtester", "Strategy Overview", "About"])

# ==============================================
# BACKTESTER PAGE
# ==============================================
if page == "Backtester":
    st.title("YMAX YMAG Backtester")

    st.header("Strategy Summaries")
    st.markdown("""
**Strategy 1:** Uses VIX/VVIX thresholds and correlation with market volatility 
to determine long positions in YMAX/YMAG—with hedging via QQQ when volatility is high.

**Strategy 2:** Enters positions only when VIX and VVIX are within a narrow “safe” range, 
exiting if conditions stray and re-entering once stability returns.

**Combination (Both Strategies):** Allows you to prioritize one strategy, 
with two modes of switching or running them in parallel.
""")

    col_sel1, col_sel2 = st.columns([1, 1])

    with col_sel1:
        st.subheader("Asset Selection")
        asset_choice = st.radio(
            "Select the asset(s) to backtest:",
            options=["YMAX", "YMAG", "Both"],
            index=2
        )

    with col_sel2:
        st.subheader("Strategy Selection")
        strategy_choice = st.radio(
            "Select strategy to backtest:",
            options=["Strategy 1 (Investment Rules)",
                     "Strategy 2 (Investment Rules)",
                     "Both Strategies"]
        )
        primary_strategy = None
        combination_mode = None
        if strategy_choice == "Both Strategies":
            st.markdown("**Specify Strategy Priority**")
            primary_strategy = st.radio(
                "Choose the primary strategy:",
                options=["Strategy 1", "Strategy 2"]
            )
            st.markdown("**Combination Mode**")
            combination_mode = st.radio(
                "Select the combination mode:",
                options=["Close by itself", "Continue running"]
            )

    st.markdown("---")
    st.subheader("Parameters")
    corr_window = st.slider("Select correlation window (days):", min_value=1, max_value=30, value=14)

    run_backtest = st.button("Run Backtest for Selected Strategy(ies)")

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
        ps_df = compute_rolling_correlations(all_assets, corr_window)
        ps_df.reset_index(inplace=True)
        prices_and_stats_df = ps_df.copy()

        # We'll store final data in these
        ymax_df_final = None
        ymag_df_final = None
        perf_df_ymax_final = None
        perf_df_ymag_final = None

        # Decide which strategy function to run
        def process_asset(asset_name, strategy):
            """
            Helper to run the specified strategy for the given asset,
            produce charts, and store final DataFrames.
            """
            if strategy == "Strategy 1":
                res = backtest_strategy_1(ps_df, asset=asset_name)
            elif strategy == "Strategy 2":
                res = backtest_strategy_2(ps_df, asset=asset_name)
            else:
                res = None
            return res

        # If user chooses "Both Strategies" => combination
        if strategy_choice == "Both Strategies":
            # If user picks "Both" assets => we'll do combination for each
            if asset_choice == "Both":
                st.markdown("## Combination Backtest - YMAX")
                combo_ymax = backtest_combined_strategies(
                    ps_df, asset="YMAX", initial_investment=10_000,
                    primary=primary_strategy, mode=combination_mode
                )
                col_ymax1, col_ymax2 = st.columns(2)
                with col_ymax1:
                    fig_val_ymax = plot_portfolio_value(combo_ymax, asset_label="Combo (YMAX)", key_prefix="combo_val_ymax")
                    st.plotly_chart(fig_val_ymax, use_container_width=True, key="combo_val_ymax")
                with col_ymax2:
                    fig_strat_ymax = plot_strategy_distribution(combo_ymax, key_prefix="combo_strat_ymax")
                    st.plotly_chart(fig_strat_ymax, use_container_width=True, key="combo_strat_ymax")

                fig_dd_ymax = plot_drawdown(combo_ymax, key_prefix="combo_dd_ymax")
                st.plotly_chart(fig_dd_ymax, use_container_width=True, key="combo_dd_ymax")

                ymax_metrics = calculate_performance_metrics(combo_ymax)
                if ymax_metrics:
                    df_ymax_perf = pd.DataFrame([ymax_metrics], index=["YMAX Combo"]).round(2)
                    st.dataframe(df_ymax_perf)
                    perf_df_ymax_final = df_ymax_perf
                else:
                    st.info("Not enough data points for YMAX combo metrics.")

                # Convert returns to % and store
                combo_ymax["Portfolio_Return"] = (combo_ymax["Portfolio_Return"] * 100).round(2)
                combo_ymax.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                ymax_df_final = combo_ymax.copy()

                st.markdown("## Combination Backtest - YMAG")
                combo_ymag = backtest_combined_strategies(
                    ps_df, asset="YMAG", initial_investment=10_000,
                    primary=primary_strategy, mode=combination_mode
                )
                col_ymag1, col_ymag2 = st.columns(2)
                with col_ymag1:
                    fig_val_ymag = plot_portfolio_value(combo_ymag, asset_label="Combo (YMAG)", key_prefix="combo_val_ymag")
                    st.plotly_chart(fig_val_ymag, use_container_width=True, key="combo_val_ymag")
                with col_ymag2:
                    fig_strat_ymag = plot_strategy_distribution(combo_ymag, key_prefix="combo_strat_ymag")
                    st.plotly_chart(fig_strat_ymag, use_container_width=True, key="combo_strat_ymag")

                fig_dd_ymag = plot_drawdown(combo_ymag, key_prefix="combo_dd_ymag")
                st.plotly_chart(fig_dd_ymag, use_container_width=True, key="combo_dd_ymag")

                ymag_metrics = calculate_performance_metrics(combo_ymag)
                if ymag_metrics:
                    df_ymag_perf = pd.DataFrame([ymag_metrics], index=["YMAG Combo"]).round(2)
                    st.dataframe(df_ymag_perf)
                    perf_df_ymag_final = df_ymag_perf
                else:
                    st.info("Not enough data points for YMAG combo metrics.")

                # Convert returns to % and store
                combo_ymag["Portfolio_Return"] = (combo_ymag["Portfolio_Return"] * 100).round(2)
                combo_ymag.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                ymag_df_final = combo_ymag.copy()

            else:
                # Single asset combination
                st.markdown(f"## Combination Backtest - {asset_choice}")
                combo_res = backtest_combined_strategies(
                    ps_df, asset=asset_choice, initial_investment=10_000,
                    primary=primary_strategy, mode=combination_mode
                )
                col1, col2 = st.columns(2)
                with col1:
                    fig_val = plot_portfolio_value(combo_res, asset_label=f"Combo ({asset_choice})", key_prefix="combo_val_single")
                    st.plotly_chart(fig_val, use_container_width=True, key="combo_val_single")
                with col2:
                    fig_strat = plot_strategy_distribution(combo_res, key_prefix="combo_strat_single")
                    st.plotly_chart(fig_strat, use_container_width=True, key="combo_strat_single")

                fig_dd = plot_drawdown(combo_res, key_prefix="combo_dd_single")
                st.plotly_chart(fig_dd, use_container_width=True, key="combo_dd_single")

                metrics_res = calculate_performance_metrics(combo_res)
                if metrics_res:
                    perf_df = pd.DataFrame([metrics_res], index=[f"{asset_choice} Combo"]).round(2)
                    st.dataframe(perf_df)
                    if asset_choice == "YMAX":
                        perf_df_ymax_final = perf_df
                    else:
                        perf_df_ymag_final = perf_df
                else:
                    st.info("Not enough data points for combo metrics.")

                # Convert returns to % and store
                combo_res["Portfolio_Return"] = (combo_res["Portfolio_Return"] * 100).round(2)
                combo_res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                if asset_choice == "YMAX":
                    ymax_df_final = combo_res.copy()
                else:
                    ymag_df_final = combo_res.copy()

        else:
            # Strategy 1 or Strategy 2 alone
            if strategy_choice == "Strategy 1 (Investment Rules)":
                chosen_strat = "Strategy 1"
            else:
                chosen_strat = "Strategy 2"

            # If "Both" assets => run the chosen strategy on each
            if asset_choice == "Both":
                st.markdown(f"## {chosen_strat} Backtest - YMAX")
                ymax_res = process_asset("YMAX", chosen_strat)
                col_ymax1, col_ymax2 = st.columns(2)
                with col_ymax1:
                    fig_val_ymax = plot_portfolio_value(ymax_res, asset_label=f"{chosen_strat} (YMAX)", key_prefix="val_ymax")
                    st.plotly_chart(fig_val_ymax, use_container_width=True, key="val_ymax")
                with col_ymax2:
                    fig_strat_ymax = plot_strategy_distribution(ymax_res, key_prefix="strat_ymax")
                    st.plotly_chart(fig_strat_ymax, use_container_width=True, key="strat_ymax")

                fig_dd_ymax = plot_drawdown(ymax_res, key_prefix="dd_ymax")
                st.plotly_chart(fig_dd_ymax, use_container_width=True, key="dd_ymax")

                ymax_metrics = calculate_performance_metrics(ymax_res)
                if ymax_metrics:
                    df_ymax_perf = pd.DataFrame([ymax_metrics], index=["YMAX Strategy"]).round(2)
                    st.dataframe(df_ymax_perf)
                    perf_df_ymax_final = df_ymax_perf
                else:
                    st.info("Not enough data points for YMAX metrics.")

                # Convert returns to % and store
                ymax_res["Portfolio_Return"] = (ymax_res["Portfolio_Return"] * 100).round(2)
                ymax_res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                ymax_df_final = ymax_res.copy()

                st.markdown(f"## {chosen_strat} Backtest - YMAG")
                ymag_res = process_asset("YMAG", chosen_strat)
                col_ymag1, col_ymag2 = st.columns(2)
                with col_ymag1:
                    fig_val_ymag = plot_portfolio_value(ymag_res, asset_label=f"{chosen_strat} (YMAG)", key_prefix="val_ymag")
                    st.plotly_chart(fig_val_ymag, use_container_width=True, key="val_ymag")
                with col_ymag2:
                    fig_strat_ymag = plot_strategy_distribution(ymag_res, key_prefix="strat_ymag")
                    st.plotly_chart(fig_strat_ymag, use_container_width=True, key="strat_ymag")

                fig_dd_ymag = plot_drawdown(ymag_res, key_prefix="dd_ymag")
                st.plotly_chart(fig_dd_ymag, use_container_width=True, key="dd_ymag")

                ymag_metrics = calculate_performance_metrics(ymag_res)
                if ymag_metrics:
                    df_ymag_perf = pd.DataFrame([ymag_metrics], index=["YMAG Strategy"]).round(2)
                    st.dataframe(df_ymag_perf)
                    perf_df_ymag_final = df_ymag_perf
                else:
                    st.info("Not enough data points for YMAG metrics.")

                # Convert returns to % and store
                ymag_res["Portfolio_Return"] = (ymag_res["Portfolio_Return"] * 100).round(2)
                ymag_res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                ymag_df_final = ymag_res.copy()

            else:
                # Single asset
                st.markdown(f"## {chosen_strat} Backtest - {asset_choice}")
                res = process_asset(asset_choice, chosen_strat)
                col1, col2 = st.columns(2)
                with col1:
                    fig_val = plot_portfolio_value(res, asset_label=f"{chosen_strat} ({asset_choice})", key_prefix="val_single")
                    st.plotly_chart(fig_val, use_container_width=True, key="val_single")
                with col2:
                    fig_strat = plot_strategy_distribution(res, key_prefix="strat_single")
                    st.plotly_chart(fig_strat, use_container_width=True, key="strat_single")

                fig_dd = plot_drawdown(res, key_prefix="dd_single")
                st.plotly_chart(fig_dd, use_container_width=True, key="dd_single")

                metrics_res = calculate_performance_metrics(res)
                if metrics_res:
                    perf_df = pd.DataFrame([metrics_res], index=[f"{asset_choice} Strategy"]).round(2)
                    st.dataframe(perf_df)
                    if asset_choice == "YMAX":
                        perf_df_ymax_final = perf_df
                    else:
                        perf_df_ymag_final = perf_df
                else:
                    st.info("Not enough data points for metrics.")

                # Convert returns to % and store
                res["Portfolio_Return"] = (res["Portfolio_Return"] * 100).round(2)
                res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
                if asset_choice == "YMAX":
                    ymax_df_final = res.copy()
                else:
                    ymag_df_final = res.copy()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # EXPORT BUTTON
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        st.markdown("---")
        if (ymax_df_final is not None) or (ymag_df_final is not None):
            export_button = st.button("Export Results to Excel")
            if export_button:
                with pd.ExcelWriter("Prices_and_stats_df.xlsx", engine="xlsxwriter") as writer:
                    # 1) Description sheet
                    export_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    description_data = {
                        "Parameter": [
                            "Strategy Selected",
                            "Correlation Window",
                            "Assets Chosen",
                            "Primary Strategy (if combo)",
                            "Combination Mode (if combo)",
                            "Export Date",
                        ],
                        "Value": [
                            strategy_choice,
                            corr_window,
                            asset_choice,
                            primary_strategy if primary_strategy else "",
                            combination_mode if combination_mode else "",
                            export_date,
                        ],
                    }
                    desc_df = pd.DataFrame(description_data)
                    desc_df.to_excel(writer, sheet_name="Description", index=False)

                    # 2) Prices_and_stats_df
                    prices_and_stats_df.to_excel(writer, sheet_name="Prices_and_stats_df", index=False)

                    # 3) Ymax Trading Results
                    if ymax_df_final is not None:
                        ymax_df_final.to_excel(writer, sheet_name="Ymax Trading Results", index=False)

                    # 4) YMAG Trading Results
                    if ymag_df_final is not None:
                        ymag_df_final.to_excel(writer, sheet_name="YMAG Trading Results", index=False)

                    # 5) YMAX Performance
                    if perf_df_ymax_final is not None:
                        perf_df_ymax_final.to_excel(writer, sheet_name="YMAX Performance", index=True)

                    # 6) YMAG Performance
                    if perf_df_ymag_final is not None:
                        perf_df_ymag_final.to_excel(writer, sheet_name="YMAG Performance", index=True)

                st.success("✅ Performance DataFrames successfully saved to 'Prices_and_stats_df.xlsx'")

    else:
        st.info("Click 'Run Backtest for Selected Strategy(ies)' to see results.")

# ==============================================
# STRATEGY OVERVIEW PAGE
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
# ABOUT PAGE
# ==============================================
elif page == "About":
    st.title("About")
    st.write("This page is under construction.")
