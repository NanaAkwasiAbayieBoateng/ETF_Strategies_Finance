import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="ETF Momentum Quality", layout="wide")

# --- Logic: Momentum Calculation ---
def get_quality_momentum(series):
    """Calculates Slope * R^2 for the last 90 days."""
    if len(series) < 90: 
        return 0
    # Log transform for exponential growth slope
    y = np.log(series.tail(90).values)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    # Annualized slope multiplied by consistency (R-squared)
    return (slope * 252) * (r_value**2)

# --- Logic: Drawdown Calculation ---
def calculate_drawdown(series):
    """Calculates the daily drawdown percentage from the peak."""
    # Running maximum (high-water mark)
    peak = series.cummax()
    # Percentage drop from peak
    drawdown = (series - peak) / peak
    return drawdown

# --- UI: Sidebar ---
st.sidebar.header("Strategy Parameters")
selected_tickers = st.sidebar.multiselect(
    "ETFs to Rank",
    options=["QQQ", "SMH", "VGT", "XLK", "URA", "SCHD", "ARKK"],
    default=["QQQ", "SMH", "VGT"]
)

backtest_years = st.sidebar.slider("Backtest Years", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.write("**Metric:** $Slope \\times R^2$ (90-day)")

# --- Main Logic ---
st.title("ETF Momentum Quality Dashboard")

if not selected_tickers:
    st.info("Please select at least one ETF in the sidebar to begin.")
else:
    # Always include SPY for comparison
    all_tickers = list(set(selected_tickers + ['SPY']))
    
    # Download data
    with st.spinner("Fetching market data..."):
        df = yf.download(all_tickers, period=f"{backtest_years}y", auto_adjust=True)['Close']
    
    # Calculate scores for the selected ETFs
    scores = {t: get_quality_momentum(df[t]) for t in selected_tickers}
    
    # Normalize price to 100 for comparison
    norm_df = (df / df.iloc[0]) * 100
    
    # --- Visualization 1: Relative Performance ---
    fig_perf = go.Figure()
    
    for t in all_tickers:
        is_spy = (t == 'SPY')
        line_style = dict(width=2, dash='dash') if is_spy else dict(width=4)
        
        # Format label with momentum score if applicable
        label = f"{t} (Score: {scores[t]:.2f})" if t in scores else t
        
        fig_perf.add_trace(go.Scatter(
            x=norm_df.index, 
            y=norm_df[t], 
            name=label, 
            line=line_style
        ))

    fig_perf.update_layout(
        template="plotly_white", 
        hovermode="x unified", 
        title="Relative Performance: Growth of $100",
        yaxis_title="Normalized Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- Visualization 2: Drawdown Chart ---
    st.subheader("Risk Analysis: Historical Drawdowns")
    fig_dd = go.Figure()

    for t in all_tickers:
        dd_series = calculate_drawdown(df[t])
        is_spy = (t == 'SPY')
        line_style = dict(width=1, dash='dash') if is_spy else dict(width=1.5)
        
        fig_dd.add_trace(go.Scatter(
            x=dd_series.index,
            y=dd_series * 100,  # Convert to percentage
            name=f"{t} Drawdown",
            fill='tozeroy',     # Shaded area chart
            line=line_style
        ))

    fig_dd.update_layout(
        template="plotly_white",
        title="Peak-to-Trough Declines (%)",
        yaxis_title="Percent Drop from Peak",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Summary Table ---
    st.subheader("Momentum & Risk Ranking")
    
    summary_list = []
    for t in selected_tickers:
        mdd = calculate_drawdown(df[t]).min() * 100
        summary_list.append({
            "Ticker": t,
            "Momentum Score": scores[t],
            "Max Drawdown (%)": mdd
        })
        
    ranking_df = pd.DataFrame(summary_list)
    ranking_df = ranking_df.sort_values(by='Momentum Score', ascending=False)
    
    # Format the table for display
    st.table(ranking_df.style.format({
        "Momentum Score": "{:.2f}",
        "Max Drawdown (%)": "{:.2f}%"
    }))