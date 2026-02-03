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
    """Calculates Slope * R^2 for the last 90 days[cite: 5]."""
    if len(series) < 90: 
        return 0
    # Log transform for exponential growth slope [cite: 4]
    y = np.log(series.tail(90).values)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    # Annualized slope multiplied by consistency (R-squared) [cite: 5]
    return (slope * 252) * (r_value**2)

# --- UI: Sidebar ---
st.sidebar.header("Strategy Parameters")
selected_tickers = st.sidebar.multiselect(
    "ETFs to Rank",
    options=["QQQ", "SMH", "VGT", "XLK", "URA", "SCHD", "ARKK"],
    default=["QQQ", "SMH", "VGT"] # [cite: 3]
)

backtest_years = st.sidebar.slider("Backtest Years", 1, 10, 5) # [cite: 3]

st.sidebar.markdown("---")
st.sidebar.write("**Metric:** $Slope \\times R^2$ (90-day) [cite: 3]")

# --- Main Logic ---
st.title("ETF Momentum Quality Dashboard")

if not selected_tickers:
    st.info("Please select at least one ETF in the sidebar to begin.")
else:
    # Always include SPY for comparison [cite: 5]
    all_tickers = list(set(selected_tickers + ['SPY']))
    
    # Download data [cite: 4]
    with st.spinner("Fetching market data..."):
        df = yf.download(all_tickers, period=f"{backtest_years}y", auto_adjust=True)['Close']
    
    # Calculate scores for the selected ETFs [cite: 5]
    scores = {t: get_quality_momentum(df[t]) for t in selected_tickers}
    
    # Normalize price to 100 for comparison [cite: 5]
    norm_df = (df / df.iloc[0]) * 100 # [cite: 5]
    
    # --- Visualization: Plotly ---
    fig = go.Figure()
    
    for t in all_tickers:
        is_spy = (t == 'SPY')
        line_style = dict(width=2, dash='dash') if is_spy else dict(width=4)
        
        # Format the label with the momentum score [cite: 5]
        label = f"{t} (Score: {scores[t]:.2f})" if t in scores else t
        
        fig.add_trace(go.Scatter(
            x=norm_df.index, 
            y=norm_df[t], 
            name=label, 
            line=line_style
        ))

    fig.update_layout(
        template="plotly_white", 
        hovermode="x unified", 
        title="Relative Performance: Growth of $100",
        yaxis_title="Normalized Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics Table ---
    st.subheader("Momentum Ranking")
    ranking_df = pd.DataFrame.from_dict(scores, orient='index', columns=['Momentum Score'])
    ranking_df = ranking_df.sort_values(by='Momentum Score', ascending=False)
    st.table(ranking_df)
