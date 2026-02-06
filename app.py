import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="ETF Momentum Quality", layout="wide")

# --- Metadata: Expanded ETF List ---
etf_metadata = {
    # --- BROAD MARKET & CORE ---
    'SPY': ['S&P 500 ETF Trust', 'State Street'],
    'VOO': ['S&P 500 ETF', 'Vanguard'],
    'QQQ': ['Nasdaq 100 Trust', 'Invesco'],
    'IWM': ['Russell 2000 ETF', 'BlackRock'],
    'DIA': ['Dow Jones Industrial Average', 'State Street'],
    'VT':  ['Total World Stock ETF', 'Vanguard'],
    # --- TECHNOLOGY & SEMICONDUCTORS ---
    'XLK': ['Technology Select Sector SPDR', 'State Street'],
    'VGT': ['Information Technology ETF', 'Vanguard'],
    'SMH': ['Semiconductor ETF', 'VanEck'],
    'SOXX':['PHLX Semiconductor ETF', 'BlackRock'],
    'IGV': ['Expanded Tech-Software ETF', 'BlackRock'],
    'ARKK':['Innovation ETF', 'ARK Invest'],
    # --- DEFENSIVE & INCOME SECTORS ---
    'XLV': ['Health Care Select Sector SPDR', 'State Street'],
    'XLP': ['Consumer Staples Select Sector SPDR', 'State Street'],
    'XLU': ['Utilities Select Sector SPDR', 'State Street'],
    'SCHD':['Schwab US Dividend Equity ETF', 'Schwab'],
    'VIG': ['Dividend Appreciation ETF', 'Vanguard'],
    'JEPI':['JPMorgan Equity Premium Income', 'JPMorgan'],
    # --- CYCLICAL & VALUE SECTORS ---
    'XLF': ['Financial Select Sector SPDR', 'State Street'],
    'XLY': ['Consumer Discretionary SPDR', 'State Street'],
    'XLI': ['Industrial Select Sector SPDR', 'State Street'],
    'XLE': ['Energy Select Sector SPDR', 'State Street'],
    'XLB': ['Materials Select Sector SPDR', 'State Street'],
    'XLRE':['Real Estate Select Sector SPDR', 'State Street'],
    'KRE': ['Regional Banking ETF', 'State Street'],
    # --- INTERNATIONAL & EMERGING MARKETS ---
    'VEA': ['Developed Markets ETF', 'Vanguard'],
    'EEM': ['Emerging Markets ETF', 'BlackRock'],
    'VXUS':['Total International Stock ETF', 'Vanguard'],
    'VGK': ['FTSE Europe ETF', 'Vanguard'],
    'EWJ': ['MSCI Japan ETF', 'BlackRock'],
    'MCHI':['MSCI China ETF', 'BlackRock'],
    # --- COMMODITIES & ALTERNATIVES ---
    'GLD': ['Gold Shares', 'State Street'],
    'IAU': ['Gold Trust', 'BlackRock'],
    'SLV': ['Silver Trust', 'BlackRock'],
    'DBC': ['Commodity Index Tracking Fund', 'Invesco'],
    'URA': ['Uranium ETF', 'Global X'],
    'BITO':['Bitcoin Strategy ETF', 'ProShares'],
    # --- BONDS & FIXED INCOME ---
    'BND': ['Total Bond Market ETF', 'Vanguard'],
    'AGG': ['U.S. Aggregate Bond ETF', 'BlackRock'],
    'TLT': ['20+ Year Treasury Bond ETF', 'BlackRock'],
    'IEF': ['7-10 Year Treasury Bond ETF', 'BlackRock'],
    'SHY': ['1-3 Year Treasury Bond ETF', 'BlackRock'],
    'LQD': ['Investment Grade Corp Bond', 'BlackRock'],
    'HYG': ['High Yield Corporate Bond ETF', 'BlackRock'],
    'TIP': ['Treasury Inflation-Protected', 'BlackRock'],
    # --- THEMATIC & FACTOR ---
    'MTUM':['MSCI USA Momentum Factor', 'BlackRock'],
    'QUAL':['MSCI USA Quality Factor', 'BlackRock'],
    'VLUE':['MSCI USA Value Factor', 'BlackRock'],
    'USMV':['MSCI USA Min Volatility', 'BlackRock'],
    'TAN': ['Solar ETF', 'Invesco'],
    'ICLN':['Global Clean Energy ETF', 'BlackRock'],
    'MJ':  ['Alternative Harvest ETF', 'ETFMG']
}

# --- Logic: Momentum Calculation ---
def get_quality_momentum(series):
    """Calculates Slope * R^2 for the last 90 days."""
    if len(series) < 90: return 0
    # Log transform for exponential growth slope
    y = np.log(series.tail(90).values)
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    # Annualized slope multiplied by consistency (R-squared)
    return (slope * 252) * (r_value**2)

# --- Logic: Drawdown Calculation ---
def calculate_drawdown(series):
    """Calculates the daily drawdown percentage from the peak."""
    peak = series.cummax()
    return (series - peak) / peak

# --- UI: Sidebar ---
st.sidebar.header("Strategy Parameters")

# Create display labels: "Ticker - Name"
ticker_options = [f"{k} - {v[0]}" for k, v in etf_metadata.items()]

selected_display = st.sidebar.multiselect(
    "ETFs to Rank",
    options=ticker_options,
    default=["QQQ - Nasdaq 100 Trust", "SMH - Semiconductor ETF", "VGT - Information Technology ETF"]
)

# Extract just the ticker for internal logic
selected_tickers = [item.split(" - ")[0] for item in selected_display]

backtest_years = st.sidebar.slider("Backtest Years", 1, 10, 5)

# --- Main Logic ---
st.title("ETF Momentum Quality Dashboard")

if not selected_tickers:
    st.info("Please select at least one ETF in the sidebar to begin.")
else:
    all_tickers = list(set(selected_tickers + ['SPY']))
    
    with st.spinner("Fetching market data..."):
        # Fix: Ensure download returns only the Close price correctly
        df = yf.download(all_tickers, period=f"{backtest_years}y")['Close']
    
    scores = {t: get_quality_momentum(df[t]) for t in selected_tickers}
    norm_df = (df / df.iloc[0]) * 100
    
    # --- Visualization 1: Relative Performance ---
    fig_perf = go.Figure()
    for t in all_tickers:
        is_spy = (t == 'SPY')
        line_style = dict(width=2, dash='dash') if is_spy else dict(width=4)
        
        # Use full name from metadata if available for the legend
        full_name = etf_metadata.get(t, [""])[0]
        label = f"{t} ({full_name})" if not is_spy else "SPY (S&P 500)"
        
        fig_perf.add_trace(go.Scatter(x=norm_df.index, y=norm_df[t], name=label, line=line_style))

    fig_perf.update_layout(template="plotly_white", title="Growth of $100", yaxis_title="Normalized Price")
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- Visualization 2: Drawdown Chart ---
    st.subheader("Risk Analysis: Historical Drawdowns")
    fig_dd = go.Figure()
    for t in all_tickers:
        dd_series = calculate_drawdown(df[t])
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series * 100, name=f"{t} Drawdown", fill='tozeroy'))

    fig_dd.update_layout(template="plotly_white", title="Peak-to-Trough Declines (%)", yaxis_title="Percent Drop")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Summary Table ---
    st.subheader("Momentum & Risk Ranking")
    summary_list = []
    for t in selected_tickers:
        mdd = calculate_drawdown(df[t]).min() * 100
        summary_list.append({
            "Ticker": t,
            "Name": etf_metadata[t][0],
            "Issuer": etf_metadata[t][1],
            "Momentum Score": scores[t],
            "Max Drawdown (%)": mdd
        })
        
    ranking_df = pd.DataFrame(summary_list).sort_values(by='Momentum Score', ascending=False)
    st.table(ranking_df.style.format({"Momentum Score": "{:.2f}", "Max Drawdown (%)": "{:.2f}%"}))
    
    # --- Visualization 4: Portfolio Optimization (Min Volatility) ---
    st.subheader("Optimal Allocation: Minimum Volatility")

    # FIX: Define returns_df before using it in the covariance calculation
    returns_df = df[selected_tickers].pct_change().dropna()
    
    if not returns_df.empty:
        # Annualized covariance matrix
        cov_matrix = returns_df.cov() * 252 
        num_assets = len(selected_tickers)

        # Optimization function: Minimize Portfolio Variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Constraints: Weights must sum to 100%
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: No short selling (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        # Initial guess: Equal weighting
        init_guess = [1. / num_assets] * num_assets

        from scipy.optimize import minimize
        opt_result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if opt_result.success:
            opt_weights = opt_result.x
            
            # Create a Bar Chart for Allocations
            fig_opt = go.Figure(data=[go.Bar(
                x=selected_tickers,
                y=opt_weights * 100,
                text=np.round(opt_weights * 100, 1),
                textposition='auto',
                marker_color='teal'
            )])

            fig_opt.update_layout(
                title="Suggested Portfolio Weights (%) for Lowest Risk",
                yaxis_title="Allocation (%)",
                template="plotly_white",
                yaxis_range=[0, 105]
            )

            st.plotly_chart(fig_opt, use_container_width=True)
            
            # Comparison Metrics
            portfolio_vol = np.sqrt(opt_result.fun) * 100
            spy_returns = df['SPY'].pct_change().dropna()
            spy_vol = (spy_returns.std() * np.sqrt(252)) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Optimized Portfolio Volatility", f"{portfolio_vol:.2f}%")
            col2.metric("S&P 500 (SPY) Volatility", f"{spy_vol:.2f}%")
            
            st.write(f"**Strategy Note:** By following this allocation, your portfolio's historical volatility would be **{portfolio_vol:.2f}%**, compared to a simple SPY holding of **{spy_vol:.2f}%**.")