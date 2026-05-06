"""
app.py — GARCH Volatility Forecasting Dashboard

Three-tab Streamlit application:
  1. Model Fit    — Fit GARCH(1,1) or EGARCH(1,1) to any ticker, view fitted vol
  2. Forecasting  — Multi-step conditional volatility forecasts with VaR
  3. Diagnostics  — Standardised residuals, ACF, Ljung-Box test, model comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from garch_model import (
    fit_garch, fit_egarch,
    rolling_volatility, ewma_volatility,
    ljung_box_test, var_forecast,
)
from market_data import get_returns, get_stock_info, realized_volatility

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GARCH Volatility Forecaster",
    page_icon="📈",
    layout="wide",
)

_dark = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "ticker": "SPY",
    "period": "2y",
    "model": "GARCH(1,1)",
    "result": None,
    "returns": None,
    "dates": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GARCH Volatility")
    st.markdown("---")

    ticker = st.text_input("Ticker", value=st.session_state.ticker).upper().strip()
    period = st.selectbox("History", ["1y", "2y", "3y", "5y"], index=1)
    model_choice = st.radio("Model", ["GARCH(1,1)", "EGARCH(1,1)"])
    lam = st.slider("EWMA λ", 0.80, 0.99, 0.94, 0.01)

    fit_btn = st.button("Fit Model", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**VaR Parameters**")
    confidence = st.slider("Confidence Level", 0.90, 0.999, 0.99, 0.001, format="%.3f")
    horizon = st.slider("Horizon (days)", 1, 30, 1)
    portfolio_value = st.number_input("Portfolio Value ($)", value=1_000_000, step=100_000)

# ── Fetch and fit ─────────────────────────────────────────────────────────────
if fit_btn and ticker:
    with st.spinner(f"Fetching {ticker} data and fitting {model_choice}..."):
        returns_series = get_returns(ticker, period)
        if returns_series.empty:
            st.sidebar.error(f"Could not fetch data for {ticker}")
        else:
            returns_arr = returns_series.values
            if model_choice == "GARCH(1,1)":
                result = fit_garch(returns_arr)
            else:
                result = fit_egarch(returns_arr)

            st.session_state.ticker = ticker
            st.session_state.period = period
            st.session_state.model = model_choice
            st.session_state.result = result
            st.session_state.returns = returns_arr
            st.session_state.dates = returns_series.index

result = st.session_state.result
returns = st.session_state.returns
dates = st.session_state.dates

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fit, tab_fc, tab_diag = st.tabs(["Model Fit", "Forecasting", "Diagnostics"])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — MODEL FIT
# ═══════════════════════════════════════════════════════════════════
with tab_fit:
    st.markdown(f"### {st.session_state.ticker} — {st.session_state.model} Conditional Volatility")

    if result is None:
        st.info("Select a ticker and click **Fit Model** to begin.")
    else:
        info = get_stock_info(st.session_state.ticker)
        st.caption(f"{info['name']} | Sector: {info['sector']}")

        # Parameter table
        c1, c2, c3, c4, c5 = st.columns(5)
        if hasattr(result, "gamma"):
            params = {"ω": result.omega, "α": result.alpha, "γ (leverage)": result.gamma,
                      "β": result.beta, "Persistence |β|": abs(result.beta)}
        else:
            params = {"ω": result.omega, "α": result.alpha, "β": result.beta,
                      "α+β": result.persistence, "Long-run Vol": f"{result.long_run_vol:.1%}"}

        for col, (name, val) in zip([c1, c2, c3, c4, c5], params.items()):
            col.metric(name, f"{val:.4f}" if isinstance(val, float) else val)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Log-Likelihood", f"{result.log_likelihood:.2f}")
        m2.metric("AIC", f"{result.aic:.2f}")
        m3.metric("BIC", f"{result.bic:.2f}")
        m4.metric("Converged", "Yes" if result.converged else "No")

        # Conditional vol vs realised
        cond_vol = np.sqrt(result.fitted_variance * 252)
        roll_vol = rolling_volatility(returns, window=21)
        ewma_vol = ewma_volatility(returns, lam=lam)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=cond_vol, name=f"{result.model_type} Vol",
                                 line=dict(color="#00d4aa", width=1.5)))
        fig.add_trace(go.Scatter(x=dates, y=roll_vol, name="21d Realised Vol",
                                 line=dict(color="#ffd700", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=dates, y=ewma_vol, name=f"EWMA (λ={lam})",
                                 line=dict(color="#ff6b6b", width=1, dash="dash")))
        fig.update_layout(**_dark, title="Annualised Conditional Volatility",
                          yaxis_title="Volatility", yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Return series
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dates, y=returns, name="Log-returns",
                                  line=dict(color="#a0a0a0", width=0.8)))
        fig2.add_trace(go.Scatter(x=dates, y=2 * np.sqrt(result.fitted_variance),
                                  name="+2σ band", line=dict(color="#ff6b6b", width=1),
                                  fill=None))
        fig2.add_trace(go.Scatter(x=dates, y=-2 * np.sqrt(result.fitted_variance),
                                  name="-2σ band", line=dict(color="#ff6b6b", width=1),
                                  fill="tonexty", fillcolor="rgba(255,107,107,0.08)"))
        fig2.update_layout(**_dark, title="Returns with ±2σ GARCH Bands",
                           yaxis_title="Log Return", height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTING
# ═══════════════════════════════════════════════════════════════════
with tab_fc:
    st.markdown("### Conditional Volatility Forecast")

    if result is None:
        st.info("Fit a model first.")
    else:
        fc_steps = st.slider("Forecast Horizon (days)", 1, 60, 20)
        fc_vol = result.forecast_vol(fc_steps)
        fc_var = result.forecast(fc_steps)

        last_date = dates[-1] if dates is not None else pd.Timestamp.today()
        fc_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=fc_steps)

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("1-Day Forward Vol", f"{fc_vol[0]:.2%}")
        fc2.metric(f"{fc_steps}-Day Forward Vol", f"{fc_vol[-1]:.2%}")
        fc3.metric("Long-run Vol", f"{result.long_run_vol:.2%}")

        fig = go.Figure()
        hist_len = min(60, len(dates))
        hist_vol = np.sqrt(result.fitted_variance[-hist_len:] * 252)
        hist_dates = dates[-hist_len:]
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_vol, name="Fitted Vol (history)",
                                 line=dict(color="#00d4aa", width=2)))
        fig.add_trace(go.Scatter(x=fc_dates, y=fc_vol, name="Forecast",
                                 line=dict(color="#ffd700", width=2, dash="dot")))
        fig.add_hline(y=result.long_run_vol, line_dash="dash", line_color="gray",
                      annotation_text="Long-run vol")
        fig.update_layout(**_dark, title="Volatility Forecast",
                          yaxis_title="Annualised Vol", yaxis_tickformat=".1%", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Value-at-Risk Forecast")
        var_info = var_forecast(result, confidence=confidence,
                                horizon=horizon, portfolio_value=portfolio_value)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Daily Vol (sigma)", f"{var_info['daily_vol']:.4f}")
        v2.metric(f"{confidence:.1%} VaR", f"{var_info['var_pct']:.2%}")
        v3.metric(f"{horizon}-day VaR (USD)", f"${var_info['var_dollar']:,.0f}")
        v4.metric("Horizon", f"{horizon} day(s)")

        conf_levels = [0.90, 0.95, 0.99, 0.995, 0.999]
        var_rows = []
        for cl in conf_levels:
            vi = var_forecast(result, confidence=cl, horizon=horizon, portfolio_value=portfolio_value)
            var_rows.append({"Confidence": f"{cl:.1%}", "VaR %": f"{vi['var_pct']:.3%}",
                             "VaR USD": f"${vi['var_dollar']:,.0f}"})
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("### Model Diagnostics")

    if result is None:
        st.info("Fit a model first.")
    else:
        std_res = result.std_residuals

        lb = ljung_box_test(std_res, lags=10)
        lb2 = ljung_box_test(std_res ** 2, lags=10)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("LB Q(10) on z", f"{lb['Q_stat']:.2f}")
        d2.metric("p-value", f"{lb['p_value']:.4f}",
                  delta="ARCH effects remain" if lb['p_value'] < 0.05 else "No ARCH effects",
                  delta_color="inverse")
        d3.metric("LB Q(10) on z2", f"{lb2['Q_stat']:.2f}")
        d4.metric("p-value", f"{lb2['p_value']:.4f}",
                  delta="ARCH effects remain" if lb2['p_value'] < 0.05 else "No ARCH effects",
                  delta_color="inverse")

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Standardised Residuals", "Histogram vs N(0,1)",
                                            "ACF of z", "ACF of z^2"])

        fig.add_trace(go.Scatter(y=std_res, mode="lines",
                                 line=dict(color="#a0a0a0", width=0.8), name="z"), row=1, col=1)
        fig.add_hline(y=2, line_dash="dash", line_color="#ff6b6b", opacity=0.5, row=1, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="#ff6b6b", opacity=0.5, row=1, col=1)

        z_vals = np.linspace(-4, 4, 200)
        fig.add_trace(go.Histogram(x=std_res, nbinsx=50, histnorm="probability density",
                                   marker_color="#00d4aa", opacity=0.6, name="Empirical"), row=1, col=2)
        from scipy.stats import norm as snorm
        fig.add_trace(go.Scatter(x=z_vals, y=snorm.pdf(z_vals),
                                 line=dict(color="#ffd700"), name="N(0,1)"), row=1, col=2)

        max_lag = 20
        acf_z = [np.corrcoef(std_res[:-k], std_res[k:])[0, 1] for k in range(1, max_lag + 1)]
        acf_z2 = [np.corrcoef(std_res[:-k] ** 2, std_res[k:] ** 2)[0, 1] for k in range(1, max_lag + 1)]
        ci = 1.96 / np.sqrt(len(std_res))

        for lag, val in enumerate(acf_z, 1):
            fig.add_trace(go.Bar(x=[lag], y=[val], marker_color="#00d4aa",
                                 showlegend=False), row=2, col=1)
        for lag, val in enumerate(acf_z2, 1):
            fig.add_trace(go.Bar(x=[lag], y=[val], marker_color="#ffd700",
                                 showlegend=False), row=2, col=2)
        for row, col in [(2, 1), (2, 2)]:
            fig.add_hline(y=ci, line_dash="dot", line_color="#ff6b6b", opacity=0.7, row=row, col=col)
            fig.add_hline(y=-ci, line_dash="dot", line_color="#ff6b6b", opacity=0.7, row=row, col=col)

        fig.update_layout(**_dark, height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
