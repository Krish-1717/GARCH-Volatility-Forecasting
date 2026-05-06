"""
garch_model.py — GARCH(1,1) and EGARCH(1,1) volatility models.

Fits conditional volatility to log-return time series via maximum likelihood.
Returns in-sample fitted variance, out-of-sample forecasts, and model
diagnostics (AIC, BIC, Ljung-Box test on standardised residuals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, chi2
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GARCHResult:
    """Fitted GARCH(1,1) model."""
    omega: float
    alpha: float
    beta: float
    mu: float
    log_likelihood: float
    aic: float
    bic: float
    fitted_variance: np.ndarray
    std_residuals: np.ndarray
    returns: np.ndarray
    converged: bool
    model_type: str = "GARCH(1,1)"

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """Annualised long-run (unconditional) volatility."""
        denom = 1.0 - self.persistence
        if denom <= 0:
            return float("nan")
        return float(np.sqrt(self.omega / denom) * np.sqrt(252))

    def forecast(self, steps: int = 10) -> np.ndarray:
        """
        Multi-step variance forecast from the last fitted value.

        Var(h) = omega/(1-alpha-beta) + (alpha+beta)^h * (Var(0) - omega/(1-alpha-beta))
        """
        long_run = self.omega / max(1.0 - self.persistence, 1e-8)
        var0 = self.fitted_variance[-1]
        h = np.arange(1, steps + 1)
        forecasts = long_run + self.persistence ** h * (var0 - long_run)
        return np.maximum(forecasts, 1e-10)

    def forecast_vol(self, steps: int = 10) -> np.ndarray:
        """Annualised volatility forecasts."""
        return np.sqrt(self.forecast(steps) * 252)

@dataclass
class EGARCHResult:
    """Fitted EGARCH(1,1) model."""
    omega: float
    alpha: float
    gamma: float   # asymmetry (leverage) term
    beta: float
    mu: float
    log_likelihood: float
    aic: float
    bic: float
    fitted_log_variance: np.ndarray
    std_residuals: np.ndarray
    returns: np.ndarray
    converged: bool
    model_type: str = "EGARCH(1,1)"

    @property
    def fitted_variance(self) -> np.ndarray:
        return np.exp(self.fitted_log_variance)

    @property
    def long_run_vol(self) -> float:
        """Approximate annualised long-run volatility."""
        long_run_log_var = self.omega / max(1.0 - abs(self.beta), 1e-8)
        return float(np.sqrt(np.exp(long_run_log_var) * 252))

    def forecast(self, steps: int = 10) -> np.ndarray:
        """Multi-step variance forecast (Jensen's inequality approximation)."""
        log_var_last = self.fitted_log_variance[-1]
        long_run_log_var = self.omega / max(1.0 - abs(self.beta), 1e-8)
        h = np.arange(1, steps + 1)
        log_var_h = long_run_log_var + self.beta ** h * (log_var_last - long_run_log_var)
        return np.exp(log_var_h)

    def forecast_vol(self, steps: int = 10) -> np.ndarray:
        return np.sqrt(self.forecast(steps) * 252)


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1) MLE
# ─────────────────────────────────────────────────────────────────────────────

def _garch_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    mu, omega, alpha, beta = params
    T = len(returns)
    eps = returns - mu

    # Variance recursion
    var = np.empty(T)
    var[0] = np.var(returns)
    for t in range(1, T):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        if var[t] <= 0:
            return 1e10

    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + eps ** 2 / var)
    return -ll  # minimise negative log-likelihood

def fit_garch(returns: np.ndarray) -> GARCHResult:
    """
    Fit GARCH(1,1) to a 1-D array of log-returns via MLE.

    Parameters
    ----------
    returns : np.ndarray  (decimal, e.g. 0.01 for 1%)
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    var0 = np.var(returns)

    # Initial parameters: mu=mean, omega=var*(1-0.85), alpha=0.08, beta=0.85
    x0 = [returns.mean(), var0 * 0.05, 0.08, 0.85]
    bounds = [
        (None, None),       # mu
        (1e-8, None),       # omega
        (1e-8, 0.999),      # alpha
        (1e-8, 0.999),      # beta
    ]
    # Constraint: alpha + beta < 1
    constraints = [{"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3]}]

    res = minimize(
        _garch_log_likelihood,
        x0,
        args=(returns,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    mu, omega, alpha, beta = res.x
    eps = returns - mu

    # Fitted variance series
    var = np.empty(T)
    var[0] = np.var(returns)
    for t in range(1, T):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        var[t] = max(var[t], 1e-10)

    ll = -res.fun
    n_params = 4
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(T) - 2 * ll

    return GARCHResult(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        mu=float(mu),
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        fitted_variance=var,
        std_residuals=eps / np.sqrt(var),
        returns=returns,
        converged=res.success,
    )

# ─────────────────────────────────────────────────────────────────────────────
# EGARCH(1,1) MLE
# ─────────────────────────────────────────────────────────────────────────────

def _egarch_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    mu, omega, alpha, gamma, beta = params
    T = len(returns)
    eps = returns - mu

    log_var = np.empty(T)
    log_var[0] = np.log(max(np.var(returns), 1e-10))
    E_abs = np.sqrt(2 / np.pi)  # E[|z|] for standard normal

    for t in range(1, T):
        z = eps[t - 1] / np.exp(0.5 * log_var[t - 1])
        log_var[t] = (
            omega
            + alpha * (abs(z) - E_abs)
            + gamma * z
            + beta * log_var[t - 1]
        )

    var = np.exp(log_var)
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + eps ** 2 / var)
    return -ll


def fit_egarch(returns: np.ndarray) -> EGARCHResult:
    """Fit EGARCH(1,1) to log-returns via MLE."""
    returns = np.asarray(returns, dtype=float)
    T = len(returns)

    x0 = [returns.mean(), -0.1, 0.08, -0.05, 0.90]
    bounds = [
        (None, None),       # mu
        (None, None),       # omega (can be negative in EGARCH)
        (0.0, None),        # alpha
        (None, None),       # gamma (leverage)
        (-0.9999, 0.9999),  # beta
    ]

    res = minimize(
        _egarch_log_likelihood,
        x0,
        args=(returns,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    mu, omega, alpha, gamma, beta = res.x
    eps = returns - mu
    E_abs = np.sqrt(2 / np.pi)

    log_var = np.empty(T)
    log_var[0] = np.log(max(np.var(returns), 1e-10))
    for t in range(1, T):
        z = eps[t - 1] / np.exp(0.5 * log_var[t - 1])
        log_var[t] = omega + alpha * (abs(z) - E_abs) + gamma * z + beta * log_var[t - 1]

    ll = -res.fun
    n_params = 5
    aic = 2 * n_params - 2 * ll
    bic = n_params * np.log(T) - 2 * ll

    return EGARCHResult(
        omega=float(omega),
        alpha=float(alpha),
        gamma=float(gamma),
        beta=float(beta),
        mu=float(mu),
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        fitted_log_variance=log_var,
        std_residuals=eps / np.sqrt(np.exp(log_var)),
        returns=returns,
        converged=res.success,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> dict:
    """Ljung-Box Q-test on squared standardised residuals (ARCH effects)."""
    sq = residuals ** 2
    T = len(sq)
    mean_sq = sq.mean()
    acf = np.array([
        np.corrcoef(sq[:-k], sq[k:])[0, 1] if k > 0 else 1.0
        for k in range(lags + 1)
    ])
    Q = T * (T + 2) * sum(acf[k] ** 2 / (T - k) for k in range(1, lags + 1))
    p_value = 1 - chi2.cdf(Q, df=lags)
    return {"Q_stat": float(Q), "p_value": float(p_value), "lags": lags}


def rolling_volatility(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling historical volatility (annualised)."""
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    vol = np.full(T, np.nan)
    for i in range(window, T + 1):
        vol[i - 1] = np.std(returns[i - window:i], ddof=1) * np.sqrt(252)
    return vol


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    JP Morgan RiskMetrics EWMA volatility (annualised).
    lam = 0.94 is the standard daily decay factor.
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    var = np.empty(T)
    var[0] = returns[0] ** 2
    for t in range(1, T):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return np.sqrt(var * 252)


def var_forecast(result,
                 confidence: float = 0.99,
                 horizon: int = 1,
                 portfolio_value: float = 1_000_000) -> dict:
    """
    Value-at-Risk forecast using GARCH conditional variance.

    Assumes normally distributed standardised residuals.
    """
    sigma_h = float(np.sqrt(result.forecast(horizon)[-1]))
    z = norm.ppf(1 - confidence)
    var_pct = -z * sigma_h
    var_dollar = var_pct * portfolio_value
    return {
        "horizon_days": horizon,
        "confidence": confidence,
        "daily_vol": float(sigma_h),
        "var_pct": float(var_pct),
        "var_dollar": float(var_dollar),
    }
