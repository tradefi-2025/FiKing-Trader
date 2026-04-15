"""
S3 — Risk Management (Position Sizing)
Stratify Platform · Core Computation Module

Implements three position-sizing methods:
  - fixed_fractional   : Fixed % of account risked per trade
  - kelly              : Kelly Criterion (confidence-weighted, half-Kelly applied)
  - cvar               : CVaR-constrained sizing

Each function returns a RiskResult dataclass.
A METHOD_REGISTRY dict maps user-supplied strings to the correct function.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Mirrors the S3 output schema defined in the Stratify service catalogue."""
    position_size: float               # units / shares / contracts
    notional: float                    # position_size * entry_price
    stop_loss_price: float             # auto-computed or user-supplied
    risk_amount: float                 # max loss in account currency
    sizing_method_used: str
    kelly_f: Optional[float] = None    # raw full-Kelly fraction (if Kelly used)
    cvar_estimate: Optional[float] = None  # tail-loss estimate (if CVaR used)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stop-loss helpers
# ---------------------------------------------------------------------------

def compute_atr_stop(
    entry_price: float,
    atr: float,
    signal_direction: str,
    multiplier: float = 2.0,
) -> float:
    """
    Derive stop-loss price from ATR when no explicit stop is provided.

    Args:
        entry_price:      Executed or expected fill price.
        atr:              Average True Range (pre-computed by S10).
        signal_direction: 'BUY' or 'SELL'.
        multiplier:       ATR multiplier (default 2.0; common range 1.5–2.5).

    Returns:
        Stop-loss price below entry for BUY, above entry for SELL.
    """
    if signal_direction.upper() == "BUY":
        return entry_price - multiplier * atr
    return entry_price + multiplier * atr


# ---------------------------------------------------------------------------
# Method 1 — Fixed Fractional
# ---------------------------------------------------------------------------

def fixed_fractional(
    account_value: float,
    entry_price: float,
    risk_pct: float,
    stop_loss_price: Optional[float] = None,
    atr: Optional[float] = None,
    signal_direction: str = "BUY",
    atr_multiplier: float = 2.0,
    max_concentration: float = 0.20,
) -> RiskResult:
    """
    Fixed-fractional position sizing.

    Position size is derived so that the maximum dollar loss
    (if the stop is hit) equals ``account_value * risk_pct``.

    Formula:
        position_size = (account_value * risk_pct) / |entry_price - stop_loss_price|

    Args:
        account_value:     Total tradeable capital in account currency.
        entry_price:       Expected fill price per unit.
        risk_pct:          Fraction of account to risk (e.g. 0.01 = 1 %).
        stop_loss_price:   Explicit stop level.  If None, derived from ATR.
        atr:               Average True Range; required if stop_loss_price is None.
        signal_direction:  'BUY' or 'SELL'; used for ATR-based stop direction.
        atr_multiplier:    Multiplier applied to ATR when auto-computing stop.
        max_concentration: Max fraction of account in a single position (default 0.20).

    Returns:
        RiskResult with sizing_method_used='FIXED_FRACTIONAL'.

    Raises:
        ValueError: If neither stop_loss_price nor atr is provided,
                    or if risk_pct is not in (0, 1).
    """
    warn_list: list[str] = []

    if not (0 < risk_pct < 1):
        raise ValueError(f"risk_pct must be in (0, 1), got {risk_pct}")

    if stop_loss_price is None:
        if atr is None or atr <= 0:
            raise ValueError("Provide stop_loss_price or a valid atr > 0.")
        stop_loss_price = compute_atr_stop(
            entry_price, atr, signal_direction, atr_multiplier
        )

    stop_distance = abs(entry_price - stop_loss_price)
    if stop_distance == 0:
        raise ValueError("entry_price == stop_loss_price; stop distance is zero.")

    risk_amount = account_value * risk_pct

    # Edge case: all returns positive → risk_pct calibration is unreliable
    if risk_pct == 0:
        warn_list.append("RISK_PCT_ZERO: risk_pct is zero; fallback to 0.01 applied.")
        risk_pct = 0.01
        risk_amount = account_value * risk_pct

    position_size = risk_amount / stop_distance
    notional = position_size * entry_price

    # Concentration guard
    if notional > account_value * max_concentration:
        warn_list.append(
            f"CONCENTRATION_RISK: notional ({notional:.2f}) exceeds "
            f"{max_concentration*100:.0f}% of account ({account_value:.2f}). "
            "Position capped."
        )
        notional = account_value * max_concentration
        position_size = notional / entry_price
        risk_amount = position_size * stop_distance

    return RiskResult(
        position_size=position_size,
        notional=notional,
        stop_loss_price=stop_loss_price,
        risk_amount=risk_amount,
        sizing_method_used="FIXED_FRACTIONAL",
        warnings=warn_list,
    )


# ---------------------------------------------------------------------------
# Method 2 — Kelly Criterion
# ---------------------------------------------------------------------------

def kelly(
    account_value: float,
    entry_price: float,
    confidence: float,
    expected_gain_pct: float,
    expected_loss_pct: float,
    stop_loss_price: Optional[float] = None,
    atr: Optional[float] = None,
    signal_direction: str = "BUY",
    atr_multiplier: float = 2.0,
    kelly_fraction: float = 0.5,
    max_concentration: float = 0.20,
) -> RiskResult:
    """
    Kelly Criterion position sizing with half-Kelly dampening.

    Full-Kelly formula:
        f* = (p * b - (1 - p)) / b

    where:
        p = probability of winning  (maps to ``confidence`` from Signaling)
        b = net odds ratio          = expected_gain_pct / expected_loss_pct

    Half-Kelly (default) applies ``kelly_fraction = 0.5`` to full-Kelly,
    significantly reducing variance without sacrificing much long-run growth.

    If f* <= 0 the signal has no statistical edge; position_size is set to 0
    and a NO_EDGE warning is emitted.

    Args:
        account_value:      Total tradeable capital.
        entry_price:        Expected fill price per unit.
        confidence:         P(win) from Signaling agent (0 < confidence < 1).
        expected_gain_pct:  Expected return if trade wins (e.g. 0.03 = 3 %).
        expected_loss_pct:  Expected loss if trade loses (e.g. 0.015 = 1.5 %).
                            Must be positive.
        stop_loss_price:    Explicit stop; derived from ATR if None.
        atr:                ATR value from S10; used if stop_loss_price is None.
        signal_direction:   'BUY' or 'SELL'.
        atr_multiplier:     ATR multiplier for auto stop.
        kelly_fraction:     Dampening factor (0.5 = half-Kelly; recommended).
        max_concentration:  Max fraction of account per position.

    Returns:
        RiskResult with sizing_method_used='KELLY', kelly_f set to full-Kelly f*.
    """
    warn_list: list[str] = []

    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if expected_loss_pct <= 0:
        raise ValueError("expected_loss_pct must be > 0.")
    if expected_gain_pct <= 0:
        raise ValueError("expected_gain_pct must be > 0.")

    if stop_loss_price is None:
        if atr is None or atr <= 0:
            raise ValueError("Provide stop_loss_price or a valid atr > 0.")
        stop_loss_price = compute_atr_stop(
            entry_price, atr, signal_direction, atr_multiplier
        )

    p = confidence
    b = expected_gain_pct / expected_loss_pct
    full_kelly_f = (p * b - (1 - p)) / b
    applied_f = kelly_fraction * full_kelly_f

    if full_kelly_f <= 0:
        warn_list.append(
            "NO_EDGE: Kelly fraction is non-positive; signal has no statistical "
            "edge. Position set to zero."
        )
        notional = 0.0
        position_size = 0.0
        stop_distance = abs(entry_price - stop_loss_price)
        risk_amount = 0.0
        return RiskResult(
            position_size=position_size,
            notional=notional,
            stop_loss_price=stop_loss_price,
            risk_amount=risk_amount,
            sizing_method_used="KELLY",
            kelly_f=full_kelly_f,
            warnings=warn_list,
        )

    notional = applied_f * account_value
    stop_distance = abs(entry_price - stop_loss_price)
    position_size = notional / entry_price
    risk_amount = position_size * stop_distance

    if notional > account_value * max_concentration:
        warn_list.append(
            f"CONCENTRATION_RISK: Kelly notional ({notional:.2f}) exceeds "
            f"{max_concentration*100:.0f}% cap. Position capped."
        )
        notional = account_value * max_concentration
        position_size = notional / entry_price
        risk_amount = position_size * stop_distance

    return RiskResult(
        position_size=position_size,
        notional=notional,
        stop_loss_price=stop_loss_price,
        risk_amount=risk_amount,
        sizing_method_used="KELLY",
        kelly_f=full_kelly_f,
        warnings=warn_list,
    )


# ---------------------------------------------------------------------------
# Method 3 — CVaR-Constrained Sizing
# ---------------------------------------------------------------------------

def cvar(
    account_value: float,
    entry_price: float,
    historical_returns: np.ndarray,
    max_portfolio_loss_tolerance: float,
    alpha: float = 0.05,
    stop_loss_price: Optional[float] = None,
    atr: Optional[float] = None,
    signal_direction: str = "BUY",
    atr_multiplier: float = 2.0,
    max_concentration: float = 0.20,
) -> RiskResult:
    """
    CVaR (Conditional Value at Risk) constrained position sizing.

    Sizes the position so that the expected loss in the worst ``alpha`` fraction
    of outcomes does not exceed ``max_portfolio_loss_tolerance``.

    Formula:
        CVaR_alpha = -mean( returns[returns <= VaR_alpha] )
        position_size = max_portfolio_loss_tolerance / CVaR_alpha / entry_price

    Args:
        account_value:               Total tradeable capital.
        entry_price:                 Expected fill price.
        historical_returns:          1-D array of past log/simple returns for
                                     the asset (sourced from S10 lookback window).
        max_portfolio_loss_tolerance: Max acceptable loss in account currency
                                     (e.g. account_value * 0.02 = 2 % of capital).
        alpha:                       Tail probability threshold (default 0.05 = 5 %).
        stop_loss_price:             Explicit stop; derived from ATR if None.
        atr:                         ATR from S10.
        signal_direction:            'BUY' or 'SELL'.
        atr_multiplier:              ATR multiplier for auto stop.
        max_concentration:           Max fraction of account per position.

    Returns:
        RiskResult with sizing_method_used='CVAR', cvar_estimate set.

    Warns:
        SMALL_TAIL_SAMPLE_WARNING if the tail has fewer than 5 observations;
        falls back to fixed_fractional in that case.
    """
    warn_list: list[str] = []

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if max_portfolio_loss_tolerance <= 0:
        raise ValueError("max_portfolio_loss_tolerance must be > 0.")

    if stop_loss_price is None:
        if atr is None or atr <= 0:
            raise ValueError("Provide stop_loss_price or a valid atr > 0.")
        stop_loss_price = compute_atr_stop(
            entry_price, atr, signal_direction, atr_multiplier
        )

    returns = np.asarray(historical_returns, dtype=float)
    var_threshold = np.quantile(returns, alpha)
    tail_returns = returns[returns <= var_threshold]

    # Small tail sample guard (from Stratify spec)
    if len(tail_returns) < 5:
        warn_list.append(
            "SMALL_TAIL_SAMPLE_WARNING: CVaR tail has fewer than 5 observations. "
            "Falling back to FIXED_FRACTIONAL with risk_pct=0.01."
        )
        fallback = fixed_fractional(
            account_value=account_value,
            entry_price=entry_price,
            risk_pct=0.01,
            stop_loss_price=stop_loss_price,
            max_concentration=max_concentration,
        )
        fallback.warnings = warn_list + fallback.warnings
        fallback.sizing_method_used = "CVAR→FIXED_FRACTIONAL_FALLBACK"
        return fallback

    cvar_estimate = float(-np.mean(tail_returns))

    if cvar_estimate <= 0:
        warn_list.append(
            "RISK_PCT_ZERO: CVaR estimate is non-positive (all returns positive). "
            "Falling back to FIXED_FRACTIONAL with risk_pct=0.01."
        )
        fallback = fixed_fractional(
            account_value=account_value,
            entry_price=entry_price,
            risk_pct=0.01,
            stop_loss_price=stop_loss_price,
            max_concentration=max_concentration,
        )
        fallback.warnings = warn_list + fallback.warnings
        fallback.sizing_method_used = "CVAR→FIXED_FRACTIONAL_FALLBACK"
        return fallback

    # notional = max tolerated loss / CVaR (as fraction of 1 unit)
    notional = max_portfolio_loss_tolerance / cvar_estimate
    position_size = notional / entry_price
    stop_distance = abs(entry_price - stop_loss_price)
    risk_amount = position_size * stop_distance

    if notional > account_value * max_concentration:
        warn_list.append(
            f"CONCENTRATION_RISK: CVaR notional ({notional:.2f}) exceeds "
            f"{max_concentration*100:.0f}% cap. Position capped."
        )
        notional = account_value * max_concentration
        position_size = notional / entry_price
        risk_amount = position_size * stop_distance

    return RiskResult(
        position_size=position_size,
        notional=notional,
        stop_loss_price=stop_loss_price,
        risk_amount=risk_amount,
        sizing_method_used="CVAR",
        cvar_estimate=cvar_estimate,
        warnings=warn_list,
    )


# ---------------------------------------------------------------------------
# Method Registry — string → function mapper
# ---------------------------------------------------------------------------

METHOD_REGISTRY: dict[str, Callable] = {
    "fixed_fractional": fixed_fractional,
    "kelly":            kelly,
    "cvar":             cvar,
}

VALID_METHODS = list(METHOD_REGISTRY.keys())


def compute_position_size(method: str, **kwargs) -> RiskResult:
    """
    Dispatcher: resolve a user-supplied method string and call the
    corresponding sizing function with the provided keyword arguments.

    Args:
        method:   One of 'fixed_fractional', 'kelly', 'cvar'
                  (case-insensitive).
        **kwargs: All arguments forwarded to the selected sizing function.
                  See individual function signatures for required parameters.

    Returns:
        RiskResult from the selected sizing method.

    Raises:
        ValueError: If ``method`` is not in METHOD_REGISTRY.

    Example:
        >>> result = compute_position_size(
        ...     method="kelly",
        ...     account_value=100_000,
        ...     entry_price=150.0,
        ...     confidence=0.72,
        ...     expected_gain_pct=0.04,
        ...     expected_loss_pct=0.02,
        ...     atr=3.0,
        ...     signal_direction="BUY",
        ... )
        >>> print(result.position_size, result.notional, result.warnings)
    """
    key = method.strip().lower()
    if key not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown sizing method '{method}'. "
            f"Valid options: {VALID_METHODS}"
        )
    fn = METHOD_REGISTRY[key]
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python s3_risk_management.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    ACCOUNT  = 100_000.0
    PRICE    = 150.0
    ATR      = 3.0
    RETURNS  = np.random.normal(0.001, 0.02, 500)  # synthetic for demo only

    tests = [
        dict(
            method="fixed_fractional",
            account_value=ACCOUNT,
            entry_price=PRICE,
            risk_pct=0.01,
            atr=ATR,
            signal_direction="BUY",
        ),
        dict(
            method="kelly",
            account_value=ACCOUNT,
            entry_price=PRICE,
            confidence=0.62,
            expected_gain_pct=0.04,
            expected_loss_pct=0.02,
            atr=ATR,
            signal_direction="BUY",
        ),
        dict(
            method="cvar",
            account_value=ACCOUNT,
            entry_price=PRICE,
            historical_returns=RETURNS,
            max_portfolio_loss_tolerance=ACCOUNT * 0.02,
            alpha=0.05,
            atr=ATR,
            signal_direction="BUY",
        ),
    ]

    for t in tests:
        m = t.pop("method")
        result = compute_position_size(method=m, **t)
        print(f"\n{'='*55}")
        print(f"Method : {result.sizing_method_used}")
        print(f"Size   : {result.position_size:.4f} units")
        print(f"Notional: ${result.notional:,.2f}")
        print(f"Stop   : ${result.stop_loss_price:.2f}")
        print(f"Risk $  : ${result.risk_amount:,.2f}")
        if result.kelly_f is not None:
            print(f"Kelly f*: {result.kelly_f:.4f}")
        if result.cvar_estimate is not None:
            print(f"CVaR    : {result.cvar_estimate:.6f}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
