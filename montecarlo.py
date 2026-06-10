import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


def fetch_price_data(ticker: str, period: str = "2y") -> pd.Series:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    return hist["Close"]


def compute_gbm_params(prices: pd.Series) -> tuple[float, float, float]:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean() * 252
    sigma = log_returns.std() * np.sqrt(252)
    S0 = float(prices.iloc[-1])
    return mu, sigma, S0


def run_monte_carlo(
    S0: float,
    mu: float,
    sigma: float,
    T: float = 1.0,
    n_steps: int = 252,
    n_simulations: int = 10000,
    risk_free_rate: float = 0.065,
) -> np.ndarray:
    dt = T / n_steps
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    random_shocks = np.random.standard_normal((n_simulations, n_steps))
    step_returns = np.exp(drift + diffusion * random_shocks)
    paths = np.ones((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        paths[:, t] = paths[:, t - 1] * step_returns[:, t - 1]
    return paths


def compute_risk_metrics(
    paths: np.ndarray,
    S0: float,
    risk_free_rate: float = 0.065,
    confidence_level: float = 0.95,
) -> dict:
    final_prices = paths[:, -1]
    returns = (final_prices - S0) / S0

    var = np.percentile(returns, (1 - confidence_level) * 100)
    cvar = returns[returns <= var].mean()

    annual_returns = returns
    excess_returns = annual_returns - risk_free_rate
    sharpe = excess_returns.mean() / annual_returns.std() if annual_returns.std() > 0 else 0.0

    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_drawdown = drawdowns.min(axis=1).mean()

    prob_profit = (final_prices > S0).mean()
    expected_price = final_prices.mean()
    median_price = np.median(final_prices)
    price_p10 = np.percentile(final_prices, 10)
    price_p90 = np.percentile(final_prices, 90)

    skewness = stats.skew(returns)
    kurt = stats.kurtosis(returns)

    return {
        "expected_price": expected_price,
        "median_price": median_price,
        "price_p10": price_p10,
        "price_p90": price_p90,
        "var_95": var,
        "cvar_95": cvar,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "prob_profit": prob_profit,
        "skewness": skewness,
        "kurtosis": kurt,
    }


def run_sensitivity_analysis(
    S0: float,
    mu: float,
    base_sigma: float,
    T: float = 1.0,
    n_steps: int = 252,
    n_simulations: int = 5000,
) -> pd.DataFrame:
    sigma_range = np.linspace(base_sigma * 0.5, base_sigma * 1.5, 7)
    results = []
    for sigma in sigma_range:
        paths = run_monte_carlo(S0, mu, sigma, T, n_steps, n_simulations)
        metrics = compute_risk_metrics(paths, S0)
        results.append({
            "sigma": round(sigma, 4),
            "expected_price": round(metrics["expected_price"], 2),
            "var_95": round(metrics["var_95"], 4),
            "cvar_95": round(metrics["cvar_95"], 4),
            "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
            "max_drawdown": round(metrics["max_drawdown"], 4),
        })
    return pd.DataFrame(results)


def plot_simulation_results(
    paths: np.ndarray,
    metrics: dict,
    ticker: str,
    S0: float,
    T: float,
    n_steps: int,
    sensitivity_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Monte Carlo Simulation — {ticker.upper()}   |   {len(paths):,} Paths   |   Horizon: {int(T * 252)}d",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    time_axis = np.linspace(0, T * 252, n_steps + 1)
    sample_idx = np.random.choice(len(paths), size=min(300, len(paths)), replace=False)
    for i in sample_idx:
        ax1.plot(time_axis, paths[i], alpha=0.05, color="steelblue", linewidth=0.4)
    ax1.plot(time_axis, np.percentile(paths, 5, axis=0), "r--", linewidth=1.4, label="5th / 95th Pctile")
    ax1.plot(time_axis, np.percentile(paths, 95, axis=0), "r--", linewidth=1.4)
    ax1.plot(time_axis, np.median(paths, axis=0), "gold", linewidth=2, label="Median Path")
    ax1.plot(time_axis, np.mean(paths, axis=0), "lime", linewidth=2, label="Mean Path")
    ax1.axhline(S0, color="white", linewidth=1, linestyle=":", alpha=0.7, label=f"S₀ = {S0:.2f}")
    ax1.set_facecolor("#0d1117")
    ax1.set_xlabel("Trading Days", color="white")
    ax1.set_ylabel("Price (₹ / $)", color="white")
    ax1.set_title("Simulated Price Paths (GBM)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=7, facecolor="#1c1f26", labelcolor="white")
    ax1.spines[:].set_color("#333")

    ax2 = fig.add_subplot(gs[0, 2])
    final_prices = paths[:, -1]
    ax2.hist(final_prices, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
    ax2.axvline(metrics["expected_price"], color="gold", linewidth=1.5, label=f'E[P]: {metrics["expected_price"]:.2f}')
    ax2.axvline(metrics["price_p10"], color="tomato", linewidth=1.2, linestyle="--", label=f'P10: {metrics["price_p10"]:.2f}')
    ax2.axvline(metrics["price_p90"], color="lime", linewidth=1.2, linestyle="--", label=f'P90: {metrics["price_p90"]:.2f}')
    ax2.set_facecolor("#0d1117")
    ax2.set_xlabel("Final Price", color="white")
    ax2.set_ylabel("Frequency", color="white")
    ax2.set_title("Terminal Price Distribution", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=7, facecolor="#1c1f26", labelcolor="white")
    ax2.spines[:].set_color("#333")

    ax3 = fig.add_subplot(gs[1, :2])
    sim_returns = (final_prices - S0) / S0
    ax3.hist(sim_returns, bins=80, color="#4a90d9", edgecolor="none", alpha=0.85)
    var_val = metrics["var_95"]
    ax3.axvline(var_val, color="red", linewidth=2, label=f'VaR 95%: {var_val:.2%}')
    ax3.axvline(metrics["cvar_95"], color="darkred", linewidth=2, linestyle="--", label=f'CVaR 95%: {metrics["cvar_95"]:.2%}')
    ax3.fill_betweenx(
        [0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 1000],
        sim_returns.min(),
        var_val,
        alpha=0.2,
        color="red",
    )
    ax3.set_facecolor("#0d1117")
    ax3.set_xlabel("Simulated Return", color="white")
    ax3.set_ylabel("Frequency", color="white")
    ax3.set_title("Return Distribution with VaR / CVaR", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(fontsize=8, facecolor="#1c1f26", labelcolor="white")
    ax3.spines[:].set_color("#333")

    ax4 = fig.add_subplot(gs[1, 2])
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    mean_drawdown = drawdowns.mean(axis=0)
    worst_drawdown = drawdowns.min(axis=0)
    ax4.fill_between(time_axis, worst_drawdown, 0, alpha=0.3, color="tomato", label="Worst Path")
    ax4.plot(time_axis, mean_drawdown, color="orange", linewidth=1.5, label="Mean Drawdown")
    ax4.set_facecolor("#0d1117")
    ax4.set_xlabel("Trading Days", color="white")
    ax4.set_ylabel("Drawdown", color="white")
    ax4.set_title("Drawdown Profile", color="white")
    ax4.tick_params(colors="white")
    ax4.legend(fontsize=7, facecolor="#1c1f26", labelcolor="white")
    ax4.spines[:].set_color("#333")

    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(sensitivity_df["sigma"], sensitivity_df["var_95"], "r-o", markersize=4, label="VaR 95%")
    ax5.plot(sensitivity_df["sigma"], sensitivity_df["cvar_95"], "darkred", linestyle="--", marker="s", markersize=4, label="CVaR 95%")
    ax5_twin = ax5.twinx()
    ax5_twin.plot(sensitivity_df["sigma"], sensitivity_df["sharpe_ratio"], "gold", linestyle="-.", marker="^", markersize=4, label="Sharpe")
    ax5_twin.set_ylabel("Sharpe Ratio", color="gold")
    ax5_twin.tick_params(axis="y", colors="gold")
    ax5.set_facecolor("#0d1117")
    ax5.set_xlabel("Volatility (σ)", color="white")
    ax5.set_ylabel("Risk Metric", color="white")
    ax5.set_title("Sensitivity: VaR / CVaR / Sharpe vs Volatility", color="white")
    ax5.tick_params(colors="white")
    ax5.legend(fontsize=7, loc="lower left", facecolor="#1c1f26", labelcolor="white")
    ax5_twin.legend(fontsize=7, loc="upper right", facecolor="#1c1f26", labelcolor="white")
    ax5.spines[:].set_color("#333")
    ax5_twin.spines[:].set_color("#333")

    ax6 = fig.add_subplot(gs[2, 2])
    labels = [
        f'Expected: {metrics["expected_price"]:.2f}',
        f'Median: {metrics["median_price"]:.2f}',
        f'VaR 95%: {metrics["var_95"]:.2%}',
        f'CVaR 95%: {metrics["cvar_95"]:.2%}',
        f'Sharpe: {metrics["sharpe_ratio"]:.3f}',
        f'Max DD: {metrics["max_drawdown"]:.2%}',
        f'P(Profit): {metrics["prob_profit"]:.2%}',
        f'Skew: {metrics["skewness"]:.3f}',
        f'Kurtosis: {metrics["kurtosis"]:.3f}',
    ]
    ax6.set_facecolor("#0d1117")
    ax6.axis("off")
    ax6.set_title("Risk Summary", color="white", fontsize=11, fontweight="bold")
    for i, label in enumerate(labels):
        key, val = label.split(": ", 1)
        ax6.text(0.02, 0.88 - i * 0.105, key + ":", transform=ax6.transAxes,
                 color="#aaa", fontsize=9)
        ax6.text(0.58, 0.88 - i * 0.105, val, transform=ax6.transAxes,
                 color="white", fontsize=9, fontweight="bold")

    fig.patch.set_facecolor("#0d1117")
    plt.savefig(f"monte_carlo_{ticker.lower()}.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.show()
    print(f"\nPlot saved → monte_carlo_{ticker.lower()}.png")


def print_summary(ticker: str, S0: float, mu: float, sigma: float, metrics: dict, n_simulations: int) -> None:
    width = 56
    print("\n" + "=" * width)
    print(f"  Monte Carlo Risk Report — {ticker.upper()}")
    print("=" * width)
    print(f"  Spot Price (S₀)      : {S0:>10.2f}")
    print(f"  Annual Drift (μ)     : {mu:>10.4f}  ({mu*100:.2f}%)")
    print(f"  Annual Volatility (σ): {sigma:>10.4f}  ({sigma*100:.2f}%)")
    print(f"  Simulations          : {n_simulations:>10,}")
    print("-" * width)
    print(f"  Expected Price       : {metrics['expected_price']:>10.2f}")
    print(f"  Median Price         : {metrics['median_price']:>10.2f}")
    print(f"  10th Percentile      : {metrics['price_p10']:>10.2f}")
    print(f"  90th Percentile      : {metrics['price_p90']:>10.2f}")
    print(f"  P(Profit > S₀)       : {metrics['prob_profit']:>10.2%}")
    print("-" * width)
    print(f"  VaR  (95%, 1Y)       : {metrics['var_95']:>10.2%}")
    print(f"  CVaR (95%, 1Y)       : {metrics['cvar_95']:>10.2%}")
    print(f"  Sharpe Ratio         : {metrics['sharpe_ratio']:>10.4f}")
    print(f"  Avg Max Drawdown     : {metrics['max_drawdown']:>10.2%}")
    print(f"  Return Skewness      : {metrics['skewness']:>10.4f}")
    print(f"  Return Excess Kurt.  : {metrics['kurtosis']:>10.4f}")
    print("=" * width + "\n")


def main(
    ticker: str = "RELIANCE.NS",
    horizon_days: int = 252,
    n_simulations: int = 10000,
    history_period: str = "2y",
    risk_free_rate: float = 0.065,
    run_sensitivity: bool = True,
) -> None:
    np.random.seed(42)
    print(f"\nFetching data for {ticker} ...")
    prices = fetch_price_data(ticker, period=history_period)
    print(f"  {len(prices)} trading days loaded  |  Latest close: {prices.iloc[-1]:.2f}")

    mu, sigma, S0 = compute_gbm_params(prices)
    T = horizon_days / 252

    print(f"\nRunning {n_simulations:,} Monte Carlo simulations ...")
    paths = run_monte_carlo(S0, mu, sigma, T=T, n_steps=horizon_days,
                            n_simulations=n_simulations, risk_free_rate=risk_free_rate)

    metrics = compute_risk_metrics(paths, S0, risk_free_rate=risk_free_rate)
    print_summary(ticker, S0, mu, sigma, metrics, n_simulations)

    sensitivity_df = pd.DataFrame()
    if run_sensitivity:
        print("Running volatility sensitivity analysis ...")
        sensitivity_df = run_sensitivity_analysis(S0, mu, sigma, T=T,
                                                  n_steps=horizon_days, n_simulations=3000)
        print("\nSensitivity Table:")
        print(sensitivity_df.to_string(index=False))

    plot_simulation_results(paths, metrics, ticker, S0, T, horizon_days, sensitivity_df)


if __name__ == "__main__":
    main(
        ticker="RELIANCE.NS",
        horizon_days=252,
        n_simulations=10000,
        history_period="2y",
        risk_free_rate=0.065,
        run_sensitivity=True,
    )