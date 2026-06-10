# Monte-Carlo-Simulation-
Monte Carlo stock price simulator using Geometric Brownian Motion — real-time data, 10K path simulations, VaR/CVaR/Sharpe/drawdown analytics and volatility sensitivity analysis.

This project simulates future stock price paths using Geometric Brownian Motion calibrated to real historical data. The engine fetches live closing prices, estimates annualised drift and volatility, and runs large-scale parallel simulations to build a full return distribution — not just a point estimate. Risk metrics including Value at Risk, Conditional VaR, Sharpe Ratio, and maximum drawdown are computed across the distribution, and a volatility sensitivity sweep shows how each metric responds to changes in market conditions. Built with a latency-conscious, vectorised pipeline and outputs a dark-themed 6-panel dashboard saved as PNG.


