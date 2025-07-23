# Heston Model Stock Price Simulation with Mean Reversion

This project implements a modified version of the Heston stochastic volatility model with a custom **mean reversion mechanism** toward the 52-week average price.

### 📈 Features:
- Heston simulation with and without mean reversion
- Black-Scholes comparison
- Backtesting on historical data (2020–2023)
- RMSE & MedAE metrics
- Visualization: mean/median paths, percentiles, and final distribution

### 🧠 Why Mean Reversion?
Mean reversion helps simulate real-world stock behavior more accurately, especially for volatile stocks like TSLA, by nudging paths back toward historical averages.

### 📊 Output:
- Simulated forward paths
- Model comparison against actual stock data
- Visualized backtests and error metrics

### 🛠️ How to Run:
```bash
python heston_model.py
