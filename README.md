# 📈 Heston Model Stock Simulation with Mean Reversion

This project implements a modified **Heston stochastic volatility model** with a custom **mean reversion strategy** toward the 52-week average price. It enhances realism when modeling high-growth or volatile stocks like Tesla and Gamestop, while remaining generalizable to large caps.

---

## 🚀 Features

- Heston model with and without mean reversion
- Black-Scholes model for baseline comparison
- Backtesting vs. historical prices (2020–2023)
- RMSE & MedAE accuracy metrics
- Visualization: simulated paths, percentiles, histograms, and comparisons
- Modular, object-oriented structure (with docstrings)

---

## 📊 Visual Outputs

- Simulated Monte Carlo stock price paths
- Final price distribution histogram
- 25th–75th percentile confidence bands
- Backtest comparison vs. actual price (mean & median)

---

## 📁 Project Structure

```
.
├── heston_model.py         # Full simulation and backtesting code
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore              # Git exclusions

```

---

## 🧠 Mean Reversion Strategy

In addition to stochastic volatility, this project adds a price correction term to pull the price toward its 52-week average:

- Adjusts strength based on deviation from 52-week average
- Exponentially decays over time
- Weighted by expected return (mu)
- Stronger effect in early simulation steps

This mimics behavior often seen in growth stocks that revert after sharp moves.

---

## 🔧 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/adambrown06/Custom-Heston-Model.git
   cd Custom-Heston-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python heston_model.py
   ```

---

## 📉 Sample Outputs

Save your plots from `matplotlib.pyplot` using `plt.savefig("results/filename.png")` in your code.

---

## 🤖 Future Additions:

- Option pricing integration
- Live feed forecasting
- Portfolio-level modeling

  ---

## 📚 References

- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*
- Black & Scholes (1973). *The Pricing of Options and Corporate Liabilities*
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

## 👤 Author

Developed by Adam Brown
[LinkedIn](https://www.linkedin.com/in/adam-brown-007a70234/) • [GitHub](https://github.com/adambrown06)

---
