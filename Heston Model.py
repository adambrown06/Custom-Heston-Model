import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

"""
Heston Model Overview:
----------------------
The Heston model introduces stochastic volatility into the standard Black-Scholes framework.

Core Dynamics (S = stock price, V = variance):
    dS_t = μ * S_t * dt + sqrt(V_t) * S_t * dW1_t
    dV_t = κ(θ - V_t) * dt + σ * sqrt(V_t) * dW2_t

Discretization:
    S_{t+1} = S_t + μ * S_t * dt + sqrt(V_t) * S_t * sqrt(dt) * Z1
    V_{t+1} = V_t + κ(θ - V_t) * dt + σ * sqrt(V_t) * sqrt(dt) * Z2

Where:
    - Z1 and Z2 are standard normal variables with correlation ρ
    - dW2_t = ρ * dW1_t + sqrt(1 - ρ²) * dZ_t (correlated Brownian motions)

Custom Modification:
--------------------
To improve realism, a **mean reversion strategy toward the 52-week average price** is added to the S_t update step.
This dynamic pulls the simulated stock prices back toward a long-term average, scaled by:
    - Deviation from the 52-week average
    - Decay over time
    - Estimated return magnitude (μ)
    - Stronger influence in early steps due to possible skew of extreme returns
This approach aims to capture the tendency of stock prices to revert to their historical averages, enhancing the model's realism.
----------------------
"""

class HestonModel:
    def __init__(self, stock_symbol):
        # Basic setup
        self.stock_symbol = stock_symbol
        self.data = None

        # Model parameters
        self.S0 = None       # Initial stock price
        self.V0 = None       # Initial variance
        self.mu = None       # Expected return
        self.theta = None    # Long-term variance
        self.kappa = 2.0     # Mean reversion speed of variance
        self.rho = -0.8      # Correlation between dW1 and dW2
        self.sigma = 0.7     # Volatility of volatility
        self.dt = 1/252      # Time step size (daily)
        self.S_bs = None     # Black-Scholes simulated stock prices

        # Custom strategy variables
        self.average_52 = None   # 52-week moving average of stock price

        # Simulation storage
        self.S_sim = []
        self.V_sim = []

    def fetch_data(self):
        # Download 2 years of daily stock data
        self.data = yf.download(self.stock_symbol, period='3y', interval='1d')
        if self.data.empty:
            raise ValueError("No data fetched. Please check the stock symbol and date range.")
        return "Data fetched successfully."

    def calculate_parameters(self):
        if self.data is None or self.data.empty:
            raise ValueError("Stock data not available. Run fetch_data() first.")

        # Initial stock price = most recent close
        self.S0 = float(self.data['Close'].iloc[-1])

        # Calculate log returns and rolling variance
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        rolling_var = log_returns.rolling(window=30).var().dropna()

        if rolling_var.empty:
            raise ValueError("Rolling variance is empty. Not enough data to compute variance.")

        # Annualized values
        mu_log = float(log_returns.mean()) * 252
        self.V0 = float(rolling_var.iloc[-1]) * 252
        self.theta = float(rolling_var.mean()) * 252
        self.mu = mu_log + 0.5 * self.V0  # Adjusted expected return

        # Compute 52-week moving average
        self.average_52 = float(self.data['Close'].iloc[-252:].mean())

    def simulate_paths(self, num_paths, use_mean_reversion=True):
        if use_mean_reversion: # Can toggle mean reversion to have plain Heston model or custom Heston model
            reversion = 1
        else:
            reversion = 0
        num_steps = 252  # Simulate 1 year of trading days
        self.S_sim = np.zeros((num_paths, num_steps))
        self.V_sim = np.zeros((num_paths, num_steps))
        self.S_sim[:, 0] = self.S0
        self.V_sim[:, 0] = self.V0

        if not np.isfinite(self.rho):
            raise ValueError(f"Invalid rho value: {self.rho}")

        for t in range(1, num_steps):
            # Mean reversion strategy toward 52-week average
            deviation = np.abs(self.S_sim[:, t-1] - self.average_52) / self.average_52
            decay = np.exp(-t / 250)  # Time decay of influence

            # Dynamically adjust beta based on return magnitude
            if abs(self.mu) > 0.3:
                base_strength = 0.15
            elif abs(self.mu) > 0.1:
                base_strength = 0.1
            else:
                base_strength = 0.05
            beta_array = reversion * (base_strength + deviation * decay)

            # Stronger mean reversion in early steps to account for potential early skew
            multiplier = 1.5 if t < 50 else 1.0

            # Generate correlated Brownian motions
            Z1 = np.random.normal(0, 1, num_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, num_paths)

            # Update variance
            V_prev = self.V_sim[:, t-1]
            self.V_sim[:, t] = V_prev + self.kappa * (self.theta - V_prev) * self.dt \
                               + self.sigma * np.sqrt(np.maximum(V_prev, 0)) * np.sqrt(self.dt) * Z2
            self.V_sim[:, t] = np.maximum(self.V_sim[:, t], 0)

            # Update stock price with added mean-reversion term
            S_prev = self.S_sim[:, t-1]
            drift = self.mu * S_prev * self.dt
            reversion = multiplier * beta_array * (self.average_52 - S_prev) * self.dt
            diffusion = np.sqrt(np.maximum(V_prev, 0)) * S_prev * np.sqrt(self.dt) * Z1
            self.S_sim[:, t] = S_prev + drift + reversion + diffusion
            self.S_sim[:, t] = np.maximum(self.S_sim[:, t], 0)

    def plot_paths(self):
        plt.figure(figsize=(14, 7))
        days = np.arange(self.S_sim.shape[1])
        for i in range(self.S_sim.shape[0]):
            plt.plot(days, self.S_sim[i], alpha=0.1, linewidth=1)
        plt.title(f'Simulated Stock Price Paths for {self.stock_symbol}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.xticks(np.arange(0, 252, 25))
        plt.grid(True)
        plt.show()

    def plot_average_vs_median_path(self):
        if len(self.S_sim) == 0:
            raise ValueError("No simulations found. Run simulate_paths() first.")

        avg_path = np.mean(self.S_sim, axis=0)
        median_path = np.median(self.S_sim, axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(avg_path, color='blue', linewidth=2.5, label='Mean Price')
        plt.plot(median_path, color='green', linewidth=2.5, label='Median Price')
        plt.title(f'Mean vs Median Simulated Price Path for {self.stock_symbol}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_percentile_bands(self):
        if len(self.S_sim) == 0:
            raise ValueError("No simulations found. Run simulate_paths() first.")

        days = np.arange(self.S_sim.shape[1])
        mean_path = np.mean(self.S_sim, axis=0)
        p25 = np.percentile(self.S_sim, 25, axis=0)
        p75 = np.percentile(self.S_sim, 75, axis=0)

        plt.figure(figsize=(12, 6))
        plt.plot(days, mean_path, color='blue', label='Mean Price', linewidth=2)
        plt.fill_between(days, p25, p75, color='lightblue', alpha=0.5, label='25th–75th Percentile')
        plt.title(f'Simulated Price with 25th–75th Percentile Bands for {self.stock_symbol}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_final_price_distribution(self):
        if len(self.S_sim) == 0:
            raise ValueError("No simulations found. Run simulate_paths() first.")

        final_prices = self.S_sim[:, -1]

        plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=25, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of Final Simulated Stock Prices for {self.stock_symbol}')
        plt.xlabel('Final Stock Price')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    


class Backtest(HestonModel): #Backtesting over past data to see if it is realistic
    def __init__(self, stock_symbol):
        super().__init__(stock_symbol)
        self.backtest_data = None
        self.actual_price = None
        self.heston_custom_mean = None
        self.heston_custom_med = None      
        self.heston_plain_mean = None
        self.heston_plain_med = None
        self.bs_mean = None
        self.bs_median = None
        
    def fetch_backtest_data(self, start_date, end_date):
        self.backtest_data = yf.download(self.stock_symbol, start=start_date, end=end_date, interval='1d')
        if self.backtest_data.empty:
            raise ValueError("No backtest data fetched. Please check the stock symbol and date range.")
        return "Backtest data fetched successfully."
    def backtest_parameters(self):
        if self.backtest_data is None or self.backtest_data.empty:
            raise ValueError("Backtest data not available. Run fetch_backtest_data() first.")

        # Initial stock price = most recent close
        self.S0 = float(self.backtest_data['Close'].iloc[-1])

        # Calculate log returns and rolling variance
        log_returns = np.log(self.backtest_data['Close'] / self.backtest_data['Close'].shift(1)).dropna()
        rolling_var = log_returns.rolling(window=30).var().dropna()

        if rolling_var.empty:
            raise ValueError("Rolling variance is empty. Not enough data to compute variance.")

        # Annualized values
        mu_log = float(log_returns.mean()) * 252
        self.V0 = float(rolling_var.iloc[-1]) * 252
        self.theta = float(rolling_var.mean()) * 252
        self.mu = mu_log + 0.5 * self.V0
        # Compute 52-week moving average
        self.average_52 = float(self.backtest_data['Close'].iloc[-252:].mean())
    def simulate_backtest(self, num_paths):
        super().simulate_paths(num_paths)
    def plot_backtest_paths(self):
        super().plot_paths()
    def plot_backtest_average_vs_median_path(self):
        super().plot_average_vs_median_path()
    def plot_backtest_percentile_bands(self):
        super().plot_percentile_bands()
    def plot_backtest_final_price_distribution(self):
        super().plot_final_price_distribution()
    def backtest_black_scholes(self, num_paths=1000, num_steps=252):
        self.S_bs = np.zeros((num_paths, num_steps))
        self.S_bs[:, 0] = self.S0

        for t in range(1, num_steps):
            Z = np.random.normal(0, 1, num_paths)
            self.S_bs[:, t] = self.S_bs[:, t-1] * np.exp((self.mu - 0.5 * self.V0) * self.dt + np.sqrt(self.V0) * np.sqrt(self.dt) * Z)
    def calculate_mean_median(self):
        # Custom Heston median and mean
        self.heston_custom_med = np.median(self.S_sim, axis=0)
        self.heston_custom_mean = np.mean(self.S_sim, axis=0)

        # Simulate plain Heston (without mean reversion)
        self.simulate_paths(1000, use_mean_reversion=False)
        self.heston_plain_med = np.median(self.S_sim, axis=0)
        self.heston_plain_mean = np.mean(self.S_sim, axis=0)

        # Black-Scholes already simulated
        self.bs_median = np.median(self.S_bs, axis=0)
        self.bs_mean = np.mean(self.S_bs, axis=0)

        # Actual price
        actual = yf.download(self.stock_symbol, '2023-01-01', '2024-01-01', interval='1d')['Close']
        self.actual_price = actual.values[:504]

    def calculate_rmse(self, predicted): #root mean square error
        return np.sqrt(np.mean((predicted - self.actual_price) ** 2))
    def calculate_MedAE(self, predicted): #median absolute error
        return np.median(np.abs(predicted - self.actual_price))
    def plot_backtest_mean(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.heston_custom_mean, label='Heston w/ Mean Reversion', linewidth=2)
        plt.plot(self.heston_plain_mean, label='Heston w/o Mean Reversion', linestyle='-.', linewidth=2)
        plt.plot(self.bs_mean, label='Black-Scholes', linestyle='--', linewidth=2)
        plt.plot(self.actual_price, label='Actual Price', linestyle=':', linewidth=2, color='black')
        plt.title(f'{self.stock_symbol}: Backtest - All Models Mean vs Actual Price')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_backtest_median(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.heston_custom_med, label='Heston w/ Mean Reversion', linewidth=2)
        plt.plot(self.heston_plain_med, label='Heston w/o Mean Reversion', linestyle='-.', linewidth=2)
        plt.plot(self.bs_median, label='Black-Scholes', linestyle='--', linewidth=2)
        plt.plot(self.actual_price, label='Actual Price', linestyle=':', linewidth=2, color='black')
        plt.title(f'{self.stock_symbol}: Backtest - All Models Median vs Actual Price') 
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
if __name__ == "__main__":
    print("="*60)
    print("    HESTON MODEL SIMULATION STARTED")
    print("="*60 + "\n")
    stock = 'TSLA' # Specify the stock symbol here, e.g., 'AAPL', 'GOOGL', etc.
    # Run simulation
    test = HestonModel(stock)
    test.fetch_data()
    test.calculate_parameters()
    test.simulate_paths(1000)
    test.plot_paths()
    test.plot_average_vs_median_path()
    test.plot_percentile_bands()
    test.plot_final_price_distribution()

    # Display sim parameters
    print(f"""\n{"="*60}
    Calculated Parameters:
    - S0        = {test.S0:.2f}
    - mu        = {test.mu:.6f}
    - V0        = {test.V0:.6f}
    - theta     = {test.theta:.6f}
    - avg_52w   = {test.average_52:.2f}
    """)

    print("\n" + "="*60)
    print("    HESTON MODEL SIMULATION COMPLETE — BEGIN BACKTESTING")
    print("="*60 + "\n")

    # Run backtest
    backtest = Backtest(stock)
    backtest.fetch_backtest_data('2020-01-01', '2023-01-01')
    backtest.backtest_parameters()
    backtest.simulate_backtest(1000)
    backtest.plot_backtest_paths()
    backtest.plot_backtest_average_vs_median_path()
    backtest.plot_backtest_percentile_bands()
    backtest.plot_backtest_final_price_distribution()
    backtest.backtest_black_scholes(num_paths=1000, num_steps=252)
    backtest.calculate_mean_median()
    backtest.plot_backtest_mean()
    backtest.plot_backtest_median()

    # Display backtest parameters
    print(f"""\nBacktest Parameters:
        - S0        = {backtest.S0:.2f}             
        - mu        = {backtest.mu:.6f}
        - V0        = {backtest.V0:.6f}
        - theta     = {backtest.theta:.6f}
        - avg_52w   = {backtest.average_52:.2f}
    """)

    # Calculate and display RMSE and MedAE for each model
    print("\n" + "="*60)
    print("    MODEL PERFORMANCE METRICS")
    print("="*60 + "\n")
    print(f"RMSE (Heston w/ Mean Reversion): {backtest.calculate_rmse(backtest.heston_custom_mean):.4f}")
    print(f"RMSE (Heston w/o Mean Reversion): {backtest.calculate_rmse(backtest.heston_plain_mean):.4f}")
    print(f"RMSE (Black-Scholes): {backtest.calculate_rmse(backtest.bs_mean):.4f}")
    print('\n')
    print(f"MedAE (Heston w/ Mean Reversion): {backtest.calculate_MedAE(backtest.heston_custom_med):.4f}")
    print(f"MedAE (Heston w/o Mean Reversion): {backtest.calculate_MedAE(backtest.heston_plain_med):.4f}")
    print(f"MedAE (Black-Scholes): {backtest.calculate_MedAE(backtest.bs_median):.4f}")
    print("\n" + "="*60)
    print("    BACKTESTING COMPLETE")