# Multi-Ticker Backtester with Walk-Forward & Cost Sensitivity

This project provides:
- A realistic (educational) backtester handling market & limit orders, partial fills, per-share & per-dollar commissions, slippage & price-impact via ADV.
- SMA crossover strategy across multiple tickers.
- Walk-forward validation and transaction-cost sensitivity analysis.

Requirements
- Python 3.9+
- Install: pip install -r requirements.txt

Quick start
1. Create a virtualenv and install:
   pip install -r requirements.txt

2. Edit the ticker list and date range in `run_backtest.py` if desired.

3. Run:
   python run_backtest.py

Outputs:
- CSV summary of walk-forward steps and metrics in `results/`
- Sensitivity run metrics in `results/`
- Optionally: You can adapt the strategy class in `strategies.py`.

Notes
- This is an educational backtester. For live trading or heavy research use a production-grade engine (QuantConnect Lean, Backtrader with enhancements, vectorbt, etc).
- The fill model is simplified: it uses next-day Open for market fills and high/low for limit fills, with partial fills limited by ADV * max_participation.
- Always validate results and add more realistic microstructure if you trade intraday or with large capital.