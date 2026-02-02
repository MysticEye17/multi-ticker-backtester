import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import yfinance as yf
from backtester import Backtester, SimpleFillModel
from strategies import SMAEvolutionStrategy
from walkforward import rolling_walkforward, slice_price_data
from datetime import datetime

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def fetch_data(tickers, start='2010-01-01', end=None):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        df = df[['Open','High','Low','Close','Adj Close','Volume']]
        df.index = pd.to_datetime(df.index)
        data[t] = df.dropna()
    # align index by intersection automatically in backtester
    return data

def evaluate_walkforward(price_data, strategy_cls, initial_cash=100_000, **bt_kwargs):
    # define rolling windows
    all_idx = sorted({d for df in price_data.values() for d in df.index})
    start_date = all_idx[0]
    end_date = all_idx[-1]
    windows = rolling_walkforward(price_data, start_date, end_date, train_years=3, test_years=1, step_months=12)
    results = []
    for (ts, te, vs, ve) in windows:
        # train slice and test slice
        train_pd = slice_price_data(price_data, ts, te)
        test_pd = slice_price_data(price_data, vs, ve)
        # simple parameter grid on SMA lengths
        best_params = None
        best_metric = -np.inf
        for short in [20,50,80]:
            for long in [100,150,200]:
                if short >= long: continue
                strat = strategy_cls(short=short, long=long)
                bt = Backtester(train_pd, initial_cash=initial_cash, **bt_kwargs)
                eq = bt.run(strat)
                perf = bt.performance(eq)
                score = perf['CAGR'] - 0.5*abs(perf['Max Drawdown'])  # simple objective
                if score > best_metric:
                    best_metric = score
                    best_params = (short, long)
        # evaluate best on test
        strat = strategy_cls(short=best_params[0], long=best_params[1])
        bt_test = Backtester(test_pd, initial_cash=initial_cash, **bt_kwargs)
        eq_test = bt_test.run(strat)
        perf_test = bt_test.performance(eq_test)
        results.append({
            'train_start': ts, 'train_end': te, 'test_start': vs, 'test_end': ve,
            'best_short': best_params[0], 'best_long': best_params[1],
            'train_score': best_metric,
            'test_CAGR': perf_test['CAGR'], 'test_Sharpe': perf_test['Sharpe (ann)'],
            'test_MaxDD': perf_test['Max Drawdown'],
            'final_equity': eq_test['equity'].iloc[-1]
        })
    return pd.DataFrame(results)

def sensitivity_analysis(price_data, strategy_cls, slippage_vals, commission_per_share_vals, **bt_base_kwargs):
    rows = []
    for s in tqdm(slippage_vals, desc="slippage"):
        for cps in commission_per_share_vals:
            bt_kwargs = dict(bt_base_kwargs)
            fill_model = SimpleFillModel(slippage_pct=s, max_participation=0.1)
            bt_kwargs['fill_model'] = fill_model
            bt_kwargs['commission_per_share'] = cps
            strat = strategy_cls(short=50, long=200)
            bt = Backtester(price_data, initial_cash=100_000, **bt_kwargs)
            eq = bt.run(strat)
            perf = bt.performance(eq)
            rows.append({'slippage_pct': s, 'commission_per_share': cps, **perf})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    tickers = ['SPY', 'AAPL', 'MSFT', 'GOOG']  # example basket
    print("Downloading data...")
    price_data = fetch_data(tickers, start='2010-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    print("Running walk-forward... (this will take a few minutes)")
    wf_res = evaluate_walkforward(price_data, SMAEvolutionStrategy, initial_cash=100_000)
    wf_res.to_csv(os.path.join(RESULT_DIR, 'walkforward_results.csv'), index=False)
    print("Walk-forward done. Saved to results/walkforward_results.csv")
    print("Running sensitivity sweep...")
    slippage_vals = [0.0000, 0.0005, 0.001, 0.002]  # 0bp, 0.05%, 0.1%, 0.2%
    cps_vals = [0.0, 0.001, 0.005, 0.01]  # per-share cost
    sens = sensitivity_analysis(price_data, SMAEvolutionStrategy, slippage_vals, cps_vals)
    sens.to_csv(os.path.join(RESULT_DIR, 'sensitivity.csv'), index=False)
    print("Sensitivity analysis saved to results/sensitivity.csv")