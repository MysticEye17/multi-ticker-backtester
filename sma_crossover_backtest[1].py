# Simple SMA crossover backtest (educational)
# Requirements: pip install yfinance pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
from math import prod

def get_data(ticker, start='2015-01-01', end=None):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[['Open','High','Low','Close','Adj Close','Volume']]
    df = df.dropna()
    return df

def sma_signals(df, short=50, long=200):
    df = df.copy()
    df['sma_short'] = df['Adj Close'].rolling(short).mean()
    df['sma_long']  = df['Adj Close'].rolling(long).mean()
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
    df.loc[df['sma_short'] <= df['sma_long'], 'signal'] = 0
    df['position'] = df['signal'].shift(1).fillna(0)  # enter next day
    return df.dropna()

def simulate(df, initial_cash=100000, commission_per_trade=1.0, slippage_pct=0.001):
    cash = initial_cash
    shares = 0
    equity_curve = []
    prev_pos = 0

    for idx, row in df.iterrows():
        price = row['Open']  # assume we fill at next open (simplification)
        pos = int(row['position'])  # 0 or 1

        # position change -> trade
        if pos != prev_pos:
            # if entering long
            if pos == 1:
                # buy as many shares as possible with cash (simple sizing: all-in)
                size = int(cash / (price * (1 + slippage_pct)))
                cost = size * price * (1 + slippage_pct) + commission_per_trade
                if size > 0 and cost <= cash:
                    shares = size
                    cash -= cost
            # if exiting to cash
            else:
                if shares > 0:
                    proceeds = shares * price * (1 - slippage_pct) - commission_per_trade
                    cash += proceeds
                    shares = 0
            prev_pos = pos

        # mark-to-market
        market_value = shares * row['Adj Close']
        total = cash + market_value
        equity_curve.append({'date': idx, 'cash': cash, 'shares': shares, 'market_value': market_value, 'equity': total})

    eq = pd.DataFrame(equity_curve).set_index('date')
    eq['returns'] = eq['equity'].pct_change().fillna(0)
    return eq

def performance(eq, trading_days=252):
    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq['equity'].iloc[-1] / eq['equity'].iloc[0]) ** (1/years) - 1
    sharpe = (eq['returns'].mean() / eq['returns'].std()) * np.sqrt(trading_days)
    rolling_max = eq['equity'].cummax()
    drawdown = (eq['equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe (ann)': sharpe,
        'Max Drawdown': max_dd
    }

if __name__ == "__main__":
    ticker = "SPY"
    df = get_data(ticker, start='2015-01-01')
    df = sma_signals(df, short=50, long=200)
    eq = simulate(df, initial_cash=100000, commission_per_trade=1.0, slippage_pct=0.0005)
    perf = performance(eq)
    print("Performance:", {k: round(v,4) if isinstance(v,float) else v for k,v in perf.items()})
    # Optionally: eq['equity'].plot()