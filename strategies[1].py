import pandas as pd
from typing import Dict
import numpy as np

class SMAEvolutionStrategy:
    """
    Multi-ticker SMA crossover strategy that outputs dollar targets per ticker.
    - short & long in days
    - equal-weight portfolio among tickers when signal=1
    - position sizing: equal-dollar allocation across all tickers that have a long signal
    """
    def __init__(self, short=50, long=200, min_periods=1):
        assert short < long
        self.short = short
        self.long = long
        self.min_periods = min_periods
        self.sma_cache = {}

    def compute_signals(self, price_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> Dict[str, int]:
        signals = {}
        for t, df in price_data.items():
            # compute SMAs if needed
            if t not in self.sma_cache:
                self.sma_cache[t] = pd.DataFrame(index=df.index)
                self.sma_cache[t]['Adj Close'] = df['Adj Close']
                self.sma_cache[t]['sma_short'] = df['Adj Close'].rolling(self.short, min_periods=self.min_periods).mean()
                self.sma_cache[t]['sma_long'] = df['Adj Close'].rolling(self.long, min_periods=self.min_periods).mean()
            row = self.sma_cache[t].loc[date]
            signal = 1 if row['sma_short'] > row['sma_long'] else 0
            signals[t] = int(signal)
        return signals

    def __call__(self, date, price_data, positions, cash) -> Dict[str, float]:
        signals = self.compute_signals(price_data, date)
        longs = [t for t,s in signals.items() if s==1]
        total_equity = cash + sum(positions[t] * price_data[t].loc[date]['Adj Close'] for t in price_data.keys())
        targets = {}
        if len(longs) == 0:
            # flatten to cash
            for t in price_data.keys():
                targets[t] = 0.0
            return targets
        weight = 1.0 / len(longs)
        for t in price_data.keys():
            if t in longs:
                targets[t] = total_equity * weight
            else:
                targets[t] = 0.0
        return targets