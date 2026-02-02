import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class Order:
    ticker: str
    date: pd.Timestamp
    side: int  # +1 buy, -1 sell
    shares: int
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float] = None
    id: Optional[int] = None

@dataclass
class Fill:
    ticker: str
    date: pd.Timestamp
    shares: int
    price: float
    commission: float
    fee: float  # per-dollar fee if any
    order_id: Optional[int] = None

class SimpleFillModel:
    """
    Fill model that:
    - For market orders: fills at next day's open (or same day open if used) with slippage_pct applied.
    - For limit orders: fills if limit crosses day's high/low. Partial fills limited by ADV participation.
    - Partial fills limited by max_participation * ADV (ADV computed externally and passed in).
    """
    def __init__(self, adv_lookback=20, max_participation=0.1, slippage_pct=0.0005):
        self.adv_lookback = adv_lookback
        self.max_participation = max_participation
        self.slippage_pct = slippage_pct

    def adv(self, df: pd.DataFrame, date: pd.Timestamp) -> float:
        # ADV ending the previous trading day
        window = df.loc[:date].tail(self.adv_lookback)
        if len(window) == 0:
            return df['Volume'].median() or 1_000
        return max(1.0, window['Volume'].mean())

    def execute_market(self, df: pd.DataFrame, date: pd.Timestamp, order: Order) -> Fill:
        # use the day's Open as execution price and slippage
        row = df.loc[date]
        base_px = row['Open']
        # slippage direction: buys pay higher, sells receive lower
        exec_px = base_px * (1 + order.side * self.slippage_pct)
        return Fill(order.ticker, date, order.shares, exec_px, commission=0.0, fee=0.0, order_id=order.id)

    def try_fill_limit(self, df: pd.DataFrame, date: pd.Timestamp, order: Order) -> Tuple[int, Optional[float]]:
        row = df.loc[date]
        # if buy limit: fill if low <= limit <= high and limit <= high (i.e., price went as low as limit)
        filled_shares = 0
        fill_price = None
        if order.order_type != 'limit' or order.limit_price is None:
            return 0, None
        if order.side == 1:
            # buy limit: fill if low <= limit
            if row['Low'] <= order.limit_price:
                filled_shares = order.shares
                fill_price = order.limit_price
        else:
            # sell limit: fill if high >= limit
            if row['High'] >= order.limit_price:
                filled_shares = order.shares
                fill_price = order.limit_price
        return filled_shares, fill_price

    def cap_by_liquidity(self, allowed_shares: int, adv: float) -> int:
        cap = int(max(1, adv * self.max_participation))
        return min(allowed_shares, cap)

class Backtester:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],  # each df must have Open, High, Low, Close, Adj Close, Volume indexed by date
        initial_cash: float = 100_000,
        commission_per_share: float = 0.005,
        commission_per_trade: float = 0.0,
        per_dollar_fee_pct: float = 0.0,
        fill_model: Optional[SimpleFillModel] = None
    ):
        self.price_data = price_data
        self.tickers = list(price_data.keys())
        self.initial_cash = initial_cash
        self.commission_per_share = commission_per_share
        self.commission_per_trade = commission_per_trade
        self.per_dollar_fee_pct = per_dollar_fee_pct
        self.fill_model = fill_model or SimpleFillModel()
        # state
        self.cash = initial_cash
        self.positions = {t: 0 for t in self.tickers}
        self.position_value = {t: 0.0 for t in self.tickers}
        self.equity_curve = []
        self.order_counter = 0
        self.fills: List[Fill] = []
        # aligned calendar
        self.dates = self._common_calendar()

    def _common_calendar(self) -> List[pd.Timestamp]:
        # intersection of all tickers' index
        idxs = [set(df.index) for df in self.price_data.values()]
        common = sorted(list(set.intersection(*idxs)))
        return common

    def target_to_orders(self, date: pd.Timestamp, targets: Dict[str, float]) -> List[Order]:
        # targets: ticker -> target dollar allocation (positive means long amount in dollars)
        orders = []
        # determine current equity
        total_equity = self.cash + sum(self.positions[t] * self.price_data[t].loc[date]['Adj Close'] for t in self.tickers)
        for t, target_dollars in targets.items():
            px = self.price_data[t].loc[date]['Open']
            desired_shares = int(np.floor(target_dollars / px)) if px > 0 else 0
            delta = desired_shares - self.positions[t]
            if delta == 0:
                continue
            side = 1 if delta > 0 else -1
            shares = abs(delta)
            self.order_counter += 1
            ord = Order(ticker=t, date=date, side=side, shares=shares, order_type='market', id=self.order_counter)
            orders.append(ord)
        return orders

    def step(self, date: pd.Timestamp, targets: Dict[str, float]):
        # create orders and execute them
        orders = self.target_to_orders(date, targets)
        # shuffle deterministic order: by ticker name
        orders = sorted(orders, key=lambda o: o.ticker)
        for o in orders:
            df = self.price_data[o.ticker]
            adv = self.fill_model.adv(df, date)
            max_shares = self.fill_model.cap_by_liquidity(o.shares, adv)
            if max_shares <= 0:
                continue
            # for this simple implementation, treat all as market orders
            o_partial = Order(o.ticker, o.date, o.side, max_shares, o.order_type, o.limit_price, o.id)
            fill = self.fill_model.execute_market(df, date, o_partial)
            # compute commission and fee
            trade_value = fill.shares * fill.price
            commission = self.commission_per_trade + self.commission_per_share * fill.shares
            fee = trade_value * self.per_dollar_fee_pct
            # adjust cash and positions
            net_cash_change = -fill.shares * fill.price * o.side - commission - fee if o.side == 1 else fill.shares * fill.price * (-o.side) - commission - fee
            # buy: side=1 -> cash decreases by shares*price + costs
            # sell: side=-1 -> cash increases by shares*price - costs
            if o.side == 1:
                self.cash -= fill.shares * fill.price
                self.cash -= commission + fee
                self.positions[o.ticker] += fill.shares
            else:
                self.cash += fill.shares * fill.price
                self.cash -= commission + fee
                self.positions[o.ticker] -= fill.shares
            fill.commission = commission
            fill.fee = fee
            self.fills.append(fill)
        # mark-to-market
        total_mv = 0.0
        for t in self.tickers:
            px = self.price_data[t].loc[date]['Adj Close']
            mv = self.positions[t] * px
            self.position_value[t] = mv
            total_mv += mv
        total_equity = self.cash + total_mv
        self.equity_curve.append({'date': date, 'cash': self.cash, 'positions': dict(self.positions), 'position_value': dict(self.position_value), 'equity': total_equity})

    def run(self, signal_generator):
        """
        signal_generator(date, price_data, positions, cash) -> targets dict: ticker -> target dollar allocation
        """
        self.equity_curve = []
        for date in self.dates:
            targets = signal_generator(date, self.price_data, self.positions, self.cash)
            self.step(date, targets)
        eq = pd.DataFrame(self.equity_curve).set_index('date')
        eq['returns'] = eq['equity'].pct_change().fillna(0)
        return eq

    def performance(self, eq, trading_days=252):
        total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq['equity'].iloc[-1] / eq['equity'].iloc[0]) ** (1/years) - 1 if years>0 else np.nan
        sharpe = (eq['returns'].mean() / (eq['returns'].std()+1e-12)) * np.sqrt(trading_days)
        rolling_max = eq['equity'].cummax()
        drawdown = (eq['equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Sharpe (ann)': sharpe,
            'Max Drawdown': max_dd
        }