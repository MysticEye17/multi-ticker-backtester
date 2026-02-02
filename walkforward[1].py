import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from copy import deepcopy

def rolling_walkforward(price_data: Dict[str, pd.DataFrame], start_date: pd.Timestamp, end_date: pd.Timestamp,
                        train_years=3, test_years=1, step_months=12) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Returns a list of (train_start, train_end, test_start, test_end) date ranges.
    - train_years, test_years specify lengths
    - step_months is how far the window moves each iteration (commonly equal to test length)
    """
    windows = []
    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(years=test_years)
    step_delta = pd.DateOffset(months=step_months)
    cur_train_start = pd.Timestamp(start_date)
    while True:
        train_start = cur_train_start
        train_end = train_start + train_delta - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + test_delta - pd.Timedelta(days=1)
        if test_end > end_date:
            break
        windows.append((train_start, train_end, test_start, test_end))
        cur_train_start = cur_train_start + step_delta
    return windows

def slice_price_data(price_data: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    sliced = {t: df.loc[(df.index>=start)&(df.index<=end)].copy() for t,df in price_data.items()}
    return sliced