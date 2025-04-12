# Import libraries
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from bayes_opt import BayesianOptimization

# OANDA API Credentials
# Note: Replace with your own API key securely (see README for instructions)
OANDA_API_KEY = "YOUR_OANDA_API_KEY_HERE"  # Placeholder; do not hardcode
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

class RealisticBacktester:
    def __init__(self):
        self.cached_data = {}
        self.slippage = 0.02  # 2 pips for JPY pairs
        self.partial_fill_prob = 0.1
        self.order_delay = 2
        self.spread = 0.02  # 2 pips spread for JPY pairs
        self.commission_per_lot = 0.5  # 0.5 pips commission

    def fetch_candles(self, instrument, granularity, from_time=None, to_time=None):
        cache_key = f"{instrument}_{granularity}_{from_time}_{to_time}"
        if cache_key in self.cached_data:
            print(f"Using cached data for {instrument} ({granularity})")
            return self.cached_data[cache_key]
            
        print(f"Fetching {granularity} candles for {instrument} from {from_time} to {to_time}...")
        all_data = []
        current_from = from_time
        max_candles = 5000
        
        while current_from < to_time:
            chunk_to = min(current_from + timedelta(hours=max_candles * 4), to_time)
            params = {
                "granularity": granularity,
                "from": current_from.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "to": chunk_to.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            request = InstrumentsCandles(instrument=instrument, params=params)
            response = client.request(request)
            candles = [candle for candle in response['candles'] if candle.get('complete')]
            if not candles:
                break
            df_chunk = pd.DataFrame([{
                'time': pd.to_datetime(candle['time']),
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': candle['volume']
            } for candle in candles])
            all_data.append(df_chunk)
            current_from = chunk_to
            time.sleep(1)
        if all_data:
            full_df = pd.concat(all_data).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
            self.cached_data[cache_key] = full_df
            print(f"Fetched {len(full_df)} candles")
            return full_df
        raise ValueError("No data retrieved")

    def execute_trades(self, df, pip_value, sl_pips, tp_pips):
        trades = []
        active_positions = []
        
        df['signal'] = df['signal'].fillna('NONE')  # Ensure no NaN signals
        
        for i in range(0, len(df)-1):
            current = df.iloc[i]
            
            if current['signal'] != 'NONE' and pd.notna(current['signal_time']) and current['signal_time'] == current['time']:
                if current['signal'] == 'BUY':
                    executed_price, fill_ratio = self.simulate_execution('BUY_STOP', current['close'], current, pip_value)
                    sl = executed_price - sl_pips * pip_value
                    tp = executed_price + tp_pips * pip_value
                    pos_type = 'LONG'
                elif current['signal'] == 'SELL':
                    executed_price, fill_ratio = self.simulate_execution('SELL_STOP', current['close'], current, pip_value)
                    sl = executed_price + sl_pips * pip_value
                    tp = executed_price - tp_pips * pip_value
                    pos_type = 'SHORT'
                
                if fill_ratio > 0:
                    entry_time_with_delay = current['time'] + timedelta(seconds=self.order_delay)
                    active_positions.append({
                        'entry_price': executed_price,
                        'sl': sl,
                        'tp': tp,
                        'type': pos_type,
                        'entry_time': entry_time_with_delay,
                        'size': fill_ratio
                    })
            
            for pos in active_positions[:]:
                if current['time'] <= pos['entry_time']:
                    continue
                
                duration = (current['time'] - pos['entry_time']).total_seconds() / 60
                commission_cost = self.commission_per_lot * pos['size']
                
                if pos['type'] == 'LONG':
                    if current['low'] <= pos['sl']:
                        pips = -sl_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['sl'], 'pips': pips,
                            'type': 'LONG', 'outcome': 'SL', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                    elif current['high'] >= pos['tp']:
                        pips = tp_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['tp'], 'pips': pips,
                            'type': 'LONG', 'outcome': 'TP', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                else:
                    if current['high'] >= pos['sl']:
                        pips = -sl_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['sl'], 'pips': pips,
                            'type': 'SHORT', 'outcome': 'SL', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                    elif current['low'] <= pos['tp']:
                        pips = tp_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['tp'], 'pips': pips,
                            'type': 'SHORT', 'outcome': 'TP', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
        
        return trades

    def simulate_execution(self, order_type, price, candle, pip_value):
        if order_type == 'BUY_STOP':
            executed_price = max(price, candle['open']) + self.slippage + self.spread
            return executed_price, 1.0 if np.random.random() >= self.partial_fill_prob else 0.5
        elif order_type == 'SELL_STOP':
            executed_price = min(price, candle['open']) - self.slippage - self.spread
            return executed_price, 1.0 if np.random.random() >= self.partial_fill_prob else 0.5
        return price, 1.0

    def evaluate_performance(self, trades):
        if not trades:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        pips = np.array([t['pips'] for t in trades])
        durations = np.array([t['duration'] for t in trades])
        total_pips = pips.sum()
        total_trades = len(trades)
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if losses.any() else float('inf')
        expectancy = pips.mean() if total_trades > 0 else 0
        sharpe_ratio = pips.mean() / pips.std() * np.sqrt(252) if pips.std() != 0 else 0  # Annualized
        sortino_ratio = pips.mean() / np.std(losses) * np.sqrt(252) if losses.std() != 0 else 0  # Annualized
        running = np.cumsum(pips)
        peak = running[0]
        max_dd = 0
        for value in running:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        avg_duration = durations.mean() if durations.any() else 0
        downside_risk = np.std(losses) if losses.any() else 0
        return total_pips, total_trades, win_rate, profit_factor, expectancy, sharpe_ratio, sortino_ratio, max_dd, avg_duration, downside_risk

# Strategy Detection Function
def detect_equal_highs_lows(data, tolerance, pip_value):
    data['signal'] = None
    data['signal_time'] = pd.Series(np.nan, index=data.index, dtype='datetime64[ns, UTC]')
    
    eq_high = (data['high'] - data['high'].shift(1)).abs() < tolerance * pip_value
    eq_low = (data['low'] - data['low'].shift(1)).abs() < tolerance * pip_value
    eq_buy = eq_low & (data['close'] > data['open']) & (data['close'].shift(-1) > data['high'])
    eq_sell = eq_high & (data['close'] < data['open']) & (data['close'].shift(-1) < data['low'])
    
    data.loc[eq_buy, 'signal'] = 'BUY'
    data.loc[eq_buy, 'signal_time'] = data.loc[eq_buy, 'time']
    data.loc[eq_sell, 'signal'] = 'SELL'
    data.loc[eq_sell, 'signal_time'] = data.loc[eq_sell, 'time']
    
    return data

# Backtest Function for Optimization
def backtest_optimization(tester, data, tolerance, sl_pips, tp_pips, pip_value):
    data_with_signals = detect_equal_highs_lows(data.copy(), tolerance, pip_value)
    trades = tester.execute_trades(data_with_signals, pip_value, sl_pips, tp_pips)
    metrics = tester.evaluate_performance(trades)
    return trades, *metrics

# Bayesian Optimization Objective
def objective(tolerance, sl_pips, tp_pips):
    global tester, data, pip_value, iteration, all_results
    trades, total_pips, total_trades, win_rate, profit_factor, expectancy, sharpe_ratio, sortino_ratio, max_dd, avg_duration, downside_risk = backtest_optimization(tester, data, tolerance, sl_pips, tp_pips, pip_value)
    iteration += 1
    objective_value = total_pips
    all_results.append({
        'iteration': iteration,
        'total_pips': total_pips,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_dd': max_dd,
        'avg_duration': avg_duration,
        'downside_risk': downside_risk,
        'objective': objective_value,
        'tolerance': tolerance,
        'sl_pips': sl_pips,
        'tp_pips': tp_pips,
        'trades': trades
    })
    print(f"\n=== Backtest Results ===")
    print(f"Total Pips: {total_pips:.1f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Expectancy: {expectancy:.2f} pips/trade")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_dd:.1f} pips")
    print(f"Avg Trade Duration: {avg_duration:.1f} minutes")
    print(f"Total Trades: {total_trades}")
    print(f"Objective (Total Pips - Penalty): {objective_value:.1f}")
    print(f"| {iteration:<9} | {objective  objective_value:.3e} | {tolerance:.2f}     | {sl_pips:.2f}     | {tp_pips:.2f}     |")
    return objective_value

# Institutional Summary Function
def institutional_summary(best_result):
    trades = best_result['trades']
    pips = np.array([t['pips'] for t in trades])
    durations = np.array([t['duration'] for t in trades])
    
    # Monthly Performance (2020-2024, aggregate by year-month)
    monthly_pips = {}
    for trade in trades:
        year_month = trade['exit_time'].strftime('%Y-%m')
        monthly_pips[year_month] = monthly_pips.get(year_month, 0) + trade['pips']
    
    # Trade Duration Buckets
    duration_buckets = {
        '<4h': len([d for d in durations if d < 240]),
        '4h-12h': len([d for d in durations if 240 <= d < 720]),
        '12h-24h': len([d for d in durations if 720 <= d < 1440]),
        '>24h': len([d for d in durations if d >= 1440])
    }
    
    print("\n=== Institutional Trader Summary (Best Parameters) ===")
    print(f"Parameters: tolerance={best_result['tolerance']:.2f}, SL={best_result['sl_pips']:.2f}, TP={best_result['tp_pips']:.2f}")
    print(f"Total Pips: {best_result['total_pips']:.1f}")
    print(f"Annualized Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Annualized Sortino Ratio: {best_result['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {best_result['max_dd']:.1f} pips")
    print(f"Downside Risk (Std Dev of Losses): {best_result['downside_risk']:.1f} pips")
    print(f"Profit Factor: {best_result['profit_factor']:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    print("\nMonthly Pip Performance (2020-2024):")
    print("| Year-Month | Pips      |")
    print("|------------|-----------|")
    for year_month, pips in sorted(monthly_pips.items()):
        print(f"| {year_month:<10} | {pips:>9.1f} |")
    
    print("\nTrade Duration Distribution:")
    print("| Range    | Trades |")
    print("|----------|--------|")
    for bucket, count in duration_buckets.items():
        print(f"| {bucket:<8} | {count:>6} |")
    
    print("\nSuggested Chart: Equity Curve")
    print("Plot cumulative pips over time using trade exit timestamps and pip profits.")
    print("X-axis: Date (2020-2024), Y-axis: Cumulative Pips")

# Run Bayesian Optimization
if __name__ == "__main__":
    instrument = "GBP_JPY"
    print(f"\nBacktesting {instrument} on H4 with Equal Highs and Lows...")
    
    # Initialize backtester and fetch data once
    tester = RealisticBacktester()
    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 12, 31, tzinfo=timezone.utc)
    data = tester.fetch_candles(instrument, "H4", start_time, end_time)
    pip_value = 0.01
    
    # Initialize global variables
    iteration = 0
    all_results = []
    
    # Define parameter bounds
    pbounds = {
        'tolerance': (5.0, 30.0),
        'sl_pips': (10.0, 50.0),
        'tp_pips': (30.0, 150.0)
    }
    
    # Run Bayesian Optimization
    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points=5, n_iter=115)  # Total 120 iterations
    
    # Display Best Parameters
    best_params = optimizer.max['params']
    best_objective = optimizer.max['target']
    best_result = max(all_results, key=lambda x: x['objective'])
    print("\n=============================================================")
    print("\n=== Best Parameters ===")
    print(f"tolerance: {best_params['tolerance']:.2f}")
    print(f"sl_pips: {best_params['sl_pips']:.2f}")
    print(f"tp_pips: {best_params['tp_pips']:.2f}")
    print(f"Best Objective: {best_objective:.1f}")
    
    # Find Top 5 by Win Rate
    sorted_by_winrate = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)[:5]
    print("\n=== Top 5 Parameters by Win Rate ===")
    for i, result in enumerate(sorted_by_winrate, 1):
        print(f"\nTop {i}:")
        print(f"tolerance: {result['tolerance']:.2f}")
        print(f"sl_pips: {result['sl_pips']:.2f}")
        print(f"tp_pips: {result['tp_pips']:.2f}")
        print(f"Total Pips: {result['total_pips']:.1f}")
        print(f"Win Rate: {result['win_rate']:.2%}")
        print(f"Objective: {result['objective']:.1f}")
    
    # Institutional Summary
    institutional_summary(best_result)

# README
# Overview:
# This script implements a realistic backtesting framework for a forex trading strategy on the GBP/JPY pair using OANDA's API. It fetches H4 candlestick data, detects trading signals based on equal highs and lows, and optimizes parameters (tolerance, stop-loss, take-profit) using Bayesian optimization. The backtester simulates real-world conditions like slippage, partial fills, spreads, and commissions. Performance metrics (pips, win rate, Sharpe ratio, etc.) are calculated, and an institutional-style summary is provided. The script is designed to run in Google Colab.
#
# Usage Instructions:
# 1. Open Google Colab (colab.research.google.com) and create a new notebook.
# 2. Install required libraries: `!pip install oandapyV20 bayesian-optimization`.
# 3. Set up your OANDA API key:
#    - Replace `OANDA_API_KEY = "YOUR_OANDA_API_KEY_HERE"` with your key.
#    - Alternatively, use Colab's input prompt: `OANDA_API_KEY = input("Enter OANDA API key: ")`.
#    - Do not hardcode credentials in the script to ensure security.
# 4. Copy and paste this code into a cell and run it using Shift + Enter.
# 5. Review outputs: backtest results for each iteration, best parameters, top 5 by win rate, and institutional summary (pips, Sharpe ratio, monthly performance, trade durations).
#
# Dependencies:
# - Python 3.x
# - Libraries: oandapyV20, pandas, numpy, bayesian-optimization
# - Note: Install `oandapyV20` and `bayesian-optimization` in Colab; pandas and numpy are pre-installed.
#
# Adapting to Your Needs:
# - Change the instrument by modifying `instrument = "GBP_JPY"` (e.g., to "EUR_USD").
# - Adjust the time frame by changing `granularity = "H4"` (e.g., to "H1", "D").
# - Modify the date range in `start_time` and `end_time` for different periods.
# - Update `pbounds` to change optimization ranges for `tolerance`, `sl_pips`, or `tp_pips`.
# - Customize the `detect_equal_highs_lows` function to implement your own signal logic.
# - Adjust `pip_value`, `slippage`, `spread`, or `commission_per_lot` in `RealisticBacktester` to match your broker’s conditions.
#
# Notes:
# - Ensure a stable internet connection, as the script fetches data from OANDA’s API.
# - The script uses a practice environment (`environment="practice"`). For live trading, update to `environment="live"` with caution.
# - Bayesian optimization runs 120 iterations, which may take time depending on data size and Colab’s resources.
# - Cache is used to avoid redundant API calls; clear `tester.cached_data` if data updates are needed.
# - For large datasets, monitor Colab’s memory usage to avoid crashes.
# - Contact for support or enhancements (e.g., adding new metrics, strategies, or visualizations).