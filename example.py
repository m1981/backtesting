import pandas as pd
import yfinance as yf
import vectorbt as vbt
import numpy as np

# Fetch GOOG data
data = yf.download('GOOG', start='2020-01-01', end='2025-01-01', progress=False)

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Get close prices
close = data['Close']

# Calculate SMAs using vectorbt
sma_fast = vbt.MA.run(close, window=10, short_name='SMA10')
sma_slow = vbt.MA.run(close, window=20, short_name='SMA20')

# Generate signals
entries = sma_fast.ma_crossed_above(sma_slow.ma)
exits = sma_fast.ma_crossed_below(sma_slow.ma)

# Create portfolio
pf = vbt.Portfolio.from_signals(
    close, 
    entries, 
    exits,
    init_cash=10000,
    fees=0.002,
    freq='1D'
)

# Print comprehensive results
print("="*60)
print("VECTORBT BACKTEST SUMMARY - SMA CROSSOVER STRATEGY")
print("="*60)
print(f"Start Date:              {close.index[0].strftime('%Y-%m-%d')}")
print(f"End Date:                {close.index[-1].strftime('%Y-%m-%d')}")
print(f"\nInitial Cash:            $10,000.00")
print(f"Final Portfolio Value:   ${pf.value().iloc[-1]:,.2f}")
print(f"Total Return:            {pf.total_return() * 100:.2f}%")

# Buy & Hold comparison
buy_hold_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
print(f"Buy & Hold Return:       {buy_hold_return:.2f}%")
print(f"Strategy vs Buy & Hold:  {(pf.total_return() * 100 - buy_hold_return):+.2f}%")

print(f"\nWin Rate:                {pf.trades.win_rate * 100:.2f}%")
print(f"Sharpe Ratio:            {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown:            {pf.max_drawdown() * 100:.2f}%")
print(f"\nTotal Trades:            {pf.trades.count}")
print(f"Winning Trades:          {pf.trades.winning.count}")
print(f"Losing Trades:           {pf.trades.losing.count}")
print(f"Profit Factor:           {pf.trades.profit_factor:.2f}")
print("="*60 + "\n")

# Show profitability
total_return_pct = pf.total_return() * 100
if total_return_pct > 0:
    print(f"✅ PROFITABLE: Strategy gained {total_return_pct:.2f}%")
    profit = pf.value().iloc[-1] - 10000
    print(f"   Profit: ${profit:,.2f}")
else:
    print(f"❌ NOT PROFITABLE: Strategy lost {total_return_pct:.2f}%")
    loss = pf.value().iloc[-1] - 10000
    print(f"   Loss: ${loss:,.2f}")

# Plot results
pf.plot().show()