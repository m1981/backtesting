import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell()

# Fetch GOOG data with proper formatting
data = yf.download('GOOG', start='2020-01-01', end='2025-01-01', progress=False)

# Handle MultiIndex columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    # Get the first level (Open, High, Low, Close, Volume)
    data.columns = data.columns.get_level_values(0)

# Select only the OHLCV columns we need
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure index name is 'Date'
data.index.name = 'Date'

# Run backtest
bt = Backtest(data, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True,
              finalize_trades=True)

output = bt.run()

# First, let's see what keys are available
print("Available metrics in output:")
print(output)
print("\n" + "="*60)
print("BACKTEST SUMMARY - SMA CROSSOVER STRATEGY")
print("="*60)
print(f"Start Date:              {data.index[0].strftime('%Y-%m-%d')}")
print(f"End Date:                {data.index[-1].strftime('%Y-%m-%d')}")
print(f"\nInitial Cash:            $10,000.00")
print(f"Final Portfolio Value:   ${output['Equity Final [$]']:,.2f}")
print(f"Total Return:            {output['Return [%]']:.2f}%")
print(f"Buy & Hold Return:       {output['Buy & Hold Return [%]']:.2f}%")

if 'Return [%]' in output and 'Buy & Hold Return [%]' in output:
    outperformance = output['Return [%]'] - output['Buy & Hold Return [%]']
    print(f"\nStrategy vs Buy & Hold:  {outperformance:+.2f}%")

print(f"\nWin Rate:                {output['Win Rate [%]']:.2f}%")
print(f"Best Day:                {output['Best Day [%]']:.2f}%")
print(f"Worst Day:               {output['Worst Day [%]']:.2f}%")
print(f"\nSharpe Ratio:            {output['Sharpe Ratio']:.2f}")
print(f"Sortino Ratio:           {output['Sortino Ratio']:.2f}")
print(f"Max Drawdown:            {output['Max. Drawdown [%]']:.2f}%")
print(f"Avg. Drawdown:           {output['Avg. Drawdown [%]']:.2f}%")
print(f"\nTotal Trades:            {output['# Trades']:.0f}")
print(f"Winning Trades:          {output['Won Trades']:.0f}")
print(f"Losing Trades:           {(output['# Trades'] - output['Won Trades']):.0f}")
print(f"Profit Factor:           {output['Profit Factor']:.2f}")
print(f"\nReturn/DD Ratio:         {output['Return/DD']:.2f}")
print("="*60 + "\n")

# Show if profitable
if output['Return [%]'] > 0:
    print(f"✅ PROFITABLE: Strategy gained {output['Return [%]']:.2f}%")
    profit = output['Equity Final [$]'] - 10000
    print(f"   Profit: ${profit:,.2f}")
else:
    print(f"❌ NOT PROFITABLE: Strategy lost {output['Return [%]']:.2f}%")
    loss = output['Equity Final [$]'] - 10000
    print(f"   Loss: ${loss:,.2f}")

bt.plot()