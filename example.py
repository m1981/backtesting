import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

def fetch_and_format_data(symbol, start='2020-01-01', end='2025-01-01'):
    """Fetch and format data for backtesting"""
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.index.name = 'Date'
    return data

def run_multi_symbol_backtest(symbols, strategy_class=SmaCross, cash=10000, commission=0.002):
    """Run backtest on multiple symbols and display results"""
    results = {}
    
    # Create figure with subplots
    n_symbols = len(symbols)
    fig = plt.figure(figsize=(15, 4 * n_symbols))
    gs = GridSpec(n_symbols, 2, figure=fig, width_ratios=[3, 1])
    
    print("="*80)
    print("MULTI-SYMBOL BACKTEST RESULTS")
    print("="*80)
    
    for i, symbol in enumerate(symbols):
        try:
            # Fetch data
            data = fetch_and_format_data(symbol)
            
            # Run backtest
            bt = Backtest(data, strategy_class, cash=cash, commission=commission)
            result = bt.run()
            results[symbol] = result
            
            # Create subplot for equity curve
            ax1 = fig.add_subplot(gs[i, 0])
            equity_curve = result._equity_curve
            ax1.plot(equity_curve.index, equity_curve['Equity'], label=f'{symbol} Portfolio', linewidth=2)
            ax1.set_title(f'{symbol} - Equity Curve', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Create subplot for key metrics
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.axis('off')
            
            # Key metrics text
            metrics_text = f"""
Return: {result['Return [%]']:.1f}%
Buy & Hold: {result['Buy & Hold Return [%]']:.1f}%
Sharpe: {result['Sharpe Ratio']:.2f}
Max DD: {result['Max. Drawdown [%]']:.1f}%
Trades: {result['# Trades']}
Win Rate: {result['Win Rate [%]']:.1f}%
            """.strip()
            
            ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
            
            # Print summary
            print(f"\n{symbol}:")
            print(f"  Return: {result['Return [%]']:.1f}% | Buy&Hold: {result['Buy & Hold Return [%]']:.1f}%")
            print(f"  Sharpe: {result['Sharpe Ratio']:.2f} | Max DD: {result['Max. Drawdown [%]']:.1f}%")
            print(f"  Trades: {result['# Trades']} | Win Rate: {result['Win Rate [%]']:.1f}%")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    plt.tight_layout()
    plt.show()
    
    return results

# Example usage
symbols = ['GOOG', 'AAPL', 'MSFT', 'TSLA']
results = run_multi_symbol_backtest(symbols)

# Optional: Create summary comparison table
summary_df = pd.DataFrame({
    symbol: {
        'Return [%]': result['Return [%]'],
        'Buy & Hold [%]': result['Buy & Hold Return [%]'],
        'Sharpe Ratio': result['Sharpe Ratio'],
        'Max Drawdown [%]': result['Max. Drawdown [%]'],
        'Win Rate [%]': result['Win Rate [%]'],
        '# Trades': result['# Trades']
    }
    for symbol, result in results.items()
}).T

print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)
print(summary_df.round(2))
