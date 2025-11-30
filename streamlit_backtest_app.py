import streamlit as st
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import tempfile
import os
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Page configuration
st.set_page_config(
    page_title="Stock Backtesting Dashboard",
    page_icon="📊",
    layout="wide"
)


class BaseStrategy(Strategy, ABC):
    """Abstract base class for all trading strategies"""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return strategy display name"""
        pass

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> dict:
        """Return strategy parameters for UI configuration"""
        pass

    @classmethod
    def configure_parameters(cls, **params):
        """Configure strategy parameters dynamically"""
        for param, value in params.items():
            if hasattr(cls, param):
                setattr(cls, param, value)


class SmaCrossStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    n1 = 10  # Fast MA
    n2 = 20  # Slow MA

    @classmethod
    def get_name(cls) -> str:
        return "SMA Crossover"

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            'n1': {'label': 'Fast MA Period', 'min': 5, 'max': 50, 'default': 10, 'step': 1},
            'n2': {'label': 'Slow MA Period', 'min': 10, 'max': 200, 'default': 20, 'step': 5}
        }

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

class EmaCrossStrategy(BaseStrategy):
    """Exponential Moving Average Crossover Strategy"""
    n1 = 12
    n2 = 26

    @classmethod
    def get_name(cls) -> str:
        return "EMA Crossover"

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            'n1': {'label': 'Fast EMA Period', 'min': 5, 'max': 50, 'default': 12, 'step': 1},
            'n2': {'label': 'Slow EMA Period', 'min': 10, 'max': 200, 'default': 26, 'step': 1}
        }

    def init(self):
        close = self.data.Close
        self.ema1 = self.I(EMA, close, self.n1)
        self.ema2 = self.I(EMA, close, self.n2)

    def next(self):
        if crossover(self.ema1, self.ema2):
            self.position.close()
            self.buy()
        elif crossover(self.ema2, self.ema1):
            self.position.close()
            self.sell()

class RsiStrategy(BaseStrategy):
    """RSI Mean Reversion Strategy"""
    rsi_period = 14
    rsi_upper = 70
    rsi_lower = 30

    @classmethod
    def get_name(cls) -> str:
        return "RSI Mean Reversion"

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            'rsi_period': {'label': 'RSI Period', 'min': 5, 'max': 50, 'default': 14, 'step': 1},
            'rsi_upper': {'label': 'RSI Upper Threshold', 'min': 60, 'max': 90, 'default': 70, 'step': 5},
            'rsi_lower': {'label': 'RSI Lower Threshold', 'min': 10, 'max': 40, 'default': 30, 'step': 5}
        }

    def init(self):
        def rsi(arr, n):
            return talib.RSI(arr, timeperiod=n)

        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)

    def next(self):
        if self.rsi[-1] < self.rsi_lower and not self.position:
            self.buy()
        elif self.rsi[-1] > self.rsi_upper and self.position:
            self.sell()


class StrategyRegistry:
    """Registry for all available strategies"""

    _strategies = {
        'sma_cross': SmaCrossStrategy,
        'ema_cross': EmaCrossStrategy,
        'rsi_mean_reversion': RsiStrategy
    }

    @classmethod
    def get_all_strategies(cls) -> dict:
        return {key: strategy.get_name() for key, strategy in cls._strategies.items()}

    @classmethod
    def get_strategy(cls, strategy_key: str) -> BaseStrategy:
        return cls._strategies.get(strategy_key)

    @classmethod
    def register_strategy(cls, key: str, strategy_class: BaseStrategy):
        cls._strategies[key] = strategy_class


@st.cache_data
def fetch_and_format_data(symbol, start, end):
    """Fetch and format data for backtesting"""
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.index.name = 'Date'
        return data
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return None


def run_backtest_for_symbol(symbol, data, strategy_class, strategy_params,
                            cash, commission):
    """Run backtest for a single symbol with dynamic strategy"""
    try:
        # Configure strategy parameters
        strategy_class.configure_parameters(**strategy_params)

        # Run backtest
        bt = Backtest(data, strategy_class, cash=cash, commission=commission)
        result = bt.run()

        # Generate chart HTML
        temp_dir = tempfile.gettempdir()
        chart_path = os.path.join(temp_dir, f'{symbol}_backtest.html')
        bt.plot(filename=chart_path, open_browser=False)

        with open(chart_path, 'r') as f:
            chart_html = f.read()

        return result, chart_html

    except Exception as e:
        st.error(f"Error running backtest for {symbol}: {e}")
        return None, None


def main():
    st.title("📊 Multi-Symbol Stock Backtesting Dashboard")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Symbol input
        st.subheader("Symbols")
        symbols_input = st.text_area(
            "Enter ticker symbols (one per line)",
            value="AAPL\nMSFT\nGOOG\nTSLA",
            height=100
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]

        st.markdown("---")

        # Date range
        st.subheader("Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime(2020, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )

        st.markdown("---")

        # Strategy selection
        st.subheader("Strategy Selection")
        available_strategies = StrategyRegistry.get_all_strategies()
        selected_strategy_key = st.selectbox(
            "Choose Strategy",
            options=list(available_strategies.keys()),
            format_func=lambda x: available_strategies[x],
            index=0
        )

        selected_strategy_class = StrategyRegistry.get_strategy(selected_strategy_key)
        strategy_params = selected_strategy_class.get_parameters()

        st.markdown("---")

        # Dynamic strategy parameters
        st.subheader("Strategy Parameters")
        param_values = {}

        for param_name, param_config in strategy_params.items():
            param_values[param_name] = st.slider(
                param_config['label'],
                min_value=param_config['min'],
                max_value=param_config['max'],
                value=param_config['default'],
                step=param_config['step']
            )

        st.markdown("---")

        # Backtest parameters
        st.subheader("Backtest Settings")
        initial_cash = st.number_input(
            "Initial Cash ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )

        commission = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            format="%.2f"
        ) / 100

        st.markdown("---")

        # Run button
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    # Main content area
    if not run_button:
        st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest'")

        # Display example metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbols", len(symbols))
        col2.metric("Initial Capital", f"${initial_cash:,.0f}")
        col3.metric("Commission", f"{commission * 100:.2f}%")
        col4.metric("Strategy", "SMA Cross")

        st.markdown("---")
        st.markdown("""
        ### About This App
        This backtesting dashboard allows you to:
        - Test multiple stock symbols simultaneously
        - Customize Moving Average strategy parameters
        - View interactive charts from backtesting.py
        - Compare strategy performance vs Buy & Hold
        - Analyze key performance metrics

        ### Strategy: Simple Moving Average Crossover
        - **Buy Signal**: Fast MA crosses above Slow MA
        - **Sell Signal**: Slow MA crosses above Fast MA
        """)

    else:
        if fast_ma >= slow_ma:
            st.error("❌ Fast MA must be less than Slow MA. Please adjust parameters.")
            return

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = {}
        chart_htmls = {}

        # Run backtests for each symbol
        for i, symbol in enumerate(symbols):
            status_text.text(f"Processing {symbol}... ({i + 1}/{len(symbols)})")

            # Fetch data
            data = fetch_and_format_data(symbol, start_date, end_date)

            if data is not None and len(data) > slow_ma:
                # Run backtest
                result, chart_html = run_backtest_for_symbol(
                    symbol, data, SmaCross, initial_cash, commission,
                    fast_ma, slow_ma
                )

                if result is not None:
                    results[symbol] = result
                    chart_htmls[symbol] = chart_html
            else:
                st.warning(f"⚠️ Insufficient data for {symbol}")

            progress_bar.progress((i + 1) / len(symbols))

        status_text.empty()
        progress_bar.empty()

        if not results:
            st.error("❌ No successful backtests. Please check your symbols and date range.")
            return

        # Display summary metrics
        st.header("📈 Performance Summary")

        # Calculate overall statistics
        winners = sum(1 for r in results.values()
                      if r['Return [%]'] > r['Buy & Hold Return [%]'])
        total = len(results)
        success_rate = (winners / total * 100) if total > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Symbols", total)
        col2.metric("Outperformed", f"{winners}/{total}")
        col3.metric("Success Rate", f"{success_rate:.1f}%")

        avg_return = sum(r['Return [%]'] for r in results.values()) / len(results)
        col4.metric("Avg Return", f"{avg_return:.2f}%",
                    delta=f"{avg_return:.2f}%")

        st.markdown("---")

        # Summary table
        st.subheader("📊 Detailed Results Table")

        summary_data = []
        for symbol, result in results.items():
            strategy_return = result['Return [%]']
            buy_hold_return = result['Buy & Hold Return [%]']
            diff = strategy_return - buy_hold_return

            summary_data.append({
                'Symbol': symbol,
                'Strategy Return (%)': strategy_return,
                'Buy & Hold (%)': buy_hold_return,
                'Difference (%)': diff,
                'Outperformed': '✓' if diff > 0 else '✗',
                'Sharpe Ratio': result['Sharpe Ratio'],
                'Max Drawdown (%)': result['Max. Drawdown [%]'],
                'Win Rate (%)': result['Win Rate [%]'],
                '# Trades': result['# Trades']
            })

        summary_df = pd.DataFrame(summary_data)

        # Style the dataframe
        def highlight_performance(row):
            if row['Outperformed'] == '✓':
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #f8d7da'] * len(row)

        styled_df = summary_df.style.apply(highlight_performance, axis=1) \
            .format({
            'Strategy Return (%)': '{:.2f}',
            'Buy & Hold (%)': '{:.2f}',
            'Difference (%)': '{:+.2f}',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown (%)': '{:.2f}',
            'Win Rate (%)': '{:.1f}'
        })

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Download button for CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary CSV",
            data=csv,
            file_name=f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # Display individual charts
        st.header("📊 Individual Backtest Charts")
        st.markdown("Interactive charts generated by backtesting.py")

        for symbol in results.keys():
            if symbol in chart_htmls:
                result = results[symbol]
                strategy_return = result['Return [%]']
                buy_hold_return = result['Buy & Hold Return [%]']
                outperformed = strategy_return > buy_hold_return

                # Symbol header with metrics
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

                with col1:
                    status_icon = "✅" if outperformed else "❌"
                    st.subheader(f"{status_icon} {symbol}")

                with col2:
                    st.metric("Strategy Return", f"{strategy_return:.2f}%")

                with col3:
                    st.metric("Buy & Hold", f"{buy_hold_return:.2f}%")

                with col4:
                    diff = strategy_return - buy_hold_return
                    st.metric("Difference", f"{diff:+.2f}%")

                with col5:
                    st.metric("Sharpe Ratio", f"{result['Sharpe Ratio']:.2f}")

                # Display the chart
                with st.expander(f"View {symbol} Chart", expanded=True):
                    # Embed the HTML chart with proper height
                    st.components.v1.html(chart_htmls[symbol], height=630, scrolling=True)

                st.markdown("---")


if __name__ == "__main__":
    main()