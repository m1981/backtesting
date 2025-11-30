import streamlit as st
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import traceback

# Page configuration
st.set_page_config(
    page_title="Stock Backtesting Dashboard",
    page_icon="📊",
    layout="wide"
)


def EMA(values, n):
    """Calculate Exponential Moving Average"""
    return pd.Series(values).ewm(span=n, adjust=False).mean()


def RSI(values, n):
    """Calculate RSI indicator"""
    deltas = np.diff(values)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(values)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(values)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


def MACD(values, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    values = pd.Series(values)
    ema_fast = values.ewm(span=fast).mean()
    ema_slow = values.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


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
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        if self.rsi[-1] < self.rsi_lower and not self.position:
            self.buy()
        elif self.rsi[-1] > self.rsi_upper and self.position:
            self.position.close()


class MacdStrategy(BaseStrategy):
    """MACD Signal Line Zero Cross Strategy"""
    signal_period = 14
    fast_period = 12
    slow_period = 26

    @classmethod
    def get_name(cls) -> str:
        return "MACD Zero Cross"

    @classmethod
    def get_parameters(cls) -> dict:
        return {
            'signal_period': {'label': 'Signal Period', 'min': 5, 'max': 30, 'default': 14, 'step': 1},
            'fast_period': {'label': 'Fast EMA Period', 'min': 5, 'max': 20, 'default': 12, 'step': 1},
            'slow_period': {'label': 'Slow EMA Period', 'min': 20, 'max': 50, 'default': 26, 'step': 1}
        }

    def init(self):
        close = self.data.Close
        macd_line, signal_line, histogram = self.I(MACD, close, self.fast_period, self.slow_period, self.signal_period)
        self.signal_line = signal_line

    def next(self):
        # Buy when signal line crosses above 0
        if len(self.signal_line) >= 2:
            if self.signal_line[-2] <= 0 and self.signal_line[-1] > 0 and not self.position:
                self.buy()
            # Sell when signal line crosses below 0
            elif self.signal_line[-2] >= 0 and self.signal_line[-1] < 0 and self.position:
                self.position.close()


class StrategyRegistry:
    """Registry for all available strategies"""

    _strategies = {
        'sma_cross': SmaCrossStrategy,
        'ema_cross': EmaCrossStrategy,
        'rsi_mean_reversion': RsiStrategy,
        'macd_zero_cross': MacdStrategy
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
    print(f"[DEBUG] Fetching data for {symbol} from {start} to {end}")
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        print(f"[DEBUG] Downloaded {len(data)} rows for {symbol}")

        if data.empty:
            print(f"[DEBUG] Empty data for {symbol}")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data.index.name = 'Date'
        print(f"[DEBUG] Successfully formatted data for {symbol}")
        return data
    except Exception as e:
        print(f"[DEBUG] Error fetching {symbol}: {e}")
        print(traceback.format_exc())
        st.error(f"Error fetching {symbol}: {e}")
        return None


def run_backtest_for_symbol(symbol, data, strategy_class, strategy_params,
                            cash, commission):
    """Run backtest for a single symbol with dynamic strategy"""
    print(f"[DEBUG] Starting backtest for {symbol}")
    print(f"[DEBUG] Strategy: {strategy_class.__name__}")
    print(f"[DEBUG] Parameters: {strategy_params}")
    print(f"[DEBUG] Cash: {cash}, Commission: {commission}")

    try:
        # Configure strategy parameters
        print(f"[DEBUG] Configuring strategy parameters...")
        strategy_class.configure_parameters(**strategy_params)

        # Verify parameters were set
        for param, value in strategy_params.items():
            actual_value = getattr(strategy_class, param, None)
            print(f"[DEBUG] Parameter {param}: expected={value}, actual={actual_value}")

        # Run backtest
        print(f"[DEBUG] Creating Backtest object...")
        bt = Backtest(data, strategy_class, cash=cash, commission=commission)

        print(f"[DEBUG] Running backtest...")
        result = bt.run()
        print(f"[DEBUG] Backtest complete. Return: {result['Return [%]']:.2f}%")

        # Generate chart HTML
        print(f"[DEBUG] Generating chart...")
        temp_dir = tempfile.gettempdir()
        chart_path = os.path.join(temp_dir, f'{symbol}_backtest.html')
        print(f"[DEBUG] Chart path: {chart_path}")

        bt.plot(filename=chart_path, open_browser=False)
        print(f"[DEBUG] Chart saved")

        with open(chart_path, 'r') as f:
            chart_html = f.read()
        print(f"[DEBUG] Chart HTML loaded, length: {len(chart_html)}")

        return result, chart_html

    except Exception as e:
        print(f"[DEBUG] ERROR in backtest for {symbol}: {e}")
        print(traceback.format_exc())
        st.error(f"Error running backtest for {symbol}: {e}")
        return None, None


def main():
    print("\n[DEBUG] ========== MAIN FUNCTION STARTED ==========")

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
        print(f"[DEBUG] Symbols: {symbols}")

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

        print(f"[DEBUG] Date range: {start_date} to {end_date}")

        st.markdown("---")

        # Strategy selection
        st.subheader("Strategy Selection")
        available_strategies = StrategyRegistry.get_all_strategies()
        print(f"[DEBUG] Available strategies: {available_strategies}")

        selected_strategy_key = st.selectbox(
            "Choose Strategy",
            options=list(available_strategies.keys()),
            format_func=lambda x: available_strategies[x],
            index=0
        )
        print(f"[DEBUG] Selected strategy key: {selected_strategy_key}")

        selected_strategy_class = StrategyRegistry.get_strategy(selected_strategy_key)
        print(f"[DEBUG] Selected strategy class: {selected_strategy_class}")

        strategy_params = selected_strategy_class.get_parameters()
        print(f"[DEBUG] Strategy parameters config: {strategy_params}")

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

        print(f"[DEBUG] Parameter values from UI: {param_values}")

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

        print(f"[DEBUG] Initial cash: {initial_cash}, Commission: {commission}")

        st.markdown("---")

        # Run button
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
        print(f"[DEBUG] Run button clicked: {run_button}")

    # Main content area
    if not run_button:
        print("[DEBUG] Run button NOT clicked - showing info page")
        st.info("👈 Configure your backtest parameters in the sidebar and click 'Run Backtest'")

        # Display example metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Symbols", len(symbols))
        col2.metric("Initial Capital", f"${initial_cash:,.0f}")
        col3.metric("Commission", f"{commission * 100:.2f}%")
        col4.metric("Strategy", available_strategies[selected_strategy_key])

        st.markdown("---")
        st.markdown(f"""
        ### About This App
        This backtesting dashboard allows you to:
        - Test multiple stock symbols simultaneously
        - Choose from multiple trading strategies
        - Customize strategy parameters
        - View interactive charts from backtesting.py
        - Compare strategy performance vs Buy & Hold
        - Analyze key performance metrics

        ### Current Strategy: {available_strategies[selected_strategy_key]}
        {selected_strategy_class.__doc__}
        """)

    else:
        print("[DEBUG] ========== BACKTEST EXECUTION STARTED ==========")

        # Validate parameters for MA strategies
        if selected_strategy_key in ['sma_cross', 'ema_cross']:
            print(f"[DEBUG] Validating MA parameters: n1={param_values['n1']}, n2={param_values['n2']}")
            if param_values['n1'] >= param_values['n2']:
                print("[DEBUG] VALIDATION FAILED: n1 >= n2")
                st.error("❌ Fast MA/EMA must be less than Slow MA/EMA. Please adjust parameters.")
                return
            print("[DEBUG] Validation passed")

        # Progress tracking
        print("[DEBUG] Creating progress tracking...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = {}
        chart_htmls = {}

        # Get maximum period needed for validation
        max_period = max(param_values.values())
        print(f"[DEBUG] Max period needed: {max_period}")

        # Run backtests for each symbol
        print(f"[DEBUG] Starting loop for {len(symbols)} symbols")
        for i, symbol in enumerate(symbols):
            print(f"\n[DEBUG] ===== Processing symbol {i + 1}/{len(symbols)}: {symbol} =====")
            status_text.text(f"Processing {symbol}... ({i + 1}/{len(symbols)})")

            # Fetch data
            print(f"[DEBUG] Fetching data for {symbol}...")
            data = fetch_and_format_data(symbol, start_date, end_date)

            if data is not None:
                print(f"[DEBUG] Data fetched successfully. Rows: {len(data)}, Max period: {max_period}")

                if len(data) > max_period:
                    print(f"[DEBUG] Sufficient data. Running backtest...")
                    # Run backtest
                    result, chart_html = run_backtest_for_symbol(
                        symbol, data, selected_strategy_class, param_values,
                        initial_cash, commission
                    )

                    if result is not None:
                        print(f"[DEBUG] Backtest successful for {symbol}")
                        results[symbol] = result
                        chart_htmls[symbol] = chart_html
                    else:
                        print(f"[DEBUG] Backtest returned None for {symbol}")
                else:
                    print(f"[DEBUG] Insufficient data: {len(data)} <= {max_period}")
                    st.warning(f"⚠️ Insufficient data for {symbol} (need > {max_period}, got {len(data)})")
            else:
                print(f"[DEBUG] Data fetch returned None for {symbol}")

            progress_bar.progress((i + 1) / len(symbols))

        status_text.empty()
        progress_bar.empty()

        print(f"[DEBUG] Backtest loop complete. Results count: {len(results)}")

        if not results:
            print("[DEBUG] NO RESULTS - showing error")
            st.error("❌ No successful backtests. Please check your symbols and date range.")
            return

        print("[DEBUG] ========== DISPLAYING RESULTS ==========")

        # Display summary metrics
        st.header("📈 Performance Summary")

        # Calculate overall statistics
        winners = sum(1 for r in results.values()
                      if r['Return [%]'] > r['Buy & Hold Return [%]'])
        total = len(results)
        success_rate = (winners / total * 100) if total > 0 else 0

        print(f"[DEBUG] Winners: {winners}/{total}, Success rate: {success_rate:.1f}%")

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
        print("[DEBUG] Creating summary table...")

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
        print(f"[DEBUG] Summary DataFrame created with {len(summary_df)} rows")

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
        print("[DEBUG] Displaying individual charts...")

        for symbol in results.keys():
            print(f"[DEBUG] Rendering chart for {symbol}")
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
                    st.components.v1.html(chart_htmls[symbol], height=730, scrolling=True)

                st.markdown("---")
            else:
                print(f"[DEBUG] No chart HTML for {symbol}")

        print("[DEBUG] ========== MAIN FUNCTION COMPLETE ==========\n")


if __name__ == "__main__":
    main()