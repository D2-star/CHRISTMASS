import pandas as pd
import numpy as np
import logging
import ta  # Make sure to install the 'ta' library via pip

# Configure logging
logging.basicConfig(level=logging.INFO)

# Example Data Generation
def generate_sample_data(num_rows=100):
    np.random.seed(0)
    dates = pd.date_range(start='2022-01-01', periods=num_rows, freq='min')  # 1-minute frequency
    data = {
        'date': dates,
        'high': np.random.rand(num_rows) * 100 + 100,  # Random high prices
        'low': np.random.rand(num_rows) * 100,  # Random low prices
        'close': np.random.rand(num_rows) * 100 + 50  # Random close prices
    }
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)  # Set 'date' as the index
    return df

# Indicator 8: Multi-Timeframe MACD Scalping Strategy
def indicator8_macd(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe MACD Scalping Strategy
    Returns a summary of buy, sell, or hold signals across timeframes.
    """
    logging.debug(f"MACD Indicator input data: {data}")

    def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line, signal_line

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        macd, signal = calculate_macd(timeframe_data)

        buy_signal = macd.iloc[-1] > signal.iloc[-1] and total_buys < 4
        sell_signal = macd.iloc[-1] < signal.iloc[-1] and total_sells < 4

        if buy_signal:
            total_buys += 1
            logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
        elif sell_signal:
            total_sells += 1
            logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
        else:
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"MACD Indicator output signals: {summary}")

    return summary
# Indicator 9: Multi-Timeframe ADX Scalping Strategy

def indicator9_adx(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe ADX Scalping Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"ADX Indicator input data: {data}")

    def calculate_adx(data, period=14):
        """
        Calculate ADX, plus DI, and minus DI.
        """
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        adx = (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=period).mean() * 100

        return adx, plus_di, minus_di

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        # Check if the 'timeframe' column exists
        if 'timeframe' not in data.columns:
            logging.error("The DataFrame does not contain a 'timeframe' column.")
            return "Error: 'timeframe' column missing."

        # Filter data for this timeframe
        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        # Calculate ADX, plus DI, and minus DI
        adx, plus_di, minus_di = calculate_adx(timeframe_data)

        # Determine the signal for the current timeframe
        if adx.iloc[-1] > 25:  # ADX indicates a strong trend
            if plus_di.iloc[-1] > minus_di.iloc[-1] and total_buys < 4:  # Bullish trend
                total_buys += 1
                logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
            elif minus_di.iloc[-1] > plus_di.iloc[-1] and total_sells < 4:  # Bearish trend
                total_sells += 1
                logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
            else:
                logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")
        else:
            logging.info(f"Timeframe M{timeframe}: Weak Trend (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"ADX Indicator output signals: {summary}")

    return summary
# Indicator 10: Multi-Timeframe Bollinger Bands
# Indicator 10: Multi-Timeframe Bollinger Bands
def indicator10_bollinger_bands(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe Bollinger Bands Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Bollinger Bands Indicator input data: {data}")

    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands: upper, lower, and middle bands.
        """
        middle_band = data['close'].rolling(window=period).mean()
        upper_band = middle_band + (data['close'].rolling(window=period).std() * std_dev)
        lower_band = middle_band - (data['close'].rolling(window=period).std() * std_dev)
        return upper_band, lower_band, middle_band

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        # Check if the 'timeframe' column exists
        if 'timeframe' not in data.columns:
            logging.error("The DataFrame does not contain a 'timeframe' column.")
            return "Error: 'timeframe' column missing."

        # Filter data for this timeframe
        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        # Calculate Bollinger Bands
        upper_band, lower_band, middle_band = calculate_bollinger_bands(timeframe_data)

        # Initialize a flag for the current timeframe
        signal_generated = False

        # Determine the signal for the current timeframe
        if not signal_generated and timeframe_data['close'].iloc[-1] < lower_band.iloc[-1]:  # Oversold
            total_buys += 1
            logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
            signal_generated = True
        elif not signal_generated and timeframe_data['close'].iloc[-1] > upper_band.iloc[-1]:  # Overbought
            total_sells += 1
            logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
            signal_generated = True
        else:
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Bollinger Bands Indicator output signals: {summary}")

    return summary
# Indicator 11: Multi-Timeframe Stochastic Oscillator
# Indicator 11: Multi-Timeframe Stochastic Oscillator
def indicator11_stochastic(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe Stochastic Oscillator Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Stochastic Indicator input data: {data}")

    def calculate_stochastic(data, k_period=14, d_period=3):
        """
        Calculate Stochastic %K and %D.
        """
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        stoch_k = ((data['close'] - low_min) / (high_max - low_min)) * 100
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        # Check if the 'timeframe' column exists
        if 'timeframe' not in data.columns:
            logging.error("The DataFrame does not contain a 'timeframe' column.")
            return "Error: 'timeframe' column missing."

        # Filter data for this timeframe
        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        # Calculate Stochastic components
        stoch_k, stoch_d = calculate_stochastic(timeframe_data)

        # Initialize a flag for the current timeframe
        signal_generated = False

        # Determine the signal for the current timeframe
        if not signal_generated and stoch_k.iloc[-1] < 20 and stoch_k.iloc[-1] > stoch_d.iloc[-1]:  # Bullish signal
            total_buys += 1
            logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
            signal_generated = True
        elif not signal_generated and stoch_k.iloc[-1] > 80 and stoch_k.iloc[-1] < stoch_d.iloc[-1]:  # Bearish signal
            total_sells += 1
            logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
            signal_generated = True
        else:
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Stochastic Indicator output signals: {summary}")

    return summary
# Indicator 13: Multi-Timeframe Donchian Channel
# Indicator 13: Multi-Timeframe Donchian Channel
def indicator13_donchian(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe Donchian Channel Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Donchian Channel Indicator input data: {data}")

    def calculate_donchian(data, period):
        """
        Calculate Donchian Channels: upper band, lower band, and middle line.
        """
        upper_band = data['high'].rolling(window=period).max()
        lower_band = data['low'].rolling(window=period).min()
        middle_line = (upper_band + lower_band) / 2
        return upper_band, lower_band, middle_line

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        # Check if the 'timeframe' column exists
        if 'timeframe' not in data.columns:
            logging.error("The DataFrame does not contain a 'timeframe' column.")
            return "Error: 'timeframe' column missing."

        # Filter data for this timeframe
        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        # Calculate Donchian Channels
        upper_band, lower_band, middle_line = calculate_donchian(timeframe_data, 20)

        # Initialize a flag for the current timeframe
        signal_generated = False

        # Determine the signal for the current timeframe
        if not signal_generated and timeframe_data['close'].iloc[-1] > upper_band.iloc[-1]:  # Buy signal
            if timeframe_data['close'].iloc[-1] > middle_line.iloc[-1]:  # Confirm momentum
                total_buys += 1
                logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
                signal_generated = True
        elif not signal_generated and timeframe_data['close'].iloc[-1] < lower_band.iloc[-1]:  # Sell signal
            if timeframe_data['close'].iloc[-1] < middle_line.iloc[-1]:  # Confirm momentum
                total_sells += 1
                logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
                signal_generated = True
        else:
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Donchian Channel Indicator output signals: {summary}")

    return summary
