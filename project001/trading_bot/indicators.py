import pandas as pd
import numpy as np
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def f_ma(src, length, ma_type):
    """Calculate moving average based on the given type."""
    return src.ewm(span=length, adjust=False).mean() if ma_type == "EMA" else src.rolling(window=length).mean()

def indicator1(data: pd.DataFrame) -> pd.Series:
    logging.debug(f"Indicator 1 input data: {data}")
    sma_short = data['close'].rolling(window=5).mean()
    sma_long = data['close'].rolling(window=20).mean()
    signal = np.where(sma_short > sma_long, "buy", "sell")
    logging.info(f"Indicator 1 output signal: {signal[-1]}")
    return pd.Series(signal, index=data.index)

def indicator2(data: pd.DataFrame) -> pd.Series:
    logging.debug(f"Indicator 2 input data: {data}")
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(to_replace=0, value=np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = np.where(rsi < 30, "buy", np.where(rsi > 70, "sell", "hold"))
    logging.info(f"Indicator 2 output signal: {signal[-1]}")
    return pd.Series(signal, index=data.index)

def indicator3(data: pd.DataFrame) -> pd.Series:
    logging.debug(f"Indicator 3 input data: {data}")
    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    signal = np.where(macd > signal_line, "buy", "sell")
    logging.info(f"Indicator 3 output signal: {signal[-1]}")
    return pd.Series(signal, index=data.index)

def indicator4(data: pd.DataFrame) -> pd.Series:
    logging.debug(f"Indicator 4 input data: {data}")
    rolling_mean = data['close'].rolling(window=20).mean()
    rolling_std = data['close'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    signal = np.where(data['close'] < lower_band * 0.95, "buy",
                      np.where(data['close'] > upper_band, "sell", "hold"))
    logging.info(f"Indicator 4 output signal: {signal[-1]}")
    return pd.Series(signal, index=data.index)

def indicator5(data: pd.DataFrame) -> pd.Series:
    logging.debug(f"Indicator 5 input data: {data}")
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    moving_average = typical_price.rolling(window=14).mean()
    mean_deviation = (typical_price - moving_average).abs().rolling(window=14).mean()
    cci = (typical_price - moving_average) / (0.015 * mean_deviation.replace(to_replace=0, value=np.nan))
    signal = np.where(cci > 100, "sell", np.where(cci < -100, "buy", "hold"))
    logging.info(f"Indicator 5 output signal: {signal[-1]}")
    return pd.Series(signal, index=data.index)

def indicator6(data: pd.DataFrame, ma_lengths: list, ma_type: str) -> str:
    logging.debug(f"Indicator 6 input data: {data}")

    total_buys = 0
    total_sells = 0

    # Define cloud lengths
    clouds = [
        (ma_lengths[0][0], ma_lengths[0][1]),  # Cloud 1
        (ma_lengths[1][0], ma_lengths[1][1]),  # Cloud 2
        (ma_lengths[2][0], ma_lengths[2][1]),  # Cloud 3
        (ma_lengths[3][0], ma_lengths[3][1])   # Cloud 4
    ]

    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15

    for timeframe in timeframes:
        # Check if 'timeframe' column exists
        if 'timeframe' not in data.columns:
            logging.error("The DataFrame does not contain a 'timeframe' column.")
            return "Error: 'timeframe' column missing."

        timeframe_data = data[data['timeframe'] == timeframe].copy()

        if timeframe_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        for index, (short_len, long_len) in enumerate(clouds):
            # Calculate moving averages
            timeframe_data.loc[:, f'Cloud{index + 1}_short'] = f_ma(timeframe_data['close'], short_len, ma_type)
            timeframe_data.loc[:, f'Cloud{index + 1}_long'] = f_ma(timeframe_data['close'], long_len, ma_type)

            short_ma = timeframe_data[f'Cloud{index + 1}_short']
            long_ma = timeframe_data[f'Cloud{index + 1}_long']

            if short_ma.empty or long_ma.empty:
                logging.warning(f"Moving averages not computed for Cloud ({short_len}, {long_len}) in timeframe {timeframe}.")
                continue

            # Log cloud status
            if short_ma.iloc[-1] > long_ma.iloc[-1]:
                total_buys += 1
                logging.info(f"Cloud {index + 1} Status: Bullish for timeframe M{timeframe}")
            elif short_ma.iloc[-1] < long_ma.iloc[-1]:
                total_sells += 1
                logging.info(f"Cloud {index + 1} Status: Bearish for timeframe M{timeframe}")

    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    summary = f"{total_buys} buys and {total_sells} sells across all timeframes"
    logging.info(f"Indicator 6 output signals: {summary}")

    return summary
 

def indicator_30(data: pd.DataFrame, length: int = 70, mult: float = 1.2) -> pd.Series:
    """
    Indicator 30 calculates buy and sell signals using ZLEMA and volatility.

    Parameters:
        data (pd.DataFrame): Input data with 'timeframe', 'close', 'high', and 'low' columns.
        length (int): Look-back window for ZLEMA and ATR calculations (default: 70).
        mult (float): Multiplier for volatility (default: 1.2).

    Returns:
        pd.Series: Buy ("buy") and sell ("sell") signals aligned with the data index.
    """
    logging.debug("Indicator 30 input data received.")

    # Step 1: Filter for the 3-minute timeframe
    timeframe = 3
    timeframe_data = data[data['timeframe'] == timeframe].copy()

    if timeframe_data.empty:
        logging.warning(f"No data for timeframe: {timeframe}. Returning empty series.")
        return pd.Series(dtype='object')

    # Extract necessary data
    close_prices = timeframe_data['close'].values
    high_prices = timeframe_data['high'].values
    low_prices = timeframe_data['low'].values

    if len(close_prices) < length:
        logging.error(f"Not enough data points for the specified length: {len(close_prices)} < {length}")
        return pd.Series(dtype='object')

    # Helper Function: EMA
    def ema(data, length):
        """Exponential Moving Average calculation."""
        alpha = 2 / (length + 1)
        ema_values = np.zeros_like(data, dtype=np.float64)
        ema_values[0] = data[0]
        for i in range(1, len(data)):
            ema_values[i] = (data[i] - ema_values[i - 1]) * alpha + ema_values[i - 1]
        return ema_values

    # Step 2: ZLEMA Calculation
    def calculate_zlema(src, length):
        """Zero-Lag EMA calculation."""
        lag = math.floor((length - 1) / 2)
        adjusted_src = src + (src - np.roll(src, lag))
        adjusted_src[:lag] = src[:lag]  # Handle edge cases
        return ema(adjusted_src, length)

    # Step 3: Volatility Calculation
    def calculate_volatility(high, low, close, length, mult):
        """Volatility calculation based on highest ATR."""
        # Calculate ATR
        tr = np.maximum.reduce([
            high[1:] - low[1:], 
            np.abs(high[1:] - close[:-1]), 
            np.abs(low[1:] - close[:-1])
        ])
        atr_values = ema(tr, length)
        atr_values = np.insert(atr_values, 0, atr_values[0])  # Pad start

        # Find highest ATR over length * 3 bars
        length_multiplier = length * 3
        highest_atr = pd.Series(atr_values).rolling(window=length_multiplier, min_periods=1).max().values
        return highest_atr * mult

    # Step 4: Calculate ZLEMA and Volatility
    zlema_values = calculate_zlema(close_prices, length)
    volatility_values = calculate_volatility(high_prices, low_prices, close_prices, length, mult)

    # Step 5: Identify Bullish and Bearish Trends
    trend = np.zeros(len(close_prices), dtype=int)
    for i in range(1, len(close_prices)):
        if close_prices[i] > zlema_values[i] + volatility_values[i] and close_prices[i - 1] <= zlema_values[i - 1] + volatility_values[i - 1]:
            trend[i] = 1  # Bullish crossover
        elif close_prices[i] < zlema_values[i] - volatility_values[i] and close_prices[i - 1] >= zlema_values[i - 1] - volatility_values[i - 1]:
            trend[i] = -1  # Bearish crossunder
        else:
            trend[i] = trend[i - 1]  # Maintain last trend if no crossover

    # Step 6: Generate Buy/Sell Signals
    signals = np.where(trend > 0, "buy", "sell")

    # Align with the original DataFrame index
    logging.info(f"Indicator 30 output signal: {signals[-1]}")
    return pd.Series(signals, index=timeframe_data.index)