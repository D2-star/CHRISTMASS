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

# Indicator 14: Multi-Timeframe Fibonacci Level

def indicator14_fibonacci(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe Fibonacci Levels Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Fibonacci Indicator input data: {data}")

    def calculate_fibonacci_levels(high, low):
        """
        Calculate Fibonacci levels based on high and low prices.
        """
        return {
            'level_23_6': low + (high - low) * 0.236,
            'level_38_2': low + (high - low) * 0.382,
            'level_50_0': low + (high - low) * 0.500,
            'level_61_8': low + (high - low) * 0.618,
            'level_78_6': low + (high - low) * 0.786,
        }

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    for timeframe in timeframes:
        # Resample data for the current timeframe
        resampled_data = data.resample(f'{timeframe}T').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })

        if resampled_data.empty:
            logging.warning(f"No data for timeframe: {timeframe}. Skipping...")
            continue

        # Calculate Fibonacci levels
        fibonacci_levels = calculate_fibonacci_levels(resampled_data['high'].max(), resampled_data['low'].min())
        
        # Generate signals for the current timeframe
        if resampled_data['close'].iloc[-1] <= fibonacci_levels['level_38_2'] and total_buys < 4:  # Buy signal
            total_buys += 1
            logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
        elif resampled_data['close'].iloc[-1] >= fibonacci_levels['level_61_8'] and total_sells < 4:  # Sell signal
            total_sells += 1
            logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
        else:  # Hold
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Fibonacci Indicator output signals: {summary}")

    return summary

# Indicator 15: ICT Strategy
def indicator15_ict(data: pd.DataFrame) -> str:
    """
    ICT Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"ICT Indicator input data: {data}")

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0
    signals = []

    for timeframe in timeframes:
        # Resample data for the current timeframe
        resampled_data = data.resample(f'{timeframe}T').agg({
            'close': 'last'
        })

        signal = []
        for i in range(len(resampled_data)):
            action = "hold"
            if i > 0 and resampled_data['close'].iloc[i] > resampled_data['close'].iloc[i - 1] and total_buys < 4:  # Buy signal
                action = "buy"
                total_buys += 1
                logging.info(f"Timeframe {timeframe}: Bullish (Buy)")
            elif i > 0 and resampled_data['close'].iloc[i] < resampled_data['close'].iloc[i - 1] and total_sells < 4:  # Sell signal
                action = "sell"
                total_sells += 1
                logging.info(f"Timeframe {timeframe}: Bearish (Sell)")
            else:  # Hold
                logging.info(f"Timeframe {timeframe}: Neutral (Hold)")
            signal.append(action)

        signals.append(pd.Series(signal, index=resampled_data.index))

    # Combine signals from all timeframes
    combined_signal = signals[-1]
    for sig in signals[:-1]:
        combined_signal = combined_signal.combine(sig, lambda x, y: "buy" if x == "buy" or y == "buy" else ("sell" if x == "sell" or y == "sell" else "hold"))

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"ICT Indicator output signals: {summary}")

    return summary

# Indicator 16: Price Action Strategy
def indicator16_price_action(data: pd.DataFrame) -> str:
    """
    Price Action Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Price Action Indicator input data: {data}")

    # Define timeframes
    timeframes = [1, 3, 5, 15]  # Representing M1, M3, M5, M15
    total_buys = 0
    total_sells = 0

    signals = []

    for timeframe in timeframes:
        # Resample data for the current timeframe
        resampled_data = data.resample(f'{timeframe}T').agg({
            'close': 'last'
        })

        signal = []
        for i in range(len(resampled_data)):
            action = "hold"
            if i >= 1:
                if resampled_data['close'].iloc[i] > resampled_data['close'].iloc[i - 1] and total_buys < 4:  # Buy signal
                    action = "buy"
                    total_buys += 1
                    logging.info(f"Timeframe {timeframe}: Bullish (Buy)")
                elif resampled_data['close'].iloc[i] < resampled_data['close'].iloc[i - 1] and total_sells < 4:  # Sell signal
                    action = "sell"
                    total_sells += 1
                    logging.info(f"Timeframe {timeframe}: Bearish (Sell)")
            else:  # Hold
                logging.info(f"Timeframe {timeframe}: Neutral (Hold)")
            signal.append(action)

        signals.append(pd.Series(signal, index=resampled_data.index))

    # Combine signals from all timeframes
    combined_signal = signals[-1]
    for sig in signals[:-1]:
        combined_signal = combined_signal.combine(sig, lambda x, y: "buy" if x == "buy" or y == "buy" else ("sell" if x == "sell" or y == "sell" else "hold"))

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Price Action Indicator output signals: {summary}")

    return summary

# Indicator 17: Multi-Timeframe CCI
def indicator17_cci(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe CCI (Commodity Channel Index) Strategy
    Returns a summary of buy, sell, or hold signals across timeframes.
    """
    logging.debug(f"CCI Indicator input data: {data}")

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

        typical_price = (timeframe_data['high'] + timeframe_data['low'] + timeframe_data['close']) / 3
        sma = typical_price.rolling(window=20).mean()
        mad = (typical_price - sma).abs().rolling(window=20).mean()
        cci = (typical_price - sma) / (0.015 * mad)

        # Initialize a flag for the current timeframe
        signal_generated = False

        for i in range(len(cci)):
            if not signal_generated:
                if cci.iloc[i] > 100:  # Sell signal
                    total_sells += 1
                    logging.info(f"Timeframe M{timeframe}: Bearish (Sell) at index {i}")
                    signal_generated = True  # Prevent further signals for this timeframe
                elif cci.iloc[i] < -100:  # Buy signal
                    total_buys += 1
                    logging.info(f"Timeframe M{timeframe}: Bullish (Buy) at index {i}")
                    signal_generated = True  # Prevent further signals for this timeframe

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"CCI Indicator output signals: {summary}")

    return summary
# Indicator 18: Multi-Timeframe RSI
def indicator18_rsi(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe RSI (Relative Strength Index) Strategy
    Returns a summary of buy, sell, or hold signals across timeframes.
    """
    logging.debug(f"RSI Indicator input data: {data}")

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

        delta = timeframe_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Initialize a flag for the current timeframe
        signal_generated = False

        for i in range(len(rsi)):
            if not signal_generated:
                if rsi.iloc[i] < 30:  # Buy signal
                    total_buys += 1
                    logging.info(f"Timeframe M{timeframe}: Bullish (Buy) at index {i}")
                    signal_generated = True  # Prevent further signals for this timeframe
                elif rsi.iloc[i] > 70:  # Sell signal
                    total_sells += 1
                    logging.info(f"Timeframe M{timeframe}: Bearish (Sell) at index {i}")
                    signal_generated = True  # Prevent further signals for this timeframe

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"RSI Indicator output signals: {summary}")

    return summary
def indicator19_parabolic_sar(data: pd.DataFrame) -> str:
    """
    Multi-Timeframe Parabolic SAR Strategy
    Returns a summary of buy, sell, or hold signals across specified timeframes.
    """
    logging.debug(f"Parabolic SAR Indicator input data: {data}")

    def calculate_parabolic_sar(data, acceleration=0.02, maximum=0.2):
        """
        Calculate Parabolic SAR.
        """
        sar = [data['close'].iloc[0]]  # Initial SAR value
        ep = data['high'].iloc[0]  # Initial extreme point
        trend = 1  # 1 for uptrend, -1 for downtrend
        af = acceleration  # Acceleration factor

        for i in range(1, len(data)):
            sar.append(sar[-1] + af * (ep - sar[-1]))

            if trend == 1:  # Uptrend
                if data['low'].iloc[i] < sar[-1]:  # Trend reversal
                    trend = -1
                    sar[-1] = ep  # Set SAR to the extreme point
                    ep = data['low'].iloc[i]  # Set new extreme point
                    af = acceleration  # Reset acceleration factor
                else:
                    if data['high'].iloc[i] > ep:
                        ep = data['high'].iloc[i]
                        af = min(af + acceleration, maximum)  # Increase acceleration factor
            else:  # Downtrend
                if data['high'].iloc[i] > sar[-1]:  # Trend reversal
                    trend = 1
                    sar[-1] = ep  # Set SAR to the extreme point
                    ep = data['high'].iloc[i]  # Set new extreme point
                    af = acceleration  # Reset acceleration factor
                else:
                    if data['low'].iloc[i] < ep:
                        ep = data['low'].iloc[i]
                        af = min(af + acceleration, maximum)  # Increase acceleration factor

        return pd.Series(sar)

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

        # Calculate Parabolic SAR
        sar = calculate_parabolic_sar(timeframe_data)

        # Determine the signal for the current timeframe
        if timeframe_data['close'].iloc[-1] > sar.iloc[-1] and total_buys < 4:  # Buy signal
            total_buys += 1
            logging.info(f"Timeframe M{timeframe}: Bullish (Buy)")
        elif timeframe_data['close'].iloc[-1] < sar.iloc[-1] and total_sells < 4:  # Sell signal
            total_sells += 1
            logging.info(f"Timeframe M{timeframe}: Bearish (Sell)")
        else:
            logging.info(f"Timeframe M{timeframe}: Neutral (Hold)")

    # Log the totals
    logging.info(f"Total Buys: {total_buys}, Total Sells: {total_sells}")

    # Create a summary with counts
    summary = f"{total_buys} buys and {total_sells} sells across all timeframes."
    logging.info(f"Parabolic SAR Indicator output signals: {summary}")

    return summary
