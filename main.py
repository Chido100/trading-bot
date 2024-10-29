from fastapi import FastAPI, HTTPException
from binance.client import Client
from binance.streams import BinanceSocketManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from datetime import datetime, timedelta
import asyncio
import logging
from decouple import config
import math
import time



# Initialize FastAPI app
app = FastAPI()

# Binance API credentials (replace with your actual keys)
BINANCE_API_KEY = config("BINANCE_API_KEY")
BINANCE_SECRET_KEY = config("BINANCE_SECRET_KEY")

# Initialize Binance REST API client
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

# Binance WebSocket stream manager
bsm = BinanceSocketManager(binance_client)

# Logger setup
logging.basicConfig(filename='crypto_trade_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Global variables
real_time_data = {}
active_orders = {}
model = RandomForestClassifier()

# Utility function: Calculate RSI
def calculate_rsi(prices, window=14):
    delta = pd.Series(prices).diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Utility function: Calculate MACD
def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = pd.Series(prices).ewm(span=short_window, adjust=False).mean()
    long_ema = pd.Series(prices).ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

# Utility function: Calculate Bollinger Bands
def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    sma = pd.Series(prices).rolling(window=window).mean()
    std = pd.Series(prices).rolling(window=window).std()
    upper_band = sma + (num_std_dev * std)
    lower_band = sma - (num_std_dev * std)
    return sma, upper_band, lower_band

# Machine Learning: Train model on historical data
def train_model(symbol: str, data: pd.DataFrame):
    data['RSI'] = calculate_rsi(data['close'])
    data['MACD'], data['MACD_signal'] = calculate_macd(data['close'])
    _, data['upper_band'], data['lower_band'] = calculate_bollinger_bands(data['close'])
    
    data.dropna(inplace=True)  # Remove rows with missing values

    # Prepare features and target
    features = ['RSI', 'MACD', 'MACD_signal', 'upper_band', 'lower_band']
    X = data[features]
    y = (data['close'].shift(-1) > data['close']).astype(int)  # 1 for buy, 0 for sell

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)  # Train the model
    
    # Validate the model with test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained for {symbol} with accuracy: {accuracy}")

    return model

# Machine Learning: Predict trade action using trained model
def predict_trade_action(rsi, macd, macd_signal, upper_band, lower_band):
    try:
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[-1]
        if isinstance(macd, pd.Series):
            macd = macd.iloc[-1]
        if isinstance(macd_signal, pd.Series):
            macd_signal = macd_signal.iloc[-1]
        if isinstance(upper_band, pd.Series):
            upper_band = upper_band.iloc[-1]
        if isinstance(lower_band, pd.Series):
            lower_band = lower_band.iloc[-1]

        # Log the indicators before making a prediction
        logging.info(f"Predicting with features - RSI: {rsi}, MACD: {macd}, MACD_signal: {macd_signal}, Upper Band: {upper_band}, Lower Band: {lower_band}")

        # Prepare the feature dataframe
        features = pd.DataFrame([[rsi, macd, macd_signal, upper_band, lower_band]], 
                                columns=['RSI', 'MACD', 'MACD_signal', 'upper_band', 'lower_band'])

        # Make the prediction using the trained model
        prediction = model.predict(features)[0]

        # Debugging: Force a "buy" if the RSI is not extremely high, and MACD is positive
        if prediction == 0 and rsi < 75 and macd > macd_signal:
            logging.warning("Adjusting prediction to BUY based on MACD trend.")
            prediction = 1

        logging.info(f"Prediction result: {prediction} (1 for buy, 0 for sell)")
        return prediction  # 1 for buy, 0 for sell

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise



def calculate_position_size(symbol: str, price: float):
    try:
        # Retry fetching symbol information up to 3 times if necessary
        symbol_info = None
        for attempt in range(3):
            symbol_info = binance_client.get_symbol_info(symbol)
            if symbol_info:
                break
            time.sleep(1)  # Wait 1 second before retrying
        
        if not symbol_info:
            raise ValueError(f"Failed to retrieve symbol information for {symbol} after multiple attempts.")
        
        # Log the retrieved symbol_info for debugging
        logging.info(f"Retrieved symbol information for {symbol}: {symbol_info}")

        # Get filters for step size and minimum notional value
        lot_size_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        min_notional_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'NOTIONAL' or f.get('filterType') == 'MIN_NOTIONAL'), None)
        
        # Ensure filters are found, and log if they are missing
        if not lot_size_filter:
            logging.error(f"LOT_SIZE filter not found for {symbol}. Full info: {symbol_info}")
            return None
        
        if not min_notional_filter:
            logging.error(f"NOTIONAL filter not found for {symbol}. Full info: {symbol_info}")
            return None

        step_size = float(lot_size_filter['stepSize'])
        min_notional = float(min_notional_filter.get('minNotional', 0))

        # Retrieve the current USDT balance
        balance_info = binance_client.get_asset_balance(asset='USDT')
        usdt_balance = float(balance_info['free'])

        # Calculate the raw position size based on 10% of the USDT balance
        trade_percentage = 0.10  # Adjusted to 10% for testing
        raw_position_size = (usdt_balance * trade_percentage) / price

        # Ensure the trade meets the minimum notional requirement
        if raw_position_size * price < min_notional:
            logging.warning(f"Raw position size {raw_position_size} does not meet the minimum notional value.")
            raw_position_size = min_notional / price  # Adjust to meet minimum notional

        # Round to the nearest valid step size and format the position size
        precision = int(round(-math.log(step_size, 10)))
        position_size = round(raw_position_size, precision)

        # Re-check if the adjusted position size is still valid
        if position_size * price < min_notional:
            logging.error(f"Adjusted position size {position_size} does not meet the minimum notional requirement. Insufficient balance to proceed.")
            return None

        logging.info(f"Calculated position size for {symbol}: {position_size}")
        return position_size

    except Exception as e:
        logging.error(f"Error calculating position size for {symbol}: {str(e)}")
        raise



# Utility function: Update indicators with advanced options
def update_indicators(symbol: str, price: float):
    if symbol not in real_time_data:
        klines = binance_client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=26)
        historical_prices = [float(kline[4]) for kline in klines]  # Closing prices
        real_time_data[symbol] = historical_prices

        data = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        data['close'] = data['close'].astype(float)

        train_model(symbol, data)
        logging.info(f"Fetched {len(historical_prices)} days of historical data and trained model for {symbol}")

    real_time_data[symbol].append(price)
    if len(real_time_data[symbol]) > 26:
        real_time_data[symbol].pop(0)

    if len(real_time_data[symbol]) >= 26:
        sma = pd.Series(real_time_data[symbol]).rolling(window=20).mean().iloc[-1]
        rsi = calculate_rsi(real_time_data[symbol]).iloc[-1]
        macd_line, signal_line = calculate_macd(real_time_data[symbol])
        macd = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        _, upper_band, lower_band = calculate_bollinger_bands(real_time_data[symbol])
        
        return sma, rsi, macd, macd_signal, upper_band, lower_band
    else:
        logging.info(f"Not enough data to calculate indicators for {symbol}.")
        return None, None, None, None, None, None

# WebSocket handler: Process incoming live data
async def handle_trade_update(msg):
    symbol = msg['s']
    price = float(msg['c'])  # Use 'c' for the latest price (close price)
    print(f"Received trade update for {symbol}: {price}")

    sma, rsi, macd, macd_signal, upper_band, lower_band = update_indicators(symbol, price)

    if pd.notna([sma, rsi, macd, macd_signal, upper_band, lower_band]).all():
        print(f"Indicators for {symbol} - SMA: {sma}, RSI: {rsi}, MACD: {macd}, Upper Band: {upper_band}, Lower Band: {lower_band}")
        
        action = predict_trade_action(rsi, macd, macd_signal, upper_band, lower_band)
        
        if action == 1:
            await execute_real_time_trade(symbol, "BUY", price, lower_band)
        elif action == 0:
            await execute_real_time_trade(symbol, "SELL", price, upper_band)
    else:
        logging.info(f"Skipping trade for {symbol}. Missing or invalid indicator values.")



# Real-time order execution with stop-loss and take-profit
async def execute_real_time_trade(symbol: str, side: str, current_price: float, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.05):
    try:
        # Check if there's an active order for this symbol
        if symbol in active_orders:
            # Retrieve the order details to check its status
            order_id = active_orders[symbol]['orderId']
            order_status = binance_client.get_order(symbol=symbol, orderId=order_id)
            
            # If the order is still open, we skip new trade attempts
            if order_status.get('status') in ['NEW', 'PARTIALLY_FILLED']:
                print(f"Order already active for {symbol}, skipping")
                return
            
            # If the order is no longer active, remove it from active_orders
            elif order_status.get('status') in ['FILLED', 'CANCELED', 'EXPIRED', 'REJECTED']:
                del active_orders[symbol]
        
        # Calculate stop-loss and take-profit prices
        stop_loss_price = current_price * (1 - stop_loss_pct) if side == "BUY" else current_price * (1 + stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct) if side == "BUY" else current_price * (1 - take_profit_pct)
        
        # Calculate the position size based on available balance
        quantity = calculate_position_size(symbol, current_price)
        if quantity is None:
            print(f"Error: Could not calculate position size for {symbol}. Skipping order.")
            logging.error(f"Error: Could not calculate position size for {symbol}. Skipping order.")
            return  # Skip order placement if quantity is invalid

        if side == "BUY":
            print(f"Attempting to place buy order for {symbol} at {current_price} with quantity {quantity}")
            try:
                # Log available balance before trying to buy
                balance_info = binance_client.get_asset_balance(asset='USDT')
                usdt_balance = float(balance_info['free'])
                logging.info(f"Available USDT balance: {usdt_balance}")

                buy_order = binance_client.order_market_buy(
                    symbol=symbol, 
                    quantity=quantity
                )
                active_orders[symbol] = buy_order

                print(f"Buy order placed for {symbol}: {buy_order}")
                logging.info(f"Buy order placed for {symbol} - Quantity: {quantity}, Price: {current_price}")

                # OCO orders
                print(f"Setting stop-loss at {stop_loss_price} and take-profit at {take_profit_price} for {symbol}")
                oco_order = binance_client.create_oco_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    quantity=quantity,
                    price=take_profit_price,  # Take-profit price
                    stopPrice=stop_loss_price,  # Stop-loss trigger price
                    stopLimitPrice=stop_loss_price * 0.995,  # Slightly lower stop-limit price
                    stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC
                )
                print(f"OCO order placed for {symbol}: {oco_order}")

            except Exception as e:
                logging.error(f"Error executing buy trade for {symbol}: {str(e)}")
                print(f"Error executing buy trade for {symbol}: {str(e)}")

        elif side == "SELL":
            print(f"Placing sell order for {symbol} at {current_price}")
            try:
                sell_order = binance_client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                active_orders[symbol] = sell_order

                print(f"Setting stop-loss at {stop_loss_price} and take-profit at {take_profit_price} for {symbol}")
                
                # Place OCO order for stop-loss and take-profit
                oco_order = binance_client.create_oco_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    quantity=quantity,
                    price=take_profit_price,  # Take-profit price
                    stopPrice=stop_loss_price,  # Stop-loss trigger price
                    stopLimitPrice=stop_loss_price * 1.005,  # Slightly higher stop-limit price
                    stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC
                )
                print(f"OCO order placed for {symbol}: {oco_order}")

            except Exception as e:
                logging.error(f"Error executing sell trade for {symbol}: {str(e)}")
                print(f"Error executing sell trade for {symbol}: {str(e)}")

        logging.info(f"Trade executed for {symbol} - Side: {side}, Price: {current_price}")
    
    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {str(e)}")
        print(f"Error executing trade for {symbol}: {str(e)}")
        raise



# Replace the bsm.close() call with bsm.stop()
async def start_websocket(symbols):
    try:
        # Loop through each symbol and set up a WebSocket
        for symbol in symbols:
            # Create a symbol ticker socket using the updated method
            socket = bsm.symbol_ticker_socket(symbol=symbol)

            # Handle the WebSocket stream in an async loop
            async with socket as stream:
                while True:
                    msg = await stream.recv()  # Receive messages from the WebSocket
                    await handle_trade_update(msg)  # Process the received message

    except Exception as e:
        logging.error(f"Error with WebSocket for {symbol}: {str(e)}")
        print(f"Error with WebSocket for {symbol}: {str(e)}")
    finally:
        await bsm.stop()  # Ensure WebSocket manager stops after execution



# FastAPI endpoint: Trigger real-time trading
@app.post("/start_trading")
async def start_trading(symbols: list[str]):
    try:
        logging.info(f"Starting real-time trading for {symbols}")
        asyncio.create_task(start_websocket(symbols))
        return {"message": "Real-time trading started for symbols: " + ", ".join(symbols)}
    
    except Exception as e:
        logging.error(f"Error starting trading: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting trading")



# FastAPI endpoint: Stop trading
@app.post("/stop_trading")
async def stop_trading(symbols: list[str]):
    try:
        logging.info(f"Stopping trading for {symbols}")
        for symbol in symbols:
            if symbol in active_orders:
                order = active_orders[symbol]
                binance_client.cancel_order(symbol=symbol, orderId=order['orderId'])
                del active_orders[symbol]
        return {"message": "Trading stopped for symbols: " + ", ".join(symbols)}
    
    except Exception as e:
        logging.error(f"Error stopping trading: {str(e)}")
        raise HTTPException(status_code=500, detail="Error stopping trading")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
            