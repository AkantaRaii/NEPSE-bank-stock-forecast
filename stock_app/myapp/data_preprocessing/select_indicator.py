import talib
from talib import abstract
import pandas as pd
from .fetch_stock_data import fetch

lasso_selected_fetures = {'ADBL':['close', 'BOP', 'DX', 'EXPMA', 'SMA10', 'BOLL_upper'],
            'NABIL':['close', 'noOfTransaction', 'change', 'BOP', 'DX', 'TRIX', 'OBV', 'ADOSC', 'ATR', 'TRANGE', 'HT_LEAD_SINE', 'KDJ.J', 'BOLL_middle', 'BOLL_lower', 'WR', 'PSY'],
            'NIFRA':['close', 'OBV', 'SMA10'],
            'SANIMA':['close', 'ADXR', 'BOP'],
            'EBL':['low', 'close', 'BOP', 'OBV', 'BOLL_lower', 'AR']}

def get_selected_indicator(stock):
    # Load the df for the current stock
    df = fetch(stock)

    # Momentum indicators
    df['ADX'] = abstract.ADX(df, timeperiod=14)
    df['ADXR'] = abstract.ADXR(df, timeperiod=14)
    df['BOP'] = abstract.BOP(df)
    df['CCI'] = abstract.CCI(df, timeperiod=14)
    df['DX'] = abstract.DX(df, timeperiod=14)
    df['MACD'] = abstract.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)['macd']
    df['MACDEXT'] = abstract.MACDEXT(df, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)['macd']
    df['MACDFIX'] = abstract.MACDFIX(df, fastperiod=12, slowperiod=26, signalperiod=9)['macd']
    df['TRIX'] = abstract.TRIX(df, timeperiod=30)
    df['MFI'] = abstract.MFI(df, timeperiod=14)
    df['MOM'] = abstract.MOM(df, timeperiod=10)
    df['PPO'] = abstract.PPO(df, fastperiod=12, slowperiod=26, matype=0)
    df['ROC'] = abstract.ROC(df, timeperiod=10)
    df['RSI'] = abstract.RSI(df, timeperiod=14)
    df['WILLR'] = abstract.WILLR(df, timeperiod=14)

    df['DPO'] = df['close'] - talib.SMA(df['close'], timeperiod=20)
    df['DMA'] = talib.SMA(df['close'], timeperiod=10) - talib.SMA(df['close'], timeperiod=50)
    df['EXPMA'] = df['close'].ewm(span=12, adjust=False).mean()

    # Volume indicators
    df['OBV'] = abstract.OBV(df)
    df['MAOBV'] = talib.SMA(df['OBV'], timeperiod=10)
    df['ADOSC'] = abstract.ADOSC(df, fastperiod=3, slowperiod=10)

    # Volatility indicators
    df['ATR'] = abstract.ATR(df, timeperiod=14)
    df['NATR'] = abstract.NATR(df, timeperiod=14)
    df['TRANGE'] = abstract.TRANGE(df)

    # Cycle indicators
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close'])
    df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close'])
    df['HT_PHASOR_real'], df['HT_PHASOR_imag'] = talib.HT_PHASOR(df['close'])
    df['HT_SINE'], df['HT_LEAD_SINE'] = talib.HT_SINE(df['close'])

    # Moving average indicator (BBI - Custom calculation)
    df['BBI'] = (talib.SMA(df['close'], timeperiod=3) + talib.SMA(df['close'], timeperiod=6) +
                 talib.SMA(df['close'], timeperiod=12) + talib.SMA(df['close'], timeperiod=24)) / 4
    df['SMA10'] = abstract.SMA(df, timeperiod=10)

    # Overbought and oversold indicators (KDJ - Custom calculation)
    df['KDJ.K'], df['KDJ.D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['KDJ.J'] = 3 * df['KDJ.K'] - 2 * df['KDJ.D']
    df['BOLL_upper'], df['BOLL_middle'], df['BOLL_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

    # BIAS (Bias Ratio) - Custom calculation
    df['BIAS1'] = (df['close'] - talib.SMA(df['close'], timeperiod=6)) / talib.SMA(df['close'], timeperiod=6)
    df['BIAS2'] = (df['close'] - talib.SMA(df['close'], timeperiod=12)) / talib.SMA(df['close'], timeperiod=12)
    df['BIAS3'] = (df['close'] - talib.SMA(df['close'], timeperiod=24)) / talib.SMA(df['close'], timeperiod=24)

    # WR (Williams %R)
    df['WR'] = abstract.WILLR(df, timeperiod=14)

    # Energy indicator (PSY) - Custom calculation
    df['PSY'] = (df['close'] > df['close'].shift(1)).rolling(window=12).mean()

    # MTM (Momentum)
    df['MTM'] = talib.MOM(df['close'], timeperiod=10)

    # BRAR (Bullish and Bearish Resistance and Support Indicator) - Custom calculation
    df['BR'] = talib.SMA(df['close'] - df['low'], timeperiod=26) / talib.SMA(df['high'] - df['low'], timeperiod=26)
    df['AR'] = talib.SMA(df['high'] - df['open'], timeperiod=26) / talib.SMA(df['open'] - df['low'], timeperiod=26)
    lasso_selected_fetures[stock].insert(0,'date')
    df=df[lasso_selected_fetures[stock]]
    df = df.dropna()
    

    print(f'\nlast date in select indicator {df['date'].iloc[-1]} and closeing is{df['close'].iloc[-1]}\n')
    return df