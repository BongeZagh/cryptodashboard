def collect_data(timeframe='4h', limit=500):
    # This function downloads candlestick data from
    # binance futures market

    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema

    # define the market
    exchange_f = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # or 'margin'
        }})

    all_coins_f = list(exchange_f.load_markets().keys())

    all_candles_f = []
    for symbol in all_coins_f:
        df = pd.DataFrame(exchange_f.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
        df['symbol'] = symbol

        df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Vol', 'Symbol']

        df['Datetime'] = df['Datetime'].apply(
            lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(x / 1000.)))

        df['Cap'] = df['Close'] * df['Vol']

        # df['price_ch_15m'] = df["Close"].pct_change(1)
        # df['vol_ch_15m'] = df["Vol"].pct_change(1)

        df['price_ch_1'] = df["Close"].pct_change(1)
        df['vol_ch_1'] = df["Vol"].pct_change(1)

        df['price_ch_4'] = df["Close"].pct_change(4)
        df['vol_ch_4'] = df["Vol"].pct_change(4)

        df['price_ch_8'] = df["Close"].pct_change(8)
        df['vol_ch_8'] = df["Vol"].pct_change(8)

        df['price_ch_24'] = df["Close"].pct_change(24)
        df['vol_ch_24'] = df["Vol"].pct_change(24)

        pd.options.mode.chained_assignment = None  # default='warn'

        df['20sma'] = df['Close'].rolling(window=20).mean()

        df['stddev'] = df['Close'].rolling(window=20).std()
        df['lower_band'] = df['20sma'] - (2 * df['stddev'])
        df['upper_band'] = df['20sma'] + (2 * df['stddev'])

        df['TR'] = abs(df['High'] - df['Low'])
        df['ATR'] = df['TR'].rolling(window=20).mean()

        df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
        df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

        def in_squeeze(df):
            return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

        df['squeeze_on'] = df.apply(in_squeeze, axis=1)

        df['squeeze_on_shift_2'] = df['squeeze_on'].shift(2)

        def out_squeeze(df):
            return df['squeeze_on_shift_2'] and not df['squeeze_on']

        df['squeeze_out'] = df.apply(out_squeeze, axis=1)

        if df.iloc[-2]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
            print("{} is coming out the squeeze".format(symbol))

        all_candles_f.append(df)

    all_candles_f = pd.concat(all_candles_f)

    return all_candles_f


# Python program to convert a list to string
# Python program to convert a list to string

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1


def plot_pat(data, symbol):
    data = data[data['Symbol'] == symbol]

    dt = data.Datetime

    data = data.set_index('Datetime')

    price = data.Close

    high = data.High

    low = data.Low

    max_idx = list(argrelextrema(high.values, np.greater, order=10)[0])
    min_idx = list(argrelextrema(low.values, np.less, order=10)[0])

    peak_1 = high.values[max_idx]
    peak_2 = low.values[min_idx]

    peaks_p = list(peak_1) + list(peak_2)

    peaks_idx = list(max_idx) + list(min_idx)

    peaks_idx_dt = np.array(dt.values[peaks_idx])

    peaks_p = np.array(list(peak_1) + list(peak_2))

    final_data = pd.DataFrame({"price": peaks_p, "datetime": peaks_idx_dt})

    final_data = final_data.sort_values(by=['datetime'])

    peaks_idx_dt = final_data.datetime

    peaks_p = final_data.price

    current_idx = np.array(list(final_data.datetime[-4:]) + list(dt[-1:]))

    current_pat = np.array(list(final_data.price[-4:]) + list(low[-1:]))

    start = min(current_idx)

    end = max(current_idx)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]

    data = data.reset_index()
    data_n = data[data['Datetime'] >= start]

    candlestick = go.Candlestick(x=data_n['Datetime'], open=data_n['Open'], high=data_n['High'], low=data_n['Low'],
                                 close=data_n['Close'])

    pat = go.Scatter(x=current_idx, y=current_pat, line={'color': 'blue'})

    fig = go.Figure(data=[candlestick, pat])

    fig.layout.xaxis.type = 'category'

    fig.update_layout(
        width=800,
        height=600)

    fig.layout.xaxis.rangeslider.visible = False
    fig.show()


def peak_detect(df, order=10):
    # this function is to detect four price peaks,then
    # combine them with latest data point to define four
    # moiving segments for harmonic pattern detection.
    from scipy.signal import argrelextrema

    dt = df.Datetime

    df = df.set_index('Datetime')

    price = df.Close

    high = df.High

    low = df.Low

    max_idx = list(argrelextrema(high.values, np.greater, order=order)[0])
    min_idx = list(argrelextrema(low.values, np.less, order=order)[0])

    peak_1 = high.values[max_idx]
    peak_2 = low.values[min_idx]

    peaks_p = list(peak_1) + list(peak_2)

    peaks_idx = list(max_idx) + list(min_idx)

    peaks_idx_dt = np.array(dt.values[peaks_idx])

    peaks_p = np.array(list(peak_1) + list(peak_2))

    final_df = pd.DataFrame({"price": peaks_p, "datetime": peaks_idx_dt})

    final_df = final_df.sort_values(by=['datetime'])

    peaks_idx_dt = final_df.datetime

    peaks_p = final_df.price

    current_idx = np.array(list(final_df.datetime[-4:]) + list(dt[-1:]))

    current_pat = np.array(list(final_df.price[-4:]) + list(price[-1:]))

    start = min(current_idx)

    end = max(current_idx)

    XA = current_pat[1] - current_pat[0]
    AB = current_pat[2] - current_pat[1]
    BC = current_pat[3] - current_pat[2]
    CD = current_pat[4] - current_pat[3]
    moves = [XA, AB, BC, CD]
    symbol = df['Symbol'].unique().tolist()
    return current_idx, current_pat, start, end, moves, high, low, final_df, symbol


def bull_bat(moves, symbol):
    try:
        err_allowed = 0.05
        XA = moves[0]
        AB = moves[1]
        BC = moves[2]
        CD = moves[3]

        M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
        W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

        AB_range = np.array([0.5 - err_allowed, 0.5 + err_allowed]) * abs(XA)
        BC_range = np.array([0.5 - err_allowed, 0.618 + err_allowed]) * abs(AB)
        CD_range = np.array([2 - err_allowed, 2 + err_allowed]) * abs(BC)

        bat_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

        if M_pat and bat_pat:
            return (symbol)

        else:
            return ([])
    except Exception as e:
        return ([])


def bear_bat(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.5 - err_allowed, 0.5 + err_allowed]) * abs(XA)
    BC_range = np.array([0.5 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([2 - err_allowed, 2 + err_allowed]) * abs(BC)

    bat_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))
    # bat_pat = True

    if W_pat and bat_pat:
        return (symbol)

    else:
        return ([])


def bull_gartley(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 1.618 + err_allowed]) * abs(BC)

    gartley_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))
    # gartley_pat = True

    if M_pat and gartley_pat:
        return (symbol)

    else:
        return ([])


def bear_gartley(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 1.618 + err_allowed]) * abs(BC)

    gartley_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

    if W_pat and gartley_pat:
        return (symbol)

    else:
        return ([])


def bull_crab(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.5 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([3.14 - err_allowed, 3.14 + err_allowed]) * abs(BC)

    crab_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

    if M_pat and crab_pat:
        return (symbol)

    else:
        return ([])


def bear_crab(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.5 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([3.14 - err_allowed, 3.14 + err_allowed]) * abs(BC)

    crab_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

    if W_pat and crab_pat:
        return (symbol)

    else:
        return ([])


def bull_butterfly(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.5 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 1.618 + err_allowed]) * abs(BC)

    butterfly_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

    if M_pat and butterfly_pat:
        return (symbol)
    else:
        return ([])


def bear_butterfly(moves, symbol):
    err_allowed = 0.05
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]

    M_pat = (XA > 0 and AB < 0 and BC > 0 and CD < 0)
    W_pat = (XA < 0 and AB > 0 and BC < 0 and CD > 0)

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.5 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 1.618 + err_allowed]) * abs(BC)

    butterfly_pat = (AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and abs(CD) >= abs(BC))

    if W_pat and butterfly_pat:
        return (symbol)
    else:
        return ([])


def detect_harmonic(data, coins, order=10):
    import numpy as np
    import pandas as pd
    import ccxt
    import time
    import dateutil
    from datetime import datetime
    from functools import reduce
    from scipy.signal import argrelextrema


    bull_bats = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bull_coin = bull_bat(moves, symbol)

            if bull_coin != []:
                bull_coin = listToString(bull_coin)
                bull_bats.append(bull_coin)
        except:
            pass

    bull_bats = [s.replace("/USDT", "") for s in bull_bats]

    bear_bats = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bear_coin = bear_bat(moves, symbol)

            if bear_coin != []:
                bear_coin = listToString(bear_coin)
                bear_bats.append(bear_coin)
        except:
            pass

    bear_bats = [s.replace("/USDT", "") for s in bear_bats]

    bull_gartleys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bull_coin = bull_gartley(moves, symbol)

            if bull_coin != []:
                bull_coin = listToString(bull_coin)
                bull_gartleys.append(bull_coin)
        except:
            pass

    bull_gartleys = [s.replace("/USDT", "") for s in bull_gartleys]

    bear_gartleys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bear_coin = bear_gartley(moves, symbol)

            if bear_coin != []:
                bear_coin = listToString(bear_coin)
                bear_gartleys.append(bear_coin)
        except:
            pass

    bear_gartleys = [s.replace("/USDT", "") for s in bear_gartleys]

    bull_crabs = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bull_coin = bull_crab(moves, symbol)

            if bull_coin != []:
                bull_coin = listToString(bull_coin)
                bull_crabs.append(bull_coin)
        except:
            pass

    bull_crabs = [s.replace("/USDT", "") for s in bull_crabs]

    bear_crabs = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bear_coin = bear_crab(moves, symbol)

            if bear_coin != []:
                bear_coin = listToString(bear_coin)
                bear_crabs.append(bear_coin)
        except:
            pass

    bear_crabs = [s.replace("/USDT", "") for s in bear_crabs]

    bull_butterflys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bull_coin = bull_butterfly(moves, symbol)

            if bull_coin != []:
                bull_coin = listToString(bull_coin)
                bull_butterflys.append(bull_coin)
        except:
            pass

    bull_butterflys = [s.replace("/USDT", "") for s in bull_butterflys]

    bear_butterflys = []
    for i in coins:
        try:
            data_new = data[data['Symbol'] == i]
            current_idx, current_pat, start, end, moves, high, low, final_data, symbol = peak_detect(data=data_new,
                                                                                                     order=order)
            bear_coin = bear_butterfly(moves, symbol)

            if bear_coin != []:
                bear_coin = listToString(bear_coin)
                bear_butterflys.append(bear_coin)
        except:
            pass

    bear_butterflys = [s.replace("/USDT", "") for s in bear_butterflys]

    harmonic = pd.DataFrame({"bull_bats": ",".join(bull_bats),
                             "bear_bats": ",".join(bear_bats),
                             "bull_crabs": ",".join(bull_crabs),
                             "bear_crabs": ",".join(bear_crabs),
                             "bull_gartleys": ",".join(bull_gartleys),
                             "bear_gartleys": ",".join(bear_gartleys),
                             "bull_butterflys": ",".join(bull_butterflys),
                             "bear_butterflys": ",".join(bear_butterflys)}, index=[0]).T
    harmonic.columns = ["coins"]
    return harmonic

def brk_out(data):
    break_outs=[]
    for i in data['Symbol'].unique().tolist():
        df = data[data['Symbol'] == i]
        if df['squeeze_out'].iloc[-1]:
            break_out = i
            break_out = listToString(break_out)
            break_outs.append(break_out)

    break_outs = [s.replace("/USDT", "") for s in break_outs]

    break_out_info = "盘整突破："+', '.join(list(break_outs))
    return break_out_info
