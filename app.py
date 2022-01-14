import streamlit as st
import pandas as pd
import requests
import tweepy
import ccxt
import plotly.express as px
from functions import *

import numpy as np
import pandas as pd
import time
import dateutil
from datetime import datetime, timedelta
from functools import reduce
from scipy.signal import argrelextrema
import plotly.graph_objects as go

data_1h = collect_data(timeframe = '1h', limit = 1000)
data_4h = collect_data(timeframe = '4h', limit = 1000)
data_8h = collect_data(timeframe = '8h', limit = 1000)
data_24h = collect_data(timeframe = '1d', limit = 1000)

coins = data_1h['Symbol'].unique().tolist()

current_time = pd.DataFrame(data_1h[data_1h['Symbol'] == 'ETH/USDT']['Datetime']).iloc[-1, 0]

recent_candles = data_1h[data_1h['Datetime'] == current_time]

brk_out_1h=brk_out(data_1h)
brk_out_4h=brk_out(data_4h)
brk_out_8h=brk_out(data_8h)
brk_out_24h=brk_out(data_24h)

pierce_3ave_1h= pierce_3ave(data_1h)
pierce_3ave_4h= pierce_3ave(data_4h)
pierce_3ave_8h= pierce_3ave(data_8h)
pierce_3ave_24h= pierce_3ave(data_24h)

harmonic_1h = detect_harmonic(data_1h, coins,  order = 10)
harmonic_4h = detect_harmonic(data_4h, coins, order = 10)
harmonic_8h = detect_harmonic(data_8h, coins, order = 10)
harmonic_24h = detect_harmonic(data_24h, coins, order = 10)


fig_price_ch_24_up = px.bar(
    recent_candles.sort_values(by='price_ch_24', ascending=False).head(10),
    x="price_ch_24",
    y="Symbol",
    orientation="h",
    text='price_ch_24',
    color='Symbol',
    title="<b> 24小时涨幅榜</b>",
    template="plotly_white"

)
fig_price_ch_24_up.update_layout(yaxis={'categoryorder': 'total ascending'},
                                 xaxis=dict(title='24小时价格变化'),
                                 showlegend=False)
fig_price_ch_24_up.update_traces(texttemplate='%{text:.1%}', textposition='auto')

fig_price_ch_24_down = px.bar(
    recent_candles.sort_values(by='price_ch_24', ascending=False).tail(10),
    x="price_ch_24",
    y="Symbol",
    orientation="h",
    text='price_ch_24',
    color='Symbol',
    title="<b> 24小时跌幅榜</b>",
    template="plotly_white"
)
fig_price_ch_24_down.update_layout(yaxis={'categoryorder': 'total descending'},
                                   xaxis=dict(title='24小时价格变化'),
                                   showlegend=False)
fig_price_ch_24_down.update_traces(texttemplate='%{text:.1%}', textposition='auto')

# 8小时
fig_price_ch_8_up = px.bar(
    recent_candles.sort_values(by='price_ch_8', ascending=False).head(10),
    x="price_ch_8",
    y="Symbol",
    orientation="h",
    text='price_ch_8',
    color='Symbol',
    title="<b> 8小时涨幅榜</b>",
    template="plotly_white"

)
fig_price_ch_8_up.update_layout(yaxis={'categoryorder': 'total ascending'},
                                xaxis=dict(title='8小时价格变化'),
                                showlegend=False)
fig_price_ch_8_up.update_traces(texttemplate='%{text:.1%}', textposition='auto')

fig_price_ch_8_down = px.bar(
    recent_candles.sort_values(by='price_ch_8', ascending=False).tail(10),
    x="price_ch_8",
    y="Symbol",
    orientation="h",
    text='price_ch_8',
    color='Symbol',
    title="<b> 8小时跌幅榜</b>",
    template="plotly_white"
)
fig_price_ch_8_down.update_layout(yaxis={'categoryorder': 'total descending'},
                                  xaxis=dict(title='8小时价格变化'),
                                  showlegend=False)
fig_price_ch_8_down.update_traces(texttemplate='%{text:.1%}', textposition='auto')

# 4小时
fig_price_ch_4_up = px.bar(
    recent_candles.sort_values(by='price_ch_4', ascending=False).head(10),
    x="price_ch_4",
    y="Symbol",
    orientation="h",
    text='price_ch_4',
    color='Symbol',
    title="<b> 4小时涨幅榜</b>",
    template="plotly_white"

)
fig_price_ch_4_up.update_layout(yaxis={'categoryorder': 'total ascending'},
                                xaxis=dict(title='4小时价格变化'),
                                showlegend=False)
fig_price_ch_4_up.update_traces(texttemplate='%{text:.1%}', textposition='auto')

fig_price_ch_4_down = px.bar(
    recent_candles.sort_values(by='price_ch_4', ascending=False).tail(10),
    x="price_ch_4",
    y="Symbol",
    orientation="h",
    text='price_ch_4',
    color='Symbol',
    title="<b> 4小时跌幅榜</b>",
    template="plotly_white"
)
fig_price_ch_4_down.update_layout(yaxis={'categoryorder': 'total descending'},
                                  xaxis=dict(title='4小时价格变化'),
                                  showlegend=False)
fig_price_ch_4_down.update_traces(texttemplate='%{text:.1%}', textposition='auto')

# 1小时
fig_price_ch_1_up = px.bar(
    recent_candles.sort_values(by='price_ch_1', ascending=False).head(10),
    x="price_ch_1",
    y="Symbol",
    orientation="h",
    text='price_ch_1',
    color='Symbol',
    title="<b> 1小时涨幅榜</b>",
    template="plotly_white"

)
fig_price_ch_1_up.update_layout(yaxis={'categoryorder': 'total ascending'},
                                xaxis=dict(title='1小时价格变化'),
                                showlegend=False)
fig_price_ch_1_up.update_traces(texttemplate='%{text:.1%}', textposition='auto')

fig_price_ch_1_down = px.bar(
    recent_candles.sort_values(by='price_ch_1', ascending=False).tail(10),
    x="price_ch_1",
    y="Symbol",
    orientation="h",
    text='price_ch_1',
    color='Symbol',
    title="<b> 1小时跌幅榜</b>",
    template="plotly_white"
)
fig_price_ch_1_down.update_layout(yaxis={'categoryorder': 'total descending'},
                                  xaxis=dict(title='1小时价格变化'),
                                  showlegend=False)
fig_price_ch_1_down.update_traces(texttemplate='%{text:.1%}', textposition='auto')

st.markdown('''# **东哥币圈扫描器**
一个实用的币圈行情扫描器.
''')
mn_time = pd.to_datetime(current_time)-timedelta(hours = 6)

# exec(open("harmonic_detector.py").read())


# for coin in coins:
#     candlestick = plot_pat(data_1h, coin)
#     st.pyplot(candlestick)
st.write(mn_time)

st.header('''TTM 挤压扫描器''')

left_column, right_column = st.columns(2)
left_column.subheader('1小时：')
left_column.write(brk_out_1h)
left_column.write(pierce_3ave_1h)

right_column.subheader('4小时：')
right_column.write(brk_out_4h)
right_column.write(pierce_3ave_4h)

left_column, right_column = st.columns(2)
left_column.subheader('8小时：')
left_column.write(brk_out_8h)
left_column.write(pierce_3ave_8h)

right_column.subheader('24小时：')
right_column.write(brk_out_24h)
right_column.write(pierce_3ave_24h)

st.header('谐波形态扫描器')

left_column, right_column = st.columns(2)
left_column.write('1小时谐波形态：')
left_column.dataframe(harmonic_1h)
right_column.write('4小时谐波形态：')
right_column.dataframe(harmonic_4h)

left_column, right_column = st.columns(2)
left_column.write('8小时谐波形态：')
left_column.dataframe(harmonic_8h)
right_column.write('24小时谐波形态：')
right_column.dataframe(harmonic_24h)

st.header('币圈行情动态')

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_price_ch_24_up, use_container_width=True)
right_column.plotly_chart(fig_price_ch_24_down, use_container_width=True)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_price_ch_8_up, use_container_width=True)
right_column.plotly_chart(fig_price_ch_8_down, use_container_width=True)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_price_ch_4_up, use_container_width=True)
right_column.plotly_chart(fig_price_ch_4_down, use_container_width=True)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_price_ch_1_up, use_container_width=True)
right_column.plotly_chart(fig_price_ch_1_down, use_container_width=True)


# symbol = st.sidebar.text_input("Symbol", value='BTC', max_chars=5)

# r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

# data = r.json()

# for message in data['messages']:
#     st.image(message['user']['avatar_url'])
#     st.write(message['user']['username'])
#     st.write(message['created_at'])
#     st.write(message['body'])
