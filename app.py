import time
import threading
import ccxt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Non-blocking Realtime StatArb Bot", page_icon="⚡", layout="wide")

# --- Global Shared Data (스레드 간 공유 데이터) ---
if 'bot_engine' not in st.session_state:
    st.session_state.bot_engine = {
        'balance': 1000.0,
        'position': 0,
        'entry_price_A': 0.0,
        'entry_price_B': 0.0,
        'trade_history': [],
        'tick_data': pd.DataFrame(columns=['time', 'price_A', 'price_B', 'z_score']),
        'curr_z': 0.0,
        'is_running': False
    }

exchange = ccxt.bitget({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

# --- 백그라운드 실시간 감시 스레드 함수 ---
def background_realtime_loop(symbol_A, symbol_B, entry_z, target_profit, stop_z, trade_usd, fee_rate):
    """화면 리로딩과 관계없이 멈추지 않고 1초마다 시세를 감시/체결하는 스레드"""
    engine = st.session_state.bot_engine
    
    while engine['is_running']:
        try:
            # 1. 시세 수집 (화면 렌더링에 영향받지 않음)
            ticker_A = exchange.fetch_ticker(symbol_A)
            ticker_B = exchange.fetch_ticker(symbol_B)
            price_A, price_B = float(ticker_A['last']), float(ticker_B['last'])
            now_str = pd.Timestamp.now().strftime('%H:%M:%S')

            # 2. 데이터 업데이트
            df = engine['tick_data']
            new_row = pd.DataFrame([{'time': now_str, 'price_A': price_A, 'price_B': price_B, 'z_score': 0.0}])
            df = pd.concat([df, new_row], ignore_index=True).tail(15)
            
            if len(df) >= 4:
                y, x = df['price_A'].astype(float), df['price_B'].astype(float)
                model = sm.OLS(y, sm.add_constant(x)).fit()
                hedge_ratio = float(model.params['price_B']) if 'price_B' in model.params else 1.0

                spread = y - (hedge_ratio * x)
                std_val, mean_val = float(spread.std()), float(spread.mean())
                curr_z = float((spread.iloc[-1] - mean_val) / std_val) if std_val != 0 else 0.0
                df['z_score'] = (spread - mean_val) / std_val if std_val != 0 else 0.0
                
                engine['curr_z'] = curr_z
                engine['tick_data'] = df

                # 3. 백그라운드 0초 진입/청산 정산 (딜레이 없음)
                if engine['position'] == 0:
                    if abs(curr_z) >= entry_z:
                        entry_fee = (trade_usd * 2) * fee_rate
                        engine['balance'] -= entry_fee
                        engine['position'] = 1 if curr_z <= -entry_z else -1
                        engine['entry_price_A'], engine['entry_price_B'] = price_A, price_B
                        engine['trade_history'].append(
                            f"⚡ [{now_str}] [백그라운드 즉시체결] {'Long' if engine['position']==1 else 'Short'} 진입 @ Z={curr_z:.2f}"
                        )
                else:
                    # 실시간 손익 정산
                    if engine['position'] == 1:
                        pnl_A = (price_A - engine['entry_price_A']) / engine['entry_price_A']
                        pnl_B = (engine['entry_price_B'] - price_B) / engine['entry_price_B']
                    else:
                        pnl_A = (engine['entry_price_A'] - price_A) / engine['entry_price_A']
                        pnl_B = (price_B - engine['entry_price_B']) / engine['entry_price_B']

                    net_usd = (trade_usd * ((pnl_A + pnl_B) / 2)) - ((trade_usd * 2) * fee_rate)

                    if net_usd >= target_profit or abs(curr_z) >= stop_z:
                        engine['balance'] += net_usd
                        type_str = "손절" if abs(curr_z) >= stop_z else "익절"
                        engine['trade_history'].append(
                            f"🎯 [{now_str}] [백그라운드 즉시청산] {type_str} 순손익: ${net_usd:+.3f} | 잔고: ${engine['balance']:.2f}"
                        )
                        engine['position'] = 0

        except Exception as e:
            pass

        time.sleep(0.8) # 0.8초 주기로 감시

# --- UI 대시보드 화면 ---
st.title("⚡ 백그라운드 스레드 기반 무지연(Zero-Delay) 봇")
st.caption("시세 감시 및 체결이 백그라운드 스레드에서 무한히 작동하므로 화면 리로딩 딜레이가 없습니다.")

symbol_A = st.sidebar.text_input("자산 A", "DOGE/USDT:USDT")
symbol_B = st.sidebar.text_input("자산 B", "SHIB/USDT:USDT")
entry_z = st.sidebar.number_input("진입 Z-Score", value=0.3, step=0.1)
target_profit = st.sidebar.number_input("익절 목표 ($)", value=0.10, step=0.02)
stop_z = st.sidebar.number_input("손절 Z-Score", value=2.5, step=0.1)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶️ 봇 시작"):
    st.session_state.bot_engine['is_running'] = True
    t = threading.Thread(
        target=background_realtime_loop, 
        args=(symbol_A, symbol_B, entry_z, target_profit, stop_z, 100.0, 0.0002),
        daemon=True
    )
    t.start()
    st.success("백그라운드 실시간 스레드 가동 시작!")

if col_btn2.button("⏹️ 봇 중지"):
    st.session_state.bot_engine['is_running'] = False
    st.warning("봇이 중지되었습니다.")

# 백그라운드 데이터 화면 표시
engine = st.session_state.bot_engine
m1, m2, m3 = st.columns(3)
m1.metric("가상 잔고", f"${engine['balance']:.2f}")
m2.metric("실시간 Z-Score", f"{engine['curr_z']:.2f}")
m3.metric("포지션 상태", "LONG" if engine['position']==1 else ("SHORT" if engine['position']==-1 else "NONE"))

st.subheader("📜 백그라운드 매매 실시간 기록")
for log in reversed(engine['trade_history']):
    st.info(log)

# 화면은 2초에 한 번만 느긋하게 새로고침 (체결 속도에는 전혀 영향 없음)
time.sleep(2)
st.rerun()
