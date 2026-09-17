import time
import ccxt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Bitget Pre-Order StatArb Bot", page_icon="⚡", layout="wide")

# --- Session States ---
if 'balance' not in st.session_state: st.session_state.balance = 1000.0
if 'position' not in st.session_state: st.session_state.position = 0 
if 'pending_order' not in st.session_state: st.session_state.pending_order = None # 미리 대기 중인 주문 정보
if 'entry_price_A' not in st.session_state: st.session_state.entry_price_A = 0.0
if 'entry_price_B' not in st.session_state: st.session_state.entry_price_B = 0.0
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'total_fees_paid' not in st.session_state: st.session_state.total_fees_paid = 0.0
if 'tick_history' not in st.session_state: 
    st.session_state.tick_history = pd.DataFrame(columns=['time', 'price_A', 'price_B', 'z_score'])

exchange = ccxt.bitget({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

@st.cache_data(ttl=60)
def get_high_volatility_symbols(top_n=20):
    try:
        tickers = exchange.fetch_tickers()
        valid_tickers = []
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT:USDT') and t.get('quoteVolume') and t['quoteVolume'] > 5000000:
                change = abs(t.get('percentage', 0))
                valid_tickers.append({'symbol': symbol, 'volume': t['quoteVolume'], 'volatility': change})
        df_vol = pd.DataFrame(valid_tickers).sort_values(by='volatility', ascending=False)
        return df_vol.head(top_n)
    except Exception:
        return pd.DataFrame(columns=['symbol', 'volatility', 'volume'])

def fetch_realtime_tickers(symbol_A, symbol_B):
    ticker_A = exchange.fetch_ticker(symbol_A)
    ticker_B = exchange.fetch_ticker(symbol_B)
    return float(ticker_A['last']), float(ticker_B['last'])

# --- UI Header ---
st.title("⚡ 미리 지정가 대기(Pre-Order) 0초 체결 페어트레이딩 봇")
st.caption("진입 임계값에 도달하기 전 미리 지정가 호가를 대기시켜 딜레이와 슬리피지를 제거합니다.")

vol_df = get_high_volatility_symbols(20)
vol_symbols = vol_df['symbol'].tolist() if not vol_df.empty else ["DOGE/USDT:USDT", "SHIB/USDT:USDT", "PEPE/USDT:USDT", "BONK/USDT:USDT"]

# Sidebar Settings
st.sidebar.header("🔥 코인 및 미리 주문 설정")
selected_A = st.sidebar.selectbox("자산 A", vol_symbols, index=0)
selected_B = st.sidebar.selectbox("자산 B", vol_symbols, index=min(1, len(vol_symbols)-1))
symbol_A, symbol_B = selected_A, selected_B

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 지정가 대기 옵션")
# 지정가 수수료(Maker Fee) 0.02% 적용
fee_rate = st.sidebar.number_input("지정가(Maker) 수수료율 (%)", value=0.02, step=0.01) / 100.0

entry_z = st.sidebar.number_input("진입 확정 Z-Score 기준", value=0.3, step=0.1)
pre_order_z = st.sidebar.number_input("⚡ 미리 주문 대기 Z-Score 구간", value=0.2, step=0.05, help="이 구간에 오면 호가창에 매수/매도 주문을 미리 집어넣습니다.")

target_net_profit = st.sidebar.number_input("🎯 익절 목표 순수익 ($)", value=0.10, step=0.02)
stop_z = st.sidebar.number_input("손절 Z-Score", value=2.5, step=0.1)
trade_usd = st.sidebar.number_input("1회 거래금액 ($)", value=100.0)

auto_sec_run = st.sidebar.checkbox("⏱️ 1초 단위 자동 갱신", value=True)

# --- Real-time Logic ---
try:
    price_A, price_B = fetch_realtime_tickers(symbol_A, symbol_B)
    now_str = pd.Timestamp.now().strftime('%H:%M:%S')

    new_data = pd.DataFrame([{'time': now_str, 'price_A': float(price_A), 'price_B': float(price_B), 'z_score': 0.0}])
    st.session_state.tick_history = pd.concat([st.session_state.tick_history, new_data], ignore_index=True).tail(15)
    df_tick = st.session_state.tick_history.copy()
    df_tick['price_A'] = df_tick['price_A'].astype(float)
    df_tick['price_B'] = df_tick['price_B'].astype(float)

    current_pnl_pct = 0.0
    current_net_usd = 0.0

    if len(df_tick) >= 4:
        y, x = df_tick['price_A'], df_tick['price_B']
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        hedge_ratio = float(model.params['price_B']) if 'price_B' in model.params else 1.0

        spread = y - (hedge_ratio * x)
        std_val = float(spread.std())
        mean_val = float(spread.mean())
        curr_z = float((spread.iloc[-1] - mean_val) / std_val) if std_val != 0 else 0.0
        df_tick['z_score'] = (spread - mean_val) / std_val if std_val != 0 else 0.0
        st.session_state.tick_history['z_score'] = df_tick['z_score']

        if st.session_state.position != 0:
            if st.session_state.position == 1:
                pnl_A = (price_A - st.session_state.entry_price_A) / st.session_state.entry_price_A
                pnl_B = (st.session_state.entry_price_B - price_B) / st.session_state.entry_price_B
            else:
                pnl_A = (st.session_state.entry_price_A - price_A) / st.session_state.entry_price_A
                pnl_B = (price_B - st.session_state.entry_price_B) / st.session_state.entry_price_B

            current_pnl_pct = ((pnl_A + pnl_B) / 2) * 100.0
            gross_usd = trade_usd * (current_pnl_pct / 100.0)
            exit_fee = (trade_usd * 2) * fee_rate # 지정가 수수료 적용
            current_net_usd = gross_usd - exit_fee

        # ★ 프리-오더(미리 주문 대기 및 0초 체결) 로직 ★
        if st.session_state.position == 0:
            # 1. 미리 주문 대기 단계 (Pre-Order Placement)
            if abs(curr_z) >= pre_order_z and abs(curr_z) < entry_z:
                st.session_state.pending_order = "LONG" if curr_z < 0 else "SHORT"
            
            # 2. 목표 도달 시 딜레이 없이 즉시 체결 (0-Delay Execution)
            elif abs(curr_z) >= entry_z:
                entry_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance -= entry_fee
                st.session_state.total_fees_paid += entry_fee
                st.session_state.position = 1 if curr_z <= -entry_z else -1
                st.session_state.entry_price_A, st.session_state.entry_price_B = price_A, price_B
                
                order_type_str = "미리대기 지정가 0초 체결" if st.session_state.pending_order else "즉시 체결"
                st.session_state.trade_history.append(
                    f"⚡ [{now_str}] [{order_type_str}] {'Long' if st.session_state.position==1 else 'Short'} 진입 완료 (수수료 절감 적용: 0.02%)"
                )
                st.session_state.pending_order = None
            else:
                st.session_state.pending_order = None # 조건 이탈 시 대기 취소

        else:
            # 청산 로직 (목표 달성 시 지정가 즉시 청산)
            if current_net_usd >= target_net_profit or abs(curr_z) >= stop_z:
                exit_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance += current_net_usd
                st.session_state.total_fees_paid += exit_fee
                type_str = "강제손절" if abs(curr_z) >= stop_z else "지정가익절"
                st.session_state.trade_history.append(
                    f"🎯 [{now_str}] 청산({type_str}) 순손익: ${current_net_usd:+.3f} | 잔고: ${st.session_state.balance:.2f}"
                )
                st.session_state.position = 0

        # UI Layout
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("🔥 비트겟 변동성 TOP 20 코인")
            st.dataframe(vol_df[['symbol', 'volatility', 'volume']].reset_index(drop=True), use_container_width=True)

        with col_right:
            st.subheader("⚡ 실시간 호가 대기 현황판")
            m1, m2, m3 = st.columns(3)
            m1.metric("가상 계좌 잔고", f"${st.session_state.balance:.2f}")
            m2.metric("실시간 Z-Score", f"{curr_z:.2f}")
            
            # 미리 대기 상태 표시
            if st.session_state.pending_order:
                pending_str = f"🟡 지정가 대기 중 ({st.session_state.pending_order})"
            else:
                pending_str = "⚪ 대기 없음"
            m3.metric("호가창 대기 상태", pending_str)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"1초 실시간 시세 ({symbol_A} vs {symbol_B})", "Z-Score 및 지정가 대기 구간"))
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['price_A']/df_tick['price_A'].iloc[0], name=symbol_A), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['price_B']/df_tick['price_B'].iloc[0], name=symbol_B), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['z_score'], name="Z-Score", line=dict(color='yellow')), row=2, col=1)
        
        # 지정가 미리 대기 라인 시각화
        fig.add_hline(y=pre_order_z, line_dash="dot", line_color="orange", row=2, col=1, annotation_text="숏 미리 대기")
        fig.add_hline(y=-pre_order_z, line_dash="dot", line_color="orange", row=2, col=1, annotation_text="롱 미리 대기")
        fig.add_hline(y=entry_z, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"초단위 시세 수집 중... ({len(df_tick)}/4 초 수집 완료)")

except Exception as e:
    st.error(f"시세 조회 중 오류 발생: {e}")

st.subheader("📜 0초 체결 매매 기록")
for log in reversed(st.session_state.trade_history):
    st.info(log)

if auto_sec_run:
    time.sleep(1)
    st.rerun()
