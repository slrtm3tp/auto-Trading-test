import time
import ccxt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Bitget Fast-Profit StatArb Bot", page_icon="🎯", layout="wide")

# --- Session States ---
if 'balance' not in st.session_state: st.session_state.balance = 1000.0
if 'position' not in st.session_state: st.session_state.position = 0 
if 'entry_price_A' not in st.session_state: st.session_state.entry_price_A = 0.0
if 'entry_price_B' not in st.session_state: st.session_state.entry_price_B = 0.0
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'total_fees_paid' not in st.session_state: st.session_state.total_fees_paid = 0.0
if 'tick_history' not in st.session_state: 
    st.session_state.tick_history = pd.DataFrame(columns=['time', 'price_A', 'price_B', 'z_score', 'pred_direction'])

# --- Exchange Setup ---
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

def predict_hyper_momentum(df_tick, fast_p=2, slow_p=4):
    if len(df_tick) < slow_p:
        return "NEUTRAL", 0.0
    z_series = df_tick['z_score']
    fast_ema = z_series.ewm(span=fast_p).mean().iloc[-1]
    slow_ema = z_series.ewm(span=slow_p).mean().iloc[-1]
    momentum = fast_ema - slow_ema
    if momentum > 0.001:
        return "UP", momentum   
    elif momentum < -0.001:
        return "DOWN", momentum 
    else:
        return "NEUTRAL", momentum

# --- UI Interface ---
st.title("🎯 목표 순수익 즉시 익절형 초단타 페어트레이딩 봇")
st.caption("수수료를 제외한 실질 순수익이 설정한 목표 금액에 도달하면 즉시 익절하여 수익을 사수합니다.")

vol_df = get_high_volatility_symbols(20)
vol_symbols = vol_df['symbol'].tolist() if not vol_df.empty else ["DOGE/USDT:USDT", "SHIB/USDT:USDT", "PEPE/USDT:USDT", "BONK/USDT:USDT"]

# Sidebar Settings
st.sidebar.header("🔥 코인 선택")
selected_A = st.sidebar.selectbox("자산 A", vol_symbols, index=0)
selected_B = st.sidebar.selectbox("자산 B", vol_symbols, index=min(1, len(vol_symbols)-1))

symbol_A, symbol_B = selected_A, selected_B

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 익절 및 민감도 설정")
lookback_ticks = st.sidebar.slider("분석 윈도우 (N초)", 5, 30, 10)
fee_rate = st.sidebar.number_input("거래 수수료율 (%)", value=0.06, step=0.01) / 100.0

entry_z = st.sidebar.number_input("초민감 진입 Z-Score 기준", value=0.3, step=0.1)

# ★ 핵심 추가: 수수료 뗀 후 이 금액 이상 벌면 즉시 익절 ★
target_net_profit = st.sidebar.number_input("🎯 즉시 익절 목표 순수익 ($)", value=0.15, step=0.05, help="이 금액 이상 수익이 나면 Z-Score와 상관없이 즉시 익절합니다.")

exit_z = st.sidebar.number_input("기존 청산 Z-Score 기준", value=0.1, step=0.05)
stop_z = st.sidebar.number_input("손절 Z-Score", value=2.5, step=0.1)
trade_usd = st.sidebar.number_input("1회 거래금액 ($)", value=100.0)

auto_sec_run = st.sidebar.checkbox("⏱️ 1초 단위 자동 갱신", value=True)

# --- Real-time Processing ---
try:
    price_A, price_B = fetch_realtime_tickers(symbol_A, symbol_B)
    now_str = pd.Timestamp.now().strftime('%H:%M:%S')

    new_data = pd.DataFrame([{
        'time': now_str, 
        'price_A': float(price_A), 
        'price_B': float(price_B),
        'z_score': 0.0,
        'pred_direction': 'NEUTRAL'
    }])

    st.session_state.tick_history = pd.concat([st.session_state.tick_history, new_data], ignore_index=True).tail(lookback_ticks)
    df_tick = st.session_state.tick_history.copy()
    df_tick['price_A'] = df_tick['price_A'].astype(float)
    df_tick['price_B'] = df_tick['price_B'].astype(float)

    current_pnl_pct = 0.0
    current_gross_usd = 0.0
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

        pred_dir, momentum_val = predict_hyper_momentum(df_tick)
        st.session_state.tick_history.iloc[-1, st.session_state.tick_history.columns.get_loc('pred_direction')] = pred_dir

        if st.session_state.position != 0:
            if st.session_state.position == 1:
                pnl_A = (price_A - st.session_state.entry_price_A) / st.session_state.entry_price_A
                pnl_B = (st.session_state.entry_price_B - price_B) / st.session_state.entry_price_B
            else:
                pnl_A = (st.session_state.entry_price_A - price_A) / st.session_state.entry_price_A
                pnl_B = (price_B - st.session_state.entry_price_B) / st.session_state.entry_price_B

            current_pnl_pct = ((pnl_A + pnl_B) / 2) * 100.0
            current_gross_usd = trade_usd * (current_pnl_pct / 100.0)
            exit_fee = (trade_usd * 2) * fee_rate
            current_net_usd = current_gross_usd - exit_fee

        # 진입 로직
        if st.session_state.position == 0:
            if (curr_z <= -entry_z and pred_dir == "UP") or momentum_val > 0.015:
                entry_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance -= entry_fee
                st.session_state.total_fees_paid += entry_fee
                st.session_state.position = 1
                st.session_state.entry_price_A, st.session_state.entry_price_B = price_A, price_B
                st.session_state.trade_history.append(f"🟢 [{now_str}] 진입: Long {symbol_A} / Short {symbol_B}")

            elif (curr_z >= entry_z and pred_dir == "DOWN") or momentum_val < -0.015:
                entry_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance -= entry_fee
                st.session_state.total_fees_paid += entry_fee
                st.session_state.position = -1
                st.session_state.entry_price_A, st.session_state.entry_price_B = price_A, price_B
                st.session_state.trade_history.append(f"🔴 [{now_str}] 진입: Short {symbol_A} / Long {symbol_B}")

        else:
            # ★ 핵심 청산 조건: 목표 순수익 달성 OR Z값 정상화 OR 강제 손절 ★
            is_target_profit = current_net_usd >= target_net_profit
            is_z_target = abs(curr_z) <= exit_z and current_net_usd > 0
            is_stop = abs(curr_z) >= stop_z

            if is_target_profit or is_z_target or is_stop:
                exit_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance += current_net_usd
                st.session_state.total_fees_paid += exit_fee
                
                if is_stop:
                    type_str = "강제손절"
                    log_icon = "🛑"
                elif is_target_profit:
                    type_str = "목표달성즉시익절"
                    log_icon = "💰"
                else:
                    type_str = "복귀익절"
                    log_icon = "🟢"

                st.session_state.trade_history.append(
                    f"{log_icon} [{now_str}] 청산({type_str}) 순손익: ${current_net_usd:+.3f} | 잔고: ${st.session_state.balance:.2f}"
                )
                st.session_state.position = 0

        # UI Layout
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("🔥 비트겟 변동성 TOP 20 코인")
            st.dataframe(vol_df[['symbol', 'volatility', 'volume']].reset_index(drop=True), use_container_width=True)

        with col_right:
            st.subheader("📊 실시간 분석 및 즉시 익절 모니터")
            m1, m2, m3 = st.columns(3)
            m1.metric("가상 계좌 잔고", f"${st.session_state.balance:.2f}")
            m2.metric("실시간 Z-Score", f"{curr_z:.2f}")
            
            pnl_status = "💰 익절 목표 도달중" if current_net_usd >= target_net_profit else ("🟢 순수익 구간" if current_net_usd > 0 else "🟠 수수료 미달")
            m3.metric("수수료 차감 후 순익", f"${current_net_usd:+.3f}", delta=pnl_status, delta_color="normal" if current_net_usd > 0 else "inverse")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"1초 실시간 시세 ({symbol_A} vs {symbol_B})", "1초 Z-Score 추적"))
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['price_A']/df_tick['price_A'].iloc[0], name=symbol_A), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['price_B']/df_tick['price_B'].iloc[0], name=symbol_B), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_tick['time'], y=df_tick['z_score'], name="Z-Score", line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=entry_z, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_layout(height=450, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"초단위 시세 수집 중... ({len(df_tick)}/4 초 수집 완료)")

except Exception as e:
    st.error(f"시세 조회 중 오류 발생: {e}")

st.subheader("📜 매매 기록")
for log in reversed(st.session_state.trade_history):
    if "🟢" in log or "💰" in log:
        st.success(log)
    elif "🛑" in log or "🔴" in log:
        st.error(log)
    else:
        st.info(log)

if auto_sec_run:
    time.sleep(1)
    st.rerun()
