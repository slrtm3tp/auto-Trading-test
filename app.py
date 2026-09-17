import time
import ccxt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Bitget Hyper-Sensitive StatArb Bot", page_icon="🚀", layout="wide")

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
    """극초고속 모멘텀 엔진: 미세 틱 변화에도 즉시 반응"""
    if len(df_tick) < slow_p:
        return "NEUTRAL", 0.0
    
    z_series = df_tick['z_score']
    fast_ema = z_series.ewm(span=fast_p).mean().iloc[-1]
    slow_ema = z_series.ewm(span=slow_p).mean().iloc[-1]
    
    momentum = fast_ema - slow_ema
    
    # 임계값을 0.001로 극단적으로 낮춤
    if momentum > 0.001:
        return "UP", momentum   
    elif momentum < -0.001:
        return "DOWN", momentum 
    else:
        return "NEUTRAL", momentum

# --- UI Interface ---
st.title("🚀 민감도 극대화(Hyper-Sensitive) 초단타 페어트레이딩 봇")
st.caption("신호 발생 빈도를 극대화하여 미세한 틱 변화에도 수시로 진입 및 청산을 수행합니다.")

vol_df = get_high_volatility_symbols(20)
vol_symbols = vol_df['symbol'].tolist() if not vol_df.empty else ["DOGE/USDT:USDT", "SHIB/USDT:USDT", "PEPE/USDT:USDT", "BONK/USDT:USDT"]

# Sidebar Settings
st.sidebar.header("🔥 코인 선택")
selected_A = st.sidebar.selectbox("자산 A", vol_symbols, index=0)
selected_B = st.sidebar.selectbox("자산 B", vol_symbols, index=min(1, len(vol_symbols)-1))

symbol_A, symbol_B = selected_A, selected_B

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 극대화 민감도 설정")
lookback_ticks = st.sidebar.slider("분석 윈도우 (N초)", 5, 30, 10)
fee_rate = st.sidebar.number_input("거래 수수료율 (%)", value=0.06, step=0.01) / 100.0

# 잦은 매매를 위한 최저 진입 수치
entry_z = st.sidebar.number_input("초민감 진입 Z-Score 기준", value=0.3, step=0.1)
exit_z = st.sidebar.number_input("청산 Z-Score 기준", value=0.1, step=0.05)
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

        # ⚡ 극초고속 민감 매매 로직: 미세 튀임에도 즉각 진입
        if st.session_state.position == 0:
            if (curr_z <= -entry_z and pred_dir == "UP") or momentum_val > 0.015:
                entry_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance -= entry_fee
                st.session_state.total_fees_paid += entry_fee
                st.session_state.position = 1
                st.session_state.entry_price_A, st.session_state.entry_price_B = price_A, price_B
                st.session_state.trade_history.append(f"🟢 [{now_str}] [초민감 진입] Long {symbol_A} / Short {symbol_B} (모멘텀: {momentum_val:+.4f})")

            elif (curr_z >= entry_z and pred_dir == "DOWN") or momentum_val < -0.015:
                entry_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance -= entry_fee
                st.session_state.total_fees_paid += entry_fee
                st.session_state.position = -1
                st.session_state.entry_price_A, st.session_state.entry_price_B = price_A, price_B
                st.session_state.trade_history.append(f"🔴 [{now_str}] [초민감 진입] Short {symbol_A} / Long {symbol_B} (모멘텀: {momentum_val:+.4f})")

        else:
            is_z_target = abs(curr_z) <= exit_z
            is_net_positive = current_net_usd > 0
            is_stop = abs(curr_z) >= stop_z

            if (is_z_target and is_net_positive) or is_stop:
                exit_fee = (trade_usd * 2) * fee_rate
                st.session_state.balance += current_net_usd
                st.session_state.total_fees_paid += exit_fee
                
                log_icon = "🛑" if is_stop else "🟢"
                type_str = "강제손절" if is_stop else "빠른익절"

                st.session_state.trade_history.append(
                    f"{log_icon} [{now_str}] 청산({type_str}) 순손익: ${current_net_usd:+.3f} (수익률 {current_pnl_pct:+.2f}%) | 잔고: ${st.session_state.balance:.2f}"
                )
                st.session_state.position = 0

        # UI Layout
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("🔥 비트겟 변동성 TOP 20 코인")
            st.dataframe(vol_df[['symbol', 'volatility', 'volume']].reset_index(drop=True), use_container_width=True)

        with col_right:
            st.subheader("📊 실시간 분석 및 극초고속 모멘텀 현황")
            m1, m2, m3 = st.columns(3)
            m1.metric("가상 계좌 잔고", f"${st.session_state.balance:.2f}")
            m2.metric("실시간 Z-Score", f"{curr_z:.2f}")
            
            pred_color = "normal" if pred_dir == "UP" else ("inverse" if pred_dir == "DOWN" else "off")
            pred_label = "🚀 상승 모멘텀 감지" if pred_dir == "UP" else ("🚀 하락 모멘텀 감지" if pred_dir == "DOWN" else "➡️ 관망")
            m3.metric("실시간 틱 모멘텀", pred_label, delta=f"모멘텀: {momentum_val:+.4f}", delta_color=pred_color)

            st.markdown("---")
            
            p1, p2 = st.columns(2)
            if st.session_state.position != 0:
                gross_color = "normal" if current_gross_usd >= 0 else "inverse"
                p1.metric(
                    label="보유 코인 평가 수익률 (Gross)", 
                    value=f"{current_pnl_pct:+.2f}%", 
                    delta=f"${current_gross_usd:+.3f}",
                    delta_color=gross_color
                )

                if current_net_usd > 0:
                    net_status = "🟢 청산 가능 (순수익 발생)"
                    net_color = "normal"
                else:
                    net_status = "🟠 수수료 미달 (대기 필요)"
                    net_color = "inverse"

                p2.metric(
                    label="수수료 차감 후 예상 순손익 (Net)", 
                    value=f"${current_net_usd:+.3f}", 
                    delta=net_status,
                    delta_color=net_color
                )
            else:
                p1.metric("보유 코인 평가 수익률 (Gross)", "0.00%", delta="포지션 없음", delta_color="off")
                p2.metric("수수료 차감 후 예상 순손익 (Net)", "$0.000", delta="포지션 없음", delta_color="off")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"1초 실시간 시세 ({symbol_A} vs {symbol_B})", "1초 Z-Score & 극초고속 모멘텀 분석"))
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

st.subheader("📜 초고속 진입 매매 기록")
for log in reversed(st.session_state.trade_history):
    if "🟢" in log:
        st.success(log)
    elif "🛑" in log or "🔴" in log:
        st.error(log)
    else:
        st.info(log)

if auto_sec_run:
    time.sleep(1)
    st.rerun()
