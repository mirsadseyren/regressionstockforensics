import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import linregress

# Import from regression.py
from regression import (
    load_data, get_tickers_from_file, find_best_candidate, run_simulation,
    STOX_FILE, LOOKBACK_DAYS, MIN_SLOPE, MIN_R_SQUARED, 
    VOLUME_STOP_RATIO, STOP_LOSS_RATE, REBALANCE_FREQ, START_CAPITAL, COMMISSION_RATE,
    MAX_ATR_PERCENT
)

st.set_page_config(page_title="Regression Bot Dashboard", layout="wide")

st.title("🚀 Momentum Regression Bot")

# --- SIDEBAR: Ayarlar ---
st.sidebar.header("⚙️ Strateji Parametreleri")

lookback_days = st.sidebar.number_input("Lookback Days", value=LOOKBACK_DAYS)
min_slope = st.sidebar.number_input("Min Slope (Eğim)", value=MIN_SLOPE, format="%.4f")
min_r2 = st.sidebar.number_input("Min R-Squared", value=MIN_R_SQUARED)
stop_loss = st.sidebar.number_input("Stop Loss Rate", value=STOP_LOSS_RATE)
vol_stop_ratio = st.sidebar.number_input("Volume Stop Ratio", value=VOLUME_STOP_RATIO)
atr_limit = st.sidebar.number_input("ATR Filter Rate", value=MAX_ATR_PERCENT, format="%.3f")
start_capital = st.sidebar.number_input("Starting Capital (TL)", value=float(START_CAPITAL), step=1000.0)

# --- VERİ YÜKLEME ---
@st.cache_data(ttl=3600*12) # 12 saat cache
def get_data():
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        return None
    return load_data(tickers)

with st.spinner("Veriler yükleniyor..."):
    all_data = get_data()

if all_data is None:
    st.error(f"{STOX_FILE} bulunamadı veya hisse yok!")
    st.stop()

# --- TABLAR ---
tab1, tab2, tab3 = st.tabs(["🔍 Günlük Tarama", "📈 Simülasyon Backtest", "📊 Hisse Analizi"])

# === TAB 1: GÜNLÜK TARAMA ===
with tab1:
    st.header("Bugünün Sinyalleri")
    
    col1, col2 = st.columns(2)
    with col1:
        date_gap = st.number_input("Geriye Dönük Gün (0 = Bugün)", min_value=0, value=0)
    
    if st.button("Taramayı Başlat"):
        # Son tarihi bul
        if isinstance(all_data.columns, pd.MultiIndex):
            last_date = all_data['Close'].index[-1]
        else:
            last_date = all_data.index[-1]
            
        target_date = last_date - timedelta(days=date_gap)
        st.info(f"Analiz Tarihi: {target_date.date()}")
        
        # Regression.py'deki find_best_candidate fonksiyonunu çağırıyoruz
        # Not: find_best_candidate min_slope ve min_r2'yi global'den alıyor.
        # Streamlit parametrelerini oraya geçirmek için fonksiyonu güncellememiz gerekirdi.
        # Şimdilik global'deki varsayılanları kullanıyor.
        candidates = find_best_candidate(target_date, all_data, lookback_days, max_atr_percent=atr_limit)
        
        # Filtreleme (Global parametreleri ezmek için burada tekrar filtreleyebiliriz ama 
        # find_best_candidate içinde zaten bir filtre var. En temizi fonksiyonu güncellemekti.)
        # Şimdilik dönen adayları buradaki parametrelere göre tekrar süzelim:
        filtered_candidates = [
            c for c in candidates 
            if c['slope'] >= min_slope and c['r2'] >= min_r2
        ]
        
        st.write(f"Bulunan Aday: {len(filtered_candidates)}")
        
        if filtered_candidates:
            df_candidates = pd.DataFrame(filtered_candidates)
            df_candidates['Ticker'] = df_candidates['t'].str.replace('.IS', '')
            
            # Formatlama
            df_display = df_candidates[['Ticker', 'price', 'slope', 'r2', 'score', 'vol_curr', 'vol_avg']].copy()
            df_display.columns = ['Ticker', 'Fiyat', 'Eğim', 'R2', 'Skor', 'Hacim', 'Ort. Hacim']
            
            st.dataframe(df_display.style.format({
                'Fiyat': "{:.2f}",
                'Eğim': "{:.4f}",
                'R2': "{:.2f}",
                'Skor': "{:.4f}",
                'Hacim': "{:,.0f}",
                'Ort. Hacim': "{:,.0f}"
            }))
            
            st.success(f"🎯 En İyi Seçim: **{filtered_candidates[0]['t']}**")
        else:
            st.warning("Kriterlere uygun hisse bulunamadı.")

# === TAB 2: BACKTEST ===
with tab2:
    st.header("Geçmiş Performans Simülasyonu")
    st.write("Belirlenen parametrelerle son 1 yılın simülasyonunu çalıştırır.")
    
    if st.button("Simülasyonu Çalıştır"):
        with st.spinner("Simülasyon yapılıyor..."):
            # Parametreleri fonksiyona geçiyoruz
            daily_vals, trade_history, final_bal = run_simulation(
                all_data, 
                lookback_days=lookback_days,
                min_slope=min_slope, # run_simulation içinde henüz kullanılmıyor (TODO)
                min_r2=min_r2,       # run_simulation içinde henüz kullanılmıyor (TODO)
                stop_loss_rate=stop_loss,
                volume_stop_ratio=vol_stop_ratio,
                start_capital=start_capital,
                max_atr_percent=atr_limit
            )
            
            roi = ((final_bal - start_capital) / start_capital) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Başlangıç", f"{start_capital:,.0f} TL")
            c2.metric("Bitiş", f"{final_bal:,.0f} TL")
            c3.metric("Getiri (ROI)", f"%{roi:.2f}", delta_color="normal")
            
            # Grafik
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_vals.index, y=daily_vals.values, mode='lines', name='Portföy Değeri'))
            fig.add_trace(go.Scatter(x=daily_vals.index, y=[start_capital]*len(daily_vals), mode='lines', name='Başlangıç', line=dict(dash='dash', color='gray')))
            fig.update_layout(title="Portföy Gelişimi", xaxis_title="Tarih", yaxis_title="TL")
            st.plotly_chart(fig, use_container_width=True)
            
            # İşlem Geçmişi
            st.subheader("İşlem Geçmişi")
            if trade_history:
                df_hist = pd.DataFrame(trade_history, columns=["Tarih", "Hisse", "Lot", "Fiyat", "İşlem", "Nakit", "Bilgi"])
                st.dataframe(df_hist)
            else:
                st.write("İşlem yok.")

# === TAB 3: HİSSE ANALİZİ ===
with tab3:
    st.header("Tekil Hisse Regresyon Analizi")
    
    tickers_list = [t.replace('.IS', '') for t in get_tickers_from_file(STOX_FILE)]
    selected_ticker = st.selectbox("Hisse Seçin", tickers_list)
    ticker_full = selected_ticker + ".IS"
    
    if st.button("Analiz Et"):
        # Veriyi al
        if isinstance(all_data.columns, pd.MultiIndex):
            try:
                series = all_data['Close'][ticker_full].dropna()
            except:
                series = all_data.xs('Close', axis=1, level=0)[ticker_full].dropna()
        else:
            series = all_data[ticker_full].dropna()
            
        # Son N gün
        series_window = series.tail(lookback_days)
        
        # Regresyon
        y = np.log(series_window.values)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Regresyon çizimi (Lineer ölçekte görselleştirmek için exp alacağız)
        # log(y) = a + bx  => y = exp(a + bx)
        reg_line_log = intercept + slope * x
        reg_line = np.exp(reg_line_log)
        
        # Grafik
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series_window.index, y=series_window.values, mode='markers+lines', name='Fiyat'))
        fig2.add_trace(go.Scatter(x=series_window.index, y=reg_line, mode='lines', name='Exp. Regresyon', line=dict(color='orange')))
        
        st.write(f"**Slope (Eğim):** {slope:.5f}")
        st.write(f"**R-Squared:** {r_value**2:.4f}")
        
        st.plotly_chart(fig2, use_container_width=True)
