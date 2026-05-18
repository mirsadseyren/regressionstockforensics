import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import linregress
from fix_cache import fix_cache
import subprocess
import sys
import os
import json

# Import from regression_nonperiodic.py
from regression_nonperiodic import (
    load_data, get_tickers_from_file, find_best_candidate, run_simulation,
    get_vectorized_metrics, get_clean_data,
    STOX_FILE, LOOKBACK_DAYS, MIN_SLOPE, MIN_R_SQUARED, 
    STOP_LOSS_RATE, REBALANCE_FREQ, START_CAPITAL, COMMISSION_RATE,
    MAX_ATR_PERCENT, SLOPE_STOP_FACTOR
)

st.set_page_config(page_title="Regression Bot Dashboard", layout="wide")

st.title("🚀 Momentum Regression Bot (Non-Periodic)")

# --- SIDEBAR: Ayarlar ---
st.sidebar.header("⚙️ Strateji Parametreleri")

lookback_days = st.sidebar.number_input("Lookback Days", value=LOOKBACK_DAYS)
min_slope = st.sidebar.number_input("Min Slope (Eğim)", value=MIN_SLOPE, format="%.4f")
min_r2 = st.sidebar.number_input("Min R-Squared", value=MIN_R_SQUARED)
stop_loss = st.sidebar.number_input("Stop Loss Rate", value=STOP_LOSS_RATE)

rebalance_options = ['1D', '3D', '5D', '7D', '15D', '1M', '2M']
default_ix = rebalance_options.index(REBALANCE_FREQ) if REBALANCE_FREQ in rebalance_options else 3
rebalance_freq = st.sidebar.selectbox("Max Holding Period (Rebalance Freq)", options=rebalance_options, index=default_ix)

atr_limit = st.sidebar.number_input("ATR Filter Rate", value=MAX_ATR_PERCENT, format="%.3f")
use_slope_stop = st.sidebar.checkbox("Daily Min Return Stop (Aktif/Pasif)", value=True)
if use_slope_stop:
    slope_stop_pct = st.sidebar.slider("Daily Min Return Stop (%)", min_value=-10.0, max_value=10.0, value=0.5, step=0.1, help="Hissenin her gün yapması gereken minimum yüzde değişim. Bu değerin altında kalırsa satılır.")
    slope_stop_factor = slope_stop_pct / 100.0
else:
    slope_stop_factor = 0.0
start_capital = st.sidebar.number_input("Starting Capital (TL)", value=float(START_CAPITAL), step=1000.0)

# --- VERİ YÜKLEME ---
if 'force_refresh' not in st.session_state:
    st.session_state.force_refresh = False

if 'daily_scan_data' not in st.session_state:
    st.session_state.daily_scan_data = None
    
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
    
if 'single_analysis_data' not in st.session_state:
    st.session_state.single_analysis_data = None

if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = None

@st.cache_data(ttl=3600*12) # 12 saat cache
def get_data(force=False):
    tickers = get_tickers_from_file(STOX_FILE)
    if not tickers:
        return None
    
    # Veriyi indir veya cache'den yükle
    data = load_data(tickers, force_refresh=force)
    
    # Veriyi fix_cache.py'deki mantıkla düzelt
    if data is not None and not data.empty:
        data = fix_cache(data)
        
        # Eğer yeni veri indirildiyse, düzeltilmiş halini tekrar cache dosyasına yaz
        if force:
            from regression_nonperiodic import DATA_CACHE_FILE
            data.to_pickle(DATA_CACHE_FILE)
            
    return get_clean_data(data)

# Yan menüye yenileme butonu
if st.sidebar.button("🔄 Verileri Güncelle"):
    st.session_state.force_refresh = True
    st.cache_data.clear()
    st.rerun()



# To make it cleaner, I will rewrite the button block and add the input just before it.
num_indices = st.sidebar.number_input("Seçilecek Endeks Sayısı", min_value=1, max_value=50, value=10)

if st.sidebar.button("📊 Endeksleri Analiz Et ve Getir"):
    with st.sidebar.status("Endeks analizi yapılıyor...", expanded=True) as status:
        try:
            st.write(f"Script çalıştırılıyor ({num_indices} endeks)...")
            # Script yolunu bul
            script_path = os.path.join(os.path.dirname(__file__), "endeksler", "endeks1y.py")
            
            # Subprocess ile çalıştır (argüman ekle)
            result = subprocess.run([sys.executable, script_path, str(num_indices)], capture_output=True, text=True, check=True)
            
            st.write("Hisseler güncellendi.")
            status.update(label="İşlem Tamamlandı!", state="complete", expanded=False)
            
            # Verileri force refresh yap
            st.session_state.force_refresh = True
            st.cache_data.clear()
            st.rerun()
            
        except subprocess.CalledProcessError as e:
            status.update(label="Hata Oluştu!", state="error")
            st.error(f"Script hatası: {e.stderr}")
        except Exception as e:
            status.update(label="Hata Oluştu!", state="error")
            st.error(f"Hata: {str(e)}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Seçili Endeksler")
    try:
        if os.path.exists('selected_indices.json'):
            with open('selected_indices.json', 'r', encoding='utf-8') as f:
                selected_indices = json.load(f)
            
            # Güzel bir formatta göster
            st.sidebar.caption(f"Toplam {len(selected_indices)} endeks seçili:")
            for idx in selected_indices:
                st.sidebar.text(f"• {idx}")
        else:
            st.sidebar.info("Henüz endeks listesi oluşturulmadı.")
    except Exception as e:
        st.sidebar.error(f"Liste yüklenirken hata: {e}")

with st.spinner("Veriler yükleniyor..."):
    # session_state'deki force_refresh'i kullan ve sonra sıfırla
    is_force = st.session_state.force_refresh
    all_data = get_data(force=is_force)
    if is_force:
        st.session_state.force_refresh = False

if all_data is None:
    st.error(f"{STOX_FILE} bulunamadı veya hisse yok!")
    st.stop()

# --- METRİK HESAPLAMA (Vektörize) ---
# Son tarihi bul (Robust erişim)
if isinstance(all_data.columns, pd.MultiIndex):
    try:
        closes = all_data['Close']
    except:
        closes = all_data.xs('Close', axis=1, level=0)
else:
    closes = all_data

# En az bir hissenin verisi olan son gerçek günü bul
valid_dates = closes.index[closes.notna().any(axis=1)]
last_available_date = valid_dates[-1] if not valid_dates.empty else closes.index[-1]

@st.cache_data(ttl=3600) # 1 saat cache
def get_metrics(_data, _lookback):
    return get_vectorized_metrics(_data, _lookback)

with st.spinner("Metrikler hesaplanıyor..."):
    # lookback_days değişirse burası tetiklenir
    precalc = get_metrics(all_data, lookback_days)

# --- TABLAR ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Günlük Tarama", "📈 Simülasyon Backtest", "📊 Hisse Analizi", "📅 Fırsat Zaman Çizelgesi"])

# === TAB 1: GÜNLÜK TARAMA ===
with tab1:
    st.header("Bugünün Sinyalleri")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("Analiz Tarihi Seçin", value=last_available_date.date())
    
    if st.button("Taramayı Başlat"):
        target_date = pd.Timestamp(selected_date)
        st.info(f"Analiz Tarihi: {target_date.date()}")
        
        candidates = find_best_candidate(
            target_date, 
            all_data, 
            lookback_days=lookback_days, 
            max_atr_percent=atr_limit,
            min_slope=min_slope,
            min_r2=min_r2,
            precalc=precalc
        )
        st.session_state.daily_scan_data = candidates
        
    if st.session_state.daily_scan_data is not None:
        candidates = st.session_state.daily_scan_data
        filtered_candidates = candidates if candidates else []
        
        st.write(f"Bulunan Aday: {len(filtered_candidates)}")
        
        if filtered_candidates:
            df_candidates = pd.DataFrame(filtered_candidates)
            df_candidates['Ticker'] = df_candidates['t'].str.replace('.IS', '')
            
            # Formatlama
            df_display = df_candidates[['Ticker', 'price', 'slope', 'r2', 'score', 'vol_curr', 'vol_avg']].copy()
            df_display.columns = ['Ticker', 'Fiyat', 'Eğim', 'R2', 'Alım Fırsatı (%)', 'Hacim', 'Ort. Hacim']
            
            # Alım Fırsatını yüzdelik yap
            df_display['Alım Fırsatı (%)'] = df_display['Alım Fırsatı (%)'] * 100
            
            st.dataframe(df_display.style.format({
                'Fiyat': "{:.2f}",
                'Eğim': "{:.4f}",
                'R2': "{:.2f}",
                'Alım Fırsatı (%)': "{:.2f}%",
                'Hacim': "{:,.0f}",
                'Ort. Hacim': "{:,.0f}"
            }).background_gradient(subset=['Alım Fırsatı (%)'], cmap='RdYlGn'))
            
            st.success(f"🎯 En İyi Seçim: **{filtered_candidates[0]['t']}**")
        else:
            st.warning("Kriterlere uygun hisse bulunamadı.")

# === TAB 2: BACKTEST ===
with tab2:
    st.header("Geçmiş Performans Simülasyonu")
    st.write("Belirlenen parametrelerle son 1 yılın simülasyonunu çalıştırır.")
    
    if st.button("Simülasyonu Çalıştır"):
        with st.spinner("Simülasyon yapılıyor..."):
            # Simülasyonu çalıştır
            p_bar = st.progress(0)
            def p_callback(val):
                p_bar.progress(val)

            daily_vals, trade_history, final_bal = run_simulation(
                all_data, 
                lookback_days=lookback_days,
                min_slope=min_slope,
                min_r2=min_r2,
                stop_loss_rate=stop_loss,
                slope_stop_factor=slope_stop_factor,
                start_capital=start_capital,
                max_atr_percent=atr_limit,
                rebalance_freq=rebalance_freq,
                progress_callback=p_callback
            )
            p_bar.empty()
            st.session_state.simulation_data = (daily_vals, trade_history, final_bal)
            
    if st.session_state.simulation_data is not None:
        daily_vals, trade_history, final_bal = st.session_state.simulation_data
        
        roi = ((final_bal - start_capital) / start_capital) * 100
        
        # Başarı İstatistikleri
        successful_trades = 0
        unsuccessful_trades = 0
        for trade in trade_history:
            action = str(trade[4])
            if action == 'ALIS':
                continue
                
            info = str(trade[6])
            
            # Kırmızı (Başarısız) - Zarar
            if 'P/L: %-' in info:
                unsuccessful_trades += 1
            # Yeşil (Başarılı) - Kar veya SÜRE DOLDU/VOLUME STOP (Zarar değilse)
            elif 'P/L: %' in info or 'SÜRE DOLDU' in action or 'VOLUME STOP' in action:
                successful_trades += 1
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Başlangıç", f"{start_capital:,.0f} TL")
        c2.metric("Bitiş", f"{final_bal:,.0f} TL")
        c3.metric("Getiri (ROI)", f"%{roi:.2f}")
        c4.metric("Başarılı İşlem ✅", f"{successful_trades}")
        c5.metric("Başarısız İşlem ❌", f"{unsuccessful_trades}")
        
        # Grafik
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_vals.index, y=daily_vals.values, mode='lines', name='Portföy Değeri'))
        fig.add_trace(go.Scatter(x=daily_vals.index, y=[start_capital]*len(daily_vals), mode='lines', name='Başlangıç', line=dict(dash='dash', color='gray')))
        fig.update_layout(title="Portföy Gelişimi", xaxis_title="Tarih", yaxis_title="TL")
        st.plotly_chart(fig, width="stretch")
        
        # --- AYLIK GETİRİ TABLOSU ---
        st.subheader("📅 Aylık Performans Raporu")
        
        # Ay sonu değerlerini al
        monthly_resampled = daily_vals.resample('ME').last()
        monthly_returns = monthly_resampled.pct_change() * 100
        
        # İlk ayın getirisini başlangıç sermayesine göre hesapla
        if not monthly_resampled.empty:
            monthly_returns.iloc[0] = (monthly_resampled.iloc[0] / start_capital - 1) * 100
        
        df_monthly = pd.DataFrame({
            'Dönem': monthly_returns.index.strftime('%Y - %B'),
            'Net Kar/Zarar (%)': monthly_returns.values
        })
        
        # Tabloyu göster
        st.dataframe(df_monthly.style.format({
            'Net Kar/Zarar (%)': "{:+.2f}%"
        }).background_gradient(subset=['Net Kar/Zarar (%)'], cmap='RdYlGn', vmin=-15, vmax=15), width="stretch")
        
        # İşlem Geçmişi
        st.subheader("İşlem Geçmişi")
        if trade_history:
            df_hist = pd.DataFrame(trade_history, columns=["Tarih", "Hisse", "Lot", "Fiyat", "İşlem", "Nakit", "Bilgi"])
            
            def style_trades(row):
                action = str(row['İşlem'])
                if action in ['SATIS', 'STOP LOSS', 'HACIM STOP', 'VOLUME STOP', 'SLOPE STOP'] or 'SÜRE DOLDU' in action:
                    info = row['Bilgi']
                    if 'P/L: %-' in info:
                        return ['background-color: #ff4b4b; color: white; font-weight: bold'] * len(row) # Kırmızı
                    elif 'P/L: %' in info:
                        return ['background-color: #21c35a; color: white; font-weight: bold'] * len(row) # Yeşil
                    elif 'SÜRE DOLDU' in action or 'VOLUME STOP' in action:
                        return ['background-color: #21c35a; color: white; font-weight: bold'] * len(row) # Yeşil
                return [''] * len(row)

            st.dataframe(df_hist.style.apply(style_trades, axis=1))
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
        
        st.session_state.single_analysis_data = (series_window, reg_line, slope, r_value)

    if st.session_state.single_analysis_data:
        series_window, reg_line, slope, r_value = st.session_state.single_analysis_data
        
        # Grafik
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=series_window.index, y=series_window.values, mode='markers+lines', name='Fiyat'))
        fig2.add_trace(go.Scatter(x=series_window.index, y=reg_line, mode='lines', name='Exp. Regresyon', line=dict(color='orange')))
        
        st.write(f"**Slope (Eğim):** {slope:.5f}")
        st.write(f"**R-Squared:** {r_value**2:.4f}")
        
        st.plotly_chart(fig2, width="stretch")

# === TAB 4: OPPORTUNITY TIMELINE ===
with tab4:
    st.header("📅 Fırsat Zaman Çizelgesi")
    st.write("Son 1 yıl boyunca stratejiye uygun tüm alım fırsatlarını tarar ve bir zaman çizelgesinde gösterir.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        scan_step = st.slider("Tarama Sıklığı (Gün)", min_value=1, max_value=10, value=3)
        scan_lookback = st.number_input("Geriye Dönük Tarama (Gün)", value=365)
    
    if st.button("Fırsatları Tara"):
        opportunities = []
        
        # Son tarihi bul (Tab 3 yöntemindeki gibi robust erişim)
        if isinstance(all_data.columns, pd.MultiIndex):
            try:
                closes = all_data['Close']
            except:
                closes = all_data.xs('Close', axis=1, level=0)
        else:
            closes = all_data
            
        # En az bir hissenin verisi olan son gerçek günü bul
        valid_dates = closes.index[closes.notna().any(axis=1)]
        latest_date = valid_dates[-1] if not valid_dates.empty else closes.index[-1]
            
        start_scan = latest_date - timedelta(days=scan_lookback)
        scan_dates = pd.date_range(start=start_scan, end=latest_date, freq=f'{scan_step}D')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, current_date in enumerate(scan_dates):
            status_text.text(f"Taranıyor: {current_date.date()}")
            
            # find_best_candidate'e parametrelerimizi geçiriyoruz
            candidates = find_best_candidate(
                current_date, 
                all_data, 
                lookback_days=lookback_days, 
                max_atr_percent=atr_limit,
                min_slope=min_slope,
                min_r2=min_r2,
                precalc=precalc
            )
            
            # Filtreleme (İçeride yapıldığı için direkt ekliyoruz)
            if candidates:
                for c in candidates:
                    opportunities.append({
                        'Date': current_date,
                        'Ticker': c['t'].replace('.IS', ''),
                        'Price': c['price'],
                        'Slope': c['slope'],
                        'R2': c['r2'],
                        'Score': c['score'],
                        'Günlük Hacim': c.get('vol_curr', 0),
                        'Ortalama Hacim': c.get('vol_avg', 0)
                    })
            
            progress_bar.progress((idx + 1) / len(scan_dates))
        
        status_text.text("Tarama tamamlandı!")
        st.session_state.timeline_data = opportunities
        
    if st.session_state.timeline_data is not None:
        opportunities = st.session_state.timeline_data
        
        if opportunities:
            df_opps = pd.DataFrame(opportunities)
            
            # Görselleştirme: Zaman Çizelgesi (Hisse Bazlı Basit Trend)
            df_opps = df_opps.sort_values(['Ticker', 'Date'])
            
            fig3 = go.Figure()
            
            # Her ticker için ayrı bir çizgi (trace) ekleyerek daha temiz bir görünüm ve efsane (legend) sağlıyoruz
            for ticker, group in df_opps.groupby('Ticker'):
                fig3.add_trace(go.Scatter(
                    x=group['Date'],
                    y=group['Score'] * 100,
                    mode='lines+markers',
                    name=ticker,
                    marker=dict(size=6),
                    line=dict(width=2),
                    hovertemplate=(
                        f"<b>{ticker}</b><br>" +
                        "Tarih: %{x}<br>" +
                        "Skor: %{y:.2f}%<extra></extra>"
                    )
                ))
            
            fig3.update_layout(
                title="Hisse Bazlı Alım Fırsatı Trendleri",
                xaxis_title="Tarih",
                yaxis_title="Skor (%)",
                height=600,
                legend_title="Hisseler",
                hovermode="x unified" # Aynı tarihteki tüm hisseleri bir arada görmek için
            )
            
            st.plotly_chart(fig3, width="stretch")
            
            # Tablo Görünümü
            st.subheader("Fırsat Listesi")
            df_display_opps = df_opps.copy()
            df_display_opps['Score'] = df_display_opps['Score'] * 100
            df_display_opps.rename(columns={'Score': 'Alım Fırsatı (%)'}, inplace=True)
            
            st.dataframe(df_display_opps.sort_values('Date', ascending=False).style.format({
                'Price': "{:.2f}",
                'Slope': "{:.4f}",
                'R2': "{:.2f}",
                'Alım Fırsatı (%)': "{:.2f}%",
                'Günlük Hacim': "{:,.0f}",
                'Ortalama Hacim': "{:,.0f}"
            }).background_gradient(subset=['Alım Fırsatı (%)'], cmap='RdYlGn'))
        else:
            st.warning("Bu periyotta parametrelere uygun fırsat bulunamadı.")

