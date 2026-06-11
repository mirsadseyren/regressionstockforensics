import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import warnings

warnings.filterwarnings('ignore')

from regression_nonperiodic import (
    load_data, get_tickers_from_file, get_vectorized_metrics, STOX_FILE
)

def fetch_financials(tickers):
    financial_data = []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        print(f"Fetching data for {ticker} ({i+1}/{total})...")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            data = {
                'Ticker': ticker,
                'PD/DD (Price to Book)': info.get('priceToBook', np.nan),
                'FD/FAVÖK (EV/EBITDA)': info.get('enterpriseToEbitda', np.nan),
                'F/K (Trailing PE)': info.get('trailingPE', np.nan),
                'FAVÖK Marjı (EBITDA Margin)': info.get('ebitdaMargins', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Ortalama Hacim (Avg Volume)': info.get('averageVolume', np.nan)
            }
            financial_data.append(data)
            time.sleep(0.1) # Small delay to avoid rate limiting
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            
    return pd.DataFrame(financial_data)

def main():
    # 1. Load trade results
    print("Loading test_trades_results.csv...")
    try:
        df = pd.read_csv('test_trades_results.csv', sep=';', decimal=',')
    except FileNotFoundError:
        print("Error: test_trades_results.csv not found.")
        return

    # Ensure numeric columns
    df['Confidence Score'] = pd.to_numeric(df['Confidence Score'], errors='coerce')
    df['Actual P/L (%)'] = pd.to_numeric(df['Actual P/L (%)'], errors='coerce')
    df['Expected P/L (%)'] = pd.to_numeric(df['Expected P/L (%)'], errors='coerce')
    
    # Derive Win Rate
    df['Kazanma İhtimali (%)'] = (df['Confidence Score'] / df['Expected P/L (%)']) * 100
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Drop rows without targets
    df = df.dropna(subset=['Confidence Score', 'Actual P/L (%)'])

    # 2. Extract slope and r2 from historical data
    print("Borsa verileri yükleniyor ve vektörize metrikler (Eğim, R2) hesaplanıyor...")
    try:
        tickers = get_tickers_from_file(STOX_FILE)
        all_data = load_data(tickers)
        precalc = get_vectorized_metrics(all_data, lookback_days=20)
        
        slopes_df = precalc['slopes']
        r2_df = precalc['r2']
        
        slopes_list = []
        r2_list = []
        
        for idx, row in df.iterrows():
            date = row['Date']
            ticker = row['Ticker']
            
            try:
                date_idx = slopes_df.index.get_indexer([date], method='pad')[0]
                if date_idx >= 0 and ticker in slopes_df.columns:
                    slope_val = slopes_df.iloc[date_idx][ticker]
                    r2_val = r2_df.iloc[date_idx][ticker]
                else:
                    slope_val = np.nan
                    r2_val = np.nan
            except Exception:
                slope_val = np.nan
                r2_val = np.nan
                
            slopes_list.append(slope_val)
            r2_list.append(r2_val)
            
        df['Eğim (Slope)'] = slopes_list
        df['R2'] = r2_list
        print("Eğim ve R2 başarıyla eklendi.")
    except Exception as e:
        print(f"Eğim ve R2 hesaplanırken hata oluştu: {e}")
        df['Eğim (Slope)'] = np.nan
        df['R2'] = np.nan

    # 3. Get unique tickers
    unique_tickers = df['Ticker'].unique()
    
    # 4. Fetch financial data from yfinance
    print(f"\nFound {len(unique_tickers)} unique tickers. Fetching financial data...")
    financials_df = fetch_financials(unique_tickers)
    
    # 5. Merge datasets
    merged_df = pd.merge(df, financials_df, on='Ticker', how='left')
    
    # 6. Metrics to analyze against Actual P/L (%)
    metrics = [
        'PD/DD (Price to Book)', 
        'FD/FAVÖK (EV/EBITDA)', 
        'F/K (Trailing PE)', 
        'FAVÖK Marjı (EBITDA Margin)',
        'ROE',
        'ROA',
        'Eğim (Slope)',
        'R2',
        'Confidence Score',
        'Expected P/L (%)',
        'Kazanma İhtimali (%)',
        'Ortalama Hacim (Avg Volume)'
    ]
    
    target = 'Actual P/L (%)'
    
    # --- NORMALIZATION STEP ---
    print("\nVeriler araştırma için normalize ediliyor (Z-Score Standardization)...")
    normalized_df = merged_df.copy()
    cols_to_normalize = metrics + [target]
    
    for col in cols_to_normalize:
        normalized_df[col] = normalized_df[col].replace([np.inf, -np.inf], np.nan)
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val != 0 and not pd.isna(std_val):
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            
    # --- CORRELATION CALCULATION & COMPOSITE SCORE ---
    print(f"\n--- Correlations with {target} (Normalized Data) ---")
    correlations = {}
    for metric in metrics:
        temp_df = normalized_df[[metric, target]].dropna()
        if len(temp_df) > 5:
            corr = temp_df[metric].corr(temp_df[target])
            correlations[metric] = corr
            print(f"Correlation {metric} vs {target}: {corr:.4f}")
        else:
            correlations[metric] = 0.0
            print(f"Correlation {metric} vs {target}: Insufficient data (0.0)")

    print("\nKorelasyon ağırlıklı Finansal Güven Skoru hesaplanıyor...")
    # Finansal Güven Skoru sıfırdan başlatalım
    normalized_df['Finansal Güven Skoru'] = 0.0
    
    # Sadece finansal metrikleri dahil edelim (Teknik ve küme metrikleri hariç)
    financial_metrics = [
        'PD/DD (Price to Book)', 
        'FD/FAVÖK (EV/EBITDA)', 
        'F/K (Trailing PE)', 
        'FAVÖK Marjı (EBITDA Margin)',
        'ROE',
        'ROA',
        'Ortalama Hacim (Avg Volume)'
    ]
    
    # Her finansal metriği (korelasyon * 100) ile ağırlıklandırıp topla.
    for metric in financial_metrics:
        corr = correlations.get(metric, 0.0)
        if pd.isna(corr):
            corr = 0.0
        
        weight = corr * 100
        metric_norm = normalized_df[metric].fillna(0)
        normalized_df['Finansal Güven Skoru'] += weight * metric_norm
        
    # İki skoru birbiriyle çarpıyoruz (Finansal Güven Skoru * Küme Güven Skoru)
    confidence_norm = normalized_df['Confidence Score'].fillna(0)
    normalized_df['Finansal x Küme Güveni'] = normalized_df['Finansal Güven Skoru'] * confidence_norm
    
    # Yeni metrikleri listeye ekleyelim ki onlar için de plot çizilsin
    metrics.append('Finansal Güven Skoru')
    metrics.append('Finansal x Küme Güveni')
    
    print("\n--- Yeni Kompozit Skorların Korelasyonları ---")
    for metric in ['Finansal Güven Skoru', 'Finansal x Küme Güveni']:
        temp_df = normalized_df[[metric, target]].dropna()
        if len(temp_df) > 5:
            corr = temp_df[metric].corr(temp_df[target])
            print(f"Correlation {metric} vs {target}: {corr:.4f}")
    
    # 8. Draw scatter plots for NORMALIZED data
    print("\nGenerating scatter plots for normalized data...")
    os.makedirs('correlation_plots/normalized', exist_ok=True)
    
    for metric in metrics:
        temp_df = normalized_df.dropna(subset=[metric, target]).copy()
        
        # Remove extreme outliers for better visualization on normalized scale
        if not temp_df.empty and len(temp_df) > 10:
            q_low = temp_df[metric].quantile(0.01)
            q_hi  = temp_df[metric].quantile(0.99)
            temp_df = temp_df[(temp_df[metric] <= q_hi) & (temp_df[metric] >= q_low)]
        
        if not temp_df.empty and len(temp_df) > 1:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=temp_df, x=metric, y=target, alpha=0.6)
            
            # Add trendline
            sns.regplot(data=temp_df, x=metric, y=target, scatter=False, color='red')
            corr = temp_df[metric].corr(temp_df[target])
            plt.title(f'NORMALIZED: {metric} vs {target}\nCorr: {corr:.2f}')
            plt.xlabel(f"{metric} (Z-Score)")
            plt.ylabel(f"{target} (Z-Score)")
            
            plt.tight_layout()
            safe_metric_name = metric.split('(')[0].strip().replace('/', '_').replace(' ', '_').replace('İ', 'I').replace('ı', 'i')
            filename = f"correlation_plots/normalized/{safe_metric_name}_vs_Kar_normalized_scatter.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot: {filename}")

    # Ekstra: Finansal Güven Skoru vs Expected P/L (%)
    print("\nEkstra grafik çiziliyor: Finansal Güven Skoru vs Expected P/L (%)")
    target2 = 'Expected P/L (%)'
    temp_df = normalized_df[['Finansal Güven Skoru', target2]].dropna()
    if not temp_df.empty and len(temp_df) > 1:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=temp_df, x='Finansal Güven Skoru', y=target2, alpha=0.6)
        sns.regplot(data=temp_df, x='Finansal Güven Skoru', y=target2, scatter=False, color='red')
        corr = temp_df['Finansal Güven Skoru'].corr(temp_df[target2])
        plt.title(f'NORMALIZED: Finansal Güven Skoru vs {target2}\nCorr: {corr:.2f}')
        plt.xlabel('Finansal Güven Skoru (Z-Score)')
        plt.ylabel(f'{target2} (Z-Score)')
        plt.tight_layout()
        filename = "correlation_plots/normalized/Finansal_Guven_Skoru_vs_Expected_PL_scatter.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")

    # Save the datasets (both original and normalized)
    merged_df.to_csv('correlation_plots/merged_data_with_financials_raw.csv', index=False, sep=';', decimal=',')
    normalized_df.to_csv('correlation_plots/normalized/merged_data_with_financials_normalized.csv', index=False, sep=';', decimal=',')
    print("\nDone! Normalized plots and datasets are saved in the 'correlation_plots' and 'correlation_plots/normalized' folders.")

if __name__ == "__main__":
    main()
