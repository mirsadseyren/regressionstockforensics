import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

def main():
    print("--- GELİŞMİŞ ML METRİKLERİ VE PORKCHOP ANALİZİ ---")
    
    # 1. Veri Yükleme
    print("Geçmiş işlem havuzu yükleniyor (historical_trade_metrics.csv)...")
    try:
        df = pd.read_csv('historical_trade_metrics.csv')
    except FileNotFoundError:
        print("Hata: historical_trade_metrics.csv bulunamadı.")
        return

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['slope', 'r2', 'score', 'pl_pct'])
    
    # 2. KNN ile ML Skorlarını Hesaplama
    features = ['slope', 'r2', 'score']
    X = df[features].values
    y = df['pl_pct'].values
    
    print("Özellikler ölçeklendiriliyor ve KNN çalıştırılıyor...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    K = 50
    knn = NearestNeighbors(n_neighbors=K+1, algorithm='auto', n_jobs=-1)
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_scaled)
    neighbor_indices = indices[:, 1:]
    neighbor_pls = y[neighbor_indices]
    
    exp_pls = np.mean(neighbor_pls, axis=1)
    win_rates = np.mean(neighbor_pls > 0, axis=1) * 100
    confidence_scores = (win_rates / 100) * exp_pls
    
    df['Beklenen 7G Kar (%)'] = exp_pls
    df['Kazanma İhtimali (%)'] = win_rates
    df['ML Güven Skoru'] = confidence_scores
    
    df.rename(columns={
        'pl_pct': '7 Günlük Kar (%)',
        'slope': 'Regresyon Eğimi',
        'r2': 'R2',
        'score': 'Uzaklık Skoru (Score)'
    }, inplace=True)
    
    target = '7 Günlük Kar (%)'
    metrics = [
        'Regresyon Eğimi',
        'R2',
        'Uzaklık Skoru (Score)',
        'Beklenen 7G Kar (%)',
        'Kazanma İhtimali (%)',
        'ML Güven Skoru'
    ]
    
    # Aykırı değerleri kırpma (Görselleştirme ve Bining için)
    print("Aykırı değerler kırpılıyor (%1 ve %99)...")
    plot_df = df.copy()
    
    q_low_target = plot_df[target].quantile(0.01)
    q_hi_target = plot_df[target].quantile(0.99)
    plot_df = plot_df[(plot_df[target] >= q_low_target) & (plot_df[target] <= q_hi_target)]
    
    for metric in metrics:
        q_low = plot_df[metric].quantile(0.01)
        q_hi = plot_df[metric].quantile(0.99)
        plot_df = plot_df[(plot_df[metric] >= q_low) & (plot_df[metric] <= q_hi)]

    # --- STANDARDIZE VE ÇANLAŞTIRMA (QuantileTransformer) ---
    print("Veriler korelasyon için çan eğrisine (Normal Dağılım) standardize ediliyor...")
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    normalized_data = qt.fit_transform(plot_df[metrics + [target]])
    normalized_df = pd.DataFrame(normalized_data, columns=metrics + [target], index=plot_df.index)
    
    print("\n--- Çanlaştırılmış (Normal) Veri ile Korelasyonlar ---")
    corr_matrix = normalized_df.corr()
    target_corrs = corr_matrix[target].sort_values(ascending=False)
    for index, value in target_corrs.items():
        if index != target:
            print(f"Korelasyon {index} vs {target}: {value:.4f}")
            
    # Klasör oluştur
    out_dir = 'correlation_plots/ml_metrics_advanced'
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Çanlaştırılmış (Normal Dağılım) Veri Korelasyon Matrisi', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/correlation_heatmap_normalized.png')
    plt.close()

    # --- PORKCHOP & VERİMLİLİK ISI HARİTALARI (Orijinal Değerler ile) ---
    print("\nPorkchop Yoğunluk ve Verimlilik Isı Haritaları çiziliyor...")
    
    plot_df['Is Profit'] = (plot_df[target] > 0).astype(int)
    
    for metric in metrics:
        print(f"Grafik oluşturuluyor: {metric}...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Grid ayarı
        ax1 = plt.subplot(2, 2, 1) # Çan Eğrisi
        ax2 = plt.subplot(2, 2, 2) # Verimlilik Isı Haritası
        ax3 = plt.subplot(2, 1, 2) # Porkchop Plot
        
        # 1. Çan Eğrisi (Dağılım)
        sns.histplot(plot_df[metric], kde=True, ax=ax1, color='skyblue', bins=50)
        ax1.set_title(f'{metric} Dağılımı (Çan Eğrisi)', fontsize=12, fontweight='bold')
        ax1.set_xlabel(metric)
        ax1.set_ylabel('Frekans')
        
        # 2. Verimlilik Isı Haritası
        # Metriği 10 parçaya böl (Quantile tabanlı)
        try:
            plot_df['bin'] = pd.qcut(plot_df[metric], q=10, duplicates='drop')
        except ValueError:
            # Eğer qcut başarısız olursa normal cut kullan
            plot_df['bin'] = pd.cut(plot_df[metric], bins=10)
            
        bin_stats = plot_df.groupby('bin').agg(
            Kazanma_Ihtimali=('Is Profit', lambda x: x.mean() * 100),
            Ortalama_Kar=(target, 'mean'),
            Islem_Sayisi=('Is Profit', 'count')
        ).reset_index()
        
        # Formatlama: Aralıkları string yap
        bin_stats['bin'] = bin_stats['bin'].astype(str)
        bin_stats.set_index('bin', inplace=True)
        
        # Isı haritası için sadece Kazanma İhtimali ve Ortalama Kar
        heatmap_data = bin_stats[['Kazanma_Ihtimali', 'Ortalama_Kar']]
        
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='Greens', ax=ax2, linewidths=.5, cbar=True)
        ax2.set_title(f'Sayı Değerlerine Göre Verimlilik (Isı Haritası)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sayı Değeri Aralıkları')
        ax2.set_xlabel('Performans Metrikleri')
        
        # 3. Porkchop Yoğunluk Haritası (2D KDE / Histplot)
        # Yoğun noktaları renklendirmek için 2D Histogram ve KDE
        sns.histplot(data=plot_df, x=metric, y=target, bins=50, pmax=.9, cmap="turbo", cbar=True, ax=ax3, cbar_kws={'label': 'İşlem Yoğunluğu'})
        
        # Ortalamayı gösteren trend çizgisi
        sns.regplot(data=plot_df.sample(min(5000, len(plot_df))), x=metric, y=target, scatter=False, color='red', line_kws={"linewidth": 3}, ax=ax3)
        
        # Yatay sıfır çizgisi (Kar/Zarar sınırı)
        ax3.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        
        ax3.set_title(f'Porkchop Yoğunlaşma Haritası: {metric} vs {target}', fontsize=14, fontweight='bold')
        ax3.set_xlabel(f"{metric} (Orijinal Sayı Değerleri)", fontsize=12)
        ax3.set_ylabel(target, fontsize=12)
        
        plt.tight_layout()
        safe_metric_name = metric.split('(')[0].strip().replace('%', 'Pct').replace('/', '_').replace(' ', '_').replace('İ', 'I').replace('ı', 'i').replace('ğ', 'g')
        filename = f"{out_dir}/{safe_metric_name}_advanced_analysis.png"
        plt.savefig(filename, dpi=150)
        plt.close()
    
    print(f"\nİşlem Tamam! Tüm gelişmiş grafikler '{out_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
