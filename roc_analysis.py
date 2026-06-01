"""
test_trades_results.csv için ROC Curve, Korelasyon ve Yanılma İstatistikleri Analizi
=====================================================================================
- Confidence Score → tahmin skoru (sürekli değişken)
- Is Profit        → gerçek sonuç (True / False)
- Met Expectation  → beklentiyi karşılayıp karşılamadığı
- Korelasyon analizi: Confidence Score ile diğer metrikler arası ilişki
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI backend gerektirmez
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")

# ── Türkçe ondalık ayırıcı ile CSV okuma ──
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_trades_results.csv")

df = pd.read_csv(CSV_PATH, sep=";", decimal=",")

# Boolean sütunları düzelt
df["Is Profit"]        = df["Is Profit"].astype(str).str.strip().str.lower() == "true"
df["Met Expectation"]  = df["Met Expectation"].astype(str).str.strip().str.lower() == "true"

# Sayısal sütunları garanti et
for col in ["Confidence Score", "Expected P/L (%)", "Actual P/L (%)", "Deviation"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["Confidence Score", "Is Profit"], inplace=True)

# ── Etiketler ──
y_true   = df["Is Profit"].astype(int).values
scores   = df["Confidence Score"].values
y_true_met = df["Met Expectation"].astype(int).values

# ═══════════════════════════════════════════════════════════
#  ROC CURVE  – Is Profit
# ═══════════════════════════════════════════════════════════
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

j_scores  = tpr - fpr
best_idx  = np.argmax(j_scores)
best_thresh = thresholds[best_idx]

# ═══════════════════════════════════════════════════════════
#  ROC CURVE  – Met Expectation
# ═══════════════════════════════════════════════════════════
fpr_met, tpr_met, thresholds_met = roc_curve(y_true_met, scores)
roc_auc_met = auc(fpr_met, tpr_met)

j_met      = tpr_met - fpr_met
best_idx_m = np.argmax(j_met)
best_thresh_met = thresholds_met[best_idx_m]

# ═══════════════════════════════════════════════════════════
#  Precision-Recall Curve – Is Profit
# ═══════════════════════════════════════════════════════════
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, scores)
avg_prec = average_precision_score(y_true, scores)

# ═══════════════════════════════════════════════════════════
#  Optimal eşikte Confusion Matrix
# ═══════════════════════════════════════════════════════════
y_pred_opt = (scores >= best_thresh).astype(int)
cm = confusion_matrix(y_true, y_pred_opt)

tn, fp, fn, tp = cm.ravel()
accuracy    = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0
ppv         = tp / (tp + fp) if (tp + fp) else 0
npv         = tn / (tn + fn) if (tn + fn) else 0
f1          = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0
fpr_val     = fp / (fp + tn) if (fp + tn) else 0
fnr_val     = fn / (fn + tp) if (fn + tp) else 0

# ═══════════════════════════════════════════════════════════
#  KORELASYON ANALİZİ – Confidence Score ile
# ═══════════════════════════════════════════════════════════
corr_cols = {
    "Expected P/L (%)": df["Expected P/L (%)"],
    "Actual P/L (%)":   df["Actual P/L (%)"],
    "Deviation":        df["Deviation"],
    "Is Profit (int)":  y_true,
    "Met Expectation (int)": y_true_met,
}

print("=" * 70)
print("  TEST TRADES – YANILMA İSTATİSTİKLERİ & ROC ANALİZİ")
print("=" * 70)
print(f"\n  Toplam işlem sayısı         : {len(df)}")
print(f"  Kârlı işlem (Is Profit=True): {y_true.sum()}  ({y_true.mean()*100:.1f}%)")
print(f"  Zararlı işlem               : {(1-y_true).sum()}  ({(1-y_true).mean()*100:.1f}%)")
print(f"  Beklenti karşılanan         : {y_true_met.sum()}  ({y_true_met.mean()*100:.1f}%)")

print(f"\n{'─'*70}")
print(f"  ROC AUC (Is Profit)         : {roc_auc:.4f}")
print(f"  ROC AUC (Met Expectation)   : {roc_auc_met:.4f}")
print(f"  Average Precision (Is Profit): {avg_prec:.4f}")

print(f"\n{'─'*70}")
print(f"  Optimal Eşik (Youden's J)   : {best_thresh:.4f}")
print(f"  ► Accuracy                  : {accuracy*100:.2f}%")
print(f"  ► Sensitivity (TPR/Recall)  : {sensitivity*100:.2f}%")
print(f"  ► Specificity (TNR)         : {specificity*100:.2f}%")
print(f"  ► Precision (PPV)           : {ppv*100:.2f}%")
print(f"  ► NPV                       : {npv*100:.2f}%")
print(f"  ► F1 Score                  : {f1:.4f}")
print(f"  ► False Positive Rate (FPR) : {fpr_val*100:.2f}%")
print(f"  ► False Negative Rate (FNR) : {fnr_val*100:.2f}%")

print(f"\n{'─'*70}")
print("  Confusion Matrix (Optimal Eşik = {:.4f})".format(best_thresh))
print(f"                    Tahmin: Zarar   Tahmin: Kâr")
print(f"  Gerçek: Zarar       {tn:>5}          {fp:>5}")
print(f"  Gerçek: Kâr         {fn:>5}          {tp:>5}")

# ── Korelasyonlar ──
print(f"\n{'─'*70}")
print("  CONFIDENCE SCORE KORELASYON ANALİZİ")
print(f"{'─'*70}")
print(f"  {'Değişken':<25} {'Pearson r':>10} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12}")
print(f"  {'─'*70}")

for name, values in corr_cols.items():
    mask = ~np.isnan(values) & ~np.isnan(scores)
    if mask.sum() < 3:
        continue
    pr, pp = stats.pearsonr(scores[mask], values[mask])
    sr, sp = stats.spearmanr(scores[mask], values[mask])
    sig_p = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
    sig_s = "***" if sp < 0.001 else ("**" if sp < 0.01 else ("*" if sp < 0.05 else ""))
    print(f"  {name:<25} {pr:>9.4f}{sig_p:<1} {pp:>12.2e} {sr:>11.4f}{sig_s:<1} {sp:>12.2e}")

# Point-biserial korelasyon (Is Profit binary vs Confidence Score)
rpb, ppb = stats.pointbiserialr(y_true, scores)
print(f"\n  Point-Biserial (Is Profit ~ CS): r = {rpb:.4f}, p = {ppb:.2e}")

rpb_met, ppb_met = stats.pointbiserialr(y_true_met, scores)
print(f"  Point-Biserial (Met Exp ~ CS)  : r = {rpb_met:.4f}, p = {ppb_met:.2e}")

# ═══════════════════════════════════════════════════════════
#  Confidence Score aralıklarına göre doğruluk
# ═══════════════════════════════════════════════════════════
bins = [0, 5, 8, 10, 13, 16, 20, 30, 100]
labels = ["0-5", "5-8", "8-10", "10-13", "13-16", "16-20", "20-30", "30+"]
df["CS_Bin"] = pd.cut(df["Confidence Score"], bins=bins, labels=labels, right=False)

print(f"\n{'─'*70}")
print("  Confidence Score Aralığına Göre Başarı Oranları")
print(f"{'─'*70}")
print(f"  {'Aralık':<10} {'İşlem':>6} {'Kârlı':>6} {'Kâr%':>7} {'BeklKarş':>8} {'BeklKarş%':>9} {'Ort.Getiri':>10}")
print(f"  {'─'*62}")

for label in labels:
    subset = df[df["CS_Bin"] == label]
    if len(subset) == 0:
        continue
    n       = len(subset)
    n_prof  = subset["Is Profit"].sum()
    pct_p   = n_prof / n * 100
    n_met   = subset["Met Expectation"].sum()
    pct_m   = n_met / n * 100
    avg_ret = subset["Actual P/L (%)"].mean()
    print(f"  {label:<10} {n:>6} {n_prof:>6} {pct_p:>6.1f}% {n_met:>8} {pct_m:>8.1f}% {avg_ret:>9.2f}%")

print("=" * 70)

# ═══════════════════════════════════════════════════════════
#  GÖRSELLEŞTIRME  (6-panel figure)
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
gs  = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.28)

colors_p = {
    "bg":      "#0d1117",
    "card":    "#161b22",
    "accent1": "#58a6ff",
    "accent2": "#f0883e",
    "accent3": "#3fb950",
    "accent4": "#bc8cff",
    "accent5": "#f778ba",
    "accent6": "#79c0ff",
    "grid":    "#21262d",
    "text":    "#c9d1d9",
    "subtext": "#8b949e",
    "diag":    "#484f58",
}

def style_ax(ax, title):
    ax.set_facecolor(colors_p["card"])
    ax.set_title(title, color=colors_p["text"], fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=colors_p["subtext"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(colors_p["grid"])
    ax.grid(True, alpha=0.15, color=colors_p["grid"])

# ── 1) ROC Curve – Is Profit ──
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, f"ROC Curve – Is Profit  (AUC = {roc_auc:.3f})")
ax1.fill_between(fpr, tpr, alpha=0.15, color=colors_p["accent1"])
ax1.plot(fpr, tpr, color=colors_p["accent1"], lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
ax1.plot([0, 1], [0, 1], ls="--", color=colors_p["diag"], lw=1, label="Rastgele (AUC = 0.5)")
ax1.scatter(fpr[best_idx], tpr[best_idx], s=120, color=colors_p["accent2"], zorder=5,
            edgecolors="white", linewidths=1.5,
            label=f"Optimal Eşik = {best_thresh:.2f}")
ax1.set_xlabel("False Positive Rate (FPR)", color=colors_p["subtext"], fontsize=10)
ax1.set_ylabel("True Positive Rate (TPR)", color=colors_p["subtext"], fontsize=10)
ax1.legend(loc="lower right", fontsize=9, facecolor=colors_p["card"],
           edgecolor=colors_p["grid"], labelcolor=colors_p["text"])
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)

# ── 2) ROC Curve – Met Expectation ──
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, f"ROC Curve – Met Expectation  (AUC = {roc_auc_met:.3f})")
ax2.fill_between(fpr_met, tpr_met, alpha=0.15, color=colors_p["accent4"])
ax2.plot(fpr_met, tpr_met, color=colors_p["accent4"], lw=2.5, label=f"ROC (AUC = {roc_auc_met:.3f})")
ax2.plot([0, 1], [0, 1], ls="--", color=colors_p["diag"], lw=1, label="Rastgele (AUC = 0.5)")
ax2.scatter(fpr_met[best_idx_m], tpr_met[best_idx_m], s=120, color=colors_p["accent2"], zorder=5,
            edgecolors="white", linewidths=1.5,
            label=f"Optimal Eşik = {best_thresh_met:.2f}")
ax2.set_xlabel("False Positive Rate (FPR)", color=colors_p["subtext"], fontsize=10)
ax2.set_ylabel("True Positive Rate (TPR)", color=colors_p["subtext"], fontsize=10)
ax2.legend(loc="lower right", fontsize=9, facecolor=colors_p["card"],
           edgecolor=colors_p["grid"], labelcolor=colors_p["text"])
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)

# ── 3) Precision-Recall Curve ──
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, f"Precision-Recall Curve  (AP = {avg_prec:.3f})")
ax3.fill_between(recall_vals, precision_vals, alpha=0.15, color=colors_p["accent3"])
ax3.plot(recall_vals, precision_vals, color=colors_p["accent3"], lw=2.5,
         label=f"PR Curve (AP = {avg_prec:.3f})")
baseline = y_true.mean()
ax3.axhline(y=baseline, ls="--", color=colors_p["diag"], lw=1,
            label=f"Rastgele Baseline = {baseline:.2f}")
ax3.set_xlabel("Recall", color=colors_p["subtext"], fontsize=10)
ax3.set_ylabel("Precision", color=colors_p["subtext"], fontsize=10)
ax3.legend(loc="upper right", fontsize=9, facecolor=colors_p["card"],
           edgecolor=colors_p["grid"], labelcolor=colors_p["text"])
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(0, 1.05)

# ── 4) Confusion Matrix Heatmap ──
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, f"Confusion Matrix  (Eşik = {best_thresh:.2f})")

cm_norm = cm.astype(float) / cm.sum()
im = ax4.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=cm_norm.max() * 1.2)

labels_cm = [["TN\n(Doğru Zarar)", "FP\n(Yanlış Kâr)"],
             ["FN\n(Kaçırılan Kâr)", "TP\n(Doğru Kâr)"]]

for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = cm_norm[i, j] * 100
        text_color = "white" if cm_norm[i, j] > cm_norm.max() * 0.5 else colors_p["text"]
        ax4.text(j, i, f"{labels_cm[i][j]}\n{val}  ({pct:.1f}%)",
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color=text_color)

ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Zarar (0)", "Kâr (1)"], color=colors_p["subtext"], fontsize=10)
ax4.set_yticklabels(["Zarar (0)", "Kâr (1)"], color=colors_p["subtext"], fontsize=10)
ax4.set_xlabel("Tahmin", color=colors_p["subtext"], fontsize=11)
ax4.set_ylabel("Gerçek", color=colors_p["subtext"], fontsize=11)

# ── 5) Scatter: Confidence Score vs Actual P/L ──
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "Confidence Score vs Actual P/L (%)")

mask_profit = df["Is Profit"].values
mask_loss   = ~df["Is Profit"].values

ax5.scatter(df.loc[mask_loss, "Confidence Score"],
            df.loc[mask_loss, "Actual P/L (%)"],
            c=colors_p["accent2"], alpha=0.45, s=25, label="Zarar", edgecolors="none")
ax5.scatter(df.loc[mask_profit, "Confidence Score"],
            df.loc[mask_profit, "Actual P/L (%)"],
            c=colors_p["accent3"], alpha=0.45, s=25, label="Kâr", edgecolors="none")

# Trend line
z = np.polyfit(scores, df["Actual P/L (%)"].values, 1)
p = np.poly1d(z)
x_line = np.linspace(scores.min(), scores.max(), 100)
ax5.plot(x_line, p(x_line), color=colors_p["accent5"], lw=2, ls="--",
         label=f"Trend (y = {z[0]:.2f}x + {z[1]:.1f})")
ax5.axhline(0, color=colors_p["diag"], lw=0.8, ls=":")

ax5.set_xlabel("Confidence Score", color=colors_p["subtext"], fontsize=10)
ax5.set_ylabel("Actual P/L (%)", color=colors_p["subtext"], fontsize=10)
ax5.legend(loc="upper left", fontsize=9, facecolor=colors_p["card"],
           edgecolor=colors_p["grid"], labelcolor=colors_p["text"])

# ── 6) Confidence Score aralığına göre bar chart ──
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "Confidence Score Aralığına Göre Kâr/Beklenti Oranı")

bin_data = []
for label in labels:
    subset = df[df["CS_Bin"] == label]
    if len(subset) == 0:
        bin_data.append((label, 0, 0, 0))
        continue
    n = len(subset)
    pct_p = subset["Is Profit"].sum() / n * 100
    pct_m = subset["Met Expectation"].sum() / n * 100
    bin_data.append((label, n, pct_p, pct_m))

x_pos = np.arange(len(bin_data))
width = 0.35
bars1 = ax6.bar(x_pos - width/2, [d[2] for d in bin_data], width,
                color=colors_p["accent3"], alpha=0.8, label="Kâr Oranı %")
bars2 = ax6.bar(x_pos + width/2, [d[3] for d in bin_data], width,
                color=colors_p["accent4"], alpha=0.8, label="Beklenti Karşılama %")

# Üst etiketler (işlem sayısı)
for i, d in enumerate(bin_data):
    if d[1] > 0:
        ax6.text(i, max(d[2], d[3]) + 2, f"n={d[1]}",
                 ha="center", va="bottom", fontsize=8, color=colors_p["subtext"])

ax6.set_xticks(x_pos)
ax6.set_xticklabels([d[0] for d in bin_data], color=colors_p["subtext"], fontsize=9)
ax6.set_xlabel("Confidence Score Aralığı", color=colors_p["subtext"], fontsize=10)
ax6.set_ylabel("Oran (%)", color=colors_p["subtext"], fontsize=10)
ax6.legend(loc="upper left", fontsize=9, facecolor=colors_p["card"],
           edgecolor=colors_p["grid"], labelcolor=colors_p["text"])
ax6.set_ylim(0, 100)

# ── Başlık ──
fig.suptitle("Test Trades – Yanılma İstatistikleri, ROC & Korelasyon Analizi",
             color="white", fontsize=17, fontweight="bold", y=0.99)

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roc_analysis_output.png")
plt.savefig(SAVE_PATH, dpi=180, bbox_inches="tight", facecolor=colors_p["bg"])
print(f"\n  ✅ Grafik kaydedildi → {SAVE_PATH}")
print("\n  Not: plt.show() kullanmak isterseniz Agg backend yerine TkAgg kullanın.")
