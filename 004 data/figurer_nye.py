"""
LOG650 – Tilleggsvisualisering (Figur 3–6)

Produserer fire nye figurer basert på CSV-filer fra analyse.py.
Kjør analyse.py først, deretter dette skriptet.

Krever:
  resultater_segment_sesong.csv
  resultater_segment_volum.csv
  resultater_sku_mape.csv
  resultater_xgboost_individuell.csv
  resultater_maaned.csv
  eda_resultater.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR   = "."
FS_THRESHOLD = 0.64

MODEL_ORDER  = ["Naiv", "Holt-Winters", "ARIMA", "XGBoost"]
MODEL_LABELS = ["Naiv", "Holt-Winters", "ARIMA", "XGBoost\n(global)"]
MODEL_SHORT  = {"Naiv": "Naiv", "Holt-Winters": "HW", "ARIMA": "ARIMA", "XGBoost": "XGBoost"}

MODEL_COLORS = {
    "Naiv":         "#aaaaaa",
    "Holt-Winters": "#5b9bd5",
    "ARIMA":        "#ed7d31",
    "XGBoost":      "#70ad47",
    "XGBoost_ind":  "#9dc3e6",
}


# ── Figur 3: Gruppert søylediagram — MAPE per segment ─────────────────────────
def fig3_segment_bars():
    """
    To paneler side om side:
      A. Sesongstyrke (tydelig vs. svak/ingen sesong)
      B. Volumnivå (høyt / middels / lavt)
    Visualiserer Tabell 3 og 4 fra rapporten som grupperte søylediagram.
    """
    sesong = pd.read_csv(f"{OUTPUT_DIR}/resultater_segment_sesong.csv")
    volum  = pd.read_csv(f"{OUTPUT_DIR}/resultater_segment_volum.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figur 3: Median MAPE per modell — segmentert analyse (testperiode Jan–Des 2025)",
        fontsize=12, fontweight="bold"
    )

    x = np.arange(len(MODEL_ORDER))

    # ── Panel A: Sesongstyrke ──────────────────────────────────────────────────
    seg_pivot = sesong.pivot(index="Modell", columns="Segment", values="Median_MAPE")
    seg_pivot = seg_pivot.reindex(MODEL_ORDER)

    ses_segs = ["Tydelig sesong", "Svak/ingen sesong"]
    ses_cols = ["#1b4f72", "#7fb3d3"]
    bar_w = 0.35

    for j, (seg, col) in enumerate(zip(ses_segs, ses_cols)):
        if seg not in seg_pivot.columns:
            continue
        offset = (j - 0.5) * bar_w
        vals = seg_pivot[seg].values
        bars = ax1.bar(x + offset, vals, bar_w, label=seg,
                       color=col, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.8,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax1.set_ylabel("Median MAPE (%)", fontsize=10)
    ax1.set_title("A. Sesongstyrke", fontsize=11, pad=8)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax1.set_axisbelow(True)

    # ── Panel B: Volumnivå ─────────────────────────────────────────────────────
    vol_pivot = volum.pivot(index="Modell", columns="Segment", values="Median_MAPE")
    vol_pivot = vol_pivot.reindex(MODEL_ORDER)

    vol_segs = ["Høyt volum", "Middels volum", "Lavt volum"]
    vol_cols = ["#6e2f6e", "#a569bd", "#d7bde2"]
    bar_w2 = 0.22

    for j, (seg, col) in enumerate(zip(vol_segs, vol_cols)):
        if seg not in vol_pivot.columns:
            continue
        offset = (j - 1) * bar_w2
        vals = vol_pivot[seg].values
        bars = ax2.bar(x + offset, vals, bar_w2, label=seg,
                       color=col, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.8,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax2.set_ylabel("Median MAPE (%)", fontsize=10)
    ax2.set_title("B. Volumnivå", fontsize=11, pad=8)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figur3_segment_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Lagret: figur3_segment_bars.png")


# ── Figur 4: Boksplott — MAPE-fordeling per modell ────────────────────────────
def fig4_boxplot():
    """
    Viser full fordeling av MAPE over alle 105 SKU-er per modell,
    inkludert XGBoost individuell. Supplerer Tabell 2 ved å synliggjøre
    spredning og uteliggere utover medianverdiene.
    """
    sku_df = pd.read_csv(f"{OUTPUT_DIR}/resultater_sku_mape.csv")
    ind_df = pd.read_csv(f"{OUTPUT_DIR}/resultater_xgboost_individuell.csv")

    keys   = ["Naiv", "Holt-Winters", "ARIMA", "XGBoost", "XGBoost_ind"]
    labels = ["Naiv", "Holt-Winters", "ARIMA", "XGBoost\n(global)", "XGBoost\n(individuell)"]
    colors = [MODEL_COLORS[k] for k in keys]

    data = []
    for key in keys:
        if key == "XGBoost_ind":
            vals = ind_df["MAPE"].dropna().values
        else:
            vals = sku_df[key].dropna().values
        data.append(vals)

    # Kapp Y-akse ved 85. persentil — viser boksene tydelig, kutter ekstremverdier
    all_vals = np.concatenate(data)
    ylim_top = np.nanpercentile(all_vals, 85) * 1.2
    n_hidden = sum(v > ylim_top for v in all_vals)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.45, markeredgecolor="none"),
        widths=0.55,
        showfliers=False,   # Ekskluder individuelle uteligger-punkter for ryddighet
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.82)

    # Median-annoteringer over hver boks
    for i, d in enumerate(data):
        med = np.nanmedian(d)
        q75 = np.nanpercentile(d, 75)
        ax.text(i + 1, q75 + ylim_top * 0.03,
                f"Median: {med:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Referanselinje ved samlet median for alle modeller
    overall_median = np.nanmedian(all_vals)
    ax.axhline(overall_median, color="gray", linestyle=":", linewidth=1.2,
               label=f"Samlet median ({overall_median:.1f}%)", alpha=0.7)
    ax.legend(fontsize=9, loc="upper right")

    ax.set_ylabel("MAPE (%)", fontsize=11)
    ax.set_ylim(0, ylim_top)
    ax.set_title(
        "Figur 4: Fordeling av MAPE per modell — alle 105 SKU-er (testperiode Jan–Des 2025)",
        fontsize=12, fontweight="bold"
    )
    note = f"Merk: {n_hidden} ekstreme observasjoner over {ylim_top:.0f}% er utelatt for lesbarhet."
    ax.text(0.5, -0.08, note, transform=ax.transAxes,
            ha="center", fontsize=8.5, color="gray")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figur4_mape_boxplot.png", dpi=150)
    plt.close()
    print("  Lagret: figur4_mape_boxplot.png")


# ── Figur 5: Linjediagram — månedlig MAPE i testperioden ──────────────────────
def fig5_monthly_errors():
    """
    Viser median MAPE per måned (Jan–Des 2025) for de fire hovedmodellene.
    Avdekker om prognosefeil er sesongavhengige — f.eks. om visse måneder
    er systematisk vanskeligere å predikere enn andre.
    """
    df = pd.read_csv(f"{OUTPUT_DIR}/resultater_maaned.csv")

    months      = df["Maaned"].tolist()
    x           = np.arange(len(months))
    line_styles = ["-o", "-s", "-^", "-D"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for model_name, ls in zip(MODEL_ORDER, line_styles):
        col = f"{model_name}_MAPE"
        if col not in df.columns:
            continue
        vals  = df[col].values
        label = "XGBoost (global)" if model_name == "XGBoost" else model_name
        ax.plot(x, vals, ls,
                label=label,
                color=MODEL_COLORS[model_name],
                linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=10)
    ax.set_ylabel("Median MAPE (%)", fontsize=11)
    ax.set_xlabel("Testmåned (2025)", fontsize=10)
    ax.set_title(
        "Figur 5: Månedlig median MAPE per modell — testperiode Jan–Des 2025",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figur5_maaned_mape.png", dpi=150)
    plt.close()
    print("  Lagret: figur5_maaned_mape.png")


# ── Figur 6: Varmekart — beste modell per kombinert segment ───────────────────
def fig6_heatmap():
    """
    2×3-matrise: sesongstyrke (rad) × volumnivå (kolonne).
    Farge = beste modell, tekst = modellnavn + MAPE + antall SKU-er.
    Gjør hybridstrategien fra avsnitt 8.3 visuelt umiddelbar.
    """
    sku_df = pd.read_csv(f"{OUTPUT_DIR}/resultater_sku_mape.csv").set_index("SKU")
    eda_df = pd.read_csv(f"{OUTPUT_DIR}/eda_resultater.csv", index_col=0)

    mean_vol    = eda_df["Mean_monthly_volume"]
    low_thresh  = mean_vol.quantile(1 / 3)
    high_thresh = mean_vol.quantile(2 / 3)

    def vol_label(v):
        if v >= high_thresh:
            return "Høyt"
        return "Middels" if v >= low_thresh else "Lavt"

    def ses_label(fs):
        return "Tydelig sesong" if (not pd.isna(fs) and fs >= FS_THRESHOLD) else "Svak/ingen sesong"

    ses_rows = ["Tydelig sesong", "Svak/ingen sesong"]
    vol_cols = ["Høyt", "Middels", "Lavt"]

    # Finn beste modell og MAPE for hvert kombinert segment
    table = {}
    for ses in ses_rows:
        for vol in vol_cols:
            seg_skus = [
                sku for sku in sku_df.index
                if sku in eda_df.index
                and ses_label(eda_df.loc[sku, "Fs_sesongstyrke"]) == ses
                and vol_label(eda_df.loc[sku, "Mean_monthly_volume"]) == vol
            ]
            best_model, best_mape = None, np.inf
            for model in MODEL_ORDER:
                if model not in sku_df.columns:
                    continue
                vals = sku_df.loc[
                    [s for s in seg_skus if s in sku_df.index], model
                ].dropna().values
                if len(vals) > 0:
                    med = float(np.median(vals))
                    if med < best_mape:
                        best_mape, best_model = med, model
            table[(ses, vol)] = {
                "model": best_model,
                "mape":  round(best_mape, 1) if best_model else np.nan,
                "n":     len(seg_skus),
            }

    # ── Tegn varmekart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    n_cols, n_rows = len(vol_cols), len(ses_rows)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    for ri, ses in enumerate(ses_rows):
        for ci, vol in enumerate(vol_cols):
            cell  = table[(ses, vol)]
            model = cell["model"]
            color = MODEL_COLORS.get(model, "#cccccc") if model else "#cccccc"

            rect = plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                  color=color, alpha=0.88, zorder=1)
            ax.add_patch(rect)

            if model:
                short = MODEL_SHORT.get(model, model)
                ax.text(ci, ri + 0.15, short,
                        ha="center", va="center", zorder=2,
                        fontsize=14, fontweight="bold", color="white")
                ax.text(ci, ri - 0.18,
                        f"MAPE: {cell['mape']:.1f}%  (n={cell['n']})",
                        ha="center", va="center", zorder=2,
                        fontsize=9.5, color="white")

    # Cellelinjer
    for c in np.arange(-0.5, n_cols, 1):
        ax.axvline(c, color="white", linewidth=2.5, zorder=3)
    for r in np.arange(-0.5, n_rows, 1):
        ax.axhline(r, color="white", linewidth=2.5, zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"{v} volum" for v in vol_cols], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ses_rows, fontsize=11)
    ax.set_xlabel("Volumnivå", fontsize=12)
    ax.set_ylabel("Sesongstyrke", fontsize=12)
    ax.set_title(
        "Figur 6: Beste modell per kombinert segment — median MAPE (Jan–Des 2025)",
        fontsize=12, fontweight="bold"
    )

    legend_patches = [
        mpatches.Patch(color=MODEL_COLORS[m],
                       label=m if m != "XGBoost" else "XGBoost (global)")
        for m in MODEL_ORDER
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              bbox_to_anchor=(1.27, 1.02), fontsize=9.5, title="Beste modell")
    ax.set_aspect("auto")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figur6_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Lagret: figur6_heatmap.png")


# ── Hovedprogram ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Genererer figur3–figur5 (krever CSV-filer fra analyse.py)...")
    fig3_segment_bars()
    fig4_boxplot()
    fig5_monthly_errors()
    print("\nFerdig! Produserte filer:")
    print("  figur3_segment_bars.png  — Gruppert søylediagram: MAPE per segment")
    print("  figur4_mape_boxplot.png  — Boksplott: MAPE-fordeling per modell")
    print("  figur5_maaned_mape.png   — Linjediagram: månedlig MAPE i testperioden")
