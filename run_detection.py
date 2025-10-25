#!/usr/bin/env python3
# run_detection.py
# Requiere: yfinance, pandas, numpy, matplotlib, scikit-learn

import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# ---------------- CONFIG ----------------
TICKERS = ["^MXX", "^SPX", "^VIX", "^GSPC","^BVSP","^N225","BYMA.BA"]   # edita si quieres
PERIOD = "3y"
INTERVAL = "1d"
ROLLING_WINDOW = 20        # para rolling mean/std (bandas)
SIGMA = 2.0
CONTAMINATION = 0.05       # para IsolationForest
ROLLING_STD_WINDOW = 10    # ventana para calcular la feature roll_std usada por IF
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- helpers ----------------
def normalize_df(df):
    """Normaliza columnas, resetea index datetime a columna 'datetime' y lower-case keys."""
    if df is None or df.empty:
        return pd.DataFrame()
    # si hay MultiIndex (por ejemplo cuando se piden varios tickers) convertir a strings simples
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col]).strip() for col in df.columns.values]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    # asegurar datetime
    if 'datetime' not in df.columns:
        # buscar la primera columna datetime
        for c in df.columns[:4]:
            if np.issubdtype(df[c].dtype, np.datetime64):
                df = df.rename(columns={c: 'datetime'})
                break
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)
    # asegurar close
    if 'close' not in df.columns:
        candidates = [c for c in df.columns if 'close' in c]
        if candidates:
            df = df.rename(columns={candidates[0]: 'close'})
    # si sigue sin close, usar la primera numérica después de datetime
    if 'close' not in df.columns:
        numeric_cols = [c for c in df.columns if c != 'datetime' and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: 'close'})
    # dropna close
    if 'close' in df.columns:
        df = df.dropna(subset=['close']).reset_index(drop=True)
    return df

def fetch_data(ticker, period=PERIOD, interval=INTERVAL):
    """Descarga y devuelve DataFrame normalizado (datetime + close al menos)."""
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval,
                         progress=False, threads=False, auto_adjust=True)
    except Exception as e:
        print(f"  ERROR yf.download {ticker}: {e}")
        return pd.DataFrame()
    df = normalize_df(df)
    return df

def compute_bands(df, col='close', wind=ROLLING_WINDOW, sigma=SIGMA):
    s = df[col].astype(float)
    rolling_mean = s.rolling(window=wind, min_periods=max(1, wind//2)).mean()
    rolling_std = s.rolling(window=wind, min_periods=max(1, wind//2)).std().fillna(0)
    lower = rolling_mean - sigma * rolling_std
    upper = rolling_mean + sigma * rolling_std
    is_anom = (s <= lower) | (s >= upper)
    return is_anom.fillna(False), rolling_mean, lower, upper

def compute_if_flags(df, col='close', contamination=CONTAMINATION, roll_std_window=ROLLING_STD_WINDOW):
    """
    Devuelve siempre 4 cosas:
      - df_with_features (DataFrame con 'ret' y 'roll_std')
      - if_flag  (pd.Series bool)
      - scores   (pd.Series floats)
      - Xs_scaled (np.array) o None si no hay suficientes datos
    """
    data = df.copy()
    # crear features en el dataframe devuelto
    data['ret'] = data[col].pct_change().fillna(0).astype(float)
    data['roll_std'] = data['ret'].rolling(window=roll_std_window, min_periods=1).std().fillna(0).astype(float)

    X = data[['ret','roll_std']].values
    # si hay muy pocos puntos devolvemos estructuras vacías/seguras
    if len(X) < 5:
        if_flag = pd.Series([False]*len(data), index=data.index)
        scores = pd.Series([0.0]*len(data), index=data.index)
        return data, if_flag, scores, None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
    model.fit(Xs)
    preds = model.predict(Xs)              # -1 outlier, 1 inlier
    scores = model.decision_function(Xs)  # higher = more normal

    if_flag = pd.Series(preds == -1, index=data.index)

    return data, if_flag, pd.Series(scores, index=data.index), Xs

def save_if_plot(df, xs_scaled, scores, if_flags, ticker, date_for_file):
    """Crea y guarda el gráfico 2-panel (decision surface + histogram) por ticker."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        ax = axes[0]
        if xs_scaled is not None:
            pad = 0.5
            xx, yy = np.meshgrid(
                np.linspace(xs_scaled[:, 0].min() - pad, xs_scaled[:, 0].max() + pad, 300),
                np.linspace(xs_scaled[:, 1].min() - pad, xs_scaled[:, 1].max() + pad, 300),
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            model = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=0)
            model.fit(xs_scaled)
            Z = model.decision_function(grid).reshape(xx.shape)
            c = ax.contourf(xx, yy, Z, levels=60, cmap="Spectral_r", alpha=0.95)
            cb = fig.colorbar(c, ax=ax)
            cb.set_label("decision_function (más alto = más normal)", fontsize=10)
            # scaled dataframe for plotting
            xs_df = pd.DataFrame(xs_scaled, columns=["ret_s","roll_std_s"], index=df.index)
            ax.scatter(xs_df.loc[~if_flags, 'ret_s'], xs_df.loc[~if_flags, 'roll_std_s'],
                       c="#1f77b4", s=28, alpha=0.6, label=f"Inliers n={(~if_flags).sum()}",
                       edgecolors="k", linewidth=0.35, zorder=4)
            ax.scatter(xs_df.loc[if_flags, 'ret_s'], xs_df.loc[if_flags, 'roll_std_s'],
                       c="magenta", s=88, marker="X", label=f"Outliers n={if_flags.sum()}",
                       edgecolors="white", linewidth=0.9, zorder=6)
        else:
            ax.text(0.5,0.5,"Pocos datos para decision surface", ha='center', va='center')
        ax.set_xlabel("ret (estandarizado)")
        ax.set_ylabel("rolling std (estandarizado)")
        ax.set_title(f"IsolationForest decision surface {ticker}")
        ax.legend(fontsize="small", loc="upper right")

        # Histograma de scores
        axh = axes[1]
        axh.hist(scores, bins=40, color="lightgray", edgecolor="k", alpha=0.9)
        axh.set_title("Histograma de decision_function")
        axh.set_xlabel("score (decision_function)")
        axh.set_ylabel("count")
        if if_flags.any():
            outlier_median = scores[if_flags].median()
            axh.axvline(outlier_median, color="magenta", linestyle="--", linewidth=1.6, label="mediana outliers")
            axh.legend(fontsize="small")

        png_path = os.path.join(OUT_DIR, f"{ticker}_if_plot_{date_for_file}.png")
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return png_path
    except Exception as e:
        print(f"  ERROR saving IF plot for {ticker}: {e}")
        try:
            plt.close('all')
        except:
            pass
        return None

# ---------------- main ----------------
def main():
    summaries = []
    rows_all = []

    for ticker in TICKERS:
        print(f"\nProcessing {ticker} ...")
        try:
            df = fetch_data(ticker)
            if df.empty:
                print(f"  no data for {ticker} -> skipping")
                continue

            # compute bands
            bandas_flag, rolling_mean, lower, upper = compute_bands(df, col='close', wind=ROLLING_WINDOW, sigma=SIGMA)

            # compute IF flags and scores and scaled features - devolvemos siempre 4 items
            res = compute_if_flags(df, col='close', contamination=CONTAMINATION, roll_std_window=ROLLING_STD_WINDOW)

            # DEBUG: imprimir info sobre res para ver qué devolvió la función
            print("  DEBUG compute_if_flags returned type:", type(res))
            if isinstance(res, tuple):
                print("  DEBUG compute_if_flags tuple length:", len(res))
            else:
                print("  DEBUG compute_if_flags not tuple")

            # unpack robusto; soporta versiones antiguas por compatibilidad
            if isinstance(res, tuple):
                if len(res) == 4:
                    df_with_features, if_flag, scores, xs_scaled = res
                elif len(res) == 3:
                    df_with_features, if_flag, scores = res
                    xs_scaled = None
                else:
                    # fallback seguro
                    df_with_features = res[0] if len(res) > 0 else df.copy()
                    if_flag = pd.Series([False]*len(df_with_features), index=df_with_features.index)
                    scores = pd.Series([0.0]*len(df_with_features), index=df_with_features.index)
                    xs_scaled = None
            else:
                df_with_features = df.copy()
                if_flag = pd.Series([False]*len(df_with_features), index=df_with_features.index)
                scores = pd.Series([0.0]*len(df_with_features), index=df_with_features.index)
                xs_scaled = None

            # ahora sí definimos date_for_file (usar last timestamp)
            last_dt = df['datetime'].iloc[-1]
            date_for_file = last_dt.date().isoformat()

            # generate IF plot (2-panel) and save -- usamos df_with_features y xs_scaled
            png = save_if_plot(df_with_features, xs_scaled, scores, if_flag, ticker, date_for_file)
            print(f"  saved IF-plot: {png}")

            # derive combined flags (usamos if_flag y bandas_flag alineados por index)
            both_flag = bandas_flag & if_flag
            bandas_only = bandas_flag & (~if_flag)
            if_only = if_flag & (~bandas_flag)
            combined_or = bandas_flag | if_flag
            combined_and = both_flag

            # attach to df minimal columns (usamos df to preserve datetime/close alignment)
            df_out = pd.DataFrame({
                'datetime': df['datetime'],
                'close': df['close'].astype(float),
                'rolling_mean': rolling_mean,
                'lower_band': lower,
                'upper_band': upper,
                'bandas_anom': bandas_flag.astype(int),
                'if_anom': if_flag.astype(int),
                'both_anom': both_flag.astype(int),
                'bandas_only': bandas_only.astype(int),
                'if_only': if_only.astype(int),
                'combined_or': combined_or.astype(int),
                'combined_and': combined_and.astype(int)
            })
            df_out['ticker'] = ticker

            # append to accumulator
            rows_all.append(df_out)

            summaries.append({
                "date": date_for_file,
                "ticker": ticker,
                "rows": len(df_out),
                "anomalies_bandas": int(bandas_flag.sum()),
                "anomalies_if": int(if_flag.sum()),
                "anomalies_both": int(both_flag.sum()),
                "anomalies_or": int(combined_or.sum())
            })

        except Exception as e:
            print(f"  ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # combine and save single CSV
    if rows_all:
        all_df = pd.concat(rows_all, ignore_index=True)
        cols = ['datetime','ticker','close','rolling_mean','lower_band','upper_band',
                'bandas_anom','if_anom','both_anom','bandas_only','if_only','combined_or','combined_and']
        cols = [c for c in cols if c in all_df.columns]
        combined_path = os.path.join(OUT_DIR, "all_tickers_anomalies.csv")
        all_df[cols].to_csv(combined_path, index=False)
        print("\nSaved combined CSV:", combined_path)
    else:
        print("\nNo data collected for any ticker.")

    # also save summary_daily.csv (one row per ticker/day) replace mode
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_csv = os.path.join(OUT_DIR, "summary_daily.csv")
        if os.path.exists(summary_csv):
            prev = pd.read_csv(summary_csv)
            for r in summaries:
                prev = prev[~((prev['date']==r['date']) & (prev['ticker']==r['ticker']))]
            merged = pd.concat([prev, summary_df], ignore_index=True)
            merged.to_csv(summary_csv, index=False)
        else:
            summary_df.to_csv(summary_csv, index=False)
        print("Saved summary_daily.csv")
    print("\nDone.")

if __name__ == "__main__":
    main()



#### imgágenes repo
# Generar mapping de ticker -> image_url para Power BI (ejemplo GitHub raw)
repo_user = "midaslab23"
repo_name = "conections"
branch = "main"             # o la rama donde estés guardando out/
out_folder = "out"

mappings = []
for ticker in TICKERS:
    safe = ticker.replace("^","").replace("/","_")  # sanitizar para filename
    # filename exacto debe coincidir con lo que guardas: e.g. MXX_if_plot_2025-10-23.png
    # si usas fecha dinámica, puedes elegir la última fecha o usar patrón sin fecha
    filename = f"{safe}_if_plot_{datetime.utcnow().date().isoformat()}.png"
    url = f"https://raw.githubusercontent.com/{repo_user}/{repo_name}/refs/heads/{branch}/{out_folder}/%5E{filename}"
    mappings.append({"ticker": ticker, "image_url": url})



pd.DataFrame(mappings).to_csv(os.path.join(OUT_DIR, "ticker_images.csv"), index=False)
print("Saved ticker_images.csv")
