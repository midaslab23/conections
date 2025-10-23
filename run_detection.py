# run_detection.py
import os, io, sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import IsolationForest
# al inicio, junto a otros imports
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # opcional: silenciar la warning de yfinance


# ---------- CONFIG ----------
TICKERS = ["^MXX", "^SPX", "^VIX","^GSPC"]          # edit as needed
PERIOD = "3Y"
INTERVAL = "1d"
ROLLING_WINDOW = 20
SIGMA = 2.0
CONTAMINATION = 0.05
OUT_DIR = "out"                     # local output folder
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
def fetch_intraday(ticker, period=PERIOD, interval=INTERVAL):
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col]).strip() for col in df.columns.values]
    df = df.reset_index()
    df.columns = [str(c).lower().replace(' ', '_') for c in df.columns]
    if 'datetime' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    if 'close' not in df.columns:
        candidates = [c for c in df.columns if 'close' in c]
        if candidates:
            df = df.rename(columns={candidates[0]: 'close'})
    return df

# ---------- funciones corregidas para AND/OR y guardado de bandas ----------
def bands_anomalies(df, col='close', wind=ROLLING_WINDOW, sigma=SIGMA):
    s = df[col].astype(float)
    rolling_mean = s.rolling(window=wind, min_periods=max(1, wind//2)).mean()
    rolling_std = s.rolling(window=wind, min_periods=max(1, wind//2)).std().fillna(0)
    lower = rolling_mean - sigma * rolling_std
    upper = rolling_mean + sigma * rolling_std
    is_anom = (s <= lower) | (s >= upper)
    # devolver también las series de bandas para guardarlas en CSV
    return is_anom.fillna(False), rolling_mean, lower, upper

def isolationforest_anomalies(df, col='close', contamination=CONTAMINATION):
    data = df.copy()
    if col not in data.columns:
        return pd.Series([False]*len(df), index=df.index)
    data['ret'] = data[col].pct_change().fillna(0).astype(float)
    X = data[['ret']].values
    if len(X) < 5:
        return pd.Series([False]*len(df), index=df.index)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
    model.fit(X)
    preds = model.predict(X)
    return pd.Series(preds == -1, index=df.index)

def plot_and_save(df, ticker, date_for_file, save_csv=True):
    """
    df: DataFrame con al menos 'datetime' y 'close'
    Esta función asumirá que df ya tiene las columnas de flag o las calcula aquí.
    """
    # recalcular bandas y IF para obtener las series de bandas para CSV
    bandas_flag, rolling_mean, lower, upper = bands_anomalies(df)
    if_flag = isolationforest_anomalies(df)
    # flags separados
    both_flag = bandas_flag & if_flag
    bandas_only = bandas_flag & (~if_flag)
    if_only = if_flag & (~bandas_flag)

    # añadir columnas para guardar
    df = df.copy()
    df['rolling_mean'] = rolling_mean
    df['lower_band'] = lower
    df['upper_band'] = upper
    df['bandas_anom'] = bandas_flag
    df['if_anom'] = if_flag
    df['both_anom'] = both_flag
    df['bandas_only'] = bandas_only
    df['if_only'] = if_only
    df['combined_or'] = bandas_flag | if_flag
    df['combined_and'] = both_flag

    # conteos para titulo/resumen
    n_bandas = int(bandas_flag.sum())
    n_if = int(if_flag.sum())
    n_both = int(both_flag.sum())
    n_or = int((bandas_flag | if_flag).sum())

    # plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df['datetime'], df['close'], label='close', linewidth=1)
    # bandas sombreadas (usar rolling_mean, lower, upper)
    ax.fill_between(df['datetime'], df['lower_band'], df['upper_band'], color='gray', alpha=0.18,
                    label=f'rolling ±{SIGMA}σ (w={ROLLING_WINDOW})')

    # plot: bandas-only (naranja), IF-only (rojo), both (negro estrella)
    if not df[df['bandas_only']].empty:
        an_b = df[df['bandas_only']]
        ax.scatter(an_b['datetime'], an_b['close'], marker='o', color='orange', s=40,
                   label=f'Bandas only (n={len(an_b)})')
    if not df[df['if_only']].empty:
        an_if = df[df['if_only']]
        ax.scatter(an_if['datetime'], an_if['close'], marker='x', color='red', s=45,
                   label=f'IF only (n={len(an_if)})')
    if not df[df['both_anom']].empty:
        an_both = df[df['both_anom']]
        ax.scatter(an_both['datetime'], an_both['close'], marker='*', color='black', s=90,
                   label=f'Both (n={len(an_both)})')

    ax.set_title(f"{ticker} | rows={len(df)} | Bandas={n_bandas} | IF={n_if} | Both={n_both} | OR={n_or}")
    ax.set_xlabel("datetime"); ax.set_ylabel("price")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.xticks(rotation=30)

    # guardar CSV (con las columnas de bandas) y PNG
    if save_csv:
        csv_path = os.path.join(OUT_DIR, f"{ticker}_anomalies_{date_for_file}.csv")
        # orden de columnas recomendado
        cols_order = ['datetime','close','rolling_mean','lower_band','upper_band',
                      'bandas_anom','if_anom','both_anom','bandas_only','if_only','combined_or','combined_and']
        # algunas columnas pueden no existir si df era corto; filtrar
        cols_to_save = [c for c in cols_order if c in df.columns]
        df.to_csv(csv_path, index=False)
    else:
        csv_path = None

    png_path = os.path.join(OUT_DIR, f"{ticker}_plot_{date_for_file}.png")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # return paths y conteos
    return {"csv": csv_path, "png": png_path, "rows": len(df),
            "bandas": n_bandas, "if": n_if, "both": n_both, "or": n_or}

# ---------- main (fragmento) ----------
summaries = []   # <- asegurarse de inicializar antes del for
for ticker in TICKERS:
    ...

# ---------- main (fragmento) ----------
for ticker in TICKERS:
    try:
        print("Processing", ticker)
        df = fetch_intraday(ticker)
        if df.empty:
            print("  no data for", ticker)
            continue

        last_dt = df['datetime'].iloc[-1]
        date_for_file = last_dt.date().isoformat()

        out = plot_and_save(df, ticker, date_for_file, save_csv=True)
        print("  saved", out["csv"], out["png"])
        summaries.append({
            "date": date_for_file,
            "ticker": ticker,
            "rows": out["rows"],
            "anomalies_bandas": out["bandas"],
            "anomalies_if": out["if"],
            "anomalies_both": out["both"],
            "anomalies_or": out["or"]
        })
    except Exception as e:
        print(f"  ERROR processing {ticker}: {e}")
        # opcional: guardar trace para debugging
        import traceback
        traceback.print_exc()
        continue
# ---- crear archivo combinado (timeseries) ----
import glob

# encuentra todos los CSV por ticker (puedes ajustar el pattern si cambias nombres)
csv_files = glob.glob(os.path.join(OUT_DIR, "*_anomalies_*.csv"))
dfs = []
for f in csv_files:
    try:
        tmp = pd.read_csv(f, parse_dates=['datetime'])
        # intentar inferir ticker del nombre de archivo si no tiene columna
        if 'ticker' not in tmp.columns:
            # filename like out/TICKER_anomalies_YYYY-MM-DD.csv
            fname = os.path.basename(f)
            ticker_name = fname.split("_anomalies_")[0]
            tmp['ticker'] = ticker_name
        dfs.append(tmp)
    except Exception as e:
        print("Error leyendo", f, e)
if dfs:
    all_df = pd.concat(dfs, ignore_index=True)
    # opcional: ordenar por ticker+datetime
    if 'datetime' in all_df.columns:
        all_df = all_df.sort_values(['ticker','datetime']).reset_index(drop=True)
    combined_path = os.path.join(OUT_DIR, "all_tickers_anomalies.csv")
    all_df.to_csv(combined_path, index=False)
    print("Saved combined timeseries:", combined_path)
else:
    print("No csv files found to combine.")

