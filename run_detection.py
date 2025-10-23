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

def bands_anomalies(df, col='close', wind=ROLLING_WINDOW, sigma=SIGMA):
    s = df[col].astype(float)
    rolling_mean = s.rolling(window=wind, min_periods=max(1, wind//2)).mean()
    rolling_std = s.rolling(window=wind, min_periods=max(1, wind//2)).std().fillna(0)
    lower = rolling_mean - sigma * rolling_std
    upper = rolling_mean + sigma * rolling_std
    is_anom = (s <= lower) | (s >= upper)
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

def plot_and_save(df, ticker, date_for_file):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df['datetime'], df['close'], label='close', linewidth=1)
    _, rolling_mean, lower, upper = bands_anomalies(df)
    ax.fill_between(df['datetime'], lower, upper, color='gray', alpha=0.18)
    an_b = df[df['bandas_anom']]
    an_if = df[df['if_anom']]
    an_both = df[df['combined']]
    if not an_b.empty:
        ax.scatter(an_b['datetime'], an_b['close'], marker='o', color='orange', s=40)
    if not an_if.empty:
        ax.scatter(an_if['datetime'], an_if['close'], marker='x', color='red', s=45)
    if not an_both.empty:
        ax.scatter(an_both['datetime'], an_both['close'], marker='*', color='black', s=90)
    ax.set_title(f"{ticker} | rows={len(df)} | Bandas={len(an_b)} | IF={len(an_if)} | Combined={int(df['combined'].sum())}")
    ax.set_xlabel("datetime"); ax.set_ylabel("price"); ax.grid(alpha=0.3)
    png = os.path.join(OUT_DIR, f"{ticker}_plot_{date_for_file}.png")
    fig.savefig(png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return png

# ---------- main ----------
summaries = []
for ticker in TICKERS:
    print("Processing", ticker)
    df = fetch_intraday(ticker)
    if df.empty:
        print("  no data for", ticker)
        continue
    df['bandas_anom'], _, _, _ = bands_anomalies(df)
    df['if_anom'] = isolationforest_anomalies(df)
    df['combined'] = df['bandas_anom'] | df['if_anom']
    last_date = df['datetime'].iloc[-1].date().isoformat()
    csv_path = os.path.join(OUT_DIR, f"{ticker}_anomalies_{last_date}.csv")
    df.to_csv(csv_path, index=False)
    png_path = plot_and_save(df, ticker, last_date)
    print("  saved", csv_path, png_path)
    summaries.append({
        "date": last_date,
        "ticker": ticker,
        "rows": len(df),
        "anomalies_bandas": int(df['bandas_anom'].sum()),
        "anomalies_if": int(df['if_anom'].sum()),
        "anomalies_combined": int(df['combined'].sum())
    })

# update summary_daily.csv (replace same date+ticker)
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

print("Done. Outputs in", OUT_DIR)
