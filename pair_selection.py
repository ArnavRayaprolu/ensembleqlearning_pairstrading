import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import StandardScaler

# Load and Clean Data
df = pd.read_csv("nifty50_hourly_prices2.csv", index_col=0, parse_dates=True)
df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

tickers = df.columns.tolist()
n = len(tickers)

# Normalise Prices for SSD
scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=tickers)

# Initialize Matrices
pval_matrix = np.full((n, n), np.nan)
ssd_matrix = np.full((n, n), np.nan)
zscore_volatility_matrix = np.full((n, n), np.nan)

print("Running Analysis...")

# Compute the Tests
for i in range(n):
    for j in range(i + 1, n):
        t1, t2 = tickers[i], tickers[j]
        s1, s2 = (df[t1]), (df[t2])

        # Skip Constant Series
        if s1.std() == 0 or s2.std() == 0:
            continue

        # Cointegration
        try:
            _, pval, _ = coint(s1, s2)
        except Exception:
            pval = np.nan
        pval_matrix[i, j] = pval
        pval_matrix[j, i] = pval

        # Z Score Crossings
        spread = s1 - s2
        mean = spread.mean()
        std = spread.std()
        if std == 0:
            continue
        zscore = (spread - mean) / std
        zero_crossings = ((zscore.shift(1) * zscore) < 0).sum()
        zscore_volatility_matrix[i, j] = zero_crossings
        zscore_volatility_matrix[j, i] = zero_crossings

# Normalise and Rank
pval_rank = pd.DataFrame(pval_matrix, index=tickers, columns=tickers).rank(axis=1, method='average', ascending=True)
zscore_rank = pd.DataFrame(zscore_volatility_matrix, index=tickers, columns=tickers).rank(axis=1, method='average', ascending=False)

# Combine Scores
combined_df = 0.5 * pval_rank + 0.5 * zscore_rank

# Extract Top 10 Pairs
flattened = []
for i in range(n):
    for j in range(i + 1, n):
        score = combined_df.iloc[i, j]
        if not np.isnan(score):
            flattened.append((tickers[i], tickers[j], score))

top10 = sorted(flattened, key=lambda x: x[2])[:10]

print("\nTop 10 Pair Rankings")
for a, b, score in top10:
    print(f"{a} - {b} | Combined Score: {score:.2f}")

# Plot Price Series 
for idx, (a, b, score) in enumerate(top10[:5], 1):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[a], label=a)
    plt.plot(df.index, df[b], label=b)
    plt.title(f'Pair {idx}: {a} & {b} | Combined Score: {score:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()
