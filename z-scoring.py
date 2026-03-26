import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load preprocessed close prices
df = pd.read_csv("nifty50_hourly_prices.csv", index_col=0, parse_dates=True)

selected_pairs = [
    ("BPCL", "CIPLA"),
    ("CIPLA", "NTPC"),
    ("ADANIENT", "TATACONSUM"),
    ("EICHERMOT", "M&M"),
    ("HDFCBANK", "ICICIBANK")
]


zscore_df = pd.DataFrame(index=df.index)

plt.figure(figsize=(15, 22))
for i, (a, b) in enumerate(selected_pairs, 1):
    y = df[a].values
    x = df[b].values

    spread = y - x

    mean = np.mean(spread)
    std = np.std(spread)
    z = (spread - mean) / std

    pair_name = f"{a}_{b}"
    zscore_df[pair_name] = z

    plt.subplot(9, 1, i)
    plt.plot(df.index, z, label=pair_name, color="blue")
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1, color='green', linestyle='--', label='+1 SD')
    plt.axhline(-1, color='green', linestyle='--', label='-1 SD')
    plt.axhline(2, color='red', linestyle='--', label='+2 SD')
    plt.axhline(-2, color='red', linestyle='--', label='-2 SD')
    plt.title(f"Z-score of Cointegrated Spread: {pair_name}")
    plt.legend(loc='upper right')
    plt.grid(True)

plt.tight_layout()
plt.show()

zscore_df.to_csv("zscore_spreads.csv")