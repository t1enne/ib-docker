import matplotlib.pyplot as plt
import pandas as pd
from src.utils import get_ols_fit_model, read_candles


def pnd(symbols: list[str]):
    df1 = read_candles(symbols[0].upper())
    df2 = read_candles(symbols[1].upper())
    s1 = df1["Close"]
    s2 = df2["Close"]
    model = get_ols_fit_model(s1, s2)
    alpha, beta = model.params
    scaled_s2 = alpha + beta * s2
    spread_series = s1 - scaled_s2
    z_score = (spread_series - spread_series.mean()) / spread_series.std()
    df = pd.DataFrame(
        {
            symbols[0]: s1,
            f"{symbols[1]}_scaled": scaled_s2,
            "spread": spread_series,
            "z_score": z_score,
        }
    )

    # --- 4) Plot normalized series ---
    fig, axes = plt.subplots(figsize=(12, 5), nrows=2)
    axes[0].plot(s1, label=symbols[0])
    axes[0].plot(df[f"{symbols[1]}_scaled"], label=f"{symbols[1]} scaled")
    axes[0].legend()
    axes[0].set_title("Cointegration-Based Normalization")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Close")
    axes[0].grid()

    # --- 5) Plot normalized spread ---
    axes[1].plot(df["z_score"], label="Normalized Spread")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].legend()
    axes[1].set_title("Spread (Mean-Reverting Series)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Z")
    axes[1].grid()

    fig.tight_layout()
    plt.show()
