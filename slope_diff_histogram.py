import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date,datetime
from pathlib import Path

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

def read_data(filepath):
    return pd.read_csv(filepath)


def filter_c18o(df):
    return df[df["molecule"].astype(str).str.strip() == "C18O"].copy()


def add_slope_kind(df):
    """
    Normalize slope labels to 'slope1' and 'slope2'
    """
    labels = df["slope_label"].astype(str).str.lower().str.strip()

    df["_slope_kind"] = np.nan
    df.loc[labels.str.contains(r"slope\s*1"), "_slope_kind"] = "slope1"
    df.loc[labels.str.contains(r"slope\s*2"), "_slope_kind"] = "slope2"

    return df


def compute_delta_alpha(df):
    """
    For each core, compute alpha(slope2) - alpha(slope1)
    """
    df = df.dropna(subset=["core", "alpha", "_slope_kind"])

    wide = df.pivot_table(
        index="core",
        columns="_slope_kind",
        values="alpha",
        aggfunc="mean"
    )

    wide = wide.dropna(subset=["slope1", "slope2"])
    wide["delta_alpha"] = wide["slope2"] - wide["slope1"]

    return wide["delta_alpha"].values


def plot_histogram(values, bin_size=0.5,save=False):
    if len(values) == 0:
        raise ValueError("No valid C18O cores with both slope1 and slope2")

    vmin, vmax = values.min(), values.max()
    bins = np.arange(
        bin_size * np.floor(vmin / bin_size),
        bin_size * np.ceil(vmax / bin_size) + bin_size,
        bin_size
    )

    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, edgecolor="black")
    plt.xlabel("Δalpha = alpha_b − alpha_r",size=16)
    plt.ylabel("Number of cores per bin",size=16)
    plt.title("C18O Δalpha histogram")
    plt.tight_layout()
    plt.tick_params(axis="both", labelsize=14)
    plt.xlim(-2.1,2.1)

    fig_path = os.path.join(Path(outdir),'histogram_delta_alpha'+"_"+today+".png")
    if save:
        plt.savefig(fig_path, dpi=150)
    plt.show()


def main(filepath, bin_size=0.5,save=False):
    df = read_data(filepath)
    df = filter_c18o(df)
    df = add_slope_kind(df)
    delta_alpha = compute_delta_alpha(df)
    plot_histogram(delta_alpha, bin_size,save=save)


if __name__ == "__main__":
    outdir='Figures'
    main("cores_slopes.csv", bin_size=0.5,save=True)
