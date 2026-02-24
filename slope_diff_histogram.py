import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date,datetime
from pathlib import Path

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)


def histogram_to_pdf(counts, bin_width):
    counts = np.asarray(counts)
    pdf = counts / (counts.sum() * bin_width)
    return pdf

def compute_histogram(values, bin_centers, bin_width):
    edges = np.concatenate([
        bin_centers - bin_width / 2,
        [bin_centers[-1] + bin_width / 2]
    ])
    counts, _ = np.histogram(values, bins=edges)
    return counts

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
    plt.hist(values, bins=bins, edgecolor="black",color='C1')
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


def plot_pdf_comparison(bin_centers, pdf_sim, pdf_obs,save=False):
    plt.figure(figsize=(7, 4.5))
    plt.step(bin_centers, pdf_obs, where="mid", label="Observation",linewidth=2,color='C1')
    plt.step(bin_centers, pdf_sim, where="mid", label="Simulation",linewidth=2,color='gray')
    plt.xlabel("Δalpha",size=16)
    plt.ylabel("Probability density",size=16)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis="both", labelsize=14)

    fig_path = os.path.join(Path(outdir),'histogram_comparison_delta_alpha'+"_"+today+".png")
    if save:
        plt.savefig(fig_path, dpi=150)

    plt.show()


def main_plot_his_data(filepath, bin_size=0.5,save=False):
    df = read_data(filepath)
    df = filter_c18o(df)
    df = add_slope_kind(df)
    delta_alpha = compute_delta_alpha(df)
    plot_histogram(delta_alpha, bin_size,save=save)


def main_compare_pdf(filepath, bin_width = 0.5,save=False):

    df = read_data(filepath)
    df = filter_c18o(df)
    df = add_slope_kind(df)
    delta_alpha = compute_delta_alpha(df)

    sim_bin_centers = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25])
    sim_counts = np.array([1, 3, 13, 18, 4, 1])  # approximate from image


    sim_pdf = histogram_to_pdf(sim_counts, bin_width)

    obs_counts = compute_histogram(delta_alpha, sim_bin_centers, bin_width)
    obs_pdf = histogram_to_pdf(obs_counts, bin_width)

    plot_pdf_comparison(sim_bin_centers, sim_pdf, obs_pdf,save=save)

if __name__ == "__main__":
    outdir='Figures'

    # main_plot_his_data("cores_slopes.csv", bin_size=0.5,save=True)
    main_compare_pdf("cores_slopes.csv",bin_width = 0.5,save=True)