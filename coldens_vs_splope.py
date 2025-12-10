import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----- Load data -----
df = pd.read_csv("cores_slopes.csv", comment="#")

fig, ax = plt.subplots(figsize=(6, 4.5))

# Color by slope_label (only two colors)
color_map = {
    "Slope1": "tab:orange",
    "Slope2": "tab:blue",
}

# Marker by molecule
marker_map = {
    "C18O": "o",
    "SO": "s",
}

marker_map = {
    "C18O": "o",
    "SO": "s",
    "N2D+": "P",
}

marker_size = {
    "C18O": 12,
    "SO": 16,
}

default_marker = "o"

# ----- Plot points -----
for _, row in df.iterrows():
    color = color_map.get(row["slope_label"], "k")
    marker = marker_map.get(row["molecule"], default_marker)
    size = marker_size.get(row["molecule"], 12)

    ax.errorbar(
        row["N_H2_deconv"]*1e4,
        row["alpha"],
        yerr=row["alpha_err"],
        fmt=marker,
        color=color,
        ecolor=color,
        markersize= size,
        markeredgecolor="k",
        capsize=3,
        linestyle="none",
    )

ax.set_xscale("log")
ax.set_xlabel(r"$N_{\mathrm{H}_2}$ (cm$^{-2}$)", fontsize=12)
ax.set_ylabel(r"Velocity exponent $\alpha$", fontsize=12)
ax.set_title(r"$N_{\mathrm{H}_2}$ vs velocity exponent $\alpha$")

ax.grid(True, which="both", linestyle="--", alpha=0.4)

# ----- Legends -----
# Legend for slopes (colors)
slope_handles = [
    Line2D([0], [0], marker="o", linestyle="none",
           color=color_map["Slope1"], label="Slope1"),
    Line2D([0], [0], marker="o", linestyle="none",
           color=color_map["Slope2"], label="Slope2"),
]

# Legend for molecules (markers) – build only for molecules present in the data
unique_molecules = df["molecule"].unique()
mol_handles = [
    Line2D([0], [0], marker=marker_map.get(m, default_marker),
           linestyle="none", color="black", label=m)
    for m in unique_molecules
]

# legend1 = ax.legend(handles=slope_handles, title="Slope", loc="upper left")
legend2 = ax.legend(handles=mol_handles, title="Molecule", loc="best")
# ax.add_artist(legend1)  # so the first legend doesn't get overwritten

plt.tight_layout()
plt.savefig('slope_vs_coldens_single_lines_and_doubles_deconv.png',bbox_inches='tight',  dpi=300)
plt.show()
