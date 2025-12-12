import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def parse_alpha_line(line):
    parts = line.split()

    name = parts[0]
    molecule = parts[1]
    density = float(parts[2].rstrip(',')) * 1e4

    alpha = float(parts[3])
    alpha_err = float(parts[4])

    color_label = parts[-1]

    return name, molecule, density, alpha, alpha_err, color_label

def parse_line(line):
    parts = line.split()

    name = parts[0]
    molecule = parts[1]

    density = float(parts[2].rstrip(',')) #* 1e4

    alpha = float(parts[3])
    alpha_err = float(parts[4])
    A = float(parts[5])

    color = parts[-1]

    data_vals = list(map(float, parts[6:-1]))
    N = len(data_vals)
    if N % 3 != 0:
        raise ValueError("Number of data values is not divisible by 3")

    npts = N // 3
    positions = np.array(data_vals[:npts])
    velocities = np.array(data_vals[npts:2*npts])
    v_unc = np.array(data_vals[2*npts:])

    return {
        "name": name,
        "molecule": molecule,
        "density": density,
        "alpha": alpha,
        "alpha_err": alpha_err,
        "A": A,
        "positions": positions,
        "velocities": velocities,
        "v_unc": v_unc,
        "color": color,
    }


def plot_file_all_together(filename):
    entries = []

    # --- Read & parse all lines first ---
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(parse_line(line))

    if not entries:
        raise RuntimeError("No valid data found")

    # --- Set up colormap ---
    densities = np.array([e["density"] for e in entries])

    norm = mcolors.Normalize(vmin=densities.min(), vmax=densities.max())

    # Define the start and end colors using names or hex codes
    # 'gold' is a named color in Matplotlib
    # 'green' is also a named color
    # colors = ["darkgoldenrod", "darkgreen"]
    # colors = ["maroon", "lightsalmon"]
    colors = ["dodgerblue", "purple"]

    # Create the custom colormap
    # The 'from_list' method takes a name for the new colormap and the list of colors
    cmap_gold_to_green = LinearSegmentedColormap.from_list("gold_to_green", colors)


    # cmap = cm.BuPu
    cmap = cmap_gold_to_green

    fig, ax = plt.subplots()

    # --- Plot everything ---
    for e in entries:
        color = cmap(norm(e["density"]))

        pos = e["positions"]
        vel = e["velocities"]
        v_unc = e["v_unc"]

        # Data with error bars
        ax.errorbar(
            pos, vel, yerr=v_unc,
            fmt='o', color=color, alpha=0.85
        )

        # Power-law fit
        x_fit = np.linspace(pos.min(), pos.max(), 300)
        y_fit = e["A"] * x_fit ** e["alpha"]
        ax.plot(x_fit, y_fit, color=color, linestyle='--', alpha=0.85)

    # --- Labels & colorbar ---
    ax.set_xlabel("Radial offset (au)")
    ax.set_ylabel(r"$\delta V$ [km/s]")
    ax.set_title("Velocity vs Position (colored by density)")
    ax.grid(True)

    ax.set_xlim(1500,8000)
    ax.set_ylim(0,0.5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"volume dnsity $\times 10^4$ ")

    # Log scales
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    plt.savefig(
        "Figures/coldens_and_slope/cold_dense_low-dens.png",
        bbox_inches='tight', dpi=300)
    plt.show()


def plot_alpha_vs_density(filename):

    # Marker by molecule
    marker_map = {
        "C18O": "o",
        "SO": "^",
        "N2D+": "x",
    }

    # Color by source label (FIXED mapping)
    color_map = {
        "blue": "C0",
        "red": "C1",
    }

    fig, ax = plt.subplots()

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            name, molecule, density, alpha, alpha_err, color_label = parse_alpha_line(line)

            marker = marker_map.get(molecule, "o")
            color = color_map.get(color_label, "k")

            ax.errorbar(
                density,
                alpha,
                yerr=alpha_err,
                fmt=marker,
                color=color,
                markersize=9,
                capsize=4,
                elinewidth=1.2,
                markeredgecolor="k",
                markeredgewidth=0.8,
                linestyle="none",
            )

    ax.set_xlabel(r"volume density $(cm^{-3})$")
    ax.set_ylabel(r"velocity exponent $\alpha$")
    ax.set_title(r"$\alpha$ vs Density")

    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # --- Legends ---
    molecule_legend = [
        Line2D([0], [0], marker="o", color="k", linestyle="", label="C18O"),
        Line2D([0], [0], marker="^", color="k", linestyle="", label="SO"),
        Line2D([0], [0], marker="x", color="k", linestyle="", label="N2D+"),
    ]

    color_legend = [
        Line2D([0], [0], marker="o", color="C0", linestyle="", label="blue"),
        Line2D([0], [0], marker="o", color="C1", linestyle="", label="red"),
    ]

    leg1 = ax.legend(handles=molecule_legend, title="Molecule", loc="upper left")
    ax.add_artist(leg1)
    # ax.legend(handles=color_legend, title="Source", loc="upper right")

    plt.savefig(
        "Figures/coldens_and_slope/cold_dense_vs_alpha_dec10.png",
        bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    filename = 'cores_parameters.txt'
    plot_alpha_vs_density(filename)
    # plot_file_all_together(filename)
