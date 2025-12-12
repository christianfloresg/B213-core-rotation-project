import os
import numpy as np
import matplotlib.pyplot as plt
from make_TP_maps import find_the_spectrum_for_a_source, ALMATPData
from spectral_cube import SpectralCube
from astropy import units as u
from astropy.constants import c

from pvextractor import Path, extract_pv_slice
from astropy.coordinates import SkyCoord

from astropy.modeling import models, fitting
from astropy.wcs import WCS
from collections.abc import Iterable


"""
Get into the correct environment
conda activate astropy
"""
# ---------- small utility ----------

def open_fits_file(folder,fits_name):

    # Step 1: Load the datacube and WCS
    cube = SpectralCube.read(os.path.join(folder,fits_name))  # Load the datacube
    header = cube.header
    wcs = cube.wcs  # Extract WCS information

    return cube, wcs

# ---------- small utility ----------
def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx

def accurate_reference_frequency(filename):
    '''
    We take the rest frequency from a file not from the header
    '''

    try:
        lower_idx_spw_name = filename.find('spw')
        spw_name = filename[lower_idx_spw_name:lower_idx_spw_name + 5]
        print(spw_name)
    except:
        print('There is no "spw" substring in this data file, I cannot guess the molecule')


    with open('molecule_rest_freq.txt') as f:
        file = f.readlines()
        for lines in file:
            if spw_name in lines:
                freq_rest = lines.split()[1]
                molec_name = lines.split()[2]
    return float(freq_rest)

def update_WCS_coordinates(initial_ww, filename):
    rest_frequency = accurate_reference_frequency(filename)

    ww = initial_ww
    spectral_axis_index = ww.wcs.spec

    # Extract current WCS parameters for the spectral axis
    crval_freq = ww.wcs.crval[spectral_axis_index]  # Reference frequency (CRVAL)
    cdelt_freq = ww.wcs.cdelt[spectral_axis_index]  # Increment per pixel in frequency
    crpix_freq = ww.wcs.crpix[spectral_axis_index]  # Reference pixel

    print(c.to('km/s').value, 'THIS TRANS')
    # Convert reference frequency to velocity using the Doppler formula
    crval_vel = c.to('km/s').value * (1 - crval_freq / rest_frequency)  # Reference velocity
    cdelt_vel = c.to('km/s').value * (
            -cdelt_freq / rest_frequency)  # Velocity increment (negative because velocity decreases with frequency)

    # Update the WCS parameters
    ww.wcs.ctype[spectral_axis_index] = 'VELO-LSR'  # Change CTYPE to velocity
    ww.wcs.crval[spectral_axis_index] = crval_vel  # Reference velocity
    ww.wcs.cdelt[spectral_axis_index] = cdelt_vel  # Increment in velocity
    ww.wcs.crpix[spectral_axis_index] = crpix_freq  # Keep reference pixel the same
    ww.wcs.cunit[spectral_axis_index] = 'km/s'  # Set the spectral axis units to velocity

    # Reinitialize the WCS to apply changes
    ww.wcs.set()

    return ww


def save_entry(filename, name, molecule, color, *containers):
    """
    Append one line to `filename`:

        name  values_from_all_containers...  color

    - containers may be lists, tuples, numpy arrays, or dicts
    - dicts contribute only their values
    - floats are written with 5 decimals
    - do not append if (name, color) already exist
    """

    # 1. Check for existing (name, color)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == name and parts[1] == molecule and parts[-1] == color:
                    return False

    values = []

    # 2. Flatten all containers
    for c in containers:
        # If dict → take values
        if isinstance(c, dict):
            iterable = c.values()
        else:
            iterable = c

        # Strings are iterable but should NOT be treated as containers here
        if isinstance(iterable, str):
            raise TypeError("String passed where numeric container expected")

        for x in iterable:
            print(x)
            # values.append(f"{float(x):.8f}")
            values.append(f"{float(x):.5g}")

    # 3. Write line
    line = " ".join([name] + [molecule] + values + [color]) + "\n"

    with open(filename, "a") as f:
        f.write(line)

    return True

def edge_pixel_from_angle(alpha, nx, ny):
    alpha = alpha % 360
    alpha_rad = np.deg2rad(alpha)

    x_c = nx / 2 - 0.5
    y_c = ny / 2 - 0.5

    if 0 <= alpha <= 45 or 315 < alpha < 360:
        yf = ny + 0.5
        xf = x_c - np.tan(alpha_rad) * ny / 2

        y2f = 0.5
        x2f = x_c + np.tan(alpha_rad) * ny / 2

    elif 45 < alpha <= 135:
        xf = 0.5
        yf = y_c + (nx / 2) / np.tan(alpha_rad)

        x2f = nx + 0.5
        y2f = y_c - (nx / 2) / np.tan(alpha_rad)

    elif 135 < alpha <= 225:
        yf = 0.5
        xf = x_c + np.tan(alpha_rad) * ny / 2

        y2f = ny + 0.5
        x2f = x_c - np.tan(alpha_rad) * ny / 2

    elif 225 < alpha <= 315:
        xf = nx + 0.5
        yf = y_c - (nx / 2) / np.tan(alpha_rad)

        x2f = 0.5
        y2f = y_c + (nx / 2) / np.tan(alpha_rad)

    else:
        raise ValueError("Angle out of bounds after normalization.")

    return [xf, yf], [x2f, y2f]


def define_pv_path_from_angle(cube, angle, position='edge'):
    '''
    define a pv path based on the center of the field, and an angle
    '''
    # Define the angle alpha (measured north from east)
    alpha = angle * u.deg  # Adjust angle as needed

    cube_wcs = WCS(cube.header)
    wcs = cube_wcs.wcs
    print(cube_wcs)
    # Get the spatial dimensions (x, y)
    ny, nx = cube.shape[1], cube.shape[2]  # Assuming cube.shape = [x, y, spectral, ]
    print(nx, ny, 'spatial coordinates')

    # Get the reference pixel for the spectral axis
    ref_pixel_spectral = cube_wcs.wcs.crpix[2] - 1  # Subtract 1 because FITS is 1-indexed
    print(ref_pixel_spectral, 'sectral index  coordinates')

    edge_pixel1, edge_pixel2 = edge_pixel_from_angle(angle, nx, ny)
    print(edge_pixel1, 'edge pixels 1')
    print(edge_pixel2, 'edge pixels 2')

    world_coords_edge1, freq_edge1 = cube_wcs.pixel_to_world(edge_pixel1[0], edge_pixel1[1], ref_pixel_spectral)
    world_coords_edge2, freq_edge2 = cube_wcs.pixel_to_world(edge_pixel2[0], edge_pixel2[1], ref_pixel_spectral)

    edge_ra1 = world_coords_edge1.ra.deg
    edge_dec1 = world_coords_edge1.dec.deg

    edge_ra2 = world_coords_edge2.ra.deg
    edge_dec2 = world_coords_edge2.dec.deg

    # Create the Path object
    ra_values = [edge_ra1, edge_ra2]
    dec_values = [edge_dec1, edge_dec2]

    print(position)

    if position == 'center_r':
        center_pixel = [nx / 2 - 0.5,
                        ny / 2 - 0.5]  # Center pixel coordinates | the center of the pixel is at 0.5 units
        world_coords_center, freq_center = cube_wcs.pixel_to_world(center_pixel[0], center_pixel[1], ref_pixel_spectral)

        center_ra = world_coords_center.ra.deg
        ceter_dec = world_coords_center.dec.deg

        ra_values = [center_ra, edge_ra1]
        dec_values = [ceter_dec, edge_dec1]

    elif position == 'center_b':

        center_pixel = [nx / 2 - 0.5,
                        ny / 2 - 0.5]  # Center pixel coordinates | the center of the pixel is at 0.5 units
        world_coords_center, freq_center = cube_wcs.pixel_to_world(center_pixel[0], center_pixel[1], ref_pixel_spectral)

        center_ra = world_coords_center.ra.deg
        ceter_dec = world_coords_center.dec.deg

        # Create the Path
        ra_values = [center_ra, edge_ra2]
        dec_values = [ceter_dec, edge_dec2]

    path = Path(SkyCoord(ra_values, dec_values, unit="deg", frame="fk5"), width=28 * u.arcsec)

    return path

def nyquist_sample_indices(radii_deg, beam_arcsec=28.0, nyquist_factor=0.5):
    """
    Choose spatial columns ~ every (nyquist_factor*beam_arcsec).
    radii_deg: 1D world coords (deg) along PV x-axis (monotonic).
    """
    radii_deg = np.asarray(radii_deg)
    d_deg = np.nanmedian(np.diff(radii_deg))
    if not np.isfinite(d_deg) or d_deg == 0:
        return np.arange(len(radii_deg), dtype=int)
    arcsec_per_pix = abs(d_deg) * 3600.0
    target_step_arcsec = beam_arcsec * nyquist_factor
    stride = int(max(1, round(target_step_arcsec / arcsec_per_pix)))
    return np.arange(0, len(radii_deg), stride, dtype=int)

def fit_gaussian_centroid(vel_kms, intensity, vmin_kms, vmax_kms):
    """
    Fit Gaussian1D + Const1D to spectrum within [vmin_kms, vmax_kms].
    Returns: mu (km/s), mu_err (km/s), model (callable or None), success (bool).
    If fitting fails, falls back to argmax (mu_err=np.nan, success=False).
    """
    vel_kms = np.asarray(vel_kms)
    intensity = np.asarray(intensity)

    win = (vel_kms >= vmin_kms) & (vel_kms <= vmax_kms)
    if np.count_nonzero(win) < 5:
        return np.nan, np.nan, None, False

    vv = vel_kms[win]
    yy = intensity[win]
    if not (np.isfinite(vv).all() and np.isfinite(yy).any()):
        return np.nan, np.nan, None, False

    # initial guesses
    jmax = np.nanargmax(yy)
    amp0 = max(yy[jmax] - np.nanmedian(yy), 1e-6)
    mu0  = float(vv[jmax])
    sig0 = max(0.1, 0.1*(vmax_kms - vmin_kms))  # km/s
    c0   = 0 # float(np.nanmedian(yy))

    g0 = models.Gaussian1D(amplitude=amp0, mean=mu0, stddev=sig0) + models.Const1D(amplitude=c0)
    g0.bounds['mean_0'] = (vmin_kms, vmax_kms)
    g0.bounds['stddev_0'] = (0.01, (vmax_kms - vmin_kms))
    g0.bounds['amplitude_0'] = (0.0, 1e3)

    fitter = fitting.LevMarLSQFitter()
    try:
        g = fitter(g0, vv, yy, maxiter=500)
        mu = float(g.mean_0.value)

        # 1-σ uncertainty on the centroid from covariance, if available
        mu_err = np.nan
        cov = fitter.fit_info.get('param_cov', None)
        if cov is not None and np.all(np.isfinite(cov)):
            # parameter order: [amplitude_0, mean_0, stddev_0, amplitude_1]
            mu_var = cov[1, 1]
            if np.isfinite(mu_var) and mu_var >= 0:
                mu_err = float(np.sqrt(mu_var))

        return mu, mu_err, (lambda x: g(x)), True
    except Exception:
        # fallback: argmax in window
        jj = np.nanargmax(yy)
        return float(vv[jj]), np.nan, None, False

# --- your extractor (unchanged) ---

def extract_pv_cut(source_name, molecule, degree_angle=0, position='edge'):
    folder_destination = os.path.join('TP_FITS', source_name)
    full_filename_path = find_the_spectrum_for_a_source(folder_destination, molecule)
    name_of_fits = full_filename_path.split('/')[-1]
    cube, wcs = open_fits_file(folder_destination, name_of_fits)
    path = define_pv_path_from_angle(cube, degree_angle, position)
    pv_slice = extract_pv_slice(cube, path)
    initial_ww = WCS(pv_slice.header)
    ww = update_WCS_coordinates(initial_ww, name_of_fits)
    return cube, path, pv_slice, ww

# --- main: minimal, readable, with optional lightweight diagnostics ---

def peak_value_for_each_radius(
    source_name,
    molecule,
    degree_angle,
    vel_range=[2, 11],
    plot_vel_range=[2,11],
    beam_arcsec=28.0,
    nyquist_factor=0.5,
    diag=False,        # show small spectra+fit panels
    diag_every=2,      # plot every Nth sampled column when diag=True
    max_diag=16,        # cap to avoid huge figures
    save=False
):
    positions = ['center_r', 'center_b']
    colors = ['C0', 'C1']

    pv_list, radii_list, vaxis_list = [], [], []
    vtrace_list, verr_list = [], []

    if plot_vel_range is None:
        plot_vmin_kms, plot_vmax_kms = map(float, vel_range)

    else:
        plot_vmin_kms, plot_vmax_kms = map(float, plot_vel_range)

    vmin_kms, vmax_kms = map(float, vel_range)


    for pos in positions:
        _, _, pv, ww = extract_pv_cut(source_name, molecule, degree_angle, position=pos)

        nx = pv.header['NAXIS1']  # spatial
        ny = pv.header['NAXIS2']  # velocity
        x = np.arange(nx)
        y = np.arange(ny)

        # world axes
        radial_deg = ww.pixel_to_world_values(x, 0)[0]     # deg
        vel_ms     = ww.pixel_to_world_values(0, y)[1]     # m/s
        vel_kms    = vel_ms / 1e3

        x_sel = nyquist_sample_indices(radial_deg, beam_arcsec, nyquist_factor)
        img = np.asarray(pv.data)  # shape (ny, nx)

        # diagnostics setup (small, simple)
        if diag:
            cols = 3
            rows = int(np.ceil(min(max_diag, len(x_sel[::diag_every])) / cols))
            figd, axd = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
            axd = axd.ravel()
            plotted = 0

        v_centroids, v_errors = [], []
        for k, j in enumerate(x_sel):
            spec = img[:, j]
            mu, mu_err, model, ok = fit_gaussian_centroid(vel_kms, spec, vmin_kms, vmax_kms)
            v_centroids.append(mu)
            v_errors.append(mu_err)

            # tiny diagnostic panel
            if diag and (k % diag_every == 0) and (plotted < max_diag):
                ax = axd[plotted]; plotted += 1
                win = (vel_kms >= vmin_kms) & (vel_kms <= vmax_kms)
                ax.plot(vel_kms[win], spec[win], lw=1.1, label='data')
                if model is not None:
                    ax.plot(vel_kms[win], model(vel_kms[win]), lw=1.1, label='fit')
                    ax.plot(vel_kms[win], spec[win]-model(vel_kms[win]), lw=0.9, alpha=0.7, label='resid')
                ax.axvline(mu, ls='--', lw=1.0, color='k')
                off_arcmin = radial_deg[j]*60.0
                ax.set_title(f"{pos} | off={off_arcmin:.2f}' | {'OK' if ok else 'FALLBACK'}", fontsize=9)
                ax.set_xlabel("V [km/s]"); ax.set_ylabel("Jy/beam")
                if model is not None: ax.legend(fontsize=8, loc='best')

                if save:
                    plt.savefig(
                        "Figures/pv_diagrams/velocity_gradients_with_fit/velocity_gradient_diag_plot_"+pos+'_'+ source_name + '_' + molecule + '_angle_+' + str(
                            degree_angle) + '.png',bbox_inches='tight',  dpi=300)
        if diag:
            # turn off unused panels
            for ax in axd[plotted:]:
                ax.axis('off')
            figd.suptitle(f"Diagnostics: {source_name} {molecule} | angle={degree_angle}° | {pos}", fontsize=12)
            figd.tight_layout()

            plt.show()


        #### This parts saves the images, and the positions and velocities of all relevant data points.
        #### This is done after the Gaussian is fit to the PV cut.
        pv_list.append(img)
        radii_list.append(radial_deg[x_sel]*60.0)  # arcmin
        vaxis_list.append(vel_kms)
        vtrace_list.append(np.array(v_centroids))
        verr_list.append(np.array(v_errors))

    # ---- overview plots (PV + traced centroids with 1σ error bars) ----
    plt.figure(figsize=(15, 6))

    ax1 = plt.subplot(131)
    ax1.imshow(
        pv_list[0], origin="lower", aspect="auto", cmap="viridis",
        extent=(radii_list[0][0], radii_list[0][-1], vaxis_list[0][0], vaxis_list[0][-1])
    )
    ax1.errorbar(radii_list[0], vtrace_list[0], yerr=verr_list[0], fmt='o', ms=4, capsize=2, color=colors[1])
    ax1.set_xlabel("Offset [arcmin]"); ax1.set_ylabel("Velocity [km/s]")
    ax1.set_ylim(plot_vmin_kms, plot_vmax_kms); ax1.set_title("center_r")


    ax2 = plt.subplot(132)
    ax2.imshow(
        pv_list[1], origin="lower", aspect="auto", cmap="viridis",
        extent=(radii_list[1][0], radii_list[1][-1], vaxis_list[1][0], vaxis_list[1][-1])
    )
    ax2.errorbar(radii_list[1], vtrace_list[1], yerr=verr_list[1], fmt='o', ms=4, capsize=2, color=colors[0])
    ax2.set_xlabel("Offset [arcmin]"); ax2.set_ylabel("Velocity [km/s]")
    ax2.set_ylim(plot_vmin_kms, plot_vmax_kms); ax2.set_title("center_b")

    v_lsr = vtrace_list[0][0]

    print('velocity centroid: ',v_lsr)
    ax3 = plt.subplot(133)

    # Build |V - Vlsr| (you already have v_lsr)
    dv_r = np.abs(vtrace_list[0][1:] - v_lsr)
    dv_b = np.abs(vtrace_list[1][1:] - v_lsr)

    # Convert radii to AU (your current recipe)
    dpc = 130
    r_r_au = (radii_list[0][1:]) * 60.0 * dpc
    r_b_au = (radii_list[1][1:]) * 60.0 * dpc

    verr_list_r=verr_list[0][1:]
    verr_list_b=verr_list[1][1:]

    # Plot points with error bars (centroid errors are also δV errors)
    ax3.errorbar(r_r_au, dv_r, yerr=verr_list_r, fmt='o', ms=5, capsize=2,mec='k',
                 color=colors[1], label='center_r data')
    ax3.errorbar(r_b_au, dv_b, yerr=verr_list_b, fmt='o', ms=5, capsize=2,mec='k',
                 color=colors[0], label='center_b data')

    # Log scales
    # ax3.set_xscale("log")
    # ax3.set_yscale("log")

    # --- NEW: fit power laws on the same axes and draw the lines ---
    fit_red = fit_powerlaw_and_plot(ax3, r_r_au, dv_r, yerr=verr_list_r, color=colors[1], label='center_r')
    fit_blue = fit_powerlaw_and_plot(ax3, r_b_au, dv_b, yerr=verr_list_b, color=colors[0], label='center_b')


    ax3.set_xlabel("Radial offset [au]")
    ax3.set_ylabel(r"$\delta V$ [km/s]")
    ax3.set_title("Peak Velocity vs Radius")
    ax3.grid(alpha=0.3, which='both')
    ax3.legend()

    plt.suptitle(
        f"{source_name} {molecule} | angle={degree_angle}° | v_lsr={v_lsr:.2f} | Nyquist={nyquist_factor * beam_arcsec:.0f}\"",
        fontsize=13)
    plt.tight_layout()

    # Optionally print results to console for quick copy/paste
    print("Power-law fits:")
    print("  center_r: alpha = {:.3f} ± {:.3f}, A = {:.3e}".format(
        fit_red['alpha'], fit_red['alpha_err'], fit_red['A']))
    print("  center_b: alpha = {:.3f} ± {:.3f}, A = {:.3e}".format(
        fit_blue['alpha'], fit_blue['alpha_err'], fit_blue['A']))


    if save:
        plt.savefig("Figures/pv_diagrams/velocity_gradients_with_fit/velocity_gradient_plot_"+source_name+'_'+molecule+'_angle_+'+str(degree_angle)+'.png',
                    bbox_inches='tight',dpi=300)

    plt.show()

    save_entry('cores_parameters.txt', source_name, molecule, 'red', fit_red, r_r_au, dv_r, verr_list_r )
    save_entry('cores_parameters.txt', source_name, molecule, 'blue', fit_blue, r_b_au, dv_b, verr_list_b)

    return {
        'radii_arcmin': radii_list,
        'vel_axis_kms': vaxis_list,
        'vtrace_kms': vtrace_list,
        'vtrace_err_kms': verr_list,
        'powerlaw_fits': {
            'center_r': fit_red,
            'center_b': fit_blue
        }
    }

    # return minimal outputs if you want to post-process
    # return {
    #     'radii_arcmin': radii_list,
    #     'vel_axis_kms': vaxis_list,
    #     'vtrace_kms': vtrace_list,
    #     'vtrace_err_kms': verr_list
    # }


def fit_powerlaw_and_plot(ax, x, y, yerr=None, color='C0', label=''):
    """
    Fit y = A * x^alpha by linear regression in log10 space and
    draw the fitted line on the given axis (assumed to be log-log).
    Returns dict(alpha, alpha_err, A).

    ax   : matplotlib Axes (already created; can be linear or log, we’ll just plot)
    x, y : 1D arrays (must be >0 to be usable on log scale)
    yerr : optional 1D array of 1σ errors on y (same units as y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)

    # keep only finite, positive points (required for log)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if yerr is not None:
        m &= np.isfinite(yerr)  # allow 0 errors; we’ll handle below
    if not np.any(m):
        return {'alpha': np.nan, 'alpha_err': np.nan, 'A': np.nan}

    xx = x[m]
    yy = y[m]
    if yerr is not None:
        ee = yerr[m]
        # convert σ(y) -> σ(log10 y): σ_logy ≈ σ_y / (y ln 10)
        with np.errstate(divide='ignore', invalid='ignore'):
            sig_logy = ee / (yy * np.log(10))
        # weights for np.polyfit are “w”, minimizing sum( w * (y - f)^2 )
        # For 1/σ^2 weighting in log space -> w = 1 / σ^2
        # np.polyfit uses w directly, not 1/σ, so we pass w = 1/σ_logy
        # but since it multiplies the residuals by w, standard choice is w=1/σ
        # We’ll follow numpy docs: w = 1/σ (log space)
        w = 1.0 / np.where(np.isfinite(sig_logy) & (sig_logy > 0), sig_logy, np.nan)
        goodw = np.isfinite(w) & (w > 0)
        xx, yy, w = xx[goodw], yy[goodw], w[goodw]
        X = np.log10(xx)
        Y = np.log10(yy)
        if len(X) < 2:
            return {'alpha': np.nan, 'alpha_err': np.nan, 'A': np.nan}
        (alpha, b), cov = np.polyfit(X, Y, deg=1, w=w, cov=True)
    else:
        X = np.log10(xx)
        Y = np.log10(yy)
        if len(X) < 2:
            return {'alpha': np.nan, 'alpha_err': np.nan, 'A': np.nan}
        (alpha, b), cov = np.polyfit(X, Y, deg=1, cov=True)

    alpha_err = float(np.sqrt(cov[0, 0])) if (cov is not None and np.isfinite(cov[0, 0])) else np.nan
    A = 10.0**b

    # draw fitted line over data span
    rx = np.linspace(xx.min(), xx.max(), 200)
    ax.plot(rx, A * rx**alpha, '--', color=color, lw=1,
            label=(f"{label} fit: α={alpha:.2f}±{alpha_err:.2f}" if label else None))
    return {'alpha': float(alpha), 'alpha_err': alpha_err, 'A': float(A)}

# --- example ---
if __name__ == "__main__":
    core = 'M493'
    molecule = 'C18O'
    angle = 136.8 #-25
    vel_range = [6.6, 8.0]

    peak_value_for_each_radius(
        core, molecule, angle, vel_range,
        plot_vel_range=None,
        beam_arcsec=28.0,
        nyquist_factor=0.50,  # 14"
        diag=True,           # quick visual check
        diag_every=1,        # every 2nd column
        max_diag=16,
        save=False
    )