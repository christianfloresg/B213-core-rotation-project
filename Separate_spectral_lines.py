import os
import matplotlib.pyplot as plt
import matplotlib
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from mpl_toolkits.axes_grid1 import ImageGrid
from make_TP_maps import find_all_spectra_for_a_molecule
from make_TP_maps import average_over_n_first_axis, find_the_spectrum_for_a_source
from astropy import units as u
import pyspeckit
import astropy.io.fits as pyfits
from spectral_cube import SpectralCube
from pyspeckit.spectrum.units import SpectroscopicAxis
import csv
from TwoD_spectral_maps import average_over_intervals

def read_csv_parameters(file_path):
    """
    Reads a CSV file containing parameters for multiple sources, ignoring comment lines (#).

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - params_dict (dict): A dictionary with source names as keys and their parameters as values.
    """
    params_dict = {}

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        # Read header, ignoring commented lines
        for header in reader:
            if header and not header[0].startswith("#"):
                break  # Found the valid header row

        # Convert header into column keys
        header = [h.strip() for h in header]

        # Process each row while ignoring commented lines
        for row in reader:
            if row and not row[0].startswith("#"):
                data = dict(zip(header, row))  # Match values with column names

                source = data["source"].strip()
                molecule = data["molecule"].strip()
                n_components = int(data["n_components"])
                guesses = [float(x) for x in data["guesses"].split(",")]

                params_dict[source] = {
                    "molecule": molecule,
                    "n_components": n_components,
                    "guesses": guesses
                }

    return params_dict

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_each_pixel(folder, molecule, num_gaussians, guesses, binning_factor=2):
    '''
    Fit each pixel of a datacube using a series of gaussinas defined by
    num_gaussians and with initial guesses given.
    NOTE:
    The limits are defined directly in the code and might be changed.
    There is a signal-to-noise cut for the gaussian fitting that can also be changed. currently = 5
    '''
    # Load your data cube
    if molecule is not None:
        full_filename_path = find_the_spectrum_for_a_source(folder, molecule)
        filename = os.path.basename(full_filename_path)

    full_path = os.path.join(folder, filename)
    cube = SpectralCube.read(full_path)

    # Define the rest frequency for the transition (adjust as necessary)
    rest_frequency = 219.5603541 * u.GHz

    # Convert the spectral axis to velocity using the radio convention
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio', rest_value=rest_frequency)

    axis_to_bin_y = 1  # Axis to bin along (0: spectral, 1: y-axis, 2: x-axis)
    axis_to_bin_x = 2  # Axis to bin along (0: spectral, 1: y-axis, 2: x-axis)

    # Apply the downsampling (binning)
    binned_cube2 = cube.downsample_axis(factor=binning_factor, axis=axis_to_bin_y)
    binned_cube = binned_cube2.downsample_axis(factor=binning_factor, axis=axis_to_bin_x)

    # Initialize the pyspeckit Cube with the data and the new spectral axis
    pcube = pyspeckit.Cube(cube=binned_cube)

    # Generate initial guesses for fitting N Gaussian components
    # Each Gaussian requires three parameters: amplitude, centroid, and width
    limits = []
    limited = []

    # guesses = [20, 5.5, 0.1, 30, 6.8, 0.1, 30, 7.0, 0.1]

    for i in range(num_gaussians):
        # Example initial guesses; adjust based on your data characteristics
        # guesses.extend([10, 4 + i, 0.1])  # Amplitude, Centroid, Sigma
        limits.extend([(3, 120), (guesses[3*i + 1] - 1.0, guesses[3*i + 1] + 1.0), (0.03, 0.8)])  # Adjust limits as necessary
        limited.extend([(True, True), (True, True), (True, True)])  # Enforce all limits

    # Perform Gaussian fitting with N components
    pcube.fiteach(fittype='gaussian', guesses=guesses,
                  limits=limits,
                  limited=limited,
                  start_from_point=(0, 0), multicore=4, signal_cut=5)

    plot_gaussian_maps(pcube, num_gaussians)

    return binned_cube, pcube


def plot_gaussian_maps(pcube, num_gaussians, save=True):
    plt.figure(figsize=(10, 3 * num_gaussians))

    levels = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])

    for i in range(num_gaussians):
        amplitude_map = pcube.parcube[3 * i, :, :]
        centroid_map = pcube.parcube[3 * i + 1, :, :]
        sigma_map = pcube.parcube[3 * i + 2, :, :]
        moment0_map = amplitude_map * sigma_map * np.sqrt(2 * np.pi)

        peak = np.percentile(moment0_map, 99)
        levels_i = levels * peak

        fig_fraction = 0.048
        fig_pad = 0.04
        plt.subplot(num_gaussians, 3, 3 * i + 1)
        plt.imshow(moment0_map, origin='lower', cmap='viridis', vmin=np.percentile(moment0_map, 1), vmax=peak)
        plt.title(f'Moment Zero Gauss {i + 1}')
        plt.colorbar(fraction=fig_fraction, pad=fig_pad, label='Integrated Intensity (Jy/beam * km/s)')
        contour = plt.contour(moment0_map, levels=levels_i, colors="black")
        plt.clabel(contour, inline=True, fontsize=8)

        # Apply masking to exclude zero values
        nonzero_values_centroid = centroid_map[centroid_map != 0]  # Keeps only non-zero elements
        plt.subplot(num_gaussians, 3, 3 * i + 2)
        plt.imshow(centroid_map, origin='lower', cmap='coolwarm', vmin=np.percentile(nonzero_values_centroid, 2), vmax=np.percentile(centroid_map, 98))
        plt.title(f'Centroid {i + 1}')
        plt.colorbar(fraction=fig_fraction, pad=fig_pad, label='Velocity (km/s)')

        nonzero_values_sigma = sigma_map[centroid_map != 0]  # Keeps only non-zero elements
        plt.subplot(num_gaussians, 3, 3 * i + 3)
        plt.imshow(sigma_map, origin='lower', cmap='jet', vmin=np.percentile(nonzero_values_sigma, 2), vmax=np.percentile(sigma_map, 98))
        plt.title(f'Sigma {i + 1}')
        plt.colorbar(fraction=fig_fraction, pad=fig_pad, label='Velocity (km/s)')

    plt.tight_layout()

    if save:
        source_name = folder.split('/')[-1]
        plt.savefig(os.path.join(save_folder, 'Separated_lines_moment_maps', source_name +'_n_gauss_'+ str(num_gaussians)),
                    bbox_inches='tight', dpi=300)
    plt.show()



def plot_gaussian_spectral_maps(folder, molecule, num_gaussians, guesses, save=True, plot=False):
    """
    Overplots the best-fit Gaussian models on the observed spectra using ImageGrid layout.

    Parameters:
    - folder: str, path to the data folder.
    - molecule: str, molecule name to identify the data file.
    - n_components: int, number of Gaussian components fitted.
    """
    def create_subplot_fit(cube, velocity, start_y, start_x, end_y, end_x, title):
        """
        Creates an ImageGrid subplot of the spectra with the fitted Gaussian models overlaid.
        """
        fig = plt.figure(figsize=(16, 16))
        grid = ImageGrid(fig, 111, nrows_ncols=(end_y - start_y, end_x - start_x), axes_pad=0.0, aspect=False)

        ### Because RA is defined opposite to pixels in array, this factor helps flip the x-axis of the plot
        grid_size_x = end_x - start_x
        invert_x_axis_factor = grid_size_x - 1
        max_value = np.nanmax(cube)

        for count, ax in enumerate(grid):
            row = count % (end_x - start_x)
            column = count // (end_y - start_y)

            # observed_spectrum = cube[:, start_y + column, start_x + row].value
            observed_spectrum = cube[:, start_y + invert_x_axis_factor - column, start_x + row].value

            ax.plot(velocity, observed_spectrum, label="Observed Spectrum", color='black',alpha=0.7)
            total_gaussian = np.zeros_like(velocity)  # Initialize array for summed Gaussian

            for i in range(num_gaussians):
                amp = pcube.parcube[3 * i, start_y + invert_x_axis_factor - column, start_x + row]
                cen = pcube.parcube[3 * i + 1, start_y + invert_x_axis_factor - column, start_x + row]
                sig = pcube.parcube[3 * i + 2, start_y + invert_x_axis_factor - column, start_x + row]

                gaussian_fit = gaussian(velocity, amp, cen, sig)
                total_gaussian += gaussian_fit  # Sum all Gaussian components

                ax.plot(velocity, gaussian_fit, linestyle="--", label=f"Gaussian Fit {i + 1}",alpha=0.7, linewidth=1.0)

            # Plot the summed Gaussian model
            if num_gaussians>1:
                ax.plot(velocity, total_gaussian, linestyle="--", label="Sum of Gaussians", color='red', linewidth=0.5)

            min_vel = min(initial_guess_for_gaussians[1::3])
            max_vel = max(initial_guess_for_gaussians[1::3])
            ax.set_xlim(min_vel-1.5 , max_vel+1.5)
            ax.set_ylim(-1, max_value*1.01)
            ax.tick_params(axis='both', which='major', labelsize=6)


        plt.suptitle(title, fontsize=18)
        fig.supylabel('Intensity (Jy/Beam)', fontsize=14)
        fig.supxlabel('Velocity (km/s)', fontsize=14)
        if save:
            title_to_save = title.replace(" ", "")
            fig.savefig(os.path.join(save_folder, 'Separated_lines_spectral_maps', source_name +'_n_gauss_'+ str(num_gaussians) + '_'+ title_to_save),
                        bbox_inches='tight', dpi=500)

    # Fit N-Gaussian components
    cube, pcube = fit_each_pixel(folder, molecule, num_gaussians, guesses)
    source_name = folder.split('/')[-1]
    velocity_axis = cube.spectral_axis.value  # Extract velocity axis
    grid_size_y, grid_size_x = cube.shape[1], cube.shape[2]

    # print(grid_size_x,grid_size_y)
    # # Plot for different quadrants
    # create_subplot_fit(cube, velocity_axis, 0, 0, 16, 16, 'Bottom-Left Quadrant')
    # create_subplot_fit(cube, velocity_axis, 0, 16, 16, 32, 'Bottom-Right Quadrant')
    # create_subplot_fit(cube, velocity_axis, 16, 0, 32, 16, 'Top-Left Quadrant')
    # create_subplot_fit(cube, velocity_axis, 16, 16, 32, 32, 'Top-Right Quadrant')
    create_subplot_fit(cube, velocity_axis, 0, 0, grid_size_y, grid_size_x, source_name+'_Full')
    if plot:
        plt.show()


def create_spectral_maps_just_data(path,molecule=None,filename=None,save=False,show=True,binning=1):
    '''
    The current griding of the data is 34x32. This is inconvenient to produce the spectral map
    I will just ignore the true size and assume it is 32x32 and produce a grid of spectra of
    size 8x8 with an average of 4 spaxcels.
    Since the FOV is ~1' 40'', and the beam size is 28.6 arcsec, there are only about 4
    beams pear cube side. It also makes sense to create a grid of 4x4 with average of 8 paxcels.
    '''
    aperture_radius = 14.3 ## A 12m antenna at 219.56 GHz or 1.365 mm produces an angular resolution of 28.6 arcsec

    def create_subplot(data, velocity, start_y, start_x, end_y, end_x, title):
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(end_y - start_y, end_x - start_x),
                         axes_pad=0.0, aspect=False)

        max_value = np.nanmax(data)

        for count, ax in enumerate(grid):
            row = count % (end_x - start_x)
            column = count // (end_y - start_y)

            spectrum = data[:, start_y + column, start_x + row]

            ax.plot(velocity, spectrum)
            ax.set_xlim(3, 11)
            ax.set_ylim(-1, max_value + 0.5)

        plt.suptitle(title, fontsize=18)
        fig.supylabel('Intensity (Jy/Beam)', fontsize=14)
        fig.supxlabel('Velocity (km/s)', fontsize=14)

        
    if molecule!=None:
        full_filename_path = find_the_spectrum_for_a_source(path, molecule)
        filename = full_filename_path.split('/')[-1]
    print(filename)

    # elif filename!=None:
        
    data_cube = ALMATPData(path, filename)
    velocity = data_cube.vel
    image = data_cube.ppv_data
    velocity_resolution = data_cube.velocity_resolution

    if binning>1:
        image = average_over_n_first_axis(image,binning)
        velocity = average_over_n_first_axis(velocity,binning)
        velocity_resolution =velocity_resolution*binning

    source_name = path.split('/')[-1]

    D, H, W = image.shape
    grid_size_y = H
    grid_size_x = W 

    subplot_size_y, subplot_size_x = 8, 8

    # Top-left quadrant
    create_subplot(image, velocity, 0, 0, 8, 8, 'Top-Left Quadrant')

    # Top-right quadrant
    create_subplot(image, velocity, 0, 8, 8, 16, 'Top-Right Quadrant')

    # Bottom-left quadrant
    create_subplot(image, velocity, 8, 0, 16, 8, 'Bottom-Left Quadrant')

    # Bottom-right quadrant
    create_subplot(image, velocity, 8, 8, 16, 16, 'Bottom-Right Quadrant')

    if save:
        # save_path =
        fig.savefig(os.path.join(*[save_folder,data_cube.molec_name,source_name])+'_binning_'+str(binning), bbox_inches='tight',dpi=300)
    if show:
        plt.show()

if __name__ == "__main__":

    save_folder = 'Figures/Separated_lines/'

    folder = 'TP_FITS/M365'
    molecule = 'C18O'

    params_file = "gaussian_params_guess.csv"  # Path to your CSV file
    parameters = read_csv_parameters(params_file)
    number_of_gaussian_components = parameters[folder]["n_components"]
    initial_guess_for_gaussians = parameters[folder]["guesses"]

    # Plot the Gaussian fits over the spectral maps
    plot_gaussian_spectral_maps(folder, molecule,number_of_gaussian_components,initial_guess_for_gaussians)