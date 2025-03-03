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

def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_each_pixel(folder,molecule):
    # Load your data cube
    if molecule!=None:
        full_filename_path = find_the_spectrum_for_a_source(folder, molecule)
        filename = full_filename_path.split('/')[-1]

    full_path = os.path.join(folder,filename)
    cube = SpectralCube.read(full_path)

    # Define the rest frequency for the C18O (2-1) transition
    rest_frequency = 219.5603541 * u.GHz

    # Convert the spectral axis to velocity using the radio convention
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio', rest_value=rest_frequency)

    # print('the array ',cube.spectral_axis.value)

    # Initialize the pyspeckit Cube with the data and the new spectral axis
    pcube = pyspeckit.Cube(cube=cube)

    # Generate initial guesses for fitting two Gaussian components
    # Each Gaussian requires three parameters: amplitude, centroid, and width
    # For two Gaussians, we need six parameters in total
    
    guesses = np.array([10,5.5,0.2, 30,7,0.2])

    # Define limits for each parameter
    # Format: [(min1, max1), (min2, max2), ..., (minN, maxN)]
    limits = [(3, 40),  # Amplitude1: must be between 0 and 100
              (4, 6),  # Center1: must be between 5 and 6
              (0.1, 0.5),  # Sigma1: must be between 0.5 and 2
              (10, 80),  # Amplitude2: must be between 0 and 100
              (6, 8),  # Center2: must be between 6 and 8
              (0.1, 0.5)]  # Sigma2: must be between 0.5 and 2

    # Indicate which limits are to be enforced
    # Format: [(min1_bool, max1_bool), (min2_bool, max2_bool), ..., (minN_bool, maxN_bool)]
    limited = [(True, True),  # Amplitude1 is bounded on both sides
               (True, True),  # Center1 is bounded on both sides
               (True, True),  # Sigma1 is bounded on both sides
               (True, True),  # Amplitude2 is bounded on both sides
               (True, True),  # Center2 is bounded on both sides
               (True, True)]  # Sigma2 is bounded on both sides

    # Perform Gaussian fitting with two components
    pcube.fiteach(fittype='gaussian', guesses=guesses,
                  limits=limits,
                  limited=limited,
                  start_from_point=(0,0), multicore=4, signal_cut=5)

    plot_gaussian_maps(pcube)

    return cube, pcube


def plot_gaussian_maps(pcube):

    # Access fitted parameters
    amplitude_map1 = pcube.parcube[0, :, :]
    centroid_map1 = pcube.parcube[1, :, :]
    sigma_map1 = pcube.parcube[2, :, :]

    amplitude_map2 = pcube.parcube[3, :, :]
    centroid_map2 = pcube.parcube[4, :, :]
    sigma_map2 = pcube.parcube[5, :, :]

    peak1 = np.percentile(amplitude_map1, 99)
    levels = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    levels1 = levels * peak1

    peak2 = np.percentile(amplitude_map2, 99)
    levels2 = levels * peak2

    plt.figure(figsize=(20, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(amplitude_map1, origin='lower', cmap='viridis',vmin=np.percentile(amplitude_map1, 1),vmax= peak1)
    plt.title('Amplitude 1')
    plt.colorbar()
    contour = plt.contour(amplitude_map1, levels=levels1, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)


    plt.subplot(2, 3, 2)
    plt.imshow(centroid_map1, origin='lower', cmap='coolwarm',vmin=np.percentile(centroid_map1, 1),vmax=np.percentile(centroid_map1, 99))
    plt.title('Centroid 1')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(sigma_map1, origin='lower', cmap='jet',vmin=np.percentile(sigma_map1, 1),vmax=np.percentile(sigma_map1, 99))
    plt.title('Sigma 1')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(amplitude_map2, origin='lower', cmap='viridis',vmin=np.percentile(amplitude_map2, 1),vmax=peak2)
    plt.colorbar()
    contour = plt.contour(amplitude_map2, levels=levels2, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)
    plt.title('Amplitude 2')

    plt.subplot(2, 3, 5)
    plt.imshow(centroid_map2, origin='lower', cmap='coolwarm',vmin=np.percentile(centroid_map2, 1),vmax=np.percentile(centroid_map2, 99))
    plt.title('Centroid 2')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(sigma_map2, origin='lower', cmap='jet', vmin=np.percentile(sigma_map2, 1),vmax=np.percentile(sigma_map2, 99))
    plt.title('Sigma 2')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_gaussian_spectral_maps(folder,molecule):
    """
    Overplots the best-fit Gaussian models on the observed spectra using ImageGrid layout.
    """

    def create_subplot_fit(cube, velocity, start_y, start_x, end_y, end_x, title):

        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(end_y - start_y, end_x - start_x), axes_pad=0.0, aspect=False)

        for count, ax in enumerate(grid):
            row = count % (end_x - start_x)
            column = count // (end_y - start_y)
            # if row >= grid_size_x or column >= grid_size_y:
            #     continue

            observed_spectrum = cube[:, start_y + column, start_x + row].value
            amp1, cen1, sig1 = amplitude_map1[start_y + column, start_x + row], \
                               centroid_map1[start_y + column, start_x + row], \
                               sigma_map1[start_y + column, start_x + row]
            amp2, cen2, sig2 = amplitude_map2[start_y + column, start_x + row],\
                               centroid_map2[start_y + column, start_x + row], \
                               sigma_map2[start_y + column, start_x + row]

            gaussian_fit1 = gaussian(velocity_axis, amp1, cen1, sig1)
            gaussian_fit2 = gaussian(velocity_axis, amp2, cen2, sig2)

            ax.plot(velocity_axis, observed_spectrum, label="Observed Spectrum")
            ax.plot(velocity_axis, gaussian_fit1, linestyle="--", label="Gaussian Fit 1", color="red")
            ax.plot(velocity_axis, gaussian_fit2, linestyle="--", label="Gaussian Fit 2", color="blue")

            ax.set_xlim(3, 11)
            ax.set_ylim(-1, max(observed_spectrum) + 0.5)
            ax.tick_params(axis='both', which='major', labelsize=6)

        plt.suptitle(title, fontsize=18)
        fig.supylabel('Intensity (Jy/Beam)', fontsize=14)
        fig.supxlabel('Velocity (km/s)', fontsize=14)
        
    cube, pcube = fit_each_pixel(folder, molecule)

    velocity_axis = cube.spectral_axis.value  # Extract velocity axis
    amplitude_map1 = pcube.parcube[0, :, :]
    centroid_map1 = pcube.parcube[1, :, :]
    sigma_map1 = pcube.parcube[2, :, :]

    amplitude_map2 = pcube.parcube[3, :, :]
    centroid_map2 = pcube.parcube[4, :, :]
    sigma_map2 = pcube.parcube[5, :, :]

    grid_size_y, grid_size_x = cube.shape[1], cube.shape[2]

    # Top-left quadrant
    create_subplot_fit(cube, velocity_axis, 0, 0, 8, 8, 'Top-Left Quadrant')

    # Top-right quadrant
    create_subplot_fit(cube, velocity_axis, 0, 8, 8, 16, 'Top-Right Quadrant')

    # Bottom-left quadrant
    create_subplot_fit(cube, velocity_axis, 8, 0, 16, 8, 'Bottom-Left Quadrant')

    # Bottom-right quadrant
    create_subplot_fit(cube, velocity_axis, 8, 8, 16, 16, 'Bottom-Right Quadrant')
    
    plt.show()

def create_spectral_maps(path,molecule=None,filename=None,save=False,show=True,binning=1):
    '''
    The current griding of the data is 34x32. This is inconvenient to produce the spectral map
    I will just ignore the true size and assume it is 32x32 and produce a grid of spectra of
    size 8x8 with an average of 4 spaxcels.
    Since the FOV is ~1' 40'', and the beam size is 28.6 arcsec, there are only about 4
    beams pear cube side. It also makes sense to create a grid of 4x4 with average of 8 paxcels.
    '''
    aperture_radius = 14.3 ## A 12m antenna at 219.56 GHz or 1.365 mm produces an angular resolution of 28.6 arcsec

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

if __name__ == "__main__":

    save_folder = 'Figures/Spectral_maps/'

    ### Creation of a single spectral map
    # source ='M490'
    # folder_destination = os.path.join('TP_FITS', source)
    # create_spectral_maps(path = 'TP_FITS/M273', molecule = 'C18O' ,save=False,binning=2)


    ### Creation of a maps for all sources for a given molecule
    folder = 'TP_FITS/M273/'
    molecule = 'C18O'
    # fit_each_pixel(folder,molecule)
    # mass_produce_spectral_maps(folder_fits, molecule='C18O',binning=1)

    # Plot the Gaussian fits over the spectral maps
    plot_gaussian_spectral_maps(folder, molecule)