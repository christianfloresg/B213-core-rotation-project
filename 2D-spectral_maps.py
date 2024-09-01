import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from mpl_toolkits.axes_grid1 import ImageGrid
from make_TP_maps import find_all_spectra_for_a_molecule

def average_over_intervals(data_cube, N, M):
    """
    Averages a 3D data cube over the second and third dimensions every N and M intervals.

    Parameters:
    data_cube (numpy array): The 3D array to be averaged.
    N (int): The interval for averaging along the second dimension.
    M (int): The interval for averaging along the third dimension.

    Returns:
    numpy array: The averaged 3D array.
    """
    # Get the shape of the original data cube
    D, H, W = data_cube.shape

    # Determine the new shape after averaging
    new_H = H // N
    new_W = W // M

    # Initialize the output array with the new shape
    averaged_cube = np.zeros((D, new_H, new_W))

    for i in range(new_H):
        for j in range(new_W):
            # Slice the original cube to take N x M intervals
            slice_chunk = data_cube[:, i * N:(i + 1) * N, j * M:(j + 1) * M]
            # Calculate the mean over the second and third dimensions
            averaged_cube[:, i, j] = np.mean(slice_chunk, axis=(1, 2))

    return averaged_cube

def create_spectral_maps(path,filename,save=False,show=True):
    '''
    The current griding of the data is 34x32. This is inconvenient to produce the spectral map
    I will just ignore the true size and assume it is 32x32 and produce a grid of spectra of
    size 8x8 with an average of 4 spaxcels.
    Since the FOV is ~1' 40'', and the beam size is 28.6 arcsec, there are only about 4
    beams pear cube side. It also makes sense to create a grid of 4x4 with average of 8 paxcels.
    '''
    aperture_radius = 14.3 ## A 12m antenna at 219.56 GHz or 1.365 mm produces an angular resolution of 28.6 arcsec

    data_cube = ALMATPData(path, filename)
    velocity = data_cube.vel
    image = data_cube.ppv_data
    velocity_resolution = data_cube.velocity_resolution

    source_name = path.split('/')[-1]

    D, H, W = image.shape
    averaged_pixels = 4
    grid_size_y = H // averaged_pixels
    grid_size_x = W // averaged_pixels

    averaged_cube = average_over_intervals(image, averaged_pixels, averaged_pixels)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(grid_size_y, grid_size_x),  # creates 2x2 grid of Axes
                     axes_pad=0.0, aspect=False  # pad between Axes in inch.
                     )

    max_value = np.nanmax(averaged_cube)

    ### Because RA is defined opposite to pixels in array, this factor helps flip the x-axis of the plot
    invert_x_axis_factor = grid_size_x - 1

    if data_cube.velocity_resolution > 1:
        velocity_limits =(-2, 17)
    else:
        velocity_limits = (3, 11)

    for count,ax in enumerate(grid):
        row =  count % grid_size_x
        column = count // grid_size_y

        spectrum = averaged_cube[:, invert_x_axis_factor - column, row]

        plot = ax.plot(velocity, spectrum)

        ax.set_xlim(velocity_limits)
        ax.set_ylim(-1,max_value+0.5)

    plt.suptitle('Source: '+source_name+' -  Molec: ' + data_cube.molec_name, fontsize=18)
    fig.supylabel('Intensity (K)', fontsize= 14)
    fig.supxlabel('Velocity (km/s)', fontsize= 14)
    if save:
        # save_path =
        fig.savefig(os.path.join(*[save_folder,data_cube.molec_name,source_name]), bbox_inches='tight',dpi=300)
    if show:
        plt.show()

def mass_produce_spectral_maps(folders_path,molecule='12CO'):
    '''
    Apply the plot_moment_maps(path, filename) function
    to all the files in a folder, for  agiven molcule.
    '''
    array_of_paths = find_all_spectra_for_a_molecule(folders_path, molecule)

    for source_path in array_of_paths:
        try:
            create_spectral_maps(path=os.path.join(folders_path,source_path.split('/')[0]),
                             filename=source_path.split('/')[1],
                             save=True,show=False)
        except IndexError as err:
            print('map '+source_path.split('/')[0]+' was not produced. Check the moment maps.')

if __name__ == "__main__":

    save_folder = 'Figures/Spectral_maps/'

    ### Creation of a single spectral map
    # source ='M275'
    # folder_destination = os.path.join('TP_FITS', source)
    # name_of_fits = 'member.uid___A001_X15aa_X29e.M275_sci.spw17.cube.I.sd.fits'
    # create_spectral_maps(path=folder_destination, filename=name_of_fits ,save=True)

    ### Creation of a maps for all sources for a given molecule
    folder_fits = 'TP_FITS'
    mass_produce_spectral_maps(folder_fits, molecule='DCO+')