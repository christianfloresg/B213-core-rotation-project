import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from mpl_toolkits.axes_grid1 import ImageGrid


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

def create_spectral_maps(path,filename):
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

    averaged_cube = average_over_intervals(image, N, M)

    # average_spectrum = np.nanmean(image, axis=(1, 2))

    # number_of_sources = len( sorted(next(os.walk(folders_path))[1]))
    # grid_size = int(math.ceil(number_of_sources ** 0.5))
    # fig = plt.figure(figsize=(15., 15.))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                  nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of Axes
    #                  axes_pad=0.3, aspect=False  # pad between Axes in inch.
    #                  )

    # print(np.shape(image[50,:,:]))
    # plt.imshow(image[50,:,:])
    # plt.show()

if __name__ == "__main__":
    source ='M262'
    folder_destination = os.path.join('TP_FITS', source)
    name_of_fits = 'member.uid___A001_X15aa_X29e.M262_sci.spw19.cube.I.sd.fits'

    # moment_maps_with_continuum(path='moment_maps_fits/'+source+'/', filename='C18O_'+source,save=True)
    # skycoord_object = SkyCoord('04:17:32 +27:41:35', unit=(u.hourangle, u.deg))

    create_spectral_maps(path=folder_destination, filename=name_of_fits)