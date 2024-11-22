import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import map_coordinates
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import os
from pvextractor import Path, extract_pv_slice
from astropy.constants import c
# from astropy.visualization.wcsaxes import CoordinateHelper
from matplotlib.ticker import FormatStrFormatter


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

def open_fits_file(folder,fits_name):

    # Step 1: Load the datacube and WCS
    cube = SpectralCube.read(os.path.join(folder,fits_name))  # Load the datacube
    header = cube.header
    wcs = cube.wcs  # Extract WCS information

    return cube, wcs

def update_WCS_coordinates(initial_ww,filename):

    rest_frequency = accurate_reference_frequency(filename)

    ww = initial_ww
    spectral_axis_index = ww.wcs.spec

    # Extract current WCS parameters for the spectral axis
    crval_freq = ww.wcs.crval[spectral_axis_index]  # Reference frequency (CRVAL)
    cdelt_freq = ww.wcs.cdelt[spectral_axis_index]  # Increment per pixel in frequency
    crpix_freq = ww.wcs.crpix[spectral_axis_index]  # Reference pixel

    print(c.to('km/s').value,'THIS TRANS')
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

def extract_pv_cut(folder,fits_name):

    cube, wcs = open_fits_file(folder,fits_name)
    data = cube.unmasked_data[:].value
    # Step 2: Define the spatial path for the PV cut
    # Define start and end points of the path in celestial coordinates
    start_point = SkyCoord("4h17m39s", "+27d55m12s", frame="fk5")
    end_point = SkyCoord("4h17m34s", "+27d56m12s", frame="fk5")

    # Create a linear path between the points
    ra_values = [start_point.ra.deg, end_point.ra.deg]  # Convert RA to degrees
    dec_values = [start_point.dec.deg, end_point.dec.deg]  # Convert Dec to degrees

    # Create the Path
    path = Path(SkyCoord(ra_values, dec_values, unit="deg", frame="fk5"), width=1 * u.arcsec)

    # Step 3: Extract the PV slice
    pv_slice = extract_pv_slice(cube, path)

    initial_ww = WCS(pv_slice.header)
    ww = update_WCS_coordinates(initial_ww,fits_name)

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121, projection=cube.wcs.celestial)
    ax.imshow(data[510,:,:])
    path.show_on_axis(ax, spacing=1, color='r')

    ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
    ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")

    ax2 = plt.subplot(122, projection=ww)
    # im = ax.imshow(pv_slice.data)

    # spatial_extent = [0, pv_slice.data.shape[1]]
    pv_image = ax2.imshow(
        pv_slice.data,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        # extent=(spatial_extent[0], spatial_extent[1], 0, 10)
    )

    vel_min = 4*1e3  # Minimum velocity in meter/s
    vel_max = 8.*1e3  # Maximum velocity in meter/s

    # Convert velocity range to pixel indices
    spatial_placeholder = 0  # Assuming single spatial point or fixed placeholder
    velocities = np.array([vel_min, vel_max])  # Velocity range

    # Replace 'None' with placeholders for spatial axes
    velocity_pixels = ww.world_to_pixel_values(spatial_placeholder,velocities)
    # velocity_pixels = ww.world_to_pixel_values(0,15)

    pixel_min = velocity_pixels[1][0]  # Corresponding to vel_min
    pixel_max = velocity_pixels[1][1]  # Corresponding to vel_max
    print(pixel_min,pixel_max)

    ax2.set_ylim(pixel_min, pixel_max)

    # ax2.set_ylim(0, 10)  # World coordinate limits for velocity
    ax2.coords[1].set_format_unit(u.km / u.s)  # Format the velocity axis
    ax2.coords[0].set_format_unit(u.arcmin)  # Format the spatial axis

    ax2.set_xlabel("Offset [arcmin]")
    ax2.set_ylabel("Velocity [km/s]")

    plt.colorbar(pv_image, ax=ax2, label="Intensity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    core = 'M275'
    folder_destination = os.path.join('TP_FITS', core)
    name_of_fits = 'member.uid___A001_X15aa_X29e.M275_sci.spw21.cube.I.sd.fits'
    extract_pv_cut(folder=folder_destination,fits_name=name_of_fits)