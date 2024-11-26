import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
# from scipy.ndimage import map_coordinates
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import os
from pvextractor import Path, extract_pv_slice
from astropy.constants import c
from matplotlib.ticker import FormatStrFormatter
from make_TP_maps import find_the_spectrum_for_a_source, ALMATPData

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

def define_pv_path_from_coordinates(start_coordinate=["4h17m39s", "+27d55m12s"],
                                    end_coordinate=["4h17m34s", "+27d56m12s"]):


    start_point = SkyCoord(start_coordinate[0], start_coordinate[1], frame="fk5")
    end_point = SkyCoord(end_coordinate[0], end_coordinate[1], frame="fk5")

    # Create a linear path between the points
    ra_values = [start_point.ra.deg, end_point.ra.deg]  # Convert RA to degrees
    dec_values = [start_point.dec.deg, end_point.dec.deg]  # Convert Dec to degrees
    # Create the Path
    path = Path(SkyCoord(ra_values, dec_values, unit="deg", frame="fk5"), width=10 * u.arcsec)

    return path

def define_pv_path_from_angle(cube,angle):
    '''
    define a pv path based on the center of the field, and an angle
    '''
    # Define the angle alpha (measured north from east)
    alpha = angle * u.deg  # Adjust angle as needed

    cube_wcs = WCS(cube.header)
    wcs = cube_wcs.wcs
    print(cube_wcs)
    # Get the spatial dimensions (x, y)
    nx, ny = cube.shape[1], cube.shape[2]  # Assuming cube.shape = [x, y, spectral, ]
    print(nx,ny,'spatial coordinates')

    # Get the reference pixel for the spectral axis
    ref_pixel_spectral = cube_wcs.wcs.crpix[2] - 1  # Subtract 1 because FITS is 1-indexed
    print(ref_pixel_spectral,'sectral index  coordinates')

    center_pixel = [nx / 2, ny / 2]  # Center pixel coordinates

    world_coords_center, freq_center = cube_wcs.pixel_to_world(center_pixel[0], center_pixel[1], ref_pixel_spectral)

    center_ra = world_coords_center.ra.deg
    center_dec = world_coords_center.dec.deg
    
    # Convert alpha to radians
    alpha_rad = np.deg2rad(alpha)

    # Get the pixel scale in degrees
    pixel_scale_x = np.abs(cube_wcs.wcs.cdelt[0])  # Degrees per pixel in RA
    pixel_scale_y = np.abs(cube_wcs.wcs.cdelt[1])  # Degrees per pixel in Dec

    # Calculate the diagonal distance in degrees
    fov_diag = np.sqrt((nx * pixel_scale_x)**2 + (ny * pixel_scale_y)**2) / 2

    # Calculate RA/Dec offsets
    delta_ra = fov_diag * np.cos(alpha_rad) / np.cos(np.deg2rad(center_dec))  # Adjust for declination
    delta_dec = fov_diag * np.sin(alpha_rad)

    # Compute start and end points
    start_ra = center_ra - delta_ra
    start_dec = center_dec - delta_dec
    end_ra = center_ra + delta_ra
    end_dec = center_dec + delta_dec

    # Create the Path object
    ra_values = [start_ra.value, end_ra.value]
    dec_values = [start_dec.value, end_dec.value]

    path = Path(SkyCoord(ra_values, dec_values, unit="deg", frame="fk5"), width=25 * u.arcsec)

    return path

def extract_pv_cut(source_name,molecule,degree_angle=0):

    folder_destination = os.path.join('TP_FITS', source_name)
    full_filename_path = find_the_spectrum_for_a_source(folder_destination, molecule)

    name_of_fits = full_filename_path.split('/')[-1]

    cube, wcs = open_fits_file(folder_destination,name_of_fits)
    # Step 2: Define the spatial path for the PV cut
    # Define start and end points of the path in celestial coordinates

    # path = define_pv_path_from_coordinates(["4h17m39s", "+27d55m12s"],["4h17m34s", "+27d56m12s"])
    path = define_pv_path_from_angle(cube,degree_angle)

    # Step 3: Extract the PV slice
    pv_slice = extract_pv_slice(cube, path)

    initial_ww = WCS(pv_slice.header)
    ww = update_WCS_coordinates(initial_ww,name_of_fits)

    return cube, path, pv_slice, ww

def plot_extraced_pv_cut(source_name,molecule,degree_angle=0):
    
    cube, path, pv_slice, ww = extract_pv_cut(source_name, molecule, degree_angle=degree_angle)

    data = cube.unmasked_data[:].value

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121, projection=cube.wcs.celestial)
    ax.imshow(data[0,:,:])
    # path.show_on_axis(ax, spacing=1, color='r')
    path.show_on_axis(ax, spacing=5,
                          edgecolor='w', linestyle=':',
                          linewidth=0.75)

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

    vel_min = 6.*1e3  # Minimum velocity in meter/s
    vel_max = 7.5*1e3  # Maximum velocity in meter/s

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
    ax2.coords[1].set_major_formatter('x.x')
    ax2.coords[0].set_major_formatter('x.xx')

    ax2.set_xlabel("Offset [arcmin]")
    ax2.set_ylabel("Velocity [km/s]")

    plt.colorbar(pv_image, ax=ax2, label="Intensity")
    plt.tight_layout()
    plt.show()

def plot_pv_and_moment_one(source_name,molecule,path,degree_angle=0):

    mom_one_name = molecule+'_'+core
    data_cube = ALMATPData(path, mom_one_name+'_mom1.fits')
    image_mom_1 = data_cube.ppv_data
    unique_moment_1_val=np.unique(image_mom_1)

    cube, path, pv_slice, ww = extract_pv_cut(source_name, molecule, degree_angle=degree_angle)
    data = cube.unmasked_data[:].value

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121, projection=cube.wcs.celestial)
    moment_one_plot = ax.imshow(image_mom_1, cmap="coolwarm", origin='lower',vmin=unique_moment_1_val[1])

    # path.show_on_axis(ax, spacing=1, color='r')
    path.show_on_axis(ax, spacing=5,
                          edgecolor='k', linestyle=':',
                          linewidth=0.75)

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

    vel_min = 5.0*1e3  # Minimum velocity in meter/s
    vel_max = 8.0*1e3  # Maximum velocity in meter/s

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
    ax2.coords[1].set_major_formatter('x.x')
    ax2.coords[0].set_major_formatter('x.xx')

    ax2.set_xlabel("Offset [arcmin]")
    ax2.set_ylabel("Velocity [km/s]")

    plt.colorbar(moment_one_plot, ax=ax, label="velocity")
    # plt.colorbar(pv_image, ax=ax2, label="Intensity")

    plt.suptitle(source_name + ' ' + molecule, fontsize=18)
    plt.tight_layout()
    plt.savefig("Figures/pv_diagrams/"+source_name+'_'+molecule+'.png', bbox_inches='tight',dpi=300)
    plt.show()


if __name__ == "__main__":
    core = 'M505'
    molecule = 'C18O'
    angle = 45+90
    path = os.path.join('moment_maps_fits',core)
    plot_pv_and_moment_one(core, molecule, path, degree_angle=angle)
    # plot_extraced_pv_cut(source_name=core,molecule=molecule,degree_angle=angle)
    # print(full_filename_path)

    # folder_destination = 'linearfit_demo'
    # name_of_fits= 'hl_tau.h2co.contsub.selfcal.robust0.5.pbcor.subim.fits'
    # extract_pv_cut(source_name=core,molecule=molecule,degree_angle=angle)
