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

# def edge_pixel_from_angle(alpha,nx,ny):
#     '''
#     calculate the position of the edge pixel of a cube
#     let's define it east from north.
#     '''
#
#     x_c = nx / 2 - 0.5
#     y_c =  ny / 2 - 0.5
#
#     alpha_rad = np.pi*alpha/180.
#     if 0<=alpha<=45:
#         yf = ny +0.5
#         xf = x_c - np.tan(alpha_rad)*ny/2
#
#         y2f = 0.5
#         x2f = x_c + np.tan(alpha_rad)*ny/2
#
#     if 45<alpha<=135:
#         xf = 0.5
#         yf = y_c + nx/2 / np.tan(alpha_rad)
#
#         x2f =  nx + 0.5
#         y2f = y_c - nx/2 / np.tan(alpha_rad)
#
#     if 135<alpha<=180:
#         yf = 0.5
#         xf = x_c + np.tan(alpha_rad)*ny/2
#
#         y2f = ny + 0.5
#         x2f = x_c - np.tan(alpha_rad)*ny/2
#
#     return [xf,yf],[x2f,y2f]

def define_pv_path_from_angle(cube,angle,position='edge'):
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
    print(nx,ny,'spatial coordinates')

    # Get the reference pixel for the spectral axis
    ref_pixel_spectral = cube_wcs.wcs.crpix[2] - 1  # Subtract 1 because FITS is 1-indexed
    print(ref_pixel_spectral,'sectral index  coordinates')

    edge_pixel1, edge_pixel2 = edge_pixel_from_angle(angle,nx,ny)
    print(edge_pixel1,'edge pixels 1')
    print(edge_pixel2,'edge pixels 2')

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

    
    if position=='center_r':
        center_pixel = [nx / 2 - 0.5, ny / 2 -0.5]  # Center pixel coordinates | the center of the pixel is at 0.5 units
        world_coords_center, freq_center = cube_wcs.pixel_to_world(center_pixel[0], center_pixel[1], ref_pixel_spectral)

        center_ra = world_coords_center.ra.deg
        ceter_dec = world_coords_center.dec.deg

        ra_values = [center_ra, edge_ra1]
        dec_values = [ceter_dec, edge_dec1]

    elif position=='center_b':

        center_pixel = [nx / 2 - 0.5, ny / 2 -0.5]  # Center pixel coordinates | the center of the pixel is at 0.5 units
        world_coords_center, freq_center = cube_wcs.pixel_to_world(center_pixel[0], center_pixel[1], ref_pixel_spectral)

        center_ra = world_coords_center.ra.deg
        ceter_dec = world_coords_center.dec.deg

        # Create the Path
        ra_values = [center_ra, edge_ra2]
        dec_values = [ceter_dec, edge_dec2]

    path = Path(SkyCoord(ra_values, dec_values, unit="deg", frame="fk5"), width=28 * u.arcsec)

    return path

def extract_pv_cut(source_name,molecule,degree_angle=0, position='edge'):

    folder_destination = os.path.join('TP_FITS', source_name)
    full_filename_path = find_the_spectrum_for_a_source(folder_destination, molecule)

    name_of_fits = full_filename_path.split('/')[-1]

    cube, wcs = open_fits_file(folder_destination,name_of_fits)
    # Step 2: Define the spatial path for the PV cut
    # Define start and end points of the path in celestial coordinates

    # path = define_pv_path_from_coordinates(["4h17m39s", "+27d55m12s"],["4h17m34s", "+27d56m12s"])
    path = define_pv_path_from_angle(cube,degree_angle,position)

    # Step 3: Extract the PV slice
    pv_slice = extract_pv_slice(cube, path)

    initial_ww = WCS(pv_slice.header)
    ww = update_WCS_coordinates(initial_ww,name_of_fits)

    return cube, path, pv_slice, ww

def plot_extraced_pv_cut(source_name,molecule,degree_angle=0):
    '''
    plot the PV cut extracted along an angle. The PV cut is made either from
    center to both edges or from edge-to-edge
    '''
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
    '''
    plot in two panels the moment one map woth an overlay of the direction of the PV cut and width
    and also the resulting PV cut.
    '''
    moment_map_name = molecule+'_'+core

    data_cube = ALMATPData(path, moment_map_name+'_mom0.fits')
    image_mom_0 = data_cube.ppv_data
    unique_moment_0_val=np.unique(image_mom_0)
    peak = np.nanmax(image_mom_0)
    levels = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    levels = levels * peak


    data_cube = ALMATPData(path, moment_map_name+'_mom1.fits')
    image_mom_1 = data_cube.ppv_data
    unique_moment_1_val=np.unique(image_mom_1)

    cube, path, pv_slice, ww = extract_pv_cut(source_name, molecule, degree_angle=degree_angle)
    data = cube.unmasked_data[:].value

    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121, projection=cube.wcs.celestial)
    moment_one_plot = ax.imshow(image_mom_1, cmap="coolwarm", origin='lower',vmin=unique_moment_1_val[1])
    contour = ax.contour(image_mom_0, levels=levels, colors="gray",linewidths=0.7)
    plt.clabel(contour, inline=True, fontsize=8)

    path.show_on_axis(ax, spacing=5,
                          edgecolor='k', linestyle='--',
                          linewidth=1)

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

    # vel_min = 5.0*1e3  # Minimum velocity in meter/s
    # vel_max = 8.0*1e3  # Maximum velocity in meter/s

    vel_min = 4.0*1e3  # Minimum velocity in meter/s
    vel_max = 7.0*1e3  # Maximum velocity in meter/s

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

    plt.suptitle(source_name + ' ' + molecule + '   angle_deg = ' + str(degree_angle), fontsize=18)
    plt.tight_layout()
    plt.savefig("Figures/pv_diagrams/automatic_angle/"+source_name+'_'+molecule+'_angle_+'+str(degree_angle)+'.png', bbox_inches='tight',dpi=300)
    plt.show()


def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx

def peak_value_for_each_radius(source_name,molecule,degree_angle,v_lsr,vel_range=[2,11]):
    '''
    Calculate the velocity for each radius from the PV diagram. Several methods should be implemented,
    such as the 'Spine' or strongest emission. It can also be just fitted with a polynomial order for continuity
    and then extract the points. It would be important to consider the Nyquist sampling, right?

    I am missing an option to only consider part of the velocity range for clearly-separated lines.
    '''

    positions = ['center_r','center_b']
    colors=['C0','C1']
    pv_data = []
    radii = []
    velocities = []
    peak_velocities = []

    min_vel, max_vel = vel_range[0]*1e3, vel_range[1]*1e3

    for count, value in enumerate(positions):
        cube, path, pv_slice, ww = extract_pv_cut(source_name,molecule,degree_angle, position=value)
        # Find the index of the maximum intensity along the y-axis (velocity axis) for each x

        # print(cube)
        x = np.arange(pv_slice.header['NAXIS1'])
        y = np.arange(pv_slice.header['NAXIS2'])
        
        radial_array = ww.pixel_to_world_values(x, 0)[0]
        velocity_array = ww.pixel_to_world_values(0, y)[1]

        print(velocity_array)
        min_vel_index = closest(velocity_array,min_vel)
        max_vel_index = closest(velocity_array, max_vel)

        print('min max index')
        print(min_vel_index,max_vel_index)
        pv_data_image = pv_slice.data

        print('pv data shape',np.shape(pv_data_image))
        peak_indices = np.argmax(pv_data_image[min_vel_index:max_vel_index,:], axis=0) + min_vel_index

        print(' peak indices ', peak_indices)
        spatial_placeholder = 0  # Assuming single spatial point or fixed placeholder
        # Convert these indices to velocity values
        peak_velocity = ww.pixel_to_world_values(spatial_placeholder,peak_indices)[1]/1e3

        radial_array_arcmin = radial_array*60.
        velocity_array_kms= velocity_array/1e3

        radii.append(radial_array_arcmin)
        velocities.append(velocity_array_kms)
        pv_data.append(pv_data_image)
        peak_velocities.append(peak_velocity)

    plt.figure(figsize=(15, 6))
    # plt.subplot(111, projection=ww)
    ax1 = plt.subplot(131)

    pv_image = ax1.imshow(
        pv_data[0],
        origin="lower",
        aspect="auto",
        # cmap="viridis",extent=(radial_array_arcmin[0], radial_array_arcmin[-1],
        #         velocity_array_kms[0], velocity_array_kms[-1])
        cmap="viridis", extent=(radii[0][0], radii[0][-1],
                                velocities[0][0], velocities[0][-1])
    )

    ax1.scatter(radii[0], peak_velocities[0],color=colors[0])

    ax2 = plt.subplot(132)

    pv_image = ax2.imshow(
        pv_data[1],
        origin="lower",
        aspect="auto",
        # cmap="viridis",extent=(radial_array_arcmin[0], radial_array_arcmin[-1],
        #         velocity_array_kms[0], velocity_array_kms[-1])
        cmap="viridis", extent=(radii[1][0], radii[1][-1],
                                velocities[1][0], velocities[1][-1])
    )
    ax2.scatter(radii[1], peak_velocities[1],color=colors[1])


    ax1.set_ylim(4,8)
    ax2.set_ylim(4,8)

    ax1.set_xlabel("Offset [arcmin]")
    ax1.set_ylabel("Velocity [km/s]")
    ax2.set_xlabel("Offset [arcmin]")
    ax2.set_ylabel("Velocity [km/s]")

    ax3 = plt.subplot(133)

    if peak_velocities[0][0] == peak_velocities[1][0]:
        v_lsr = peak_velocities[0][0]

    # print('v_lsr', peak_velocities[0][0], v_lsr,peak_velocities[1][0])
    velocity_gradient_r = ((peak_velocities[0]-v_lsr)**2 )**0.5
    velocity_gradient_b = ((peak_velocities[1]-v_lsr)**2 )**0.5

    ax3.scatter(radii[0], velocity_gradient_r,color=colors[0])
    ax3.scatter(radii[1], velocity_gradient_b,color=colors[1])

    ax3.set_xlabel("Radial offset [arcmin]")
    ax3.set_ylabel(r" $\delta$V [km/s]")
    ax3.set_title("Peak Velocity vs Radius")
    plt.grid()

    ax3.set_ylim(-0.1,0.5)

    plt.suptitle(source_name + ' ' + molecule + '   angle_deg = ' + str(degree_angle)+ '  v_lsr = ' + str(round(v_lsr,2)), fontsize=18)

    plt.tight_layout()
    # plt.savefig("Figures/pv_diagrams/velocity_gradients/"+source_name + '_' + molecule+
    #             '_angle_+'+str(degree_angle)+'_v_lsr = ' + str(round(v_lsr,2))+'.png', bbox_inches='tight',dpi=300)
    plt.show()

def velocity_gradient():
    '''
    plot the velocity gradient by
    '''

if __name__ == "__main__":
    core = 'M236'
    molecule = 'C18O'
    angle = 67.9## measured north from east
    v_lsr=6.72
    vel_range = [5.0, 8.0] ## in km/s
    path = os.path.join('moment_maps_fits',core)
    # plot_pv_and_moment_one(core, molecule, path, degree_angle=angle)
    # plot_extraced_pv_cut(source_name=core,molecule=molecule,degree_angle=angle)
    # print(full_filename_path)

    peak_value_for_each_radius(core,molecule,angle,v_lsr,vel_range)