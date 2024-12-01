import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData, make_average_spectrum_data, find_the_spectrum_for_a_source, find_all_spectra_for_a_molecule
from make_TP_maps import closest_idx
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs.utils import pixel_to_skycoord


def mass_produce_spectral_comparison(large_map, coordinate_file,folders_path,molecule_Nobeyama,molecule_ALMA):
    '''
    We mass produce a comparison between Nobeyama and ALMA data
    by calling compare_nobeyama_and_ALMA many times.
    '''
    array_of_paths = find_all_spectra_for_a_molecule(folders_path, molecule_ALMA)

    for source_path in array_of_paths:
        print(source_path.split('/')[0])
        try:
            compare_nobeyama_and_ALMA(large_map, coordinate_file, source_name = source_path.split('/')[0],
                                      molecule=molecule_Nobeyama,save=True)
        except IndexError as err:
            print('comparison for '+source_path.split('/')[0]+' was not produced.')

def compare_nobeyama_and_ALMA(large_map,coordinate_file,source_name,radius_in_arcsec=60.0,molecule='',save=False):
    '''
    We compare the spectra taken from the Nobeyama observatory with the spectra of ALMA data
    '''
    spectrum_big, velocity_big = create_spectrum_from_position(large_map,
                                  coordinate_file,
                                  source_name=source_name)

    full_filename_path = find_the_spectrum_for_a_source(folders_path=os.path.join('TP_FITS',source_name), spw_or_molec='C18O')
    filename = full_filename_path.split('/')
    
    spectrum_small, velocity_small = make_average_spectrum_data(path=os.path.join('TP_FITS',source_name), filename=filename[-1])

    colors=['k','r']
    alpha=[1,0.85]

    plt.plot(velocity_small, spectrum_small / np.nanmax(spectrum_small), "-", color=colors[0], lw=2, label='C18O ALMA')
    plt.plot(velocity_big, spectrum_big/ np.nanmax(spectrum_big), "-", color=colors[1], lw=2, label=molecule+' Nobeyama')
    plt.xlabel("velocity [km/s]", size=15)
    plt.ylabel("Intensity", size=15)
    plt.tick_params(axis='both', direction='in')
    plt.xlim(3, 9)

    plt.legend()
    plt.title(str(source_name) , fontsize=18)

    if save:
        save_fig_name = 'comparison_ALMA_vs_Nobeyama_'+molecule+'_source_'+source_name+'.png'
        plt.savefig(os.path.join('Figures/Nobeyama/', save_fig_name), bbox_inches='tight',dpi=300)

    plt.show()


def create_spectrum_from_position(large_map,coordinate_file,source_name,radius_in_arcsec=60.0,plot=False):
    """
    Average spectrum of the central beam.

    """
    source_list,ra_list,dec_list = read_source_positions(coordinate_file)

    for ii, name in enumerate(source_list):
        if name in source_name:
            skycoord_object = SkyCoord(str(ra_list[ii]) + ' ' + str(dec_list[ii]), unit=(u.hourangle, u.deg))
            print(source_name)

    data_cube = ALMATPData(path='do-not-save/Nobeyama_data/', filename=large_map)
    velocity = data_cube.vel/1e3 ## from m/s to km/s

    pixel_scale_ra = data_cube.wcs.wcs.cdelt[0] * 3600  # arcseconds per pixel
    pixel_scale_dec = data_cube.wcs.wcs.cdelt[1] * 3600  # arcseconds per pixel
    aperture_radius = abs(radius_in_arcsec/pixel_scale_ra)

    try:
        x_center, y_center = data_cube.wcs.celestial.world_to_pixel(skycoord_object) ## This one if 3D cube
    except:
        raise UnboundLocalError('can not get coordinates for '+str(source_name)+" Check your source name or the list of coordinates")

        # Initialize a list to store the pixel values within the aperture
    center_beam_values = []
    # Iterate over a square region, but filter by distance to make it circular
    for xx in range(int(x_center - aperture_radius), int(x_center + aperture_radius) + 1):
        for yy in range(int(y_center - aperture_radius), int(y_center + aperture_radius) + 1):
            # Calculate the distance from the center
            distance = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

            # Check if the distance is within the aperture radius
            if distance <= aperture_radius:
                # Append the data at this pixel position
                center_beam_values.append(data_cube.ppv_data[:, yy, xx ])

    # Convert center_beam_values to a NumPy array for easy manipulation
    center_beam_values = np.array(center_beam_values)
    average_spectrum = np.nanmean(center_beam_values, axis=0)

    if plot:
        plt.plot(velocity,average_spectrum)
        plt.show()
    return average_spectrum, velocity

def show_spatial_location_of_spectra(large_map, coordinate_file,source_name,radius_in_arcsec=360.0, molecule='C18O'):
    '''
    plot moment zero pf Nobeyama data and mark the position of the sources
    from the coordinate file
    '''

    #### Open the data using standard codes
    data_cube = ALMATPData(path='do-not-save/Nobeyama_data/', filename=large_map)
    image = data_cube.ppv_data
    velocity = data_cube.vel/1e3 ## from m/s to km/s
    velocity_resolution = data_cube.velocity_resolution/1e3  ## from m/s to km/s

    #### Compute moment zero map
    lower_idx, upper_idx = closest_idx(velocity, 4.8), closest_idx(velocity, 7.0)
    moment_0_map = np.max(image[lower_idx:upper_idx,:,:], axis=0)
    max_map = np.nanmax(moment_0_map)*0.3
    min_map = np.nanmin(moment_0_map)

    #### Plot the map
    fig = plt.figure(figsize=(6,6))
    fig1 = fig.add_subplot(111, projection=data_cube.wcs, slices=('x','y',150))
    unique_moment_0_val = np.unique(moment_0_map)
    mom0_im = fig1.imshow(moment_0_map, cmap="viridis", origin='lower', vmin=min_map,vmax=max_map)

    ### Change the limits
    pixel_limits_ra = 250
    pixel_limits_dec =150
    fig1.set_xlim(-0.5+pixel_limits_ra, data_cube.ny - 0.5 -pixel_limits_ra)
    fig1.set_ylim(-0.5+pixel_limits_dec, data_cube.nx - 0.5 -pixel_limits_dec)

    ### Add labels and display
    fig1.set_xlabel('Right Ascension')
    fig1.set_ylabel('Declination')

    ### Overplot

    source_list,ra_list,dec_list = read_source_positions(coordinate_file)

    pixel_scale_ra = data_cube.wcs.wcs.cdelt[0] * 3600  # arcseconds per pixel
    pixel_scale_dec = data_cube.wcs.wcs.cdelt[1] * 3600  # arcseconds per pixel
    aperture_radius = abs(radius_in_arcsec / pixel_scale_ra)
    plt.title(str(molecule) , fontsize=18)

    for ii, name in enumerate(source_list):
        # if name in source_name:
        skycoord_object = SkyCoord(str(ra_list[ii]) + ' ' + str(dec_list[ii]), unit=(u.hourangle, u.deg))
        print(name)

        s = SphericalCircle(skycoord_object, aperture_radius * u.arcsec,
                            edgecolor='red', facecolor='none',
                            transform=fig1.get_transform('fk5'), linewidth=2, linestyle='-')
        fig1.add_patch(s)
    plt.show()

    filename='Nobeyama_HCN_source_locations.png'
    fig.savefig(os.path.join('Figures/Nobeyama/',filename),dpi=300, bbox_inches='tight')

def read_source_positions(text_file):
    '''
    We take the rest frequency from a file not from the header
    '''
    source_list=[]
    ra_list=[]
    dec_list=[]
    with open(text_file) as f:
        file = f.readlines()
        for lines in file:
            if lines[0]!='#':
                source_list.append(lines.split()[0])
                ra_list.append(lines.split()[1].strip(','))
                dec_list.append(lines.split()[2])

    return source_list,ra_list,dec_list

if __name__ == "__main__":

    # source ='M262'
    # molecule='HCOp'
    # compare_nobeyama_and_ALMA(large_map=molecule+'_2022-2024_01kms_spheroidal_xyb_base.fits',
    #                                coordinate_file='Herschel_coordinates_from_ALMA.txt',
    #                                source_name=source, molecule=molecule,save=True)
    folder_fits = 'TP_FITS'
    molecule_Nobeyama='H13COp'
    mass_produce_spectral_comparison(large_map=molecule_Nobeyama+'_2022-2024_01kms_spheroidal_xyb_base.fits',
                                   coordinate_file='Herschel_coordinates_from_ALMA.txt',
                                     folders_path=folder_fits, molecule_Nobeyama=molecule_Nobeyama
                                     ,molecule_ALMA='C18O')

    # show_spatial_location_of_spectra(large_map=molecule_Nobeyama+'_2022-2024_01kms_spheroidal_xyb_base.fits',
    #                                coordinate_file='Herschel_coordinates_from_ALMA.txt',source_name='M262',
    #                                  molecule=molecule_Nobeyama)
