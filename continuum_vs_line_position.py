import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle


def moment_maps_with_continuum(path, filename, skycoord_object, save_name, save=True ):
    '''
    Plot moment zero, one, and two maps in a grid
    '''

    aperture_radius = 14.3 ## A 12m antenna at 219.56 GHz or 1.365 mm produces an angular resolution of 28.6 arcsec

    data_cube = ALMATPData(path, filename+'_mom0.fits')
    image_mom_0 = data_cube.ppv_data

    data_cube = ALMATPData(path, filename+'_mom1.fits')
    image_mom_1 = data_cube.ppv_data


    fig = plt.figure(figsize=(12,6))

    peak = np.nanmax(image_mom_0)
    levels = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    levels = levels * peak

    ### check if BTS produced maps or SNR too low.

    ## Moment zero
    unique_moment_0_val=np.unique(image_mom_0)
    fig1 = fig.add_subplot(121,projection=data_cube.wcs)
    mom0_im = fig1.imshow(image_mom_0, cmap="viridis", origin='lower',vmin=unique_moment_0_val[1])
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mom0_im, fraction=0.048, pad=0.04, label='Integrated Intensity (Jy/beam * km/s)')
    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)

    unique_moment_1_val=np.unique(image_mom_1)
    fig2 = fig.add_subplot(122,projection=data_cube.wcs)
    mom1_im = fig2.imshow(image_mom_1, cmap="coolwarm", origin='lower',vmin=unique_moment_1_val[1])
    cbar = plt.colorbar(mom1_im, fraction=0.048, pad=0.04, label='Velocity (km/s)')
    contour = fig2.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)


    # fig.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.1)
    fig1.set_title('moment 0')
    fig2.set_title('moment 1')

    ## create the position patch of the Herschel data
    s = SphericalCircle(skycoord_object, aperture_radius * u.arcsec,
                        edgecolor='white', facecolor='none',
                        transform=fig1.get_transform('fk5'))
    fig1.add_patch(s)

    plt.suptitle(filename , fontsize=18)
    

    if save:
        fig.savefig(os.path.join('Figures/Herschel_position',save_name), bbox_inches='tight')
    plt.show()

def mass_produce_moment_maps_with_continuum(coordinate_file, folder_fits,molecule='C18O'):
    '''
    Apply the plot_moment_maps(path, filename) function
    to all the files in a folder, for  agiven molcule.
    '''
    source_list,ra_list,dec_list = read_source_positions(coordinate_file)

    folder_list = sorted(next(os.walk(folder_fits))[1])

    for ii, sources in enumerate(source_list):
        folder_index = search_string_in_list(sources, folder_list)
        full_folder_dir= os.path.join(folder_fits,folder_list[folder_index[0]])
        try:
            skycoord_object = SkyCoord( str(ra_list[ii])+' '+str(dec_list[ii]), unit=(u.hourangle, u.deg))
            moment_maps_with_continuum(path=full_folder_dir, filename= molecule+'_'+folder_list[folder_index[0]],
                                       skycoord_object = skycoord_object, save_name = molecule+'_'+ sources)
        except IndexError as err:
            print('map '+sources+' was not produced. Check the moment maps.')


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

def search_string_in_list(search_term, string_list):
    """
    Searches for a string in a list of strings and returns the indices where it is found.

    Parameters:
    search_term (str): The string to search for.
    string_list (list of str): The list of strings to search within.

    Returns:
    list of int: A list of indices where the search_term is found.
    """
    indices = [i for i, s in enumerate(string_list) if search_term in s]
    return indices


if __name__ == "__main__":
    # source ='M262'
    # moment_maps_with_continuum(path='moment_maps_fits/'+source+'/', filename='C18O_'+source,save=True)
    # skycoord_object = SkyCoord('04:17:32 +27:41:35', unit=(u.hourangle, u.deg))

    mass_produce_moment_maps_with_continuum(coordinate_file='Herschel_coordinates.txt',
                                            folder_fits='moment_maps_fits', molecule='C18O')