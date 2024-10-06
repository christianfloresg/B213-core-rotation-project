import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData, closest_idx, calculate_peak_SNR
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from mpl_toolkits.axes_grid1 import ImageGrid
from make_TP_maps import find_all_spectra_for_a_molecule
from make_TP_maps import average_over_n_first_axis, find_the_spectrum_for_a_source
import matplotlib.cm as cm


def channel_maps(path, molecule, n_channels=[3, 3], FOV_size=6, initial_vel=0, final_vel=10, nsigma=1,
                 cube=[], save_folder='./', filename='default', title_image='',
                save=False):
    '''
    rms_level is in Jy/beam
    '''

    if molecule!=None:
        full_filename_path = find_the_spectrum_for_a_source(path, molecule)
        filename = full_filename_path.split('/')[-1]

    data_cube = ALMATPData(path, filename)
    velocity = data_cube.vel
    cube = data_cube.ppv_data
    velocity_resolution = data_cube.velocity_resolution
    velocity_channels = data_cube.nz
    cube_size_nx = data_cube.nx
    cube_size_ny = data_cube.ny

    peak_signal, noise = calculate_peak_SNR(path, filename, velo_limits=[initial_vel, final_vel],
                                            binning=1,separate=True)

    initial_channel, upper_idx = closest_idx(velocity, initial_vel), closest_idx(velocity, final_vel)
    channel_step = int( abs(upper_idx - initial_channel)/(n_channels[0]*n_channels[1]) )

    invert_wavelength_cube = False

    vel = velocity #/ 1.e5

    if invert_wavelength_cube:
        cube = np.flip(cube, axis=0)
        vel = velocity[::-1] *-1 #/ 1.e5 * -1

    nx, ny = n_channels  ##CHANGE 3x5 12CO 3x5 13CO
    xsize = 6.0 * ny
    ysize = 6.0 * nx
    subplot_size = FOV_size  ##from, -2.5 tp 2.5 arsec in x and y direction
    fig = plt.figure(figsize=(xsize - 0.1, ysize))  # xsize-0.1, ysize 12CO
    nblc = int((nx - 1) * ny) + 1
    maximo = peak_signal
    minimo = noise * nsigma

    levs_prev = np.linspace(minimo, maximo, 25)
    levs = [round(elem, 4) for elem in levs_prev]  ##CHANGE

    ra_del = data_cube.total_size_ra#self.ImsizeRA
    dec_del = data_cube.total_size_dec#self.ImsizeDEC


    # print(ra_del,dec_del)
    
    ra = np.linspace(-ra_del / 2.0, ra_del / 2.0, cube_size_nx)
    dec = np.linspace(-dec_del / 2.0, dec_del / 2.0, cube_size_ny)
    X, Y = np.meshgrid(ra, dec)

    new_cmap = cm.gist_heat
    # new_cmap = truncate_colormap(my_cmap, 0.0, 1.0)

    for k in np.arange(nx * ny):

        channel_interval = initial_channel + channel_step * k  ####CHANGE #12CO 14 + 2*k ## 13CO 14 + 2*k
        mult = int(velocity_channels / (nx * ny))
        slice0 = cube[channel_interval, :, :]
        npos = 1 + k
        f = fig.add_subplot(nx, ny, npos)
        pax = f.imshow(slice0, origin='lower', extent=(ra[0] * -1, ra[-1] * -1, dec[0], dec[-1]),
                       cmap=new_cmap, vmin=minimo,
                       vmax=maximo)

        # star = f.plot(0, 0, marker='*', markersize=6, color='yellow')
        veltxt = '%1.2f' % (vel[channel_interval])
        f.text(-0.6 * FOV_size, 0.8 * FOV_size, veltxt, fontsize=20, color='w')
        plt.xlim(subplot_size, -subplot_size)
        plt.ylim(-subplot_size, subplot_size)
        f.axes.set_aspect('equal')
        # plt.savefig('pngs/' + str(k).zfill(3) + '.png', pad_inches=0)

        # plot ticks and grid at 0.5 arcsecond intervals
        # in bottom left corner panel

        if npos == nblc:
            f.xaxis.set_visible(True)
            f.yaxis.set_visible(True)
            f.set_xlabel(r'$\Delta \alpha$ (")', fontsize=22)
            f.set_ylabel(r'$\Delta \delta$ (")', fontsize=22)
            f.tick_params(axis='both', which='both', labelsize=20)

            # try:
            #     elips = Ellipse(
            #         xy=(-2.2, -1.9),
            #         width=self.bmaj, height=self.bmin, angle=self.bpa + 90,
            #         facecolor='grey', edgecolor='white')
            # except:

            # elips = Ellipse(
            #     xy=(0.8 * FOV_size, -0.8 * FOV_size),
            #     width=self.bmin, height=self.bmaj, angle=self.bpa - 90,
            #     facecolor='grey', edgecolor='white')
            # f.add_artist(elips)

        else:
            f.xaxis.set_visible(False)
            f.yaxis.set_visible(False)

    ## vertical position of the cbar

    ## horizontal position of the cbar


    cbar_position = fig.add_axes([0.13, 0.93, 0.30, 0.02])
    cbar = fig.colorbar(pax, cax=cbar_position, orientation='horizontal', format='%.0f')

    plt.title(r'$\rm I_{\nu}$ (Jy beam$^{-1}$)', fontsize=22)
    cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(wspace=0.011, hspace=0.011)
    fig.suptitle(str(title_image), fontsize=20)

    if save:
        figname = os.path.join(save_folder, filename + '.pdf')
        plt.savefig(figname, bbox_inches='tight')

    else:
        plt.show()


def mass_produce_spectral_maps(folders_path,molecule='12CO',binning=1):
    '''
    Apply the plot_moment_maps(path, filename) function
    to all the files in a folder, for  agiven molcule.
    '''
    array_of_paths = find_all_spectra_for_a_molecule(folders_path, molecule)

    for source_path in array_of_paths:
        try:
            create_spectral_maps(path=os.path.join(folders_path,source_path.split('/')[0]),
                             filename=source_path.split('/')[1],
                             save=True,show=False,binning=binning)
        except IndexError as err:
            print('map '+source_path.split('/')[0]+' was not produced. Check the moment maps.')

if __name__ == "__main__":

    save_folder = '.'

    ### Creation of a single spectral map
    source ='M449'
    folder_destination = os.path.join('TP_FITS', source)
    channel_maps(path = 'TP_FITS/M449', molecule = 'C18O' ,
                         n_channels=[5, 4], FOV_size=45, initial_vel=4.5, final_vel=8, nsigma=5., save=True)


    ### Creation of a maps for all sources for a given molecule
    # folder_fits = 'TP_FITS'
    # mass_produce_spectral_maps(folder_fits, molecule='C18O',binning=1)