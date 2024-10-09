import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from make_TP_maps import ALMATPData, closest_idx, calculate_peak_SNR, find_the_spectrum_for_a_source
from make_TP_maps import find_all_spectra_for_a_molecule
import matplotlib.cm as cm
import matplotlib.colors as colors

def _sinh(x):
    return np.sinh(x)

def _arcsinh(x):
    return np.arcsinh(x)

def channel_maps(path, molecule, n_channels=[3, 3], initial_vel=0, channel_step=1, nsigma=1,
                save_folder='./', filename='default', title_image=True,
                save=False):
    """
    Create channel maps for a given source
    You can define the gridding/number of the channel maps with n_channels.
    It is possible to select the velocity range with initial vel, and final vel.
    """
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


    vel = velocity #/ 1.e5
    if vel[0]>vel[1]:
        print('I need to flip the axis')
        cube = np.flip(cube, axis=0)
        vel = velocity[::-1] #*-1 #/ 1.e5 * -1

    final_vel = initial_vel + abs(velocity_resolution)*channel_step*n_channels[0]*n_channels[1]
    print('This is the velocity range consider to calculate the peak intensity ',initial_vel,final_vel)

    peak_signal, noise = calculate_peak_SNR(path, filename, velo_limits=[initial_vel, final_vel],
                                            binning=1,separate=True)
    initial_channel, upper_idx = closest_idx(vel, initial_vel), closest_idx(vel, final_vel)

    # channel_step = #int( abs(upper_idx - initial_channel)/(n_channels[0]*n_channels[1]) )

    # if channel_step<1:
    #     channel_step=1


    nx, ny = n_channels  ##CHANGE 3x5 12CO 3x5 13CO
    xsize = 6.0 * ny
    ysize = 6.0 * nx

    ra_del = data_cube.total_size_ra#self.ImsizeRA
    dec_del = data_cube.total_size_dec#self.ImsizeDEC
    FOV_size = min(ra_del,dec_del)

    fig = plt.figure(figsize=(xsize - 0.1, ysize))  # xsize-0.1, ysize 12CO
    nblc = int((nx - 1) * ny) + 1
    maximo = peak_signal
    minimo = noise * nsigma

    levs_prev = np.linspace(minimo, maximo, 25)
    levs = [round(elem, 4) for elem in levs_prev]  ##CHANGE


    # print(ra_del,dec_del)
    
    ra = np.linspace(-ra_del / 2.0, ra_del / 2.0, cube_size_nx)
    dec = np.linspace(-dec_del / 2.0, dec_del / 2.0, cube_size_ny)
    X, Y = np.meshgrid(ra, dec)

    new_cmap = cm.gist_heat
    # new_cmap = cm.cividis
    # new_cmap = cm.viridis

    norm_new = colors.FuncNorm((_arcsinh, _sinh), vmin=minimo, vmax=maximo)


    for k in np.arange(nx * ny):

        channel_interval = initial_channel + channel_step * k  ####CHANGE #12CO 14 + 2*k ## 13CO 14 + 2*k
        mult = int(velocity_channels / (nx * ny))
        slice0 = cube[channel_interval, :, :]
        npos = 1 + k
        f = fig.add_subplot(nx, ny, npos)
        pax = f.imshow(slice0, origin='lower', extent=(ra[0] * -1, ra[-1] * -1, dec[0], dec[-1]),
                       cmap=new_cmap,norm=norm_new)
                       # , vmin=minimo, vmax=maximo)

        # star = f.plot(0, 0, marker='*', markersize=6, color='yellow')
        veltxt = '%1.2f' % (vel[channel_interval])
        f.text(-0.6 * FOV_size/2, 0.8 * FOV_size/2, veltxt, fontsize=20, color='w')
        plt.xlim(ra_del/2., -ra_del/2.)
        plt.ylim(-dec_del/2., dec_del/2.)
        f.axes.set_aspect('auto')

        # plt.savefig('pngs/' + str(k).zfill(3) + '.png', pad_inches=0)

        # plot ticks and grid at 0.5 arcsecond intervals
        # in bottom left corner panel

        if npos == nblc:
            f.xaxis.set_visible(True)
            f.yaxis.set_visible(True)
            f.set_xlabel(r'$\Delta \alpha$ (")', fontsize=22)
            f.set_ylabel(r'$\Delta \delta$ (")', fontsize=22)
            f.tick_params(axis='both', which='both', labelsize=20)

        else:
            f.xaxis.set_visible(False)
            f.yaxis.set_visible(False)

    ## vertical position of the cbar

    ## horizontal position of the cbar

    cbar_position = fig.add_axes([0.12, 0.91, 0.40, 0.03])
    cbar = fig.colorbar(pax, cax=cbar_position, orientation='horizontal', format='%.0f')

    plt.title(r'$\rm I_{\nu}$ (Jy beam$^{-1}$)', fontsize=26)
    cbar.ax.tick_params(labelsize=20)

    plt.subplots_adjust(wspace=0.011, hspace=0.011)
    source_name = data_cube.source_name

    if title_image:
        fig.suptitle(str(source_name), fontsize=30)

    if save:
        molecule_name = data_cube.molec_name
        save_folder = 'Figures/channel_maps/'
        fig.savefig(os.path.join(*[save_folder,molecule_name,source_name])+'.pdf', bbox_inches='tight',dpi=300)
        # plt.savefig(figname, bbox_inches='tight')

    else:
        plt.show()


def mass_produce_channel_maps(folders_path,molecule='12CO',n_channels=[4, 5], initial_vel=4.5, channel_step=1, nsigma=5.):
    '''
    Apply the channel_maps() function
    to all the files in a folder, for  agiven molcule.
    '''
    array_of_paths = find_all_spectra_for_a_molecule(folders_path, molecule)

    for source_path in array_of_paths:
        try:
            path = os.path.join(folders_path, source_path.split('/')[0])
            print('we are currently producing a channel map for '+path)
            channel_maps(path, molecule,
                         n_channels, initial_vel, channel_step, nsigma, save=True)

        except IndexError as err:
            print('map '+source_path.split('/')[0]+' was not produced. Check the moment maps.')

if __name__ == "__main__":

    save_folder = 'channel_maps'

    ###### initial vel for C18O 4.5 to 8.0
    ###### for N2D+ -> 5.5 to 7.5
    
    ### Creation of a single channel map
    # source ='M426'
    # folder_destination = os.path.join('TP_FITS', source)
    # channel_maps(path = folder_destination, molecule = 'SO' ,
    #                      n_channels=[4, 5], initial_vel=5.0, channel_step=1, nsigma=3., save=True)

    ### Creation of a channel maps for all sources for a given molecule
    ### For C18O I use n_channels=[4, 5], initial_vel=4.5, channel_step=2, nsigma=5.
    ### For N2D+ n_channels=[3, 5], initial_vel=6.2, channel_step=1, nsigma=3.
    ### For SO n_channels=[4, 5], initial_vel=5.0, channel_step=2, nsigma=3.

    folder_fits = 'TP_FITS'
    mass_produce_channel_maps(folder_fits, molecule='C18O',n_channels=[4, 5], initial_vel=4.5, channel_step=2, nsigma=5.)