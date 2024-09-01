import os
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import sys
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Gaussian1DKernel
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Astropy modules to deal with coordinates
from astropy.wcs import WCS
from astropy.wcs import Wcsprm
from astropy.io import fits
from astropy.wcs import utils

#BTS code developed by Seamus to get nice moment maps using cube masking
#instead of sigma clippping

module_path = os.path.abspath(os.path.join('/Users/christianflores/Documents/Work/BTS-master/')) # or the path to your source code
# module_path = os.path.abspath(os.path.join('/Users/christianflores/Documents/Work/external_codes/BTS-master'))
sys.path.insert(0, module_path)
import BTS
sys.path.insert(0, module_path)


class ALMATPData:
    def __init__(self, path, filename):
        self.image = 1

        #         try:
        data_cube = fits.open(os.path.join(path, filename))
        #         except:
        #              data_cube = fits.open(os.path.join(path, filename + '.fits'))

        self.filename = filename
        self.header = data_cube[0].header
        self.ppv_data = data_cube[0].data


        # If the data has a 4 dimension, turn it into 3D
        if (np.shape(data_cube[0].data)[0] == 1):
            self.ppv_data = data_cube[0].data[0, :, :, :]

        self.nx = self.header['NAXIS1']
        self.ny = self.header['NAXIS2']


        try:
            lower_idx_spw_name = self.filename.find('spw')
            self.spw_name = self.filename[lower_idx_spw_name:lower_idx_spw_name+5]
            print(self.spw_name)
        except:
            print('There is no "spw" substring in this data file, I cannot guess the molecule')

        try:
            self.nz = self.header['NAXIS3']
            self.vel = self.get_vel(self.header)
            dv = self.vel[1] - self.vel[0]
            print('velocity resolution', dv)
            if (dv < 0):
                dv = dv * -1
            self.velocity_resolution = dv

        except:
            print('This is a 2D image')

        self.wcs = WCS(self.header)

    def get_vel(self, head):

        ### If the header data is stored as frequency then convert to velocity [in km/s]
        if "f" in head['CTYPE3'].lower():

            df = head['CDELT3']
            nf = head['CRPIX3']
            fr = head['CRVAL3']

            ff = np.zeros(head["NAXIS3"])
            for ii in range(0, len(ff)):
                ff[ii] = fr + (ii - nf + 1) * df

            rest = self.accurate_reference_frequency()  # head["RESTFRQ"]

            vel = (rest - ff) / rest * 299792.458
            return vel

        elif "v" in head['CTYPE3'].lower():

            refnv = head["CRPIX3"]
            refv = head["CRVAL3"]
            dv = head["CDELT3"]
            ### Construct the velocity axis

            vel = np.zeros(head["NAXIS3"])
            for ii in range(0, len(vel)):
                vel[ii] = refv + (ii - refnv + 1) * dv

            return vel

        else:

            print("The CTYPE3 variable in the fitsfile header does not start with F for frequency or V for velocity")
            return

    def accurate_reference_frequency(self):
        '''
        We take the rest frequency from a file not from the header
        '''
        with open('molecule_rest_freq.txt') as f:
            file = f.readlines()
            for lines in file:
                if self.spw_name in lines:
                    freq_rest =lines.split()[1]
                    self.molec_name = lines.split()[2]
        return float(freq_rest)

def molecule_to_spectral_window(molecule):
    '''
    We take the rest frequency from a file not from the header
    '''
    with open('molecule_rest_freq.txt') as f:
        file = f.readlines()
        for lines in file:
            if molecule in lines:
                freq_rest =lines.split()[1]
                spw_name = lines.split()[0]
    return str(spw_name), float(freq_rest)

def closest_idx(lst, val):
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()
    return idx

def get_files_in_directory(directory_path):
    '''
    This function returns a list of the
    files and directories in directory_path
    '''
    try:
        #get a list of files and directory
        file_list = os.listdir(directory_path)

        # Get only file names in directory
        file_list = [file for file in file_list if os.path.isfile(os.path.join(directory_path, file))]

        return file_list
    except OSError as e:
        print(f"Error: {e}")
        return []


def mass_produce_moment_maps(folder_fits,molecule='12CO'):
    '''
    Apply the plot_moment_maps(path, filename) function
    to all the files in a folder, for  agiven molcule.
    '''
    folder_list = sorted(next(os.walk(folder_fits))[1])
    print(folder_list)
    for sources in folder_list:
        full_folder_dir= os.path.join(folder_fits,sources)
        # print(full_folder_dir)
        try:
            plot_moment_maps(path=full_folder_dir, filename= molecule+'_'+sources)
        except IndexError as err:
            print('map '+sources+' was not produced. Check the moment maps.')
            
def plot_moment_maps(path, filename):
    '''
    Plot moment zero, one, and two maps in a grid
    '''
    data_cube = ALMATPData(path, filename+'_mom0.fits')
    image_mom_0 = data_cube.ppv_data

    data_cube = ALMATPData(path, filename+'_mom1.fits')
    image_mom_1 = data_cube.ppv_data

    data_cube = ALMATPData(path, filename+'_mom2.fits')
    image_mom_2 = data_cube.ppv_data


    fig = plt.figure(figsize=(12,12))

    peak = np.nanmax(image_mom_0)
    levels = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    levels = levels * peak

    ### check if BTS produced maps or SNR too low.

    ## Moment zero
    unique_moment_0_val=np.unique(image_mom_0)
    fig1 = fig.add_subplot(221,projection=data_cube.wcs)
    mom0_im = fig1.imshow(image_mom_0, cmap="viridis", origin='lower',vmin=unique_moment_0_val[1])
    # divider = make_axes_locatable(fig1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mom0_im, fraction=0.048, pad=0.04, label='Integrated Intensity (Jy/beam * km/s)')
    contour = fig1.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)

    ## Moment one
    unique_moment_1_val=np.unique(image_mom_1)
    fig2 = fig.add_subplot(222,projection=data_cube.wcs)
    mom1_im = fig2.imshow(image_mom_1, cmap="coolwarm", origin='lower',vmin=unique_moment_1_val[1])
    cbar = plt.colorbar(mom1_im,fraction=0.048, pad=0.04, label='Velocity (km/s)')

    ##moment two
    unique_moment_2_val=np.unique(image_mom_2)
    fig3 = fig.add_subplot(223,projection=data_cube.wcs)
    mom2_im = fig3.imshow(image_mom_2, cmap="seismic", origin='lower',vmin=unique_moment_2_val[1])
    cbar = plt.colorbar(mom2_im,fraction=0.048, pad=0.04, label='Velocity width (km/s)')

    fig4 = fig.add_subplot(224,projection=data_cube.wcs)
    mom1_im = fig4.imshow(image_mom_1, cmap="coolwarm", origin='lower',vmin=unique_moment_1_val[1])
    cbar = plt.colorbar(mom1_im, fraction=0.048, pad=0.04, label='Velocity (km/s)')
    contour = fig4.contour(image_mom_0, levels=levels, colors="black")
    plt.clabel(contour, inline=True, fontsize=8)


    # contour = fig2.contour(image, levels=levels, colors="black")


    # fig.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.1)
    fig1.set_title('moment 0')
    fig2.set_title('moment 1')
    fig3.set_title('moment 2')

    plt.suptitle(filename , fontsize=18)

    fig.savefig(os.path.join('Figures',filename), bbox_inches='tight')
    plt.show()
    
def make_average_spectrum_data(path, filename):
    """
    Average spectrum of the whole cube.
    """
    count = 0
    data_cube = ALMATPData(path, filename)
    velocity = data_cube.vel
    image = data_cube.ppv_data
    average_spectrum = np.nanmedian(image, axis=(1, 2))

    return average_spectrum, velocity

def plot_average_spectrum(path,filename):
    """
    This one plots the average spectrum
    """
    spectrum, velocity = make_average_spectrum_data(path,filename)
    plt.figure()
    # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)
    plt.xlabel("velocity [km/s]")
    plt.ylabel("Intensity")
    # Set the value for horizontal line
    y_horizontal_line = 0
    plt.axhline(y_horizontal_line, color='red', linestyle='-')
#     plt.axvline(Vsys, color='red', linestyle='--')
    plt.plot(velocity,spectrum,"-",color="black",lw=1)
    plt.tick_params(axis='both', direction='in')
    plt.xlim(-20,20)
    plt.show()


def calculate_peak_SNR(path, filename, velo_limits=[-20, 20]):
    '''
    Calculates the peak SNR over the whole cube.
    It is possible to set velocity limits for the calculation
    of noise in line-free regions.
    The noise is calculated as the std of images in line-free channels,
    averaged over many channels.
    '''

    data_cube = ALMATPData(path, filename)
    image = data_cube.ppv_data
    velocity = data_cube.vel
    peak_signal_in_cube = np.nanmax(image)

    val_down, val_up = velo_limits[0], velo_limits[1]
    lower_idx, upper_idx = closest_idx(velocity, val_down), closest_idx(velocity, val_up)

    print(lower_idx, upper_idx)
    array_of_noise_lower = np.nanstd(image[:lower_idx, :, :], axis=0)
    array_of_noise_upper = np.nanstd(image[upper_idx:, :, :], axis=0)

    average_noise_images = (np.nanmean(array_of_noise_lower) + np.nanmean(array_of_noise_upper)) / 2.

    return round(peak_signal_in_cube / average_noise_images, 1)

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y


def gaussian_parameters_of_spectra(velocity, spectrum, guess=[], plot=False):
    '''
    guess is a 3-tuple of centroid, amplitude, and width
    Need to improve for non-overalaping centers.
    Basically, each component is restricted in a portion
    of the larger velocity array
    '''
    if guess == []:
        guess = [7, 5, 2]

    rms = np.nanstd(spectrum[10:40])
    #     gaussian_kernel = Gaussian1DKernel(3)
    #     smoothed_spectrum = convolve(spectrum,gaussian_kernel)

    n_bound = int(len(guess) / 3.)
    bounds = ((2, rms * 8, 1e-3) * n_bound, (10, 100, 10) * n_bound)

    popt, pcov = curve_fit(func, velocity, spectrum, p0=guess, bounds=bounds)
    average_centroid = np.nanmedian(popt[::3])
    fit = func(velocity, *popt)

    if plot:
        #         plt.plot(velocity,smoothed_spectrum)
        plt.plot(velocity, spectrum)
        plt.plot(velocity, fit, 'r-')
        plt.xlim(average_centroid - 5, average_centroid + 5)
        #         plt.xlim(0,10)
        plt.show()

    return popt

def find_all_spectra_for_a_molecule(folders_path, spw_or_molec='.spw27.'):
    '''
    Find the path to all the spectra of a given molecule
    you can use the spectral windown number or the molecule name - dou ke yi
    spw_or_molec e.g., = 'spw27' or '.spw27.' or 'C18O', etc.
    '''

    try:
        spectral_window_name = molecule_to_spectral_window(spw_or_molec)[0] ## first index is spw name
    except:
        print('please check that your spectral window coincides with molcules in "molecule_rest_freq" file')

    array_of_paths=[]
    molecule = spectral_window_name
    folder_list = sorted(next(os.walk(folders_path))[1])
    for each_folder in folder_list:
        filenames = get_files_in_directory(os.path.join(folders_path,each_folder))
        for names in filenames:
            if molecule in names:
                array_of_paths.append(os.path.join(each_folder,names))
    return array_of_paths


def plot_grid_of_spectra(folders_path,spw_numbers=['.spw27.','.spw21.'],normalized=False):

    number_of_sources = len( sorted(next(os.walk(folders_path))[1]))
    grid_size = int(math.ceil(number_of_sources ** 0.5))
    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of Axes
                     axes_pad=0.3, aspect=False  # pad between Axes in inch.
                     )

    colors=['k','C1']
    alpha=[1,0.85]
    line_width=[1.0,1.2]
    counter=0
    for molecules in spw_numbers:
        array_of_paths = find_all_spectra_for_a_molecule(folders_path,molecules)

        for ax, sources in zip(grid, array_of_paths):
            #     for sources in array_of_paths:
            spectrum, velocity = make_average_spectrum_data(path='TP_FITS',
                                                            filename=sources)
            SNR = calculate_peak_SNR(path='TP_FITS', filename=sources)
            if normalized:
                plot = ax.plot(velocity,spectrum/np.nanmax(spectrum),color=colors[counter],alpha=alpha[counter],
                               lw=line_width[counter])
                ax.set_ylim(-0.4,1.3)
            else:
                plot = ax.plot(velocity, spectrum)
                
            ax.text(x=0.05, y=0.95-counter*0.1, s='SNR = ' + str(int(SNR)), ha='left', va='top',
                    transform=ax.transAxes, size=12, color=plot[0].get_color())#'purple')
            print(abs(velocity[10] - velocity[11]))
            # ax.set_xlim(-6, 18)
            ax.set_xlim(0, 12)

            ax.set_title(sources.split('/')[0])
        counter=counter+1

    plt.suptitle(' vs '.join(spw_numbers) , fontsize=18)

    save_fig_name = 'Grid_of_spectra_' + '_vs_'.join(spw_numbers) +'.png'
    fig.savefig(os.path.join('Figures',save_fig_name), bbox_inches='tight')

    plt.show()

def create_moment_masking_parameterfile(source,destination,fits_file_name):
    '''
    Copy the parameter file needed to run BTS and create moment maps
    Modify the files themselves so they have the appropriate input data
    '''

    cube = ALMATPData(destination,fits_file_name)
    molecule = cube.molec_name

    ### copying the file
    moment_param_filename = destination.split('/')[-1] + '_'+molecule+'_moments.param' ## Name of the cube.param file
    full_path_moment_param_filename = os.path.join(destination,moment_param_filename) ## full name including path
    copy_text_files(source, full_path_moment_param_filename)

    ### modifying the file
    new_fits_path = os.path.join(destination,fits_file_name)
    replace_line(full_path_moment_param_filename, 'data_in_file_name', new_fits_path)
    save_folder = os.path.join('moment_maps_fits',destination.split('/')[-1])
    output_base = os.path.join(save_folder,molecule+'_'+destination.split('/')[-1])
    replace_line(full_path_moment_param_filename, 'output_base', output_base)


    ### make directory to save file if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    return full_path_moment_param_filename

def copy_text_files(source,destination):
    '''
    # Copy the content of the moment masking parameter to the folder of the fits files
    # source to destination is the folder
    '''

    try:
        shutil.copyfile(source, destination)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except:
        print("Error occurred while copying file.")


def replace_line(file_name, key_text, new_text):
    lines = open(file_name, 'r').readlines()
    for count, line in enumerate(lines):
    # for line in lines:
        if key_text in line:
            text_to_change = line.split()[2]
            replaced_line = line.replace(text_to_change, new_text)
            line_num = count
            # print(text_to_change)
    lines[line_num] = replaced_line
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def compute_moment_maps_for_one_molecule(folders_path='TP_FITS',spw_number='.spw27.'):
    '''
    Compute moment maps for all the folders in TP FITS for a given molecule
    The issue is that specific parameters tunning cannot be generated.
    '''
    array_of_paths = find_all_spectra_for_a_molecule(folders_path,spw_number)

    for sources in array_of_paths:

        core = sources.split('/')[0]
        folder_destination = os.path.join('TP_FITS',core)
        name_of_fits = sources.split('/')[1]

        print(folder_destination,name_of_fits)
        filename = create_moment_masking_parameterfile(source='Fit_cube_example.param', destination=folder_destination,
                                                       fits_file_name=name_of_fits)
        param = BTS.read_parameters(filename)
        # # # Run the function to make the moments using the moment-masking technique
        BTS.make_moments(param)

if __name__ == "__main__":

    # plot_grid_of_spectra(folders_path='TP_FITS', spw_numbers=['C18O','N2D+'],normalized=True)

    # compute_moment_maps_for_one_molecule(folders_path='TP_FITS',spw_number='C18O')
    # mass_produce_moment_maps(folder_fits='moment_maps_fits', molecule='C18O')


    ####Run single sources
    core = 'M295_296_280_281'
    folder_destination = os.path.join('TP_FITS',core)
    name_of_fits = 'member.uid___A001_X15aa_X2b8.M295_M296_M280_M281_sci.spw17.cube.I.sd.fits'
    filename = create_moment_masking_parameterfile(source='Fit_cube_example.param', destination=folder_destination,
                                                   fits_file_name=name_of_fits)
    param = BTS.read_parameters(filename)
    # # # # # Run the function to make the moments using the moment-masking technique
    BTS.make_moments(param)

    plot_moment_maps(path='moment_maps_fits/'+core+'/', filename='DCO+_'+core)

