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

#Astropy modules to deal with coordinates
from astropy.wcs import WCS
from astropy.wcs import Wcsprm
from astropy.io import fits
from astropy.wcs import utils

#BTS code developed by Seamus to get nice moment maps using cube masking
#instead of sigma clippping

# module_path = os.path.abspath(os.path.join('/Users/christianflores/Documents/Work/BTS-master/')) # or the path to your source code
module_path = os.path.abspath(os.path.join('/Users/christianflores/Documents/Work/external_codes/BTS-master'))
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
            self.nz = self.header['NAXIS3']
            self.vel = self.get_vel(self.header)
            dv = self.vel[1] - self.vel[0]
            if (dv < 0):
                dv = dv * -1

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

            rest = self.accurate_reference_frequency(self.filename)  # head["RESTFRQ"]

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

    def accurate_reference_frequency(self, filename):
        if "spw17" in filename:
            freq_rest = 2.16278749E+11  # c-C3H2
            freq_rest = 2.16112628E+11  #
            mole_name = "DCO+"

        if "spw19" in filename:
            freq_rest = 2.19949433E+11
            mole_name = "SO"

        if "spw21" in filename:
            freq_rest = 2.19560353E+11
            mole_name = "C18O"

        if "spw23" in filename:
            freq_rest = 2.30538000E+11
            mole_name = "12CO"

        if "spw25" in filename:
            freq_rest = 2.31220768E+11
            mole_name = "13CS"

        if "spw27" in filename:
            freq_rest = 2.31321635E+11
            mole_name = "N2D+"
        return float(freq_rest)  # ,mole_name


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
#     plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)
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

def find_all_spectra_for_a_molecule(folders_path):
    '''
    Find the path to all the spectra of a given molecule
    '''
    array_of_paths=[]
    molecule = '.spw19.'
    folder_list = next(os.walk(folders_path))[1]
    for each_folder in folder_list:
        filenames = get_files_in_directory(os.path.join(folders_path,each_folder))
        for names in filenames:
            if molecule in names:
                array_of_paths.append(os.path.join(each_folder,names))
    return array_of_paths


def plot_grid_of_spectra(folders_path):
    array_of_paths = find_all_spectra_for_a_molecule(folders_path)

    number_of_sources = len(array_of_paths)
    grid_size = int(math.ceil(number_of_sources ** 0.5))
    fig = plt.figure(figsize=(15., 15.))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of Axes
                     axes_pad=0.3, aspect=False  # pad between Axes in inch.
                     )

    for ax, sources in zip(grid, array_of_paths):
        #     for sources in array_of_paths:
        spectrum, velocity = make_average_spectrum_data(path='TP_FITS',
                                                        filename=sources)

        SNR = calculate_peak_SNR(path='TP_FITS', filename=sources)
        ax.plot(velocity, spectrum)
        print(abs(velocity[10] - velocity[11]))
        ax.set_xlim(-6, 18)
        ax.set_title(sources.split('/')[0])

        ax.text(x=0.05, y=0.9, s='SNR = ' + str(int(SNR)), ha='left', va='top',
                transform=ax.transAxes, size=12, color='purple')

    #     fig.savefig('this_spw25_n.png', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    param = BTS.read_parameters("TP_FITS/M236/Fit_cube.param")

    print(os.getcwd())

    # # Run the function to make the moments using the moment-masking technique
    BTS.make_moments(param)

    # # Using the generated mask, fit the entire datacube
    # BTS.fit_a_fits(param)