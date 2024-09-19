import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from make_TP_maps import *
from scipy.signal import correlate



def convolution_functions(folders_path,molecules,save=False):
    """
    Convolve functions to find best center
    NOT QUITE WORKING YET
    """

    if isinstance(molecules, list):
        print("The argument is already a list.")
    elif isinstance(molecules, str):
        print("The argument is a string. Converting it to a list.")
        molecules = [molecules]
    else:
        print("The argument is neither a list nor a string.")
        return None

    plt.figure()

    spectra_array=[]
    velocity_array=[]
    velocity_range=[-50.,50]


    for counter,molec in enumerate(molecules):
        full_filename_path = find_the_spectrum_for_a_source(folders_path, molec)
        filename = full_filename_path.split('/')

        spectrum, velocity = make_average_spectrum_data(folders_path,filename[-1])

        min_idx = closest_idx(velocity, velocity_range[0])
        max_idx = closest_idx(velocity, velocity_range[1])
        # print(max_idx,min_idx)
        if min_idx<max_idx:
            new_spectrum=spectrum[min_idx:max_idx]/np.nanmax(spectrum[min_idx:max_idx])
            new_velocity = velocity[min_idx:max_idx]

        else:
            new_spectrum= spectrum[max_idx:min_idx]/np.nanmax(spectrum[max_idx:min_idx])
            new_velocity = velocity[max_idx:min_idx]

        print(len(new_spectrum))
        spectra_array.append(new_spectrum)
        velocity_array.append(new_velocity)
        plt.plot(new_velocity,new_spectrum)

    plt.show()
    print(np.nanmedian(velocity))
    convolution= np.convolve(spectra_array[0],spectra_array[1],'same')
    correlated = correlate(spectra_array[0],spectra_array[1][::-1],'same')
    print(len(correlated),len(convolution))

    plt.plot(velocity_array[0],correlated)
    plt.plot(velocity_array[0],convolution)
    #
    # plt.xlabel("velocity [km/s]", size=15)
    # plt.ylabel("Intensity", size=15)
    # plt.tick_params(axis='both', direction='in')
    # plt.xlim(3, 10)
    plt.show()

def fit_gaussians_to_spectra(folders_path,molecules,normalized=False,save=False):
    """
    fit spectra with gaussian to obtain basic parameters
    """
    if isinstance(molecules, list):
        print("The argument is already a list.")
    elif isinstance(molecules, str):
        print("The argument is a string. Converting it to a list.")
        molecules = [molecules]
    else:
        print("The argument is neither a list nor a string.")
        return None

    plt.figure()

    colors=['k','r']
    alpha=[1,0.85]

    # guess_params = [[7, 5, 2],[5, 5, 2, 8, 5, 2]]
    guess_params = [[7, 5, 2],[7, 5, 2]]

    velocity_centroids=[]
    velocity_at_peaks=[]
    spectra_array=[]
    for counter,molec in enumerate(molecules):
        full_filename_path = find_the_spectrum_for_a_source(folders_path, molec)
        filename = full_filename_path.split('/')

        SNR = calculate_peak_SNR(folders_path, filename=filename[-1])
        spectrum, velocity = make_average_spectrum_data(folders_path,filename[-1])
        spectra_array.append(spectrum)
        # spectrum = gaussian_filter(spectrum,sigma=3)

        popt = gaussian_parameters_of_spectra(velocity, spectrum, guess=guess_params[counter], plot=False)
        fit = multi_gaussian(velocity, *popt)

        velocity_centroids.append(popt[0])
        print('Best Gaussian parameters = ', popt)

        #### find velocity of the peak emission
        peak_emission = np.nanmax(spectrum)
        peak_index= closest_idx(spectrum, peak_emission)
        peak_velocity = velocity[peak_index]
        print('velocity at peak position = ', peak_velocity)
        velocity_at_peaks.append(peak_velocity)

        # plt.title("Averaged Spectrum ("+mole_name+") @"+dir_each)
        plt.xlabel("velocity [km/s]", size=15)
        plt.ylabel("Intensity", size=15)
        if normalized:
            plt.plot(velocity,spectrum/np.nanmax(spectrum),"-",color=colors[counter],lw=2,label=molec)
            plt.plot(velocity, fit/np.nanmax(fit), 'b--', label="Gaussian Fit")

        else:
            plt.plot(velocity,spectrum,"-",color=colors[counter],lw=2)
            plt.plot(velocity, fit, 'g--', label="Gaussian Fit")

        plt.tick_params(axis='both', direction='in')
        plt.xlim(3,10)
    plt.legend()

    print('velocity centroid difference = ', round(velocity_centroids[1]-velocity_centroids[0],3))

    print('velocity peak difference = ', round(velocity_at_peaks[1]-velocity_at_peaks[0],3))

    plt.text(x=3.5, y=0.9, s='v.c.d = ' + str(round(velocity_centroids[1]-velocity_centroids[0],3)) +' km/s', ha='left', va='top',
             size=12)

    plt.text(x=3.5, y=0.7, s='v.p.d = ' + str(round(velocity_at_peaks[1]-velocity_at_peaks[0],3)) +' km/s', ha='left', va='top',
             size=12)

    plt.title(folders_path.split('/')[-1] , fontsize=18)
    save_fig_name = 'fitted_averaged_spectra_'+ folders_path.split('/')[-1] + '_' + '_vs_'.join(molecules) + 'ref3.png'
    save_folder = os.path.join('Figures', 'spectra_comparison')

    print(save_fig_name)
    if save:
        plt.savefig(os.path.join(save_folder, save_fig_name), bbox_inches='tight',dpi=300)

    plt.show()


def multi_gaussian(x, *params):
    n_gaussians = len(params) // 3
    y = np.zeros_like(x)

    for i in range(n_gaussians):
        centroid = params[i * 3]
        amplitude = params[i * 3 + 1]
        width = params[i * 3 + 2]
        y += amplitude * np.exp(-(x - centroid) ** 2 / (2 * width ** 2))

    return y


def gaussian_parameters_of_spectra(velocity, spectrum, guess=[], plot=False):
    '''
    guess is a list of 3-tuples for each Gaussian component: (centroid, amplitude, width)
    e.g., [centroid1, amplitude1, width1, centroid2, amplitude2, width2, ...]
    '''
    if guess == []:
        guess = [7, 5, 2]  # Default to a single Gaussian guess

    # Calculate the RMS of the spectrum (ignoring NaNs) for bounds
    rms = np.nanstd(spectrum[10:40])

    # Number of Gaussians to fit
    n_gaussians = len(guess) // 3

    # Define bounds: these should be tuples that define (min_bounds, max_bounds) for each parameter
    # Example bounds per Gaussian: centroid (2, 10), amplitude (rms*8, 100), width (0.001, 10)
    min_bounds = [2, rms * 8, 0.001] * n_gaussians
    max_bounds = [10, 100, 10] * n_gaussians
    bounds = (min_bounds, max_bounds)

    # Perform the fit
    popt, pcov = curve_fit(multi_gaussian, velocity, spectrum, p0=guess, bounds=bounds)

    # Calculate the average centroid for plotting purposes
    average_centroid = np.nanmedian(popt[::3])

    # Generate the fitted curve
    fit = multi_gaussian(velocity, *popt)

    # Plot the result if requested
    if plot:
        plt.plot(velocity, spectrum, label="Observed Spectrum")
        plt.plot(velocity, fit, 'r-', label="Gaussian Fit")
        plt.xlim(average_centroid - 5, average_centroid + 5)
        plt.legend()
        plt.show()

    return popt

if __name__ == "__main__":

    # fit_gaussians_to_spectra(folders_path='TP_FITS/M456', molecules=['N2D+','C18O'],
    #                       normalized=True,save=False)

    convolution_functions(folders_path='TP_FITS/M456', molecules=['N2D+','C18O'],
                          save=False)