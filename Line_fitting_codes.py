import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization.wcsaxes import SphericalCircle
from make_TP_maps import *

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

    guess_params = [[7, 5, 2],[5, 5, 2, 8, 5, 2]]
    # guess_params = [[7, 5, 2],[7, 5, 2]]

    velocity_centroids=[]
    for counter,molec in enumerate(molecules):
        full_filename_path = find_the_spectrum_for_a_source(folders_path, molec)
        filename = full_filename_path.split('/')

        SNR = calculate_peak_SNR(folders_path, filename=filename[-1])
        spectrum, velocity = make_average_spectrum_data(folders_path,filename[-1])

        popt = gaussian_parameters_of_spectra(velocity, spectrum, guess=guess_params[counter], plot=False)
        fit = multi_gaussian(velocity, *popt)

        print(popt)
        
        velocity_centroids.append(popt[0])
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
        plt.xlim(3,9)
    plt.legend()

    print('velocity centroid difference = ', abs(velocity_centroids[1]-velocity_centroids[0]))
    plt.title(folders_path.split('/')[-1] , fontsize=18)

    save_fig_name = 'Averaged_spectra_'+ folders_path.split('/')[-1] + '_' + '_vs_'.join(molecules) + 'ref3.png'
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

    fit_gaussians_to_spectra(folders_path='TP_FITS/M308', molecules=['N2D+','C18O'],
                          normalized=True,save=False)