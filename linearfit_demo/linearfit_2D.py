import sys
import os
import numpy as np
sys.path.append('/Users/christianflores/Documents/Work/GitHub/B213-core-rotation-project')

from make_TP_maps import find_the_spectrum_for_a_source

sys.path.append('/Users/christianflores/Documents/Work/GitHub/B213-core-rotation-project/imfits')
from imfits import Imfits, au

my_path = os.getcwd()
os.chdir(my_path)


def main():
    # ------------- get moment map -------------

    source_name = 'M503'
    folder_path = os.path.join('../TP_FITS',source_name)
    filename = find_the_spectrum_for_a_source(folder_path, spw_or_molec='C18O')
    print(filename)
    # vrange = [5.7, 7.2] # velocity range with more than 3sigma detection
    vrange = [5.6, 7.3] # velocity range with more than 3sigma detection

    # # moment map
    cube = Imfits(filename)

    noise_level = cube.getrms_cube(vwindows=[[50,250],[750,950]])
    print('noise level in Jy/beam ', noise_level)
    mome_one_file = filename.replace(filename.split('/')[-1], 'velocity_gradient_' +source_name + '.mom1.fits')

    cube.getmoments([1], vrange = vrange, threshold = [3.*noise_level, 1000.],
        outname = mome_one_file, overwrite = True)

    # # --------------- 2D linear fit ------------------
    # # linear fit
    #  filename.replace('.fits', '.mom1.fits')
    mom = Imfits(mome_one_file)
    au.lnfit2d(mom,
    [np.mean(vrange), 0.1, 0.1], # initla guess for [v0, a, b]
    rfit = 46.0, # Radius of a circle (in arcsec) in which fitting is performed
    outfig=True, outname = source_name + '_gradient_direction')


if __name__ == '__main__':
    main()
