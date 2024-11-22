from imfits import Imfits, au


def main():
    # ------------- get moment map -------------
    f = 'hl_tau.h2co.contsub.selfcal.robust0.5.pbcor.subim.fits'
    rms = 1.65e-3
    vrange = [1.6, 13.9] # velocity range with more than 3sigma detection

    # moment map
    cube = Imfits(f)
    cube.getmoments([1], vrange = vrange, threshold = [3.*rms, 1000.],
        outname = f.replace('.fits', '.mom1.fits'), overwrite = True)


    # --------------- 2D linear fit ------------------
    # linear fit
    f = 'hl_tau.h2co.contsub.selfcal.robust0.5.pbcor.subim.mom1.fits'
    mom = Imfits(f)
    au.lnfit2d(mom,
    [1., 0., 0.], # initla guess for [v0, a, b]
    rfit = 1.6, # Radius of a circule in which fitting is performed
    )


if __name__ == '__main__':
    main()
