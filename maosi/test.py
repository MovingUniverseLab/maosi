import psf
import scene
import instrument
import observation
import numpy as np
import pylab as py
import pyfits
import time

Nstars = 100
stars_x = np.random.rand(Nstars) * 6000 * 0.1 *-1
stars_y = np.random.rand(Nstars) * 6000 * 0.1
stars_f_tmp = np.random.rand(Nstars)
stars_f = (stars_f_tmp**-0.5) - 1.0
stars_f *= 1e6 / stars_f_tmp.max()

def prepare_test_imaka(rootdir='/Users/jlu/work/imaka/sims/psfs/'):
    psf_file = rootdir + 'PSFs_Imaka88-offner-10x10field_5GS8rad-Nyquist.fits'

    print('Loading PSF grid from: ')
    print(psf_file)
    psf_grid_raw = pyfits.getdata(psf_file)

    return psf_grid_raw

def test_imaka(psf_grid_raw):
    time_start = time.time()
    
    if psf_grid_raw is None:
        psf_grid_raw = prepare_test()
    
    h4rg = instrument.Instrument((6000,6000), 4.0, 10., 4.)
    h4rg.scale = 0.1
    sources = scene.Scene(stars_x, stars_y, stars_f)
    psfgrid = psf.PSF_grid(psf_grid_raw)

    print('Making Image: {0} sec'.format(time.time() - time_start))
    obs = observation.Observation(h4rg, sources, psfgrid, 4, 3.0)
    print('Saving Image: {0} sec'.format(time.time() - time_start))
    obs.save_to_fits('tmp.fits', clobber=True)

    # print 'Displaying Image'
    # py.clf()
    # py.imshow(obs.img, cmap='gist_heat')

    # # Zoom in
    # py.axis([2000, 3000, 2000, 3000])


    # Print out some runtime information.    
    time_end = time.time()
    run_time = time_end - time_start
    print('Total Time:    {0} seconds'.format(run_time))
    print('Time Per Star: {0} seconds'.format(run_time / Nstars))

    return obs

    

    
