import numpy as np
import pylab as py
from astropy.io import fits as pyfits
from scipy.interpolate import RectBivariateSpline
import math
import pdb
from astropy.table import Table

class Observation(object):
    def __init__(self, instrument, scene, psf_grid, wave, background,
                 origin=[0,0], PA=0):
        """
        background - Background in electrons per second
        origin - a 2D array giving the pixel coordinates that correspond
                 to the 0, 0 point in the arcsecond-based coordinate
                 system for the scene. This (along with the PA and the
                 instrument scale) allows the scene coordinates to be
                 converted into pixel coordinates. 
        PA - the position angle (east of north) of the Y-axis of the
             detector in degrees. Default is zero.
        """
        # This will be the image in electrons... convert to DN at the end.
        img = np.zeros(instrument.array_size, dtype=float)

        itime_tot = instrument.itime * instrument.coadds
        flux_to_counts = itime_tot / instrument.gain

        # Add the background and dark current in electrons
        img += (background + instrument.dark_current) * flux_to_counts

        # Total readnoise in electrons
        readnoise = instrument.readnoise / instrument.gain
        readnoise /= math.sqrt(instrument.fowler)

        # i and j are the coordinates into the PSF array. Make it 0 at the center.
        # i goes along the x direction (2nd index in image array)
        # j goes along the y direction (1st index in image array)
        psf_j = np.arange(psf_grid.psf.shape[3]) - (psf_grid.psf.shape[3] / 2)
        psf_i = np.arange(psf_grid.psf.shape[4]) - (psf_grid.psf.shape[4] / 2)

        psf_j_scaled = psf_j * (psf_grid.psf_scale[wave] / instrument.scale)
        psf_i_scaled = psf_i * (psf_grid.psf_scale[wave] / instrument.scale)

        x, y = convert_scene_to_pixels(scene, instrument, origin, PA)
    

        keep_idx = []

        # Add the point sources
        print( 'Observation: Adding stars one by one.')
        for ii in range(len(x)):
            if ii % 1000 == 0:
                print( ii)
            # Fetch the appropriate interpolated PSF and scale by flux.
            # This is only good to a single pixel.
            try:
                psf = psf_grid.get_local_psf(x[ii], y[ii], wave)
            except ValueError as err:
                # Skip this star.
                continue
            
            psf *= scene.flux[ii] * flux_to_counts

            if psf.min() < 0:
                pdb.set_trace()

            # Project this PSF onto the detector at this position.
            # This includes sub-pixel shifts and scale changes.

            # Coordinates of the PSF's pixels at this star's position
            psf_i_old = psf_i_scaled + x[ii]
            psf_j_old = psf_j_scaled + y[ii]

            # Make the interpolation object.
            # Can't keep this because we have a spatially variable PSF.
            psf_interp = RectBivariateSpline(psf_j_old, psf_i_old, psf, kx=1, ky=1)

            # New grid of points to evaluate at for this star.
            xlo = int(psf_i_old[0])
            xhi = int(psf_i_old[-1])
            ylo = int(psf_j_old[0]) + 1
            yhi = int(psf_j_old[-1]) + 1

            # Remove sections that will be off the edge of the image
            if xlo < 0:
                xlo = 0
            if xhi > img.shape[1]:
                xhi = img.shape[1]
            if ylo < 0:
                ylo = 0
            if yhi > img.shape[0]:
                yhi = img.shape[0]
                
            # Interpolate the PSF onto the new grid.
            psf_i_new = np.arange(xlo, xhi)
            psf_j_new = np.arange(ylo, yhi)
            psf_star = psf_interp(psf_j_new, psf_i_new, grid=True)

            # Add the PSF to the image.
            img[ylo:yhi, xlo:xhi] += psf_star

            keep_idx.append(ii)

        print( 'Observation: Finished adding stars.')
        
        #####
        # ADD NOISE: Up to this point, the image is complete; but noise free.
        #####
        # Add Poisson noise from dark, sky, background, stars.
        img_noise = np.random.poisson(img, img.shape)
        
        # Add readnoise. We get readnoise for every coadd. 
        readnoise_all_reads = np.random.normal(loc=0, scale=readnoise, size=img.shape.append(instrument.coadds))
        img_noise += readnoise_all_reads.sum(axis=2)

        # Save the image to the object
        self.img = img + img_noise

        # Create a table containing the information about the stars planted.
        stars_x = x[keep_idx]
        stars_y = y[keep_idx]
        stars_counts = scene.flux[keep_idx] * flux_to_counts
        stars = Table((stars_x, stars_y, stars_counts),
                        names=("xpix", "ypix", "counts"),
                        meta={'name':'stars table'})
        self.stars = stars

        # TO DO: Add support for saturation and creating a *.max file with the
        # saturation limit of this image included.
        # sat_in_DN = instrument.saturation * instrument.coadds / instrument.gain
        # SAVE TO *.max FILE. inside save_to_fits().
        
        return

    def save_to_fits(self, fitsfile, clobber=False):
        pyfits.writeto(fitsfile, self.img, clobber=clobber)

        self.stars.write(fitsfile.replace('.fits', '_stars_table.fits'), format='fits', overwrite=clobber)

        return

def convert_scene_to_pixels(scene, instrument, origin, posang):
    # Remeber that python images are indexed with img[y, x].
    
    # Get the X and Y pixel positions of the sources in the scene.
    # We assume they are in arcseconds increasing to the North and East.
    x_tmp = (scene.xpos / instrument.scale)
    y_tmp = (scene.ypos / instrument.scale)

    # Rotate to the proper PA (rotate counter-clockwise by the specified angle)
    sina = math.sin(math.radians(posang))
    cosa = math.cos(math.radians(posang))

    x_pix = (x_tmp * cosa) + (y_tmp * -sina)
    y_pix = (x_tmp * sina) + (y_tmp * cosa)

    # Flip the x-axis to increase to the right.
    x_pix *= -1.0

    x_pix += origin[0]
    y_pix += origin[1]

    return x_pix, y_pix

