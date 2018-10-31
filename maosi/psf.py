import numpy as np
import pylab as py
from astropy.io import fits as pyfits
import pdb
from scipy import interpolate as interp

class PSF_grid(object):
    """
    Container for a grid of PSFs. This function hosts
    all of the interpolation routines such that you can "get_psf"
    anywhere in the focal plane.
    """


    def __init__(self, psf,
                 wave_array=[487, 625, 770, 870, 1020, 1250, 1650, 2120],
                 grid_shape=[11,11]):
        """
        Load up a FITS file that contains at least 1, or possibly a grid (list)
        of PSFs. These are stored and you can interpolate between them
        if necessary.
        """
        self.img_scale = 0.10  # arcseconds per pixel
        
        # Fix wave_array to be a float
        wave_array = np.array(wave_array, dtype=float)
        wave_shape = psf.shape[0]

        if wave_shape != len(wave_array):
            print( 'Problem with PSF shape and wave_array shape' )
            
        # Reshape the array to get the X and Y positions
        psf = psf.reshape((wave_shape, grid_shape[0], grid_shape[1],
                           psf.shape[2], psf.shape[3]))
        psf = np.swapaxes(psf, 1, 2)

        # scale array = lambda / 2D (Nyquist sampled)
        tel_diam = 2.235  # meters
        psf_scale = wave_array * (206264.8 * 1e-9) / (2.0 * tel_diam) # arcsec / pixel
        
        # Calculate the positions of all these PSFs. We assume that the
        # outermost PSFs are at the corners such that all observed stars
        # are internal to these corners.
        x_pos = np.array(np.mgrid[0:grid_shape[1]], dtype=float)
        y_pos = np.array(np.mgrid[0:grid_shape[0]], dtype=float)

        # Need to multiply by some of the array size properties.
        # Note that this assumes a pixel scale.
        fov = 10.    # arcmin
        fov *= 60.   # arcsec
        fov /= self.img_scale # pixels

        x_pos *= fov / x_pos[-1]
        y_pos *= fov / y_pos[-1]

        xx, yy = np.meshgrid(x_pos, y_pos)

        # Reshape back to a PSF list.
        # Reshape the array to get the X and Y positions
        n_psfs = psf.shape[1] * psf.shape[2]
        psf = psf.reshape((wave_shape, n_psfs,
                           psf.shape[3], psf.shape[4]))
        xx = xx.reshape(n_psfs)
        yy = yy.reshape(n_psfs)
        
        self.psf = psf
        self.psf_x = xx  # 2D array
        self.psf_y = yy  # 2D array
        # self.psf_x = x_pos  # 1D array
        # self.psf_y = y_pos  # 1D array
        self.psf_wave = wave_array
        self.wave_shape = wave_shape
        self.grid_shape = grid_shape
        self.psf_scale = psf_scale

        self.interpolator = [None for ii in self.psf_wave]
        
        return

    @classmethod
    def from_file(cls, psf_file,
                  wave_array=[487, 625, 770, 870, 1020, 1250, 1650, 2120],
                  grid_shape=[11,11]):
        # 4D array with [wave, grid_idx, flux_x, flux_y]
        psf = pyfits.getdata(psf_file)

        return cls(psf, wave_array=wave_array, grid_shape=grid_shape)

        
    def get_local_psf(self, x, y, wave_idx, method='bilinear'):
        """
        Return an interpolated PSF at the requested [x, y] location.
        Interpolation method is nearest neighbor ('neighbor', default) or
        bilinear interpolation ('bilinear').
        """
        psf_x = self.psf_x
        psf_y = self.psf_y

        # We can't handle stars that are outside our PSF grid.
        # But fail gracefully with an exception. 
        if (x < psf_x.min()) or (x > psf_x.max()):
            raise ValueError('x is outside the valid PSF grid region')
        if (y < psf_y.min()) or (y > psf_y.max()):
            raise ValueError('y is outside the valid PSF grid region')
        
        # Find the PSF
        if method == 'neighbor':
            pidx = np.argmin(np.hypot(psf_x - x, psf_y - y))
            
            psf_loc = self.psf[wave_idx, pidx]

        elif method == 'bilinear':
            
            # Distance of all PSFs to the star
            dr = np.hypot(psf_x - x, psf_y - y)

            # Find PSFs in each quadrant
            xlo_ylo = np.where((psf_x <= x) & (psf_y <= y))[0]
            xlo_yhi = np.where((psf_x <= x) & (psf_y > y))[0]
            xhi_ylo = np.where((psf_x > x) & (psf_y <= y))[0]
            xhi_yhi = np.where((psf_x > x) & (psf_y > y))[0]

            # Find the closest PSF in each quadrant
            idx_xlo_ylo = np.argmin(dr[xlo_ylo])
            idx_xlo_yhi = np.argmin(dr[xlo_yhi])
            idx_xhi_ylo = np.argmin(dr[xhi_ylo])
            idx_xhi_yhi = np.argmin(dr[xhi_yhi])

            # Select the image of the closest PSF in each quadrant
            psf_xlo_ylo = self.psf[wave_idx, xlo_ylo[idx_xlo_ylo]]
            psf_xlo_yhi = self.psf[wave_idx, xlo_yhi[idx_xlo_yhi]]
            psf_xhi_ylo = self.psf[wave_idx, xhi_ylo[idx_xhi_ylo]]
            psf_xhi_yhi = self.psf[wave_idx, xhi_yhi[idx_xhi_yhi]]

            # Select the x distance of the closest PSF in each quadrant
            dx_xlo_ylo = x - psf_x[xlo_ylo[idx_xlo_ylo]]
            dx_xlo_yhi = x - psf_x[xlo_yhi[idx_xlo_yhi]]
            dx_xhi_ylo = psf_x[xhi_ylo[idx_xhi_ylo]] - x
            dx_xhi_yhi = psf_x[xhi_yhi[idx_xhi_yhi]] - x

            # Select the y distance of the closest PSF in each quadrant
            dy_xlo_ylo = y - psf_y[xlo_ylo[idx_xlo_ylo]]
            dy_xlo_yhi = psf_y[xlo_yhi[idx_xlo_yhi]] - y
            dy_xhi_ylo = y - psf_y[xhi_ylo[idx_xhi_ylo]]
            dy_xhi_yhi = psf_y[xhi_yhi[idx_xhi_yhi]] - y
            
            # Calculate the "sides" of the four PSFs asterism
            dx_bottom = psf_x[xhi_ylo[idx_xhi_ylo]] - psf_x[xlo_ylo[idx_xlo_ylo]]
            dx_top = psf_x[xhi_yhi[idx_xhi_yhi]] - psf_x[xlo_yhi[idx_xlo_yhi]]
            dy_left = psf_y[xlo_yhi[idx_xlo_yhi]] - psf_y[xlo_ylo[idx_xlo_ylo]]
            dy_right = psf_y[xhi_yhi[idx_xhi_yhi]] - psf_y[xhi_ylo[idx_xhi_ylo]]

            # Calculate the weight of the four closest PSFs
            weight_xlo_ylo = (1 - (dx_xlo_ylo / dx_bottom)) * (1 - (dy_xlo_ylo / dy_left))
            weight_xlo_yhi = (1 - (dx_xlo_yhi / dx_top)) * (1 - (dy_xlo_yhi / dy_left))
            weight_xhi_ylo = (1 - (dx_xhi_ylo / dx_bottom)) * (1 - (dy_xhi_ylo / dy_right))
            weight_xhi_yhi = (1 - (dx_xhi_yhi / dx_top)) * (1 - (dy_xhi_yhi / dy_right))

            psf_loc = weight_xlo_ylo * psf_xlo_ylo + \
                      weight_xlo_yhi * psf_xlo_yhi + \
                      weight_xhi_ylo * psf_xhi_ylo + \
                      weight_xhi_yhi * psf_xhi_yhi

        else:
            # Nearest Neighbor:
            r = np.hypot(psf_x - x, psf_y - y)
            rdx = np.argmin(r)

            psf_loc = self.psf[wave_idx, psf_x[rdx], psf_y[rdx]]
            
        return psf_loc

    def get_local_psf_nn(self, x, y, wave_idx):
        """
        Return an the nearest neighbor PSF at the requested [x, y] location.
        This works for requests beyond the grid boundaries, but leads to 
        dsiscrete jumps.
        """
        psf_x = self.psf_x
        psf_y = self.psf_y

        dx = x - psf_x
        dy = y - psf_y

        xidx = np.argmin(np.abs(dx))
        yidx = np.argmin(np.abs(dy))
        
        psf_loc = self.psf[wave_idx, yidx, xidx]
            
        return psf_loc
    
    def plot_psf_grid(self, wave_idx, psf_size=[50,50]):
        # Chop down the PSFs to the plotting region
        # and at the wavelength requested.
        psf_shape = self.psf.shape[-2:]
        psf_x_lo = int((psf_shape[1] / 2.0) - (psf_size[1] / 2.0))
        psf_y_lo = int((psf_shape[0] / 2.0) - (psf_size[0] / 2.0))
        psf_x_hi = psf_x_lo + psf_size[1]
        psf_y_hi = psf_y_lo + psf_size[0]

        psf = self.psf[wave_idx, :, :, psf_y_lo:psf_y_hi, psf_x_lo:psf_x_hi]
        psf_shape = psf.shape[-2:]
        grid_shape = psf.shape[0:2]

        img = np.zeros((psf_shape[0] * grid_shape[0],
                        psf_shape[1] * grid_shape[1]), dtype=float)

        for xx in range(grid_shape[1]):
            for yy in range(grid_shape[0]):
                xlo = 0 + (xx * psf_shape[1])
                xhi = xlo + psf_shape[1]
                ylo = 0 + (yy * psf_shape[0])
                yhi = ylo + psf_shape[0]
                
                img[ylo:yhi, xlo:xhi] = psf[yy, xx, :, :]

        py.clf()
        py.imshow(img)

        return


    

    
