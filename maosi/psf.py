import numpy as np
import pylab as py
from astropy.io import fits
import fnmatch
import os
import pdb
from scipy import interpolate as interp

class PSF_grid(object):
    """
    Container for a grid of PSFs. This function hosts
    all of the interpolation routines such that you can "get_psf"
    anywhere in the focal plane.
    """


    def __init__(self, psf,
                 wave_array=[870],
                 grid_shape=[11,11],
                 grid_step=1.0,
                 psf_scale=0.00396028):
        """
        Load up a FITS file that contains at least 1, or possibly a grid (list)
        of PSFs. These are stored and you can interpolate between them
        if necessary.

        psf : np.array, len=N_wave * N_grid_x * N_grid_y * N_psf_pix_x * N_psf_pix_y]

        wave_array : list
            List of wavelengths in nm.
        
        grid_shape : list, len=2
            Shape of grid e.g. [11,11]
        
        grid_step : float
            grid step size in arcsec

        psf_scale : float
            PSF pixel scale in arcsec / pix
        """
        self.img_scale = psf_scale  # arcseconds per pixel             
        
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
        #tel_diam = 2.235  # meters
        #psf_scale = wave_array * (206264.8 * 1e-9) / (2.0 * tel_diam) # arcsec / pixel
        
        # Calculate the positions of all these PSFs. We assume that the
        # outermost PSFs are at the corners such that all observed stars
        # are internal to these corners.
        x_pos = np.array(np.mgrid[0:grid_shape[1]], dtype=float) * grid_step
        y_pos = np.array(np.mgrid[0:grid_shape[0]], dtype=float) * grid_step

        # Need to multiply by some of the array size properties.
        # Note that this assumes a pixel scale.
        xx, yy = np.meshgrid(x_pos, y_pos)
                     
        # Reshape back to a PSF list.
        # Reshape the array to get the X and Y positions
        n_psfs = psf.shape[1] * psf.shape[2]
        psf = psf.reshape((wave_shape, n_psfs,
                           psf.shape[3], psf.shape[4]))
        
        xx = xx.reshape(n_psfs)
        yy = yy.reshape(n_psfs)
        
        self.psf = psf
                     
        self.psf_x_2d = xx  # 2D array
        self.psf_y_2d = yy  # 2D array
        self.psf_x = x_pos  # 1D array
        self.psf_y = y_pos  # 1D array
                     
        self.psf_wave = wave_array
        self.wave_shape = wave_shape
        self.grid_shape = grid_shape
        self.psf_scale = np.array([0.00396028])
        self.fov = fov # pixels

        self.interpolator = [None for ii in self.psf_wave]
        
        return

    @classmethod
    def from_file(cls, psf_file,
                  wave_array=[487, 625, 770, 870, 1020, 1250, 1650, 2120],
                  grid_shape=[11,11]):
        # 4D array with [wave, grid_idx, flux_x, flux_y]
        psf = fits.getdata(psf_file)

        return cls(psf, wave_array=wave_array, grid_shape=grid_shape)

        
    def get_local_psf(self, x, y, wave_idx, method='bilinear'):
        """
        Return an interpolated PSF at the requested [x, y] location.
        Interpolation method is nearest neighbor ('neighbor', default) or
        bilinear interpolation ('bilinear').
        """
        psf_x = self.psf_x_2d
        psf_y = self.psf_y_2d
        
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
            xlo_ylo = np.where((psf_x <= x) & (psf_y <= y))
            xlo_yhi = np.where((psf_x <= x) & (psf_y > y))
            xhi_ylo = np.where((psf_x > x) & (psf_y <= y))
            xhi_yhi = np.where((psf_x > x) & (psf_y > y))

            # Find the closest PSF in each quadrant
            idx_xlo_ylo = np.argmin(dr[xlo_ylo])
            idx_xlo_yhi = np.argmin(dr[xlo_yhi])
            idx_xhi_ylo = np.argmin(dr[xhi_ylo])
            idx_xhi_yhi = np.argmin(dr[xhi_yhi])

            # Select the image of the closest PSF in each quadrant
            psf_xlo_ylo = self.psf[wave_idx, xlo_ylo[0][idx_xlo_ylo], xlo_ylo[1][idx_xlo_ylo]]
            psf_xlo_yhi = self.psf[wave_idx, xlo_yhi[0][idx_xlo_yhi], xlo_yhi[1][idx_xlo_yhi]]
            psf_xhi_ylo = self.psf[wave_idx, xhi_ylo[0][idx_xhi_ylo], xhi_ylo[1][idx_xhi_ylo]]
            psf_xhi_yhi = self.psf[wave_idx, xhi_yhi[0][idx_xhi_yhi], xhi_yhi[1][idx_xhi_yhi]]

            x_xlo_ylo = psf_x[xlo_ylo[0][idx_xlo_ylo], xlo_ylo[1][idx_xlo_ylo]]
            y_xlo_ylo = psf_y[xlo_ylo[0][idx_xlo_ylo], xlo_ylo[1][idx_xlo_ylo]]
            x_xlo_yhi = psf_x[xlo_yhi[0][idx_xlo_yhi], xlo_yhi[1][idx_xlo_yhi]]
            y_xlo_yhi = psf_y[xlo_yhi[0][idx_xlo_yhi], xlo_yhi[1][idx_xlo_yhi]]
            x_xhi_ylo = psf_x[xhi_ylo[0][idx_xhi_ylo], xhi_ylo[1][idx_xhi_ylo]]
            y_xhi_ylo = psf_y[xhi_ylo[0][idx_xhi_ylo], xhi_ylo[1][idx_xhi_ylo]]
            x_xhi_yhi = psf_x[xhi_yhi[0][idx_xhi_yhi], xhi_yhi[1][idx_xhi_yhi]]
            y_xhi_yhi = psf_y[xhi_yhi[0][idx_xhi_yhi], xhi_yhi[1][idx_xhi_yhi]]

            # Select the x distance of the closest PSF in each quadrant
            dx_xlo_ylo = x - x_xlo_ylo
            dx_xlo_yhi = x - x_xlo_yhi
            dx_xhi_ylo = x_xhi_ylo - x
            dx_xhi_yhi = x_xhi_yhi - x

            # Select the y distance of the closest PSF in each quadrant
            dy_xlo_ylo = y - y_xlo_ylo
            dy_xlo_yhi = y_xlo_yhi - y
            dy_xhi_ylo = y - y_xhi_ylo
            dy_xhi_yhi = y_xhi_yhi - y
            
            # Calculate the "sides" of the four PSFs asterism
            dx_bottom = x_xhi_ylo - x_xlo_ylo
            dx_top = x_xhi_yhi - x_xlo_yhi
            dy_left = y_xlo_yhi - y_xlo_ylo
            dy_right = y_xhi_yhi - y_xhi_ylo

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

        # Normalize the PSF.
        psf_loc /= psf_loc.sum()

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

class MAOS_PSF_grid_from_line(PSF_grid):
    from paarti import psfs as paarti_psfs

    def __init__(self, directory, seed, wave_idx=None):
        """
        Load up a MAOS run where we have run a single line of PSFs,
        usually along the X direction. Turn these into a 2D grid
        through reflection and interpolation.

        directory : str
            Name of directory where MAOS was run.
        
        seed : int
            Seed of MAOS run.
        
        wave_idx : list or None
            List of indices of wavelengths to keep.
        """
        filelist = os.listdir(directory)
        fits_files = fnmatch.filter(filelist, f'evlpsfcl_{seed}_x*_y*.fits')
        n_pos = len(fits_files)
        first_file = True

        # Loop through all the PSF FITS files in the MAOS run directory.
        
        for i, FITSfilename in enumerate(fits_files):
            with fits.open(directory + FITSfilename) as psfFITS:
                # Use the first file to define some arrays.
                if first_file:
                    if wave_idx == None:
                        wave_idx = np.arange(len(psfFITS))

                    n_wvls = len(wave_idx)
                    
                    wavelength = np.empty(n_wvls, dtype=float)
                    pixel_scale = np.empty(n_wvls, dtype=float)
                    
                    pos = np.empty([n_pos, 2])
                    
                    psf_x_size = psfFITS[0].data.shape[1]
                    psf_y_size = psfFITS[0].data.shape[0]
                    
                    psfs = np.empty([n_wvls, n_pos, psf_y_size, psf_x_size])
                    first_file = False

                # Loop through the different wavelengths and load them up.
                for ww in wave_idx:
                    header = psfFITS[ww].header
                    data = psfFITS[ww].data
                    
                    wavelength[ww] = header['wvl']*1E9
                    pixel_scale[ww] = header['dp']
                    
                    psfs[ww, i, :, :] = data
                    pos[i, 0] = header['theta'].real
                    pos[i, 1] = header['theta'].imag


        # Now artificially make a grid from a line along X direction.
        x_old = pos[:, 0]
        y_old = pos[:, 1]
        psf_old = psfs
        
        # Reflect across Y (to get negative x values)
        x_new1 = np.append(pos[1:, 0] * -1, x_old, axis=0)
        y_new1 = np.append(pos[1:, 1]     , y_old, axis=0)
        psf_new1 = np.append(psfs[:, 1:, :, :], psf_old, axis=1)

        # Rotate to get Y (but drop 0 entry so no duplication)
        x_new2 = np.append(pos[1:, 1], x_new1, axis=0)
        y_new2 = np.append(pos[1:, 0], y_new1, axis=0)
        psf_new2 = np.append(psfs[:, 1:, :, :], psf_new1, axis=1)

        x_new3 = np.append(pos[1:, 1]     , x_new2, axis=0)
        y_new3 = np.append(pos[1:, 0] * -1, y_new2, axis=0)
        psf_new3 = np.append(psfs[:, 1:, :, :], psf_new2, axis=1)

        pos = np.array([x_new3, y_new3]).T
        psfs = psf_new3

        ##########
        # Now interpolate between the PSFs onto a regular grid.
        ##########
        # Here is the new grid coordinates. Don't bother extrapolating.
        grid_max = int(np.max(pos) / np.sqrt(2.0))  # scale down by sqrt(2)
        idx_grid_good = np.where(pos[:, 0] < grid_max)[0]  # find positions within this range
        grid_max = pos[idx_grid_good, 0].max() # find the maximum position allowed.
        dgrid = np.abs(pos[1, 1] - pos[0, 1])
        
        x_grid_1d = np.arange(-grid_max, grid_max+1, dgrid)
        y_grid_1d = np.arange(-grid_max, grid_max+1, dgrid)
        x_grid_2d, y_grid_2d = np.meshgrid(x_grid_1d, y_grid_1d)

        # Make a new PSF grid with the right shape.
        #  shape = [N_wvls, N_pos_in_x, N_pos_in_y, PSF_shape_x, PSF_shape_y]
        psf_grid_new = np.zeros([n_wvls, x_grid_2d.shape[0], x_grid_2d.shape[1],
                                 psfs.shape[2], psfs.shape[3]])
        
        from scipy.interpolate import LinearNDInterpolator
        for ww in range(len(wavelength)):
            interp = LinearNDInterpolator(pos, psfs[ww])

            psf_grid_new[ww] = interp(x_grid_2d, y_grid_2d)

        # Now save all our variables.
        self.psf = psf_grid_new
        self.psf_x_2d = x_grid_2d
        self.psf_y_2d = y_grid_2d
        self.psf_x = x_grid_1d
        self.psf_y = y_grid_1d

        self.psf_wave = wavelength
        self.wave_shape = len(wavelength)
        self.grid_shape = np.array(self.psf.shape[1:3])
        self.psf_scale = pixel_scale
        self.fov = self.psf.shape[3]

        return
        
        


        

