import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table
from maosi.scene import Scene
from maosi.instrument import Instrument
from maosi.observation import Observation
from maosi.psf import PSF_grid
import time
import pdb

class GCstars(Scene):
    def __init__(self, label_file='/g/lu/data/gc/source_list/label.dat', instr=None):
        self.label_file = label_file
        
        self.stars = read_label_dat(label_file)

        # Start off our object with all the positions at time t_naught. This
        # isn't strictly observable, but it will do for now.
        x_now = self.stars['x']
        y_now = self.stars['y']
            
        f_now = 10**((self.stars['Kmag'] - instr.ZP_mag) / -2.5) * instr.ZP_flux
        mag_now = self.stars['Kmag']
        name_now = self.stars['name']

        super(self.__class__, self).__init__(x_now, y_now, f_now, mag_now,
                                             name_now)

        return

    def move_to_epoch(self, year):
        """
        year in decimals

        This will modify self.xpos, ypos using velocities from the initial
        label.dat file.
        """
        dt = time - self.stars['t0']
        self.xpos = self.stars['x'] + (dt * self.stars['vx'])
        self.ypos = self.stars['y'] + (dt * self.stars['vy'])

        return


class Grid(Scene):
    def __init__(self, n_grid, mag, noise=False, instr=None):
        
        # Prepare a grid of positions
        img_size = 10 # Approximate size of the image (")
        row = np.delete(np.arange(-(img_size / 2), (img_size / 2), (img_size / (n_grid + 1))), 0)
        grid = np.asarray([row] * n_grid)
        
        if noise:
            x_noise = np.random.uniform(low=-noise/2, high=noise/2, size=len(grid.flatten()))
            y_noise = np.random.uniform(low=-noise/2, high=noise/2, size=len(grid.flatten()))
        else:
            x_noise = np.zeros(len(grid.flatten()))
            y_noise = np.zeros(len(grid.flatten()))
        
        x_now = Table.Column(data=grid.flatten()+x_noise, name='x')
        y_now = Table.Column(data=grid.flatten('F')+y_noise, name='x')

        f_now = Table.Column(data=[10 ** ((mag - instr.ZP_mag) / -2.5) * instr.ZP_flux] * (n_grid ** 2), name='Kmag')
        mag_now = Table.Column(data=[mag] * (n_grid ** 2), name='Kmag')
        name_now = Table.Column(data=['dummy_star'] * (n_grid ** 2), name='name')

        super(self.__class__, self).__init__(x_now, y_now, f_now, mag_now,
                                             name_now)

        return

class IrregularGrid(Scene):
    def __init__(self, n_grid, mag, instr):
        
        # Prepare a grid of positions
        img_size = 10 # Approximate size of the image (")
        row = np.delete(np.arange(-(img_size / 2), (img_size / 2), (img_size / (n_grid + 1))), 0)
        grid = np.asarray([row] * n_grid)

        avg_sep = img_size / n_grid
        rand_sep = avg_sep / 3.0

        np.random.seed(1001)
        
        x_tmp = np.random.uniform(low=-rand_sep, high=rand_sep, size=len(grid.flatten()))
        y_tmp = np.random.uniform(low=-rand_sep, high=rand_sep, size=len(grid.flatten()))
        
        x_now = Table.Column(data=grid.flatten() + x_tmp, name='x')
        y_now = Table.Column(data=grid.flatten('F') + y_tmp, name='y')
        f_now = Table.Column(data=[10 ** ((mag - instr.ZP_mag) / -2.5) * instr.ZP_flux] * (n_grid ** 2), name='Kmag')
        mag_now = Table.Column(data=[mag] * (n_grid ** 2), name='Kmag')
        name_now = Table.Column(data=['dummy_star'] * (n_grid ** 2), name='name')

        super(self.__class__, self).__init__(x_now, y_now, f_now, mag_now,
                                             name_now)

        return
    
class NIRC2(Instrument):
    def __init__(self):
        array_size = np.array((1024, 1024))
        readnoise = 60.0    # electrons
        dark_current = 0.1  # electrons / sec / pix
        gain = 4.0          # electrons per DN
        
        super(self.__class__, self).__init__(array_size, readnoise, dark_current, gain)

        self.scale = 0.009954   # "/pix
        self.tint = 2.8
        self.coadds = 10
        self.fowler = 8
        self.name = 'NIRC2'

        # Built in conversion that a 9th magnitude star should give roughly
        # 24000 DN on a NIRC2 central pixel in the K-band in a 2.8 sec exposure.
        # The zeropoint calculated is integrated (has an aperture correction to
        # go from the central pixel value based on Gunther's PSF grid...
        # it is approximate).
        self.aper_corr = 387.605 
        self.ZP_flux = (24000.0 * 4 / 2.8) * self.aper_corr
        self.ZP_mag = 9.0

        return


class PSF_grid_NIRC2_Kp(PSF_grid):
    """
    Container for a grid of NIRC2 PSFs. This function hosts
    all of the interpolation routines such that you can "get_psf"
    anywhere in the focal plane.
    """
    def __init__(self, psf, grid_points):

        """
        Load up a FITS file that contains at least 1, or possibly a grid
        of PSFs. These are stored and you can interpolate between them
        if necessary.
        """
        psf_scale = [0.009954]  # arcseconds per pixel
        
        # Fix wave_array to be a float
        wave_array=[2120]        
        wave_shape = 1

        if wave_shape != len(wave_array):
            print('Problem with PSF shape and wave_array shape')


        # Reshape the array to get the X and Y positions
        psf = psf.reshape((wave_shape, psf.shape[0],
                           psf.shape[1], psf.shape[2]))

        grid = grid_points.reshape((wave_shape, grid_points.shape[0], grid_points.shape[1]))
        #grid = np.swapaxes(grid, 1, 2)

        # Calculate the positions of all these PSFs. We assume that the
        # outermost PSFs are at the corners such that all observed stars
        # are internal to these corners. These are 1D arrays.
        x_pos = grid[0, :, 0]
        y_pos = grid[0, :, 1]

        self.psf = psf
        self.psf_x = x_pos   # 1D array
        self.psf_y = y_pos   # 1D array
        self.psf_wave = wave_array
        self.wave_shape = wave_shape
        self.psf_scale = psf_scale
        
        self.interpolator = [None for ii in self.psf_wave]

        return
        

    
def read_label_dat(label_file='/g/lu/data/gc/source_list/label.dat'):
    gcstars = Table.read(label_file, format='ascii')

    gcstars.rename_column('col1', 'name')
    gcstars.rename_column('col2', 'Kmag')
    gcstars.rename_column('col3', 'x')
    gcstars.rename_column('col4', 'y')
    gcstars.rename_column('col5', 'xerr')
    gcstars.rename_column('col6', 'yerr')
    gcstars.rename_column('col7', 'vx')
    gcstars.rename_column('col8', 'vy')
    gcstars.rename_column('col9', 'vxerr')
    gcstars.rename_column('col10', 'vyerr')
    gcstars.rename_column('col11', 't0')
    gcstars.rename_column('col12', 'use?')
    gcstars.rename_column('col13', 'r2d')

    return gcstars

def read_nirc2_psf_grid(psf_file, psf_grid_pos_file):
    print('Loading PSF grid from: ')
    print(psf_file)

    # Read in the PSF grid (single array of PSFs)
    psfs = pyfits.getdata(psf_file)

    # Read in the grid positions.
    psf_grid_pos = pyfits.getdata(psf_grid_pos_file)

    # Positify and normalize and the PSFs.
    for ii in range(psfs.shape[0]):
        # Repair them because there shouldn't be pixels with flux < 0.
        # To preserve the noise properties, just clip the values to be zero.
        psfs[ii][psfs[ii] < 0] = 0
            
        psfs[ii] /= psfs[ii].sum()

    return psfs, psf_grid_pos

def test_nirc2_img(psf_grid_raw, psf_grid_pos, outname='tmp.fits'):
    time_start = time.time()
    
    nirc2 = NIRC2()
    nirc2.coadds = 10 * 10   # 2.8 sec * 10 coadds * 10 images

    print( 'Reading GC Label.dat: {0} sec'.format(time.time() - time_start))
    stars = GCstars()

    psfgrid = PSF_grid_NIRC2_Kp(psf_grid_raw, psf_grid_pos)

    print( 'Making Image: {0} sec'.format(time.time() - time_start))
    wave_index = 0
    background = 3.0 # elect_nicerons /sec
    obs = Observation(nirc2, stars, psfgrid,
                      wave_index, background,
                      origin=np.array([631, 603]))
    
    print('Saving Image: {0} sec'.format(time.time() - time_start))

    obs.save_to_fits(outname, clobber=True)
    
    return


    
