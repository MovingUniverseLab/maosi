import numpy as np
from astropy.io import fits as pyfits
from astropy.table import Table
from scene import Scene
from instrument import Instrument
from observation import Observation
from psf import PSF_grid
import time
from glob import glob
import re
from os import makedirs, path
from scipy import ndimage
import pdb


class IRIS(Instrument):
    def __init__(self):
        array_size = np.array((4096, 4096))
        readnoise = 5  # electrons
        dark_current = 0.002  # electrons / sec / pix
        gain = 3.04  # electrons per DN

        super(self.__class__, self).__init__(array_size, readnoise, dark_current, gain)

        self.scale = 0.004  # "/pix
        self.offset = 0.6  # distance in " of the (1, 1) pixel from the optical axis
        self.tint = 30
        self.coadds = 1##########################
        self.fowler = 1####################3
        self.name = 'IRIS'

        # Built in conversion that a 9th magnitude star should give roughly
        # 24000 DN on a NIRC2 central pixel in the K-band in a 2.8 sec exposure.
        # The zeropoint calculated is integrated (has an aperture correction to
        # go from the central pixel value based on Gunther's PSF grid...
        # it is approximate).
        self.aper_corr = 387.605###################################3
        self.ZP_flux = (24000.0 * 4 / 2.8) * self.aper_corr
        self.ZP_mag = 9.0

        return

# class PSF_grid_NIRC2_Kp(PSF_grid):
#     """
#     Container for a grid of NIRC2 PSFs. This function hosts
#     all of the interpolation routines such that you can "get_psf"
#     anywhere in the focal plane.
#     """
#     def __init__(self, psf, grid_points):
# 
#         """
#         Load up a FITS file that contains at least 1, or possibly a grid
#         of PSFs. These are stored and you can interpolate between them
#         if necessary.
#         """
#         psf_scale = [0.009954]  # arcseconds per pixel
# 
#         # Fix wave_array to be a float
#         wave_array=[2120]        
#         wave_shape = 1
# 
#         if wave_shape != len(wave_array):
#             print('Problem with PSF shape and wave_array shape')
# 
# 
#         # Reshape the array to get the X and Y positions
#         psf = psf.reshape((wave_shape, int(grid_shape[0]), int(grid_shape[1]),
#                            psf.shape[1], psf.shape[2]))
# 
#         grid = grid_points.reshape((wave_shape, int(grid_shape[0]),
#                                     int(grid_shape[1]), grid_points.shape[1]))
#         #grid = np.swapaxes(grid, 1, 2)
# 
#         # Calculate the positions of all these PSFs. We assume that the
#         # outermost PSFs are at the corners such that all observed stars
#         # are internal to these corners. These are 1D arrays.
#         x_pos = grid[0, :, 0]
#         y_pos = grid[0, :, 1]
# 
#         self.psf = psf
#         self.psf_x = x_pos   # 2D array
#         self.psf_y = y_pos   # 2D array
#         self.psf_wave = wave_array
#         self.wave_shape = wave_shape
#         self.psf_scale = psf_scale
#         
#         self.interpolator = [None for ii in self.psf_wave]
# 
#         return

def convert_iris_psf_grid(psf_root, dir_out):

    # Read in the PSFs
    from scipy import ndimage
    xs = []
    ys = []
    psfs = []
    imgs = glob(psf_root + '*.fits')
    img = []

    for img in imgs:
        x_str = re.search('_x(.*)_y', img)
        xs.append(float(x_str.group(1)))
        y_str = re.search('_y(.*)_', img)
        ys.append(float(y_str.group(1)))
        psfs.append(pyfits.getdata(img))

    # Arrange the PSFs
    x_unique, x_unique_idx = np.unique(xs, return_index=True)
    y_unique, y_unique_idx = np.unique(ys, return_index=True)
    xs_order = []
    ys_order = []
    psfs_order = []

    for idx_y in y_unique_idx:

        for idx_x in x_unique_idx:
            xs_order.append(xs[idx_x])
            ys_order.append(ys[idx_y])
            idx_psf = np.where((np.array(xs) == xs[idx_x]) &
                               (np.array(ys) == ys[idx_y]))[0][0]
            psfs_order.append(psfs[idx_psf])

    # Convert the PSF positions
    iris_tmp = IRIS()
    xs_order = np.array(xs_order)
    ys_order = np.array(ys_order)
    xs_order -= iris_tmp.offset
    ys_order -= iris_tmp.offset
    xs_order /= iris_tmp.scale
    ys_order /= iris_tmp.scale
    pos_order = np.transpose(np.vstack((xs_order, ys_order)))
    
    # Convert the PSF images
    hdr = pyfits.getheader(img)
    comments = ''

    for i in range(len(hdr['comment'])):
        comment = hdr['comment'][i]
        comments += comment[2:]

    ps = float(re.search('PSF Sampling:(.*)arcsec', comments).group(1))

    # aaa = np.max(psfs_order[0])
    # aaa1 = np.sum(psfs_order[0])
    
    for i in range(len(psfs_order)):
        psfs_order[i] = ndimage.zoom(psfs_order[i], (ps / iris_tmp.scale),
                                     order=1) # Bilinear interpolation
        psfs_order[i] /= np.sum(psfs_order[i]) # Normalize to 1

    # bbb = np.max(psfs_order[0])
    # bbb1 = np.sum(psfs_order[0])

    # from matplotlib import pyplot
    # from matplotlib.colors import LogNorm
    # pyplot.figure(figsize=(6, 6))
    # pyplot.imshow(psfs_order[0],norm=LogNorm())

    # pyplot.figure(figsize=(6, 6))
    # pyplot.imshow(psfs_order[0], norm=LogNorm())

    # img_orig = pyfits.open(path.join(dir_out, 'old_psf_grid.fits'))[0].data
    # np.sum(img_orig[0])

    # Save the PSFs
    if not path.exists(dir_out):
        makedirs(dir_out)

    psfs_hdul = pyfits.HDUList([pyfits.PrimaryHDU(psfs_order)])
    psfs_hdul.writeto(path.join(dir_out, 'psf_grid.fits'), overwrite=True)
    pos_hdul = pyfits.HDUList([pyfits.PrimaryHDU(pos_order)])
    pos_hdul.writeto(path.join(dir_out, 'grid_pos.fits'), overwrite=True)
