
class Instrument(object):
    def __init__(self, array_size, readnoise, dark_current, gain, tel_diam, pix_scale):
        """
        array_size - in units of pixels (2D)
        readnoise - in units of electrons per read
        dark_current - in units of electrons per second per pixel
        gain - in units of electrons per DN
        tel_diam - telescope diameter in meters.
        pix_scale - pixel scale in arcsec/pixel.
        """
        
        self.array_size = array_size
        self.readnoise = readnoise
        self.gain = gain
        self.dark_current = dark_current
        self.tel_diam = tel_diam

        # Here are a bunch of default values setup.

        # Integration time in seconds
        self.tint = 1.0

        # Coadds
        self.coadds = 1

        # Fowler Samples for multi-CDS. This is the number of reads
        # in the beginning and repeated at the end.
        self.fowler = 1

        # Pixel Scale (arcsec / pixel)
        self.scale = pix_scale

        return


