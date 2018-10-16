
class Instrument(object):
    def __init__(self, array_size, readnoise, dark_current, gain):
        """
        array_size - in units of pixels (2D)
        readnoise - in units of electrons per read
        dark_current - in units of electrons per second per pixel
        gain - in units of electrons per DN
        """
        
        self.array_size = array_size
        self.readnoise = readnoise
        self.gain = gain
        self.dark_current = dark_current

        # Here are a bunch of default values setup.

        # Integration time in seconds
        self.tint = 1.0

        # Coadds
        self.coadds = 1

        # Fowler Samples for multi-CDS. This is the number of reads
        # in the beginning and repeated at the end.
        self.fowler = 1

        # Pixel Scale (arcsec / pixel)
        self.scale = 0.1 

        return


