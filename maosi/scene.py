
class Scene(object):
    """
    Allow the user to specify some scene consisting of a set of
    point sources with specifed positions in arcseconds and fluxes.
    """

    def __init__(self, stars_x, stars_y, stars_f, stars_m, stars_n):
        """
        X Position (in arcsec)
        Y Position (in arcsec)
        Flux (in electrons/sec)
        Magnitude
        Name
        
        Positions are at PA=0 with North up and East to the left.
        Positive x increases to the East.
        """
        self.xpos = stars_x
        self.ypos = stars_y
        self.flux = stars_f
        self.mag = stars_m
        self.name = stars_n

        return

