import numpy as np
import pylab as py
import pdb
import matplotlib.pyplot as plt

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


class SPISEAClusterScene(Scene):
    """
    Make a SPISEA cluster, distribute stars in a gaussian pattern on the sky
    to mimic a real star cluster. 
    """
    def __init__(self, cluster_fwhm=5.0, cluster_age=5e6, cluster_dist=4000, cluster_AKs=0.8,
                 cluster_mass=1e3):
        """
        Generate a SPISEA cluster with a Gaussian distribution of positions.

        Optional Parameters
        -------------------
        cluster_fwhm : float
            Radius of the cluster (std) in arcsec.

        cluster_age : float
            Age of the cluster in years.

        cluster_dist : float
            Distance of the cluster in parcsec.

        cluster_AKs : float
            AKs of the cluster in magnitudes.
        
        """
        self.cluster_age = cluster_age
        self.cluster_dist = cluster_dist
        self.cluster_AKs = cluster_AKs
        self.cluster_fwhm = cluster_fwhm
        self.cluster_mass = cluster_mass

        self.make_spisea_cluster()
        gdx = (self.cluster.star_systems['isWR'] == False)
        self.cluster.star_systems = self.cluster.star_systems[gdx]
        self.generate_star_positions_arcsec(cluster_fwhm)

        # Convert magnitudes to fluxes.
        self.cluster.star_systems['f_decam_r'] =  10**(-self.cluster.star_systems['m_decam_r']/2.5)
        self.cluster.star_systems['f_decam_r'] *= 4490 * 1.51e7 * 0.14 # convert from Jy to e-/s/m^2
        # see https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html

        self.xpos = self.cluster.star_systems['x']
        self.ypos = self.cluster.star_systems['y']
        self.flux = self.cluster.star_systems['f_decam_r']
        self.mag = self.cluster.star_systems['m_decam_r']
        self.name = np.array([f'star_{ii:04d}' for ii in range(len(self.cluster.star_systems))])

        return
    

    def make_spisea_cluster(self):
        from spisea import synthetic, evolution, atmospheres, reddening, ifmr
        from spisea.imf import imf, multiplicity

        # Define isochrone parameters
        logAge = np.log10(self.cluster_age)
        AKs = self.cluster_AKs # extinction in mags
        dist = self.cluster_dist # distance in parsec
        metallicity = 0 # Metallicity in [M/H]

        # Define evolution/atmosphere models and extinction law
        evo_model = evolution.MISTv1() 
        atm_func = atmospheres.get_merged_atmosphere
        red_law = reddening.RedLawHosek18b()

        # Also specify filters for synthetic photometry (optional). Here we use 
        # the HST WFC3-IR F127M, F139M, and F153M filters
        filt_list = ['decam,r', 'wfc3,ir,f153m']

        # Specify the directory we want the output isochrone
        # table saved in. If the directory does not already exist,
        # SPISEA will create it.
        iso_dir = 'isochrones/'

        # Make IsochronePhot object. Note that this will take a minute or two, 
        # unless the isochrone has been generated previously.
        #
        # Note that this is not show all of the user options 
        # for IsochronePhot. See docs for complete list of options.
        my_iso = synthetic.IsochronePhot(logAge, AKs, dist, metallicity=0,
                                         evo_model=evo_model, atm_func=atm_func,
                                         red_law=red_law, filters=filt_list,
                                         iso_dir=iso_dir)

        imf_multi = multiplicity.MultiplicityUnresolved()
        massLimits = np.array([0.2, 0.5, 1, 120])
        powers = np.array([-1.3, -2.3, -2.3])
        my_imf = imf.IMF_broken_powerlaw(massLimits, powers, imf_multi)

        # Define total cluster mass
        mass = self.cluster_mass

        # Make cluster object
        cluster = synthetic.ResolvedCluster(my_iso, my_imf, mass)

        self.cluster = cluster

        return

    def generate_star_positions_arcsec(self, cluster_fwhm):
        """
        Randomly draw x and y positions (in arcsec) for a cluster of stars
        in a 2D gaussian distribution. Positions are saved on the self.cluster object.
        """
        stars_x = []
        stars_y = []

        n_stars = len(self.cluster.star_systems)
        x = np.random.normal(0, cluster_fwhm, size=n_stars)
        y = np.random.normal(0, cluster_fwhm, size=n_stars)

        self.cluster.star_systems['x'] = x
        self.cluster.star_systems['y'] = y

        return
        

