import numpy as np
from prospect.models import transforms
import warnings
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# NB: all units of time here are yrs (not Gyr)


def create_sfh(model):
    """
    Create the appropriate type of SFH and return it.
    The input must be a Prospector model object
    """

    # non-parametric SFHs
    if 'agebins' in model.params:
        return NonparametricSFH(**model.params)

    # parametric SFHs
    else:

        if model.config_dict['sfh']['init'] == 1:
            return ExponentialSFH(**model.params)

        if model.config_dict['sfh']['init'] == 4:
            return DelayedExponentialSFH(**model.params)

    raise NotImplementedError("{} SFH not implemented yet".format(type))



class SFH:
    """
    Parent class for star formation histories
    """

    def __call__(self, t):
        """
        Return SFR (Msun/yr) corresponding to the lookback time t (in yrs)
        """

        return self.sfr(t)


    def __repr__(self):

        attrs = vars(self)
        return (type(self).__name__ + '(' + ', '.join("%s: %s" % item for item in attrs.items()) + ')')


    def mass_formed(self, t):
        """
        Return the total mass formed (in Msun) from the formation of the galaxy
        up to the lookback time t (in yrs)
        """

        # deal with arrays in input
        if hasattr(t, "__len__"):
            ret = np.zeros_like(t)
            for i in range(len(t)):
                ret[i] = self.mass_formed(t[i])
            return ret

        return quad(self.__call__, t, 14e9)[0]


    def ageform(self, x):
        """
        Return the lookback time (in yrs) at which x percent (from 0 to 100)
        of the final stellar mass was formed. For example, x=50 returns the median age
        """

        assert (x >= 0) & (x <= 100), 'Must provide a percentage value (0 < x < 100)'
        fn = lambda t: self.mass_formed(t)/self.mass_formed(0.0) - x/100.0
        return brentq(fn, 0, self.tage)


    def mean_age(self):
        """
        Return the mass-weighted age (in yrs)
        """

        fn = lambda t: self.__call__(t) * t / self.mass_formed(0.0)
        return quad(fn, 0, self.tage)[0]


    def mean_sfr(self, time=0):
        """
        Return the SFR averaged over the past time (in yrs).
        """

        if time == 0.0:
            return float(self.sfr(0))

        return float( (self.mass_formed(0) - self.mass_formed(time)) / time )


class NonparametricSFH(SFH):
    """
    Class describing a non-parametric star formation history.
    Note that an object of this class represents a specific instance of the
    model and has zero free parameters
    """


    def __init__(self, agebins, mass, **kwargs):
        """
        Initialize SFH from a prospector model, using the parameter values in model.params
        """

        # store relevant parameters
        self.logagebins = agebins
        self.mass = np.asarray(mass, float)
        self.tage = 10**np.max(self.logagebins)


    def bin_sfr(self):
        """
        Return SFH in the form of SFR (Msun/yr) in each agebin
        """

        agebins_yrs = 10**self.logagebins.T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        sfr = self.mass / dt
        return sfr


    def sfr(self, t):
        """
        Return SFR (Msun/yr) corresponding to the input lookback time t (in yrs, scalar or array)
        """

        time = np.asanyarray(t, dtype=float)

        # extend the youngest bin all the way to zero,
        # to make it easy when calculating the current SFR
        agebins_yrs = 10**self.logagebins
        youngest_ind = np.unravel_index(np.argmin(agebins_yrs, axis=None), agebins_yrs.shape)
        if agebins_yrs[youngest_ind] > 100e6:
            warnings.warn("Extending youngest bin from {} to 0 yrs".format(agebins_yrs[youngest_ind]))
        agebins_yrs[youngest_ind] = 0.0

        # make the list of conditions
        condlist = []
        for bin in agebins_yrs:
            condlist.append( (time >= bin[0]) & (time < bin[1])  )

        # make the list of functions
        # portions not covered by condlist are set to zero by default
        funclist = self.bin_sfr()

        return np.piecewise(time, condlist, funclist)


    def time_axis(self, epsilon = 1e-4):
        """
        Return the optimal look-back time axis, in years, for plotting the SFH
        epsilon controls the spacing between the end of a bin and the beginning of next bin
        """

        # code taken from FastStepBasis.convert_sfh
        agebins_yrs = 10**self.logagebins.T
        dt = agebins_yrs[1, :] - agebins_yrs[0, :]
        bin_edges = np.unique(agebins_yrs)
        maxage = agebins_yrs.max()
        t = np.concatenate((bin_edges * (1.-epsilon), bin_edges * (1+epsilon)))
        t.sort()
        t = t[1:-1] # remove older than oldest bin, younger than youngest bin
        return t


    def mass_formed(self, t):
        """
        Return total mass (in Msun) formed by lookback time t (in yrs)
        """

        # get the mass formed before a given time, and the corresponding time
        mass_formed_after = np.cumsum( np.concatenate(([0.0], self.mass)) )
        mass_formed_before = np.sum(self.mass) - mass_formed_after
        bins_old_edge = (10**self.logagebins.T)[1]

        # return a function that interpolates linearly
        xp = np.concatenate(([0.0], bins_old_edge) )
        yp = mass_formed_before
        return interp1d(xp, yp, bounds_error=False, fill_value=(0.0, 0.0) )(t)



class ExponentialSFH(SFH):
    """
    Class describing an exponential star formation history in prospector
    Note that an object of this class represents a specific instance of the
    model and has zero free parameters
    """


    def __init__(self, tage, tau, mass, **kwargs):
        """
        Initialize SFH from a prospector model, using the parameter values in model.params
        """

        # store relevant parameters
        self.tage = np.float(tage) * 1e9
        self.tau = np.float(tau) * 1e9
        self.mass = np.float(mass)


    def sfr(self, t):
        """
        Return SFR (Msun/yr) corresponding to the input lookback time t (in yrs, scalar or array)
        """

        time = np.asanyarray(t, dtype=float)

        # make the list of conditions
        condlist = [ (time >= 0) & (time <= self.tage) ]

        # make the list of functions
        # portions not covered by condlist are set to zero by default
        _tau = self.tau
        _t0 = self.tage
        constant = (self.mass / _tau) / (np.exp(_t0/_tau) - 1.0)
        funclist = [ lambda x: constant * np.exp( x / _tau ) ]

        return np.piecewise(time, condlist, funclist)


    def time_axis(self, N = 200):
        """
        Return the optimal look-back time axis, in years, for plotting the SFH
        N is the number of elements in the output array
        """

        return np.linspace(0.0, self.tage, N)


    def mass_formed(self, t):
        """
        Return total mass (in Msun) formed by lookback time t (in yrs)
        """

        time = np.asanyarray(t, dtype=float)
        _tau = self.tau
        _t0 = self.tage

        # make the list of conditions
        condlist = [ (time >= 0) & (time <= _t0) ]

        # make the list of functions
        # portions not covered by condlist are set to zero by default
        funclist = [ lambda x: self.mass * ( np.exp(_t0/_tau) - np.exp(x/_tau) ) /  ( np.exp(_t0/_tau) - 1.0 )  ]

        return np.piecewise(time, condlist, funclist)



class DelayedExponentialSFH(SFH):
    """
    Class describing a delayed exponential star formation history in prospector
    Note that an object of this class represents a specific instance of the
    model and has zero free parameters
    """


    def __init__(self, tage, tau, mass, **kwargs):
        """
        Initialize SFH from a prospector model, using the parameter values in model.params
        """

        # store relevant parameters
        self.tage = np.float(tage) * 1e9
        self.tau = np.float(tau) * 1e9
        self.mass = np.float(mass)


    def sfr(self, t):
        """
        Return SFR (Msun/yr) corresponding to the input lookback time t (in yrs, scalar or array)
        """

        time = np.asanyarray(t, dtype=float)

        # make the list of conditions
        condlist = [ (time >= 0) & (time <= self.tage) ]

        # make the list of functions
        # portions not covered by condlist are set to zero by default
        _tau = self.tau
        _t0 = self.tage
        constant = self.mass / ( (_t0 - _tau) * np.exp(_t0/_tau) + _tau )
        funclist = [ lambda x: constant * (x/_tau) * np.exp(x/_tau) ]

        return np.piecewise(time, condlist, funclist)


    def time_axis(self, N = 200):
        """
        Return the optimal look-back time axis, in years, for plotting the SFH
        N is the number of elements in the output array
        """

        return np.linspace(0.0, self.tage, N)


    def mass_formed(self, t):
        """
        Return total mass (in Msun) formed by lookback time t (in yrs)
        """

        time = np.asanyarray(t, dtype=float)
        _tau = self.tau
        _t0 = self.tage

        # make the list of conditions
        condlist = [ (time >= 0) & (time <= _t0) ]

        # make the list of functions
        # portions not covered by condlist are set to zero by default
        denominator = 1.0 + _tau  / (_t0 - _tau) * np.exp( - _t0 / _tau )
        funclist = [ lambda x: self.mass * ( 1.0 - (x - _tau) / (_t0 - _tau) * np.exp( (x - _t0) / _tau ) ) / denominator ]

        return np.piecewise(time, condlist, funclist)
