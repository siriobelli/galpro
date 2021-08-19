import prospect
from .sfh import create_sfh
from .utils import ivarsmooth
import numpy as np
import warnings
import sys
from os.path import realpath
from matplotlib import pyplot as plt
from astropy.cosmology import WMAP9 as cosmo
from matplotlib.ticker import FormatStrFormatter

# according to the Numpy version, default_rng may not be available
try:
    rng = np.random.default_rng(7936534)
    random_choice = rng.choice
except:
    warnings.warn('default_rng not found, using older numpy.random functions')
    np.random.seed(7936534)
    random_choice = np.random.choice



class ProspectorFit:
    """
    Class describing the results of one Prospector fit. It has methods to calculate statistical properties
    of the posterior distributions for each parameter (including derived parameters), to make plots, and to
    make statistical tests.
    """


    def __init__(self, h5_filename):
        """
        Initialize from h5 file written by Prospector
        """

        # read the Prospector output
        prosp_out = prospect.io.read_results.results_from(h5_filename, dangerous=True)

        # set basic attributes
        self.input_file = realpath(h5_filename)
        self.result = prosp_out[0]
        self.obs = prosp_out[1]
        self.model = prosp_out[2]
        self.sps = prospect.io.read_results.get_sps(self.result)

        # set lists of available parameters
        self.primary_parameters = []
        self.secondary_parameters = []
        self.fixed_parameters = []
        self.special_parameters = ['mfrac', 'logmass_surv', 'mean_sfr_x', 'ageform_x', 'ageform_x_y']

        # loop through the entire config_dict
        config = self.model.config_dict
        for parname in config:

            # determine the type of parameter
            if config[parname]['isfree']:
                appropriate_list = self.primary_parameters
            else:
                if config[parname].get('depends_on') == None:
                    appropriate_list = self.fixed_parameters
                else:
                    appropriate_list = self.secondary_parameters

            # add the parameter, expanding if it's a list
            if config[parname]['N'] == 1:
                appropriate_list.append(parname)
            else:
                for i in range(config[parname]['N']):
                    appropriate_list.append(parname+'_{0}'.format(i+1))

        # add special parameters
        if 'f_outlier_spec' in self.primary_parameters + self.secondary_parameters:
            self.special_parameters.append('N_outlier_spec')

        # set auxiliary attributes
        self.mfrac_chain = None


    def __repr__(self):
        return "{}('{}')".format(type(self).__name__, self.input_file)


    def __str__(self):

        ret = '\n{}\n'.format(self.__class__)
        ret += '\ncall: \n{}\n'.format(self.__repr__())
        ret += '\nattributes: \n' + ', '.join(list(vars(self).keys())) + '\n'
        ret += '\n\nmodel\n---------\n\n{}'.format(self.list_parameters())
        ret += '\nobs\n---------\n\n{}\n'.format(self.obs.keys())
        ret += '\nresult\n---------\n\n{}\n'.format(self.result.keys())
        ret += '\nsps\n---------\n\n{}\n'.format(self.sps.__class__)
        ret += '\nother\n---------\n\n'
        ret += 'mfrac_chain length: {}'.format(None if self.mfrac_chain is None else len(self.mfrac_chain))

        return ret



    def list_parameters(self):
        """
        Return string with available parameters, split by primary and derived parameters
        """

        ret = "Primary parameters: \n{} \n\nSecondary parameters: \n{} \n\nFixed parameters: \n{} \n\nSpecial parameters: \n{} \n".format(self.primary_parameters,
            self.secondary_parameters, self.fixed_parameters, self.special_parameters)
        return ret


    def parameter_statistic(self, parm_name, statistic, percentile=None, x=None, y=None, N_random=None):
        """
        Return the specified statistic for the specified parameter.
        For the statistic you can choose among 'MAP', 'bestfit', 'mean',
        'stddev', 'median', 'percentile'. If 'percentile', then you also need to
        provide a number between 0 and 100.
        Some special parameters, such as ageform_x, depend on one or two additional numbers, x and y.
        If N_random is set, only a subset of the chain is used (to make things faster).
        """

        # check that input is valid
        valid_statistics = ['MAP', 'bestfit', 'mean', 'stddev', 'median', 'percentile']
        if statistic not in valid_statistics:
            raise ValueError("'{}' not supported; valid statistics are {}".format(statistic, valid_statistics))
        if (percentile != None) & (statistic != 'percentile'):
            raise ValueError("percentile argument can be set only when statistic='percentile'")

        # get the posterior chain for this parameter
        chain, weights = self.parameter_chain(parm_name, x, y, N_random)

        if statistic == 'MAP':
            imax = np.argmax(self.result['lnprobability'])
            return chain[imax]

        if statistic == 'bestfit':
            imax = np.argmax(self.result['lnlikelihood'])
            return chain[imax]

        if statistic == 'mean':
            return np.average(chain, weights=weights)

        if statistic == 'stddev':
            average = np.average(chain, weights=weights)
            variance = np.average((chain-average)**2, weights=weights)
            return np.sqrt(variance)

        if statistic == 'median':
            return self.weighted_percentile(chain, weights, 50)

        if statistic == 'percentile':
            if percentile == None:
                raise ValueError("when statistic='percentile', percentile argument must also be provided")
            return self.weighted_percentile(chain, weights, percentile)


    def parameter_chain(self, parm_name, x=None, y=None, N_random=None):
        """
        Return the posterior chain for a given parameter and the corresponding weights.
        If N_random is set, only a subset of the chain is returned (to make things faster).
        Some special parameters, such as ageform_x, depend on one or two additional numbers, x and y
        """

        if N_random is None:
            return self.parameter_subchain(parm_name, x=x, y=y), self.result['weights']
        else:
            w_random = random_choice(len(self.result["chain"]), size=N_random, p=self.result["weights"])
            return self.parameter_subchain(parm_name, x=x, y=y, indices=w_random), self.result['weights'][w_random]



    def parameter_subchain(self, parm_name, x=None, y=None, indices=None):
        """
        Return a subset of the posterior chain for a given parameter. If indices == None, return the full chain.
        Some of the special parameters, such as ageform_x, depend on one or two additional numbers, x and y.
        """

        if indices is None:
            w_sub = range(len(self.result['weights']))
        else:
            w_sub = indices

        # initialize chain
        chain = np.zeros(len(w_sub))

        # if the parameter is primary, take the chain directly from the results
        if parm_name in self.primary_parameters:
            w_parm = np.array([p == parm_name for p in np.array(self.primary_parameters)], dtype=bool)
            chain = self.result['chain'][w_sub, w_parm].squeeze()
            return chain

        # if the parameter is secondary, make the chain using model.set_parameters
        if parm_name in self.secondary_parameters:
            for i, w in enumerate(w_sub):
                self.model.set_parameters(self.result['chain'][w])
                chain[i] = self.model.params[parm_name][0]
            return chain

        # if the parameter is fixed, there is not much of a chain
        if parm_name in self.fixed_parameters:
            warnings.warn("{} is a fixed parameter: chain elements are identical".format(parm_name))
            return chain + self.model.params[parm_name][0]

        # special cases that must be handled individually
        if parm_name in self.special_parameters:

            if parm_name == 'mfrac':
                if self.mfrac_chain is None:
                    self.set_mfrac()
                return self.mfrac_chain[w_sub]

            if parm_name == 'logmass_surv':
                return self.parameter_subchain('logmass', indices=w_sub) + np.log10(self.parameter_subchain('mfrac', indices=w_sub))

            if parm_name == 'mean_sfr_x':
                if x==None:
                    raise ValueError("mean_sfr_x requires the extra parameter 'x', the lookback time in years over which to average the SFR")
                for i, w in enumerate(w_sub):
                    self.model.set_parameters(self.result['chain'][w])
                    sfh = create_sfh(self.model)
                    chain[i] = sfh.mean_sfr(x)
                return chain

            if parm_name == 'ageform_x':
                if x==None:
                    raise ValueError("ageform_x requires the extra parameter 'x', the fraction of mass formed (0 < x < 100)")
                for i, w in enumerate(w_sub):
                    self.model.set_parameters(self.result['chain'][w])
                    sfh = create_sfh(self.model)
                    chain[i] = sfh.ageform(x)
                return chain

            if parm_name == 'ageform_x_y':
                if (x==None) | (y==None):
                    raise ValueError("ageform_x_y requires two extra parameters 'x' and 'y', "\
                        "the fraction of mass formed at the edges of the interval (0 < x < y < 100)")
                for i, w in enumerate(w_sub):
                    self.model.set_parameters(self.result['chain'][w])
                    sfh = create_sfh(self.model)
                    chain[i] = sfh.ageform(x) - sfh.ageform(y)
                return chain

            if parm_name == 'N_outlier_spec':
                w_used = np.where(self.obs['mask'] == True)[0]
                N_pix = len(w_used)
                return N_pix * self.parameter_subchain('f_outlier_spec', indices=w_sub)


        # if we make it until here, parm_name is not good
        self.list_parameters()
        raise ValueError("{} is not a valid parameter name.")


    def get_all_parameters(self):
        """
        Return a list with all available parameters, of all types
        """

        return self.primary_parameters + self.secondary_parameters + self.fixed_parameters + self.special_parameters


    def evidence(self):
        """
        Return Bayesian evidence as (logZ, dlogZ) calculated by dynesty
        """

        imax = np.argmax(self.result['lnprobability'])
        return self.result['logz'][imax], self.result['logzerr'][imax]


    def sfh_MAP(self):
        """
        Return MAP SFH (as an object)
        """

        imax = np.argmax(self.result['lnprobability'])
        theta = self.result['chain'][imax]
        self.model.set_parameters(theta)
        return create_sfh(self.model)


    def sfh_percentile(self, time, percentile=None):
        """
        Return the specified percentile of the sfh distribution
        at a lookback time (in yrs, array) as an array.
        'percentile' can be a list, in which case a list of arrays will be output
        """

        # we first need to evaluate the SFH for each step of the chain
        sfh_all = np.zeros((len(time), len(self.result['chain'])))
        for i, theta in enumerate(self.result['chain']):
            self.model.set_parameters(theta)
            sfh_fn = create_sfh(self.model)
            sfh_all[:,i] = sfh_fn(time)

        # now use the weights to calculate the percentile SFH
        if isinstance(percentile, list) == False:
            percentile_list = [percentile]
        else:
            percentile_list = percentile
        ret = []
        for p in percentile_list:
            sfh_out = np.zeros(len(time))
            weights = self.result['weights']
            for i_time in range(len(sfh_out)):
                sfh_out[i_time] = self.weighted_percentile(sfh_all[i_time,:], weights, p)
            ret.append(sfh_out)

        if isinstance(percentile, list) == False:
            return ret[0]
        else:
            return ret


    def modelspec_MAP(self, outwave=None, peraa=False):
        """
        Return Maximum A Posteriori spectrum as (lambda, flux). This is the
        model spectrum in the observed frame, with no calibration applied. A custom wavelength axis can be
        supplied (vacuum wavelengths). If peraa is True, return the spectrum in erg/s/cm^2/AA instead of AB
        maggies.
        """

        # set wavelength
        if outwave is None:
            outwave = self.sps.wavelengths.copy()

        # maximum a posteriori
        imax = np.argmax(self.result["lnprobability"])
        theta_max = self.result["chain"][imax, :]

        # set model parameters
        self.model.set_parameters(theta_max)

        # calculate flux
        flux = self.sps.get_spectrum(outwave=outwave, filters=None, peraa=peraa, **self.model.params)[0]

        return (outwave, flux)


    def modelspec_percentile(self, outwave, percentile, peraa=False, Nrandom = 500):
        """
        Return wavelength-wise percentile of the model spectrum. If percentile is
        a list [p1, p2, ...] then the output is a list of arrays: [flux1, flux2, ...].
        This is the model spectrum in the observed frame, with no calibration applied.
        A custom wavelength axis must be supplied (vacuum wavelengths). If peraa is True,
        return the spectrum in erg/s/cm^2/AA instead of AB maggies.
        Note: to speed up things, Nrandom (default: 500) steps in the chain are
        randomly drawn according to their weights, and used to construct the spectrum
        """

        # draw 500 random chain steps
        Nrandom = 500
        w_random = random_choice(len(self.result["chain"]), size=Nrandom, p=self.result["weights"])

        # calculate and store model spectra for the randomly-drawn chain steps
        spec_all = np.zeros([len(outwave), Nrandom])
        for i, theta in enumerate(self.result['chain'][w_random]):
            self.model.set_parameters(theta)
            spec = self.sps.get_spectrum(outwave=outwave, filters=None, peraa=peraa, **self.model.params)[0]
            spec_all[:,i] = spec

        # now calculate the percentile SFH
        if isinstance(percentile, list) == False:
            percentile_list = [percentile]
        else:
            percentile_list = percentile
        ret = []
        for p in percentile_list:
            ret.append( np.percentile(spec_all, p, axis=1) )

        if isinstance(percentile, list) == False:
            return ret[0]
        else:
            return ret


    def phot_MAP(self):
        """
        Return Maximum A Posteriori photometry, corresponding to all the filters in obs,
        including the ones that are masked.
        """

        # maximum a posteriori
        imax = np.argmax(self.result["lnprobability"])
        theta_max = self.result["chain"][imax, :]

        # get the model photometry
        _, phot, _ = self.model.predict(theta_max, obs=self.obs, sps=self.sps)

        return phot


    def calibspec_MAP(self):
        """
        Return Maximum A Posteriori observed-frame, calibrated spectrum as (lambda, flux)
        """

        # maximum a posteriori
        imax = np.argmax(self.result["lnprobability"])
        theta_max = self.result["chain"][imax, :]

        # get the calibrated spectrum
        flux, _, _ = self.model.predict(theta_max, obs=self.obs, sps=self.sps)

        # get the observed wavelengths
        wave = self.obs['wavelength'].copy()

        # pixels outside the observed range make no sense, due to the polynomial correction
        wave_masked = wave[np.where(self.obs['mask'] == True)]
        w_outrange = np.where( (wave < np.min(wave_masked)) | (wave > np.max(wave_masked)) )[0]
        flux[w_outrange] = np.nan

        return (wave, flux)


    def calibspec_percentile(self, percentile, Nrandom = 500):
        """
        Return wavelength-wise percentile of the calibrated model spectrum. If percentile is
        a list [p1, p2, ...] then the output is a list of arrays: [flux1, flux2, ...].
        Note: to speed up things, Nrandom (default: 500) steps in the chain are
        randomly drawn according to their weights, and used to construct the spectrum
        """

        # draw 500 random chain steps
        Nrandom = 500
        w_random = random_choice(len(self.result["chain"]), size=Nrandom, p=self.result["weights"])

        # pixels outside the observed range make no sense, due to the polynomial correction
        wave = self.obs['wavelength'].copy()
        wave_masked = wave[np.where(self.obs['mask'] == True)]
        w_outrange = np.where( (wave < np.min(wave_masked)) | (wave > np.max(wave_masked)) )[0]

        # calculate and store calibrated model spectra for the randomly-drawn chain steps
        spec_all = np.zeros([len(self.obs['wavelength']), Nrandom])
        for i, theta in enumerate(self.result['chain'][w_random]):
            spec, _, _ = self.model.predict(theta, obs=self.obs, sps=self.sps)
            spec[w_outrange] = np.nan
            spec_all[:,i] = spec

        # now calculate the percentile SFH
        if isinstance(percentile, list) == False:
            percentile_list = [percentile]
        else:
            percentile_list = percentile
        ret = []
        for p in percentile_list:
            ret.append( np.nanpercentile(spec_all, p, axis=1) )

        if isinstance(percentile, list) == False:
            return ret[0]
        else:
            return ret

    def spec_calibration(self, peraa=False):
        """
        Return polynomial correction that must be applied (by multiplication) to
        the MAP spectrum in order to reproduce the observed spectrum.
        If peraa is True, return the calibration in erg/s/cm^2/AA instead of AB
        maggies.
        """

        # set the MAP parameters
        imax = np.argmax(self.result['lnprobability'])
        theta = self.result['chain'][imax]
        self.model.set_parameters(theta)

        # get the calibration
        wave, modelspec = self.modelspec_MAP(outwave=self.obs['wavelength'], peraa=peraa)
        calibration = self.model.spec_calibration(obs=self.obs, spec=modelspec)

        # determine valid wavelength range of spectroscopic data
        # (outside this range, the polynomial used to fit the spectral shape makes no sense)
        wave_masked = self.obs['wavelength'][np.where(self.obs["mask"] == True)]
        wavemin, wavemax = np.nanmin(wave_masked), np.nanmax(wave_masked)
        calibration[(wave < wavemin) | (wave > wavemax)] = np.nan

        return calibration


    def chisquare_spec(self, reduced=True):
        """
        Return the chi-square for the spectrum, calculated as the sum of the square of the MAP residuals.
        If reduced=True, return the chi-square divided by the number of spectral pixels used in the fit.
        """

        # get MAP spectroscopy
        _, spec_map = self.calibspec_MAP()

        # select pixels used in the fit
        w_used = np.where(self.obs['mask'] == True)[0]

        # calculate spectroscopic residuals
        res = (self.obs['spectrum'] - spec_map)/self.obs['unc']

        # return chi-square
        chisquare = np.nansum( res[w_used]**2 )
        if reduced == True:
            return chisquare/len(w_used)
        else:
            return chisquare


    def chisquare_phot(self, reduced=True):
        """
        Return the chi-square for the photometry, calculated as the sum of the square of the MAP residuals.
        If reduced=True, return the chi-square divided by the number of photometric points used in the fit.
        """

        # get MAP photometry
        phot_map = self.phot_MAP()

        # select photometry used in the fit
        w_used = np.where(self.obs["phot_mask"] == True)[0]

        # calculate photometric residuals
        res = (self.obs["maggies"] - phot_map)/self.obs["maggies_unc"]

        # return chi-square
        chisquare = np.nansum( res[w_used]**2 )
        if reduced == True:
            return chisquare/len(w_used)
        else:
            return chisquare


    def set_mfrac(self):
        """
        Calculate and store the fraction of the total stellar mass that is still in stars,
        for all the steps of the chain
        """

        barLength = 10 # Modify this to change the length of the progress bar
        print('Calculating mfrac, could take a while...', flush=True)

        Nchain = self.result['chain'].shape[0]
        self.mfrac_chain = np.zeros(Nchain)

        for i in range(Nchain):
            _, _, _mfrac = self.model.predict(self.result['chain'][i,:], self.obs, sps=self.sps)
            self.mfrac_chain[i] = _mfrac
            progress = i/(Nchain-1)
            block = int(round(barLength*progress))
            text = "\rPercent: [{}] {:.2f}%".format( "#"*block + "-"*(barLength-block), progress*100)
            sys.stdout.write(text)
            sys.stdout.flush()


    def plot_sfh(self, ax, percentile_range=95):
        """
        Plot the SFH in the provided axis object
        percentile_range sets the width of the distribution used to calculate
        the shaded area
        """

        #time = self.sfh_MAP().time_axis()
        tuniv = cosmo.age(self.parameter_statistic('zred', 'median')).value * 1e9
        time = np.linspace(0.0, tuniv, 500)

        sfh_map = self.sfh_MAP()(time)
        sfh_lo, sfh_median, sfh_hi = self.sfh_percentile(time, [50-0.5*percentile_range, 50, 50+0.5*percentile_range])

        ax.fill_between(time*1e-9, sfh_lo, sfh_hi, color='red', alpha=0.2, linewidth=0, label='Central {:.0f}%'.format(percentile_range))
        ax.plot(time*1e-9, sfh_median, color='red', lw=2, alpha=0.7, zorder=10, label='Median SFH')
        ax.plot(time*1e-9, sfh_map, color='blue', lw=2, alpha=0.7, zorder=10, label='MAP SFH')

        # prettify
        ax.set(xlabel = 'Lookback time (Gyr)',
               ylabel = r'SFR ($M_\odot$/yr)',
               yscale='log',)
               #title = 'Reconstructed Star Formation History',
               #ylim=[0,2])
        ax.legend()


    def plot_posteriors(self, fig, gridspec=None, ncols=10, show_prior=True, hspace=0.8):
        """
        Plot the posterior distribution for all parameters
        in the provided figure object (optionally, gridspec too)
        """

        # get full list of free parameters
        parnames = np.array(self.result.get('theta_labels', self.model.theta_labels()))

        # create (sub)gridspec
        nrows = int(np.ceil( len(parnames) / ncols ))
        if gridspec is None:
            posterior_grid = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=hspace)
        else:
            posterior_grid = gridspec.subgridspec(nrows=nrows, ncols=ncols, hspace=hspace)

        # create dictionary of parameter names, using their prospector names
        dic_names = {}
        for p in parnames:
            dic_names[p] = p

        # dictionary of 'good', nicely formatted names
        dic_goodnames = {
            'zred': r'$z$',
            'logzsol': r'log $Z/Z_\odot$',
            'sigma_smooth': r'$\sigma_\ast$ (km/s)',
            'logmass': r'log $\tilde M/M_\odot$',
            'mass': r'$\tilde M \ (10^{10} M_\odot)$',
            'dust2': r'$A_V$',
            'dust_index': r'$n$',
            'dust1_fraction': r'$A_\mathrm{v,birth} / A_\mathrm{v} $',
            'duste_qpah': r'$Q_\mathrm{PAH}$',
            'duste_gamma': r'$\gamma$',
            'duste_umin': r'$U_{min}$',
            'spec_norm': r'$N_S$',
            'f_outlier_spec': r'$O_S$',
            'spec_jitter': r'$J_S$',
            'tage': r'$t_0$ (Gyr)',
            'tau': r'$\tau$ (Gyr)',
        }

        # for non-parametric SFH, add nicely formatted bin names
        if 'agebins' in self.model.params:
            bin_edges = ['{:.2g}'.format(10**(t[1]-9)) for t in self.model.params['agebins']]
            for i in range(len(bin_edges)-1):
                dic_goodnames['logsfr_ratios_' +str(i+1)] = r'log $R_{' + bin_edges[i] + '}$'

        # now take the good names when present, otherwise leave the prospector one
        for k in dic_names.keys():
            if k in dic_goodnames:
                dic_names[k] = dic_goodnames[k]

        # loop through the list of parameters and plot a histogram for each
        for i, par_name in enumerate(parnames):

            # gridspec index for this panel
            ind = np.unravel_index(i, (nrows, ncols), order='C')
            ax_par = fig.add_subplot(posterior_grid[ind])

            # store bounding box for later
            if i==0:
                bb_par = ax_par.get_position()

            # set scaling for this parameter (this should be consistent
            # with the labels defined above)
            scaling = 1.0
            if par_name == 'mass':
                scaling = 1e-10

            # get the prior for this parameter
            if 'logsfr_ratios' in par_name:
                prior = lambda x: self.model.config_dict['logsfr_ratios']['prior'](x).T[0]
                nd_range = self.model.config_dict['logsfr_ratios']['prior'].range
                prior_range = (round(nd_range[0][0]), round(nd_range[1][0]))
                if prior_range[0] == prior_range[1]:
                    prior_range = (nd_range[0][0], nd_range[1][0])
                #prior_range = np.array([-4,4])
            else:
                prior = self.model.config_dict[par_name]['prior']
                prior_range = np.array(prior.range) * scaling


            # show posterior
            ax_par.hist(self.parameter_chain(par_name) * scaling, weights=self.result['weights'], range=prior_range, bins=30, histtype='stepfilled', align='mid', color='forestgreen')

            # show prior
            if show_prior == True:
                x = np.linspace(prior_range[0], prior_range[1], 100)
                normalize = 0.75 * ax_par.get_ylim()[1] / np.nanmax(np.exp(prior(x/scaling)))
                ax_par.plot(x, normalize*np.exp(prior(x/scaling)), color='purple')

            # prettify
            ax_par.set(xlim=prior_range, xlabel=dic_names[par_name],
                       yticklabels=[], yticks=[], xticks=[prior_range[0], prior_range[1]])
            ax_par.tick_params(labelsize=10)
            if par_name=='zred':
                ax_par.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax_par.get_xticklabels()[0].set_ha('left')
            ax_par.get_xticklabels()[1].set_ha('right')

            # confidence interval in the title
            value_median = self.parameter_statistic(par_name, 'median')*scaling
            value_84 = self.parameter_statistic(par_name, 'percentile', 84)*scaling
            value_16 = self.parameter_statistic(par_name, 'percentile', 16)*scaling
            ax_par.set_title(r'{:.2f} $\pm$ {:.2f}'.format(value_median, 0.5*(value_84-value_16)), fontsize=10)


    def plot_sed(self, ax, residuals=False, show_percentiles=False, show_filters=False):
        """
        Plot the photometry and the model in the provided axis object.
        If residuals is True, plot the residuals instead.
        """

        obs = self.obs

        # select photometry used in the fit
        w_used = np.where(obs["phot_mask"] == True)[0]
        w_unused = np.where(obs["phot_mask"] == False)[0]

        # get observed-frame wavelength axis with wide coverage
        redshift = self.parameter_statistic('zred', 'MAP')
        wide_wavelength = self.sps.wavelengths * (1.0 + redshift)

        # get MAP photometry
        phot_map = self.phot_MAP()

        if residuals==True:

            # plot the residuals
            res = (obs["maggies"] - phot_map)/obs["maggies_unc"]
            ax.errorbar(obs["phot_wave"][w_used], res[w_used], label='Residual',
                     marker='.', markersize=10, alpha=0.8, ls='', lw=3,
                     markerfacecolor='none', markeredgecolor='black',
                     markeredgewidth=3)
            ax.axhline(y=0)

            # plot settings
            ylabel = 'Residuals (sigma)'
            ymin, ymax = np.nanmin(res[w_used]), np.nanmax(res[w_used])
            ymin -= 0.1*(ymax-ymin)
            ymax += 0.1*(ymax-ymin)

        else:

            # plot model spectrum
            modelspec_wave, modelspec_map = self.modelspec_MAP(peraa=False)
            if show_percentiles==True:
                modelspec_lo, modelspec_hi = self.modelspec_percentile(modelspec_wave, [2.5, 97.5], peraa=False)
                ax.fill_between(modelspec_wave, modelspec_lo, modelspec_hi, color='red', alpha=0.2, linewidth=0, label='Central {:.0f}%'.format(95.0))
            ax.loglog(modelspec_wave, modelspec_map, label='Model spectrum (MAP)',
                   lw=0.7, color='red', alpha=0.7)

            # plot MAP photometry
            ax.errorbar(obs["phot_wave"], phot_map, label='Model photometry (MAP)',
                     marker='s', markersize=10, alpha=0.8, ls='', lw=3,
                     markerfacecolor='none', markeredgecolor='red',
                     markeredgewidth=3)

            # plot observed photometry
            ax.errorbar(obs["phot_wave"][w_used], obs['maggies'][w_used], yerr=obs['maggies_unc'][w_used],
                     label='Observed photometry', ecolor='green',
                     marker='o', markersize=10, ls='', lw=3, alpha=0.8,
                     markerfacecolor='none', markeredgecolor='green',
                     markeredgewidth=3)
            if len(w_unused) > 0:
                ax.errorbar(obs["phot_wave"][w_unused], obs['maggies'][w_unused], yerr=obs['maggies_unc'][w_unused],
                         label='Observed photometry (not used)', ecolor='limegreen',
                         marker='o', markersize=10, ls='', lw=3, alpha=0.6,
                         markerfacecolor='none', markeredgecolor='limegreen',
                         markeredgewidth=3)

            # plot transmission curves
            if show_filters==True:
                for f in obs['filters']:
                    w, t = f.wavelength.copy(), f.transmission.copy()
                    t = t / t.max()
                    t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
                    ax.loglog(w, t, lw=2, alpha=0.7)

            # plot settings
            chi0 = self.chisquare_phot(reduced=True)
            ax.legend(loc='best', fontsize=10, title=r'$\chi^2 / N_\mathrm{data}$' + ' = {:.2f}'.format(chi0))
            ylabel = 'Flux Density [maggies]'
            w_detected = np.where(obs['maggies']/obs['maggies_unc'] > 2.0)
            ymin, ymax = np.nanmin(obs['maggies'][w_detected])*0.8, np.nanmax(obs['maggies'][w_detected])/0.4

        # prettify
        xmin, xmax = np.min(obs["phot_wave"])*0.8, np.max(obs["phot_wave"])/0.8
        ax.set(xlabel = r'Observed wavelength ($\AA$)',
               ylabel = ylabel,
               xscale = 'log',
               xlim = [xmin, xmax],
               ylim = [ymin, ymax])


    def plot_spectrum(self, ax, smooth=True, show_calibration=True, residuals=False, show_percentiles=False, physical_units=False, peraa=False):
        """
        Plot the spectrum and the model in the provided axis object.
        If residuals is True, plot the residuals instead.
        If there is no spectroscopy in the data, show the models only, in the
        rest-frame optical (around 4000 A); the other keywords are ignored
        """

        # if there is no spectroscopy, zoom in on the 4000A region
        # and show the models
        if self.obs.get("spectrum") is None:

            # only show modelspec
            zred = self.parameter_statistic('zred', 'MAP')
            xmin, xmax = (1.0+zred) * np.array([3600, 5300])
            model_wave = np.linspace(xmin, xmax, 500)
            model_wave, model_flux = self.modelspec_MAP(model_wave)
            plt.plot(model_wave, model_flux, label='Model spectrum (MAP)', lw=1.6, color='red', alpha=0.7, zorder=6)

            if show_percentiles==True:
                model_lo, model_hi = self.modelspec_percentile(model_wave, [2.5, 97.5], peraa=False)
                ax.fill_between(model_wave, model_lo, model_hi, color='red', alpha=0.2, linewidth=0, label='Central {:.0f}%'.format(95.0), zorder=1)

            # prettify
            w_toplot = np.where( (model_wave > xmin) & (model_wave < xmax) )[0]
            ymin, ymax = np.nanmin(model_flux[w_toplot])*0.9, np.nanmax(model_flux[w_toplot])/0.9
            ax.set(xlabel = r'Observed wavelength ($\AA$)',
                ylabel = 'Flux Density [maggies]',
                xlim = [xmin, xmax],
                ylim = [ymin, ymax])
            ax.legend(loc='best', fontsize=10)

            return

        # get observed and model spectra
        obs_wave, obs_flux, obs_err = self.obs['wavelength'].copy(), self.obs['spectrum'].copy(), self.obs['unc'].copy()
        map_wave, map_flux = self.calibspec_MAP()
        if show_percentiles == True:
            lo_flux, hi_flux = self.calibspec_percentile([2.5, 97.5])
        else:
            lo_flux, hi_flux = np.full(len(obs_wave), np.nan), np.full(len(obs_wave), np.nan)

        # convert to physical units
        if physical_units == True:
            calib = self.spec_calibration(peraa=peraa)
            obs_flux /= calib
            obs_err /= calib
            map_flux /= calib
            lo_flux /= calib
            hi_flux /= calib
            if peraa==True:
                ylabel = r'Flux Density (erg $s^{-1}$ cm$^{-2}$ $\AA^{-1}$)'
            else:
                ylabel = 'Flux Density (maggies)'
        else:
            ylabel = 'Flux Density [same units as observations]'

        # smooth spectra
        if (residuals == False) & (smooth != False):

            # default value: velocity dispersion
            if smooth is True:
                smooth_sigma = self.parameter_statistic('sigma_smooth', 'MAP')
                pixel_vel = np.median((self.obs['wavelength'][1:]-self.obs['wavelength'][:-1])/self.obs['wavelength'][1:] * 3e5)
                smooth_width = int(smooth_sigma / pixel_vel)
            else:
                smooth_width = int(smooth)

            obs_flux = ivarsmooth(obs_flux, 1.0/self.obs['unc']**2, smooth_width)
            map_flux = ivarsmooth(map_flux, np.ones(len(map_flux)), smooth_width)
            lo_flux = ivarsmooth(lo_flux, np.ones(len(map_flux)), smooth_width)
            hi_flux = ivarsmooth(hi_flux, np.ones(len(map_flux)), smooth_width)

        # parts of observed spectrum actually used in the fit
        obs_flux_masked = obs_flux.copy()
        obs_flux_masked[np.where(self.obs["mask"] == False)] = np.NaN

        if residuals == True:

            # flux is now the residual flux
            obs_flux = (obs_flux - map_flux) / obs_err
            obs_flux_masked = (obs_flux_masked - map_flux) / obs_err
            ax.plot(obs_wave, obs_flux, 'o', color="skyblue", alpha=0.3, label='Observed spectrum (not used)', zorder=1)
            ax.plot(obs_wave, obs_flux_masked, 'o', color="royalblue", alpha=0.5, label='Observed spectrum', zorder=5)
            ylabel = 'Residuals (sigma)'

        else:

            # plot all spectra
            ax.plot(obs_wave, obs_flux, color="skyblue", label='Observed spectrum (not used)', zorder=1)
            ax.plot(obs_wave, obs_flux_masked, color="royalblue", label='Observed spectrum', zorder=2)
            ax.plot(map_wave, map_flux, label='Model spectrum (MAP)', lw=1.6, color='red', alpha=0.7, zorder=6)
            if show_percentiles == True:
                ax.fill_between(obs_wave, lo_flux, hi_flux, color='red', alpha=0.2, linewidth=0, label='Central {:.0f}%'.format(95.0))

            # plot polynomial used for calibration
            if show_calibration == True:
                cal = self.spec_calibration()
                cal *= np.nanmedian(obs_flux) / np.nanmedian(cal)
                ax.plot(obs_wave, cal, label='Calibration (scaled)', lw=0.8, color='green', alpha=0.5, zorder=1)

            chi0 = self.chisquare_spec(reduced=True)
            ax.legend(loc='best', fontsize=10, title=r'$\chi^2 / N_\mathrm{data}$' + ' = {:.2f}'.format(chi0))

        # establish bounds
        wave_masked = self.obs['wavelength'][np.where(self.obs["mask"] == True)]
        xrange = np.nanmax(wave_masked) - np.nanmin(wave_masked)
        xmin, xmax = np.nanmin(wave_masked) - 0.03*xrange, np.nanmax(wave_masked) + 0.03*xrange
        ymin, ymax = np.nanmin(obs_flux_masked), np.nanmax(obs_flux_masked)
        h = ymax-ymin
        ymin, ymax = ymin - 0.1*h, ymax + 0.1*h

        # prettify
        ax.axhline(0, color='black', lw=0.7, alpha=0.8, zorder=6)
        ax.set(xlabel = r'Observed wavelength ($\AA$)',
               ylabel = ylabel,
               xlim = [xmin, xmax],
               ylim = [ymin, ymax])


    def summary_figure(self, filename=None):
        """
        Make gigantic figure with SED, spectrum, posteriors, and SFH
        """

        # set up posterior grid
        N_par = len(np.array(self.result.get('theta_labels', self.model.theta_labels())))
        N_par_cols = 6
        N_par_rows = np.ceil(N_par / N_par_cols)

        # set up whole figure
        if self.obs.get("spectrum") is None:
            height_inch = np.array([8, 4, 6, 2*N_par_rows])
        else:
            height_inch = np.array([8, 8, 4, 4, 6, 2*N_par_rows])
        width_inch = 8.5
        fig = plt.figure( figsize=(width_inch, np.sum(height_inch)) )
        outer_grid = fig.add_gridspec(nrows=len(height_inch), ncols=1, height_ratios=height_inch, hspace=0.3, left=0.1, right=0.95, bottom=0.03, top=0.98)

        # set up grids
        grid_sed = outer_grid[0].subgridspec(nrows=2, ncols=1, hspace=0.0, height_ratios=[2,1])
        if self.obs.get("spectrum") is None:
            grid_spec = outer_grid[1].subgridspec(nrows=1, ncols=1)
            grid_sfh = outer_grid[2].subgridspec(nrows=2, ncols=1)
            grid_posteriors = outer_grid[3]
        else:
            grid_spec1 = outer_grid[1].subgridspec(nrows=2, ncols=1, hspace=0.0, height_ratios=[2,1])
            grid_spec2 = outer_grid[2].subgridspec(nrows=1, ncols=1)
            grid_spec3 = outer_grid[3].subgridspec(nrows=1, ncols=1)
            grid_sfh = outer_grid[4].subgridspec(nrows=2, ncols=1)
            grid_posteriors = outer_grid[5]

        # SED
        #######
        ax_sed = fig.add_subplot(grid_sed[0])
        ax_sed_res = fig.add_subplot(grid_sed[1])

        self.plot_sed(ax_sed, show_percentiles=True)
        ax_sed.set_xlabel('')
        self.plot_sed(ax_sed_res, residuals=True)

        if self.obs.get("spectrum") is None:

            # MAP spectrum in physical units
            #############################
            ax_spec = fig.add_subplot(grid_spec[0])
            self.plot_spectrum(ax_spec, show_percentiles=True, smooth=False, physical_units=True, show_calibration=False)

        else:

            # Spectrum in physical units
            #############################
            ax_spec1 = fig.add_subplot(grid_spec1[0])
            ax_spec1_res = fig.add_subplot(grid_spec1[1])

            self.plot_spectrum(ax_spec1, show_percentiles=True, smooth=True, physical_units=True, show_calibration=False)
            ax_spec1.set(title='Smoothed, calibrated spectrum', xlabel='')
            self.plot_spectrum(ax_spec1_res, residuals=True)

            # Spectrum in observed units
            #############################
            ax_spec2 = fig.add_subplot(grid_spec2[0])
            self.plot_spectrum(ax_spec2, show_percentiles=False, smooth=True, physical_units=False, show_calibration=True)
            ax_spec2.set(title='Smoothed, observed spectrum')
            ax_spec2.legend(title='')

            # Spectrum in observed units, not smoothed
            #############################
            ax_spec3 = fig.add_subplot(grid_spec3[0])
            self.plot_spectrum(ax_spec3, show_percentiles=False, smooth=False, physical_units=False, show_calibration=True)
            ax_spec3.set(title='Unsmoothed, observed spectrum')
            ax_spec3.legend().set_visible(False)

        # SFH
        #############################
        ax_sfh1 = fig.add_subplot(grid_sfh[0])
        ax_sfh2 = fig.add_subplot(grid_sfh[1])
        self.plot_sfh(ax_sfh1)
        ax_sfh1.set(yscale='linear', xlabel='')
        self.plot_sfh(ax_sfh2)
        ax_sfh2.legend().set_visible(False)
        if ax_sfh2.get_ylim()[0] < 1e-4:
            ax_sfh2.set_ylim(bottom=1e-4)

        # Posteriors
        #############################

        self.plot_posteriors(fig, grid_posteriors, ncols=N_par_cols, show_prior=True, hspace=1.2)

        # write to file
        if filename != None:
            fig.savefig(filename)


    @staticmethod
    def weighted_percentile(values, weights, percentile):
        """
        Calculate weighted percentile for an array of values with associated weights.
        Percentile must be a number between 0 and 100.
        """

        assert len(values) == len(weights), "values and weights arrays must have identical length"
        assert (percentile >= 0) and (percentile <= 100), "percentile must be a number between 0 and 100"

        # sort by the parameter values
        w_sort = np.argsort(np.asarray(values))
        values_sorted = np.asarray(values)[w_sort]
        weights_sorted = np.asarray(weights)[w_sort]

        # for each value, calculate its weighted quantile, normalized between 0 and 1
        running_quantiles = np.cumsum(weights_sorted) - 0.5 * weights_sorted
        running_quantiles -= running_quantiles[0]
        running_quantiles /= running_quantiles[-1]

        # use linear interpolation to find the exact percentile
        return np.interp(percentile/100.0, running_quantiles, values_sorted)



    @staticmethod
    def dust_attenuation(lam_input, dust1, dust2, dust_index, **extra):
        """
        Return F(obs) / F(emitted) for Kriek & Conroy law, as a function
        of rest-frame wavelength (in angstrom). Adapted from Joel.
        It includes both the diffuse and birth-cloud components; if you want the
        diffuse component only, set dust1=0
        """

        # sanitize inputs
        lam = np.atleast_1d(lam_input).astype(float)

        dd63 = 6300.00
        lamv = 5500.0
        dlam = 350.0
        lamuvb = 2175.0

        #Calzetti curve, below 6300 Angstroms, else no addition
        cal00 = np.zeros_like(lam)
        gt_dd63 = lam > dd63
        le_dd63 = ~gt_dd63
        if np.sum(gt_dd63) > 0:
            cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
        if np.sum(le_dd63) > 0:
            cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-0.198*(1E4/lam[le_dd63])**2 + 0.011*(1E4/lam[le_dd63])**3) + 1.78
        cal00 = cal00/0.44/4.05
        eb = 0.85 - 1.9 * dust_index  #KC13 Eqn 3
        #Drude profile for 2175A bump
        drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )
        attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust_index

        # birth-cloud component
        dust1_curve = dust1*(lam/lamv)**(-1.0)

        return np.exp(-attn_curve-dust1_curve)
