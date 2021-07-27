
# GalPro
A package for analyzing galaxies using [Prospector](https://github.com/bd-j/prospector).

The main class is ProspectorFit

    # create prospectorfit object
    from galpro import ProspectorFit 
    pf = ProspectorFit('prospector_output.h5')
    
    # print properties of the fit 
    print(pf)

    # check available methods
    help(pf)
    
    # print out Maximum-A-Posterior value of metallicity logZ/Zsun 
    print(pf.parameter_statistic('logzsol', 'MAP'))
    
    # plot observed SED and model
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    pf.plot_sed(ax)
    
    # make giant summary plot and save to file
    pf.summary_figure('summary_figure.pdf')


There is also a SFH (star formation history) class

    # get the Maximum-A-Posteriori star formation history and print its properties
    sfh = pf.sfh_MAP()
    print(sfh)
    
    # check available SFH methods
    help(sfh)
    
    # make a plot of the SFH
    t = sfh.time_axis()
    plt.plot(t, sfh(t))

    # calculate median age (in years)
    print(sfh.ageform(50))
