#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:54:24 2023

@author: wfw23
"""

import redback
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGenericTransient
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sne= 'SN2011kl'
data = redback.get_data.get_supernova_data_from_open_transient_catalog_data(sne)
frequencies= redback.utils.bands_to_frequency(data['band'].values)
snejoint= redback.transient.Afterglow(name=sne, flux_density=data['flux_density(mjy)'].values,
                                      time=data['time (days)'].values, data_mode='flux_density',
                                      flux_density_err=data['flux_density_error'].values, 
                                      frequency=frequencies)
ax1=snejoint.plot_data(show=False)
ax1.loglog()
plt.show()

model='tophat_and_arnett'
model_kwargs = dict(frequency=frequencies, output_format='flux_density')
priors = redback.priors.get_priors(model=model)
priors['redshift']=0.67

result = redback.fit_model(transient=snejoint, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=500, plot=False, resume=True)
ax=result.plot_lightcurve(show=False)
ax.loglog()
plt.legend(loc='lower left',bbox_to_anchor=(0, 0))
plt.show()