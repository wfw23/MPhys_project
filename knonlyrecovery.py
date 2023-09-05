#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:13:11 2023

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

times= np.linspace(0.1,40,200)
num_points=100
noise=0.2

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]
bandfreqs = (redback.utils.bands_to_frequency(bands))
print(bandfreqs)
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

params={}
params['redshift'] = 0.01
params['av'] = 0.5
params['loge0'] = 51.0
params['logn0'] = 0.5 
params['p'] = 2.3
params['logepse'] = -1.25
params['logepsb'] = -2.5
params['ksin'] = 1
params['g0'] = 1000
params['thv']= 0.02
params['thc'] = 0.05
params['mej']=0.03
params['vej_1']=0.1
params['vej_2']=0.4
params['kappa']=10
params['beta']=4
    
combined_model =  SimulateGenericTransient(model='tophat_and_twolayerstratified', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)

kntransient2_on = redback.transient.Afterglow(name='kntransient2_on', flux_density=combined_model.data['output'].values,
                                      time=combined_model.data['time'].values, data_mode='flux_density',
                                      flux_density_err=combined_model.data['output_error'].values, frequency=combined_model.data['frequency'].values)

model='two_layer_stratified_kilonova'
injection_parameters= params
model_kwargs = dict(frequency=kntransient2_on.filtered_frequencies, output_format='flux_density')
priors = redback.priors.get_priors(model=model)
priors['redshift']=0.01
priors

result2 = redback.fit_model(transient=kntransient2_on, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=1000, plot=False, resume=True,  injection_parameters=injection_parameters)
ax=result2.plot_lightcurve(show=False)
for f in frequencies:
    kn=redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, base_model='two_layer_stratified_kilonova', **params, frequency=f, 
                                                                                      output_format='flux_density')
    ax.plot(times,kn, ls=':', c='k', alpha=0.4)
ax.loglog()

f1 = mpatches.Patch(color='blueviolet', label='radio')
f2 = mpatches.Patch(color='b', label='F160W')
f3 = mpatches.Patch(color='dodgerblue', label='F110W')
f4 = mpatches.Patch(color='deepskyblue', label='lssty')
f5 = mpatches.Patch(color='turquoise', label='lsstz')
f6 = mpatches.Patch(color='mediumspringgreen', label='lssti')
f7 = mpatches.Patch(color='yellowgreen', label='lsstr',alpha=0.7)
f8 = mpatches.Patch(color='gold', label='lsstg', alpha=0.7)
f9 = mpatches.Patch(color='orange', label='lsstu')
f10=mpatches.Patch(color='orangered', label='UVOT:uvw1')
f11=mpatches.Patch(color='red', label='UV')
agline=  Line2D([0],[0],color='k', ls='--', label='afterglow', alpha=0.4)
knline=  Line2D([0],[0],color='k', ls=':', label='true fit', alpha=0.4)
plt.legend(handles=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,knline],loc='lower left',bbox_to_anchor=(0, 0))
#plt.savefig('knonly_lightcurve2_on.png', dpi='figure')
plt.show()