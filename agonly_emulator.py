#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 22:41:51 2023

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
noise=0.25

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
#frequencies=[5e9, 2e17]
frequencies=[]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

agkwargs={}
agkwargs['loge0'] = 51.5
agkwargs['logn0'] = 1
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.5
agkwargs['thc'] = 0.07
agkwargs['base_model']='tophat_redback'
knkwargs={}
knkwargs['mej']=0.03
knkwargs['vej_1']=0.1
knkwargs['vej_2']=0.4
knkwargs['kappa']=5
knkwargs['beta']=4
knkwargs['base_model']='two_layer_stratified_kilonova'
params={}
params['redshift'] = 0.01
params['av'] = 0.5
params['model_type']='kilonova'
params['afterglow_kwargs']=agkwargs
params['optical_kwargs']=knkwargs
    
combined_model =  SimulateGenericTransient(model='afterglow_and_optical', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)

significant_off_optical = redback.transient.Afterglow(name='significant_off_optical', flux_density=combined_model.data['output'].values,
                                      time=combined_model.data['time'].values, data_mode='flux_density',
                                      flux_density_err=combined_model.data['output_error'].values, frequency=combined_model.data['frequency'].values)

significant_off_optical.plot_data()
model='tophat_from_emulator'
injection_parameters= agkwargs
model_kwargs = dict(frequency=significant_off_optical.filtered_frequencies, output_format='flux_density')
priors = redback.priors.get_priors(model='tophat_redback')
priors['redshift']=0.01
priors['xiN']=1
priors

result = redback.fit_model(transient=significant_off_optical, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=500, plot=False, resume=True, injection_parameters=injection_parameters)
ax=result.plot_lightcurve(show=False)
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, av=0.5,
     **agkwargs)
    ax.plot(times, flux, ls='--', color='k', alpha=0.5)
ax.loglog()

f1 = mpatches.Patch(color='blueviolet', label='radio')
f2 = mpatches.Patch(color='blueviolet', label='F160W')
f3 = mpatches.Patch(color='blue', label='F110W')
f4 = mpatches.Patch(color='deepskyblue', label='lssty')
f5 = mpatches.Patch(color='turquoise', label='lsstz')
f6 = mpatches.Patch(color='mediumspringgreen', label='lssti')
f7 = mpatches.Patch(color='yellowgreen', label='lsstr',alpha=0.7)
f8 = mpatches.Patch(color='gold', label='lsstg', alpha=0.7)
f9 = mpatches.Patch(color='orange', label='lsstu')
f10=mpatches.Patch(color='orangered', label='UVOT:uvw1')
f11=mpatches.Patch(color='red', label='X-Ray')
agline=  Line2D([0],[0],color='k', ls='--', label='afterglow', alpha=0.4)
knline=  Line2D([0],[0],color='k', ls=':', label='kilonova', alpha=0.4)
plt.legend(handles=[f2,f3,f4,f5,f6,f7,f8,f9,f10],loc='lower left',bbox_to_anchor=(0, 0))
plt.savefig('sigoff_optical.png', dpi='figure')
plt.show()