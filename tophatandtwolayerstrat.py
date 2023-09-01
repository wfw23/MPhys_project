#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 19:56:59 2023

@author: wfw23
"""
import redback
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGenericTransient

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
params['thv']= 0.6
params['thc'] = 0.06
params['mej']=0.03
params['vej_1']=0.1
params['vej_2']=0.4
params['kappa']=10
params['beta']=4
    
combined_model =  SimulateGenericTransient(model='tophat_and_twolayerstratified', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)

combined_transient= redback.transient.Afterglow(name='combined_transient', flux_density=combined_model.data['output'].values,
                                      time=combined_model.data['time'].values, data_mode='flux_density',
                                      flux_density_err=combined_model.data['output_error'].values, frequency=combined_model.data['frequency'].values)

model='tophat_and_twolayerstratified'
injection_parameters= params
model_kwargs = dict(frequency=combined_transient.filtered_frequencies, output_format='flux_density')
all_priors = redback.priors.get_priors(model=model)
all_priors['redshift']=0.01
all_priors['av']=0.5

result_both = redback.fit_model(transient=combined_transient, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=all_priors, nlive=1000, plot=False, resume=True, injection_parameters=injection_parameters)
ax=result_both.plot_lightcurve(show=False)
ax.loglog()
plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5))
plt.show()
