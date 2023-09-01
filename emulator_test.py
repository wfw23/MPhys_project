# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import redback
import pandas as pd
from bilby.core.prior import Constraint, PriorDict, Uniform
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import numpy as np

from redback.simulate_transients import SimulateGenericTransient
times= np.linspace(0.1,40,500)
num_points=50
noise=0.25

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

params={}
params['av'] = 0.5
params['loge0'] = 49.5
params['logn0'] = 0.5 
params['p'] = 2.3
params['logepse'] = -1.25
params['logepsb'] = -2.5
params['g0'] = 1000
params['thv']= 0.5
params['thc'] = 0.06
params['redshift']=0.01
params['xiN']=1
params['base_model']='tophat_redback'
    
tophatag =  SimulateGenericTransient(model='extinction_with_afterglow_base_model', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)

emulator_redback = redback.transient.Afterglow(name='emulator_redback', flux_density=tophatag.data['output'].values,
                                      time=tophatag.data['time'].values, data_mode='flux_density',
                                      flux_density_err=tophatag.data['output_error'].values, frequency=tophatag.data['frequency'].values)

ax=emulator_redback.plot_data(show=False)
ax.loglog()
plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5))
plt.show()

model='tophat_from_emulator'
injection_parameters= params
model_kwargs = dict(frequency=emulator_redback.filtered_frequencies, output_format='flux_density')
priors = redback.priors.get_priors('tophat_redback')
priors['redshift']=0.01
priors['xiN']=1
#priors['frequency']= Uniform(minimum=10e9, maximum=10e17, name='frequency', latex_label='$frequency$', unit=None, boundary=None)
priors

result = redback.fit_model(transient=emulator_redback, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=500, plot=False, resume=True, injection_parameters=injection_parameters)

result.plot_corner()
ax=result.plot_lightcurve(show=False)
for f in frequencies:
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, av=0.5,
    base_model='tophat_redback',  thv= 0.5, loge0=49.5 , thc= 0.06, logn0=0.5, p=2.3, logepse=-1.25, logepsb=-2.5, xiN=1, g0=1000,
    output_format='flux_density' , frequency=f)
    ax.plot(times, flux, ls='--', color='k', alpha=0.5)
plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5))
ax.loglog()
plt.show()
