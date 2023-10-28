#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:18:06 2023

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
from bilby.core.prior import PriorDict, Uniform, Sine

times=  np.logspace(3.6,6.8,50)/86400
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
agkwargs['redshift'] = 0.01
'''
agkwargs['base_model']='tophat_redback'
knkwargs={}
knkwargs['mej']=0.03
knkwargs['vej_1']=0.1
knkwargs['vej_2']=0.4
knkwargs['kappa']=25
knkwargs['beta']=4
knkwargs['base_model']='two_layer_stratified_kilonova'
params={}
params['redshift'] = 0.01
params['av'] = 0.5
params['model_type']='kilonova'
params['afterglow_kwargs']=agkwargs
params['optical_kwargs']=knkwargs
'''    
agmodel =  SimulateGenericTransient(model='tophat_redback', parameters=agkwargs,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)

test_sigoff= redback.transient.Afterglow(name='test_sigoff', flux_density=agmodel.data['output'].values,
                                      time=agmodel.data['time'].values, data_mode='flux_density',
                                      flux_density_err=agmodel.data['output_error'].values, frequency=agmodel.data['frequency'].values)

test_sigoff.plot_data()
model='tophat_from_emulator'
#base_model='tophat_from_emulator'
agkwargs['av']=0.5
injection_parameters= agkwargs
model_kwargs = dict(frequency=test_sigoff.filtered_frequencies, output_format='flux_density', axis='off')#, base_model=base_model)
priors = redback.priors.get_priors(model=model)
priors['redshift']=0.01
#priors['av']=Uniform(minimum=0, maximum=2, name='av', latex_label='$av$', unit=None, boundary=None)
priors['xiN']=1


result = redback.fit_model(transient=test_sigoff, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=1000, plot=False, resume=True,injection_parameters=injection_parameters, clean=True)

#ax=result.plot_lightcurve(show=False)

'''
agkwargs['frequency']=3.45e14
flux= redback.transient_models.afterglow_models.tophat_redback(times, output_format='flux_density',
     **agkwargs)
ax.plot(times, flux, ls='--', color='k', alpha=0.5)
ax.loglog()
'''
#band_labels=['radio']
band_labels=[]
band_labels.extend(bands)
#band_labels.append('X-Ray')
ax=result.plot_lightcurve(show=False, band_labels=band_labels)

for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.afterglow_models.tophat_redback(times, output_format='flux_density', 
     **agkwargs)
    ax.plot(times, flux, ls='--', color='k', alpha=0.5)
ax.loglog()
plt.legend(loc='lower left',bbox_to_anchor=(0, 0))
plt.savefig('test_sigoff.png', dpi='figure')
plt.show()