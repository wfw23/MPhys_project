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

times= np.linspace(0.1,40,100)
num_points=40
noise=0.25


model_kwargs = {'output_format':'flux_density', 'frequency':3.45e14}

agkwargs={}
agkwargs['loge0'] = 51.5
agkwargs['logn0'] = 1
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.07
agkwargs['thc'] = 0.09
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
    
combined_model =  SimulateGenericTransient(model='afterglow_and_optical', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=False, noise_term=noise)

agdominates_on_restest = redback.transient.Afterglow(name='agdominates_on_restest', flux_density=combined_model.data['output'].values,
                                      time=combined_model.data['time'].values, data_mode='flux_density',
                                      flux_density_err=combined_model.data['output_error'].values, frequency=combined_model.data['frequency'].values)

agdominates_on_restest.plot_data()
model='extinction_with_afterglow_base_model'
base_model='tophat_from_emulator'
agkwargs['av']=0.5
injection_parameters= agkwargs
model_kwargs = dict(frequency=agdominates_on_restest.filtered_frequencies, output_format='flux_density', base_model=base_model)
priors = redback.priors.get_priors(model=base_model)
priors['redshift']=0.01
priors['av']=Uniform(minimum=0, maximum=2, name='av', latex_label='$av$', unit=None, boundary=None)
priors['xiN']=1


result = redback.fit_model(transient=agdominates_on_restest, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=1000, plot=False, resume=True,injection_parameters=injection_parameters)

ax=result.plot_lightcurve(show=False)


agkwargs['frequency']=3.45e14
flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01,
     **agkwargs)
ax.plot(times, flux, ls='--', color='k', alpha=0.5)
ax.loglog()

plt.legend(loc='lower left',bbox_to_anchor=(0, 0))
#plt.savefig('sigoff_optical.png', dpi='figure')
plt.show()