#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:05:58 2023

@author: wfw23
"""

import redback
import numpy as np
import pandas as pd
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
frequencies=[5e9, 2e17]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

agkwargs={}
agkwargs['loge0'] = 51.98
agkwargs['logn0'] = 0.76
agkwargs['p'] = 2.32
agkwargs['logepse'] = -1.15
agkwargs['logepsb'] = -2.11
agkwargs['xiN'] = 1
agkwargs['g0'] = 1685.05
agkwargs['thv']= 0.41
agkwargs['thc'] = 0.03
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
    
'''
combined_model =  SimulateGenericTransient(model='afterglow_and_optical', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)
'''
data=pd.read_csv('/home/wfw23/Mphys_proj/simulated/sig_off.csv')
data.mask(data['frequency']==5e9, inplace=True)
data.mask(data['frequency']==2e17, inplace=True)
data.dropna(how='any', inplace=True)

afterglow_values= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(data['time'].values, av=0.99, redshift=0.01, **agkwargs, output_format='flux_density',
                                                                                                  frequency=data['frequency'].values)
#subtracted data
#print(len(data))
data.mask((data['output']-data['output_error'] <= afterglow_values) & (afterglow_values <= data['output']+data['output_error']), inplace=True)
#print(len(data))
flux_density= data['output'].values - afterglow_values
#print(data['output'].values)
flux_density_err= (data['output_error']/data['output'])*flux_density
data['output']=flux_density
data['output_error']=flux_density_err
data.mask(data['output']<=5e-6, inplace=True)
data.mask(data['output_error']> (2*data['output']), inplace=True)
data.dropna(how='any', inplace=True)

subtracted_knonly_off_pred_new = redback.transient.Afterglow(name='subtracted_knonly_off_pred_new', flux_density=data['output'].values,
                                      time=data['time'].values, data_mode='flux_density',
                                      flux_density_err=data['output_error'].values, frequency=data['frequency'].values)

subtracted_knonly_off_pred_new.plot_data()
model='extinction_with_kilonova_base_model'
base_model='two_layer_stratified_kilonova'
knkwargs['av']=0.5
injection_parameters= knkwargs
model_kwargs = dict(frequency=subtracted_knonly_off_pred_new.filtered_frequencies, output_format='flux_density', base_model=base_model)
priors = redback.priors.get_priors(model=base_model)
priors['redshift']=0.01
priors['av']=Uniform(minimum=0, maximum=2, name='av', latex_label='$av$', unit=None, boundary=None)

result = redback.fit_model(transient=subtracted_knonly_off_pred_new, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=1000, plot=False, resume=True, 
                           injection_parameters=injection_parameters)

band_colors={5e9:'crimson',1.952e14:'orangered',2.601e14:'orange',3.083e14:'gold',3.454e14:'greenyellow',3.983e14:'limegreen',
             4.825e14:'mediumaquamarine',6.273e14:'c',8.152e14:'deepskyblue',1.141e15:'blue',2e17:'blueviolet'}
#band_labels=['radio']
band_labels=[]
band_labels.extend(bands)
#band_labels.append('X-Ray')
ax=result.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)

for f in frequencies:
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',
     **knkwargs)
    ax.plot(times, flux, ls='--', color='k', alpha=0.5)
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
f11=mpatches.Patch(color='red', label='X-ray')
agline=  Line2D([0],[0],color='k', ls='--', label='afterglow', alpha=0.4)
knline=  Line2D([0],[0],color='k', ls=':', label='kilonova', alpha=0.4)
plt.legend(loc='lower left',bbox_to_anchor=(0, 0))
#handles=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11],
#plt.savefig('significant_ok.png', dpi='figure')
plt.show()