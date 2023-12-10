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
import pandas as pd
from redback.simulate_transients import SimulateGenericTransient
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from bilby.core.prior import PriorDict, Uniform, Sine, Constraint

times= np.logspace(3.6,6.6,50)/86400
num_points=100
noise=0.25

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]
#frequencies=[]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

agkwargs={}
agkwargs['loge0'] = 51.0
agkwargs['logn0'] = 0.5
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.02
agkwargs['thc'] = 0.05
agkwargs['base_model']='tophat_redback'
knkwargs={}
knkwargs['mej']=0.03
knkwargs['vej_1']=0.1
knkwargs['vej_2']=0.4
knkwargs['kappa']=10
knkwargs['beta']=4
knkwargs['base_model']='two_layer_stratified_kilonova'
params={}
params['redshift'] = 0.01
params['av'] = 0.5
params['model_type']='kilonova'
params['afterglow_kwargs']=agkwargs
params['optical_kwargs']=knkwargs
    

def afterglow_constraints(parameters):
    constrained_params= parameters.copy()
    
    time=np.linspace(0.1,100,100) 
    
    if isinstance(parameters['thv'], (float,int)):
       flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=0.01, av=0.5,
           base_model='tophat_redback',  thv= parameters['thv'], loge0=parameters['loge0'], thc=parameters['thc'], logn0=parameters['logn0'], 
           p=2.3, logepse=-1.25, logepsb=-2.5, xiN=1.0, g0=1000,
           output_format='flux_density', frequency=frequencies[6]) #remember to change index when optical only
       maxflux=(max(flux))
       peaktime=(time[np.argmax(flux)])
       fluxday1= np.interp(1,time,flux)
       minflux=(fluxday1) 
    else:
        maxflux=[]
        peaktime=[]
        minflux=[]
        size=len(parameters['thv'])
        for i in range(size):
            flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=0.01, av=0.5,
                base_model='tophat_redback',  thv= parameters['thv'][i], loge0=parameters['loge0'][i] , thc=parameters['thc'][i], logn0=parameters['logn0'][i], 
                p=2.3, logepse=-1.25, logepsb=-2.5, xiN=1.0, g0=1000,
                output_format='flux_density', frequency=frequencies[6]) #remember to change index when optical only
            maxflux.append(max(flux))
            peaktime.append(time[np.argmax(flux)])
            fluxday1= np.interp(1,time,flux)
            minflux.append(fluxday1)
        
    #peak flux must be at times < 200 days
    constrained_params['peak_time']= 200 - np.array(peaktime)
    #peak flux must be greater than 10e-12
    constrained_params['max_flux']= np.array(maxflux) - (10e-12)
    #thv-thc for off axis, thc-thv for on axis
    constrained_params['alignment']= parameters['thc'] - parameters['thv']
    #filter lower flux afterglows out for ON AXIS case (comment out if needed)
    constrained_params['min_flux']= np.array(minflux) - 10e-7
    
    return constrained_params

'''
sig_on_test1 =  SimulateGenericTransient(model='afterglow_and_optical', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)
sig_on_test1.save_transient(name='sig_on_test1')
'''
data=pd.read_csv('/home/wfw23/Mphys_proj/simulated/sig_on_test1.csv')
'''
data.mask(data['frequency']==5e9, inplace=True)
data.mask(data['frequency']==2e17, inplace=True)
data.dropna(how='any', inplace=True)
'''
sig_off_agonly_test1 = redback.transient.Afterglow(name='sig_off_agonly_test1', flux_density=data['output'].values,
                                      time=data['time'].values, data_mode='flux_density',
                                      flux_density_err=data['output_error'].values, frequency=data['frequency'].values)
sig_off_agonly_test1.plot_data()

model='extinction_with_afterglow_base_model'
base_model='tophat_from_emulator'
agkwargs['av']=0.5
injection_parameters= agkwargs
model_kwargs = dict(frequency=sig_off_agonly_test1.filtered_frequencies, output_format='flux_density', base_model=base_model)
'''
priors = PriorDict(conversion_function=afterglow_constraints)
priors['max_flux']= Constraint(minimum=0, maximum=10)
priors['peak_time']= Constraint(minimum=0, maximum=200)
priors['alignment']= Constraint(minimum=0, maximum=1.57)
priors['min_flux']=Constraint(minimum=0, maximum=10)

priors.update(redback.priors.get_priors(model='tophat_redback'))
'''
priors=redback.priors.get_priors(model='tophat_redback')
priors['redshift']=0.01
priors['av']=Uniform(minimum=0, maximum=2, name='av', latex_label='$av$', unit=None, boundary=None)
priors['xiN']= 1.0
result = redback.fit_model(transient=sig_off_agonly_test1, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=priors, nlive=1000, plot=False, resume=True, injection_parameters=injection_parameters)

band_labels=['radio']
#band_labels=[]
band_labels.extend(bands)
band_labels.append('X-Ray')
ax=result.plot_lightcurve(show=False, band_labels=band_labels)

for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density',
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
plt.legend(loc='lower left',bbox_to_anchor=(0, 0))
#plt.savefig('sigoff_optical.png', dpi='figure')
plt.show()