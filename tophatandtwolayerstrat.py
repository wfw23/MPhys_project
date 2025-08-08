#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 19:56:59 2023

@author: wfw23
"""
import redback
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
from redback.simulate_transients import SimulateGenericTransient
from bilby.core.prior import PriorDict, Uniform, Sine, Constraint

times= np.logspace(3.6,6.6,50)/86400
num_points=100
noise=0.25

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]  #COMMENT OUT if w/out X-ray and radio
#frequencies=[]          #INCLUDE  if w/out X-ray and radio
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

model_kwargs = {'output_format':'flux_density', 'frequency':frequencies}

agkwargs={}
agkwargs['loge0'] = 52.0#on-ax = 51.0 , off-ax = 51.5, dom=52
agkwargs['logn0'] = 1.5 #on-ax = 0.5, off-ax = 1, dom=1.5
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.03 #on-ax = 0.03, off-ax = 0.5, dom =0.03
agkwargs['thc'] = 0.08 #on/off=0.07, dom=0.08
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
sig_on_optical =  SimulateGenericTransient(model='afterglow_and_optical', parameters=params,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=True, noise_term=noise)
'''

sig_off_joint_optical = redback.transient.Afterglow(name='sig_off_joint_optical', flux_density=data['output'].values,
data=pd.read_csv('/home/wfw23/Mphys_proj/simulated/agdom_on.csv') #sig_on.csv or sig_off.csv

#INCLUDE below if w/out X-ray/radio
'''
data.mask(data['frequency']==5e9, inplace=True)
data.mask(data['frequency']==2e17, inplace=True)
data.dropna(how='any', inplace=True)
'''

#remember to change name for new recoveries
sig_on_joint_test2 = redback.transient.Afterglow(name='new_both_agdom', flux_density=data['output'].values,
                                      time=data['time'].values, data_mode='flux_density',
                                      flux_density_err=data['output_error'].values, frequency=data['frequency'].values)

model='tophatredback_and_twolayerstratified'
params.update(agkwargs)
params.update(knkwargs)
injection_parameters= params
model_kwargs = dict(frequency=sig_off_joint_optical.filtered_frequencies, output_format='flux_density',axis='off')

#all_priors = PriorDict(conversion_function=afterglow_constraints)
#all_priors['max_flux']= Constraint(minimum=0, maximum=10)
#all_priors['peak_time']= Constraint(minimum=0, maximum=200)
#all_priors['alignment']= Constraint(minimum=0, maximum=1.57)
#all_priors['min_flux']=Constraint(minimum=10e-7, maximum=10)
all_priors=(redback.priors.get_priors(model=model))
all_priors['redshift']=0.01
all_priors['xiN']= 1.0

nwalkers=100
start_pos = bilby.core.prior.PriorDict()
for key in ['logepse','logepsb','av', 'kappa', 'beta','logn0', 'loge0']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 0.01)
for key in ['thv', 'thc', 'p', 'mej', 'vej_1', 'vej_2']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 0.001)
for key in ['g0']:
    start_pos[key] = bilby.core.prior.Normal(injection_parameters[key], 1)
pos0 = pd.DataFrame(start_pos.sample(nwalkers))

result_both = redback.fit_model(transient=sig_on_joint_test2, model=model, sampler='emcee', model_kwargs=model_kwargs,
                           prior=all_priors, plot=False, resume=True, injection_parameters=injection_parameters,
                           walks=nwalkers, nlive=2200, nburn=1000, pos0=pos0)
'''
result_both = redback.fit_model(transient=sig_on_joint_test2, model=model, sampler='nestle', model_kwargs=model_kwargs,
                           prior=all_priors, plot=False, resume=True,injection_parameters=injection_parameters,
                           nlive=1000)
'''
band_labels=['radio']       #COMMENT OUT if w/out X-ray and radio
#band_labels=[]             #INCLUDE  if w/out X-ray and radio
band_labels.extend(bands)   
band_labels.append('X-Ray') #COMMENT OUT if w/out X-ray and radio
ax=result_both.plot_lightcurve(show=False, band_labels=band_labels)
ax.loglog()
plt.legend(loc='center', bbox_to_anchor=(1.15, 0.5))
plt.show()
