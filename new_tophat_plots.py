#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:06:21 2023

@author: wfw23
"""

import redback
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import corner
from bilby.core.prior import Constraint, PriorDict, Uniform, Sine

bands = ['lsstg', 'lsstr', 'lssti','lsstz','lssty', 'lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies

that_priors = redback.priors.get_priors(model='tophat_redback')
that_priors['p']=  2.3
that_priors['loge0']= Uniform(minimum=46, maximum=53, name='loge0', latex_label='$\\log_{10}E_{0}$', unit=None, boundary=None)
that_priors['redshift']= 0.01
that_priors['logepse']= -1.25
that_priors['logepsb']= -2.5
that_priors['xiN']= 1.0
that_priors['g0']= 1000

def afterglow_constraints(parameters):
    constrained_params= parameters.copy()
    
    time=np.linspace(0.1,100,100) #make sure time arrays match
    maxflux=[]
    peaktime=[]
    minflux=[]
    for i in range(len(parameters['thv'])):
        flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(time=time, redshift=0.01, av=0.5,
            base_model='tophat_redback',  thv= parameters['thv'][i], loge0=parameters['loge0'][i] , thc=parameters['thc'][i], logn0=parameters['logn0'][i], 
            p=2.3, logepse=-1.25, logepsb=-2.5, xiN=1.0, g0=1000,
            output_format='flux_density', frequency=frequencies[4])
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

priors = PriorDict(conversion_function=afterglow_constraints)
priors['max_flux']= Constraint(minimum=0, maximum=20)
priors['peak_time']= Constraint(minimum=0, maximum=250)
priors['alignment']= Constraint(minimum=0, maximum=1.57)
priors['min_flux']=Constraint(minimum=0, maximum=10)
priors.update(redback.priors.get_priors(model='tophat_redback'))
priors['p']=  2.3
priors['thv']= Uniform(minimum=0, maximum=0.1, name='thv', latex_label='thv', unit=None, boundary=None)
priors['loge0']= Uniform(minimum=46, maximum=53, name='loge0', latex_label='$\\log_{10}E_{0}$', unit=None, boundary=None)
priors['redshift']= 0.01
priors['logepse']= -1.25
priors['logepsb']= -2.5
priors['xiN']= 1.0
priors['g0']= 1000

samples=priors.sample(5000)
afterglow_data=pd.DataFrame.from_dict(samples)
afterglow_data.to_csv("onaxis_samples_redback.csv")
priordata=that_priors.sample(5000)
priordata=pd.DataFrame.from_dict(priordata)
on_axisdata= pd.read_csv('onaxis_samples_redback.csv')

agdata= np.array([on_axisdata['logn0'],on_axisdata['loge0'],on_axisdata['thv'],on_axisdata['thc']])
agdata= agdata.transpose()

original=np.array([priordata['logn0'],priordata['loge0'],priordata['thv'],priordata['thc']])
original=original.transpose()

figure=corner.corner(agdata, labels=["logn0", "loge0", 'thv','thc'], quantiles=[0.16, 0.5, 0.84], show_titles=True, color='red')
figure=corner.corner(original, labels=["logn0", "loge0", 'thv','thc'], quantiles=[0.16, 0.5, 0.84], color='blue', fig=figure)
plt.savefig("onaxis_corner_redback.png", dpi='figure')
