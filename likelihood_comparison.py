#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:19:56 2023

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
from redback.transient_models.afterglow_models import tophat_redback, tophat_from_emulator

#times= np.linspace(0.1,40,200)
times=  np.logspace(3.94,6.6,50)/86400
num_points=40
noise=0.25


model_kwargs = {'output_format':'flux_density', 'frequency':5e9}

agkwargs={}
agkwargs['redshift'] = 0.01
agkwargs['loge0'] = 51.5
agkwargs['logn0'] = 1
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.25
agkwargs['thc'] = 0.08

afterglow_data =  SimulateGenericTransient(model='tophat_redback', parameters=agkwargs,
                                            times=times, data_points=num_points, model_kwargs=model_kwargs, 
                                            multiwavelength_transient=False, noise_term=noise)

afterglow_object = redback.transient.Afterglow(name='afterglow_object', flux_density=afterglow_data.data['output'].values,
                                      time=afterglow_data.data['time'].values, data_mode='flux_density',
                                      flux_density_err=afterglow_data.data['output_error'].values, frequency=afterglow_data.data['frequency'].values)

kwargs = dict(frequency=afterglow_data.data['frequency'].values, output_format='flux_density')
x=afterglow_object.x
y=afterglow_object.y
sigma=afterglow_object.y_err
tophat_redback_likelihood = redback.likelihoods.GaussianLikelihood(x,y ,sigma=sigma, function=tophat_redback, kwargs=kwargs)
emulator_likelihood = redback.likelihoods.GaussianLikelihood(x,y ,sigma=sigma, function=tophat_from_emulator, kwargs=kwargs)

tophat_redback_likelihood.parameters.update(agkwargs)
agkwargs.pop('xiN')
emulator_likelihood.parameters.update(agkwargs)

print(tophat_redback_likelihood.log_likelihood(), emulator_likelihood.log_likelihood())

priors=redback.priors.get_priors(model='tophat_redback')
priors['redshift']=0.01
priors['p']=2.3
priors['logepse']=-1.25
priors['logepsb']=-2.5
priors['g0']=1000
priors['xiN']=1

for i in range(20):
    samples=priors.sample()
    tophat_redback_likelihood.parameters.update(samples)
    samples.pop('xiN')
    emulator_likelihood.parameters.update(samples)
    #print(samples)
    print(tophat_redback_likelihood.log_likelihood(), emulator_likelihood.log_likelihood())

