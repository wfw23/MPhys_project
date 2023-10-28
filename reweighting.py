#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:49:47 2023

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
from bilby.core.result import reweight as rw
from redback.transient_models.afterglow_models import tophat_redback, tophat_from_emulator
from redback.transient_models.extinction_models import extinction_with_afterglow_base_model
from redback.transient_models.combined_models import tophatredback_and_twolayerstratified

times= np.logspace(3.6,6.6,50)/86400
noise=0.25


bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
frequencies=[5e9, 2e17]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()
frequencies
band_labels=['radio']
#band_labels=[]
band_labels.extend(bands)
band_labels.append('X-Ray')

agkwargs={}
agkwargs['loge0'] = 51.0
agkwargs['logn0'] = 0.5
agkwargs['p'] = 2.3
agkwargs['logepse'] = -1.25
agkwargs['logepsb'] = -2.5
agkwargs['xiN'] = 1
agkwargs['g0'] = 1000
agkwargs['thv']= 0.03
agkwargs['thc'] = 0.07
agkwargs['base_model']='tophat_redback'
agkwargs['output_format']='flux_density'

result = redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/tophatredback_and_twolayerstratified/GRBsig_on_joint_result.json')

kwargs = dict(frequency=result.meta_data['frequency'], output_format='flux_density', base_model='tophat_redback', axis='on')
x=result.meta_data['time']
y=result.meta_data['flux_density']
sigma=result.meta_data['flux_density_err']
new_likelihood = redback.likelihoods.GaussianLikelihood(x,y ,sigma=sigma, function=tophatredback_and_twolayerstratified, kwargs=kwargs)
kwargs['base_model']='tophat_from_emulator'
emulator_like= redback.likelihoods.GaussianLikelihood(x,y ,sigma=sigma, function=extinction_with_afterglow_base_model, kwargs=kwargs)

new_result= rw(result, label='rw_sig_on_joint', new_likelihood=new_likelihood)
new_result.save_to_file('rw_sig_on_joint.json')
ax2= new_result.plot_lightcurve(show=False, band_labels=band_labels)

'''
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.afterglow_models.tophat_redback(times, redshift=0.01, av=0.5,
     **agkwargs)
    ax2.plot(times, flux, ls='--', color='k', alpha=0.5)
'''
#flux= redback.transient_models.afterglow_models.tophat_redback(times, redshift=0.01, frequency=3.45e14, **agkwargs)
#ax2.plot(times,flux, ls='--', color='k')
ax2.loglog()
plt.show()


