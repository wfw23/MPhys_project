#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:11:22 2023

@author: wfw23
"""

import redback
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from bilby.core.prior import PriorDict, Uniform, Sine, Constraint

times= np.logspace(3.6,6.6,50)/86400

bands = ['F160W', 'F110W','lssty', 'lsstz','lssti', 'lsstr','lsstg','lsstu', 'uvot::uvw1']
#frequencies=[5e9, 2e17]
frequencies=[]
bandfreqs = (redback.utils.bands_to_frequency(bands))
frequencies.extend(bandfreqs)
frequencies.sort()

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

band_labels=['radio']
#band_labels=[]
band_labels.extend(bands)
band_labels.append('X-Ray')

band_colors={5e9:'crimson',1.952e14:'orangered',2.601e14:'orange',3.083e14:'gold',3.454e14:'greenyellow',3.983e14:'limegreen',
             4.825e14:'mediumaquamarine',6.273e14:'c',8.152e14:'deepskyblue',1.141e15:'blue',2e17:'blueviolet'}
'''
agon= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_afterglow_base_model/GRBsig_on_agonly_result.json')
agoff= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_afterglow_base_model/GRBsig_off_agonly_result.json')
knon= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBsig_on_knonly_result.json')
knoff= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/extinction_with_kilonova_base_model/GRBsig_off_knonly_result.json')
jointon= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/tophatredback_and_twolayerstratified/GRBsig_on_joint_result.json')
jointoff= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/tophatredback_and_twolayerstratified/GRBsig_off_joint_result.json')


fig= plt.figure(figsize=(14,7))
ax1=fig.add_subplot(2,3,1)
ax1=agon.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density', av=0.5,
     **agkwargs)
    ax1.loglog(times, flux, ls='--', color='k', alpha=0.5)
ax1.annotate('(A)', xy=(20,60), fontsize=14)

ax3=fig.add_subplot(2,3,2, sharex=ax1, sharey=ax1)
ax3=knon.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **knkwargs)
    ax3.loglog(times, flux, ls=':', color='k', alpha=0.5)
ax3.annotate('(B)', xy=(20,60), fontsize=14)

ax5=fig.add_subplot(2,3,3, sharex=ax1, sharey=ax1)
ax5=jointon.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **agkwargs)
    ax5.loglog(times, flux, ls='--', color='k', alpha=0.5)
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **knkwargs)
    ax5.loglog(times, flux, ls=':', color='k', alpha=0.5)
ax5.annotate('(C)', xy=(20,60),fontsize=14)
plt.xlim(0.06,40)

agkwargs['loge0'] = 51.5
agkwargs['logn0'] = 1
agkwargs['thv']= 0.5

ax2=fig.add_subplot(2,3,4)
ax2=agoff.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **agkwargs)
    ax2.loglog(times, flux, ls='--', color='k', alpha=0.5)
ax2.annotate('(D)', xy=(20,1),fontsize=14)

ax4=fig.add_subplot(2,3,5, sharex=ax2, sharey=ax2)    
ax4=knoff.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **knkwargs)
    ax4.loglog(times, flux, ls=':', color='k', alpha=0.5)
ax4.annotate('(E)', xy=(20,1),fontsize=14)

ax6=fig.add_subplot(2,3,6, sharex=ax2, sharey=ax2)    
ax6=jointoff.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors)
for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **agkwargs)
    ax6.loglog(times, flux, ls='--', color='k', alpha=0.5)
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **knkwargs)
    ax6.loglog(times, flux, ls=':', color='k', alpha=0.5)
ax6.annotate('(F)', xy=(20,1),fontsize=14)
plt.xlim(0.06,40)
'''
f1 = mpatches.Patch(color='crimson', label='radio')
f2 = mpatches.Patch(color='orangered', label='F160W')
f3 = mpatches.Patch(color='orange', label='F110W')
f4 = mpatches.Patch(color='gold', label='lssty')
f5 = mpatches.Patch(color='greenyellow', label='lsstz')
f6 = mpatches.Patch(color='limegreen', label='lssti')
f7 = mpatches.Patch(color='mediumaquamarine', label='lsstr',alpha=0.7)
f8 = mpatches.Patch(color='c', label='lsstg', alpha=0.7)
f9 = mpatches.Patch(color='deepskyblue', label='lsstu')
f10=mpatches.Patch(color='blue', label='UVOT:uvw1')
f11=mpatches.Patch(color='blueviolet', label='X-ray')
agline=  Line2D([0],[0],color='k', ls='--', label='afterglow', alpha=0.4)
knline=  Line2D([0],[0],color='k', ls=':', label='kilonova', alpha=0.4)
#fig.text(0.5, -0.02, 'Time since burst [days]', ha='center', fontsize=17)
#fig.text(-0.01, 0.5, 'Flux Density [mJy]', va='center', rotation='vertical', fontsize=17)

#ax4.legend(handles=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,agline,knline],loc='upper center', ncol=7, fontsize=14, bbox_to_anchor=(0.5, -0.26))

result= redback.result.read_in_result('/home/wfw23/Mphys_proj/GRBData/afterglow/flux_density/tophatredback_and_twolayerstratified/GRBsig_off_joint_optical_result.json')
result.meta_data['time'] = result.meta_data['time'][~np.isnan(result.meta_data['time'])]
result.meta_data['flux_density'] = result.meta_data['flux_density'][~np.isnan(result.meta_data['flux_density'])]
result.meta_data['flux_density_err'] = result.meta_data['flux_density_err'][~np.isnan(result.meta_data['flux_density_err'])]
result.meta_data['frequency'] = result.meta_data['frequency'][~np.isnan(result.meta_data['frequency'])]
result.meta_data['bands'] = result.meta_data['bands'][~np.isnan(result.meta_data['bands'])]
ax=result.plot_lightcurve(show=False, band_labels=band_labels, band_colors=band_colors, bands_to_plot=[195200000000000.0, 260100000000000.0, 308300000000000.0, 345400000000000.0, 398300000000000.0, 482500000000000.0, 627300000000000.0, 815200000000000.0, 1141000000000000.0])

for f in frequencies:
    agkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_afterglow_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **agkwargs)
    ax.loglog(times, flux, ls='--', color='k', alpha=0.5)
    knkwargs['frequency']=f
    flux= redback.transient_models.extinction_models.extinction_with_kilonova_base_model(times, redshift=0.01, output_format='flux_density',av=0.5,
     **knkwargs)
    ax.loglog(times, flux, ls=':', color='k', alpha=0.5)

plt.xlim(0.04,40)
plt.legend(handles=[f2,f3,f4,f5,f6,f7,f8,f9,f10,agline,knline],loc='center left', fontsize=12,bbox_to_anchor=(1.01,0.5))

plt.show()

