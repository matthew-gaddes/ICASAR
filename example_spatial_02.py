# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:17:22 2018

@author: eemeg

An example of the ICASAR software with real data that contains a deformation signals that is not visible in a single 6/12 days Sentinel-1 interferogram.  

"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path

from ICASAR_functions import ICASAR
#from auxiliary_functions import col_to_ma, r2_to_r3
from auxiliary_functions_from_other_repos  import LiCSBAS_to_LiCSAlert, get_param_par, read_img


import sys; sys.path.append('/home/matthew/university_work/python_stuff/python_scripts/')
from small_plot_functions import matrix_show
#%% Things to set

ICASAR_settings = {"n_comp" : 5,                                         # number of components to recover with ICA (ie the number of PCA sources to keep)
                   "bootstrapping_param" : (200, 0),                    # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                   "tsne_param" : (30, 12),                             # (perplexity, early_exaggeration)
                   "ica_param" : (1e-2, 150),                           # (tolerance, max iterations)
#                   "hdbscan_param" : (35,10),                           # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                   "hdbscan_param" : (55,10),                           # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                   "out_folder" : Path('example_spatial_02_outputs'),   # outputs will be saved here
                   "create_all_ifgs_flag" : True,                       # small signals are hard for ICA to extact from time series, so make it easier by creating all possible long temporal baseline ifgs from the incremental data.  
                   "load_fastICA_results" : True,                      # If all the FastICA runs already exisit, setting this to True speeds up ICASAR as they don't need to be recomputed.  
                   "figures" : "png+window"}                            # if png, saved in a folder as .png.  If window, open as interactive matplotlib figures,
                                                                         # if 'png+window', both.  
                                                                         # default is "window" as 03_clustering_and_manifold is interactive.  
                                                                    
campi_flegrei_data_file = Path('./campi_flegrei_LiCSBAS_example_data/cum.h5')                               # the main results file from LICSBAS, which includes the incremental (between epoch or acquisition) deformation.                          
#campi_flegrei_data_file = Path('./campi_flegrei_LiCSBAS_example_data/cum_filt.h5')                                            
campi_flegrei_hgt_file = Path('./campi_flegrei_LiCSBAS_example_data/hgt')                                   # The DEM from LiCSBAS, as a binary file.              
mlipar = './campi_flegrei_LiCSBAS_example_data/slc.mli.par'                                                 # Parameter file from LiCSBAS, which contains the clipped scenen width and height, which we need to open the binary dem file.  



                                              
#%% Import the data (using a funtcion that is normally found in LiCSAlert)

print(f"Opening the LiCSBAS .h5 file...", end = '')
displacement_r2, temporal_baselines = LiCSBAS_to_LiCSAlert(campi_flegrei_data_file, figures=True)                             # open the h5 file produced by LiCSBAS, lons and lats are in geocode info and same resolution as the ifgs
print(f"Done.  ")

print(f"Opening the LiCSAR/LiCSBAS hgt (DEM) file...", end = '')
width = int(get_param_par(mlipar, 'range_samples'))
length = int(get_param_par(mlipar, 'azimuth_lines'))
dem = read_img(campi_flegrei_hgt_file, length, width)
del width, length
print(f"Done.  ")



# need the temporal baselines
# need to get the temporal baselines of all the ifg combinations.  
# function to: 
        # plot ic strength against tempotal baseline, using kernel density
        # line of best fit

# need the dem
        # same function as for temporal baseline.  

#%% do ICA with ICSAR function



spatial_data = {'mixtures_r2'    : displacement_r2['incremental'],
                'mask'           : displacement_r2['mask'],
                'ifg_dates'      : temporal_baselines['daisy_chain'],                           # in form YYYYMMDD_YYYYMMDD as a list of strings.  
                'dem'            : dem,
                'lons'           : displacement_r2['lons'],
                'lats'           : displacement_r2['lats']}
                
 

S_best, time_courses, x_train_residual_ts, Iq, n_clusters, S_all_info, phUnw_mean  = ICASAR(spatial_data = spatial_data, **ICASAR_settings) 
      

#%% testing

from auxiliary_functions import create_all_ifgs, col_to_ma
_, mixtures, ifg_dates = create_all_ifgs(spatial_data['mixtures_r2'], spatial_data['ifg_dates'])                               # if ifg_dates is None, None is also returned.  

matrix_show(time_courses)

matrix_show(col_to_ma(mixtures[777,], spatial_data['mask']))

#%% How well are we fitting the data

# cf_data = spatial_data['mixtures_r2']

# f, axes = plt.subplots(3,1)
# matrix_show(cf_data, ax = axes[0], fig = f)
# matrix_show(time_courses @ S_best, ax = axes[1], fig = f)
# matrix_show(cf_data - (time_courses @ S_best), ax = axes[2], fig = f)

#%%








def visualise_ICASAR_inversion(interferograms, sources, time_courses, mask, n_data = 10):
    """
        2021_03_03 | MEG | Written.  
    """
    
    def plot_ifg(ifg, ax, mask, vmin, vmax):
        """
        """
        w = ax.imshow(col_to_ma(ifg, mask), interpolation ='none', aspect = 'equal', vmin = vmin, vmax = vmax)                                                   # 
        axin = ax.inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(w, cax=axin, orientation='horizontal')
        ax.set_yticks([])
        ax.set_xticks([])
    
    import matplotlib.pyplot as plt
    
    interferograms_mc = interferograms - np.mean(interferograms, axis = 1)[:, np.newaxis]
    interferograms_ICASAR = time_courses @ sources
    residual = interferograms_mc - interferograms_ICASAR
    
    if n_data > interferograms.shape[0]:
        n_data = interferograms.shape[0]

    
    fig, axes = plt.subplots(3, n_data, figsize = (15,7))  
    if n_data == 1:    
        axes = np.atleast_2d(axes).T                                                # make 2d, and a column (not a row)
    
    row_labels = ['Data', 'Model', 'Resid.' ]
    for ax, label in zip(axes[:,0], row_labels):
        ax.set_ylabel(label)

    for data_n in range(n_data):
        vmin = np.min(np.stack((interferograms_mc[data_n,], interferograms_ICASAR[data_n,], residual[data_n])))
        vmax = np.max(np.stack((interferograms_mc[data_n,], interferograms_ICASAR[data_n,], residual[data_n])))
        plot_ifg(interferograms_mc[data_n,], axes[0,data_n], mask, vmin, vmax)
        plot_ifg(interferograms_ICASAR[data_n,], axes[1,data_n], mask, vmin, vmax)
        plot_ifg(residual[data_n,], axes[2,data_n], mask, vmin, vmax)



visualise_ICASAR_inversion(displacement_r2['incremental'], S_best, time_courses, displacement_r2['mask'], n_data = 10)























