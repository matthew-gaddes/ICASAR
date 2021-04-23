# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:23:10 2018

@author: eemeg
"""



def ICASAR(n_comp, spatial_data = None, temporal_data = None, figures = "window", 
           bootstrapping_param = (200,0), ica_param = (1e-4, 150), tsne_param = (30,12), hdbscan_param = (35,10),
           out_folder = './ICASAR_results/', ica_verbose = 'long', inset_axes_side = {'x':0.1, 'y':0.1}, create_all_ifgs_flag = False):
    """
    Perform ICASAR, which is a robust way of applying sICA to data.  As PCA is also performed as part of this,
    the sources and time courses found by PCA are also returned.  Note that this can be run with eitehr 1d data (e.g. time series for a GPS station),
    or on 2d data (e.g. a time series of interferograms) by providing a 'mask', that is used to convert 1d row vectors to 2d masked arrays.  
    
    A note on reference areas/pixels:
        ICASAR requires each interferogram to be mean centered (ie the mean of all the pixels for a single interferogram is 0).  
        Therefore, when the time series is reconstructed using the result of ICASAR (i.e. tcs * sources), these will produce 
        the mean centered time series.  If you wish to work 
    
    Inputs:
        n_comp | int | Number of ocmponents that are retained from PCA and used as the input for ICA.  
        spatial_data | dict or None | contains 'mixtures_r2' in which the images are stored as row vectors and 'mask', which converts a row vector back to a masked array
                                       Optional:
                                           lons | rank 2 array | lons of each pixel in the image.  Changed to rank 2 in version 2.0, from rank 1 in version 1.0  .  If supplied, ICs will be geocoded as kmz.  
                                           lats | rank 2 array | lats of each pixel in the image. Changed to rank 2 in version 2.0, from rank 1 in version 1.0
                                           dem | rank 2 array | height in metres of each pixel in the image.  If supplied, IC vs dem plots will be produced.  
                                           ifg_dates | list | dates of the interferograms in the form YYYYMMDD_YYYYMMDD.  If supplied, IC strength vs temporal baseline plots will be produced.  
        temporal_data | dict or None | contains 'mixtures_r2' as time signals as row vectors and 'xvals' which are the times for each item in the time signals.   
        figures | string,  "window" / "png" / "none" / "png+window" | controls if figures are produced, noet none is the string none, not the NoneType None
        
        bootstrapping_param | tuple | (number of ICA runs with bootstrap, number of ICA runs without bootstrapping )  e.g. (100,10)
        ica_param | tuple | Used to control ICA, (ica_tol, ica_maxit)
        hdbscan_param  | tuple | Used to control the clustering (min_cluster_size, min_samples)
        tsne_param     | tuple | Used to control the 2d manifold learning  (perplexity, early_exaggeration)
        
        

        out_folder | string | if desired, can set the name of the folder results are saved to.  Should end with a /
        
        ica_verbose | 'long' or 'short' | if long, full details of ICA runs are given.  If short, only the overall progress 
        inset_axes_side | dict | inset axes side length as a fraction of the full figure, in x and y direction in the 2d figure of clustering results.  
        
        create_all_ifgs_flag | boolean | If spatial_data contains incremental ifgs (i.e. the daisy chain), these can be recombined to create interferograms 
                                    between all possible acquisitions to improve performance with lower magnitude signals (that are hard to see in 
                                    in short temporal baseline ifgs).  
                                    e.g. for 3 interferogams between 4 acquisitions: a1__i1__a2__i2__a3__i3__a4
                                    This option would also make: a1__i4__a3, a1__i5__a4, a2__i6__a4

    Outputs:
        S_best | rank 2 array | the recovered sources as row vectors (e.g. 5 x 1230)
        mask | rank 2 boolean | Same as inputs, but useful to save.  mask to convert the ifgs as rows into rank 2 masked arrays.  Used for figure outputs, an
        tcs    | rank 2 array | the time courses for the recoered sources (e.g. 17 x 5)
        source_residuals    | ? | the residual when each input mixture is reconstructed using the sources and time courses 
        Iq_sorted              | ?|     the cluster quality index for each centrotype
        n_clusters  | int | the number of clusters found.  Doens't include noise, so is almost always 1 less than the length of Iq
        S_all_info | dictionary| useful for custom plotting. Sources: all the sources in a rank 3 array (e.g. 500x500 x1200 for 6 sources recovered 200 times)
                                                            labels: label for each soure
                                                            xy: x and y coordinats for 2d representaion of all sources
        phUnw_mean | r2 array | the mean for each interfeorram.  subtract from (tcs * sources) to get back original ifgs.  
        
    History:
        2018/06/?? | MEG | Written
        2019/11/?? | MEG | Rewrite to be more robust and readable
        2020/06/03 | MEG | Update figure outputs.  
        2020/06/09 | MEG | Add a raise Exception so that data cannot have nans in it.  
        2020/06/12 | MEG | Add option to name outfolder where things are saved, and save results there as a pickle.  
        2020/06/24 | MEG | Add the ica_verbose option so that ICASAR can be run without generating too many terminal outputs.  
        2020/09/09 | MEG | Major update to now handle temporal data (as well as spatial data)
        2020/09/11 | MEG | Small update to allow an argument to be passed to plot_2d_interactive_fig to set the size of the inset axes.  
        2020/09/16 | MEG | Update to clarify the names of whether variables contain mixtures or soruces.  
        2021/04/13 | MEG | Update so that lons and lats are now rank2 tensors (ie matrices with a lon or lat for each pixel)
        2021/04/13 | MEG | Add option to create_all_ifgs_from_incremental
    
    Stack overview:
        PCA_meg2                                        # do PCA
        maps_tcs_rescale                                # rescale spatial maps from PCA so they have the same range, then rescale time courses so no change.  makes comparison easier.  )
        pca_variance_line                               # plot of variance for each PC direction
        component_plot                                  with PCA sources
        bootstrap_ICA                                   with bootstrapping
        bootstrap_ICA                                   and without bootstrapping
        bootstrapped_sources_to_centrotypes             # run HDBSCAN (clustering), TSNE (2d manifold) and make figure showing this.  Choose source most representative of each cluster (centrotype).  
        plot_2d_interactive_fig                         # interactive figure showing clustering and 2d manifold representaiton.  
        bss_components_inversion                        # inversion to get time coures for each centrotype.  
        component_plot                                  # with ICASAR sources
        r2_arrays_to_googleEarth                        # geocode spatial sources and make a .kmz for use with Google Earth.  
    """
    
    # external functions
    import numpy as np
    import numpy.ma as ma  
    import matplotlib.pyplot as plt
    import shutil                                                                # used to make/remove folders etc
    import os                                                                    # ditto
    import pickle                                                                # to save outputs.  
    from pathlib import Path
    # internal functions    
    from blind_signal_separation_funcitons import PCA_meg2
    from auxiliary_functions import  bss_components_inversion, maps_tcs_rescale, r2_to_r3, r2_arrays_to_googleEarth
    from auxiliary_functions import plot_spatial_signals, plot_temporal_signals, plot_pca_variance_line
    from auxiliary_functions import prepare_point_colours_for_2d, prepare_legends_for_2d, create_all_ifgs, signals_to_master_signal_comparison, plot_source_tc_correlations
    from auxiliary_functions_from_other_repos import plot_2d_interactive_fig, baseline_from_names, update_mask_sources_ifgs


    # Check inputs, unpack either spatial or temporal data, and check for nans
    if temporal_data is None and spatial_data is None:                                                                  # check inputs
        raise Exception("One of either spatial or temporal data must be supplied.  Exiting.  ")
    if temporal_data is not None and spatial_data is not None:
        raise Exception("Only either spatial or temporal data can be supplied, but not both.  Exiting.  ")
      
    if spatial_data is not None:
        mixtures = spatial_data['mixtures_r2']
        mask = spatial_data['mask']
        if 'ifg_dates' in spatial_data:                                                             # dates the ifgs span is optional.  
            ifg_dates = spatial_data['ifg_dates']
        else:
            ifg_dates = None                                                                        # set to None if there are none.  
        spatial = True
    else:
        mixtures = temporal_data['mixtures_r2']
        xvals = temporal_data['xvals']
        spatial = False
    if np.max(np.isnan(mixtures)):
        raise Exception("Unable to proceed as the data ('phUnw') contains Nans.  ")
                        
    # sort out various things for figures, and check input is of the correct form
    if type(out_folder) == str:
        print(f"Trying to conver the 'out_folder' arg which is a string to a pathlib Path.  ")
        out_folder = Path(out_folder)
    fig_kwargs = {"figures" : figures}
    if figures == "png" or figures == "png+window":                                                         # if figures will be png, make 
        fig_kwargs['png_path'] = out_folder                                                                  # this will be passed to various figure plotting functions
    elif figures == 'window' or figures == 'none':
        pass
    else:
        raise ValueError("'figures' should be 'window', 'png', 'png+window', or 'None'.  Exiting...")
        
    if ica_verbose == 'long':
        fastica_verbose = True
    elif ica_verbose == 'short':
        fastica_verbose = False
    else:
        print(f"'ica_verbose should be either 'long' or 'short'.  Setting to 'short' and continuing.  ")
        ica_verbose = 'short'
        fastica_verbose = False

    
    if spatial_data is not None:                                                                                      # if we're working with spatial data, we should check lons and lats as they determine if the ICs will be geocoded.  
        if ('lons' in spatial_data) and ('lats' in spatial_data):                                                       # 
            print(f"As 'lons' and 'lats' have been provided, the ICs will be geocoded.  ")
            if (len(spatial_data['lons'].shape) != 2) or (len(spatial_data['lats'].shape) != 2):
                raise Exception(f"'lons' and 'lats' should be rank 2 tensors (i.e. matrices with a lon or lat for each pixel in the interferogram.  Exiting...  ")
            ge_kmz = True
        elif ('lons' in spatial_data) and ('lats' not in spatial_data):
            raise Exception(f"Either both or neither of 'lons' and 'lats' should be provided, but only 'lons' was.  Exiting...  ")
        elif ('lons' not in spatial_data) and ('lats' in spatial_data):
            raise Exception(f"Either both or neither of 'lons' and 'lats' should be provided, but only 'lats' was.  Exiting...  ")
        else:
            ge_kmz = False
    else:
        ge_kmz = False                                                                                              # if there's no spatial data, assume that we must be working with temporal.  
            
    
    if spatial_data is not None:                                                                                      # if we're working with spatial data, we should check the ifgs and acq dates are the correct lengths as these are easy to confuse.  
        if ifg_dates is not None:
            n_ifgs = spatial_data['mixtures_r2'].shape[0]                                                               # get the number of incremental ifgs
            if n_ifgs != len(spatial_data['ifg_dates']):                                                                # and check it's equal to the list of ifg dates (YYYYMMDD_YYYYMMDD)
                raise Exception(f"There should be an equal number of incremental interferogram and dates (in the form YYYYMMDD_YYYYMMDD), but they appear to be different.  Exiting...")

        
    # create a folder that will be used for outputs
    try:
        print("Trying to remove the existing outputs folder... ", end = '')
        shutil.rmtree(out_folder)                                                                       # try to remove folder
        print('Done!')
    except:
        print("Failed!")                                                                                # 
    try:
        print("Trying to create a new outputs folder... ", end = '')                                    # try to make a new folder
        os.makedirs(out_folder)                                                                         # swtiched to makedirs (from mkdir) to hand paths with intermedeaite folders.          
        print('Done!')
    except:
        print("Failed!")                                                                                          

    n_converge_bootstrapping = bootstrapping_param[0]                 # unpack input tuples
    n_converge_no_bootstrapping = bootstrapping_param[1]
    
    

    # -1: Possibly create all interferograms from incremental
    if create_all_ifgs_flag:
        print(f"Creating all possible interferogram pairs from the incremental interferograms...", end = '')
        _, mixtures, ifg_dates = create_all_ifgs(mixtures, spatial_data['ifg_dates'])                               # if ifg_dates is None, None is also returned.  
        print(" Done!")
    if (spatial) and (ifg_dates is not None):
        temporal_baselines = baseline_from_names(ifg_dates)
    
    # 0: Mean centre the mixtures
    mixtures_mean = np.mean(mixtures, axis = 1)[:,np.newaxis]                                         # get the mean for each ifg (ie along rows.  )
    mixtures_mc = mixtures - mixtures_mean                                                           # mean centre the data (along rows)
    n_mixtures = np.size(mixtures_mc, axis = 0)     
       
    # 1: do sPCA once (and possibly create a figure of the PCA sources)
    print('Performing PCA to whiten the data....', end = "")
    PC_vecs, PC_vals, PC_whiten_mat, PC_dewhiten_mat, x_mc, x_decorrelate, x_white = PCA_meg2(mixtures_mc, verbose = False)    
    if spatial:
        x_decorrelate_rs, PC_vecs_rs = maps_tcs_rescale(x_decorrelate[:n_comp,:], PC_vecs[:,:n_comp])                                           # x decorrlates is n_comps x n_samples (e.g. 5 x 2000)
    else:
        x_decorrelate_rs = x_decorrelate[:n_comp,:]                                                                         # temporally, we don't care about A and W so much, so don't bother to rescale, just select the requierd number of components.  
        PC_vecs_rs =  PC_vecs[:,:n_comp]
        
    if fig_kwargs['figures'] != "none":
        plot_pca_variance_line(PC_vals, title = '01_PCA_variance_line', **fig_kwargs)
        if spatial:
            plot_spatial_signals(x_decorrelate_rs.T, mask, PC_vecs_rs.T, mask.shape, title = '02_PCA_sources_and_tcs', shared = 1, **fig_kwargs)
        else:
            plot_temporal_signals(x_decorrelate_rs, '02_PCA_sources', **fig_kwargs)
    print('Done!')
   
      
    # 2: do ICA multiple times   
    # First with bootstrapping
    A_hist_BS = []                                                                                      # ditto but with bootstrapping
    S_hist_BS = []
    n_ica_converge = 0
    n_ica_fail = 0
    if ica_verbose == 'short' and n_converge_bootstrapping > 0:                                 # if we're only doing short version of verbose, and will be doing bootstrapping
        print(f"FastICA progress with bootstrapping: ", end = '')
    while n_ica_converge < n_converge_bootstrapping:
        S, A, ica_converged = bootstrap_ICA(mixtures_mc, n_comp, bootstrap = True, ica_param = ica_param, verbose = fastica_verbose)                # note that this will perform PCA on the bootstrapped samples, so can be slow.  
        if ica_converged:
            n_ica_converge += 1
            A_hist_BS.append(A)                                     # record results
            S_hist_BS.append(S)                                     # record results
        else:
            n_ica_fail += 1
        if ica_verbose == 'long':
            print(f"sICA with bootstrapping has converged {n_ica_converge} of {n_converge_bootstrapping} times.   \n")              # longer (more info) update to terminal
        else:
            print(f"{int(100*(n_ica_converge/n_converge_bootstrapping))}% ", end = '')                                              # short update to terminal

    # and without bootstrapping
    A_hist_no_BS = []                                                                        # initiate to store time courses without bootstrapping
    S_hist_no_BS = []                                                                        # and recovered sources        
    n_ica_converge = 0                                                                       # reset the counters for the second lot of ica
    n_ica_fail = 0
    if ica_verbose == 'short' and n_converge_no_bootstrapping > 0:                           # if we're only doing short version of verbose, and are actually doing ICA with no bootstrapping
        print(f"FastICA progress without bootstrapping: ", end = '')
    while n_ica_converge < n_converge_no_bootstrapping:
        S, A, ica_converged = bootstrap_ICA(mixtures_mc, n_comp, bootstrap = False, ica_param = ica_param,
                                            X_whitened = x_white, dewhiten_matrix = PC_whiten_mat, verbose = fastica_verbose)               # no bootstrapping, so PCA doesn't need to be run each time and we can pass it the whitened data.  
        
        if ica_converged:
            n_ica_converge += 1
            A_hist_no_BS.append(A)                                     # record results
            S_hist_no_BS.append(S)                                     # record results
        else:
            n_ica_fail += 1
        if ica_verbose == 'long':
            print(f"sICA without bootstrapping has converged {n_ica_converge} of {n_converge_no_bootstrapping} times.   \n",)
        else:
            print(f"{int(100*(n_ica_converge/n_converge_no_bootstrapping))}% ", end = '')
            
        
       
    # 3: change data structure for sources, and compute similarities and distances between them.  
    #A_hist = A_hist_BS + A_hist_no_BS                                                                   # list containing the time courses from each run.  i.e. each is: times x n_components
    S_hist = S_hist_BS + S_hist_no_BS                                                                   # list containing the soures from each run.  i.e.: each os n_components x n_pixels
    
    if spatial:
        sources_all_r2, sources_all_r3 = sources_list_to_r2_r3(S_hist, mask)                            # convert to more useful format.  r2 one is (n_components x n_runs) x n_pixels, r3 one is (n_components x n_runs) x ny x nx, and a masked array
    else:
        sources_all_r2 = S_hist[0]                                                                      # get the sources recovered by the first run
        for S_hist_one in S_hist[1:]:                                                                   # and then loop through the rest
            sources_all_r2 = np.vstack((sources_all_r2, S_hist_one))                                    # stacking them vertically.  
                        
       
    # 4: Do clustering and 2d manifold representation, plus get centrotypes of clusters, and make an interactive plot.   
    S_best, labels_hdbscan, xy_tsne, clusters_by_max_Iq_no_noise, Iq  = bootstrapped_sources_to_centrotypes(sources_all_r2, hdbscan_param, tsne_param)        # do the clustering and project to a 2d plane.  clusters_by_max_Iq_no_noise is an array of which cluster number is best (ie has the highest Iq)
    labels_colours = prepare_point_colours_for_2d(labels_hdbscan, clusters_by_max_Iq_no_noise)                                                                # make a list of colours so that each point with the same label has the same colour, and all noise points are grey
    legend_dict = prepare_legends_for_2d(clusters_by_max_Iq_no_noise, Iq)    
    marker_dict = {'labels' : np.ravel(np.hstack((np.zeros((1, n_comp*n_converge_bootstrapping)), np.ones((1, n_comp*n_converge_no_bootstrapping)))))}        # boostrapped are labelled as 0, and non bootstrapped as 1
    marker_dict['styles'] = ['o', 'x']                                                                                                                        # bootstrapped are 'o's, and non-bootstrapped are 'x's
       
    plot_2d_labels = {'title' : '03_clustering_and_manifold_results',
                      'xlabel' : 'TSNE dimension 1',
                      'ylabel' : 'TSNE dimension 2'}
        
    if spatial:
        plot_2d_labels['title']
        spatial_data_S_all = {'images_r3' : sources_all_r3}                                                                                            # spatial data stored in rank 3 format (ie n_imaces x height x width)
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, spatial_data = spatial_data_S_all,                                                # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side,
                                fig_filename = '03_clustering_and_manifold_results', **fig_kwargs)
    
    else:
        temporal_data_S_all = {'tcs_r2' : sources_all_r2,
                               'xvals'  : temporal_data['xvals'] }                                                                               # make a dictionary of the sources recovered from each run
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, temporal_data = temporal_data_S_all,                                        # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side,
                                fig_filename = '03_clustering_and_manifold_results', **fig_kwargs)

    Iq_sorted = np.sort(Iq)[::-1]               
    n_clusters = S_best.shape[0]                                                                     # the number of sources/centrotypes is equal to the number of clusters    
    
    # 5: Make time courses using centrotypes
    tcs = np.zeros((n_mixtures, n_clusters))                                                        # store time courses as columns    
    source_residuals = np.zeros((n_mixtures,1))                                                     # initiate array to store these
    for i in range(n_mixtures):
        m, residual_centrotypes = bss_components_inversion(S_best, mixtures_mc[i,:])                # fit each of the training ifgs with the chosen ICs
        tcs[i,:] = m                                                                            # store time course
        source_residuals[i,0] = residual_centrotypes                                            # if bootstrapping, as sources come from bootstrapped data they are better for doing the fitting
    print('Done!')
    
 
    # 6: Possibly make figure of the centrotypes (chosen sources) and time courses.  
    if fig_kwargs['figures'] != "none":
        if spatial:
            plot_spatial_signals(S_best.T, mask, tcs.T, mask.shape, title = '04_ICASAR_sourcs_and_tcs', shared = 1, **fig_kwargs)               # plot the chosen sources
        else:
            plot_temporal_signals(S_best, '04_ICASAR_sources', **fig_kwargs)
        
    # 7: Possibly geocode the recovered sources and make a Google Earth file.     
    if ge_kmz:
        print('Creating a Google Earth .kmz of the geocoded independent components... ', end = '')
        S_best_r3 = r2_to_r3(S_best, mask)
        r2_arrays_to_googleEarth(S_best_r3, spatial_data['lons'], spatial_data['lats'], 'IC', out_folder = out_folder)                              # note that lons and lats should be rank 2 (ie an entry for each pixel in the ifgs)
        print('Done!')
    

    # 8: Calculate the correlations between the DEM and the ICs 
    if (spatial_data is not None) and ('dem' in spatial_data) :                                                                                      # if we're working with spatial data, we should check the ifgs and acq dates are the correct lengths as these are easy to confuse.  
        dem_ma = ma.masked_invalid(spatial_data['dem'])                                                                                                             # LiCSBAS dem uses nans, but lets switch to a masked array (with nans masked)
        dem_new_mask, S_best_new_mask, mask_both = update_mask_sources_ifgs(spatial_data['mask'], S_best, ma.getmask(dem_ma), ma.compressed(dem_ma)[np.newaxis,:])  # Odly, the masked DEM is not the same as the masked ifgs, so find pixels we have value for both of
        ic_dem_comparisons = {}
        dem_to_ic_comparisons = signals_to_master_signal_comparison(S_best_new_mask, dem_new_mask, density = True)                                                  # And then we can do kernel density plots for each IC and the DEM
    else:
        dem_to_ic_comparisons = None
        dem_ma = None
    
    # 9: Calculate the correlations between the temporal baselines and timecourses 
    if (spatial_data is not None) and ('temporal_baselines' in locals()) :                                                                                      # if we're working with spatial data, we should check the ifgs and acq dates are the correct lengths as these are easy to confuse.  
        tcs_to_tempbaselines_comparisons = signals_to_master_signal_comparison(tcs.T, np.asarray(temporal_baselines)[np.newaxis,:], density = True)               # And then we can do kernel density plots for each IC and the DEM
    else:
        tcs_to_tempbaselines_comparisons = None
  
    # 10: Plot the results of the two correlations.  
    if (spatial_data is not None):
        if ('dem' in spatial_data) or ('temporal_baselines' in locals()):                                                                   # at least one of the two things we make comparisons against must exist for it to be worth plotting the figure.  
            plot_source_tc_correlations(S_best, mask, dem_ma, dem_to_ic_comparisons, tcs_to_tempbaselines_comparisons, **fig_kwargs)
 
    

    # 11: Save the results: 
    print('Saving the key results as a .pkl file... ', end = '')                                            # note that we don't save S_all_info as it's a huge file.  
    if spatial:
        with open(out_folder / 'ICASAR_results.pkl', 'wb') as f:
            pickle.dump(S_best, f)
            pickle.dump(mask, f)
            pickle.dump(tcs, f)
            pickle.dump(source_residuals, f)
            pickle.dump(Iq_sorted, f)
            pickle.dump(n_clusters, f)
        f.close()
        print("Done!")
    else:                                                                       # if temporal data, no mask to save
        with open(out_folder / 'ICASAR_results.pkl', 'wb') as f:
            pickle.dump(S_best, f)
            pickle.dump(tcs, f)
            pickle.dump(source_residuals, f)
            pickle.dump(Iq_sorted, f)
            pickle.dump(n_clusters, f)
        f.close()
        print("Done!")

    S_all_info = {'sources' : sources_all_r2,                                                                # package into a dict to return
                  'labels' : labels_hdbscan,
                  'xy' : xy_tsne       }
    
    return S_best, tcs, source_residuals, Iq_sorted, n_clusters, S_all_info, mixtures_mean
   

#%%

def bootstrapped_sources_to_centrotypes(sources_r2, hdbscan_param, tsne_param):
    """ Given the products of the bootstrapping, run the 2d manifold and clustering algorithms to create centrotypes.  
    Inputs:
        mixtures_r2 | rank 2 array | all the sources recovered after bootstrapping.  If 5 components and 100 bootstrapped runs, this will be 500 x n_pixels (or n_times)
        hdbscan_param  | tuple | Used to control the clustering (min_cluster_size, min_samples)
        tsne_param     | tuple | Used to control the 2d manifold learning  (perplexity, early_exaggeration)
    Returns:
        S_best | rank 2 array | the recovered sources as row vectors (e.g. 5 x 1230)
        labels_hdbscan | rank 2 array | the cluster number for each of the sources in sources_all_r2 e.g 1000,
        xy_tsne | rank 2 array | the x and y coordinates of where each space is in the 2D space.  e.g. 1000x2
        clusters_by_max_Iq_no_noise | rank 1 array | clusters ranked by quality index (Iq).  e.g. 3,0,1,4,2
        Iq | list | cluster quality index for each cluster.  Entry 0 is Iq (cluster quality index) for the first cluster

    History:
        2020/08/26 | MEG | Created from a script.  
        2021_04_16 | MEG | Remove unused figure arguments.  
    """
    import numpy as np
    import hdbscan                                                               # used for clustering
    from sklearn.manifold import TSNE                                            # t-distributed stochastic neighbour embedding

    
    perplexity = tsne_param[0]                                                   # unpack tuples
    early_exaggeration = tsne_param[1]
    min_cluster_size = hdbscan_param[0]                                              
    min_samples = hdbscan_param[1] 

    # 1: Create the pairwise comparison matrix
    print('\nStarting to compute the pairwise distance matrices....', end = '')
    D, S = pairwise_comparison(sources_r2)
    print('Done!')


    #  2: Clustering with all the recovered sources   
    print('Starting to cluster the sources using HDBSCAN....', end = "")  
    clusterer_precom = hdbscan.HDBSCAN(metric = 'precomputed', min_cluster_size = min_cluster_size, 
                                       min_samples = min_samples, cluster_selection_method = 'leaf')
    labels_hdbscan = clusterer_precom.fit_predict(D)                                                                  # D is n_samples x n_samples, then returns a rank 1 which is the cluster number (ie label) for each source
    Iq = cluster_quality_index(labels_hdbscan, S)                                                                     # calculate the cluster quality index, using S (n_samples x n_samples), and the label for each one
                                                                                                                      # note that Iq is ordered by cluster, so the first value is the cluster quality index for 1st cluster (which is usually labelled -1 and the noise points)
    if np.min(labels_hdbscan) == (-1):                                                                                # if HDBSCAN has identified noise
        Iq = Iq[1:]                                                                                                   # delete the first entry, as this is the Iq of the noise (which isn't a cluster)
    clusters_by_max_Iq_no_noise = np.argsort(Iq)[::-1]                                                                # clusters by best Iqfirst (ie cluster)
    print('Done!')


    # 3:  2d manifold with all the recovered sources
    print('Starting to calculate the 2D manifold representation....', end = "")
    manifold_tsne = TSNE(n_components = 2, metric = 'precomputed', perplexity = perplexity, early_exaggeration = early_exaggeration)
    xy_tsne = manifold_tsne.fit(D).embedding_
    print('Done!' )
    
    # 4: Determine the number of clusters from HDBSCAN
    if np.min(labels_hdbscan) == (-1):                                      # if we have noise (which is labelled as -1 byt HDBSCAN), 
        n_clusters = np.size(np.unique(labels_hdbscan)) - 1                 # noise doesn't count as a cluster so we -1 from number of clusters
    else:
        n_clusters = np.size(np.unique(labels_hdbscan))                     # but if no noise, number of clusters is just number of different labels
        
    if n_clusters == 0:
        print("No clusters have been found.  Often, this is caused by running the FastICA algorithm too few times, or setting"
              "the hdbscan_param 'min_cluster_size' too high.  ")
        return None, labels_hdbscan, xy_tsne, clusters_by_max_Iq_no_noise, Iq
    
    else:
        # 4:  Centrotypes (object that is most similar to all others in the cluster)
        print('Calculating the centrotypes and associated time courses...', end = '')
        S_best_args = np.zeros((n_clusters, 1)).astype(int)                         
        for i, clust_number in enumerate(clusters_by_max_Iq_no_noise):                              # loop through each cluster in order of how good they are (i.e. highest Iq first)
            source_index = np.ravel(np.argwhere(labels_hdbscan == clust_number))                    # get the indexes of sources in this cluster
            S_this_cluster = np.copy(S[source_index, :][:, source_index])                           # similarities for just this cluster
            in_cluster_arg = np.argmax(np.sum(S_this_cluster, axis = 1))                            # the sum of a column of S_this... is the similarity between 1 source and all the others.  Look for the column that's the maximum
            S_best_args[i,0] = source_index[in_cluster_arg]                                         # conver the number in the cluster to the number overall (ie 2nd in cluster is actually 120th source)     
        S_best = np.copy(sources_r2[np.ravel(S_best_args),:])                                       # these are the centrotype sources
    
        return S_best, labels_hdbscan, xy_tsne, clusters_by_max_Iq_no_noise, Iq


     
#%%

def bootstrap_ICA(X, n_comp, bootstrap = True, ica_param = (1e-4, 150), 
                  X_whitened = None, dewhiten_matrix = None, verbose = True):
    """  A function to perform ICA either with or without boostrapping.  
    If not performing bootstrapping, performance can be imporoved by passing the whitened data and the dewhitening matrix
    (so that PCA does not have to be peroformed).  
    
    Inputs:
        X | rank2 array | data as row vectors (ie n_variables x n_samples)
        n_comp | int | number of sources to recover 
        X_whitened | rank2 array | data as row vectors (ie n_variables x n_samples), but whitened (useful if not bootstrapping)
        ica_param | tuple | Used to control ICA, (ica_tol, ica_maxit)
        X_whitened | rank2 array | data as row vectors (e.g. 10 x 20,000 for 10 ifgs of 20000 pixels), but whitened.  Useful to pass to function if not bootstapping as 
                                    this can then be calculated only once.  
        dewhiten_matrix | rank2 array | Converts the time courses recovered when using whitened data back to unwhiteend.  
                                        size is n_ifgs x n_sources.  
                                        X_white = A x S
                                        X = dewhiten x A x S
                                        Needed if not bootstrapping and don't want to  do PCA each time (as above)
        verbose | boolean | If True, the FastICA algorithm returns how many times it took to converge (or if it didn't converge)
    
    Returns:
        S | rank2 array | sources as row vectors (ie n_sources x n_samples)
        A | rank 2 array | time courses as columns (ie n_ifgs x n_sources)
        ica_success | boolean | True is the FastICA algorithm does converge.  
        
    History:
        2020/06/05 | MEG | Written
        2020/06/09 | MEG | Update to able to hand the case in which PCA fails (normally to do with finding the inverse of a matrix)
    
    """
    import numpy as np
    from blind_signal_separation_funcitons import PCA_meg2, fastica_MEG
    from auxiliary_functions import  maps_tcs_rescale
    
    n_loop_max = 1000                                                               # when trying to make bootstrapped samples, if one can't be found after this many attempts, raise an error.  Best left high.  
    
    n_ifgs = X.shape[0]
    
    # 0: do the bootstrapping and determine if we need to do PCA
    if bootstrap:
        pca_needed = True                                                                                            # PCA will always be needed if bootstrapping 
        input_ifg_args = np.arange(n_comp-1)                                                                          # initiate as a crude way to get into the loop
        n_loop = 0                                                                                                   # to count how many goes it takes to generate a good bootstrap sample
        while len(np.unique(input_ifg_args)) < n_comp and n_loop < 100:                                              # try making a list of samples to bootstrap with providing it has enough unique items for subsequent pca to work
            input_ifg_args = np.random.randint(0, n_ifgs, n_ifgs)                                                    # generate indexes of samples to select for bootstrapping 
            n_loop += 1
        if n_loop == n_loop_max:                                                                                            # if we exited beacuse we were stuck in a loop, error message and stop
            raise Exception(f'Unable to bootstrap the data as the number of training data must be sufficently'
                            f' bigger than "n_components" sought that there are "n_components" unique items in'
                            f' a bootsrapped sample.  ')                                                             # error message
        X = X[input_ifg_args, :]                                                                              # bootstrapped smaple
    else:                                                                                                           # if we're not bootstrapping, need to work out if we actually need to do PCA
        if X_whitened is not None and dewhiten_matrix is not None:
            pca_needed = False
        else:
            pca_needed = True
            print(f"Even though bootstrapping is not being used, PCA is being performed.  "
                  f"This step could be sped up significantly by running PCA beforehand and "
                  f"computing 'X_whiten' and 'dewhiten_matrix' only once.  ")
        
    # 1 get whitened data using PCA, if we need to (ie if X_whitened and dewhiten_matrix aren't provided)
    if pca_needed:
        try:
            pca_vecs, _, _, dewhiten_matrix, _, _, X_whitened = PCA_meg2(X, verbose = False)                               # pca on bootstrapped data
            pca_success = True
        except:
            pca_success = False
    else:
        pca_success = True
            
    if pca_success:                                                                                         # If PCA was a success, do ICA (note, if not neeed, success is set to True)
        X_whitened = X_whitened[:n_comp,]                                                                     # reduce dimensionality
        W, S, A_white, _, _, ica_success = fastica_MEG(X_whitened, n_comp=n_comp,  algorithm="parallel",        
                                                   whiten=False, maxit=ica_param[1], tol = ica_param[0], verbose = verbose)          # do ICA
        A = dewhiten_matrix[:,0:n_comp] @ A_white                                         # turn ICA mixing matrix back into a time courses (ie dewhiten)
        S, A = maps_tcs_rescale(S, A)                                                     # rescale so spatial maps have a range or 1 (so easy to compare)
        return S, A, ica_success
    else:                                                                                                   # or if not a success, say that 
        ica_success = False
        return None, None, ica_success



#%%


def pairwise_comparison(sources_r2):
    """ Compte the pairwise distances and similarities for ICA sources.  
        Note that this uses the absolute value of the similarities, so is invariant to sign flips of the data.  
    Inputs:
        sources_r2 | rank 2 array | sources as row vectors
    """
    import numpy as np
    
    S = np.corrcoef(sources_r2)                                                  # Similarity matrix
    S = np.abs(S)                                                                # covariance of 1 and -1 are equivalent for our case
    D = 1 - S                                                                   # convert to dissimilarity    
    return D, S
    
 


#%%

def sources_list_to_r2_r3(sources, mask = None):
    """A function to convert a list of the outputs of multiple ICA runs (which are lists) into rank 2 and rank 3 arrays.  
    Inputs:
        sources | list | list of runs of ica (e.g. 10, or 20 etc.), each item would be n_sources x n_pixels
        mask | boolean | Only needed for two_d.  Converts row vector back to masked array.  
    Outputs:
        sources_r2 | rank 2 array | each source as a row vector (e.g. n_sources_total x n_pixels)
        sources_r3 | rank 3 masked array | each source as a rank 2 image. (e.g. n_souces_total x source_height x source_width )
    History:
        2018_06_29 | MEG | Written
        2020/08/27 | MEG | Update to handle both 1d and 2d signals.  
        2020/09/11 | MEG | Change sources_r3 so that it's now a masked array (sources_r3_ma)
       
    
    """
    import numpy as np
    import numpy.ma as ma
    from auxiliary_functions import col_to_ma
    
    
    n_converge_needed = len(sources)
    n_comp = np.size(sources[0], axis = 0)
    n_pixels = np.size(sources[0], axis = 1)
    
    sources_r2 = np.zeros(((n_converge_needed * n_comp), n_pixels))                # convert from list to one big array
    for i in range(n_converge_needed):    
        sources_r2[i*n_comp:((i*n_comp) + n_comp), :] = sources[i]
    n_sources_total = np.size(sources_r2, axis = 0)

    if mask is not None:
        sources_r3 = ma.zeros((col_to_ma(sources_r2[0,:], mask).shape))[np.newaxis, :, :]               # get the size of one image (so rank 2)
        sources_r3 = ma.repeat(sources_r3, n_sources_total, axis = 0)                                    # and then extend to make rank 3
        for i in range(n_sources_total):
            sources_r3[i,:,:] = col_to_ma(sources_r2[i,:], mask)   
    else:
        sources_r3 = None
    return sources_r2, sources_r3



#%%

def cluster_quality_index(labels, S):
    """
    A function to calculate the cluster quality index (Iq).  If a cluster has only one element in it,
    the cluster quality index is set to nan (np.nan)
    Inputs:
        labels | rank 1 array | label number for each data point
        S | rank 2 array | similiarit between each data point 
    Returns:
        Iq | list | cluster quality index 
    2018_05_28 | written
    2018_05_30 | if clusters have only one point in them, set Iq to 0
    """
    import numpy as np
   
    Iq = []                                                                                         # initiate cluster quality index
    for i in np.unique(labels):                                                                     # loop through each label (there will be as many loops here as there are clusters)
        labels_1cluster = np.ravel(np.argwhere(labels == i))
        if np.size(labels_1cluster) < 2:                                                            # check if cluster has only one point in it
            Iq_temp = np.nan
        else:
            S_intra = np.copy(S[labels_1cluster, :][:,labels_1cluster])                                 # The similarties between the items in the cluster
            S_intra = np.where(np.eye(np.size(S_intra, axis = 0)) == 1, np.nan, S_intra)                # change the diagonals to nans
            S_inter = np.copy(S[labels_1cluster, :])                                                    # The similarties between the items in the cluster and those out of the cluster
            S_inter = np.delete(S_inter, labels_1cluster, axis = 1)                                     # horizontal axis remove similarities with itself
            Iq_temp = np.nanmean(S_intra) - np.mean(S_inter)                                         # Iq is the difference between the mean of the distances inside the cluster, and the mean distance between items in the cluster and out of the cluster
        Iq.append(Iq_temp)                                                                          # append whichever value of Iq (np.nan or a numeric value)
    return Iq

  