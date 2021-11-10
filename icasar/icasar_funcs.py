# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:23:10 2018

@author: eemeg
"""



def ICASAR(n_comp, spatial_data = None, temporal_data = None, figures = "window", 
           bootstrapping_param = (200,0), ica_param = (1e-4, 150), tsne_param = (30,12), hdbscan_param = (35,10),
           out_folder = './ICASAR_results/', ica_verbose = 'long', inset_axes_side = {'x':0.1, 'y':0.1}, 
           create_all_ifgs_flag = False, max_n_all_ifgs = 1000, load_fastICA_results = False):
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
        spatial_data | dict or None | Required: 
                                         displacement_r2 | rank 2 array | row vectors of the ifgs
                                         mask  | rank 2 array | mask to conver the row vectors to rank 2 masked arrays.  
                                      Optional (ie don't have to exist in the dictionary):
                                         ifg_dates | list | dates of the interferograms in the form YYYYMMDD_YYYYMMDD.  If supplied, IC strength vs temporal baseline plots will be produced.  
                                         lons | rank 2 array | lons of each pixel in the image.  Changed to rank 2 in version 2.0, from rank 1 in version 1.0  .  If supplied, ICs will be geocoded as kmz.  
                                         lats | rank 2 array | lats of each pixel in the image. Changed to rank 2 in version 2.0, from rank 1 in version 1.0
                                         dem | rank 2 array | height in metres of each pixel in the image.  If supplied, IC vs dem plots will be produced.  
                                         
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
        max_n_all_ifgs | If after creating all the ifgs there are more than this number, select only this many at random.  Useful as the number of ifgs created grows with the square of the number of ifgs.  
        load_fastICA_results | boolean | The multiple runs of FastICA are slow, so if now paramters are being changed here, previous runs can be reloaded.  

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
        2021_10_07 | MEG | Add option to limit the number of ifgs created from incremental. (e.g. if 5000 are generated but default value of 1000 is used, 1000 will be randomly chosen from the 5000)
        2021_10_20 | MEG | Also save the 2d position of each source, and its HDBSSCAN label in the .pickle file.  
    
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
    from icasar.blind_signal_separation import PCA_meg2
    from icasar.aux import  bss_components_inversion, maps_tcs_rescale, r2_to_r3, r2_arrays_to_googleEarth, dem_and_temporal_source_figure
    from icasar.aux import plot_spatial_signals, plot_temporal_signals, plot_pca_variance_line
    from icasar.aux import prepare_point_colours_for_2d, prepare_legends_for_2d, create_all_ifgs, signals_to_master_signal_comparison, plot_source_tc_correlations
    from icasar.aux2 import plot_2d_interactive_fig, baseline_from_names, update_mask_sources_ifgs

    # -10: Check for an unusual combination of inputs:
    if (create_all_ifgs_flag) and ('ifg_dates' not in spatial_data.keys()):
            raise Exception(f"'ifg_dates' (in the form yyyymmdd_yyyymmdd) are usually optional, but not if the 'create_all_ifgs_flag' is set to True.  Exiting.  " )

    # -9 Check inputs, unpack either spatial or temporal data, and check for nans
    if temporal_data is None and spatial_data is None:                                                                  # check inputs
        raise Exception("One of either spatial or temporal data must be supplied.  Exiting.  ")
    if temporal_data is not None and spatial_data is not None:
        raise Exception("Only either spatial or temporal data can be supplied, but not both.  Exiting.  ")
      
    if spatial_data is not None:                                                                    # if we have spatial data
        mixtures = spatial_data['mixtures_r2']                                                      # these are the mixtures we'll perform PCA and ICA on
        mask = spatial_data['mask']                                                                 # the mask that converts row vector mixtures into 2d (rank 2) arrays.  
        if 'ifg_dates' in spatial_data:                                                             # dates the ifgs span is optional.  
            ifg_dates = spatial_data['ifg_dates']
        else:
            ifg_dates = None                                                                        # set to None if there are none.  
        spatial = True
    if temporal_data is not None:                                                                    # if we have temporal data
        mixtures = temporal_data['mixtures_r2']                                                     # these are the mixture we'll perform PCA and ICA on.  
        xvals = temporal_data['xvals']
        spatial = False
    if np.max(np.isnan(mixtures)):
        raise Exception("Unable to proceed as the data ('phUnw') contains Nans.  ")
                        
    #-8:  sort out various things for figures, and check input is of the correct form
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
    
    # -7: Check argument
    if ica_verbose == 'long':
        fastica_verbose = True
    elif ica_verbose == 'short':
        fastica_verbose = False
    else:
        print(f"'ica_verbose should be either 'long' or 'short'.  Setting to 'short' and continuing.  ")
        ica_verbose = 'short'
        fastica_verbose = False

    
    # -6: Determine if we have both lons and lats and so can geocode the ICs (ge_kmz = True), and check both rank 2
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
            
    # -5: Check the temporal dimension of the time series and the ifg_dates agree
    if spatial_data is not None:                                                                                      # if we're working with spatial data, we should check the ifgs and acq dates are the correct lengths as these are easy to confuse.  
        if ifg_dates is not None:
            n_ifgs = spatial_data['mixtures_r2'].shape[0]                                                               # get the number of incremental ifgs
            if n_ifgs != len(spatial_data['ifg_dates']):                                                                # and check it's equal to the list of ifg dates (YYYYMMDD_YYYYMMDD)
                raise Exception(f"There should be an equal number of incremental interferogram and dates (in the form YYYYMMDD_YYYYMMDD), but they appear to be different.  Exiting...")
    
    # -4: Check the sizes of the spatial data inputs, and assign None to the DEM if it doesn't exist
    if spatial_data is not None:                                                                                      # if we're working with spatial data
        spatial_data_r2_arrays = ['mask', 'dem', 'lons', 'lats']                                                      # we need to check the spatial data is the correct resolution (ie all the same)
        spatial_data_r2_arrays_present = list(spatial_data.keys())                                                      # we alse need to determine which of these spatial data we actually have.  
        spatial_data_r2_arrays = [i for i in spatial_data_r2_arrays if i in spatial_data_r2_arrays_present]             # remove any from the check list incase they're not provided.  
        for spatial_data_r2_array1 in spatial_data_r2_arrays:                                                           # first loop through each spatial data
            for spatial_data_r2_array2 in spatial_data_r2_arrays:                                                       # second loo through each spatial data
                if spatial_data[spatial_data_r2_array1].shape != spatial_data[spatial_data_r2_array2].shape:            # check the size is equal
                    raise Exception(f"All the spatial data should be the same size, but {spatial_data_r2_array1} is of shape {spatial_data[spatial_data_r2_array1].shape}, "
                                    f"and {spatial_data_r2_array2} is of shape {spatial_data[spatial_data_r2_array2].shape}.  Exiting.")
        if 'dem' not in spatial_data_r2_arrays_present:                                                                  #  the dem is not compulsory
            spatial_data['dem'] = None                                                                                    # so set it to None if not available.  
        
    # -3: Possibly change the matplotlib backend.  
    if figures == 'png':
        plt.switch_backend('agg')                                                                       # with this backend, no windows are created during figure creation.  


    # -2: create a folder that will be used for outputs
    if os.path.exists(out_folder):                                                                      # see if the folder we'll write to exists.  
        if load_fastICA_results:                                                                        # we will need the .pkl of results from a previous run, so can't just delete the folder.  
            existing_files = os.listdir(out_folder)                                                     # get all the ICASAR outputs.  
            print(f"As 'load_fastICA' is set to True, all but the FastICA_results.pkl file will be deleted.   ")
            for existing_file in existing_files:
                if existing_file == 'FastICA_results.pkl':                                              # if it's the results from the time consuming FastICA runs...
                    pass                                                                                # ignore it    
                else:
                    os.remove(out_folder / existing_file)                                               # but if not, delete it.  
        else:
            print("Removing the existing outputs directory and creating a new empty one... ", end = '')                                 # if we don't care about the FastICA results file, just delete the folder and then make a new one.  
            shutil.rmtree(out_folder)                                                                   # try to remove folder
            os.mkdir(out_folder)
            print("Done.")
    else:
        os.mkdir(out_folder)                                                                            # if it never existed, make it.  
                                                                                           

    n_converge_bootstrapping = bootstrapping_param[0]                 # unpack input tuples
    n_converge_no_bootstrapping = bootstrapping_param[1]  
    

    # -1: Possibly create all interferograms from incremental
    if create_all_ifgs_flag:
        print(f"Creating all possible interferogram pairs from the incremental interferograms...", end = '')
        mixtures_incremental = np.copy(mixtures)                                                                                # make a copy of the originals that we can use to calculate the time courses.  
        mixtures_incremental_mc = mixtures_incremental - np.mean(mixtures_incremental, axis = 1)[:, np.newaxis]                 # mean centre the mixtures (i.e. the mean of each image is 0, so removes the effect of a reference pixel)
        mixtures, ifg_dates = create_all_ifgs(mixtures_incremental, spatial_data['ifg_dates'], max_n_all_ifgs)                               # if ifg_dates is None, None is also returned.  
        print(" Done!")          
    
    # 0: Mean centre the mixtures
    mixtures_mean = np.mean(mixtures, axis = 1)[:,np.newaxis]                                         # get the mean for each ifg (ie along rows.  )
    mixtures_mc = mixtures - mixtures_mean                                                           # mean centre the data (along rows)
    n_mixtures = np.size(mixtures_mc, axis = 0)     
       
    # 1: do sPCA once (and possibly create a figure of the PCA sources)
    print('Performing PCA to whiten the data....', end = "")
    PC_vecs, PC_vals, PC_whiten_mat, PC_dewhiten_mat, x_mc, x_decorrelate, x_white = PCA_meg2(mixtures_mc, verbose = False)    
    if spatial:
        x_decorrelate_rs, PC_vecs_rs = maps_tcs_rescale(x_decorrelate[:n_comp,:], PC_vecs[:,:n_comp])                          # rescale to new desicred range, and truncate to desired number of components.  
    else:
        x_decorrelate_rs = x_decorrelate[:n_comp,:]                                                                            # truncate to desirec number of components
        PC_vecs_rs =  PC_vecs[:,:n_comp]
    print('Done!')    
    if fig_kwargs['figures'] != "none":
        plot_pca_variance_line(PC_vals, title = '01_PCA_variance_line', **fig_kwargs)
        if spatial:
            plot_spatial_signals(x_decorrelate_rs.T, mask, PC_vecs_rs.T, mask.shape, title = '02_PCA_sources_and_tcs', shared = 1, **fig_kwargs)                      # the usual plot of the sources and their time courses (ie contributions to each ifg)                              
            if ifg_dates is not None:                                                                                                                       # if we have ifg_dates
                temporal_baselines = baseline_from_names(ifg_dates)                                                                                         # we can use these to calcaulte temporal baselines
                spatial_data_temporal_info_pca = {'temporal_baselines' : temporal_baselines, 'tcs' : PC_vecs_rs}                                                             # and use them in the following figure
            else:
                spatial_data_temporal_info_pca = None                                                                                                                        # but we might also not have them
            dem_and_temporal_source_figure(x_decorrelate_rs, spatial_data['mask'], fig_kwargs, spatial_data['dem'],                                         # also compare the sources to the DEM, and the correlation between their time courses and the temporal baseline of each interferogram.  
                                           spatial_data_temporal_info_pca, fig_title = '03_PCA_source_correlations')
        else:
            plot_temporal_signals(x_decorrelate_rs, '02_PCA_sources', **fig_kwargs)
    
   
    
    # 2: Make or load the results of the multiple ICA runs.  
    if load_fastICA_results:
        print(f"Loading the results of multiple FastICA runs.  ")
        try:
            with open(out_folder / 'FastICA_results.pkl', 'rb') as f:
                S_hist = pickle.load(f)   
                A_hist = pickle.load(f)
        except:
            print(f"Failed to open the results from the previous runs of FastICA.  Switching 'load_fastICA_results' to False and trying to continue anyway.  ")
            load_fastICA_results = False
    if not load_fastICA_results:
       print(f"No results were found for the multiple ICA runs, so these will now be performed.  ")
       S_hist, A_hist = perform_multiple_ICA_runs(n_comp, mixtures_mc, bootstrapping_param, ica_param,
                                                  x_white, PC_dewhiten_mat, ica_verbose) 
       with open(out_folder / 'FastICA_results.pkl', 'wb') as f:
            pickle.dump(S_hist, f)
            pickle.dump(A_hist, f)
            
    if S_hist[0].shape[1] != np.sum(1-spatial_data['mask']):
        raise Exception(f"There are {S_hist[0].shape[1]} pixels in the ICASAR sources that have been loaded, but"
                        f" {np.sum(1-spatial_data['mask'])} pixels in the current mask.  This normally happens when the"
                        f" FastICA results that are being loaded are from a different set of data.  If not, something "
                        f" is inconsitent with the mask and the coherent pixels.  Exiting.  ")

    
    # 3: Convert the sources from lists from each run to a single matrix.  
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
       
    plot_2d_labels = {'title' : '04_clustering_and_manifold_results',
                      'xlabel' : 'TSNE dimension 1',
                      'ylabel' : 'TSNE dimension 2'}
        
    if spatial:
        plot_2d_labels['title']
        spatial_data_S_all = {'images_r3' : sources_all_r3}                                                                                            # spatial data stored in rank 3 format (ie n_imaces x height x width)
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, spatial_data = spatial_data_S_all,                                                # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side,
                                fig_filename = plot_2d_labels['title'], **fig_kwargs)
    
    else:
        temporal_data_S_all = {'tcs_r2' : sources_all_r2,
                               'xvals'  : temporal_data['xvals'] }                                                                               # make a dictionary of the sources recovered from each run
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, temporal_data = temporal_data_S_all,                                        # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side,
                                fig_filename = plot_2d_labels['title'], **fig_kwargs)

    Iq_sorted = np.sort(Iq)[::-1]               
    n_clusters = S_best.shape[0]                                                                     # the number of sources/centrotypes is equal to the number of clusters    
    
    # 5: Make time courses using centrotypes (i.e. S_best, the spatial patterns found by ICA)
    if create_all_ifgs_flag:
        inversion_results = bss_components_inversion(S_best, [mixtures_incremental_mc, mixtures_mc])                       # invert to fit both the incremental and all possible ifgs.  
        tcs_all = inversion_results[1]['tcs'].T
    else:
        inversion_results = bss_components_inversion(S_best, [mixtures_mc])                                                 # invert to fit the incremetal ifgs.  
        tcs_all = inversion_results[0]['tcs'].T
    source_residuals = inversion_results[0]['residual']        
    tcs = inversion_results[0]['tcs'].T

 
    # 6: Possibly make figure of the centrotypes (chosen sources) and time courses.  
    if fig_kwargs['figures'] != "none":
        if spatial:
            plot_spatial_signals(S_best.T, mask, tcs.T, mask.shape, title = '05_ICASAR_sourcs_and_tcs', shared = 1, **fig_kwargs)               # plot the chosen sources
        else:
            plot_temporal_signals(S_best, '04_ICASAR_sources', **fig_kwargs)
        
    # 7: Possibly geocode the recovered sources and make a Google Earth file.     
    if ge_kmz:
        #import pdb; pdb.set_trace()
        print('Creating a Google Earth .kmz of the geocoded independent components... ', end = '')
        S_best_r3 = r2_to_r3(S_best, mask)
        r2_arrays_to_googleEarth(S_best_r3, spatial_data['lons'], spatial_data['lats'], 'IC', out_folder = out_folder)                              # note that lons and lats should be rank 2 (ie an entry for each pixel in the ifgs)
        print('Done!')
    

    # 8: Calculate the correlations between the DEM and the ICs, and the ICs time courses and the temporal baselines of the interferograms.  
    if (spatial_data is not None):
        #import pdb; pdb.set_trace()
        
        if ifg_dates is not None:                                                                                                                       # if we have ifg_dates
            spatial_data_temporal_info_ica = {'temporal_baselines' : temporal_baselines, 'tcs' : tcs_all}                                                             # use them in the following figure.  Note that time courses here are from pca
        else:
            spatial_data_temporal_info_ica = None                                                                                                                        # but we might also not have them
        dem_and_temporal_source_figure(S_best, spatial_data['mask'], fig_kwargs, spatial_data['dem'],                                                 # also compare the sources to the DEM, and the correlation between their time courses and the temporal baseline of each interferogram.  
                                       spatial_data_temporal_info_ica, fig_title = '06_ICA_source_correlations')
        
        
        

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
            pickle.dump(xy_tsne, f)
            pickle.dump(labels_hdbscan, f)
        f.close()
        print("Done!")
    else:                                                                       # if temporal data, no mask to save
        with open(out_folder / 'ICASAR_results.pkl', 'wb') as f:
            pickle.dump(S_best, f)
            pickle.dump(tcs, f)
            pickle.dump(source_residuals, f)
            pickle.dump(Iq_sorted, f)
            pickle.dump(n_clusters, f)
            pickle.dump(xy_tsne, f)
            pickle.dump(labels_hdbscan, f)
        f.close()
        print("Done!")

    S_all_info = {'sources' : sources_all_r2,                                                                # package into a dict to return
                  'labels' : labels_hdbscan,
                  'xy' : xy_tsne       }
    
    return S_best, tcs, source_residuals, Iq_sorted, n_clusters, S_all_info, mixtures_mean
   

#%%


def LiCSBAS_to_ICASAR(LiCSBAS_out_folder, filtered = False, figures = False, n_cols=5, crop_pixels = None, return_r3 = False, 
                      ref_area = False):
    """ A function to prepare the outputs of LiCSBAS for use with LiCSALERT.
    LiCSBAS uses nans for masked areas - here these are converted to masked arrays.   Can also create three figures: 1) The Full LiCSBAS ifg, and the area
    that it has been cropped to 2) The cumulative displacement 3) The incremental displacement.  

    Inputs:
        h5_file | string | path to h5 file.  e.g. cum_filt.h5
        figures | boolean | if True, make figures
        n_cols  | int | number of columns for figures.  May want to lower if plotting a long time series
        crop_pixels | tuple | coords to crop images to.  x then y, 00 is top left.  e.g. (10, 500, 600, 900).  
                                x_start, x_stop, y_start, y_stop, No checking that inputted values make sense.  
                                Note, generally better to have cropped (cliped in LiCSBAS language) to the correct area in LiCSBAS_for_LiCSAlert
        return_r3 | boolean | if True, the rank 3 data is also returns (n_ifgs x height x width).  Not used by ICASAR, so default is False
        ref_area | boolean | If True, the reference area (in pixels, x then y) used by LiCSBAS is extracted and returned to the user.  

    Outputs:
        displacment_r3 | dict | Keys: cumulative, incremental.  Stored as masked arrays.  Mask should be consistent through time/interferograms
                                Also lons and lats, which are the lons and lats of all pixels in the images (ie rank2, and not column or row vectors)    
                                Also Dem, mask, and  E N U (look vector components in east north up diretcion)
        displacment_r2 | dict | Keys: cumulative, incremental, mask.  Stored as row vectors in arrays.  
                                Also lons and lats, which are the lons and lats of all pixels in the images (ie rank2, and not column or row vectors)    
                                Also Dem, mask, and  E N U (look vector components in east north up diretcion)
        tbaseline_info | dict| imdates : acquisition dates as strings
                              daisy_chain : names of the daisy chain of ifgs, YYYYMMDD_YYYYMMDD
                              baselines : temporal baselines of incremental ifgs

    2019/12/03 | MEG | Written
    2020/01/13 | MEG | Update depreciated use of dataset.value to dataset[()] when working with h5py files from LiCSBAS
    2020/02/16 | MEG | Add argument to crop images based on pixel, and return baselines etc
    2020/11/24 | MEG | Add option to get lons and lats of pixels.  
    2021/04/15 | MEG | Update lons and lats to be packaged into displacement_r2 and displacement_r3
    2021_04_16 | MEG | Add option to also open the DEM that is in the .hgt file.  
    2021_05_07 | MEG | Change the name of baseline_info to tbaseline_info to be consistent with LiCSAlert
    2021_09_22 | MEG | Add functionality to extract the look vector componenets (ENU files)
    2021_09_23 | MEG | Add option to extract where the LiCSBAS reference area is.  
    2021_09_28 | MEG | Fix cropping option.  
    """

    import h5py as h5
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    import os
    import re
    import pathlib
    #from pathlib import Path
    
    from icasar.aux2 import add_square_plot
    from icasar.aux import col_to_ma
    
    

    def rank3_ma_to_rank2(ifgs_r3, consistent_mask = False):
        """A function to take a time series of interferograms stored as a rank 3 array,
        and convert it into the ICA(SAR) friendly format of a rank 2 array with ifgs as
        row vectors, and an associated mask.

        For use with ICA, the mask must be consistent (ie the same pixels are masked throughout the time series).

        Inputs:
            ifgs_r3 | r3 masked array | ifgs in rank 3 format
            consistent_mask | boolean | If True, areas of incoherence are consistent through the whole stack
                                        If false, a consistent mask will be made.  N.b. this step can remove the number of pixels dramatically.
        """

        n_ifgs = ifgs_r3.shape[0]
        # 1: Deal with masking
        mask_coh_water = ifgs_r3.mask                                                                                               #get the mask as a rank 3, still boolean
        if consistent_mask:
            mask_coh_water_consistent = mask_coh_water[0,]                                                                          # if all ifgs are masked in the same way, just grab the first one
        else:
            mask_coh_water_sum = np.sum(mask_coh_water, axis = 0)                                                                   # sum to make an image that shows in how many ifgs each pixel is incoherent
            mask_coh_water_consistent = np.where(mask_coh_water_sum == 0, np.zeros(mask_coh_water_sum.shape),
                                                                          np.ones(mask_coh_water_sum.shape)).astype(bool)           # make a mask of pixels that are never incoherent
        ifgs_r3_consistent = ma.array(ifgs_r3, mask = ma.repeat(mask_coh_water_consistent[np.newaxis,], n_ifgs, axis = 0))          # mask with the new consistent mask

        # 2: Convert from rank 3 to rank 2
        n_pixs = ma.compressed(ifgs_r3_consistent[0,]).shape[0]                                                        # number of non-masked pixels
        ifgs_r2 = np.zeros((n_ifgs, n_pixs))
        for ifg_n, ifg in enumerate(ifgs_r3_consistent):
            ifgs_r2[ifg_n,:] = ma.compressed(ifg)

        return ifgs_r2, mask_coh_water_consistent


    def ts_quick_plot(ifgs_r3, title):
        """
        A quick function to plot a rank 3 array of ifgs.
        Inputs:
            title | string | title
        """
        n_ifgs = ifgs_r3.shape[0]
        n_rows = int(np.ceil(n_ifgs / n_cols))
        fig1, axes = plt.subplots(n_rows,n_cols)
        fig1.suptitle(title)
        for n_ifg in range(n_ifgs):
            ax=np.ravel(axes)[n_ifg]                                                                            # get axes on it own
            matrixPlt = ax.imshow(ifgs_r3[n_ifg,],interpolation='none', aspect='equal')                         # plot the ifg
            ax.set_xticks([])
            ax.set_yticks([])
            fig1.colorbar(matrixPlt,ax=ax)                                                                       
            ax.set_title(f'Ifg: {n_ifg}')
        for axe in np.ravel(axes)[(n_ifgs):]:                                                                   # delete any unused axes
            axe.set_visible(False)

    def daisy_chain_from_acquisitions(acquisitions):
        """Given a list of acquisiton dates, form the names of the interferograms that would create a simple daisy chain of ifgs.  
        Inputs:
            acquisitions | list | list of acquistiion dates in form YYYYMMDD
        Returns:
            daisy_chain | list | names of daisy chain ifgs, in form YYYYMMDD_YYYYMMDD
        History:
            2020/02/16 | MEG | Written
        """
        daisy_chain = []
        n_acqs = len(acquisitions)
        for i in range(n_acqs-1):
            daisy_chain.append(f"{acquisitions[i]}_{acquisitions[i+1]}")
        return daisy_chain
    
        
    def baseline_from_names(names_list):
        """Given a list of ifg names in the form YYYYMMDD_YYYYMMDD, find the temporal baselines in days_elapsed
        Inputs:
            names_list | list | in form YYYYMMDD_YYYYMMDD
        Returns:
            baselines | list of ints | baselines in days
        History:
            2020/02/16 | MEG | Documented 
        """
        from datetime import datetime
                
        baselines = []
        for file in names_list:
            master = datetime.strptime(file.split('_')[-2], '%Y%m%d')   
            slave = datetime.strptime(file.split('_')[-1][:8], '%Y%m%d')   
            baselines.append(-1 *(master - slave).days)    
        return baselines
    
    def create_lon_lat_meshgrids(corner_lon, corner_lat, post_lon, post_lat, ifg):
        """ Return a mesh grid of the longitudes and latitues for each pixels.  Not tested!
        I think Corner is the top left, but not sure this is always the case
        """
        ny, nx = ifg.shape
        x = corner_lon +  (post_lon * np.arange(nx))
        y = corner_lat +  (post_lat * np.arange(ny))
        xx, yy = np.meshgrid(x,y)
        geocode_info = {'lons_mg' : xx,
                        'lats_mg' : yy}
        return geocode_info

    def get_param_par(mlipar, field):
        """
        Get parameter from mli.par or dem_par file. Examples of fields are;
         - range_samples
         - azimuth_lines
         - range_looks
         - azimuth_looks
         - range_pixel_spacing (m)
         - azimuth_pixel_spacing (m)
         - radar_frequency  (Hz)
        """
        import subprocess as subp
        value = subp.check_output(['grep', field,mlipar]).decode().split()[1].strip()
        return value
    
        
    def read_img(file, length, width, dtype=np.float32, endian='little'):
        """
        Read image data into numpy array.
        endian: 'little' or 'big' (not 'little' is regarded as 'big')
        """
        if endian == 'little':
            data = np.fromfile(file, dtype=dtype).reshape((length, width))
        else:
            data = np.fromfile(file, dtype=dtype).byteswap().reshape((length, width))
        return data

    # -1: Check for common argument errors:
    if not isinstance(LiCSBAS_out_folder, pathlib.PurePath):
        raise Exception(f"'LiCSBAS_out_folder' must be a pathlib Path, but instead is a {type(LiCSBAS_out_folder)}. Exiting.  ")
    
    # 0: Work out the names of LiCSBAS folders - not tested exhaustively! 
    LiCSBAS_folders = {}
    LiCSBAS_folders['all'] = os.listdir(LiCSBAS_out_folder)
    for LiCSBAS_folder in LiCSBAS_folders['all']:
        if bool(re.match(re.compile('TS_.'), LiCSBAS_folder)):                                                   # the timeseries output, which is named depending on mutlitlooking and clipping.  
            LiCSBAS_folders['TS_'] = LiCSBAS_folder
        else:
            pass
        if re.match(re.compile('GEOCml.+clip'), LiCSBAS_folder):                                            # see if there is a folder of multilooked and clipped
            LiCSBAS_folders['ifgs'] = LiCSBAS_folder
        elif re.match(re.compile('GEOCml.+'), LiCSBAS_folder):                                            # see if there is a folder of multilooked and clipped
            LiCSBAS_folders['ifgs'] = LiCSBAS_folder
        elif re.match(re.compile('GEOC'), LiCSBAS_folder):                                            # see if there is a folder of multilooked and clipped
            LiCSBAS_folders['ifgs'] = LiCSBAS_folder
        else:
            pass
    if 'TS_' not in LiCSBAS_folders:
        raise Exception(f"Unable to find the TS_* folder that contains the .h5 files with the LiCSBAS results.  Exiting.  ")
    

    # 1: Open the h5 file with the incremental deformation in.  
    displacement_r3 = {}                                                                                        # here each image will 1 x width x height stacked along first axis
    displacement_r2 = {}                                                                                        # here each image will be a row vector 1 x pixels stacked along first axis
    tbaseline_info = {}

    if filtered:
        cumh5 = h5.File(LiCSBAS_out_folder / LiCSBAS_folders['TS_'] / 'cum_filt.h5' ,'r')                       # either open the filtered file from LiCSBAS
    else:
        cumh5 = h5.File(LiCSBAS_out_folder / LiCSBAS_folders['TS_'] / 'cum.h5' ,'r')                            # or the non filtered file from LiCSBAS
    tbaseline_info["acq_dates"] = cumh5['imdates'][()].astype(str).tolist()                                        # get the acquisition dates
    cumulative = cumh5['cum'][()]                                                                                # get cumulative displacements as a rank3 numpy array
    cumulative *= 0.001                                                                                             # LiCSBAS default is mm, convert to m
    
    if ref_area:
       ref_str = cumh5['refarea'][()] 
       ref_xy = {'x_start' : int(ref_str.split('/')[0].split(':')[0]),                                            # convert the correct part of the string to an integer
                 'x_stop' : int(ref_str.split('/')[0].split(':')[1]),
                 'y_start' : int(ref_str.split('/')[1].split(':')[0]),
                 'y_stop' : int(ref_str.split('/')[1].split(':')[1])}
     
    
    # 2: Mask the data  
    mask_coh_water = np.isnan(cumulative)                                                                       # get where masked
    displacement_r3["cumulative"] = ma.array(cumulative, mask=mask_coh_water)                                   # rank 3 masked array of the cumulative displacement
    displacement_r3["incremental"] = np.diff(displacement_r3['cumulative'], axis = 0)                           # displacement between each acquisition - ie incremental
    if displacement_r3["incremental"].mask.shape == ():                                                         # in the case where no pixels are masked, the diff operation on the mask collapses it to nothing.  
        displacement_r3["incremental"].mask = mask_coh_water[1:]                                                # in which case, we can recreate the mask from the rank3 mask, but dropping one from the first dimension as incremental is always one smaller than cumulative.  
    n_im, length, width = displacement_r3["cumulative"].shape                                   

    # if figures:                                                 
    #     ts_quick_plot(displacement_r3["cumulative"], title = 'Cumulative displacements')
    #     ts_quick_plot(displacement_r3["incremental"], title = 'Incremental displacements')

    displacement_r2['cumulative'], displacement_r2['mask'] = rank3_ma_to_rank2(displacement_r3['cumulative'])      # convert from rank 3 to rank 2 and a mask
    displacement_r2['incremental'], _ = rank3_ma_to_rank2(displacement_r3['incremental'])                          # also convert incremental, no need to also get mask as should be same as above

    # 3: work with the acquisiton dates to produces names of daisy chain ifgs, and baselines
    tbaseline_info["ifg_dates"] = daisy_chain_from_acquisitions(tbaseline_info["acq_dates"])
    tbaseline_info["baselines"] = baseline_from_names(tbaseline_info["ifg_dates"])
    tbaseline_info["baselines_cumulative"] = np.cumsum(tbaseline_info["baselines"])                                                            # cumulative baslines, e.g. 12 24 36 48 etc
    
    # 4: get the lons and lats of each pixel in the ifgs
    geocode_info = create_lon_lat_meshgrids(cumh5['corner_lon'][()], cumh5['corner_lat'][()], 
                                            cumh5['post_lon'][()], cumh5['post_lat'][()], displacement_r3['incremental'][0,:,:])             # create meshgrids of the lons and lats for each pixel
    displacement_r2['lons'] = geocode_info['lons_mg']                                                                                        # add to the displacement dict
    displacement_r2['lats'] = geocode_info['lats_mg']
    displacement_r3['lons'] = geocode_info['lons_mg']                                                                                        # add to the displacement dict (rank 3 one)
    displacement_r3['lats'] = geocode_info['lats_mg']

    # 4: Open the parameter file to get the number of pixels in width and height (though this should agree with above)
    try:
        width = int(get_param_par(LiCSBAS_out_folder / LiCSBAS_folders['ifgs'] / 'slc.mli.par', 'range_samples'))
        length = int(get_param_par(LiCSBAS_out_folder / LiCSBAS_folders['ifgs'] / 'slc.mli.par', 'azimuth_lines'))
    except:
        print(f"Failed to open the 'slc.mli.par' file, so taking the width and length of the image from the h5 file and trying to continue.  ")
        (_, length, width) = cumulative.shape
       
    # 5: get the DEM
    try:
        dem = read_img(LiCSBAS_out_folder / LiCSBAS_folders['ifgs'] / 'hgt', length, width)
        displacement_r2['dem'] = dem                                                                      # and added to the displacement dict in the same was as the lons and lats
        displacement_r3['dem'] = dem                                                                      # 
    except:
        print(f"Failed to open the DEM from the hgt file for this volcano, but trying to continue anyway.")
    
    # 6: Get the E N U files (these are the components of the ground to satellite look vector in east north up directions.  )   
    try:
        for component in ['E', 'N', 'U']:
            look_vector_component = read_img(LiCSBAS_out_folder / LiCSBAS_folders['ifgs'] / f"{component}.geo", length, width)
            displacement_r2[component] = look_vector_component
            displacement_r3[component] = look_vector_component
    except:
        print(f"Failed to open the E N U files (look vector components), but trying to continue anyway.")
        
    if crop_pixels is not None:
        print(f"Cropping the images in x from {crop_pixels[0]} to {crop_pixels[1]} "
              f"and in y from {crop_pixels[2]} to {crop_pixels[3]} (NB matrix notation - 0,0 is top left).  ")
        
        if figures:
            ifg_n_plot = 1                                                                                      # which number ifg to plot.  Shouldn't need to change.  
            title = f'Cropped region, ifg {ifg_n_plot}'
            fig_crop, ax = plt.subplots()
            fig_crop.canvas.set_window_title(title)
            ax.set_title(title)
            ax.imshow(col_to_ma(displacement_r2['incremental'][ifg_n_plot,:], displacement_r2['mask']),
                                interpolation='none', aspect='auto')                                            # plot the uncropped ifg
        
        #import pdb; pdb.set_trace()
        for product in displacement_r3:
            if len(displacement_r3[product].shape) == 2:                                                                                  # if it's a rank 2, assume only x, y
                resized_r2 = displacement_r3[product][crop_pixels[2]:crop_pixels[3], crop_pixels[0]:crop_pixels[1]]               # and crop
                displacement_r2[product] = resized_r2
                displacement_r3[product] = resized_r2
            elif len(displacement_r3[product].shape) == 3:                                                                                # if it's a rank 3, assume times, x, y
                resized_r3 = displacement_r3[product][:, crop_pixels[2]:crop_pixels[3], crop_pixels[0]:crop_pixels[1]]            # and crop only last two dimensions
                displacement_r3[product] = resized_r3
                displacement_r2[product], displacement_r2['mask'] = rank3_ma_to_rank2(resized_r3)      # convert from rank 3 to rank 2 and a mask
            else:
                pass
            
    

        # for product in displacement_r3:
        #     print(f"{product} : {displacement_r3[product].shape}")
        
        # import pdb; pdb.set_trace()
        # for disp_dict in [displacement_r2, displacement_r3]:
        #     for product in disp_dict:
        #         if len(disp_dict[product].shape) == 2:                                                                                  # if it's a rank 2, assume only x, y
        #             disp_dict[product] = disp_dict[product][crop_pixels[2]:crop_pixels[3], crop_pixels[0]:crop_pixels[1]]               # and crop
        #         elif len(disp_dict[product].shape) == 3:                                                                                # if it's a rank 3, assume times, x, y
        #             disp_dict[product] = disp_dict[product][:, crop_pixels[2]:crop_pixels[3], crop_pixels[0]:crop_pixels[1]]            # and crop only last two dimensions
        #         else:
        #             pass
    

        if figures:
            add_square_plot(crop_pixels[0], crop_pixels[1], crop_pixels[2], crop_pixels[3], ax)                 # draw a box showing the cropped region    
  
   

    if return_r3:
        if ref_area:
            return displacement_r3, displacement_r2, tbaseline_info, ref_xy
        else:
            return displacement_r3, displacement_r2, tbaseline_info
    else:
        if ref_area:
            return displacement_r2, tbaseline_info, ref_xy
        else:
            return displacement_r2, tbaseline_info

#%%
def update_mask_sources_ifgs(mask_sources, sources, mask_ifgs, ifgs):
    """ Given two masks of pixels, create a mask of pixels that are valid for both.  Also return the two sets of data with the new masks applied.  
    Inputs:
        mask_sources | boolean rank 2| original mask
        sources  | r2 array | sources as row vectors
        mask_ifgs | boolean rank 2| new mask
        ifgs  | r2 array | ifgs as row vectors
    Returns:
        ifgs_new_mask
        sources_new_mask
        mask_both | boolean rank 2| original mask
    History:
        2020/02/19 | MEG |  Written      
        2020/06/26 | MEG | Major rewrite.  
        2021_04_20 | MEG | Add check that sources and ifgs are both rank 2 (use row vectors if only one source, but it must be rank2 and not rank 1)
    """
    import numpy as np
    import numpy.ma as ma
    from icasar.aux import col_to_ma

        
    
    def apply_new_mask(ifgs, mask_old, mask_new):
        """Apply a new mask to a collection of ifgs (or sources) that are stored as row vectors with an accompanying mask.  
        Inputs:
            ifgs | r2 array | ifgs as row vectors
            mask_old | r2 array | mask to convert a row of ifg into a rank 2 masked array
            mask_new | r2 array | the new mask to be applied.  Note that it must not unmask any pixels that are already masked.  
        Returns:
            ifgs_new_mask | r2 array | as per ifgs, but with a new mask.  
        History:
            2020/06/26 | MEG | Written
        """
        n_pixs_new = len(np.argwhere(mask_new == False))                                        
        ifgs_new_mask = np.zeros((ifgs.shape[0], n_pixs_new))                        # initiate an array to store the modified sources as row vectors    
        for ifg_n, ifg in enumerate(ifgs):                                 # Loop through each source
            ifg_r2 = col_to_ma(ifg, mask_old)                             # turn it from a row vector into a rank 2 masked array        
            ifg_r2_new_mask = ma.array(ifg_r2, mask = mask_new)              # apply the new mask   
            ifgs_new_mask[ifg_n, :] = ma.compressed(ifg_r2_new_mask)       # convert to row vector and places in rank 2 array of modified sources
        return ifgs_new_mask
    
    
    # check some inputs.  Not exhuastive!
    if (len(sources.shape) != 2) or (len(ifgs.shape) != 2):
        raise Exception(f"Both 'sources' and 'ifgs' must be rank 2 arrays (even if they are only a single source).  Exiting. ")
    
    mask_both = ~np.logical_and(~mask_sources, ~mask_ifgs)                                       # make a new mask for pixels that are in the sources AND in the current time series
    n_pixs_sources = len(np.argwhere(mask_sources == False))                                  # masked pixels are 1s, so invert with 1- bit so that non-masked are 1s, then sum to get number of pixels
    n_pixs_new = len(np.argwhere(mask_ifgs == False))                                          # ditto for new mask
    n_pixs_both = len(np.argwhere(mask_both == False))                                        # ditto for the mutual mask
    print(f"Updating masks and ICA sources.  Of the {n_pixs_sources} in the sources and {n_pixs_new} in the current LiCSBAS time series, "
          f"{n_pixs_both} are in both and can be used in this iteration of LiCSAlert.  ")
    
    ifgs_new_mask = apply_new_mask(ifgs, mask_ifgs, mask_both)                                  # apply the new mask to the old ifgs and return the non-masked elemts as row vectors.  
    sources_new_mask = apply_new_mask(sources, mask_sources, mask_both)                         # ditto for the sources.  
    
    return ifgs_new_mask, sources_new_mask, mask_both
    




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
        print('Done!' )
    
        return S_best, labels_hdbscan, xy_tsne, clusters_by_max_Iq_no_noise, Iq


#%%

def perform_multiple_ICA_runs(n_comp, mixtures_mc, bootstrapping_param, ica_param,
                              mixtures_white = None, dewhiten_matrix = None, ica_verbose = 'long'):
    """
    ICASAR requires ICA to be run many times, wither with or without bootstrapping.  This function performs this.  
    Inputs:
        n_comp | int | the number of souces we aim to recover.  
        mixutres_mc | rank 2 array | mixtures as rows, mean centered along rows.  I.e. of size n_varaibles x n_observations.  
        bootstrapping_param | tuple | (number of ICA runs with bootstrap, number of ICA runs without bootstrapping )  e.g. (100,10)
        ica_param | tuple | Used to control ICA, (ica_tol, ica_maxit)
        mixtures_white | rank 2 | mean centered and decorellated and unit variance in each dimension (ie whitened).  As per mixtures, row vectors.  
        dewhiten_matrix | rank 2 | n_comp x n_comp.  mixtures_mc = dewhiten_matrix @ mixtures_white
        ica_verbose | 'long' or 'short' | if long, full details of ICA runs are given.  If short, only the overall progress 
    Returns:
        S_best | list of rank 2 arrays | the sources from each run of the FastICA algorithm, n_comp x n_pixels.  Bootstrapped ones first, non-bootstrapped second.  
        A_hist | list of rank 2 arrays | the time courses from each run of the FastICA algorithm.  n_ifgs x n_comp.  Bootstrapped ones first, non-bootstrapped second.  
    History:
        2021_04_23 | MEG | Written
    """
    
    # 1: unpack a tuple and check a few inputs.  
    n_converge_bootstrapping = bootstrapping_param[0]                 # unpack input tuples
    n_converge_no_bootstrapping = bootstrapping_param[1]          
    
    if (n_converge_no_bootstrapping > 0) and ((mixtures_white is None) or (dewhiten_matrix is None)):
        raise Exception(f"If runs without bootstrapping are to be performed, the whitened data and the dewhitening matrix must be provided, yet one "
                        f"or more of these are 'None'.  This is as PCA is performed to whiten the data, yet if bootstrapping is not being used "
                        f"the data don't change, so PCA doesn't need to be run (and it can be computationally expensive).  Exiting.  ")
    
    # 2: do ICA multiple times   
    # First with bootstrapping
    A_hist_BS = []                                                                                      # ditto but with bootstrapping
    S_hist_BS = []
    n_ica_converge = 0
    n_ica_fail = 0
    if ica_verbose == 'short' and n_converge_bootstrapping > 0:                                 # if we're only doing short version of verbose, and will be doing bootstrapping
        print(f"FastICA progress with bootstrapping: ", end = '')
    while n_ica_converge < n_converge_bootstrapping:
        S, A, ica_converged = bootstrap_ICA(mixtures_mc, n_comp, bootstrap = True, ica_param = ica_param, verbose = ica_verbose)                # note that this will perform PCA on the bootstrapped samples, so can be slow.  
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
                                            X_whitened = mixtures_white, dewhiten_matrix = dewhiten_matrix, verbose = ica_verbose)               # no bootstrapping, so PCA doesn't need to be run each time and we can pass it the whitened data.  
        
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
    A_hist = A_hist_BS + A_hist_no_BS                                                                   # list containing the time courses from each run.  i.e. each is: times x n_components
    S_hist = S_hist_BS + S_hist_no_BS                                                                   # list containing the soures from each run.  i.e.: each os n_components x n_pixels
    
    return S_hist, A_hist


     
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
    from icasar.blind_signal_separation import PCA_meg2, fastica_MEG
    from icasar.aux import  maps_tcs_rescale
    
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
    from icasar.aux import col_to_ma
    
    
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

  