# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:23:10 2018

@author: eemeg
"""



def ICASAR(n_comp, spatial_data = None, temporal_data = None, figures = "window", 
           bootstrapping_param = (200,0), ica_param = (1e-4, 150), tsne_param = (30,12), hdbscan_param = (35,10),
           lons = None, lats = None, ge_kmz = True, out_folder = './ICASAR_results/',
           ica_verbose = 'long', inset_axes_side = {'x':0.1, 'y':0.1}):
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
        temporal_data | dict or None | contains 'mixtures_r2' as time signals as row vectors and 'xvals' which are the times for each item in the time signals.   
        figures | string,  "window" / "png" / "none" / "png+window" | controls if figures are produced, noet none is the string none, not the NoneType None
        
        bootstrapping_param | tuple | (number of ICA runs with bootstrap, number of ICA runs without bootstrapping )  e.g. (100,10)
        ica_param | tuple | Used to control ICA, (ica_tol, ica_maxit)
        hdbscan_param  | tuple | Used to control the clustering (min_cluster_size, min_samples)
        tsne_param     | tuple | Used to control the 2d manifold learning  (perplexity, early_exaggeration)
        
        lons | rank 2 array | lons of each pixel in the image.  Changed to rank 2 in version 2.0, from rank 1 in version 1.0  
        lats | rank 2 array | lats of each pixel in theimage. Changed to rank 2 in version 2.0, from rank 1 in version 1.0
        ge_kmz | Boolean | If Ture and lons and lats are provided, a .kmz of the ICs is produced for viewing in GoogleEarth
        out_folder | string | if desired, can set the name of the folder results are saved to
        
        ica_verbose | 'long' or 'short' | if long, full details of ICA runs are given.  If short, only the overall progress 
        inset_axes_side | dict | inset axes side length as a fraction of the full figure, in x and y direction in the 2d figure of clustering results.  

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
    import matplotlib.pyplot as plt
    import shutil                                                                # used to make/remove folders etc
    import os                                                                    # ditto
    import pickle                                                                # to save outputs.  
    # internal functions    
    from blind_signal_separation_funcitons import PCA_meg2
    from auxiliary_functions import  bss_components_inversion, maps_tcs_rescale, r2_to_r3, r2_arrays_to_googleEarth
    from auxiliary_functions import plot_spatial_signals, plot_temporal_signals, plot_pca_variance_line, plot_2d_interactive_fig
    from auxiliary_functions import prepare_point_colours_for_2d, prepare_legends_for_2d


    # Check inputs, unpack either spatial or temporal data, and check for nans
    if temporal_data is None and spatial_data is None:                                                                  # check inputs
        raise Exception("One of either spatial or temporal data must be supplied.  Exiting.  ")
    if temporal_data is not None and spatial_data is not None:
        raise Exception("Only either spatial or temporal data can be supplied, but not both.  Exiting.  ")
    
    if spatial_data is not None:
        mixtures = spatial_data['mixtures_r2']
        mask = spatial_data['mask']
        spatial = True
    else:
        mixtures = temporal_data['mixtures_r2']
        xvals = temporal_data['xvals']
        spatial = False
    if np.max(np.isnan(mixtures)):
        raise Exception("Unable to proceed as the data ('phUnw') contains Nans.  ")
    
    # sort out various things for figures, and check input is of the correct form
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
         
    if (len(lons.shape) != 2) or (len(lats.shape) != 2):
        raise Exception(f"'lons' and 'lats' should be rank 2 tensors (i.e. matrices with a lon or lat for each pixel in the interferogram.  Exiting...  ")
        
        
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
    
    # 0: Mean centre the mixtures
    mixtures_mean = np.mean(mixtures, axis = 1)[:,np.newaxis]                                         # get the mean for each ifg (ie along rows.  )
    mixtures_mc = mixtures - mixtures_mean                                                           # mean centre the data (along rows)
    n_mixtures = np.size(mixtures_mc, axis = 0)     
       
    # 1: do sPCA once
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

        S, A, ica_converged = bootstrap_ICA(mixtures_mc, n_comp, bootstrap = True, ica_param = ica_param, verbose = fastica_verbose)
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
                                            X_whitened = x_white, dewhiten_matrix = PC_whiten_mat, verbose = fastica_verbose)
        
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
    S_best, labels_hdbscan, xy_tsne, clusters_by_max_Iq_no_noise, Iq  = bootstrapped_sources_to_centrotypes(sources_all_r2,  hdbscan_param, tsne_param, 
                                                                                                            (n_converge_bootstrapping, n_comp))                 # clusters_by_max_Iq_no_noise is an array of which cluster number is best (ie has the highest Iq)

    labels_colours = prepare_point_colours_for_2d(labels_hdbscan, clusters_by_max_Iq_no_noise)                                          # make a list of colours so that each point with the same label has the same colour, and all noise points are grey
    legend_dict = prepare_legends_for_2d(clusters_by_max_Iq_no_noise, Iq)    
    marker_dict = {'labels' : np.ravel(np.hstack((np.zeros((1, n_comp*n_converge_bootstrapping)), np.ones((1, n_comp*n_converge_no_bootstrapping)))))}        # boostrapped are labelled as 0, and non bootstrapped as 1
    if n_converge_bootstrapping == 0:
        marker_dict['styles'] = ['x']                                                                                                           # if there are no bootstrapping, only non bootstrapped
    elif n_converge_no_bootstrapping == 0:
        marker_dict['styles'] = ['o']                                                                                                           # if there are no non bootstrapping, only bootstrapped
    else:
        marker_dict['styles'] = ['o', 'x']                                                                                                      # or both
       
    plot_2d_labels = {'title' : '03_clustering_and_manifold_results',
                      'xlabel' : 'TSNE dimension 1',
                      'ylabel' : 'TSNE dimension 2'}
    if spatial:
        plot_2d_labels['title']
        spatial_data = {'images_r3' : sources_all_r3}                                                       # spatial data stored in rank 3 format (ie n_imaces x height x width)
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, spatial_data = spatial_data,                 # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side)
    
    else:
        temporal_data_S_all = {'tcs_r2' : sources_all_r2,
                               'xvals'  : temporal_data['xvals'] }                                          # make a dictionary of the sources recovered from each run
        plot_2d_interactive_fig(xy_tsne.T, colours = labels_colours, temporal_data = temporal_data_S_all,         # make the 2d interactive plot
                                labels = plot_2d_labels, legend = legend_dict, markers = marker_dict, inset_axes_side = inset_axes_side)

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
    #import pdb; pdb.set_trace()
    from auxiliary_functions import r2_arrays_to_googleEarth
    
    if spatial:
        if ge_kmz and (lons is not None) and (lats is not None):
            print('Creating a Google Earth .kmz of the geocoded independent components... ', end = '')
            S_best_r3 = r2_to_r3(S_best, mask)
            r2_arrays_to_googleEarth(S_best_r3, lons, lats, 'IC', out_folder = out_folder)                              # note that lons and lats should be rank 2 (ie an entry for each pixel in the ifgs)
            print('Done!')
               
    # 8: Save the results: 
    print('Saving the key results as a .pkl file... ', end = '')                                            # note that we don't save S_all_info as it's a huge file.  
    if spatial:
        with open(f'{out_folder}ICASAR_results.pkl', 'wb') as f:
            pickle.dump(S_best, f)
            pickle.dump(mask, f)
            pickle.dump(tcs, f)
            pickle.dump(source_residuals, f)
            pickle.dump(Iq_sorted, f)
            pickle.dump(n_clusters, f)
        f.close()
        print("Done!")
    else:                                                                       # if temporal data, no mask to save
        with open(f'{out_folder}ICASAR_results.pkl', 'wb') as f:
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

def bootstrapped_sources_to_centrotypes(sources_r2, hdbscan_param, tsne_param, 
                                        bootstrap_settings, figures = "window", png_path='./'):
    """ Given the products of the bootstrapping, run the 2d manifold and clustering algorithms to create centrotypes.  
    Inputs:
        mixtures_r2 | rank 2 array | all the sources recovered after bootstrapping.  If 5 components and 100 bootstrapped runs, this will be 500 x n_pixels (or n_times)
        hdbscan_param  | tuple | Used to control the clustering (min_cluster_size, min_samples)
        tsne_param     | tuple | Used to control the 2d manifold learning  (perplexity, early_exaggeration)
        bootstrap_settings | tuple | (number of bootsrapped runs, number components recovered in each run), e.g. (40,6) would mean that the first 40x6 recovered soruces came from bootsrapped
                                        runs, whilst the remainder (?, 6) came from non-bootstrapped runs.  Note that the number of componnets recovered is the same for both bootstrapped
                                        and non bootstrapped.  
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
        png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
    Returns:
        S_best | rank 2 array | the recovered sources as row vectors (e.g. 5 x 1230)
        labels_hdbscan | rank 2 array | the cluster number for each of the sources in sources_all_r2 e.g 1000,
        xy_tsne | rank 2 array | the x and y coordinates of where each space is in the 2D space.  e.g. 1000x2
        clusters_by_max_Iq_no_noise | rank 1 array | clusters ranked by quality index (Iq).  e.g. 3,0,1,4,2
        Iq | list | cluster quality index for each cluster.  Entry 0 is Iq (cluster quality index) for the first cluster

    History:
        2020/08/26 | MEG | Created from a script.  
    """
    import numpy as np
    import hdbscan                                                               # used for clustering
    from sklearn.manifold import TSNE                                            # t-distributed stochastic neighbour embedding

    fig_kwargs = {"figures" : figures}
    if figures == "png" or figures == "png+window":                                                         # if figures will be png, make 
        fig_kwargs['png_path'] = png_path                                                                  # this will be passed to various figure plotting functions
    elif figures == 'window' or figures == 'none':
        pass
    else:
        raise ValueError("'figures' should be 'window', 'png', 'png+window', or 'None'.  Exiting...")

    
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
              "the hdbscan_param 'min_cluster_size' too low.  ")
       
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
        S_best = np.copy(sources_r2[np.ravel(S_best_args),:])                                   # these are the centrotype sources
    
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

#%% Superseeded

# def plot_cluster_results(labels, xy, sources_r2, sources_r3,
#                          interactive = False, order = None, Iq = None, hull = True, set_zoom = 0.2, 
#                          title = "2d manifold of sources", bootstrap_settings = None, figures = 'window', png_path = './'): 
#     """
#      A function to plot clustering results in 2d.  Interactive as hovering over a point reveals the source that it corresponds to.  
#     Only the distance between each point (D) is important, and not the exact x or y position.  
        
#     !!!!!Bootstrapped first, non-bootstrapped second in sources_r2 and sources_r3 !!!!!
    
#     Inputs:
#         labels | cluster label for each point. Clusters of (-1) are interpreted as noise and coloured grey.  If None, plots without labels (so all points are the same colour)
#         xy | many x 2 | xy and coordinates for the points. ie the output of the TSNE manifold.  
#         sources_r2 | rank 2 array | n_images x n_pixels (or n_times)             Doesn't support masked arrays.  
#         sources_r3 | rank 3 array | n_images x image rows * image columns             Doesn't support masked arrays.  Not needed if working with 1d data.  
#         interactive | boolean | if True, hovering over point shows the source, if False, doesn't.  
#         order | none or rank 1 array | the order to plot the clusters in (ie usually plot the best (eg highest Iq) first)
#         hull | boolean | If True, draw convex hulls around the clusters (with more than 3 points in them, as it's only a line for clusters of 2 points)
#         set_zoom | flt | set the size of the interactive images of the source that pop up.  
#         title | string | if supplied, applied to the figure and figure window
#         bootstrap_settings | tuple | (number of bootsrapped runs, number components recovered in each run), e.g. (40,6) would mean that the first 40x6 recovered soruces came from bootsrapped
#                                         runs, whilst the remainder (?, 6) came from non-bootstrapped runs.  Note that the number of componnets recovered is the same for both bootstrapped
#                                         and non bootstrapped.  
#         figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
#         png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
        
        
    
#     The interactive part was modified from:
#     https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
    
#     2018/06/18 | MEG: written 
#     2018/06/19 | MEG: update the hover function so can deal with two points very close together.  
#     2018/06/19 | MEG: choose between two manifolds, and choose whether interactive or not.  
#     2018/06/27 | MEG: labels of (-1) are noise, and are coloured grey automatically.  
#     2018/06/27 | MEG: Unique colours for plots with more than 10 clusters, and the possibility of expanded legends to show more possible colours
#     2018/06/28 | MEG: Add the option for no labels to be provided and all the points to be the same colour
#     2018/06/29 | MEG: Add option to plot both bootrapped and non-bootstrapped samples on the same plot with a different marker style.  
#     2020/03/03 | MEG | Add option to set the path of the png saved.  
#     2020/09/28 | MEG | Remove TSNE settings from inside function (xy position of each point must be given) and tidy up several areas.  
    
    
#     """    

#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#     from matplotlib.lines import Line2D                                  # for the manual legend
#     from scipy.spatial import ConvexHull
#     import random                                                        # for generating random colours
#     r = lambda: random.randint(0,255)                                   # lambda function for random integers for colours

#     def hover(event):
#         if line_BS.contains(event)[0]:                             # False if cursor isn't over point, True if is
#             ind = line_BS.contains(event)[1]["ind"]               # find out the index within the array from the event
#             if np.size(ind) == 1:                               # if only hovering over one point, convert from array to integer
#                 ind =  ind[0]                                    # array to integer conversion
#             else:                                               # if hovering over multiple points, pick one to show
#                 index = np.random.randint(0, np.size(ind), 1)   # of the multiple points, pick one to show at random
#                 ind = ind[index]                                # index of random point
#                 ind = ind[0]                                    # array to integer conversion
#             #print(f'Displaying source {ind}.')
#             w,h = fig.get_size_inches()*fig.dpi                 # get the figure size
#             ws = (event.x > w/2.)*-1 + (event.x <= w/2.)        # ?
#             hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
#             ab.xybox = (xybox[0]*ws, xybox[1]*hs)               # if event occurs in the top or right quadrant of the figure, change the annotation box position relative to mouse
#             ab.set_visible(True)                                # make annotation box visible
#             ab.xy =(x[ind], y[ind])                             # place it at the position of the hovered scatter point
#             im.set_data(sources_r3[ind,:,:])                           # set the image corresponding to that point
        
#         elif line_no_BS.contains(event)[0]:                             # False if cursor isn't over point, True if is
#             ind = line_no_BS.contains(event)[1]["ind"]                   # find out the index within the array from the event
#             if np.size(ind) == 1:                                        # if only hovering over one point, convert from array to integer
#                 ind =  n_BS + ind[0]                                     # array to integer conversion
#             else:                                                        # if hovering over multiple points, pick one to show
#                 index = np.random.randint(0, np.size(ind), 1)            # of the multiple points, pick one to show at random
#                 ind = ind[index]                                         # index of random point
#                 ind = n_BS + ind[0]                                    # array to integer conversion
#             #print(f'Displaying source {ind}.')
#             w,h = fig.get_size_inches()*fig.dpi                 # get the figure size
#             ws = (event.x > w/2.)*-1 + (event.x <= w/2.)        # ?
#             hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
#             ab.xybox = (xybox[0]*ws, xybox[1]*hs)               # if event occurs in the top or right quadrant of the figure, change the annotation box position relative to mouse
#             ab.set_visible(True)                                # make annotation box visible
#             ab.xy =(x[ind], y[ind])                             # place it at the position of the hovered scatter point
#             im.set_data(sources_r3[ind,:,:])                           # set the image corresponding to that point
        
#         else:
#             #if the mouse is not over a scatter point
#             ab.set_visible(False)
#         fig.canvas.draw_idle()

       
#     # 0/6: set some parameters
#     if labels is None:    
#         try:
#             n_sources_total = np.size(sources_r3, axis = 0)
#         except:
#             raise Exception("Can't work out how many sources were recovered.  Either labels (labels), a pairwise distance matrix (D), or the sources themselves (images_r3) need to be provided.  ")
#         labels = np.zeros(n_sources_total, dtype = 'int64')
#     n_sources_total = len(labels)                   
#     n_clusters = len(np.unique(labels))

#     if bootstrap_settings is None:
#         n_BS = n_sources_total
#         n_no_BS = 0
#     else:
#         n_BS = bootstrap_settings[0]                         # the number of bootstrapped runs
#         n_comp = bootstrap_settings[1]                       # the number of components sought in each run
#         n_BS = n_comp * n_BS                                    # the number of recovered sources that came from bootstrapped runs
#         n_no_BS = (n_comp * n_sources_total) - n_BS             # the number of recovered sources that came from  non-bootstrapped runs
  
#     if np.min(labels) == (-1):                                          # if some are labelled as cluster -1, these points are noise and not a member of any cluster (by convention)
#         print('Data labelled as noise will be plotted in grey.')
#         noise = True                                                    # a flag to easily record that we have noise
#         n_clusters -=1                                                  # noise doesn't count as a cluster
#     else:
#         noise = False

#     if order is None:                                                               # if no order, plot 1st cluster first, etc. 
#         order = np.arange(n_clusters)
    


#     # 1/6: xy positions for points need to be calculated if not provided
#     x = xy[:,0]                                                            # split x and y into seperate variables
#     y = xy[:,1]




#     # 2/6: set the colours and marker style for each data point depending on which cluster they are in
#     colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']   # the standard nice colours
#     if n_clusters > 10:                                                 # if we have more than 10 clsuters, generate some random colours
#         for i in range(n_clusters - 10):                                # how many more colours we need
#             colours.append(('#%02X%02X%02X' % (r(),r(),r())))           # generate them with r, a lambda fucntion
        
#     colours = colours[:n_clusters]                                      # crop to length
#     colours2 = []                                                       # new list of colours, 1st item is the colour that label 0 should be
#     for i in range(n_clusters):
#         colours2.append(colours[int(np.argwhere(order == i))])          # populate the list
#     labels_chosen_colours = []                                           # initiate a list where instead of label for each source, we have its colour
#     for one_label in(labels):                                            # convert labels to clsuter colours
#         if one_label == (-1):                                           # if noise, 
#             labels_chosen_colours.append('#c9c9c9')                     # colour is grey
#         else:
#             labels_chosen_colours.append(colours2[one_label])           # otherwise, the correct colour (nb colours 2 are reordered so the most imporant clusters have the usual blue etc. colours)
    
# #    marker_BS = ['o']                                                   # set the marker style
# #    marker_no_BS = ['v']
# #    markers_all = n_BS * marker_BS + n_no_BS * marker_no_BS             # a simple list of hte marker style for each point 
    
    
#     # 3/6: create figure and plot scatter
#     fig = plt.figure()
#     if title is None:                                                   # if an argument was given for the title
#         fig.canvas.set_window_title(f'2d cluster representation')
#     else:
#         fig.canvas.set_window_title(title)
#         fig.suptitle(title)
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('Distance 1')
#     ax.set_ylabel('Distance 2')
#     if bootstrap_settings is not None:
#         ax.set_title("'o': bootstrapped data, 'x': non-bootstrapped data")
    
#     line_BS = ax.scatter(x[:n_BS],y[:n_BS], s = 14,  marker = 'o', c = labels_chosen_colours[:n_BS])           # plot the points
    
#     #line_no_BS = ax.scatter(x[n_BS : (n_BS+n_no_BS)],y[n_BS : (n_BS+n_no_BS)], s= 14, marker = 'x', c = labels_chosen_colours[n_BS : (n_BS+n_no_BS)])           # plot the points
#     if n_no_BS == 0:                                                                # if we don't have any non bootstrapped points, plot without a colour list (can't have an empty list of colours even if no points in python 3)
#         line_no_BS = ax.scatter(x[n_BS:],y[n_BS:], s= 14, marker = 'x')           # plot the (empty)
#     else:
#         line_no_BS = ax.scatter(x[n_BS:],y[n_BS:], s= 14, marker = 'x', c = labels_chosen_colours[n_BS:])           # if we do have some non-bootstrapped points, plot them
    
        
#     # 4/6: draw the convex hulls, if required
#     if hull:
#         for i in range(n_clusters):                                                     # add the hulls around the points in each cluster
#             to_plot = np.argwhere(labels == i)
#             xy2_1_cluster = xy[np.ravel(to_plot), :]                                   # get just the 2d data for each cluster that we're looping through
#             if np.size(xy2_1_cluster, axis = 0) > 2:                           # only draw convex hulls around the points if we're asked to, and if they have 3 or more points (will fail for clusters of less than 3 points. )
#                 hull = ConvexHull(xy2_1_cluster)                                            # a hull around the points in a certain cluster
#                 for simplex in hull.simplices:                                              # loop through each vertice
#                     ax.plot(xy2_1_cluster[simplex, 0], xy2_1_cluster[simplex,1], 'k-')      # and pllot
        
    
#     # 5/6: legend on the left side
#     legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8c564b'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e377c2'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#bcbd22'), 
#                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#17becf')]
#     if n_clusters > 10:                                                          # if we have more than 10 clsuters, repeat the same colours the required number of times
#         for i in range(n_clusters-10):
#             legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#%02X%02X%02X' % (r(),r(),r())))
#     legend_elements = legend_elements[:n_clusters]                                      # crop to length

#     legend_labels = []
#     for i in order:
#         if Iq is not None:
#             legend_labels.append(f'Cluster: {i}\nIq: {np.round(Iq[i], 2)} ')                                   # make a list of strings to name each cluster
#         else:
#             legend_labels.append(f'Cluster: {i}')
#     if noise:
#         legend_labels.append('Noise')
#         legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#c9c9c9'))              # but if we have 10 clusters (which is the max we plot), Noise must be added as the 11th
                                          
#     box = ax.get_position()                                                         # Shrink current axis by 20%
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])                  # cont'd
#     ax.legend(handles = legend_elements, labels = legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))                           # Put a legend to the right of the current axis


#     # 6/6: if required, make interactive.  
#     if interactive is True:
#         im = OffsetImage(sources_r3[0,:,:], zoom=set_zoom)                               # create the annotations box
#         xybox=(50., 50.)
#         ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
#         ax.add_artist(ab)               # add it to the axes 
#         ab.set_visible(False)           # and make invisible
#         fig.canvas.mpl_connect('motion_notify_event', hover)                    # add callback for mouse moves
    
    
#     if figures == 'window':                                                                 # possibly save the output
#         pass
#     elif figures == "png":
#         fig.savefig(f"{png_path}/{title}.png")
#         plt.close()
#     elif figures == 'png+window':
#         fig.savefig(f"{png_path}/{title}.png")
#     else:
#         pass



    





  