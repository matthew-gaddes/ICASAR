# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:23:10 2018

@author: eemeg
"""



def ICASAR(phUnw, bootstrapping_param, mask, n_comp, source_order = 'Iq', figures = True, scatter_zoom = 0.2,
           ica_param = (1e-4, 150), tsne_param = (30,12), hdbscan_param = (35,10)):
    """
    Inputs:
        phUnw | rank 2 array | ifgs as rows
        bootstrapping_param | tuple | (number of ICA runs with bootstrap, number of ICA runs without bootstrapping )  e.g. (100,10)
        mask | rank 2 boolean | mask to convert the ifgs as rows into rank 2 masked arrays.  Used for figure outputs.  
        n_comp | int | Number of ocmponents that are retained from PCA and used as the input for ICA.  
        source_order | string | 'Iq' or 'tc_energy'.  Sorts the sources (and their time courses) so that the most important is first, where most important is 
                                if they came from a high quality cluster (Iq), or based on how strong (energetic) the time course is (tc_energy)
        figures | boolean | controls if figures are produced (and outputs to the screen)
        scatter_zoom | float | Sets the size of the popup previews in the 2d clusters_by_max_Iq_no_noise figure.  
        hdbscan_param  | tuple | Used to control the clustering (min_cluster_size, min_samples)
        tsne_param     | tuple | Used to control the 2d manifold learning  (perplexity, early_exaggeration)
        
    
    Outputs:
        S_best | rank 2 array | the recovered sources as row vectors (e.g. 5 x 12,300)
        tcs    | rank 2 array | the time courses for the recoered sources (e.g. 17 x 5)
        source_residuals    | ? | the residual when each input mixture is reconstructed using the sources and time courses 
        Iq              | ?|     the cluster quality index for each cluster.  Ordered by cluster number, where cluster -1 is noise ()
        n_clusters  | int | the number of clusters found.  Doens't include noise, so is almost always 1 less than the length of Iq
        S_all_info | dictionary| useful for custom plotting. Sources: all the sources in a rank 3 array (e.g. 500x500 x1200 for 6 sources recovered 200 times)
                                                            labels: label for each soure
                                                            xy: x and y coordinats for 2d representaion of all sources
        
        
    """
    # external functions
    import numpy as np
    import matplotlib.pyplot as plt
    import hdbscan
    from sklearn.manifold import TSNE                                    # t-distributed stochastic neighbour embedding
    
    
    n_converge_bootstrapping = bootstrapping_param[0]                 # unpack input tuples
    n_converge_no_bootstrapping = bootstrapping_param[1]
    perplexity = tsne_param[0]                              
    early_exaggeration = tsne_param[1]
    min_cluster_size = hdbscan_param[0]                                              
    min_samples = hdbscan_param[1] 
    ica_tol = ica_param[0]
    ica_maxit = ica_param[1]
    
    
    phUnwMC = phUnw - np.mean(phUnw, axis = 1)[:,np.newaxis]                                        # mean centre the data (along rows)
    n_ifgs = np.size(phUnwMC, axis = 0)     
       
    # do sPCA once
    print('Performing PCA to whiten the data....', end = "")
    PC_vecs, PC_vals, PC_whiten_mat, PC_dewhiten_mat, x_mc, x_decorrelate, x_white = PCA_meg2(phUnwMC, verbose = False)    
    if figures:
        pca_variance_line(PC_vals, title = 'Variance in each dimension for the preliminary whitening')
        x_decorrelate_rs, PC_vecs_rs = maps_tcs_rescale(x_decorrelate[:n_comp,:], PC_vecs[:,:n_comp])
        component_plot(x_decorrelate_rs.T, mask, PC_vecs_rs.T, mask.shape, 'sPCA / whitening results', shared = 1)
    print('Done!')
    
    
    # do ICA multiple times
    if n_converge_bootstrapping > 0 and n_converge_no_bootstrapping > 0:
        bootstrapping = 'partial'
    elif n_converge_bootstrapping == 0 and n_converge_no_bootstrapping > 0:
        bootstrapping = 'none'
    elif n_converge_bootstrapping > 0 and n_converge_no_bootstrapping == 0:
        bootstrapping = 'full'    
    
    A_hist_no_BS = []
    S_hist_no_BS = []
    A_hist_BS = []
    S_hist_BS = []
    
    n_ica_converge = 0
    n_ica_fail = 0
    while n_ica_converge < n_converge_bootstrapping:
        print(f"sICA with bootstrapping has converged {n_ica_converge} of {n_converge_bootstrapping} times.   ", end = "")
        input_ifg_args = np.arange(n_comp-1)                                                                          # initiate as a crude way to get into the loop
        n_loop = 0                                                                                                  # to count how many goes it takes to generate a good bootstrap sample
        while len(np.unique(input_ifg_args)) < n_comp and n_loop < 100:                                            # try making a list of samples to bootstrap with providing it has enough unique items for subsequent pca to work
            input_ifg_args = np.random.randint(0, n_ifgs, n_ifgs)                                                   # generate indexes of samples to select for bootstrapping 
            n_loop += 1
        if n_loop == 100:                                                                                           # if we exited beacuse we were stuck in a loop, error message and stop
            print('Unable to bootstrap the data as the number of training data must be sufficently bigger than "n_components" sought that there are "n_components" unique items in a bootsrapped sample.  ')      # error message
            sys.exit()        
        phUnwMC_bs = phUnwMC[input_ifg_args, :]                                                                   # bootstrapped smaple
        vecs_bs, _, _, _, _, _, x_white_bs = PCA_meg2(phUnwMC_bs, verbose = False)                               # pca on bootstrapped data
        x_white_bs_for_ica = x_white_bs[:n_comp, :]                                                              # reduce dimensionality
        W, S, A, _, _, ica_converged = fastica_MEG(x_white_bs_for_ica, n_comp=n_comp,  algorithm="parallel",        
                                                               whiten=False, maxit=ica_maxit, tol=ica_tol)          # do ICA
        if ica_converged is True:
            A_dewhite = PC_dewhiten_mat[:,0:n_comp] @ A             # turn ICA mixing matrix back into a time course
            S, A_dewhite = maps_tcs_rescale(S, A_dewhite)          # rescale so spatial maps have a range or 1 (so easy to compare)
            A_hist_BS.append(A_dewhite)                             # record results
            S_hist_BS.append(S)                                     # record results
            n_ica_converge += 1
        else:
            n_ica_fail += 1
       
    n_ica_converge = 0                          # reset the counters for the second lot of ica
    n_ica_fail = 0
    while n_ica_converge < n_converge_no_bootstrapping:
        print(f"sICA without bootstrapping has converged {n_ica_converge} of {n_converge_no_bootstrapping} times.   ", end = "")
        x_white_for_ica = x_white[:n_comp,:]                                                                                        # reduce the number of components as required
        W, S, A, _, _, ica_converged = fastica_MEG(x_white_for_ica, n_comp=n_comp,  algorithm="parallel",           
                                                               whiten=False, maxit=ica_maxit, tol=ica_tol)                          # do ICA
        if ica_converged is True:
            A_dewhite = PC_dewhiten_mat[:,0:n_comp] @ A                                                                                 # ICA unmixing matrix back into time courses
            S, A_dewhite = maps_tcs_rescale(S, A_dewhite)          # rescale so spatial maps have a range or 1 (so easy to compare)
            A_hist_no_BS.append(A_dewhite)
            S_hist_no_BS.append(S)
            n_ica_converge += 1
        else:
            n_ica_fail += 1
    del n_ica_converge, n_ica_fail
    
    
    # change data structure for sources, and compute similarities and distances between them.  
    print('Starting to compute the pairwise distance matrices....', end = '')
    if bootstrapping is 'full':
        S_hist_r2, S_hist_r3 = sources_list_to_r2_r3(S_hist_BS, mask)                          # combine both bootstrapped and non-bootstrapped.  
        D, S = pairwise_comparison(S_hist_r2)
    elif bootstrapping is 'partial':
        S_hist_r2, S_hist_r3 = sources_list_to_r2_r3(S_hist_BS + S_hist_no_BS, mask)                          # combine both bootstrapped and non-bootstrapped.      
        D, S = pairwise_comparison(S_hist_r2)                                           # pairwise for all the sources
    elif bootstrapping is 'none':
        S_hist_r2, S_hist_r3 = sources_list_to_r2_r3(S_hist_no_BS, mask)                          # combine both bootstrapped and non-bootstrapped.  
        D, S = pairwise_comparison(S_hist_r2)
    print('Done!')
    

    # Clustering with all the recovered sources   
    print('Starting to cluster the sources using HDBSCAN....', end = "")  
    clusterer_precom = hdbscan.HDBSCAN(metric = 'precomputed', min_cluster_size = min_cluster_size, min_samples = min_samples, cluster_selection_method = 'leaf')
    labels_hdbscan = clusterer_precom.fit_predict(D)                      # n_samples x n_features

    Iq = cluster_quality_index(labels_hdbscan, S)                                             # with the chosen number of sources, calculate the cluster quality index
    clusters_by_max_Iq = np.argsort(Iq)[::-1] + (-1)                                                                  # default for argsort is min first so reverse to max first as these are the best.   -1 as 
    if np.min(labels_hdbscan) == (-1):
        Iq_no_noise = np.delete(Iq, 0)
        clusters_by_max_Iq_no_noise = np.argsort(Iq_no_noise)[::-1]                                                                   # default for argsort is min first so reverse to max first as these are the best
    print('Done!')
    
    
    # 2d manifold with all the recovered sources
    print('Starting to calculate the 2D manifold representation....', end = "")
    manifold_tsne = TSNE(n_components = 2, metric = 'precomputed', perplexity = perplexity, early_exaggeration = early_exaggeration)
    xy_tsne = manifold_tsne.fit(D).embedding_
    print('Done!' )
    
    if figures:
        _ = plot_cluster_results(labels = labels_hdbscan, interactive = True, images_r3 = S_hist_r3, order = clusters_by_max_Iq_no_noise, Iq = Iq_no_noise, 
                               xy2 = xy_tsne, hull = False, set_zoom = scatter_zoom)
        f4 = plt.subplots(); clusterer_precom.condensed_tree_.plot(select_clusters = True)
    
    
    # Determine the number of clusters
    if np.min(labels_hdbscan) == (-1):                                      # if we have noise, 
        n_clusters = np.size(np.unique(labels_hdbscan)) - 1                 # noise doesn't count as a cluster so we -1 from number of clusters
    else:
        n_clusters = np.size(np.unique(labels_hdbscan))                     # but if no noise, number of clusters is just number of different labels
        
        
    # Centrotypes (object that is most similar to all others in the cluster)
    print('Calculating the centrotypes and associated time courses...', end = '')
    S_best_args = np.zeros((n_clusters, 1)).astype(int)       
    for i, clust_number in enumerate(clusters_by_max_Iq_no_noise):                              # loop through each cluster in order of how good they are
        source_index = np.ravel(np.argwhere(labels_hdbscan == clust_number))                     # get the indexes of sources in this cluster
        S_this_cluster = np.copy(S[source_index, :][:, source_index])                          # similarities for just this cluster
        in_cluster_arg = np.argmax(np.sum(S_this_cluster, axis = 1))                           # the sum of a column of S_this... is the similarity between 1 source and all the others.  Look for the column that's the maximum
        S_best_args[i,0] = source_index[in_cluster_arg]                                        # conver the number in the cluster to the number overall (ie 2nd in cluster is actually 120th source)     
    S_best = np.copy(S_hist_r2[np.ravel(S_best_args),:])                                       # these are the centrotype sources
    del source_index, S_this_cluster, in_cluster_arg, S_best_args                              # tidy up    
    
    # Time courses using centrotypes
    tcs = np.zeros((n_ifgs, n_clusters))                                                    # store time courses as columns    
    source_residuals = np.zeros((n_ifgs,1))                                                 # initiate array to store these
    for i in range(n_ifgs):
        m, residual_centrotypes = bss_components_inversion(S_best, phUnwMC[i,:])                          # fit each of the training ifgs with the chosen ICs
        tcs[i,:] = m                                                                              # store time course
        source_residuals[i,0] = residual_centrotypes                                                                   # if bootstrapping, as sources come from bootstrapped data they are better for doing the fitting
    print('Done!')
    
    
    if figures:
        component_plot(S_best.T, mask, tcs.T, mask.shape, 'Chosen sources and time courses', shared = 1 )                # plot the sources chosen
    
    
    if source_order is 'Iq':
        print('Recovered sources are ordered by cluster quality index (Iq).  ')
    elif source_order is 'tc_energy':
        print('Recovered sources are ordered by time course energy.  ')    
        tc_energies = np.mean(np.abs(x_train_tcs), axis = 0)                                    # reorder based on time couse strength
        clusters_by_energy_no_noise = np.argsort(tc_energies)[::-1]                             # calcaulte order for most important time course first
        S_best = S_best[clusters_by_energy_no_noise, :]                                         # re-order spatial patterns
        x_train_tcs = x_train_tcs[:, clusters_by_energy_no_noise]                               # and time courses
    else:
        print("source_order must be 'Iq' or 'tc_energy'.  Defaulting to ordering by cluster quality index.  ")
    
    S_all_info = {'sources' : S_hist_r3,
                  'labels' : labels_hdbscan,
                  'xy' : xy_tsne       }
    
  
    return S_best,  tcs, source_residuals, Iq, n_clusters, S_all_info
        
    



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

def sources_list_to_r2_r3(sources, mask):
    """A function to convert a list of the outputs of multiple ICA runs into rank 2 and rank 3 arrays.  
    Inputs:
        sources | list | list of runs of ica (e.g. 10, or 20 etc.), each item would be n_sources x n_pixels
    Outputs:
        sources_r2 | rank 2 array | each source as a row vector (e.g. n_sources_total x n_pixels)
        sources_r3 | rank 3 array | each source as a rank 2 image. (e.g. n_souces_total x source_height x source_width )
    2018_06_29 | MEG: Written
    """
    import numpy as np
    from small_plot_functions import col_to_ma
    
    n_converge_needed = len(sources)
    n_comp = np.size(sources[0], axis = 0)
    n_pixels = np.size(sources[0], axis = 1)
    
    sources_r2 = np.zeros(((n_converge_needed * n_comp), n_pixels))                # convert from list to one big array
    for i in range(n_converge_needed):    
        sources_r2[i*n_comp:((i*n_comp) + n_comp), :] = sources[i]
    n_sources_total = np.size(sources_r2, axis = 0)

    sources_r3 = np.zeros((col_to_ma(sources_r2[0,:], mask).shape))[np.newaxis, :, :]               # get the size of one image (so rank 2)
    sources_r3 = np.repeat(sources_r3, n_sources_total, axis = 0)                                    # and then extend to make rank 3
    for i in range(n_sources_total):
        sources_r3[i,:,:] = col_to_ma(sources_r2[i,:], mask)   
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



#%%
    



def plot_cluster_results(labels = None, D = None, interactive = False, images_r3 = None, order = None, Iq = None, xy2 = None, 
                         manifold = 'tsne', perplexity = 30, hull = True, set_zoom = 0.2, save = False, 
                         title = None, BS_and_no_BS_settings = None): 
    """
     A function to plot clustering results in 2d.  Interactive as hovering over a point reveals the source that it corresponds to.  
    Only the distance between each point (D) is important, and the x and y values are noramlly random. 
    However, if performing a series of plots the random x and y locations of each point can be outputted and passed to the functoin the next time it is used (as xy2).  
    
    !!!!!Bootstrapped first, non-bootstrapped second!!!!!
    
    Inputs:
        labels | cluster label for each point. Clusters of (-1) are interpreted as noise and coloured grey.  If None, plots without labels (so all points are the same colour)
        D | rank 2 array | distance metrix for all possible point pairs.  If xy2 is supplied, D is not required (as it is only used
                            by the manifold learning methods to creat the (x,y) positions for each point).  
        interactive | boolean | if True, hovering over point shows the source, if False, doesn't.  
        images_r3 | rank 3 array | n_images x image rows | image columns             Doesn't support masked arrays
        order | none or rank 1 array | the order to plot the clusters in (ie usually plot the best (eg highest Iq) first)
        xy2 | many x 2 | xy and coordinates for the points if you wan't to specify where they're drawn
        manifold | string | 'tsne' for t-distributed stochastic neighbour embedding, 'mds' for multidimension scaling.  
        perplexity | int | tsne parameter, balances attention between local and global aspects of the data.  
                            Low values lead to local variations dominating, high ?  
                            Should be lower than the number of data
        hull | boolean | If True, draw convex hulls around the clusters (with more than 3 points in them, as it's only a line for clusters of 2 points)
        set_zoom | flt | set the size of the interactive images of the source that pop up.  
        save | boolean | if True, save as .png and close.  
        title | string | if supplied, applied to the figure and figure window
        BS_and_no_BS_settings | tuple | (number of bootsrapped runs, number components recovered in each run), e.g. (40,6) would mean that the first 40x6 recovered soruces came from bootsrapped
                                        runs, whilst the remainder (however many) came from non-bootstrapped runs.  
        
        
    
    Outputs:
        xy2 | many x 2 | xy and coordinates for the points if you wan't to specify where they're drawn in a later figure.  
    
    The interactive part was modified from:
    https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
    
    2018/06/18 | MEG: written 
    2018/06/19 | MEG: update the hover function so can deal with two points very close together.  
    2018/06/19 | MEG: choose between two manifolds, and choose whether interactive or not.  
    2018/06/27 | MEG: labels of (-1) are noise, and are coloured grey automatically.  
    2018/06/27 | MEG: Unique colours for plots with more than 10 clusters, and the possibility of expanded legends to show more possible colours
    2018/06/28 | MEG: Add the option for no labels to be provided and all the points to be the same colour
    2018/06/29 | MEG: Add option to plot both bootrapped and non-bootstrapped samples on the same plot with a different marker style.  
    
    
    """    

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.lines import Line2D                                  # for the manual legend
    from sklearn.manifold import TSNE                                    # t-distributed stochastic neighbour embedding
    from sklearn.manifold import MDS                                     # multidimension scaling
    from scipy.spatial import ConvexHull
    import random                                                        # for generating random colours
    r = lambda: random.randint(0,255)                                   # lambda function for random integers for colours

    def hover(event):
        if line_BS.contains(event)[0]:                             # False if cursor isn't over point, True if is
            ind = line_BS.contains(event)[1]["ind"]               # find out the index within the array from the event
            if np.size(ind) == 1:                               # if only hovering over one point, convert from array to integer
                ind =  ind[0]                                    # array to integer conversion
            else:                                               # if hovering over multiple points, pick one to show
                index = np.random.randint(0, np.size(ind), 1)   # of the multiple points, pick one to show at random
                ind = ind[index]                                # index of random point
                ind = ind[0]                                    # array to integer conversion
            #print(f'Displaying source {ind}.')
            w,h = fig.get_size_inches()*fig.dpi                 # get the figure size
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)        # ?
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)               # if event occurs in the top or right quadrant of the figure, change the annotation box position relative to mouse
            ab.set_visible(True)                                # make annotation box visible
            ab.xy =(x[ind], y[ind])                             # place it at the position of the hovered scatter point
            im.set_data(images_r3[ind,:,:])                           # set the image corresponding to that point
        
        elif line_no_BS.contains(event)[0]:                             # False if cursor isn't over point, True if is
            ind = line_no_BS.contains(event)[1]["ind"]               # find out the index within the array from the event
            if np.size(ind) == 1:                               # if only hovering over one point, convert from array to integer
                ind =  n_BS + ind[0]                                    # array to integer conversion
            else:                                               # if hovering over multiple points, pick one to show
                index = np.random.randint(0, np.size(ind), 1)   # of the multiple points, pick one to show at random
                ind = ind[index]                                # index of random point
                ind = n_BS + ind[0]                                    # array to integer conversion
            #print(f'Displaying source {ind}.')
            w,h = fig.get_size_inches()*fig.dpi                 # get the figure size
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)        # ?
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)               # if event occurs in the top or right quadrant of the figure, change the annotation box position relative to mouse
            ab.set_visible(True)                                # make annotation box visible
            ab.xy =(x[ind], y[ind])                             # place it at the position of the hovered scatter point
            im.set_data(images_r3[ind,:,:])                           # set the image corresponding to that point
        
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

       
    # 0/6: set some parameters
    if labels is None:    
        try:
            n_sources_total = np.size(D, axis = 0)
        except:
            pass
        try:
            n_sources_total = np.size(images_r3, axis = 0)
        except:
            print("Can't work out how many sources were recovered.  Either labels (labels), a pairwise distance matrix (D), or the sources themselves (images_r3) need to be provided.  ")
            import sys
            sys.exit()       
        labels = np.zeros(n_sources_total, dtype = 'int64')
    n_sources_total = len(labels)                   
    n_clusters = len(np.unique(labels))

    if BS_and_no_BS_settings is None:
        n_BS = n_sources_total
        n_no_BS = 0
    else:
        n_BS = BS_and_no_BS_settings[0]                         # the number of bootstrapped runs
        n_comp = BS_and_no_BS_settings[1]                       # the number of components sought in each run
        n_BS = n_comp * n_BS                                    # the number of recovered sources that came from bootstrapped runs
        n_no_BS = (n_comp * n_sources_total) - n_BS             # the number of recovered sources that came from  non-bootstrapped runs
  
    if np.min(labels) == (-1):                                          # if some are labelled as cluster -1, these points are noise and not a member of any cluster (by convention)
        print('Data labelled as noise will be plotted in grey.')
        noise = True                                                    # a flag to easily record that we have noise
        n_clusters -=1                                                  # noise doesn't count as a cluster
    else:
        noise = False

    if order is None:                                                               # if no order, plot 1st cluster first, etc. 
        order = np.arange(n_clusters)
    


    # 1/6: xy positions for points need to be calculated if not provided
    if xy2 is None:                                                                     # if we aren't supplied with (xy) coords for each point, make them
        if manifold is 'tsne':
            manifold_tsne = TSNE(n_components = 2, metric = 'precomputed', perplexity = perplexity)
            xy2 = manifold_tsne.fit(D).embedding_
        elif manifold is 'mds':
            manifold_mds = MDS(n_components=2, dissimilarity='precomputed')
            xy2 = manifold_mds.fit(D).embedding_        
        else:
            raise ValueError("manifold should be 'tsne' or 'mds'.  ")
    x = xy2[:,0]                                                            # split x and y into seperate variables
    y = xy2[:,1]




    # 2/6: set the colours and marker style for each data point depending on which cluster they are in
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']   # the standard nice colours
    if n_clusters > 10:                                                 # if we have more than 10 clsuters, generate some random colours
        for i in range(n_clusters - 10):                                # how many more colours we need
            colours.append(('#%02X%02X%02X' % (r(),r(),r())))           # generate them with r, a lambda fucntion
        
    colours = colours[:n_clusters]                                      # crop to length
    colours2 = []                                                       # new list of colours, 1st item is the colour that label 0 should be
    for i in range(n_clusters):
        colours2.append(colours[int(np.argwhere(order == i))])          # populate the list
    labels_chosen_colours = []                                           # initiate a list where instead of label for each source, we have its colour
    for one_label in(labels):                                            # convert labels to clsuter colours
        if one_label == (-1):                                           # if noise, 
            labels_chosen_colours.append('#c9c9c9')                     # colour is grey
        else:
            labels_chosen_colours.append(colours2[one_label])           # otherwise, the correct colour (nb colours 2 are reordered so the most imporant clusters have the usual blue etc. colours)
    
#    marker_BS = ['o']                                                   # set the marker style
#    marker_no_BS = ['v']
#    markers_all = n_BS * marker_BS + n_no_BS * marker_no_BS             # a simple list of hte marker style for each point 
    
    
    # 3/6: create figure and plot scatter
    fig = plt.figure()
    if title is None:                                                   # if an argument was given for the title
        fig.canvas.set_window_title('2d cluster representation')
    else:
        fig.canvas.set_window_title(title)
        fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Distance 1')
    ax.set_ylabel('Distance 2')
    if BS_and_no_BS_settings is not None:
        ax.set_title("'o': bootstrapped data, 'x': non-bootstrapped data")
    
    line_BS = ax.scatter(x[:n_BS],y[:n_BS], s = 14,  marker = 'o', c = labels_chosen_colours[:n_BS])           # plot the points
    
    #line_no_BS = ax.scatter(x[n_BS : (n_BS+n_no_BS)],y[n_BS : (n_BS+n_no_BS)], s= 14, marker = 'x', c = labels_chosen_colours[n_BS : (n_BS+n_no_BS)])           # plot the points
    if n_no_BS == 0:                                                                # if we don't have any non bootstrapped points, plot without a colour list (can't have an empty list of colours even if no points in python 3)
        line_no_BS = ax.scatter(x[n_BS:],y[n_BS:], s= 14, marker = 'x')           # plot the (empty)
    else:
        line_no_BS = ax.scatter(x[n_BS:],y[n_BS:], s= 14, marker = 'x', c = labels_chosen_colours[n_BS:])           # if we do have some non-bootstrapped points, plot them
    
        
    # 4/6: draw the convex hulls, if required
    if hull:
        for i in range(n_clusters):                                                     # add the hulls around the points in each cluster
            to_plot = np.argwhere(labels == i)
            xy2_1_cluster = xy2[np.ravel(to_plot), :]                                   # get just the 2d data for each cluster that we're looping through
            if np.size(xy2_1_cluster, axis = 0) > 2:                           # only draw convex hulls around the points if we're asked to, and if they have 3 or more points (will fail for clusters of less than 3 points. )
                hull = ConvexHull(xy2_1_cluster)                                            # a hull around the points in a certain cluster
                for simplex in hull.simplices:                                              # loop through each vertice
                    ax.plot(xy2_1_cluster[simplex, 0], xy2_1_cluster[simplex,1], 'k-')      # and pllot
        
    
    # 5/6: legend on the left side
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#8c564b'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#e377c2'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f7f7f'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#bcbd22'), 
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#17becf')]
    if n_clusters > 10:                                                          # if we have more than 10 clsuters, repeat the same colours the required number of times
        for i in range(n_clusters-10):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#%02X%02X%02X' % (r(),r(),r())))
    legend_elements = legend_elements[:n_clusters]                                      # crop to length

    legend_labels = []
    for i in order:
        if Iq is not None:
            legend_labels.append(f'Cluster: {i}\nIq: {np.round(Iq[i], 2)} ')                                   # make a list of strings to name each cluster
        else:
            legend_labels.append(f'Cluster: {i}')
    if noise:
        legend_labels.append('Noise')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#c9c9c9'))              # but if we have 10 clusters (which is the max we plot), Noise must be added as the 11th
                                          
    box = ax.get_position()                                                         # Shrink current axis by 20%
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])                  # cont'd
    ax.legend(handles = legend_elements, labels = legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))                           # Put a legend to the right of the current axis


    # 6/6: if required, make interactive.  
    if interactive is True:
        im = OffsetImage(images_r3[0,:,:], zoom=set_zoom)                               # create the annotations box
        xybox=(50., 50.)
        ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data', boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)               # add it to the axes 
        ab.set_visible(False)           # and make invisible
        fig.canvas.mpl_connect('motion_notify_event', hover)                    # add callback for mouse moves
    
    
    if save:                                                                        # maybe save and close 
        print('saving')
        fig.savefig(f'clustering_output/clustering_with_{n_clusters}_clusters.png')
        plt.close()
    else:
        plt.show()
    
    return xy2



#%%
######################################################################################################################################################
######################################################################################################################################################
#                            Taken from other scripts for Github
######################################################################################################################################################
######################################################################################################################################################

    
def component_plot(spatial_map, pixel_mask, timecourse, shape, title, shared = 0, temporal_baselines = None):
    """
    Input:
        spatial map | pxc matrix of c component maps (p pixels)
        pixel_mask | mask to turn spaital maps back to regular grided masked arrays
        codings | cxt matrix of c time courses (t long)   
        shape | tuple | the shape of the grid that the spatial maps are reshaped to
        shared | 0 or 1 | if 1, spatial maps share colorbar and time courses shared vertical axis
        Temporal_baselines | x axis values for time courses.  Useful if some data are missing (ie the odd 24 day ifgs in a time series of mainly 12 day)
        
    2017/02/17 | modified to use masked arrays that are given as vectors by spatial map, but can be converted back to 
                 masked arrays using the pixel mask    
    2017/05/12 | shared scales as decrived in 'shared'
    2017/05/15 | remove shared colorbar for spatial maps
    2017/10/16 | remove limit on the number of componets to plot (was 5)
    2017/12/06 | Add a colorbar if the plots are shared, add an option for the time courses to be done in days
    2017/12/?? | add the option to pass temporal baselines to the function
    
    """
    import numpy as np
    import numpy.ma as ma  
    import matplotlib.pyplot as plt
    import matplotlib
    from small_plot_functions import remappedColorMap
    
    def linegraph(sig, ax, temporal_baselines = None):
        """ signal is a 1xt row vector """
        
        if temporal_baselines is None:
            times = sig.size
            a = np.arange(times)
        else:
            a = temporal_baselines
        ax.plot(a,sig,marker='o', color='k')
        ax.axhline(y=0, color='k', alpha=0.4) 
        
 
    
    # colour map stuff
    ifg_colours = plt.get_cmap('coolwarm')
    cmap_mid = 1 - np.max(spatial_map)/(np.max(spatial_map) + abs(np.min(spatial_map)))          # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
    if cmap_mid < (1/257):                                                  # this is a fudge so that if plot starts at 0 doesn't include the negative colorus for the smallest values
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.5, midpoint=0.5, stop=1.0, name='shiftedcmap')
    else:
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=1.0, name='shiftedcmap')
    
    #make a list of ifgs as masked arrays (and not column vectors)
    spatial_maps_ma = []
    for i in range(np.size(spatial_map,1)):
        spatial_maps_ma.append(ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask ))
        spatial_maps_ma[i].unshare_mask()
        spatial_maps_ma[i][~spatial_maps_ma[i].mask] = spatial_map[:,i].ravel()
    tmp, n_sources = spatial_map.shape
#    if n_sources > 5:
#        n_sources = 5
    del tmp
    
    f, (ax_all) = plt.subplots(2, n_sources, figsize=(15,7))
    f.suptitle(title, fontsize=14)
    f.canvas.set_window_title(title)
    for i in range(n_sources):    
        im = ax_all[0,i].imshow(spatial_maps_ma[i], cmap = ifg_colours_cent, vmin = np.min(spatial_map), vmax = np.max(spatial_map))
        ax_all[0,i].set_xticks([])
        ax_all[0,i].set_yticks([])
#        if shared == 0:
#            ax_all[0,i].imshow(spatial_maps_ma[i])
#        else:
#            im = ax_all[0,i].imshow(spatial_maps_ma[i], vmin = np.min(spatial_map) , vmax =  np.max(spatial_map))
    for i in range(n_sources):
        linegraph(timecourse[i,:], ax_all[1,i], temporal_baselines)
        if temporal_baselines is not None: 
            ax_all[1,i].set_xlabel('Days')
        if shared ==1:
            ax_all[1,i].set_ylim([np.min(timecourse) , np.max(timecourse)])
            
            
    if shared == 1:
        f.tight_layout(rect=[0, 0, 0.94, 1])
        cax = f.add_axes([0.94, 0.6, 0.01, 0.3])
        f.colorbar(im, cax=cax, orientation='vertical')
        
        
        
#%%
        
        
def maps_tcs_rescale(maps, tcs):
    """
    A function to rescale spaital maps to have unit range and rescale each's time cource (tc)
    so that there is no change to the product of the two matrices
    
    input:
        maps | array | spatial maps as rows (e.g. 2x1775)
        tcs  | array | time courses as columns (e.g. 15x2)
    Output:
        maps_scaled | array | spatial maps as rows with each row having unit range
        tcs_scaled | array | TCs scaled so that new maps x new tcs equals maps x tcs
    
    2017/05/15 | written
    
    """
    import numpy as np
    from small_maths_functions import rescale_unit_range
    
    maps_scaled , scaling = rescale_unit_range(maps)
    tcs_scaled = tcs * np.ravel(scaling)
    return maps_scaled, tcs_scaled


#%%
    


def fastica_MEG(X, n_comp=None,
            algorithm="parallel", whiten=True, fun="logcosh", fun_prime='', 
            fun_args={}, maxit=200, tol=1e-04, w_init=None, verbose = True):
    """Perform Fast Independent Component Analysis.
    Parameters
    ----------
    X : (p, n) array
        Array with n observations (statistical units) measured on p variables.
    n_comp : int, optional
        Number of components to extract. If None no dimension reduction
        is performed.
    algorithm : {'parallel','deflation'}
        Apply an parallel or deflational FASTICA algorithm.
    whiten: boolean, optional
        If true perform an initial whitening of the data. Do not set to 
        false unless the data is already white, as you will get incorrect 
        results.
        If whiten is true, the data is assumed to have already been
        preprocessed: it should be centered, normed and white.
    fun : String or Function
          The functional form of the G function used in the
          approximation to neg-entropy. Could be either 'logcosh', 'exp', 
          or 'cube'.
          You can also provide your own function but in this case, its 
          derivative should be provided via argument fun_prime
    fun_prime : Empty string ('') or Function
                See fun.
    fun_args : Optional dictionnary
               If empty and if fun='logcosh', fun_args will take value 
               {'alpha' : 1.0}
    maxit : int
            Maximum number of iterations to perform
    tol : float
          A positive scalar giving the tolerance at which the
          un-mixing matrix is considered to have converged
    w_init : (n_comp,n_comp) array
             Initial un-mixing array of dimension (n.comp,n.comp).
             If None (default) then an array of normal r.v.'s is used
 
    Results
    -------
    K : (n_comp, p) array
        pre-whitening matrix that projects data onto th first n.comp
        principal components. Returned only if whiten is True
    W : (n_comp, n_comp) array
        estimated un-mixing matrix
        The mixing matrix can be obtained by::
            w = np.asmatrix(W) * K.T
            A = w.T * (w * w.T).I
    S : (n_comp, n) array
        estimated source matrix
    Notes
    -----
    The data matrix X is considered to be a linear combination of
    non-Gaussian (independent) components i.e. X = SA where columns of S
    contain the independent components and A is a linear mixing
    matrix. In short ICA attempts to `un-mix' the data by estimating an
    un-mixing matrix W where S = W K X.
    Implemented using FastICA:
      A. Hyvarinen and E. Oja, Independent Component Analysis:
      Algorithms and Applications, Neural Networks, 13(4-5), 2000,
      pp. 411-430
      
      2017/07/19 | Merged into one function by MEG and included a PCA function for whitening
      2017/07/20 | fixed bug when giving the function whitened data
      2018/02/22 | Return a boolean flag describing if algorithm converged or not (only works with symetric estimation)
      
    """
    
    import numpy as np
    from scipy import linalg
    import types
    from pca import PCA_meg2
    
    
    def _ica_def(X, tol, g, gprime, fun_args, maxit, w_init):
        """Deflationary FastICA using fun approx to neg-entropy function
        Used internally by FastICA.
        """
        def _gs_decorrelation(w, W, j):
            """ Gram-Schmidt-like decorrelation. """
            t = np.zeros_like(w)
            for u in range(j):
                t = t + np.dot(w, W[u]) * W[u]
                w -= t
            return w
        
        n_comp = w_init.shape[0]
        W = np.zeros((n_comp, n_comp), dtype=float)
        # j is the index of the extracted component
        for j in range(n_comp):
            w = w_init[j, :].copy()
            w /= np.sqrt((w**2).sum())
            n_iterations = 0
            # we set lim to tol+1 to be sure to enter at least once in next while
            lim = tol + 1 
            while ((lim > tol) & (n_iterations < (maxit-1))):
                wtx = np.dot(w.T, X)
                gwtx = g(wtx, fun_args)
                g_wtx = gprime(wtx, fun_args)
                w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
                _gs_decorrelation(w1, W, j)
                w1 /= np.sqrt((w1**2).sum())
                lim = np.abs(np.abs((w1 * w).sum()) - 1)
                w = w1
                n_iterations = n_iterations + 1
            W[j, :] = w
        return W                    # XXXX for deflation, a converged term isn't returned

    
    def _ica_par(X, tol, g, gprime, fun_args, maxit, w_init):
        """Parallel FastICA.
        Used internally by FastICA.
        2017/05/10 | edit to? 
        """
        def _sym_decorrelation(W):
            """ Symmetric decorrelation """
            K = W @  W.T
            s, u = linalg.eigh(K) 
            # u (resp. s) contains the eigenvectors (resp. square roots of 
            # the eigenvalues) of W * W.T 
            u, W = [np.asmatrix(e) for e in (u, W)]
            W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W  # W = (W * W.T) ^{-1/2} * W
            return np.asarray(W)
    
        n, p = X.shape
        W = _sym_decorrelation(w_init)
        # we set lim to tol+1 to be sure to enter at least once in next while
        lim = tol + 1 
        it = 0
        hist_lim = np.zeros((1, maxit))                                   #initiate array for history of change of W
        hist_W = np.zeros((w_init.size, maxit))                           # and for what W actually is
        while ((lim > tol) and (it < (maxit-1))):                           # and done less than the maximum iterations
            wtx = W @ X
            gwtx = g(wtx, fun_args)
            g_wtx = gprime(wtx, fun_args)
            W1 = (gwtx @ X.T)/float(p) - ((np.diag(g_wtx.mean(axis=1))) @ W)
            W1 = _sym_decorrelation(W1)
            lim = max(abs(abs(np.diag(W1 @ W.T)) - 1))
            W = W1
            it += 1
            hist_lim[0,it] = lim                                        # recond the measure of how much W changes
            hist_W[:,it] = np.ravel(W)                                  # and what W is
        hist_lim = hist_lim[:, 0:it]                                    # crop the 0 if we finish before the max number of iterations
        hist_W = hist_W[:, 0:it]                                        # ditto
        if it < maxit-1:
            if verbose:
                print('FastICA algorithm converged in ' + str(it) + ' iterations.  ')
            converged = True
        else:
            if verbose:
                print("FastICA algorithm didn't converge in " + str(it) + " iterations.  ")
            converged = False
        return W, hist_lim, hist_W, converged
    

  
    
    algorithm_funcs = {'parallel': _ica_par,   'deflation': _ica_def}

    alpha = fun_args.get('alpha',1.0)
    if (alpha < 1) or (alpha > 2):
        raise ValueError("alpha must be in [1,2]")

    if type(fun) is str:
        # Some standard nonlinear functions
        # XXX: these should be optimized, as they can be a bottleneck.
        if fun == 'logcosh':
            def g(x, fun_args):
                alpha = fun_args.get('alpha', 1.0)
                return np.tanh(alpha * x)
            def gprime(x, fun_args):
                alpha = fun_args.get('alpha', 1.0)
                return alpha * (1 - (np.tanh(alpha * x))**2)
        elif fun == 'exp':
            def g(x, fun_args):
                return x * np.exp(-(x**2)/2)
            def gprime(x, fun_args):
                return (1 - x**2) * np.exp(-(x**2)/2)
        elif fun == 'cube':
            def g(x, fun_args):
                return x**3
            def gprime(x, fun_args):
                return 3*x**2
        else:
            raise ValueError(
                        'fun argument should be one of logcosh, exp or cube')
    elif callable(fun):
        raise ValueError('fun argument should be either a string '
                         '(one of logcosh, exp or cube) or a function') 
    else:
        def g(x, fun_args):
            return fun(x, **fun_args)
        def gprime(x, fun_args):
            return fun_prime(x, **fun_args)
#    if whiten is False:
#        print('Data must be whitened if whitening is being skipped. ')
    p, n = X.shape

    if n_comp is None:
        n_comp = min(n, p)
    if (n_comp > min(n, p)):
        n_comp = min(n, p)
        print("n_comp is too large: it will be set to %s" % n_comp)


    # whiten the data
    if whiten:
        vecs, vals, whiten_mat, dewhiten_mat, x_mc, x_decorrelate, x_white  = PCA_meg2(X)           # function determines whether to use compact trick or not
        X1 = x_white[0:n_comp, :]                                                                    # if more mixtures than components to recover, use only first few dimensions
    else:
        X1 = np.copy(X[0:n_comp, :])
        

    if w_init is None:
        w_init = np.random.normal(size=(n_comp, n_comp))
    else:
        w_init = np.asarray(w_init)
        if w_init.shape != (n_comp,n_comp):
            raise ValueError("w_init has invalid shape -- should be %(shape)s"
                             % {'shape': (n_comp,n_comp)})

    kwargs = {'tol': tol,
              'g': g,
              'gprime': gprime,
              'fun_args': fun_args,
              'maxit': maxit,
              'w_init': w_init}

    func = algorithm_funcs.get(algorithm, 'parallel')

    W, hist_lim, hist_W, converged = func(X1, **kwargs)                                    #  W unmixes the whitened data
    #del X1

    if whiten:
        S = W @ whiten_mat[0:n_comp,:] @ x_mc
        A = np.linalg.inv(W)
        A_dewhite = dewhiten_mat[:,0:n_comp] @ A    
        #S = np.dot(np.dot(W, K), X)
        return W, S, A, A_dewhite, hist_lim, hist_W, vecs, vals, x_mc, x_decorrelate, x_white, converged
    else:
        S = W @ X1
        A = np.linalg.inv(W)
        return W, S, A, hist_lim, hist_W, converged

#%%

def pca_variance_line(pc_vals, title = ''):
    """
    A function to display the cumulative variance in each dimension of some high D data
    Inputs:
        pc_vals | rank 1 array | variance in each dimension.  Most important dimension first.  
        title | string | figure title
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    
    f, ax = plt.subplots()
    pc_vals_cs = np.concatenate((np.array([0]), np.cumsum(pc_vals)))
    x_vals = np.arange(len(pc_vals_cs)) 
    ax.plot(x_vals, pc_vals_cs/pc_vals_cs[-1])
    ax.scatter(x_vals, pc_vals_cs/pc_vals_cs[-1])
    ax.set_xlabel('Component number')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Cumulative Variance')
    ax.set_ylim([0, 1])
    ax.set_title(title)


#%%

def PCA_meg2(X, verbose = False, return_dewhiten = True):
    """
    
    Input:
        X | array | rows are dimensions (e.g. 2 x 1000 normally for 2 sound recordings, or eg 225x12 for 12 15x15 pixel images)
                    Doesn't have to be mean centered
        verbose | boolean | if true, prints some info to the screen.  
        return_dewhiten | boolean | if False, doesn't return the dewhitening matrix as the pseudo inverse needed to calculate this can fail with very large matrices (e.g. 1e6)
        
    Output:
        vecs | array | eigenvectors as columns, most important first
        vals | 1d array | eigenvalues, most important first
        whiten_mat | 2d array | whitens the mean centered data
        x_mc | 2d array | mean centered data
        x_newbasis | 2d array | mean centered and decorrelated
        x_white | 2d | mean centered and decorellated and unit variance in each dimension (ie whitened)
        
    
    2016/12/16 | updated to python 3.5
    2016/03/29 | overhaul to include whitening
    2017/07/19 | include compact trick PCA and automatically determine which to use 
    2017/11/16 | fix bug in the order of eigenvectors and values - descending order now
    2018/01/16 | fix bug in how covariance matrix is caluculated for compact trick case (now dividing by samples, which
                 gives the same results as doing it with np.cov [which is used in the normal case])
    2018/01/17 | swith to eigh (from eig) as this doesn't give complex results (such as (1.2i + 1.2e17j))
                 Take abs value of eigenvalues as some of the tiny ones can become negative (floating point effect)
    2018/02/12 | fix a bug in the compact trick PCA so that vectors are now unit length and outputs correct.  
    2018/02/23 | add option to no return the dewhitening matrix as the pseudo inverse needed for this can fail with 
                 very large matrices.  
    """
    
    import numpy as np
    
    if not return_dewhiten:
        print('Will not return the dewhitening matrix.  ')
    dims, samples = X.shape
    X = X - X.mean(axis=1)[:,np.newaxis]                # mean center each row (ie dimension)

    if samples < dims and dims > 100:                   # do PCA usin the compact trick
        if verbose:
            print('There are more samples than dimensions and more than 100 dimension so using the compact trick.')
        M = (1/samples) * X.T @ X           # maximum liklehood covariance matrix.  See blog post for details on (samples) or (samples -1): https://lazyprogrammer.me/covariance-matrix-divide-by-n-or-n-1/
        e,EV = np.linalg.eigh(M)           # eigenvalues and eigenvectors    
        tmp = (X @ EV)                     # this is the compact trick
        vecs = tmp[:,::-1]                    # vectors are columns, make first (left hand ones) the important onces
        vals = np.sqrt(e)[::-1]               # also reverse the eigenvectors
        for i in range(vecs.shape[0]):        # normalise each eigenvector (ie change length to 1)
            vecs[i,:] /= vals
        vecs = vecs[:, 0:-1]                       # drop the last eigenvecto and value as it's not defined.            
        vecs = np.divide(vecs, np.linalg.norm(vecs, axis = 0)[np.newaxis, :])       # make unit length (columns)
        vals = vals[0:-1]                          # also drop the last eigenvealue
        X_pca_basis = vecs.T @ X                     # whitening using the compact trick is a bit of a fudge
        covs = np.diag(np.cov(X_pca_basis))
        covs_recip = np.reciprocal(np.sqrt(covs))
        covs_recip_mat = np.diag(covs_recip)
        whiten_mat = (covs_recip_mat @ vecs.T)
        if return_dewhiten:
            dewhiten_mat = np.linalg.pinv(whiten_mat)       # as always loose a dimension with compast trick, have to use pseudoinverse
    else:                                                   # or do PCA normally
        cov_mat = np.cov(X)                                           # dims by dims covariance matrix
        vals_noOrder, vecs_noOrder = np.linalg.eigh(cov_mat)                            # vectors (vecs) are columns, not not ordered
        order = np.argsort(vals_noOrder)[::-1]                                  # get order of eigenvalues descending
        vals = vals_noOrder[order]                                              # reorder eigenvalues
        vals = np.abs(vals)                                                     # do to floatint point arithmetic some tiny ones can be nagative which is problematic with the later square rooting
        vecs = vecs_noOrder[:,order]                                            # reorder eigenvectors
        vals_sqrt_mat = np.diag(np.reciprocal(np.sqrt(vals)))          # square roots of eigenvalues on diagonal of square matrix
        whiten_mat = vals_sqrt_mat @ vecs.T                            # eigenvectors scaled by 1/values to make variance same in all directions
        if return_dewhiten:
            dewhiten_mat = np.linalg.inv(whiten_mat)
    # use the vectors and values to decorrelate and whiten
    x_mc = np.copy(X)                       # data mean centered
    x_decorrelate =  vecs.T @ X             # data decorrelated
    x_white = whiten_mat @ X                # data whitened
  
    if return_dewhiten:
        return vecs, vals, whiten_mat, dewhiten_mat, x_mc, x_decorrelate, x_white
    else:
        return vecs, vals, whiten_mat, x_mc, x_decorrelate, x_white

#%%
        
    
    
def bss_components_inversion(sources, interferogram):
    """
    A function to fit an interferogram using components learned by BSS, and return how strongly
    each component is required to reconstruct that interferogramm, and the 
    
    Inputs:
        sources | n_sources x pixels | ie architecture I.  Mean centered
        interferogram | 1 x pixels | Doesn't have to be mean centered
        
    Outputs:
        m | rank 1 array | the strengths with which to use each source to reconstruct the ifg.  
        mean_l2norm | float | the misfit between the ifg and the ifg reconstructed from sources
    """
    interferogram -= np.mean(interferogram)                     # mean centre
    n_pixels = np.size(interferogram)
    
    d = interferogram.T                                        # a column vector (p x 1)
    g = sources.T                                              # a matrix of ICA sources and each is a column (p x n_sources)
    m = np.linalg.inv(g.T @ g) @ g.T @ d                       # m (n_sources x 1)
    d_hat = g@m
    d_resid = d - d_hat
    #mean_l2norm = np.sqrt(np.sum(d_resid**2))/n_pixels          # misfit between ifg and ifg reconstructed from sources
    mean_l2norm = np.sqrt(np.sum(d_resid**2))/n_pixels          # misfit between ifg and ifg reconstructed from sources
    
    return m, mean_l2norm
