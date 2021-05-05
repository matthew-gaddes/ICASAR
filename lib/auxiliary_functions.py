#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:19:37 2019

@author: matthew
"""


#%%



def visualise_ICASAR_inversion(interferograms, sources, time_courses, mask, n_data = 10):
    """
        2021_03_03 | MEG | Written.  
    """
    import numpy as np
    
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

#%%


def plot_source_tc_correlations(sources, mask, dem = None, dem_to_ic_comparisons = None, tcs_to_tempbaselines_comparisons = None,
                                png_path = './', figures = "window"):
    """Given information about the ICs, their correlations with the DEM, and their time courses correlations with an intererograms temporal basleine, 
    create a plot of this information.  
    Inputs:
        sources | rank 2 array | sources as row vectors.  
        mask | rank 2 boolean | to convert a source from a row vector to a rank 2 masked array.  
        dem | rank 2 array | The DEM.  Can also be a masked array.  
        dem_to_ic_comparisons | dict | keys:
                                        xyzs | list of rank 2 arrays | entry in the list for each signal, xyz are rows.  
                                        line_xys | list of rank 2 arrays | entry in the list for each signal, xy are points to plot for the lines of best fit
                                        cor_coefs | list | correlation coefficients between each signal and the master signal.  
        tcs_to_tempbaselines_comparisons| dict | keys as above.  
        png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
    Returns:
        figure
    History:
        2021_04_22 | MEG | Written.  
        2021_04_23 | MEG | Update so that axes are removed if they are not being used.  
        
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from auxiliary_functions import col_to_ma
    from auxiliary_functions_from_other_repos import remappedColorMap, truncate_colormap

    n_sources = sources.shape[0]

    # colour map stuff
    ifg_colours = plt.get_cmap('coolwarm')
    cmap_mid = 1 - np.max(sources)/(np.max(sources) + abs(np.min(sources)))                                    # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
    if cmap_mid < (1/257):                                                                                                 # this is a fudge so that if plot starts at 0 doesn't include the negative colorus for the smallest values
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.5, midpoint=0.5, stop=1.0, name='shiftedcmap')
    else:
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=1.0, name='shiftedcmap')


    f, axes = plt.subplots(3, (n_sources+1), figsize = (15,7))
    plt.subplots_adjust(wspace = 0.1)
    f.canvas.set_window_title(f"ICs and correlations")
    
    # 1: Plot the DEM:
    if dem is not None:
        terrain_cmap = plt.get_cmap('terrain')
        terrain_cmap = truncate_colormap(terrain_cmap, 0.2, 1)    
        dem_plot = axes[1,0].imshow(dem, cmap = terrain_cmap)
        axin = axes[1,0].inset_axes([0, -0.06, 1, 0.05])
        cbar_1 = f.colorbar(dem_plot, cax=axin, orientation='horizontal')
        cbar_1.set_label('Height (m)', fontsize = 8)
        axes[1,0].set_title('DEM')
        axes[1,0].set_xticks([])
        axes[1,0].set_yticks([])
    else:
        axes[1,0].set_axis_off()
        
    # 2: Find the x and y limits for the 2d scatter plots
    if dem_to_ic_comparisons is not None:                                                               # first check that it actually exists.  
        row1_all_xyzs = np.stack(dem_to_ic_comparisons['xyzs'], axis = 2)                               # merge together into a rank 3 numpy array. (3 x n_pixels x n_ics?)
        row1_xlim = (np.min(row1_all_xyzs[0,]), np.max(row1_all_xyzs[0,]))                              # x limits are min and max of the first row
        row1_ylim = (np.min(row1_all_xyzs[1,]), np.max(row1_all_xyzs[1,]))                              # y limits are min and max of the second row
    
    if tcs_to_tempbaselines_comparisons is not None:                                                    # as above.          
        row2_all_xyzs = np.stack(tcs_to_tempbaselines_comparisons['xyzs'], axis = 2)
        row2_xlim = (np.min(row2_all_xyzs[0,]), np.max(row2_all_xyzs[0,]))        
        row2_ylim = (np.min(row2_all_xyzs[1,]), np.max(row2_all_xyzs[1,]))        
    
    # 3: Loop through each IC
    for ic_n in range(n_sources):
        # 2a: Plotting the IC
        im = axes[0,ic_n+1].imshow(col_to_ma(sources[ic_n,:], mask), cmap = ifg_colours_cent, vmin = np.min(sources), vmax = np.max(sources))
        axes[0,ic_n+1].set_xticks([])
        axes[0,ic_n+1].set_yticks([])
        axes[0,ic_n+1].set_title(f"IC {ic_n}")
        
        # 2B: Plotting the IC to DEM scatter, if the data are available
        if dem_to_ic_comparisons is not None:
            axes[1,ic_n+1].scatter(dem_to_ic_comparisons['xyzs'][ic_n][0,:],
                                   dem_to_ic_comparisons['xyzs'][ic_n][1,:], c= dem_to_ic_comparisons['xyzs'][ic_n][2,:])
            axes[1,ic_n+1].plot(dem_to_ic_comparisons['line_xys'][ic_n][0,:], dem_to_ic_comparisons['line_xys'][ic_n][1,:], c = 'r')
            axes[1,ic_n+1].set_xlim(row1_xlim[0], row1_xlim[1])
            axes[1,ic_n+1].set_ylim(row1_ylim[0], row1_ylim[1])
            axes[1,ic_n+1].axhline(0, c='k')
            axes[1,ic_n+1].yaxis.tick_right()                                       # set ticks to be on the right. 
            if ic_n != (n_sources-1):
                axes[1,ic_n+1].yaxis.set_ticklabels([])                             # if it's not the last one, turn the  tick labels off
            else:
                axes[1,ic_n+1].yaxis.set_ticks_position('right')                    # but if it is, make sure they're on the right.  
                axes[1,ic_n+1].set_ylabel(f"IC")
                axes[1,ic_n+1].yaxis.set_label_position('right')
            if ic_n == int(n_sources/2):
                axes[1,ic_n+1].set_xlabel('Height (m)')
            axes[1,ic_n+1].set_title(f"CorCoef: {np.round(dem_to_ic_comparisons['cor_coefs'][ic_n],2)}", fontsize = 7, color = 'r')    
        else:
            axes[1,ic_n+1].set_axis_off()
                
                

        if tcs_to_tempbaselines_comparisons is not None:
            axes[2,ic_n+1].scatter(tcs_to_tempbaselines_comparisons['xyzs'][ic_n][0,:],
                                   tcs_to_tempbaselines_comparisons['xyzs'][ic_n][1,:], c= tcs_to_tempbaselines_comparisons['xyzs'][ic_n][2,:])
            axes[2,ic_n+1].plot(tcs_to_tempbaselines_comparisons['line_xys'][ic_n][0,:], tcs_to_tempbaselines_comparisons['line_xys'][ic_n][1,:], c = 'r')
            axes[2,ic_n+1].set_xlim(row2_xlim[0], row2_xlim[1])
            axes[2,ic_n+1].set_ylim(row2_ylim[0], row2_ylim[1])                                                    # force them to all share a y axis.  Gnerally not good as such varying scales.  
            axes[2,ic_n+1].axhline(0, c='k')
            axes[2,ic_n+1].yaxis.tick_right()

            if ic_n != (n_sources-1):
                axes[2,ic_n+1].yaxis.set_ticklabels([])                                                            # if it's not the last one, turn the  tick labels off
            else:
                axes[2,ic_n+1].yaxis.set_ticks_position('right')                                                   # but if it is, make sure they're on the right.  
                axes[2,ic_n+1].set_ylabel(f"IC strength")
                axes[2,ic_n+1].yaxis.set_label_position('right')
            
            if ic_n == int(n_sources/2):                                                                            # on roughly the middle plot....
                axes[2,ic_n+1].set_xlabel('Temporal Baseline (days)')                                               # add an x label.  
                                
            axes[2,ic_n+1].set_title(f"CorCoef: {np.round(tcs_to_tempbaselines_comparisons['cor_coefs'][ic_n],2)}", fontsize = 7, color = 'r')    
        else:
            axes[2,ic_n+1].set_axis_off()


    # 3: The ICs colorbar
    axin = axes[0,0].inset_axes([0.5, 0, 0.1, 1])            
    cbar_2 = f.colorbar(im, cax=axin, orientation='vertical')
    cbar_2.set_label('IC strength')
    axin.yaxis.set_ticks_position('left')

    # last tidying up
    for ax in [axes[0,0], axes[2,0]]:
        ax.set_axis_off()

    f.tight_layout()    
    
    if figures == 'window':                                                                 # possibly save the output
        pass
    elif figures == "png":
        f.savefig(f"{png_path}/05_IC_correlations.png")
        plt.close()
    elif figures == 'png+window':
        f.savefig(f"{png_path}/05_IC_correlations.png")
    else:
        pass  

#%%

def signals_to_master_signal_comparison(signals, master_signal, density = False):
        """ Given an array of signals (as row vectors), compare it to a single signal and plot a kernel
        density estimate, and calculate a line of best fit through the points with R**2 value.  
        Inputs:
            signals | rank 2 | signals as rows.  Even if there's only 1 signal, it still needs to be rank 2
            master_signal | rank 2 | signal as a row, but has to be rank 2
            density | boolean | If True, gaussian kernel density estimate for the points.  Can be slow.  
            
        Returns:
            signal_to_msignal_comparison | dict | keys:
                                                    xyzs | list of rank 2 arrays | entry in the list for each signal, xyz are rows.  
                                                    line_xys | list of rank 2 arrays | entry in the list for each signal, xy are points to plot for the lines of best fit
                                                    cor_coefs | list | correlation coefficients between each signal and the master signal.  
            
        History:
            2021_04_22 | MEG | Written.  
            2021_04_26 | MEG | Add check that the signals are of the same length.  
        """
        import numpy as np
        from scipy.stats import gaussian_kde
        import numpy.polynomial.polynomial as poly                                             # used for lines of best fit through dem/source plots
        
        n_signals, n_pixels = signals.shape                                                    # each signal is a row, observations of that are columns.  
        if n_pixels != master_signal.shape[1]:
            raise Exception(f"The signals aren't of the same length (2nd dimension), as 'signals' is {n_pixels} long, but 'master_signal' is {master_signal.shape[1]} long.  Exiting.  ")
        xyzs = []                                                                              # initiate
        line_xys = []
        cor_coefs = []
        print(f"Starting to calculate the 2D kernel density estimates for the signals.  Completed ", end = '')
        for signal_n, signal in enumerate(signals):                                            # signal is a row of signals, and loop through them.  
            
            # 1: Do the kernel density estimate
            xy = np.vstack((master_signal, signal[np.newaxis,:]))                              # master signal will be on X and be the top row.  
            x = xy[:1,:]                                    
            y = xy[1:2,:]
            if density:
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()                                                               # need to be sorted so that when plotted, those with the highest z value go on top.                                  
                x, y, z = x[0,idx], y[0,idx], z[idx]  
                xyzs.append(np.vstack((x,y,z)))                                                 # 3 rows, for each of x,y and z
            else:
                xyzs.append(np.vstack((x,y,np.zeros(n_pixels))))                                # if we're not doing the kernel density estimate, z value is just zeros.  
                    
            # 2: calculate the lines of best fit
            line_coefs = poly.polyfit(x, y, 1)                                                  # polynomial of order 1 (i.e. a line of best fit)
            line_yvals = (poly.polyval(x, line_coefs))                                          # calculate the lines yvalues for all x points
            line_xys.append(np.vstack((x, line_yvals)))                                         # x vals are first row, y vals are 2nd row
            
            # 3: And the correlation coefficient
            cor_coefs.append(np.corrcoef(x, y)[1,0])                                            # which is a 2x2 matrix, but we just want the off diagonal (as thats the correlation coefficient between the signals)
            
            print(f"{signal_n} ", end = '')
        print('\n')
        
        signal_to_msignal_comparison = {'xyzs' : xyzs,
                                        'line_xys' : line_xys,
                                        'cor_coefs' : cor_coefs}
        
        return signal_to_msignal_comparison



#%%

def create_all_ifgs(ifgs_r2, ifg_dates):
    """Given a rank 2 of incremental ifgs, calculate all the possible ifgs that still step forward in time (i.e. if deformation is positive in all incremental ifgs, 
    it remains positive in all the returned ifgs.)  If acquisition dates are provided, the tmeporal baselines of all the possible ifgs can also be found.  
    Inputs:
        ifgs_r2 | rank 2 array | Interferograms as row vectors.  
        ifg_dates | list of strings | dates in the form YYYYMMDD_YYYYMMDD.  As the interferograms are incremental, this should be the same length as the number of ifgs
    Returns:
        ifgs_r2 | rank 2 array | Only the ones that are non-zero (the diagonal in ifgs_r3) and in the lower left corner (so deformation isn't reversed.  )
    History:
        2021_04_13 | MEG | Written
        2021_04_19 | MEG | add funcionality to calculate the temporal baselines of all possible ifgs.  
        2021_04_29 | MEG | Add functionality to handle networks with breaks in them.  
    """
    import numpy as np
    from datetime import datetime, timedelta
    from auxiliary_functions_from_other_repos import acquisitions_from_ifg_dates
    
    def triange_lower_left_indexes(side_length):
        """ For a square matrix of size side_length, get the index of all the values that are in the lower
        left quadrant (i.e. all to the lower left of the diagonals).  
        Inputs:
            side_length | int | side length of the square.  e.g. 5 for a 5x5
        Returns:
            lower_left_indexes | rank 2 array | indexes of all elements below the diagonal.  
        History:
            2021_04_13 | MEG | Written.  
        """
        import numpy as np
        zeros_array = np.ones((side_length, side_length))                                                               # initate as ones so none will be selected.  
        zeros_array = np.triu(zeros_array)                                                                              # set the lower left to 0s
        lower_left_indexes = np.argwhere(zeros_array == 0)                                                              # select only the lower lefts
        return lower_left_indexes

    
    n_ifgs, n_pixs = ifgs_r2.shape
        
    # 1: Determine if the network is continuous, and if not split it into lists
    ifg_dates_continuous = []                                                                   # list of the dates for a continuous network
    ifgs_r2_continuous = []                                                                     # and the incremental interferograms in that network.  
    start_continuous_run = 0
    for ifg_n in range(n_ifgs-1):
        if (ifg_dates[ifg_n][9:] != ifg_dates[ifg_n+1][:8]):                                   # if the dates don't agree
            ifg_dates_continuous.append(ifg_dates[start_continuous_run:ifg_n+1])                # +1 as want to include the last date in the selection
            ifgs_r2_continuous.append(ifgs_r2[start_continuous_run:ifg_n+1,])
            start_continuous_run = ifg_n+1                                                      # +1 so that when we index the next time, it doesn't include ifg_n
        if ifg_n == n_ifgs -2:                                                                  # of if we've got to the end of the list.              
            ifg_dates_continuous.append(ifg_dates[start_continuous_run:])                       # select to the end.  
            ifgs_r2_continuous.append(ifgs_r2[start_continuous_run:,])
            
    n_networks = len(ifg_dates_continuous)                                                      # get the number of connected networks.  
    
    # for item in ifgs_r2_continuous:
    #     print(item.shape)
    # for item in ifg_dates_continuous:
    #     print(item)
    #     print('\n')

    # import copy
    # ifg_dates_copy = copy.copy(ifg_dates)
    # for ifg_list in ifg_dates_continuous:
    #     for ifg in ifg_list:
    #         try:
    #             del ifg_dates_copy[ifg_dates_copy.index(ifg)]
    #         except:
    #             pass
    # print(ifg_dates_copy)

    # 2: Loop through each continuous network and make all possible ifgs.  
    ifgs_all_r2 = []
    dates_all_r1 = []
    for n_network in range(n_networks):
        ifgs_r2_temp = ifgs_r2_continuous[n_network]
        ifg_dates_temp = ifg_dates_continuous[n_network]
        n_acq = ifgs_r2_temp.shape[0] + 1
        
        # 2a: convert from daisy chain of incremental to a relative to a single master at the start of the time series.  
        acq1_def = np.zeros((1, n_pixs))                                                # deformation is 0 at the first acquisition
        ifgs_cs = np.cumsum(ifgs_r2_temp, axis = 0)                                          # convert from incremental to cumulative.  
        ifgs_cs = np.vstack((acq1_def, ifgs_cs))
        
        # 2b: create all possible ifgs
        ifgs_cube = np.zeros((n_acq, n_acq, n_pixs))                                    # cube to store all possible ifgs in
        for i in range(n_acq):                                                          # loop through each column and add the ifgs.  
            ifgs_cube[:,i,] = ifgs_cs - ifgs_cs[i,]
           
        # 2c: Get only the positive ones (ie the lower left quadrant)    
        lower_left_indexes = triange_lower_left_indexes(n_acq)                              # get the indexes of the ifgs in the lower left corner (ie. non 0, and with unreveresed deformation.  )
        ifgs_all_r2.append(ifgs_cube[lower_left_indexes[:,0], lower_left_indexes[:,1], :])        # get those ifgs and store as row vectors.  
        
        # 2d: Calculate the dates that the new ifgs run between.  
        acq_dates = acquisitions_from_ifg_dates(ifg_dates_temp)                                                         # get the acquisitions from the ifg dates.  
        ifg_dates_all_r2 = np.empty([n_acq, n_acq], dtype='U17')                                                        # initate an array that can hold unicode strings.  
        for row_n, date1 in enumerate(acq_dates):                                                                       # loop through rows
            for col_n, date2 in enumerate(acq_dates):                                                                   # loop through columns
                ifg_dates_all_r2[row_n, col_n] = f"{date2}_{date1}"
        ifg_dates_all_r1 = list(ifg_dates_all_r2[lower_left_indexes[:,0], lower_left_indexes[:,1]])             # just get the lower left corner (like for the ifgs)

        dates_all_r1.append(ifg_dates_all_r1)

    # 3: convert lists back to a single matrix of all interferograms.  
    ifgs_all_r2 = np.vstack(ifgs_all_r2)
    dates_all_r1 = [item for sublist in dates_all_r1 for item in sublist]                                           # not clear how this works....
    
    return ifgs_all_r2, dates_all_r1


  
#%%
def plot_spatial_signals(spatial_map, pixel_mask, timecourse, shape, title, shared = 0, 
                         temporal_baselines = None, figures = "window",  png_path = './'):
    """
    Input:
        spatial map | pxc matrix of c component maps (p pixels)
        pixel_mask | mask to turn spaital maps back to regular grided masked arrays
        codings | cxt matrix of c time courses (t long)   
        shape | tuple | the shape of the grid that the spatial maps are reshaped to
        title | string | figure tite and png filename (nb .png will be added, don't include here)
        shared | 0 or 1 | if 1, spatial maps share colorbar and time courses shared vertical axis
        Temporal_baselines | x axis values for time courses.  Useful if some data are missing (ie the odd 24 day ifgs in a time series of mainly 12 day)
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
        png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
        
    Returns:
        Figure, either as a window or saved as a png
        
    2017/02/17 | modified to use masked arrays that are given as vectors by spatial map, but can be converted back to 
                 masked arrays using the pixel mask    
    2017/05/12 | shared scales as decrived in 'shared'
    2017/05/15 | remove shared colorbar for spatial maps
    2017/10/16 | remove limit on the number of componets to plot (was 5)
    2017/12/06 | Add a colorbar if the plots are shared, add an option for the time courses to be done in days
    2017/12/?? | add the option to pass temporal baselines to the function
    2020/03/03 | MEG | Add option to save figure as png and close window
    """
    
    import numpy as np
    import numpy.ma as ma  
    import matplotlib.pyplot as plt
    import matplotlib
    
    from auxiliary_functions_from_other_repos import remappedColorMap, truncate_colormap
    
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
    if cmap_mid < (1/257):                                                                       # this is a fudge so that if plot starts at 0 doesn't include the negative colorus for the smallest values
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
            
            
    if shared == 1:                                                             # if the colourbar is shared between each subplot, the axes need extending to make space for it.
        f.tight_layout(rect=[0, 0, 0.94, 1])
        cax = f.add_axes([0.94, 0.6, 0.01, 0.3])
        f.colorbar(im, cax=cax, orientation='vertical')
  
    if figures == 'window':                                                                 # possibly save the output
        pass
    elif figures == "png":
        f.savefig(f"{png_path}/{title}.png")
        plt.close()
    elif figures == 'png+window':
        f.savefig(f"{png_path}/{title}.png")
    else:
        pass
    

#%%    

def plot_temporal_signals(signals, title = None, signal_names = None,
                          figures = "window",  png_path = './'):
    """Plot a set of time signals stored in a matrix as rows.  
    Inputs:
        signals | rank 2 array | signals as row vectors.  e.g. 1x100
        title | string | figure title.  
        signals_names | list of strings | names of each signal
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
        png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
    Returns:
        Figure, either as a window or saved as a png
    History:
        2020/09/09 | MEG | Written
    
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_signals = signals.shape[0]
    fig1, axes = plt.subplots(n_signals,1, figsize = (10,6))
    if title is not None:
        fig1.canvas.set_window_title(title)
        fig1.suptitle(title)
    for signal_n, signal in enumerate(signals):
        axes[signal_n].plot(np.arange(0, signals.shape[1]), signal)
        if signal_names is not None:
            axes[signal_n].set_ylabel(signal_names[signal_n])
        axes[signal_n].grid(alpha = 0.5)
        if signal_n != (n_signals-1):
            axes[signal_n].set_xticklabels([])
            
    if figures == 'window':                                                                 # possibly save the output
        pass
    elif figures == "png":
        fig1.savefig(f"{png_path}/{title}.png")
        plt.close()
    elif figures == 'png+window':
        fig1.savefig(f"{png_path}/{title}.png")
    else:
        pass
                          # connect the figure and the function.  


#%%


def plot_pca_variance_line(pc_vals, title = '', figures = 'window', png_path = './'):
    """
    A function to display the cumulative variance in each dimension of some high D data
    Inputs:
        pc_vals | rank 1 array | variance in each dimension.  Most important dimension first.  
        title | string | figure title
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
        png_path | string or None | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
    Returns:
        figure, either as window or saved as a png
    History:
        2019/XX/XX | MEG | Written
        2020/03/03 | MEG | Add option to save as png
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
    f.canvas.set_window_title(title)
    

    if figures == 'window':
        pass
    elif figures == "png":
        f.savefig(f"{png_path}/01_pca_variance_line.png")
        plt.close()
    elif figures == 'png+window':
        f.savefig(f"{png_path}/01_pca_variance_line.png")
    else:
        pass
        
        
        
        
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
    
    def rescale_unit_range(signals):
        """
        rescale a matrix of row vectors so that each row has a range of 1.  
        Also record the scaling factor required to rescale each row vector
        signals are rows
        Input:
            signals: array with each signal as a new row 
        Output:
            signals_rescale | array | each row has a range of 1
            signals_factor | array | factor that each row is dvided by to adjust range
        """
        import numpy as np
        
        signals_rescale = np.ones(signals.shape)                                    # initiate rescaled array
        signals_factor = np.ones((np.size(signals, axis=0) , 1))                    # initiate array to record scaling factor for each row
           
        for i in np.arange(np.size(signals, axis = 0)):
            signals_factor[i,0] = (np.max(signals[i,:])-np.min(signals[i,:]))
            signals_rescale[i,:] = signals[i,:]/signals_factor[i,0]
            
        return signals_rescale, signals_factor

    
    maps_scaled , scaling = rescale_unit_range(maps)
    tcs_scaled = tcs * np.ravel(scaling)
    return maps_scaled, tcs_scaled



#%%
  
    
def bss_components_inversion(sources, interferograms):
    """
    A function to fit an interferogram using components learned by BSS, and return how strongly
    each component is required to reconstruct that interferogramm, and the 
    
    Inputs:
        sources | n_sources x pixels | ie architecture I.  Mean centered
        interferogram | n_ifgs x pixels | Doesn't have to be mean centered
        
    Outputs:
        m | rank 1 array | the strengths with which to use each source to reconstruct the ifg.  
        mean_l2norm | float | the misfit between the ifg and the ifg reconstructed from sources
    """
    import numpy as np
    
    inversion_results = []
    for interferogram in interferograms:
        interferogram -= np.mean(interferogram)                     # mean centre
        n_pixels = np.size(interferogram)
    
    
        d = interferogram.T                                        # now n_pixels x n_ifgs
        g = sources.T                                              # a matrix of ICA sources and each is a column (n_pixels x n_sources)
        
        ### Begin different types of inversions.  
        m = np.linalg.inv(g.T @ g) @ g.T @ d                       # m (n_sources x 1), least squares
        #m = g.T @ np.linalg.inv(g @ g.T) @ d                      # m (n_sources x 1), least squares with minimum norm condition.     COULDN'T GET TO WORK.  
        #m = np.linalg.pinv(g) @ d                                   # Moore-Penrose inverse of G for a simple inversion.  
        # u = 1e0                                                                 # bigger value favours a smoother m, which in turn can lead to a worse fit of the data.  1e3 gives smooth but bad fit, 1e1 is a compromise, 1e0 is rough but good fit.  
        # m = np.linalg.inv(g.T @ g + u*np.eye(g.shape[1])) @ g.T @ d;                # Tikhonov solution  
        
        
        ### end different types of inversion
        d_hat = g@m
        d_resid = d - d_hat
        mean_l2norm = np.sqrt(np.sum(d_resid**2))/n_pixels          # misfit between ifg and ifg reconstructed from sources
        
        inversion_results.append({'tcs'      : m,
                                  'model'    : d_hat,
                                  'residual' : d_resid,
                                  'l2_norm'  : mean_l2norm})
    
    return inversion_results


#%%
    

def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.  
    Useful when converting between vectors used by BSS methods results that are to be plotted
    Inputs:
        col | rank 1 array | 
        pixel_mask | array mask (rank 2)
    Outputs:
        source | rank 2 masked array | colun as a masked 2d array
    """
    import numpy.ma as ma 
    import numpy as np
    
    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()   
    return source


#%% taken from insar_tools.py

def r2_to_r3(ifgs_r2, mask):
    """ Given a rank2 of ifgs as row vectors, convert it to a rank3. 
    Inputs:
        ifgs_r2 | rank 2 array | ifgs as row vectors 
        mask | rank 2 array | to convert a row vector ifg into a rank 2 masked array        
    returns:
        phUnw | rank 3 array | n_ifgs x height x width
    History:
        2020/06/10 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
    
    n_ifgs = ifgs_r2.shape[0]
    ny, nx = col_to_ma(ifgs_r2[0,], mask).shape                                   # determine the size of an ifg when it is converter from being a row vector
    
    ifgs_r3 = np.zeros((n_ifgs, ny, nx))                                                # initate to store new ifgs
    for ifg_n, ifg_row in enumerate(ifgs_r2):                                           # loop through all ifgs
        ifgs_r3[ifg_n,] = col_to_ma(ifg_row, mask)                                  
    
    mask_r3 = np.repeat(mask[np.newaxis,], n_ifgs, axis = 0)                            # expand the mask from r2 to r3
    ifgs_r3_ma = ma.array(ifgs_r3, mask = mask_r3)                                      # and make a masked array    
    return ifgs_r3_ma


#%% Copied from small_plot_functions.py


def r2_arrays_to_googleEarth(images_r3_ma, lons, lats, layer_name_prefix = 'layer', kmz_filename = 'ICs',
                             out_folder = './'):
    """ Given one or several arrays in a rank3 array, create a multilayer Google Earth file (.kmz) of them.  
    Inputs:
        images_r3_ma | rank3 masked array |x n_images x ny x nx
        lons | rank 2 array | lons of each pixel in the image.  
        lats | rank 2 array | lats of each pixel in theimage. 
        layer_name_prefix | string | Can be used to set the name of the layes in the kmz (nb of the form layer_name_prefix_001 etc. )
        kmz_filename | string | Sets the name of the kmz produced
        out_folder | pathlib Path | path to location to save .kmz.  
    Returns:
        kmz file
    History:
        2020/06/10 | MEG | Written
        2021/03/11 | MEG | Update to handle incorrectly sized lons and lats arrays (e.g. rank2 arrays instead of rank 1)
    """
    import numpy as np
    import os
    import shutil
    import simplekml
    from pathlib import Path


    n_images = images_r3_ma.shape[0]    
    if type(out_folder) == str:                                                                                     # this should really be a path, but it could easily be a string.  
        out_folder = Path(out_folder)                                                                               # if it is a string, conver it.  
    # 0 temporary folder for intermediate pngs
    try:
        os.mkdir('./temp_kml')                                                                       # make a temporay folder to save pngs
    except:
        print("Can't create a folder for temporary kmls.  Trying to delete 'temp_kml' incase it exisits already... ", end = "")
        try:
            shutil.rmtree('./temp_kml')                                                              # try to remove folder
            os.mkdir('./temp_kml')                                                                       # make a temporay folder to save pngs
            print("Done. ")
        except:
          raise Exception("Problem making a temporary directory to store intermediate pngs" )

    # 1: Initiate the kml
    kml = simplekml.Kml()
        
    # 2 Begin to loop through each iamge
    for n_image in np.arange(n_images)[::-1]:                                           # Reverse so that first IC is processed last and appears as visible
        layer_name = f"{layer_name_prefix}_{str(n_image).zfill(3)}"                     # get the name of a layer a sttring
        r2_array_to_png(images_r3_ma[n_image,], layer_name, './temp_kml/')              # save as an intermediate .png
        
        ground = kml.newgroundoverlay(name= layer_name)                                 # add the overlay to the kml file
        ground.icon.href = f"./temp_kml/{layer_name}.png"                               # and the actual image part
    
        ground.gxlatlonquad.coords = [(lons[-1,0], lats[-1,0]), (lons[-1,-1],lats[-1,-1]),           # lon, lat of image south west, south east
                                      (lons[0,-1], lats[0,-1]), (lons[0,0],lats[0,0])]         # north east, north west  - order is anticlockwise around the square, startign in the lower left
       
    #3: Tidy up at the end
    kml.savekmz(out_folder / f"{kmz_filename}.kmz", format=False)                                    # Saving as KMZ
    shutil.rmtree('./temp_kml')    


#%% Copied from small_plot_functions.py

def r2_array_to_png(r2, filename, png_folder = './'):    
    """ Given a rank 2 array/image, save it as a png with no borders.  
    If a masked array is used, transparency for masked areas is conserved.  
    Designed for use with Google Earth overlays.  
    
    Inputs:
        r2 | rank 2 array | image / array to be saved
        filename | string | name of .png
        png_folder | string | folder to save in, end with /  e.g. ./kml_outputs/
    Returns:
        png of figure
    History:
        2020/06/10 | MEG | Written
        2021_05_05 | MEG | Change colours to coolwarm.  
        
    """
    import matplotlib.pyplot as plt
    
    f, ax = plt.subplots(1,1)
    ax.imshow(r2, cmap = plt.get_cmap('coolwarm'))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"{png_folder}{filename}.png", bbox_inches = 'tight',pad_inches = 0, transparent = True)
    plt.close()


#%%

def prepare_point_colours_for_2d(labels, cluster_order):
    """Given the label for each point (ie 1, 2 or 3 say, or -1 if noise) and the order of importance to the clusters 
    (ie cluster 3 is the most compact and isolated so has the highest Iq value, then cluster 1, then cluster 2), return 
    a list of colours for each point so they can be plotted using a standard  .scatter funtcion.  Ie all the points labelled
    3 have the same colour.  
    
    Inputs:
        label | rank 1 array | the label showing which cluster each point is in.  e.g. (1000)
        cluster_order | rank 1 array | to determine which cluster should be blue (the best one is always in blue, the 2nd best in orange etc.  )
    Returns:
        labels_chosen_colours | np array | colour for each point.  Same length as label.  
    History:
        2020/09/10 | MEG | Written
        2020/09/11 | MEG | Update so returns a numpy array and not a list (easier to index later on.  )
    """
    import numpy as np
    n_clusters = len(cluster_order)
    
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']           # the standard nice Matplotlib colours
    if n_clusters > 10:                                                         # if we have more than 10 clsuters, generate some random colours
        for i in range(n_clusters - 10):                                        # how many more colours we need (as we already have 10)
            colours.append(('#%02X%02X%02X' % (np.random.randint(0,25), 
                                               np.random.randint(0,25),
                                               np.random.randint(0,25))))       # generate colours randomly (ie a point between 0 and 255 in 3 dimensions.  )
    else:
        colours = colours[:n_clusters]                                          # crop to length if we have 10 or less colours
        
    colours2 = []                                                               # new list of colours, 1st item is the colour that label 0 should be (which is not necesarily blue)
    for i in range(n_clusters):                                                 # loop through each cluster
        colours2.append(colours[int(np.argwhere(cluster_order == i))])          # populate the list
    
    labels_chosen_colours = []                                           # initiate a list where instead of label for each source, we have its colour
    for label in(labels):                                           # Loop through each point's label
        if label == (-1):                                           # if noise, 
            labels_chosen_colours.append('#c9c9c9')                     # colour is grey
        else:               
            labels_chosen_colours.append(colours2[label])           # otherwise, the correct colour (nb colours 2 are reordered so the most imporant clusters have the usual blue etc. colours)
    labels_chosen_colours = np.asarray(labels_chosen_colours)       # convert from list to numpy array
    return labels_chosen_colours


#%%

def prepare_legends_for_2d(clusters_by_max_Iq_no_noise, Iq):
        """Given the cluster order and the cluster quality index (Iq), create a lenend ready for plot_2d_interactive_fig.  
        Inputs:
            clusters_by_max_Iq_no_noise | rank1 array | e.g. (3,2,4,1) if cluster 3 has the highest Iq.  
            Iq | list | Iq for each clusters.  1st item in list is Iq for 1st cluster.  
        Returns:
            legend_dict | dict | contains the legend symbols (a list of complicated Matplotlib 2D line things), and the labels as a list of strings.  
        History:
            2020/09/10 | MEG | Written
            
        """
        import numpy as np
        from matplotlib.lines import Line2D                                  # for the manual legend
        n_clusters = len(Iq)
        
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
        if n_clusters > 10:                                                                  # if we have more than 10 clsuters, repeat the same colours the required number of times
            for i in range(n_clusters-10):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#%02X%02X%02X' % (np.random.randint(0,255),
                                                                                                                  np.random.randint(0,255),
                                                                                                                  np.random.randint(0,255))))
        legend_elements = legend_elements[:n_clusters]                                      # crop to length
    
        legend_labels = []
        for i in clusters_by_max_Iq_no_noise:
                legend_labels.append(f'#: {i}\nIq: {np.round(Iq[i], 2)} ')                                   # make a list of strings to name each cluster
        legend_labels.append('Noise')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#c9c9c9'))              # but if we have 10 clusters (which is the max we plot), Noise must be added as the 11th
        legend_dict = {'elements' : legend_elements,
                       'labels'   : legend_labels}
        return legend_dict
