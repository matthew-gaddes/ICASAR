#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:19:37 2019

@author: matthew
"""

  
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
    
    def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the median value of a colormap, and scale the
        remaining color range (i.e. truncate the colormap so that it isn't 
        compressed on the shorter side) . Useful for data with a negative minimum and
        positive maximum where you want the middle of the colormap's dynamic
        range to be at zero.
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and 0.5; if your dataset mean is negative you should leave 
              this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0; usually the
              optimal value is abs(vmin)/(vmax+abs(vmin)) 
              Only got this to work with:
                  1 - vmin/(vmax + abs(vmin))
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              0.5 and 1.0; if your dataset mean is positive you should leave 
              this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 
              
          2017/??/?? | taken from stack exchange
          2017/10/11 | update so that crops shorter side of colorbar (so if data are in range [-1 100], 
                       100 will be dark red, and -1 slightly blue (and not dark blue))
          '''
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        
        if midpoint > 0.5:                                      # crop the top or bottom of the colourscale so it's not asymetric.  
            stop=(0.5 + (1-midpoint))
        else:
            start=(0.5 - midpoint)
        
        
        cdict = { 'red': [], 'green': [], 'blue': [], 'alpha': []  }
        # regular index to compute the colors
        reg_index = np.hstack([np.linspace(start, 0.5, 128, endpoint=False),  np.linspace(0.5, stop, 129)])
    
        # shifted index to match the data
        shift_index = np.hstack([ np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129)])
    
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)
            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))
        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)
        return newcmap

    
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
    import numpy as np
    
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


