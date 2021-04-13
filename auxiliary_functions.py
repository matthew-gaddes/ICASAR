#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:19:37 2019

@author: matthew
"""

  
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


#%% MAINTAINED IN ITS OWN REPO (interactive_2d_plot) - DON'T MODIFY HERE!  


def plot_2d_interactive_fig(xy, colours, spatial_data = None, temporal_data = None,
                        inset_axes_side = {'x':0.1, 'y':0.1}, arrow_length = 0.1, figsize = (10,6), 
                        labels = None, legend = None, markers = None):
    """ Data are plotted in a 2D space, and when hovering over a point, further information about it (e.g. what image it is)  appears in an inset axes.  
    Inputs:
        xy | rank 2 array | e.g. 2x100, the x and y positions of each data
        colours | rank 1 array | e.g. 100, value used to set the colour of each data point
        spatial_data | dict or None | contains 'images_r3' in which the images are stored as in a rank 3 array (e.g. n_images x heigh x width).  Masked arrays are supported.  
        temporal_data | dict or None | contains 'tcs_r2' as time signals as row vectors and 'xvals' which are the times for each item in the timecourse.   
        inset_axes_side | dict | inset axes side length as a fraction of the full figure, in x and y direction
        arrow_length | float | lenth of arrow from data point to inset axes, as a fraction of the full figure.  
        figsize | tuple |  standard Matplotlib figsize tuple, in inches.  
        labels | dict or None | title for title, xlabel for x axis label, and ylabel for y axis label
        legend | dict or None | elements contains the matplotilb symbols.  E.g. for a blue circle: Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4')
                                labels contains the strings for each of these.       
        markers | dict or None | dictionary containing labels (a numpy array where each number relates to a different marker style e.g. (1,0,1,0,0,0,1 etc))) 
                                 and markers (a list of the different Matplotlib marker styles e.g. ['o', 'x'])
             Returns:
        Interactive figure
    History:
        2020/09/09 | MEG | Modified from a sript in the ICASAR package.  
        2020/09/10 | MEG | Add labels, and change so that images are stored as rank3 arrays.  
        2020/09/10 | MEG | Add legend option.  
        2020/09/11 | MEG | Add option to have different markers.  
        2020/09/15 | MEG | Add option to set size of inset axes.  
    
    """
    def remove_axes2_and_arrow(fig):
        """ Given a figure that has a second axes and an annotation arrow due to a 
        point having been hovered on, remove this axes and annotation arrow.  
        Inputs:
            fig | matplotlib figure 
        Returns:
        History:
            2020/09/08 | MEG | Written
        """
        # 1: try and remove any axes except the primary one
        try:
            fig.axes[1].remove()                
        except:
            pass
        
        # 2: try and remove any annotation arrows
        for art in axes1.get_children():
            if isinstance(art, matplotlib.patches.FancyArrow):
                try:
                    art.remove()        
                except:
                    continue
            else:
                continue
        fig.canvas.draw_idle()                                          # update the figure
    
    
    def axes_data_to_fig_percent(axes_lims, fig_lims, point):
        """ Given a data point, find where on the figure it plots (ie convert from axes coordinates to figure coordinates) 
        Inputs:
            axes_xlims | tuple | usually just the return of something like: axes1.get_ylim()
            fig_lims | tuple | the limits of the axes in the figure.  usuall (0.1, 0.9)  for an axes made with something like this: axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])                  # main axes
            point |float | point in data coordinates
        Returns:
            fig_position | float | where the data point is in the figure.  (0,0) would be the lower left corner.  
        History:
            2020/09/08 | MEG | Written
            
        """
        gradient = (fig_lims[1] - fig_lims[0])/(axes_lims[1] - axes_lims[0])
        y_intercept = fig_lims[0] - (gradient * axes_lims[0])
        fig_position = (gradient * point) + y_intercept
        return fig_position
    
    def calculate_insetaxes_offset(lims, points, offset_length):
        """
        The offsets between the inset axes and the point are different depending on which quadrant of the graph the point is in.  
        Inputs:
            lims | list | length is equal to the number of dimensions.  Filled with tuples of the axes limits.  
            point | list | length is equal to the number of diemsions. Filled with points.  
            offset_length | float | length of the arrow.  
        Returns:
            offsets | list | length is equal to the number of dimensions.  Length of offset for inset axes in each dimension.  
        History:
            2020/09/08 | MEG | Written
        """
        import numpy as np
        offsets = []
        for dim_n in range(len(lims)):                                        # loop through each dimension.  
            dim_centre = np.mean(lims[dim_n])
            if points[dim_n] < dim_centre:
                offsets.append(-offset_length)
            else:
                offsets.append(offset_length)
        return offsets
    
    def hover(event):
        if event.inaxes == axes1:                                                       # determine if the mouse is in the axes
            cont, ind = sc.contains(event)                                              # cont is a boolean of if hoving on point, ind is a dictionary about the point being hovered over.  Note that two or more points can be in this.  
            if cont:                                                                    # if on point
                remove_axes2_and_arrow(fig)                                             # remove the axes and arrow created when hovering on the point (incase cursor moves from one point to next without going off a point)
                point_n = ind['ind'][0]                                                 # get the index of which data point we're hovering on in a simpler form.      
                
                # 1: Add the annotation arrow (from inset axes to data point)
                arrow_lengths = calculate_insetaxes_offset([axes1.get_xlim(), axes1.get_ylim()], 
                                                          [xy[0,point_n], xy[1,point_n]], arrow_length)                               # calculate the length of the arrow, which depends which quadrant we're in (as the arrow always go away from the plot)
                axes1.arrow(xy[0,point_n] + arrow_lengths[0], xy[1,point_n] + arrow_lengths[1],                                       # add the arrow.  Notation is all a bit backward as head is fixed at end, so it has to be drawn backwards.  
                            -arrow_lengths[0], -arrow_lengths[1], clip_on = False, zorder = 999)                                # clip_on makes sure it's visible, even if it goes off the edge of the axes.  

                # 2: Add the inset axes                
                fig_x = axes_data_to_fig_percent(axes1.get_xlim(), (0.1, 0.9), xy[0,point_n] + arrow_lengths[0])                   # convert position on axes to position in figure, ready to add the inset axes
                fig_y = axes_data_to_fig_percent(axes1.get_ylim(), (0.1, 0.9), xy[1,point_n] + arrow_lengths[1])                   # ditto for y dimension
                if arrow_lengths[0] > 0 and arrow_lengths[1] > 0:                                                          # top right quadrant
                    inset_axes = fig.add_axes([fig_x, fig_y,                                                               # create the inset axes, simple case, anochored to lower left forner
                                               inset_axes_side['x'], inset_axes_side['y']], anchor = 'SW')               
                elif arrow_lengths[0] < 0 and arrow_lengths[1] > 0:                                                        # top left quadrant
                    inset_axes = fig.add_axes([fig_x - inset_axes_side['x'], fig_y,                                        # create the inset axes, nudged in x direction, anchored to lower right corner
                                               inset_axes_side['x'], inset_axes_side['y']], anchor = 'SE')     
                elif arrow_lengths[0] > 0 and arrow_lengths[1] < 0:                                                        # lower right quadrant
                    inset_axes = fig.add_axes([fig_x, fig_y - inset_axes_side['y'],                                        # create the inset axes, nudged in y direction
                                               inset_axes_side['x'], inset_axes_side['y']], anchor = 'NW')                 
                else:                                                                                                      # lower left quadrant
                    inset_axes = fig.add_axes([fig_x - inset_axes_side['x'], fig_y - inset_axes_side['y'],                 # create the inset axes, nudged in both x and y
                                               inset_axes_side['x'], inset_axes_side['y']], anchor = 'NE')                
                
                # 3: Plot on the inset axes
                if temporal_data is not None:
                    inset_axes.plot(temporal_data['xvals'], temporal_data['tcs_r2'][point_n,])                            # draw the inset axes time course graph
                if spatial_data is not None:
                    inset_axes.imshow(spatial_data['images_r3'][point_n,])                                                      # or draw the inset axes image
                inset_axes.set_xticks([])                                                                                       # and remove ticks (and so labels too) from x
                inset_axes.set_yticks([])                                                                                       # and from y
                fig.canvas.draw_idle()                                                                                          # update the figure.  
            else:                                                                       # else not on a point
                remove_axes2_and_arrow(fig)                                             # remove the axes and arrow created when hovering on the point                       
        else:                                                                           # else not in the axes
            remove_axes2_and_arrow(fig)                                                 # remove the axes and arrow created when hovering on the point (incase cursor moves from one point to next without going off a point)
    
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # 1: Check some inputs:
    if temporal_data is None and spatial_data is None:                                                                  # check inputs
        raise Exception("One of either spatial or temporal data must be supplied.  Exiting.  ")
    if temporal_data is not None and spatial_data is not None:
        raise Exception("Only either spatial or temporal data can be supplied, but not both.  Exiting.  ")
    
    
    
    # 2: Draw the figure
    fig = plt.figure(figsize = figsize)                                             # create the figure, size set in function args.  
    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])                                      # main axes
    if markers is None:                                                             # if a dictionary about different markers is not supplied, 
        sc = axes1.scatter(xy[0,],xy[1,],c=colours, s=100)                                # draw the scatter plot, just draw them all with the default maker
    else:                                                                                                                           # but if we do have a dictionary of markers.  
        n_markers = len(markers['styles'])                                                                                          # get the number of unique markers
        for n_marker in range(n_markers):                                                                                           # loop through each marker style
            point_args = np.ravel(np.argwhere(markers['labels'] == n_marker))                                                                 # get which points have that marker style
            sc = axes1.scatter(xy[0,point_args],xy[1,point_args],c=colours[point_args], s=100, marker = markers['styles'][n_marker])      # draw the scatter plot with different marker styles
        sc = axes1.scatter(xy[0,],xy[1,],c=colours, s=100, alpha = 0.0)                                                                   # draw the scatter plot again, but with invisble markers.  As the last to be drawn, these are the ones that 
                                                                                                                                    # are hovered over, and indexing works as all the points are draw this time.  
    # 3: Try and add various labels from the labels dict
    try:
        fig.canvas.set_window_title(labels['title'])
        fig.suptitle(labels['title'])
    except:
        pass
    try:
        axes1.set_xlabel(labels['xlabel'])
    except:
        pass
    try:
        axes1.set_ylabel(labels['ylabel'])
    except:
        pass
    
    # 4: Possibly add a legend, using the legend dict.  
    if legend is not None:
        axes1.legend(handles = legend['elements'], labels = legend['labels'], 
                     bbox_to_anchor=(1., 0.5), loc = 'center right', bbox_transform=plt.gcf().transFigure)                           # Put a legend to the right of the current axis.  bbox is specified in figure coordinates.  
              
    fig.canvas.mpl_connect("motion_notify_event", hover)                                # connect the figure and the function.  


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
        out_folder | string | path to location to save .kmz.  Should have a trailing /
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
   
    
   
    # if lats[0] < lats[-1]:
    #     print(f"Reversing the lats.  lats[0] should be the bottom left, and therefore the lowest value.  Previously lats[0] was {lats[0]}, but now it is ", end = '')
    #     lats = lats[::-1]
    #     print(f"{lats[0]}.  ")


    n_images = images_r3_ma.shape[0]    
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
    kml.savekmz(f"{out_folder}{kmz_filename}.kmz", format=False)                                    # Saving as KMZ
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
        
    """
    import matplotlib.pyplot as plt
    
    f, ax = plt.subplots(1,1)
    ax.imshow(r2)
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
