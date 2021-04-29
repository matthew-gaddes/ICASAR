#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:05:11 2021

@author: matthew
"""

#%%

def plot_2d_interactive_fig(xy, colours, spatial_data = None, temporal_data = None,
                        inset_axes_side = {'x':0.1, 'y':0.1}, arrow_length = 0.1, figsize = (10,6), 
                        labels = None, legend = None, markers = None, 
                        figures = 'window', png_path = './', fig_filename = '2d_interactive_plot'):
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
        figures | string,  "window" / "png" / "png+window" | controls if figures are produced (either as a window, saved as a png, or both)
        png_path | string | if a png is to be saved, a path to a folder can be supplied, or left as default to write to current directory.  
        fig_filename | string | name of file, if you wish to set one.  Doesn't include the extension (as it's always a png).  
     Returns:
        Interactive figure
    History:
        2020/09/09 | MEG | Modified from a sript in the ICASAR package.  
        2020/09/10 | MEG | Add labels, and change so that images are stored as rank3 arrays.  
        2020/09/10 | MEG | Add legend option.  
        2020/09/11 | MEG | Add option to have different markers.  
        2020/09/15 | MEG | Add option to set size of inset axes.  
        2021_04_16 | MEG | Add figures option (png, png and window, or just window), option to save to a directory, and option to set filename.  
    
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
    fig = plt.figure(figsize = figsize)                                                                # create the figure, size set in function args.  
    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])                                                         # main axes
    if markers is None:                                                                                # if a dictionary about different markers is not supplied... 
        sc = axes1.scatter(xy[0,],xy[1,],c=colours, s=100)                                             # draw the scatter plot, just draw them all with the default marker
    else:                                                                                                                                     # but if we do have a dictionary of markers.  
        n_markers = len(markers['styles'])                                                                                                    # get the number of unique markers
        for n_marker in range(n_markers):                                                                                                     # loop through each marker style
            point_args = np.ravel(np.argwhere(markers['labels'] == n_marker))                                                                 # get which points have that marker style
            try:
                sc = axes1.scatter(xy[0,point_args], xy[1,point_args], c=colours[point_args], s=100, marker = markers['styles'][n_marker])        # draw the scatter plot with different marker styles
            except:
                pass
        sc = axes1.scatter(xy[0,],xy[1,],c=colours, s=100, alpha = 0.0)                                                                       # draw the scatter plot again with all the points (regardless of marker style), but with invisble markers.  As the last to be drawn, these are the ones that are hovered over, and indexing works as all the points are draw this time.  

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
    
    if figures == 'window':
        pass
    elif figures == "png":
        fig.savefig(f"{png_path}/{fig_filename}.png")
        plt.close()
    elif figures == 'png+window':
        fig.savefig(f"{png_path}/{fig_filename}.png")
    else:
        pass


#%% From LiCSAlert



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
    from auxiliary_functions import col_to_ma

        
    
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

def add_square_plot(x_start, x_stop, y_start, y_stop, ax, colour = 'k'):
    """Draw localization square around an area of interest, x_start etc are in pixels, so (0,0) is top left.  
    Inputs:
        x_start | int | start of box
        x_stop | int | etc. 
        y_start | int |
        y_ stop | int |
        ax | axes object | axes on which to draw
        colour | string | colour of bounding box.  Useful to change when plotting labels, and predictions from a model.  
    
    Returns:
        box on figure
        
    History:
        2019/??/?? | MEG | Written
        2020/04/20 | MEG | Document, copy to from small_plot_functions to LiCSAlert_aux_functions
    """
        
    ax.plot((x_start, x_start), (y_start, y_stop), c= colour)           # left hand side
    ax.plot((x_start, x_stop), (y_stop, y_stop), c= colour)             # bottom
    ax.plot((x_stop, x_stop), (y_stop, y_start), c= colour)             # righ hand side
    ax.plot((x_stop, x_start), (y_start, y_start), c= colour)             # top
    

#%% From InSAR tools

def acquisitions_from_ifg_dates(ifg_dates):
    """Given a list of ifg dates in the form YYYYMMDD_YYYYMMDD, get the unique YYYYMMDDs that acquisitions were made on.  
    Inputs:
        ifg_dates | list | of strings in form YYYYMMDD_YYYYMMDD.  Called imdates in LiCSBAS nomenclature.  
    Returns:
        acq_dates | list | of strings in form YYYYMMDD
    History:
        2021_04_12 | MEG | Written        
    """
    acq_dates = []
    for ifg_date in ifg_dates:                                          # loop through the dates for each ifg
        dates = ifg_date.split('_')                                     # split into two YYYYMMDDs
        for date in dates:                                              # loop through each of these
            if date not in acq_dates:                                   # if it't not already in the list...
                acq_dates.append(date)                                  # add to it
    return acq_dates


def baseline_from_names(names_list):
    """Given a list of ifg names in the form YYYYMMDD_YYYYMMDD, find the temporal baselines in days_elapsed (e.g. 12, 6, 12, 24, 6 etc.  )
    Inputs:
        names_list | list | in form YYYYMMDD_YYYYMMDD
    Returns:
        baselines | list of ints | baselines in days
    History:
        2020/02/16 | MEG | Documented
    """

    from datetime import datetime, timedelta

    baselines = []
    for file in names_list:

        master = datetime.strptime(file.split('_')[-2], '%Y%m%d')
        slave = datetime.strptime(file.split('_')[-1][:8], '%Y%m%d')
        baselines.append(-1 *(master - slave).days)
    return baselines


#%% Small plot functions

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
    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    import numpy as np
    new_cmap = colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
    return new_cmap 
