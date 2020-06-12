#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 18:19:37 2019

@author: matthew
"""

  
def component_plot(spatial_map, pixel_mask, timecourse, shape, title, shared = 0, 
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


def pca_variance_line(pc_vals, title = '', figures = 'window', png_path = './'):
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
        lons | rank 1 array | lons of each pixel in the image.  
        lats | rank 1 array | lats of each pixel in theimage
        layer_name_prefix | string | Can be used to set the name of the layes in the kmz (nb of the form layer_name_prefix_001 etc. )
        kmz_filename | string | Sets the name of the kmz produced
        out_folder | string | path to location to save .kmz.  Should have a trailing /
    Returns:
        kmz file
    History:
        2020/06/10 | MEG | Written
        2020/06
    """
    import numpy as np
    import os
    import shutil
    import simplekml
   

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
    
        ground.gxlatlonquad.coords = [(lons[0], lats[0]), (lons[-1],lats[0]),           # lon, lat of image south west, south east
                                      (lons[-1], lats[-1]), (lons[0],lats[-1])]         # north east, north west  - order is anticlockwise around the square, startign in the lower left
       
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
