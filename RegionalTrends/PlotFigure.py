# Import all necessary packages for the figure
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
import os

def plot_map(data, lon, lat, 
            crange=None, 
            label=None, 
            cmap='viridis',
            interpolation=False, 
            extent=None, 
            cbar_orientation='vertical', 
            c_ticks=10,
            c_ticks_num=True,
            x_ticks=5,
            show_x_ticks=True,
            x_ticks_num=True,
            show_y_ticks=True,
            y_ticks=5,
            y_ticks_num=True,
            extreme_colors=False,
            clabel_size=32,
            ctick_size=28,
            tick_size=28,
            save_name=None):
    """Makes a world map figure of a 2D array.

	Args:
		data (2D array with shape [lat, lon]): The data you wish to plot
        lon (1D array): Array of longitudes
        lat (1D array): Array of latitudes
        crange (list, tuple or array with 2 elements): Minimum and maximum range of colorbar
        label (string): Label name of the colorbar
        cmap (string or obj): Colormap of the figure
        interpolation (boolean): Interpolates between grid cells for smoother figure
        ticks (list, tuple or array with 2 elements): Number of ticks for x and y axes
        extent (list or array with 4 elements): Extent of the figure map (East, West, South, North)
        cbar_orientation (string): Orientation of the colorbar
        c_ticks (float): Place colorbar ticklabels every 'c_ticks' increments
        colormarker (string): Color of the marker at Lobith
        extreme_colors (boolean or list/array/tuple): See comments below
        save_name (string): Saves figure with this specified name
	Returns:
		None (makes a figure and potentially saves it to directory)
	"""
    
    plt.rcParams['axes.unicode_minus'] = False

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()

    # Create meshgrid 2D array both for longitudes and latitudes
    Lon, Lat = np.meshgrid(lon, lat)
    
    # This code right here decides whether the colorbar will have an extension
    # or not. The extension of a colorbar is when the colorbar has a triangular
    # shape at the end of one of its limits. This triangular shape indicates that
    # either the minimum or maximum (depending on where this 'triangle' is located)
    # value in the data is outside the colorbar limits.
    if crange is not None:
        if np.nanmin(data) < crange[0] and np.nanmax(data) > crange[1]:
            # In this case the plotted data has values outside both the minimum and 
            # maximum range of the colorbar, meaning that we want a 'triangle' at both ends
            extension = 'both' 
        elif np.nanmin(data) < crange[0] and np.nanmax(data) <= crange[1]:
            # Here we only want a triangle for the lower limit of the colorbar
            extension = 'min'
        elif np.nanmin(data) >= crange[0] and np.nanmax(data) > crange[1]:
            # Here we only want a triangle for the upper limit of the colorbar
            extension = 'max'
        else:
            # Here all the data is within the colorbar range, so we do not want an extension.
            extension = 'neither'
    elif crange is None:
        crange = (np.nanmin(data), np.nanmax(data))
        extension = 'neither'
    
    # This is something that I once came up with and I still like a lot, although
    # it might look a bit unusual at first glance. In short, this piece of code
    # adjusts the provided colormap so that values outside the colorbar range
    # (below the minimum or above the maximum) get their own special colors.
    if extreme_colors == True:
        # When extreme_colors is True, the colormap is modified such that
        # out-of-range values receive distinct colors. These colors are taken
        # directly from the original lower and upper bounds of the colormap.
        # For instance, if the highest color of a discrete colormap is black,
        # that same black will be used as the "over-limit" color, while it is
        # removed from the main color range.
        # THIS OPTION IS MAINLY USEFUL FOR CERTAIN DISCRETE COLORMAPS.
        base_cmap = plt.get_cmap(cmap)  # Load the base colormap
        
        # Sample all colors from the original colormap
        basemap = base_cmap(np.linspace(0, 1, base_cmap.N))
        
        # Remove the original first and last colors from the main colormap
        cmap = ListedColormap(basemap[1:-1])
        
        # Assign the removed edge colors as special extremes
        cmap.set_over(basemap[-1])  # Use the old upper limit as the over-limit color
        cmap.set_under(basemap[0])  # Use the old lower limit as the under-limit color
    
    elif isinstance(extreme_colors, (list, tuple, np.ndarray)):
        # Here the user has provided the upper and lower extreme colors manually
        # themselves, without them being extracted from the main colormap
        under_color, over_color = extreme_colors
        if under_color is not None:
            cmap.set_under(under_color) # Set over-limit color
        if over_color is not None:
            cmap.set_over(over_color) # Set under-limit color
    else:
        # If no extreme colors are used, the colormap remains untouched.
        cmap = cmap 
    
    # The user has the option to use interpolation of data for plotting.
    # Originally the values are plotted for each grid cell, which is more accurate,
    # but there may be a reason to interpolate (such as simply obtaining a nicer
    # looking figure, or to upscale the resolution). The interpolation is done
    # using gouraud shading in ax.pcolormesh.
    if interpolation == True:
        shading = 'gouraud'
    elif interpolation == False:
        shading = 'auto'
    
    
    # Here we actually finally make the figure, first we define the figure
    # with a simple PlateCarree projection
    fig, ax = plt.subplots(1, figsize=(12, 8), constrained_layout=True,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # If the extent is none, then the full domain of the map will be displayed.
    # If it is not none, only the chosen portion will be displayed.
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add coastlines to the figure with high detail
    ax.coastlines(resolution='10m', linewidth=1.5)
    ax.add_feature(cfeature.BORDERS, linestyle='-', lw=1.5) # And add country borders
    ax.add_feature(cfeature.OCEAN, facecolor='#edf0f5')
    
    # Add gridlines to the figure corresponding to the ticklabels.
    gl = ax.gridlines(draw_labels=False, x_inline=False, y_inline=False, color='xkcd:black', alpha=0.4)
    
    eps = 1e-6
    xmin, xmax = lon.min()-eps, lon.max()+eps
    ymin, ymax = lat.min()-eps, lat.max()+eps

    # ----- X ticks -----
    if show_x_ticks:
        if x_ticks_num:
            xloc = ticker.MaxNLocator(nbins=x_ticks)
            xticks = xloc.tick_values(xmin, xmax)
        else:
            xloc = ticker.MultipleLocator(x_ticks)
            xticks = xloc.tick_values(xmin, xmax)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        gl.xlocator = ticker.FixedLocator(xticks)
    else:
        ax.set_xticks([])
        gl.xlocator = ticker.NullLocator()  # removes gridlines

    # ----- Y ticks -----
    if show_y_ticks:
        if y_ticks_num:
            yloc = ticker.MaxNLocator(nbins=y_ticks)
            yticks = yloc.tick_values(ymin, ymax)
        else:
            yloc = ticker.MultipleLocator(y_ticks)
            yticks = yloc.tick_values(ymin, ymax)

        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        gl.ylocator = ticker.FixedLocator(yticks)
    else:
        ax.set_yticks([])
        gl.ylocator = ticker.NullLocator()  # removes gridlines

    # Geographic label formatting
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f', degree_symbol='°'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f', degree_symbol='°'))

    ax.tick_params(labelsize=tick_size, length=10) # Tick label size and tick length
    
    # Here we actually plot the data with PlateCarree transform and potential
    # interpolation. Note that rasterized is added to mainly decrease file size
    mesh = ax.pcolormesh(Lon, Lat, data, transform=ccrs.PlateCarree(),
                         cmap=cmap, shading=shading, rasterized=True)
        
    mesh.set_clim(crange) # Set colorbar range
        
    # Next we create a new horizontal axis to the main plot where the colorbar will go.
    # The user can choose the colorbar to be either vertical or horizontal
    divider = make_axes_locatable(ax)
    
    if cbar_orientation == 'vertical':
        # Vertical colorbar on the right of the map
        ax_cb = divider.append_axes('right', size='3.8%', pad=0.2, axes_class=plt.Axes)
        fig.add_axes(ax_cb) # Add the newly created colorbar axis to the figure
        cbar = plt.colorbar(mesh, cax=ax_cb, extend=extension, orientation='vertical')
        cbar.ax.tick_params(labelsize=ctick_size, direction='in', length=8, left=True, right=True)  # Customize colorbar ticks
        if c_ticks_num:
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks)) # Set number of ticks on colorbar
        else:
            cbar.ax.yaxis.set_major_locator(MultipleLocator(c_ticks)) # Place ticklabel every 'c_ticks' increments
    
    elif cbar_orientation == 'horizontal':
        # Horizontal colorbar below the map
        ax_cb = divider.append_axes('bottom', size='6%', pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_cb) # Add the newly created colorbar axis to the figure
        cbar = plt.colorbar(mesh, cax=ax_cb, extend=extension, orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=ctick_size, direction='in', length=8, top=True, bottom=True)  # Customize colorbar ticks
        if c_ticks_num:
            cbar.ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=c_ticks)) # Set number of ticks on colorbar
        else:
            cbar.ax.xaxis.set_major_locator(MultipleLocator(c_ticks)) # Place ticklabel every 'c_ticks' increments

    cbar.set_label(label, fontsize=clabel_size, labelpad=10)  # Set colorbar label and font size
    
    # And here we save the figure, if a save_name is provided
    if save_name is not None:
        folder = './figures/'
        save_path = os.path.join(folder, save_name + '.jpg')
        plt.savefig(save_path, format='jpg', bbox_inches='tight', dpi=800)
        
    plt.show()