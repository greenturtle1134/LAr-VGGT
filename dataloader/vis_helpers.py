import math
import numpy as np
import matplotlib.pyplot as plt

def show_patches(patches, ncols=4, inches_per_patch = 1, vrange=(0, 40)):
    """
    Displays a set of patches as a 
    :param patches: PxHxW array of patches to display
    :param ncols: number of columns for tableau (row count will be chosen automatically)
    :param inches_per_patch: scale factor for size of the resulting graph
    :returns: a matplotlib figure
    """
    P = patches.shape[0]
    nrows = math.ceil(P/ncols)
    print(f"{P} patches, creating {nrows}x{ncols} display")
    
    fig, axes = plt.subplots(figsize=(inches_per_patch*ncols, inches_per_patch*nrows), layout="constrained", nrows=nrows, ncols=ncols)
    for p, ax in zip(range(P), axes.flatten()):
        ax.imshow(patches[p], origin='lower', cmap='viridis', vmin=vrange[0], vmax=vrange[1])
    for ax in axes.flatten():
        ax.axis('off')
    
    return fig


def sparse_to_grid(coords, values, x_pix, y_pix):
    """
    Converts from the sparse 2D arrays to a dense 2D array, mostly for debugging purposes
    :param coords: coords as output by project_2d
    :param values: values as output by project_2d
    :param x_pix: number of pixels in x axis
    :param y_pix: number of pixels in y axis
    :returns: x_pix by y_pix array of values
    """
    grid = np.full((x_pix, y_pix), 0)
    x = coords[:, 0]
    y = coords[:, 1]
    grid[x, y] = values
    return grid