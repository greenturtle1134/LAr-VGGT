import math
import numpy as np

L = 768
Ls = L*np.sqrt(3)/2 # Default size of the projected "window"
def project_2d(image, proj,
               x_min = -Ls, x_max = Ls, x_pix = L,
               y_min = -Ls, y_max = Ls, y_pix = L
              ):
    """
    Projects a voxel image to a list of sparse 2D pixels
    :param image: the 3D voxel image, a NxD array. The first four columns are the three spatial dimensions, then value
    :param proj: the 2x3 projection matrix to be used
    :param x_min: minimum x value
    :param x_max: maximum x value
    :param x_pix: number of pixels on x axis
    :param y_min: minimum y value
    :param y_max: maximum y value
    :param y_pix: number of pixels on y axis
    :returns: tuple of (coords, values) where coords is a N'x2 array of integers representing coordinates of nonzero pixels, and values is a N-length array of total values
    """
    # FOR NOW ignore lower rows of projection matrix; will definitely need them later
    if proj.shape[0] > 2:
        proj = proj[:2,:]
    
    # Projection of points onto 2D plane
    projected = np.hstack([image[:, :3] @ proj.T, image[:, 3:]])

    # Prune points not within range
    projected = projected[
        (projected[:,0] >= x_min) & (projected[:,0] <= x_max) &
        (projected[:,1] >= y_min) & (projected[:,1] <= y_max)
    ]

    # Computation of integer coordinates
    projected_coords = np.column_stack((
        np.digitize(projected[:,0], np.linspace(x_min, x_max, x_pix+1)) - 1,
        np.digitize(projected[:,1], np.linspace(y_min, y_max, y_pix+1)) - 1
    ))

    # Aggregation of points into pixels
    coords, inv = np.unique(projected_coords, axis=0, return_inverse=True)
    values = np.bincount(inv, weights=projected[:,2])
    # return np.column_stack((coords, values))
    return coords, values


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


def patchify(coords, values, x_pix, y_pix, x_patch, y_patch):
    """
    Divides a 2D sparse pixel image into sparse patches.
    :param coords: Nx2 array of nonzero pixels, as returned by project_2d
    :param values: N-length array of pixel values, as returned by project_2d
    :x_pix: the number of pixels in x
    :y_pix: the number of pixels in y
    :x_patch: the size of the patch in x
    :y_patch: the size of the patch in y
    :returns: tuple (P, patch_coords, patches) where P is the number of patches, patch_coords is a Px2 containing coordinates of each non-zero patch (in units of patches), and patches is a PxD1xD2 of patch images
    """
    # Locate the patch coordinates of each nonzero pixel
    patch_coords = np.column_stack((
        np.digitize(coords[:,0], np.arange(0, x_pix, x_patch)) - 1,
        np.digitize(coords[:,1], np.arange(0, y_pix, y_patch)) - 1
    ))
    
    #Identify nonzero patches and patch assignments
    patch_coords, inv = np.unique(patch_coords, axis=0, return_inverse=True)
    
    # Get the number of patches
    P, _ = patch_coords.shape
    
    # Compute the x and y coordinates within each point
    offset_coords = coords - patch_coords[inv] * np.array([x_patch, y_patch]) # Coords to the corner of each patch
    offset_x, offset_y = offset_coords[:,0], offset_coords[:,1]
    
    # Fill in the final array
    patches = np.zeros((P, x_patch, y_patch))
    patches[inv, offset_x, offset_y] = values
    
    return P, patch_coords, patches