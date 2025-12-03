import math
import numpy as np

from scipy.spatial.transform import Rotation

L = 768
# Ls = L*np.sqrt(3)/2 # Default size of the projected "window"
Ls = L/2 # Default size of the projected "window"
def project_2d(image, proj,
               x_min = -Ls, x_max = Ls, x_pix = L,
               y_min = -Ls, y_max = Ls, y_pix = L
              ):
    """
    Projects a voxel image to a list of sparse 2D pixels, recently modified
    :param image: the 3D voxel image, a NxD array. The first three columns are the coordinates to be rotated
    :param proj: the 2x3 projection matrix to be used
    :param x_min: minimum x value
    :param x_max: maximum x value
    :param x_pix: number of pixels on x axis
    :param y_min: minimum y value
    :param y_max: maximum y value
    :param y_pix: number of pixels on y axis
    :returns: tuple of (coords, inv, depth, mask) where coords is a N'x2 array of coordinates of nonzero pixels, inv is a N array mapping points to their pixels, depth is a N array of depths, and mask is a N mask.
    """
    
    # Rotation of points as per matrix
    rotated = image[:, :3] @ proj.T

    # Prune points not within range
    mask = (rotated[:,0] >= x_min) & (rotated[:,0] < x_max-1) & (rotated[:,1] >= y_min) & (rotated[:,1] < y_max-1) # For some reason the -1 is necessary? Needs to be looked into more.
    rotated = rotated[mask]

    # Extract x and y coordinates, round to nearest pixel
    projected_x = np.trunc(x_pix * (rotated[:,0] - x_min) / (x_max - x_min))
    projected_y = np.trunc(y_pix * (rotated[:,1] - y_min) / (y_max - y_min))
    # projected_x = np.floor_divide(x_pix * (rotated[:,0] - x_min), (x_max - x_min))
    # projected_y = np.floor_divide(y_pix * (rotated[:,1] - y_min), (y_max - y_min))
    projected_coords = np.column_stack((projected_x, projected_y)).astype(int)

    # Aggregation of points into pixels
    coords, inv = np.unique(projected_coords, axis=0, return_inverse=True)

    return coords, inv, rotated[:,2], mask


def split_patches(coords, x_patch, y_patch):
    """
    Divides a 2D sparse pixel set into sparse patches.
    :param coords: Nx2 array of nonzero pixels, as returned by project_2d
    :x_patch: the size of the patch in x
    :y_patch: the size of the patch in y
    :returns: tuple (P, patch_coords, offset_coords, patch_inv) where P is the number of patches, patch_coords is a Px2 containing coordinates of each patch, offset_coords is a Nx2 containing coordinates of each point within the patch, and patch_inv is a N mapping points to patches
    """
    # Locate the patch coordinates of each nonzero pixel
    patch_coords, offset_coords = np.divmod(coords, np.array([x_patch, y_patch]))
    
    # Identify nonzero patches and patch assignments
    patch_coords, patch_inv = np.unique(patch_coords, axis=0, return_inverse=True)

    # Find the count of patches
    P, _ = patch_coords.shape

    return P, patch_coords, offset_coords, patch_inv


def construct_patches(P, offset_coords, patch_inv, values, x_patch, y_patch):
    """
    Constructs patches once pixels have already been assigned to them
    :P: the number of patches
    :offset_coords: Nx2 of within-patch coords of each pixel, as returned by split_patches
    :patch_inv: N assignment of pixels to patches, as returned by split_patches
    :values: either N or NxD of values assigned to each pixel (values can be a vector)
    :x_patch: the size of the patch in x
    :y_patch: the size of the patch in y
    :returns: PxXxY of patches, or PxXxYxD if a vector value was given
    """    
    # Allocate the final array
    if len(values.shape) > 1:
        # values have their own dimension
        patches = np.zeros((P, x_patch, y_patch, *values.shape[1:]))
    else:
        # values is only a single value
        patches = np.zeros((P, x_patch, y_patch))

    # Add values to patches (same either way)
    offset_x, offset_y = offset_coords[:,0], offset_coords[:,1]
    patches[patch_inv, offset_x, offset_y] = values
    
    return patches


def stack_patches(patches):
    """
    Stacks a bunch of patches together for batch-processing
    :param patches: nested list of shape NxS, each element is a tuple of patch arrays to concatenate of shape Px??? each. Typically this is coordinates (Px2), values (PxHxW), and depths (PxHxW)
    :returns: tuple (patch_counts, ...) where patch_counts is a NxS of counts for each view, every other element is a concatenated array
    """
    D = len(patches[0][0])
    patch_counts = np.array([[x[0].shape[0] for x in event] for event in patches])
    all_stacks = [np.concatenate([x[i] for event in patches for x in event]) for i in range(D)]
    return patch_counts, *all_stacks