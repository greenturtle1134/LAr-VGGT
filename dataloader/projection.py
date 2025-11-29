import math
import numpy as np

L = 768
Ls = L*np.sqrt(3)/2 # Default size of the projected "window"
def project_2d(image, proj,
               x_min = -Ls, x_max = Ls, x_pix = L,
               y_min = -Ls, y_max = Ls, y_pix = L
              ):
    """
    Projects a voxel image to a list of sparse 2D pixels, recently modified
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
    # # FOR NOW ignore lower rows of projection matrix; will definitely need them later
    # if proj.shape[0] > 2:
    #     proj = proj[:2,:]
    
    # Rotation of points as per matrix
    rotated = image[:, :3] @ proj.T

    # Prune points not within range
    rotated = rotated[
        (rotated[:,0] >= x_min) & (rotated[:,0] <= x_max) &
        (rotated[:,1] >= y_min) & (rotated[:,1] <= y_max)
    ]

    # Extract x and y coordinates, round to nearest pixel
    projected_x = np.trunc(x_pix * (rotated[:,0] - x_min) / (x_max - x_min)).astype(int)
    projected_y = np.trunc(y_pix * (rotated[:,1] - y_min) / (y_max - y_min)).astype(int)
    # projected_x = np.floor_divide(x_pix * (rotated[:,0] - x_min), (x_max - x_min))
    # projected_y = np.floor_divide(y_pix * (rotated[:,1] - y_min), (y_max - y_min))
    projected_coords = np.column_stack((projected_x, projected_y)).astype(int)

    # Aggregation of points into pixels
    coords, inv = np.unique(projected_coords, axis=0, return_inverse=True)
    values = np.bincount(inv, weights=image[:,3])
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
    
    # Identify nonzero patches and patch assignments
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


# def patchify_set(coords, values, ids, x_pix, y_pix, x_patch, y_patch):
#     """
#     Divides a set of 2D sparse pixel images into sparse patches.
#     :param coords: Nx2 array of nonzero pixels, as returned by project_2d
#     :param values: N-length array of pixel values, as returned by project_2d
#     :param ids: N-length array of identifications of images
#     :x_pix: the number of pixels in x
#     :y_pix: the number of pixels in y
#     :x_patch: the size of the patch in x
#     :y_patch: the size of the patch in y
#     :returns: tuple (P, patch_coords, patches) where P is the number of patches, patch_coords is a Px3 containing [x_coord, y_coord, id] of every non-zero patch, and patches is a PxD1xD2 of patch images
#     """
#     # Locate the patch coordinates of each nonzero pixel
#     patch_coords = np.column_stack((
#         np.digitize(coords[:,0], np.arange(0, x_pix, x_patch)) - 1,
#         np.digitize(coords[:,1], np.arange(0, y_pix, y_patch)) - 1,
#         ids
#     ))
    
#     #Identify nonzero patches and patch assignments
#     patch_coords, inv = np.unique(patch_coords, axis=0, return_inverse=True)  # patch_coords is now an array of [x, y, id]
    
#     # Get the number of patches
#     P, _ = patch_coords.shape
#     N = np.max(patch_coords[:3])
    
#     # Compute the x and y coordinates within each point
#     offset_coords = coords - patch_coords[inv][:2] * np.array([x_patch, y_patch]) # Coords to the corner of each patch
#     offset_x, offset_y = offset_coords[:,0], offset_coords[:,1]
#     image_id = patch_coords[inv]
    
#     # Fill in the final array
#     patches = np.zeros((P, x_patch, y_patch, N))
#     patches[inv, offset_x, offset_y, image_id] = values
    
#     return P, patch_coords, patches