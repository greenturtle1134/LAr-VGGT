import h5py as h5
import random
from scipy.spatial.transform import Rotation
from dataloader.projection import *

class Dataset:
    def __init__(self, path, pixel_res = (256, 256), patch_size = (16, 16), x_range = None, y_range = None, L = 768):
        # Infer ranges as the minimum size to contain the whole cube
        Ls = L*np.sqrt(3)/2
        if x_range is None:
            x_range = (-Ls, Ls)
        if y_range is None:
            y_range = (-Ls, Ls)

        # Store parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_pix, self.y_pix = pixel_res
        self.x_patch, self.y_patch = patch_size
        self.L = L
        
        # Load the data
        f = h5.File(path, mode = 'r')
        
        # Reshape image and clusters lists
        self.images = [x.reshape(-1, 8) for x in f["point"]]
        self.image_clusters = [x.reshape(-1, 5) for x in f["cluster"]]
        
        # Translate images to center at origin
        for x in self.images:
            x[:,0:3] -= L/2

    def choose_events(self, N, S, return_intermediates = False):
        chosen_events = random.choices(self.images, k=N)
        chosen_rotations = [[Rotation.random() for _ in range(S)] for _ in range(N)]

        projections = [[project_2d(
            event, rotation.as_matrix(),
            x_min=self.x_min,
            x_max=self.x_max,
            x_pix=self.x_pix,
            y_min=self.y_min,
            y_max=self.y_max,
            y_pix=self.y_pix
        ) for rotation in rotations] for event, rotations in zip(chosen_events, chosen_rotations)]

        patches = [[patchify(
            coords, values,
            x_pix=self.x_pix, y_pix=self.y_pix,
            x_patch=self.x_patch, y_patch=self.y_patch
        ) for coords, values in p] for p in projections]

        return (patches, projections, chosen_rotations) if return_intermediates else patches


def stack_patches(patches):
    # NOTE: patch_counts will only be a "proper" array if we're assuming an equal view count for all events!
    # This is what I'm going with for now, but it might change!
    patch_counts = np.array([[x for x, _, _ in event] for event in patches])
    all_coords = np.concatenate([x for event in patches for _, x, _ in event])
    all_patches = np.concatenate([x for event in patches for _, _, x in event])
    return patch_counts, all_coords, all_patches