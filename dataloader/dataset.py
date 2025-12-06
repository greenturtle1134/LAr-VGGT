import math
import numpy as np
import random

from scipy.spatial.transform import Rotation
import h5py as h5

from dataloader.preprocessing import *

# This function will have to change as the supervision changes!
def project_and_patchify(image, proj, x_min, x_max, x_pix, x_patch, y_min, y_max, y_pix, y_patch, threshold):
    # Compute projection
    coords, inv, depth, mask = project_2d(image, proj,
                                          x_min=x_min, x_max=x_max, x_pix=x_pix,
                                          y_min=y_min, y_max=y_max, y_pix=y_pix
                                         )
    point_values = image[:,3][mask]

    # Sum the values and take the mean distance
    values = np.bincount(inv, weights=point_values)
    depth_sums = np.bincount(inv, weights=depth*point_values)
    values = np.bincount(inv, weights=point_values)
    mean_depth = depth_sums / values
    
    # Split into patches
    P, patch_coords, offset_coords, patch_inv = split_patches(coords, x_patch, y_patch)
    patches = construct_patches(P, offset_coords, patch_inv, values, x_patch, y_patch)
    depth_patches = construct_patches(P, offset_coords, patch_inv, mean_depth, x_patch, y_patch)

    # Cut based on thresholding
    mask = patches.sum(axis=(1, 2)) > threshold
    patch_coords, patches, depth_patches = patch_coords[mask], patches[mask], depth_patches[mask]

    return patch_coords, patches, depth_patches

class Dataset:
    def __init__(self, images, image_clusters, pixel_res = (256, 256), patch_size = (16, 16), x_range = None, y_range = None, L = 768, threshold=0):
        if x_range is None:
            x_range = (-L/2, L/2)
        if y_range is None:
            y_range = (-L/2, L/2)

        # Store parameters
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_pix, self.y_pix = pixel_res
        self.x_patch, self.y_patch = patch_size
        self.L = L
        self.images = images
        self.image_clusters = image_clusters
        self.threshold = threshold

        # This is really a parameter only used for testing
        self.angle_limit = 0


    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        # If the key is a slice, return a new Dataset
        if isinstance(key, slice):
            # slice the lists
            new_images = self.images[key]
            new_clusters = self.image_clusters[key]

            # create a new instance
            new_obj = self.__class__(
                new_images, new_clusters,
                x_range = (self.x_min, self.x_max),
                y_range = (self.y_min, self.y_max),
                pixel_res = (self.x_pix, self.y_pix),
                patch_size = (self.x_patch, self.y_patch),
                L = self.L,
                threshold = self.threshold
            )
            
            return new_obj

        # If the key is an int, return the corresponding element
        return (self.images[key], self.image_clusters[key])

    def choose_events(self, N, S, return_intermediates = True, locked_rotations = None):
        chosen_events = random.sample(self.images, k=N) # if N < len(self.images) else self.images
        if locked_rotations == "orthogonal":
            bases = [Rotation.random() for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        elif locked_rotations == "limited":
            bases = [Rotation.from_euler('x', random.uniform(-90, 90), degrees=True) for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        elif locked_rotations == "limited_round":
            bases = [Rotation.from_euler('x', random.uniform(-180, 180), degrees=True) for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        elif locked_rotations == "xy":
            bases = [Rotation.from_euler('XY', [random.uniform(-180, 180), random.uniform(-180, 180)], degrees=True) for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        elif locked_rotations == "xz":
            bases = [Rotation.from_euler('XZ', [random.uniform(-180, 180), random.uniform(-180, 180)], degrees=True) for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        elif locked_rotations == "fixed":
            bases = [Rotation.from_euler('x', 0, degrees=True) for _ in range(N)]
            chosen_rotations = [[b, Rotation.from_euler('x', 90, degrees=True) * b, Rotation.from_euler('y', 90, degrees=True) * b] for b in bases]
        else:
            if locked_rotations is not None and locked_rotations != "none":
                print("Warning: unknown rotation {locked_rotations} found; using fully random")
            chosen_rotations = [[Rotation.random() for _ in range(S)] for _ in range(N)]

        results = [[project_and_patchify(
            event, rotation.as_matrix(), threshold=self.threshold,
            x_min=self.x_min, x_max=self.x_max, x_pix=self.x_pix, x_patch=self.x_patch,
            y_min=self.y_min, y_max=self.y_max, y_pix=self.y_pix, y_patch=self.y_patch
        ) for rotation in rotations] for event, rotations in zip(chosen_events, chosen_rotations)]

        return (results, chosen_rotations) if return_intermediates else results

def dataset_from_file(path, remove_noise_cluster=False, **kwargs):
    # Load the data
    f = h5.File(path, mode = 'r')
    
    # Reshape image and clusters lists
    images = [x.reshape(-1, 8) for x in f["point"]]
    image_clusters = [x.reshape(-1, 5) for x in f["cluster"]]

    # Cut out the first cluster if desired
    if remove_noise_cluster:
        images = [image[clusters[0,0]:,:] for image, clusters in zip(images, image_clusters)]
    
    # Translate images to center at origin
    for x in images:
        x[:,0:3] -= L/2

    # Construct the dataset
    return Dataset(images, image_clusters, **kwargs)