import numpy as np
import cv2
from typing import Tuple
from tools.data_structures import settings

def sample_bg(img: np.ndarray) -> tuple:
    ''' Sample the border pixels of the image and return the median color as the background color. '''
    h, w = img.shape[:2]
    # Collect edge coordinates directly into a numpy array
    top    = np.stack([(0, c) for c in range(w)])
    bottom = np.stack([(h-1, c) for c in range(w)])
    left   = np.stack([(r, 0) for r in range(1, h-1)])
    right  = np.stack([(r, w-1) for r in range(1, h-1)])
    border_arr = np.concatenate([top, bottom, left, right], axis=0)
    
    # Sample colors at these coordinates
    colors = img[border_arr[:, 0], border_arr[:, 1]]
    bg_color = tuple(np.median(colors, axis=0))
    return bg_color

def color_similar(color1: tuple, color2: tuple, tol: float) -> bool:
    ''' Returns true if color1 is within tol% of color2 for all channels.'''
    return bool(np.all((np.array(color1) >= np.array(color2) * (1 - tol)) & (np.array(color1) <= np.array(color2) * (1 + tol))))

def get_neighbors(s: settings, img: np.ndarray, pixel: tuple, blob: bool) -> list[Tuple[int, int]]:
    ''' Given a pixel coordinate (r, c), return neighbors within a configurable kernel size.
        For blob=True, returns only north and west neighbors (for DSU efficiency).
        For blob=False, returns all neighbors in a square kernel of size s.kernel_size (odd integer >= 3).
    '''
    r, c = pixel
    h, w = img.shape[:2]
    neighbors = []

    assert s.kernel_size >= 3 and s.kernel_size % 2 == 1, "kernel_size must be odd and >= 3"
    radius = s.kernel_size // 2

    if blob:
        # Only north and west neighbors for union-find efficiency
        if r > 0: neighbors.append((r - 1, c)) # North
        if c > 0: neighbors.append((r, c - 1)) # West
        if s.square_kernel and r > 0 and c > 0: neighbors.append((r - 1, c - 1)) # Northwest
        if s.square_kernel and r > 0 and c < w - 1: neighbors.append((r - 1, c + 1)) # Northeast
        return neighbors
    
    # All neighbors in the kernel, including the center pixel
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if s.square_kernel:
                    neighbors.append((nr, nc))
                else:
                    if abs(dr) + abs(dc) <= radius:
                        neighbors.append((nr, nc))
    return neighbors

def union_find(s: settings, img: np.ndarray, bg_mask: set, color_to_find: tuple) -> list[set[Tuple[int, int]]]:
    ''' Calculates the disjointed set union of all blobs in the image that are similar to color_to_find.'''
    h, w = img.shape[:2]
    blob_id = {}      # Maps a pixel (r,c) to its integer blob ID
    parent_blob = [0] # DSU parent list, indexed by blob ID
    next_blob_id = 1  # Next available blob ID

    # DSU Helper Functions
    def find(i: int) -> int:
        if parent_blob[i] == i:
            return i
        parent_blob[i] = find(parent_blob[i])
        return parent_blob[i]
    
    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent_blob[root_j] = root_i

    # Main Pass
    for r in range(h):
        for c in range(w):
            pixel = (r, c)
            if pixel in bg_mask:
                continue
            if color_similar(tuple(img[pixel]), color_to_find, s.flood_tol):
                # Get neighbor coordinates
                neighbor_coords = get_neighbors(s, img, pixel, True)

                # Build a clean set of neighbor root IDs
                processed_neighbor_ids = set()
                for neighbor in neighbor_coords:
                    if neighbor in blob_id:
                        # Find the root ID for the neighbor and add it to the set
                        processed_neighbor_ids.add(find(blob_id[neighbor]))

                # This is a new blob, add unique identifier to dictionary with pixel as key
                if not processed_neighbor_ids:
                    blob_id[pixel] = next_blob_id
                    parent_blob.append(next_blob_id)
                    parent_blob[next_blob_id] = next_blob_id
                    next_blob_id += 1
                else:
                    # 3. Work ONLY with the set of IDs
                    min_id = min(processed_neighbor_ids)
                    blob_id[pixel] = min_id
                    for nid in processed_neighbor_ids:
                        if nid != min_id:
                            union(min_id, nid)

    # Grouping Pass
    blobs_by_root = {}
    for pixel, id in blob_id.items():
        root = find(id)
        blobs_by_root.setdefault(root, set()).add(pixel)

    return sorted(list(blobs_by_root.values()), key=len, reverse=True)

def convolution(s: settings, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ''' Applies a convolutional blur to the image using a square kernel.'''
    is_color = img.ndim == 3
    
    if is_color:
        h, w, ch = img.shape
    else: # Grayscale
        h, w = img.shape
        ch = 1
    radius = s.kernel_size // 2
    # Pad image to avoid issues with edge
    if is_color:
        padded_img = np.pad(img, ((radius, radius), (radius, radius), (0,0)), mode='edge')
    else:
        padded_img = np.pad(img, ((radius, radius), (radius, radius)), mode='edge')

    output_activation = np.zeros_like(img, dtype=np.float32)

    for r in range(h):
        for c in range(w):
            # Extract kernel-sized neighborhood from input activation via slicing
            neighborhood = padded_img[r:r+s.kernel_size, c:c+s.kernel_size]
            if is_color:
                # Apply kernel to each channel
                for i in range(ch):
                     output_activation[r, c, i] = np.sum(neighborhood[:, :, i] * kernel)
            else: # Grayscale
                 output_activation[r, c] = np.sum(neighborhood * kernel)
            
    output_activation = np.clip(output_activation, 0, 255)
    return output_activation.astype(np.uint8)

def gaussian_function(x: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
    ''' Returns the value of the Gaussian function (PDF) at x. Accepts scalar or numpy array. '''
    variance = std_dev ** 2
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2) / (2 * variance))

def gaussian_derivative(x, mean, std_dev) -> np.ndarray:
    """Calculates the first derivative of the Gaussian function."""
    variance = std_dev ** 2
    return -((x - mean) / variance) * gaussian_function(x, mean, std_dev)

def sobel_gaussian_curve_sample(s: settings) -> tuple[np.ndarray, np.ndarray]:
    ''' Returns the Sobel operator kernels for the given settings.'''
    assert s.kernel_size >= 3 and s.kernel_size % 2 == 1, "kernel_size must be odd and >= 3"
    # Should always be centered at 0
    mean = s.kernel_size // 2
    std_dev = (s.kernel_size - 1) / 6.0

    x_values = np.arange(s.kernel_size)
    # Gaussian curve for smoothing
    kernel_1d_smooth = gaussian_function(x_values, mean, std_dev)
    # Derivative of Gaussian for edge detection
    kernel_1d_deriv = gaussian_derivative(x_values, mean, std_dev)
    # Sobel Kernels are the matrix product of the two 1D kernels
    kernel_x = np.outer(kernel_1d_smooth, kernel_1d_deriv)
    kernel_y = np.outer(kernel_1d_deriv, kernel_1d_smooth)

    return kernel_x, kernel_y
