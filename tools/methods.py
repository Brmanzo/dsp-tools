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
    else:
        # All neighbors in the kernel, excluding the center pixel
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0 or ((not s.square_kernel) and abs(dr) + abs(dc) > radius):
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
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

def convolution(s: settings, img: np.ndarray) -> np.ndarray:
    ''' Applies a convolutional blur to the image using a square kernel.'''
    h, w = img.shape[:2]
    blurred_img = img.copy()
    for r in range(h):
        for c in range(w):
            neighbors = get_neighbors(s, img, (r, c), False)
            blurred_img[r, c] = np.mean([img[n] for n in neighbors] + [img[r, c]], axis=0).astype(np.uint8)
    return blurred_img