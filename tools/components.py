import numpy as np
import cv2
from typing import Tuple
from tools.preprocessing import measure_image
from tools.methods import color_similar, get_neighbors, union_find
from tools.data_structures import settings


def blob_fill(s: settings, img: np.ndarray, bg_mask: set, bg_color: tuple|int, recolor: tuple|int) -> Tuple[np.ndarray, set]:
    ''' Given an image and background mask, calculates the Disjointed Set Union of all blobs to recolor.'''
    h, w = img.shape[:2]
    img_size = h * w
    blob_th = img_size * s.blob_tol
    blob_mask = set()

    sorted_blobs = union_find(s, img, bg_mask, bg_color)

    if not sorted_blobs:
        print("No blobs were found.")
        return img, blob_mask

    print(f"Found {len(sorted_blobs)} blob(s). The largest has {len(sorted_blobs[0])} pixels.")
    blobs_recolored = 0
    for blob in sorted_blobs:
        # While blob size is above threshold
        if len(blob) < blob_th:
            break
        
        blobs_recolored += 1
        for pixel in blob:
            blob_mask.add(pixel)
            img[pixel] = recolor
            
    if blobs_recolored > 0:
        print(f"Recolored {blobs_recolored} blob(s) larger than {int(blob_th)} pixels.")

    return img, blob_mask

def flood_fill(s: settings, img: np.ndarray, bg_color: tuple|int, seed: tuple, recolor: tuple|int, mode:str) -> Tuple[np.ndarray, set]:
    ''' Given a seed pixel, propagates outwards to find similar pixels within tolerance.
        Returns the recolored image as well as the set of recolored pixels as a mask. 

        Supports both grayscale (single channel) and color images.
        Lakes specifies whether or not to search whole image for blobs.
        Square_kernel specifies whether to use 4-connectivity or 8-connectivity.'''
    ocean, land, beach = set(), set(), set()
    boat = seed
    map = set((r, c) for r in range(img.shape[0]) for c in range(img.shape[1]))

    if boat[0] < 0 or boat[0] >= img.shape[0] or boat[1] < 0 or boat[1] >= img.shape[1]:
        raise ValueError("boat is off the map.")
    beach.add(boat)

    # Determine if image is grayscale or color
    is_grayscale = img.ndim == 2

    # While there is still a beach to explore
    while beach:
        pixel = beach.pop()
        # If explored pixel, skip
        if pixel in ocean or pixel in land:
            continue
        # Get pixel value
        pixel_val = img[pixel] if is_grayscale else tuple(img[pixel])
        # Ensure pixel_val is a tuple or float for color_similar
        if isinstance(pixel_val, np.ndarray):
            pixel_val = tuple(pixel_val.tolist())

        if color_similar(pixel_val, bg_color, s.flood_tol, mode):
            neighbors = get_neighbors(s, img, pixel, False)
            found_bg_neighbor = False
            for n in neighbors:
                # Also correct the neighbor value handling
                neighbor_val = img[n] if is_grayscale else tuple(img[n])
                if isinstance(neighbor_val, np.ndarray):
                    neighbor_val = tuple(neighbor_val.tolist())
                if color_similar(neighbor_val, bg_color, s.flood_tol, mode):
                    found_bg_neighbor = True
                    break #i love gooning soooooo much heheheheh -Gooner Gonzalez    frrrrr n Nolan menzo
            
            if found_bg_neighbor:
                ocean.add(pixel)
                for neighbor in neighbors:
                    if neighbor not in ocean and neighbor not in land:
                        beach.add(neighbor)
            else:
                land.add(pixel)
        else:
            land.add(pixel)
            map.discard(pixel)
        # Otherwise, mark as land (foreground)
    # If searching for background within foreground (lakes), iterate over remaining map pixels
    if s.lakes:
        img, blob_mask = blob_fill(s, img, ocean, bg_color, recolor)
        ocean.update(blob_mask)
    # Recolor ocean as specified color
    for pixel in ocean:
        img[pixel] = recolor

    return img, ocean

def alias(s: settings, img: np.ndarray, erosion_history: list[set]) -> np.ndarray:
    ''' Given the previous eroded layers, reapply the aliasing by recoloring each
        pixel to the median color of their neighbors with a fading alpha.
        
        Fade factor controls how quickly the alpha fades (higher = quicker).
        Halo pct controls how many of the eroded layers to use for aliasing (0%-100%).'''
    num_layers = len(erosion_history)
    num_layers = int(np.ceil(num_layers * s.halo_pct))
    for i in range(num_layers):
        perimeter_mask = erosion_history.pop()
        alpha = int(round(255 * ((num_layers - i) / (num_layers + 1.0)) ** s.fade_factor))
        for pixel in perimeter_mask:
            neighbors = get_neighbors(s, img, pixel, False)
            if neighbors:
                neighbor_median_color = tuple(np.median(np.array([img[p][:3] for p in neighbors],
                                                                 dtype=np.float32), axis=0).astype(np.uint8))
            else:
                neighbor_median_color = tuple(img[pixel][:3])
            img[pixel] = (*neighbor_median_color, alpha)
        if s.print_debug: print(f"Aliased layer {i+1}/{num_layers} with alpha {alpha}")
    return img

def erode(img: np.ndarray, mask: set, s: settings) -> set:
    ''' Helper function to find the next concentric inset of the given mask.'''
    inset = set()
    for pixel in mask:
        neighbors = get_neighbors(s, img, pixel, False)
        for neighbor in neighbors:
            if neighbor not in mask:
                inset.add(neighbor)
    return inset

def erosion(s: settings, img: np.ndarray, inset_mask: set, bg_mask: set, erosion_history: list[set]) -> Tuple[np.ndarray, set]:
    ''' Recursively erodes the perimeter of the foreground until the next inset is similar enough.
        Then reapplies aliasing to the eroded edges if specified. '''
    is_color = img.ndim == 3
    
    while True:
        bg_mask.update(inset_mask)
        next_inset_mask = erode(img, bg_mask, s)
        erosion_history.append(next_inset_mask)
        if not next_inset_mask:
            for pixel in inset_mask:
                img[pixel] = (0, 0, 0, 0)
            # Return the image and the updated bg_mask when erosion is complete
            return img, bg_mask

        if is_color:
            inset_pixels = np.array([img[p][:3] for p in inset_mask], dtype=np.float32)
            next_inset_pixels = np.array([img[p][:3] for p in next_inset_mask], dtype=np.float32)
        else: # Grayscale
            inset_pixels = np.array([img[p] for p in inset_mask], dtype=np.float32)
            next_inset_pixels = np.array([img[p] for p in next_inset_mask], dtype=np.float32)

        inset_median_color      = np.median(inset_pixels, axis=0)
        next_inset_median_color = np.median(next_inset_pixels, axis=0)

        if s.print_debug: print(f"Inset median color: {inset_median_color}, Next inset median color: {next_inset_median_color}")

        # keep your current outside-band test (no logic change)
        if (np.all(next_inset_median_color >= inset_median_color * (1 + s.erosion_tol)) or
            np.all(next_inset_median_color <= inset_median_color * (1 - s.erosion_tol))) and not np.all(next_inset_median_color == (0, 0, 0)):
            if s.print_debug: print("Eroding further...")
            for pixel in inset_mask:
                img[pixel] = (0, 0, 0, 0)
            return erosion(s, img, next_inset_mask, bg_mask, erosion_history)
        else:
            if s.print_debug: print("Erosion complete, aliasing edges.")
            if s.aliasing: return alias(s, img, erosion_history), bg_mask
            else: return img, bg_mask

def crop(s: settings, img: np.ndarray, mask: set) -> Tuple[np.ndarray, set]:
    ''' Given an image, automatically crops to the area of interest based on lightness channel.
        Crop is set when the average lightness of row/column diverges and reconverges from the max lightness by tol.
        Returns the cropped image and the repositioned mask.'''
    srcL = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2HLS)[:, :, 1]

    left, right = measure_image(srcL, s.crop_tol, behavior="stat comparison, max > average, by col, divergence, last, convergence")
    top, bottom = measure_image(srcL, s.crop_tol, behavior="stat comparison, max > average, by row, divergence, last, convergence")

    # If slack is true, add a 1 pixel border around the crop
    if s.crop_slack:
        left -= s.crop_slack;  right += s.crop_slack
        top  -= s.crop_slack; bottom += s.crop_slack

    # Remove pixels outside crop from mask
    mask.difference_update((r, c) for r in range(img.shape[0]) for c in range(img.shape[1]) if r < top or r > bottom or c < left or c > right)

    # Shift mask coordinates to new origin
    cropped_mask = set((r - top, c - left) for (r, c) in mask)
    
    return img[top:bottom+1, left:right+1], cropped_mask

# def convolution_blur(s: settings, img: np.ndarray) -> np.ndarray:
