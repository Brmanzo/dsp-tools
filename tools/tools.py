import sys
import numpy as np
from PIL import Image, ImageSequence
from tools.IO import get_args, output_img, output_gif
from tools.methods import sample_bg, convolution
from tools.components import flood_fill, crop, erode, erosion
from tools.data_structures import settings

def remove_bg(s: settings, img: np.ndarray) -> np.ndarray:
    ''' Given an RGBA image, removes the background by:
        1. Sampling the border pixels to determine the background color.
        2. Flood filling from the corner pixel to remove similar pixels.
        3. Optionally cropping to the area of interest based on lightness channel.
        4. Recursively eroding the edges until the next inset is similar enough.
        5. Optionally reapplying aliasing to the eroded edges.

        Lakes specifies whether or not to search whole image for color.
        Square_kernel specifies whether to use 8-connectivity or 4-connectivity.
        Crop specifies whether to auto-crop the image after flood fill.
        Aliasing specifies whether to reapply aliasing after erosion.'''
    transparent = (0, 0, 0, 0)
    # sample background color and flood from corner
    bg_color = sample_bg(img)
    bg_removed, mask = flood_fill(s, img, bg_color, (0, 0), transparent)
    # crop where the average max lightness diverges from the average
    if s.crop:
        img, mask = crop(s, bg_removed, mask)
    erosion_history = []
    # erode away aliased edges
    inset_mask = erode(img, mask, s)
    erosion_history.append(inset_mask)
    eroded_img, eroded_mask = erosion(s, img, inset_mask, mask, erosion_history)

    # final crop to remove any transparent border pixels
    if s.crop:
        eroded_img = crop(s, eroded_img, eroded_mask)[0]

    return eroded_img

def remove_bg_img(argv: list, img: np.ndarray) -> None:
    ''' Removes background from a single image file using settings from argv.'''
    s = get_args(argv)
    img_np = remove_bg(s, img)
    output_img(img_np, "output.png")

def remove_bg_gif(argv: list, gif: Image.Image) -> None:
    '''Leverages remove_bg to process each frame of a GIF and preserve animation.'''
    s = get_args(argv)
    s.crop = False  # Disable cropping for GIFs to maintain frame dimensions
    s.print_debug = False  # Disable debug output for GIF processing
    
    frames = []
    gif_frames = list(ImageSequence.Iterator(gif))
    frame_count = len(gif_frames)

    print(f"Processing GIF with {frame_count} frames...")
    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        img = remove_bg(s, np.array(frame.convert("RGBA")))
        frames.append(Image.fromarray(img, mode="RGBA"))
        print(f"Processed frame {i+1}/{frame_count}")

    output_gif(frames, "output.gif", gif.info.get('duration', 100))

def blur_img(argv: list, img: np.ndarray):
    ''' Applies a Gaussian blur to the image.'''
    s = get_args(argv)
    blurred_img = convolution(s, img)
    output_img(blurred_img, "output_blur.png")

valid_file_extensions = (".png", ".jpg", ".jpeg", ".webp")
def get_function(argv: list):
    '''Determines the function to execute based on input arguments.'''
    if "--remove_bg" in argv:
        if argv[2].endswith(valid_file_extensions):
            remove_bg_img(sys.argv, np.array(Image.open(argv[2]).convert("RGBA")))
        elif argv[2].endswith(".gif"):
            remove_bg_gif(sys.argv, Image.open(argv[2]))
    elif "--blur" in argv:
        if argv[2].endswith(valid_file_extensions):
            blur_img(sys.argv, np.array(Image.open(argv[2]).convert("RGBA")))
    else:
        print("No valid function specified.")
        return
    
