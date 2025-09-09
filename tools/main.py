#!/usr/bin/env python3
# DSP-tools/main.py
import os
import numpy as np
from PIL import Image, ImageSequence
from pathlib import Path
from tools.tools import remove_bg
from tools.preprocessing import debug_oscilloscope
from tools.data_structures import settings
import sys

def get_function(argv: list):
    '''Determines the function to execute based on input arguments.'''
    if "--remove_bg" in argv:
        if argv[2].endswith((".png")):
            remove_bg_img(sys.argv, np.array(Image.open(argv[2]).convert("RGBA")))
        elif argv[2].endswith((".gif")):
            remove_bg_gif(sys.argv, Image.open(argv[2]))
    else:
        print("No valid function specified.")
        return
    
def get_token(argv: list, prefix: str, default: str) -> str:
    '''Searches input arguments for token and returns value or default.'''
    for arg in argv:
        if arg.startswith(prefix):
            parts = arg.split("=", 1)
            if len(parts) == 2 and parts[1]:
                return parts[1]
            else:
                print(f"Invalid {prefix} value, using default {default}")
                return default
    return default

def get_args(argv: list) -> settings:
    ''' Parses command line arguments into settings object.'''
    s = settings()
    s.lakes         = "--lakes" in argv
    s.aliasing      = "--no_alias" not in argv
    s.square_kernel = "--square_kernel" in argv
    s.crop          = "--no_crop" not in argv

    s.flood_tol     = float(get_token(argv, "--flood_tol=", "0.1"))
    s.erosion_tol   = float(get_token(argv, "--erosion_tol=", "0.1"))
    s.crop_slack    =   int(get_token(argv, "--crop_slack=", "1"))
    s.crop_tol      = float(get_token(argv, "--crop_tol=", "0.1"))
    s.fade_factor   = float(get_token(argv, "--fade_factor=", "1.5"))
    s.halo_pct      = float(get_token(argv, "--halo_pct=", "0.6"))
    s.blob_th       = float(get_token(argv, "--blob_th=", "0.05"))
    return s

def debug_img(img: np.ndarray, name: str, axis: str="row") -> None:
    ''' Outputs debug image of lightness channel.'''
    debug_dir = Path.cwd() / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_oscilloscope(img, name, debug_dir, axis)

def output_img(img: np.ndarray, name: str) -> None:
    ''' Outputs the processed image to the current working directory.'''
    output_path = os.path.join(os.getcwd(), name)
    Image.fromarray(img, mode="RGBA").save(output_path)
    print(f"Processed image saved to {output_path}")

def output_gif(frames: list[Image.Image], name: str, duration: int) -> None:
    ''' Outputs the processed GIF to the current working directory.'''
    output_path = os.path.join(os.getcwd(), name)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration,
        disposal=2
    )
    print(f"Processed GIF saved to {output_path}")

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

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        return
    get_function(sys.argv)

if __name__ == "__main__":
    main()
