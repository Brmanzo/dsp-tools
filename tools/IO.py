import os
import numpy as np
from PIL import Image
from pathlib import Path
from tools.preprocessing import debug_oscilloscope
from tools.data_structures import settings

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
    s.blob_tol      = float(get_token(argv, "--blob_tol=", "0.05"))
    s.kernel_size   =   int(get_token(argv, "--kernel_size=", "3"))
    return s

def debug_img(img: np.ndarray, name: str, axis: str="row") -> None:
    ''' Outputs debug image of lightness channel.'''
    debug_dir = Path.cwd() / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_oscilloscope(img, name, debug_dir, axis)

def output_img(img: np.ndarray, name: str, mode: str) -> None:
    ''' Outputs the processed image to the current working directory.'''
    output_path = os.path.join(os.getcwd(), name)
    Image.fromarray(img, mode=mode).save(output_path)
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