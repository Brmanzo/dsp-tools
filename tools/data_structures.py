#!/usr/bin/env python3
# DSP-tools/main.py
from typing import Tuple

class settings():
    def __init__(self):
        # Program
        self.print_debug   = True
        # Flood Settings
        self.flood_tol     = 0.1
        self.lakes         = False
        self.square_kernel = False
        self.blob_tol       = 0.0005
        # Erosion Settings
        self.erosion_tol   = 0.1
        # Crop Settings
        self.crop          = True
        self.crop_slack    = 1
        self.crop_tol      = 0.1
        # Aliasing Settings
        self.aliasing      = True
        self.fade_factor   = 1.5
        self.halo_pct      = 0.6

        # Convolution Settings
        self.kernel_size    = 3

class RGBA:
    def __init__(self, color_tuple: Tuple[int, int, int, int]):
        if len(color_tuple) != 4:
            raise ValueError("Input must be a tuple of 4 integers.")
        self.r, self.g, self.b, self.a = color_tuple
    
    def __repr__(self):
        """Provides a clean string representation of the object."""
        return f"RGBA({self.r}, {self.g}, {self.b}, {self.a})"
    
    def __iter__(self):
        """Allows the class to be iterated over and unpacked like a tuple."""
        yield self.r; yield self.g; yield self.b; yield self.a

    def __getitem__(self, key: int):
        """Allows index access like my_color[0]."""
        return (self.r, self.g, self.b, self.a)[key]
    
    def grayscale(self) -> int:
        """Returns the perceived lightness (Luma) of the color."""
        return int(0.299 * self.r + 0.587 * self.g + 0.114 * self.b)