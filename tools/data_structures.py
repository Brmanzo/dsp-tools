#!/usr/bin/env python3
# DSP-tools/main.py

class settings():
    def __init__(self):
        # Program
        self.print_debug   = True
        # Flood Settings
        self.flood_tol     = 0.1
        self.lakes         = False
        self.square_kernel = False
        self.blob_th       = 0.05
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