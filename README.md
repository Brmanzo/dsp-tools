## DSP Tools

Welcome to **DSP Tools**! This project is a collection of image processing utilities, I created to practice programming algorithms and app development.

### Features

- **Background Removal** for PNGs and GIFs using:
    - Flood fill
    - Blob fill
    - Erosion
    - Aliasing/feathering
    - Cropping
- **Spatial Filtering**
    - Blurring
    - Sharpening
    - Edge Detection
- Extended functionality from `preprocessing.py` (imported from my [clash_star_tracker](https://github.com/Brmanzo/clash_star_tracker) project) to explore additional image processing applications.

### Goals

I plan to add more useful features and tools in the future. Contributions and suggestions are welcome!

### Future Ideas
- File conversion
- Pixel art downscaling
- Image slicing/ Grid detection
- FFT-based edge detection
- Preprocessing tools for ML inference
---

### How to use

python -m tools \<tool\> \<input image\> \<args\>

#### Tools:
- remove_bg (.gif supported)
- blur
- sharpen
- edge
- preprocess_text

#### Options:
- lakes (Remove internal white space if true)
- no_alias (disable aliasing)
- square_kernel (use a 3x3 kernel insted of diamond)
- no_crop (To preserve the original image dimensions)

#### Modifiers:
- flood_tol (Difference in lightness to flood over when removing background)
- erosion_tol (Difference in lightness to erode when aliasing)
- crop_slack (How much additional margin in pixels to leave when cropping)
- crop_tol (Difference in lightness to detect background from subject)
- fade_factor (controls how quickly the alpha fades (higher = quicker))
- halo_pct (controls how many of the eroded layers to use for aliasing (0%-100%))
- blob_tol (How large a blob must be to be removed (blob area/image area))
- kernel_size (kernel size, 3 -> 3x3, 5 -> 5x5)
  
