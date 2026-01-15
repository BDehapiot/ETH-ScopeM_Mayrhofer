#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# skimage
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, remove_small_holes

# scipy
from scipy.ndimage import binary_fill_holes

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    dataset = "gigyf12_dko_1.7nm"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    mskj = io.imread(Path(data_path, "out", "msk_junctions_hc_00.tif"))
    mskl = io.imread(Path(data_path, "out", "msk_labels_hc_00.tif"))
    mskj = mskj.astype(bool)

    msk1 = remove_small_holes(
        mskl > 0, area_threshold=4096, connectivity=2)
    msk2 = find_boundaries(msk1, mode="inner")
    msk3 = binary_dilation(mskj)
    mskm = msk2 & ~msk3
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(mskl, visible=0)
    vwr.add_image(
        mskj, visible=0, 
        colormap="green", blending="additive",
        )
    vwr.add_image(
        mskm, visible=0, 
        colormap="magenta", blending="additive",
        )

    