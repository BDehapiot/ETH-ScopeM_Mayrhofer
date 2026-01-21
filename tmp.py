#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# skimage
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, remove_small_holes

# scipy
from scipy.ndimage import distance_transform_edt

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    dataset = "gigyf12_dko_1.7nm"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    mskv = io.imread(Path(data_path, "out", "msk_vesicles_hc_00.tif"))
    mskj = io.imread(Path(data_path, "out", "msk_junctions_hc_00.tif"))
    mskl = io.imread(Path(data_path, "out", "msk_labels_hc_00.tif"))
    mskj = mskj.astype(bool)

    msk1 = remove_small_holes(
        mskl > 0, area_threshold=4096, connectivity=2)
    msk2 = find_boundaries(msk1, mode="inner")
    msk3 = binary_dilation(mskj)
    mskm = msk2 & ~msk3
    mska = mskj | mskm
    
    edtj = distance_transform_edt(mskj == 0)
    edtm = distance_transform_edt(mskm == 0)
    edta = distance_transform_edt(mska == 0)
    
    for props in regionprops(label(mskl), intensity_image=msk3):
        if props.intensity_max == 0:
            coords = props.coords
            edtj[tuple(coords.T)] = np.nan
    
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
    vwr.add_image(
        mska, visible=0, 
        colormap="gray", blending="additive",
        )

    vwr.add_image(
        edtj, visible=0, 
        blending="additive",
        )
    vwr.add_image(
        edtm, visible=0, 
        blending="additive",
        )
    vwr.add_image(
        edta, visible=0, 
        blending="additive",
        )


    