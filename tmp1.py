#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# skimage
from skimage.measure import label, regionprops

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    

    dataset = "gigyf12_dko_3.25nm_00"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    
    view = 3
    pix_ref = 27.2
    
    mskv = io.imread(Path(data_path, "out", f"msk_vesicles_hc_{view:02d}.tif"))
    mskj = io.imread(Path(data_path, "out", f"msk_junctions_hc_{view:02d}.tif"))
    mskl = io.imread(Path(data_path, "out", f"msk_labels_hc_{view:02d}.tif"))
    
    # Measure cells
    for props in regionprops(label(mskl)):
        if props.label == 39:
            print(props.area * ((pix_ref * 1e-3) ** 2))

    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(mskl, visible=1)
    
    # vwr.add_image(
    #     mskj, visible=0, 
    #     colormap="green", blending="additive",
    #     )
