#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

from skimage.measure import label, regionprops

#%% Function(s) ---------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    
    # Parameters
    view = 2
    pix_ref = 27.2
    dataset = "ins1e_wt_3.25nm_01"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    
    # Load
    prp =  io.imread(Path(data_path, "prp", f"prp_{view:02d}.tif"))
    mskv = io.imread(Path(data_path, "out", f"msk_vesicles_hc_{view:02d}.tif"))
    mskj = io.imread(Path(data_path, "out", f"msk_junctions_hc_{view:02d}.tif"))
    mskl = io.imread(Path(data_path, "out", f"msk_labels_hc_{view:02d}.tif"))
    mskv = mskv * 255
    
    # -------------------------------------------------------------------------
    
    print(np.max(mskl))
    mskll = label(mskl > 0, connectivity=1)
    print(np.max(mskll))
    
    dfl = []
    for props in regionprops(mskl):        
        dfl.append({
            "label" : props.label,
            "area"  : props.area,
            })
    dfl = pd.DataFrame(dfl)
    
    dfll = []
    for props in regionprops(mskll):        
        dfll.append({
            "label" : props.label,
            "area"  : props.area,
            })
    dfll = pd.DataFrame(dfll)
            
    # -------------------------------------------------------------------------
    
    # Display
    # vwr = napari.Viewer()
    # vwr.add_labels(mskl, blending="additive", opacity=0.25, visible=1)
    # vwr.add_image(mskv, blending="additive", opacity=1.0, visible=1)
    # vwr.add_image(prp, blending="additive", opacity=0.75, visible=1)