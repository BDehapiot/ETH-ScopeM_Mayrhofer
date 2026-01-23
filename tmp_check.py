#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

#%% Function(s) ---------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    
    # Parameters
    view = 0
    pix_ref = 27.2
    dataset = "ins1e_wt_3.25nm_00"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    
    # Load
    prp =  io.imread(Path(data_path, "prp", f"prp_{view:02d}.tif"))
    mskv = io.imread(Path(data_path, "out", f"msk_vesicles_hc_{view:02d}.tif"))
    mskj = io.imread(Path(data_path, "out", f"msk_junctions_hc_{view:02d}.tif"))
    mskl = io.imread(Path(data_path, "out", f"msk_labels_hc_{view:02d}.tif"))
        
    # -------------------------------------------------------------------------
    
    # Display
    vwr = napari.Viewer()
    vwr.add_labels(mskl, blending="additive", opacity=0.25, visible=1)
    vwr.add_image(mskv * 255, blending="additive", opacity=1.0, visible=1)
    vwr.add_image(prp, blending="additive", opacity=0.75, visible=1)
    # vwr.add_image(msk1b, blending="additive", colormap="magenta", visible=1)
