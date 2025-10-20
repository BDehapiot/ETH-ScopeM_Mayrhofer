#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
data_path = Path("D:\local_Mayrhofer\data")
img_name = "Ins1e_wt_1.7nm_00"

# Parameters
df = 4

#%% Function(s) ---------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    out_paths = data_path / img_name / f"level-{df}"
    img_paths = list(out_paths.glob("*.tif"))
    
    # # Load images
    imgs, stms = [], []
    for img_path in img_paths:
        imgs.append(io.imread(img_path))
        stms.append(img_path.stem)
    
    # Display
    imgs = np.stack(imgs)
    vwr = napari.Viewer()
    vwr.add_image(imgs)