#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Path
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
data_path = Path("D:\local_Mayrhofer\data")
img_name = "Ins1e_wt_1.7nm_00"

# Parameters
suffix = "vesicles"
load_name = f"model-{suffix}_1024_normal_256-31_1"

#%% Function(s) ---------------------------------------------------------------

def predict(path):
    img = io.imread(path)
    prd = (unet.predict(img, verbose=1) * 255).astype("uint8")
    save_path = data_path / (path.stem + f"_prd-{suffix}.tif")
    io.imsave(save_path, prd, check_contrast=False)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # load images
    unet = UNet(load_name=load_name)
    img_paths = list((data_path / img_name).glob("*.tif"))   
    for i, img_path in enumerate(img_paths):
        if i < 50:
            predict(img_path)
            
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(img, gamma=2)
    # vwr.add_image(prd, blending="additive", colormap="magma", opacity=0.33)