#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from pathlib import Path

# functions
from functions import load_images, get_shift, stich

# bdtools
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 16
suffixes = ["cells", "nuclei", "vesicles"]

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")

#%% Function(s) ---------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Setup prd directory
    if df == 1:
        level_path = data_path
    else:
        level_path = data_path / f"level-{df}"
    prd_path = level_path / "prds"
    if prd_path.exists():
        for item in prd_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
    else:
        prd_path.mkdir(parents=True, exist_ok=True)
    
    # Get shift
    mtds = get_shift(data_path, df=16)
        
#%% Pre_stiching predictions --------------------------------------------------

    # Load images
    imgs, _ = load_images(data_path, df=df)
        
    # Predict
    suffix = "nuclei"
    unet = UNet(load_name=list(Path.cwd().glob(f"model-{suffix}*"))[0])
    prds = unet.predict(np.stack(imgs), verbose=1)
    prds = (prds * 255).astype("uint8")
            
    # Stich
    imgs_s = stich(imgs, mtds)
    prds_s = stich(prds, mtds)
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(
        imgs_s, gamma=2, opacity=0.5)
    vwr.add_image(
        prds_s, blending="additive", colormap="bop orange", opacity=0.33)    
