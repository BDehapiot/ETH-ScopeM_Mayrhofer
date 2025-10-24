#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from pathlib import Path

# functions
from functions import load_images, get_shift, stich

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 16
suffixes = ["cells", "nuclei", "vesicles"]
target = "stiched"

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
            
#%% Predict -------------------------------------------------------------------

    # Load images
    imgs, mtds = load_images(data_path, df=df)  
    
    # Get shift
    mtds = get_shift(imgs, mtds) 
           
    if target == "stiched":
                
        # Manual normalization
        imgs = np.stack(imgs)
        imgs = imgs.astype("float32")
        imgs = norm_pct(imgs, pct_low=1, pct_high=99, mask=imgs > 0)
        
        # Stich 
        imgs_s = stich(imgs, mtds)

        # Predict
        prds = {key:[] for key in suffixes}
        for suffix in suffixes:
            load_name = list(Path.cwd().glob(f"model-{suffix}-{target}*"))[0]
            unet = UNet(load_name=load_name)
            prd = unet.predict(imgs_s, verbose=1)
            prds[suffix] = (prd * 255).astype("uint8")
            
        # Display
        vwr = napari.Viewer()
        vwr.add_image(
            imgs_s, 
            opacity=0.5
            )
        for suffix in suffixes:
            vwr.add_image(
                prds[suffix], name=suffix, colormap="bop orange",
                blending="additive",  opacity=0.33,
                ) 
