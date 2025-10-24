#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# functions
from functions import load_images, get_shift, stich

# bdtools
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 16
patch_size = 4000 // df
target = "stiched"

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")
if df == 1:
    level_path = data_path
else:
    level_path = data_path / f"level-{df}"

#%% Function(s) ---------------------------------------------------------------

def extract_from_tiles(imgs, mtds, df=16):

    # Save patches
    for img, mtd in zip(imgs, mtds):
        save_path = train_path / (mtd["stm"] + ".tif")
        io.imsave(save_path, img, check_contrast=False)
    
def extract_from_stiched(imgs, mtds, df=16):
    
    # Stich images
    mtds = get_shift(imgs, mtds)
    imgs_s = stich(imgs, mtds)
    
    # Extract patches
    patches = extract_patches(imgs_s, patch_size, 0)
    
    # Save patches
    for i, patch in enumerate(patches):
        save_path = train_path / (f"patch-{i:04d}" + ".tif")
        io.imsave(save_path, patch.astype("uint16"), check_contrast=False)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Setup train directory
    train_path = Path(Path.cwd() / "data" / f"train_{target}_level-{df}")
    if not train_path.exists():
        train_path.mkdir(parents=True, exist_ok=True)
        
    # Load images
    imgs, mtds = load_images(data_path, df=df, return_metadata=True)
    
    # Extract
    if target == "tiles":
        extract_from_tiles(imgs, mtds, df=df)
    elif target == "stiched":
        extract_from_stiched(imgs, mtds, df=df) 
