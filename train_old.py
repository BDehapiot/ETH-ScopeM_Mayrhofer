#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import normalize_images

# bdtools
from bdtools.models.unet import UNet
from bdtools.models.annotate import Annotate

#%% Inputs(general) -----------------------------------------------------------

# Paths
root_path = Path.cwd()
train_path = Path(Path.cwd(), "data", "train")

# Parameters
mask_type = "vesicles"

"The vesicles mask have been mixed with nuclei..."

#%% Inputs(model) -------------------------------------------------------------

# Procedure
procedure = {
    
    "annotate" : 1,
    "train"    : 0,
    
    }

# Build
unet_build = {
    
    "load_name"  : "model_250_normal_5000-940_1",
    "save_name"  : "",
    "root_path"  : root_path,
    "backbone"   : "resnet18",
    "activation" : "sigmoid",
    
    }

# Train
unet_train = {
    
    "preview"            : 0,
    "X_val"              : None,
    "y_val"              : None,
    
    # Preprocess
    "img_norm"           : "none", 
    "msk_type"           : "normal", 
    "patch_size"         : 250,
    "patch_overlap"      : 125,
    "downscaling_factor" : 1, 
    
    # Augment
    "iterations"         : 5000,
    "invert_p"           : 0.0,
    "gamma_p"            : 0.0, 
    "gblur_p"            : 0.0, 
    "noise_p"            : 0.0, 
    "flip_p"             : 0.5, 
    "distord_p"          : 0.5,
    
    # Train
    "epochs"             : 100,
    "batch_size"         : 16,
    "validation_split"   : 0.2,
    "metric"             : "soft_dice_coef",
    "learning_rate"      : 0.001,
    "patience"           : 20,
    
    }

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    if procedure["annotate"]:
        
        Annotate(train_path)
        
    if procedure["train"]:
            
        # Load data
        imgs, msks, imgs_old, msks_old = [], [], [], []
        for msk_path in list(train_path.rglob("*.tif")):
            if f"_mask_{mask_type}" in msk_path.name:
                img_path = Path(str(msk_path).replace(f"_mask_{mask_type}", ""))
                msk = io.imread(msk_path)
                img = io.imread(img_path).astype("float32")
                if msk_path.name.startswith("patch"):
                    msks_old.append(msk)
                    imgs_old.append(img)
                else:
                    msks.append(msk)
                    imgs.append(img)
        imgs_old = np.stack(imgs_old)
        msks_old = np.stack(msks_old)
        imgs = np.stack(imgs)
        msks = np.stack(msks)
        
        # Normalize old
        imgs_old = normalize_images(imgs_old)
        
        # Concatenate
        imgs = np.concatenate([imgs, imgs_old], axis=0)
        msks = np.concatenate([msks, imgs_old], axis=0)
        
        # Normalize all
        imgs /= 255

        # Build
        unet = UNet(**unet_build)
        
        # Train
        unet.train(imgs, msks, **unet_train) 