#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet
from bdtools.models.annotate import Annotate

#%% Inputs(general) -----------------------------------------------------------

# Procedure
annotate = 0
train = 1

# Parameters
df = 16
mask_type = "vesicles"

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")
train_path = Path(Path.cwd() / "data" / f"train_level-{df}")

#%% Inputs(model) -------------------------------------------------------------

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 0
load_name = ""

# preprocess
patch_size = 250
patch_overlap = 125
img_norm = "none"
msk_type = "normal"

# augment
iterations = 2000
invert_p = 0.0
gamma_p = 0
gblur_p = 0
noise_p = 0 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 16
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.001
patience = 20

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    if annotate:
        
        Annotate(train_path)
        
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(train_path.rglob("*.tif")):
            if f"_mask_{mask_type}" in path.name:
                if Path(str(path).replace(f"_mask_{mask_type}", "")).exists():
                    msks.append(io.imread(path))   
                    imgs.append(io.imread(str(path).replace(f"_mask_{mask_type}", "")))
        imgs = np.stack(imgs)
        msks = np.stack(msks)
                        
        # Manual normalization
        imgs = imgs.astype("float32")
        imgs = norm_pct(imgs, pct_low=1, pct_high=99, mask=imgs > 0)

        unet = UNet(
            save_name="",
            load_name=load_name,
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        # Train
        unet.train(
            
            imgs, msks, 
            X_val=None, y_val=None,
            preview=preview,
            
            # Preprocess
            img_norm=img_norm, 
            msk_type=msk_type, 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscale_factor, 
            
            # Augment
            iterations=iterations,
            invert_p=invert_p,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )