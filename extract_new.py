#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

from functions import load_images

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 16
patch_size = 4000 // df

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")
if df == 1:
    level_path = data_path
else:
    level_path = data_path / f"level-{df}"
train_path = Path(Path.cwd() / "data" / f"train_level-{df}")

#%% Function(s) ---------------------------------------------------------------

def extract(path):
    
    # Load image
    img = io.imread(img_path)
    
    # Save patch
    save_path = train_path / path.name
    io.imsave(save_path, img, check_contrast=False)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    imgs, mtds = load_images(data_path, df=df, return_metadata=True)
    for img, mtd in zip(img_paths):
        extract(img_path)        
