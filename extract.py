#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

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
    
    img_paths = list(level_path.glob("*.tif"))
    for i, img_path in enumerate(img_paths):
        extract(img_path)        
