#%% Imports -------------------------------------------------------------------

import numpy as np
np.random.seed(42)
from skimage import io
from pathlib import Path

# bdtools
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Paths
data_path  = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
train_path = Path(Path(__file__).resolve().parent, "data", "train")
patch_num  = 20
patch_size = 250

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    prp_img_paths = list(data_path.rglob("prp*.tif"))
    for path in prp_img_paths:
        img = io.imread(path)
        patches = extract_patches(img, patch_size, 0)
        p_idxs = np.random.choice(
            np.arange(0, len(patches)), size=patch_num, replace=False)
        for idx in p_idxs:
            patch = patches[idx]
            save_name = (
                str(path.parent.parent.name) + 
                f"_{path.stem}_patch-{idx:04d}.tif"
                )
            save_path = train_path / save_name
            io.imsave(save_path, patch, check_contrast=False)