#%% Imports -------------------------------------------------------------------

import os
import tifffile
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed 

#%% Inputs --------------------------------------------------------------------

# Parameters
df = 16

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")

#%% Function(s) ---------------------------------------------------------------

def load_and_downscale(img_path, level_path, df=2):
    
    # Load image
    with tifffile.TiffFile(img_path) as tif:
        img = tif.asarray()
        
    # Downscale image
    new_shape = (img.shape[0] // df, df, img.shape[1] // df, df)
    dsc = img[:new_shape[0] * df, :new_shape[2] * df] \
              .reshape(new_shape).mean(axis=(1,3)).astype(np.uint16)

    # Save downscaled image
    out_file = level_path / f"{img_path.stem}_level-{df}.tif"
    tifffile.imwrite(out_file, dsc, compression=None)
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Setup level directory
    level_path = data_path / f"level-{df}"
    if level_path.exists():
        for item in level_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
    else:
        level_path.mkdir(parents=True, exist_ok=True)
    
    # Open & rescale images
    img_paths = list(data_path.glob("*.tif"))
    Parallel(n_jobs=os.cpu_count(), batch_size=4, prefer="threads")(
        delayed(load_and_downscale)(img_path, level_path, df=df)
        for img_path in img_paths
        )