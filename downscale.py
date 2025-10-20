#%% Imports -------------------------------------------------------------------

import os
import tifffile
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed 

#%% Inputs --------------------------------------------------------------------

# Paths
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
data_path = Path("D:\local_Mayrhofer\data")
img_name = "Ins1e_wt_1.7nm_00"

# Parameters
df = 4

#%% Function(s) ---------------------------------------------------------------

def load_and_downscale(img_path, out_path, df=2):
    
    # Load image
    with tifffile.TiffFile(img_path) as tif:
        img = tif.asarray()
        
    # Downscale image
    new_shape = (img.shape[0] // df, df, img.shape[1] // df, df)
    dsc = img[:new_shape[0] * df, :new_shape[2] * df] \
              .reshape(new_shape).mean(axis=(1,3)).astype(np.uint16)

    # Save downscaled image
    out_file = out_path / f"{img_path.stem}_level-{df}.tif"
    tifffile.imwrite(out_file, dsc, compression=None)
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Setup output directory
    out_path = data_path / img_name / f"level-{df}"
    if out_path.exists():
        for item in out_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
    else:
        out_path.mkdir(parents=True, exist_ok=True)
    
    # Open & rescale images
    img_paths = list((data_path / img_name).glob("*.tif"))
    Parallel(n_jobs=os.cpu_count(), batch_size=4, prefer="threads")(
        delayed(load_and_downscale)(img_path, out_path, df=df)
        for img_path in img_paths
        )