#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

#%% Inputs --------------------------------------------------------------------

# Paths
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
data_path = Path("D:\local_Mayrhofer\data")
img_name = "Ins1e_wt_1.7nm_00"

# Parameters
df = 4

#%% Function(s) ---------------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    out_paths = data_path / img_name / f"level-{df}"
    img_paths = list(out_paths.glob("*.tif"))
    
    # Load images
    imgs, stms = [], []
    for img_path in img_paths:
        imgs.append(io.imread(img_path))
        stms.append(img_path.stem)
    
    # # Display
    # imgs = np.stack(imgs)
    # vwr = napari.Viewer()
    # vwr.add_image(imgs)
    
#%% Development ---------------------------------------------------------------

    # # -------------------------------------------------------------------------
    
    # # Take right strip of img0 and left strip of img1
    # strip0 = img0[:, -max_search:]
    # strip1 = img1[:, :max_search]

    # # Average along y (height) to make 1D profiles
    # prof0 = strip0.mean(axis=0)
    # prof1 = strip1.mean(axis=0)

    # # Cross-correlate (same mode)
    # corr = correlate(prof0 - prof0.mean(), prof1 - prof1.mean(), mode='full')
    # shift = np.argmax(corr) - (len(prof1) - 1)

    # # Convert shift to overlap in pixels
    # overlap = max_search - shift
    # overlap = np.clip(overlap, 1, max_search-1)

    # # -------------------------------------------------------------------------  
        
    # img0 = imgs[1]
    # img1 = imgs[2]
    # overlap = 100
    
    # # Get image shape
    # nY, nX = img0.shape
    
    # # Non-overlapping regions
    # left_part = img0[:, :-overlap]
    # right_part = img1[:, overlap:]
    
    # # Overlapping regions
    # left_overlap = img0[:, -overlap:]
    # right_overlap = img1[:, :overlap]
    
    # # Blend overlaps
    # blend_overlap = 0.5 * (left_overlap + right_overlap)
    
    # # Stich
    # # stitched = np.concatenate((left_part, blend_overlap, right_part), axis=1)
    # stitched = np.concatenate((left_part, left_overlap, right_part), axis=1)
    
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(np.concatenate((img0, img1), axis=1))
    # vwr = napari.Viewer()
    # vwr.add_image(stitched)
    
#%% Development ---------------------------------------------------------------

    # Imports
    import time
    from skimage.registration import phase_cross_correlation

    # Parameters
    wsearch = 200
    idxs = [0, 1, 2, 3, 4, 5, 6]
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("get shifts :", end=" ", flush=True)
    
    shifts = []
    for idx in idxs[1:]:
        
        # Extract data
        img0 = imgs[idx - 1]
        img1 = imgs[idx]
        crp0 = img0[:, -wsearch:]
        crp1 = img1[:, :wsearch]
        
        # Get shift
        shift, error, diffphase = phase_cross_correlation(
            crp0, crp1, upsample_factor=10, overlap_ratio=0.3)
        shifts.append((int(round(shift[0])), int(round(shift[1]))))
        
    # Replace outliers
    shifts_y = [s[0] for s in shifts]
    shifts_x = [s[1] for s in shifts]
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s\n")
    