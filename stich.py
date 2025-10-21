#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct

# numpy
from numpy.fft import fft2, ifft2, fftshift

# skimage
from skimage.filters import sobel

#%% Inputs --------------------------------------------------------------------

# Paths
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data")
data_path = Path("D:\local_Mayrhofer\data")
img_name = "Ins1e_wt_1.7nm_00"

# Parameters
df = 16

#%% Function(s) ---------------------------------------------------------------

def subtract_background(imgs):
    imgs = np.stack(imgs)
    med = np.median(imgs, axis=0)
    subs = imgs / med
    return list(subs)

def preprocess_images(img):
    # img = norm_pct(img)
    img = sobel(img)
    return img
    
def register_translation(img0, img1):
    
    # Cross power spectrum
    f0 = fft2(img0)
    f1 = fft2(img1)
    cross_power = f0 * f1.conj()
    cross_power /= np.abs(cross_power)
    
    # Correlation
    corr = fftshift(ifft2(cross_power))
    corr = np.abs(corr)
    
    # Shifts
    max_idx = np.unravel_index(np.argmax(corr), corr.shape)
    shift = np.array(max_idx) - np.array(corr.shape) // 2
    dy = int(shift[0])
    dx = int(shift[1])       
    
    # Metrics
    pval = corr.max()
    pnrm = pval / np.sum(corr)
    
    return dy, dx, pval, pnrm

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    out_paths = data_path / img_name / f"level-{df}"
    img_paths = list(out_paths.glob("*.tif"))
    
    # Load images
    imgs, mtds = [], []
    for img_path in img_paths:
        img = io.imread(img_path)
        stm = img_path.stem
        row = int(stm[5:8])
        col = int(stm[9:12])
        nY, nX = img.shape
        imgs.append(img)
        mtds.append({
            
            "stm" : stm,
            "row" : row, 
            "col" : col,
            "nY"  : nY, 
            "nX"  : nX, 
            
            # Tile coordinates (considering padding)
            "y0"  : row * nY + nX,
            "x0"  : col * nX + nX, 
            "y1"  : row * nY + nY * 2,
            "x1"  : col * nX + nX * 2,  
            
            })
        
    # Subtract background
    subs = subtract_background(imgs)
    
    # Preprocess images
    prps = Parallel(n_jobs=-1)(
        delayed(preprocess_images)(sub)
        for sub in subs
        )
    prps = norm_pct(prps)
    
    # Get mosaic shape
    max_row = np.max([m["row"] for m in mtds])
    max_col = np.max([m["col"] for m in mtds])
    
    # # Stich images (no registration)
    # stiched = np.zeros(((max_row + 1) * nY, (max_col + 1) * nX))
    # stiched = np.pad(stiched, ((nY, nY), (nX, nX)))
    # for prp, mtd in zip(prps, mtds):
    #     y0, y1 = mtd["y0"], mtd["y1"]
    #     x0, x1 = mtd["x0"], mtd["x1"]
    #     stiched[y0:y1, x0:x1] = prp
    
    # # Display
    # imgs = np.stack(imgs)
    # subs = np.stack(subs)
    # vwr = napari.Viewer()
    # # vwr.add_image(subs)
    # vwr.add_image(stiched, colormap="turbo")
    
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

    # # Imports
    # from skimage.registration import phase_cross_correlation

    # # Inputs 
    # xsearch = 256
    
    # t0 = time.time()
    # print("get shifts :", end=" ", flush=True)
    
    # yshifts = np.zeros((max_row, max_col), dtype=int)
    # xshifts = np.zeros((max_row, max_col), dtype=int)
    # errors = np.zeros((max_row, max_col), dtype=float)
    # for i in range(1, len(imgs)):
    #     for r in range(max_row):
    #         if mtds[i]["row"] == r:
                
    #             # Load images
    #             prp0 = prps[i - 1]
    #             prp1 = prps[i]
                        
    #             # Crop images
    #             crp0 = prp0[:, -xsearch:]
    #             crp1 = prp1[:, :xsearch]
                
    #             # Get shifts
    #             shift, error, diffphase = phase_cross_correlation(
    #                 crp0, crp1, upsample_factor=1)
    #             yshifts[mtds[i]["row"], mtds[i]["col"]] = int(round(shift[0]))
    #             xshifts[mtds[i]["row"], mtds[i]["col"]] = int(round(shift[1]))
    #             errors[mtds[i]["row"], mtds[i]["col"]] = error

    #     # Get shifts
        
    # #     # Get shifts
    # #     shift, error, diffphase = phase_cross_correlation(
    # #         crp0, crp1, upsample_factor=1)
    # #     shifts.append((int(round(shift[0])), int(round(shift[1]))))
            
    # t1 = time.time()
    # print(f"{t1 - t0:.3f}s\n")
    
    # # Display
    # imgs = np.stack(imgs)
    # vwr = napari.Viewer()
    # vwr.add_image(yshifts)
    # vwr.add_image(xshifts)
    # vwr.add_image(errors)
    
#%% Development ---------------------------------------------------------------
    
    t0 = time.time()
    print("get shifts :", end=" ", flush=True)

    nans = np.full((max_row, max_col), np.nan, dtype=float)
    ydys = nans.copy(); ydxs = nans.copy(); ypnrms = nans.copy()
    xdys = nans.copy(); xdxs = nans.copy(); xpnrms = nans.copy()   
    for i in range(1, len(imgs)):
        
        # Get y shifts
        for c in range(max_col):
            if mtds[i]["col"] == c:
                dy, dx, pval, pnrm = register_translation(prps[i - 1], prps[i])
                ydys[mtds[i]["row"], mtds[i]["col"]] = dy
                ydxs[mtds[i]["row"], mtds[i]["col"]] = dx
                ypnrms[mtds[i]["row"], mtds[i]["col"]] = pnrm
        
        # Get x shifts
        for r in range(max_row):
            if mtds[i]["row"] == r:
                dy, dx, pval, pnrm = register_translation(prps[i - 1], prps[i])
                xdys[mtds[i]["row"], mtds[i]["col"]] = dy
                xdxs[mtds[i]["row"], mtds[i]["col"]] = dx
                xpnrms[mtds[i]["row"], mtds[i]["col"]] = pnrm
            
    t1 = time.time()
    print(f"{t1 - t0:.3f}s\n")
    
    # Get modal shifts
    def get_mode(arr):
        values, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
        return int(values[np.argmax(counts)])
    ydym = get_mode(ydys)
    ydxm = get_mode(ydxs)
    xdym = get_mode(xdys)
    xdxm = get_mode(xdxs)

    # # Display
    # imgs = np.stack(imgs)
    # vwr = napari.Viewer()
    # vwr.add_image(dys)
    # vwr.add_image(dxs)
    # vwr.add_image(pnrms)
    
#%% Development ---------------------------------------------------------------

    # for i in range(1, len(imgs)):
    #     for r in range(max_row):
    #         if mtds[i]["row"] == r:
                


#%% Development ---------------------------------------------------------------

    # # Initialize
    # stiched_reg = np.zeros(((max_row + 1) * nY, (max_col + 1) * nX))
    # stiched_reg = np.pad(stiched_reg, ((nY, nY), (nX, nX)))
    
    # # Place seed
    # y0, y1 = mtds[0]["y0"], mtds[0]["y1"]
    # x0, x1 = mtds[0]["x0"], mtds[0]["x1"]
    # stiched_reg[y0:y1, x0:x1] = subs[0]
    
    # for i in range(len(prps) - 1):
    #     if i <= 3:
            
    #         # Fetch data
    #         sub0 = subs[i]
    #         sub1 = subs[i + 1]    
    #         prp0 = prps[i]
    #         prp1 = prps[i + 1] 
    #         y0, y1 = mtds[i + 1]["y0"], mtds[i + 1]["y1"]
    #         x0, x1 = mtds[i + 1]["x0"], mtds[i + 1]["x1"]
            
    #         # Get shift
    #         dy, dx, peak_val, peak_nrm = register_translation(prp0, prp1)
    #         print(f"{dy:+4d}", f"{dx:+4d}", f"{peak_val:.2e}", f"{peak_nrm:.2e}")
            
    #         # Add image
    #         if peak_nrm > 2e-04:
    #             stiched_reg[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = sub1
    #         else:
    #             stiched_reg[y0:y1, x0:x1] = sub1

    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(stiched_reg, colormap="gray", contrast_limits=[0.8, 1.2])



