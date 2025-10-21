#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
import xarray as xr
from skimage import io
from pathlib import Path
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

def normalize_background(imgs):
    imgs = np.stack(imgs)
    med = np.median(imgs, axis=0)
    subs = imgs / med
    return list(subs)

def preprocess_images(img):
    img = sobel(img)
    return img
    
def get_shift(img0, img1):
    
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
    dy, dx = int(shift[0]), int(shift[1]) 
    
    # Score
    scr = corr.max() / np.sum(corr)
    
    return dy, dx, scr

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%% Development ---------------------------------------------------------------

    # Paths
    out_paths = data_path / img_name / f"level-{df}"
    img_paths = list(out_paths.glob("*.tif"))
    
    # Fetch md
    imgs, mtds = [], []
    
    for img_path in img_paths:
        
        # Fetch
        img = io.imread(img_path)
        stm = img_path.stem
        nY, nX = img.shape
        row = int(stm[5:8])
        col = int(stm[9:12])
        y0 = row * nY
        y1 = row * nY + nY
        x0 = col * nX
        x1 = col * nX + nX

        # Append
        imgs.append(img)
        mtds.append({
            "stm" : stm, 
            "row" : row, "col" : col,
            "nY"  : nY,  "nX"  : nX, 
            "y0"  : y0,  "y1"  : y1,
            "x0"  : x0,  "x1"  : x1,  
            })

    nR = np.max([m["row"] for m in mtds])
    nC = np.max([m["col"] for m in mtds])
    
    # Normalize background
    nrms = normalize_background(imgs)
    
    # Preprocess images
    prps = Parallel(n_jobs=-1)(
        delayed(preprocess_images)(nrm)
        for nrm in nrms
        )
    prps = norm_pct(prps)
    
    # Get 2D arrays
    tmp = np.empty((nR + 1, nC + 1), dtype=object)
    mtds_2D = tmp.copy()
    imgs_2D = tmp.copy()
    nrms_2D = tmp.copy()
    prps_2D = tmp.copy()
    
    for mtd, img, nrm, prp in zip(mtds, imgs, nrms, prps):
        r, c = mtd["row"], mtd["col"]
        mtds_2D[r, c] = mtd
        imgs_2D[r, c] = img
        nrms_2D[r, c] = nrm
        prps_2D[r, c] = prp
        
#%% Development ---------------------------------------------------------------

    t0 = time.time()
    print("get shifts :", end=" ", flush=True)

    tmp = np.full((nR, nC), np.nan, dtype=float)
    dyYs = tmp.copy(); dxYs = tmp.copy(); scrY = tmp.copy()
    dyXs = tmp.copy(); dxXs = tmp.copy(); scrX = tmp.copy()   
    for i in range(1, len(imgs)):
        
        # Get y shifts
        for c in range(nC):
            if mtds[i]["col"] == c:
                dy, dx, scr = get_shift(prps[i - 1], prps[i])
                dyYs[mtds[i]["row"], mtds[i]["col"]] = dy
                dxYs[mtds[i]["row"], mtds[i]["col"]] = dx
                scrY[mtds[i]["row"], mtds[i]["col"]] = scr
        
        # Get x shifts
        for r in range(nR):
            if mtds[i]["row"] == r:
                dy, dx, scr = get_shift(prps[i - 1], prps[i])
                dyXs[mtds[i]["row"], mtds[i]["col"]] = dy
                dxXs[mtds[i]["row"], mtds[i]["col"]] = dx
                scrX[mtds[i]["row"], mtds[i]["col"]] = scr
            
    t1 = time.time()
    print(f"{t1 - t0:.3f}s\n")
        
#%% Development ---------------------------------------------------------------

    # for r in range(nR):
    #     mtdR = mtds_2D[r, :]
    #     prpR = prps_2D[r, :]
    #     for c in range(nC):
    #         if prpR[c] is not None:
                