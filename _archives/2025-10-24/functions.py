#%% Imports -------------------------------------------------------------------

import time
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_pct

# numpy
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

# skimage
from skimage.filters import sobel

#%% Function : load_images() --------------------------------------------------

def load_images(data_path, df=16, suffix="", return_metadata=False):
    
    # Initialize
    if df == 1:
        level_path = data_path
    else:
        level_path = data_path / f"level-{df}"
    img_paths = list(level_path.glob(f"*level-{df}{suffix}.tif"))
    
    imgs, mtds = [], []
    for i, img_path in enumerate(img_paths):
        
        # Load images
        img = io.imread(img_path)
        imgs.append(img)
        
        # Get metadata
        stm = img_path.stem
        nY, nX = img.shape
        row = int(stm[5:8])
        col = int(stm[9:12])
        y0 = row * nY
        y1 = row * nY + nY
        x0 = col * nX
        x1 = col * nX + nX
        mtds.append({
            "stm" : stm, "idx" : i,
            "row" : row, "col" : col,
            "nY"  : nY,  "nX"  : nX, 
            "y0"  : y0,  "y1"  : y1,
            "x0"  : x0,  "x1"  : x1,  
            })
            
    return imgs, mtds
        
#%% Function : get_shift() ----------------------------------------------------

def get_shift(imgs, mtds):
    
    # Nested function(s) ------------------------------------------------------
    
    def preprocess_image(raw):
        raw = sobel(raw)
        return raw
    
    def _get_shift(raw0, raw1):
        
        # Cross power spectrum
        f0 = fft2(raw0)
        f1 = fft2(raw1)
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
        scr = float(corr.max() / np.sum(corr))

        return dy, dx, scr
    
    # Execute -----------------------------------------------------------------
    
            
    # Get mosaic shape
    nR = np.max([m["row"] for m in mtds])
    nC = np.max([m["col"] for m in mtds])
        
    # Preprocess images
    prps = Parallel(n_jobs=-1)(
        delayed(preprocess_image)(img)
        for img in imgs
        )
    prps = norm_pct(prps)
        
    # Get 2D arrays
    tmp = np.empty((nR + 1, nC + 1), dtype=object)
    mtds_2D = tmp.copy()
    prps_2D = tmp.copy()
    for mtd, img, prp in zip(mtds, imgs, prps):
        r, c = mtd["row"], mtd["col"]
        mtds_2D[r, c] = mtd
        prps_2D[r, c] = prp

    # Get shifts

    for r in range(nR + 1):
        mtdR, prpR = mtds_2D[r, :], prps_2D[r, :]
        for c in range(nC + 1):
            if mtdR[c] is not None:
                idx = mtdR[c]["idx"]
                if mtdR[c - 1] is not None:
                    dy, dx, scr = _get_shift(prpR[c - 1], prpR[c])
                    mtds[idx]["lshift"] = (dy, dx, scr) 
                else:
                    mtds[idx]["lshift"] = (np.nan,) * 3
                
    for c in range(nC + 1):
        mtdC, prpC = mtds_2D[:, c], prps_2D[:, c]
        for r in range(nR + 1):
            if mtdC[r] is not None:
                idx = mtdC[r]["idx"]
                if mtdC[r - 1] is not None:
                    dy, dx, scr = _get_shift(prpC[r - 1], prpC[r])
                    mtds[idx]["tshift"] = (dy, dx, scr)
                else:
                    mtds[idx]["tshift"] = (np.nan,) * 3
    
    return mtds

#%% Function : stich() --------------------------------------------------------

def stich(imgs, mtds):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_mode(arr):
        arr = arr[~np.isnan(arr)]
        arr = arr[arr < 0]
        values, counts = np.unique(arr, return_counts=True)
        return int(values[np.argmax(counts)])
    
    # Execute -----------------------------------------------------------------
    
    # Get mosaic shape
    nY, nX = imgs[0].shape
    nR = np.max([m["row"] for m in mtds])
    nC = np.max([m["col"] for m in mtds])
    
    # Get modal shifts 
    ldxs =  np.array([m["lshift"][1] for m in mtds])
    tdys =  np.array([m["tshift"][0] for m in mtds])
    ldx_mode = get_mode(ldxs)
    tdy_mode = get_mode(tdys)
    
    # Stich data
    stiched = np.zeros((nR * nY, nC * nX), dtype="float32") 
    for i, mtd in enumerate(mtds):
        row, col = mtd["row"], mtd["col"]
        tdy = row * tdy_mode
        ldx = col * ldx_mode
        y0r = mtd["y0r"] = mtd["y0"] + tdy 
        y1r = mtd["y1r"] = mtd["y1"] + tdy 
        x0r = mtd["x0r"] = mtd["x0"] + ldx
        x1r = mtd["x1r"] = mtd["x1"] + ldx
        stiched[y0r:y1r, x0r:x1r] = imgs[i]    
    
    return stiched

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Parameters
    df = 16
    
    # Paths
    raw_name = "Ins1e_wt_1.7nm_00"
    data_path = Path(f"D:\local_Mayrhofer\data\{raw_name}")
            
    # -------------------------------------------------------------------------
    
    # get_shift()
    t0 = time.time()
    print("get_shift() :", end=" ", flush=True)
    
    mtds = get_shift(data_path, df=df)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # stich()
    t0 = time.time()
    print("stich() :", end=" ", flush=True)
    
    imgs, _ = load_images(data_path, df=df)
    stiched = stich(imgs, mtds)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(
        stiched, name="stiched", colormap="gray",
        )
    
