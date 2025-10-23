#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
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

# Parameters
df = 16
suffixes = ["cells", "nuclei", "vesicles"]

# Paths
img_name = "Ins1e_wt_1.7nm_00"
data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")
if df == 1:
    level_path = data_path
else:
    level_path = data_path / f"level-{df}"

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
    scr = float(corr.max() / np.sum(corr))

    return dy, dx, scr

def get_mode(arr):
    arr = arr[~np.isnan(arr)]
    arr = arr[arr < 0]
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%% Development ---------------------------------------------------------------

    # Paths
    img_paths = list(level_path.glob("*.tif"))
    
    t0 = time.time()
    print("load_data() :", end=" ", flush=True)
    
    # Load data
    imgs, mtds = [], []
    prds = {s: [] for s in suffixes}
    for i, img_path in enumerate(img_paths):
        
        # Load images
        img = io.imread(img_path)
        imgs.append(img)
        
        # Load predictions
        for suffix in suffixes:
            prds[suffix].append(io.imread(
                level_path / "prds" / (img_path.stem + f"_prd-{suffix}.tif")))
        
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
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    nR = np.max([m["row"] for m in mtds])
    nC = np.max([m["col"] for m in mtds])
    
    # Normalize background
    nrms = normalize_background(imgs)
    
    t0 = time.time()
    print("preprocess_images() :", end=" ", flush=True)
    
    # Preprocess images
    prps = Parallel(n_jobs=-1)(
        delayed(preprocess_images)(nrm)
        for nrm in nrms
        )
    prps = norm_pct(prps)
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
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
    print("get_shift() :", end=" ", flush=True)
    
    for r in range(nR + 1):
        mtdR = mtds_2D[r, :]
        prpR = prps_2D[r, :]
        for c in range(nC + 1):
            if mtdR[c] is not None:
                idx = mtdR[c]["idx"]
                if mtdR[c - 1] is not None:
                    dy, dx, scr = get_shift(prpR[c - 1], prpR[c])
                    mtdR[c]["lshift"] = mtds[idx]["lshift"] = (dy, dx, scr) 
                else:
                    mtdR[c]["lshift"] = mtds[idx]["lshift"] = (np.nan,) * 3
                
    for c in range(nC + 1):
        mtdC = mtds_2D[:, c]
        prpC = prps_2D[:, c]
        for r in range(nR + 1):
            if mtdC[r] is not None:
                idx = mtdC[r]["idx"]
                if mtdC[r - 1] is not None:
                    dy, dx, scr = get_shift(prpC[r - 1], prpC[r])
                    mtdC[r]["tshift"] = mtds[idx]["tshift"] = (dy, dx, scr)
                else:
                    mtdC[r]["tshift"] = mtds[idx]["tshift"] = (np.nan,) * 3
                
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    ldxs =  np.array([m["lshift"][1] for m in mtds])
    lscrs = np.array([m["lshift"][2] for m in mtds])
    tdys =  np.array([m["tshift"][0] for m in mtds])
    tscrs = np.array([m["tshift"][2] for m in mtds])
    ldx_mode = get_mode(ldxs)
    tdy_mode = get_mode(tdys)
    
#%% Development ---------------------------------------------------------------

    t0 = time.time()
    print("stich() :", end=" ", flush=True)

    tmp = np.zeros((nR * nY, nC * nX)) 
    nrms_s = tmp.copy()
    for suffix in suffixes:
        prds[f"{suffix}_s"] = tmp.copy()
    
    for i, mtd in enumerate(mtds):
        row, col = mtd["row"], mtd["col"]
        tdy = row * tdy_mode
        ldx = col * ldx_mode
        y0r = mtd["y0r"] = mtd["y0"] + tdy 
        y1r = mtd["y1r"] = mtd["y1"] + tdy 
        x0r = mtd["x0r"] = mtd["x0"] + ldx
        x1r = mtd["x1r"] = mtd["x1"] + ldx
        nrms_s[y0r:y1r, x0r:x1r] = nrms[i]
        for suffix in suffixes:
            prds[f"{suffix}_s"][y0r:y1r, x0r:x1r] = prds[suffix][i]

    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    prd_colors = ["bop blue", "bop orange", "bop purple"]
    vwr.add_image(
        nrms_s, name="nrms_s", colormap="gray",
        contrast_limits=[0.8, 1.1]
        )
    for i, suffix in enumerate(suffixes):
        vwr.add_image(
            prds[f"{suffix}_s"], name=f"{suffix}_s", colormap=prd_colors[i],
            blending="additive", opacity=0.1,
            )