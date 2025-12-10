#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# numpy
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

# skimage
from skimage.filters import sobel
from skimage.transform import rescale
from skimage.morphology import remove_small_objects, remove_small_holes

#%% Function(s) ---------------------------------------------------------------
                
#%% Function(s) : rescale() ---------------------------------------------------

def get_rescaling_factor(data_name, psize_ref):
    parts = data_name.split("_")
    for part in parts:
        if "nm" in part:
            psize_raw = float(part.replace("nm", ""))
            rf = psize_raw / psize_ref
    return round(rf, 4)

def rescale_image(img_path, rf):
    img = io.imread(img_path)
    img = rescale(img, rf, preserve_range=True)
    return img.astype("uint16")

#%% Function(s) : prepare() ---------------------------------------------------

def load_images(rsc_path):

    img_paths = list(rsc_path.glob("*.tif"))
    
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

def normalize_images(imgs):
    imgs = np.stack(imgs)
    imgs = imgs.astype("float32")
    imgs = norm_pct(imgs, pct_low=0.1, pct_high=99.9, mask=imgs > 0)
    return (imgs * 255).astype("uint8")

def split_images(imgs, mtds, ntiles=24):
    
    # Get tile coordinates
    rows = [m["row"] for m in mtds]
    cols = [m["col"] for m in mtds]
    minR, maxR = min(rows), max(rows)
    minC, maxC = min(cols), max(cols)
    r0s = np.arange(minR, maxR, ntiles)
    r1s = r0s + (ntiles - 1)
    if r1s[-1] > maxR: r1s[-1] = maxR
    c0s = np.arange(minC, maxC, ntiles)
    c1s = c0s + (ntiles - 1)
    if c1s[-1] > maxC: c1s[-1] = maxC
    
    # Split images
    split_imgs, split_mtds = [], []
    for r0, r1 in zip(r0s, r1s):
        for c0, c1 in zip(c0s, c1s):
            tmp_imgs, tmp_mtds = [], []
            for i, (img, mtd) in enumerate(zip(imgs, mtds)):
                if r0 <= mtd["row"] <= r1 and c0 <= mtd["col"] <= c1:
                    tmp_imgs.append(imgs[i])
                    tmp_mtds.append(mtds[i])
            split_imgs.append(tmp_imgs)
            split_mtds.append(tmp_mtds)
    
    return split_imgs, split_mtds
    
#%% Function(s) : prepare() - get_shifts() ------------------------------------

def get_shifts(imgs, mtds):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_shift(img_0, img_1):
        
        # Cross power spectrum
        f0 = fft2(img_0)
        f1 = fft2(img_1)
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
        delayed(sobel)(img)
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

    idxs = np.array([m["idx"] for m in mtds])

    for r in range(nR + 1):
        mtdR, prpR = mtds_2D[r, :], prps_2D[r, :]
        for c in range(nC + 1):
            if mtdR[c] is not None:             
                idx = int(np.flatnonzero(idxs == mtdR[c]["idx"])[0])
                if mtdR[c - 1] is not None:
                    dy, dx, scr = get_shift(prpR[c - 1], prpR[c])
                    mtds[idx]["lshift"] = (dy, dx, scr) 
                else:
                    mtds[idx]["lshift"] = (np.nan,) * 3
                
    for c in range(nC + 1):
        mtdC, prpC = mtds_2D[:, c], prps_2D[:, c]
        for r in range(nR + 1):
            if mtdC[r] is not None:
                idx = int(np.flatnonzero(idxs == mtdC[r]["idx"])[0])
                if mtdC[r - 1] is not None:
                    dy, dx, scr = get_shift(prpC[r - 1], prpC[r])
                    mtds[idx]["tshift"] = (dy, dx, scr)
                else:
                    mtds[idx]["tshift"] = (np.nan,) * 3
    
    return mtds

#%% Function(s) : prepare() - stich() -----------------------------------------

def stich_images(imgs, mtds, scaling_coeff=1):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_mode(arr):
        arr = arr[~np.isnan(arr)]
        arr = arr[arr < 0]
        values, counts = np.unique(arr, return_counts=True)
        return int(values[np.argmax(counts)])
    
    def trim_images(imgs):
        imgs = imgs[~np.all(imgs == 0, axis=1)]
        imgs = imgs[:, ~np.all(imgs == 0, axis=0)]
        return imgs
    
    # Execute -----------------------------------------------------------------
        
    # Get mosaic shape
    nY, nX = imgs[0].shape
    nR = np.max([m["row"] for m in mtds])
    nC = np.max([m["col"] for m in mtds])
    
    # Get modal shifts 
    ldxs =  np.array([m["lshift"][1] for m in mtds])
    tdys =  np.array([m["tshift"][0] for m in mtds])
    ldx_mode = get_mode(ldxs) * scaling_coeff
    tdy_mode = get_mode(tdys) * scaling_coeff
    
    # Stich data
    imgs_s = np.zeros((nR * nY + nY, nC * nX + nX), dtype="uint8") 
    for i, mtd in enumerate(mtds):
        row, col = mtd["row"], mtd["col"]
        tdy = row * tdy_mode
        ldx = col * ldx_mode
        y0r = mtd["y0"] * scaling_coeff + tdy 
        y1r = mtd["y1"] * scaling_coeff + tdy 
        x0r = mtd["x0"] * scaling_coeff + ldx
        x1r = mtd["x1"] * scaling_coeff + ldx
        imgs_s[y0r:y1r, x0r:x1r] = imgs[i]    
    
    # Trim zeros rows & cols
    imgs_s = trim_images(imgs_s)
        
    return imgs_s

#%% Function(s) : predict() ---------------------------------------------------

def predict_images(img, model_type="cells"):
    img = img.astype("float32") / 255
    load_name = list(Path.cwd().glob(f"model-{model_type}*"))[0]
    unet = UNet(load_name=load_name)
    prd = unet.predict(img, verbose=0)
    return (prd * 255).astype("uint8")

#%% Function(s) : process() ---------------------------------------------------

def get_mask(prd, thresh, min_size_o, min_size_h):
    msk = prd > thresh * 255
    msk = remove_small_objects(msk, min_size=min_size_o)
    msk = remove_small_holes(msk, area_threshold=min_size_h)
    return msk.astype("uint8")

def sync_masks(mskc, mskn, mskv):
    mskn[mskc == 0] = 0
    mskv[mskc == 0] = 0
    mskv[mskn > 0 ] = 0

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    dataset = "gigyf12_dko_1.7nm"
    data_path = Path(
        rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")
    rsc_path = data_path / "rsc"
    
    # Load
    imgs, mtds = load_images(rsc_path)
    
    # Normalize
    imgs = normalize_images(imgs)
    
    # Split
    imgs, mtds = split_images(imgs, mtds, ntiles=24)
    
    # Shift & stich
    imgs_s = []
    for i in range(len(imgs)):
        mtds[i] = get_shifts(imgs[i], mtds[i])
        imgs_s.append(stich_images(imgs[i], mtds[i]))     
        
#%%
    
    # Imports
    import napari

    # Predict
    prd = predict_images(imgs_s[0], model_type="vesicles")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(prd)
    