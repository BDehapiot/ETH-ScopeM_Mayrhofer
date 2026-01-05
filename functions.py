#%% Imports -------------------------------------------------------------------

import pandas as pd
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
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects, remove_small_holes, skeletonize)

# scipy
from scipy.ndimage import distance_transform_edt
                
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

def split_images(imgs, mtds, tiles_hw=24, tiles_ratio=0.5):
    
    # Get tile coordinates
    rows = [m["row"] for m in mtds]
    cols = [m["col"] for m in mtds]
    minR, maxR = min(rows), max(rows)
    minC, maxC = min(cols), max(cols)
    r0s = np.arange(minR, maxR, tiles_hw)
    r1s = r0s + (tiles_hw - 1)
    if r1s[-1] > maxR: r1s[-1] = maxR
    c0s = np.arange(minC, maxC, tiles_hw)
    c1s = c0s + (tiles_hw - 1)
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
            
    # Remove empty split
    tiles_min  = (tiles_hw ** 2) * tiles_ratio
    split_imgs = [
        sublist for sublist in split_imgs if len(sublist) >= tiles_min]
    split_mtds = [
        sublist for sublist in split_mtds if len(sublist) >= tiles_min]
    
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

import gc
from keras import backend as K

def predict_images(img, model_type="cells"):
    
    img = img.astype("float32") / 255
    load_name = list(Path.cwd().glob(f"model-{model_type}*"))[0]
    unet = UNet(load_name=load_name)
    prd = unet.predict(img, verbose=0)
    prd = (prd * 255).astype("uint8")
    
    del unet
    K.clear_session()
    gc.collect()   
    
    return prd

#%% Function(s) : process() ---------------------------------------------------

def get_mask(prd, thresh, min_size_o, min_size_h):
    msk = prd > thresh * 255
    msk = remove_small_objects(msk, min_size=min_size_o)
    msk = remove_small_holes(msk, area_threshold=min_size_h)
    return msk.astype("uint8")

#%% Function(s) : correct() ---------------------------------------------------
    
def sync_masks(mskc, mskn, mskv):
    mskn[mskc == 0] = 0
    mskv[mskc == 0] = 0
    mskv[mskn > 0 ] = 0 
    
def skeletonize_bounds(mskb, pad_width):
    mskb = np.pad(mskb > 0, pad_width, mode="constant", constant_values=1)
    mskb = skeletonize(mskb, method="lee")
    mskb = mskb[pad_width:-pad_width, pad_width:-pad_width]
    return mskb   

#%% Function(s) : analyse() ---------------------------------------------------

def get_vesicle_results(prp, mskv, mskb, mskl, pix_ref=27.2):
    
    resv = {
        
        "idxv" : [],
        "area" : [],
        "ints" : [],
        "dist" : [],
        "idxc" : [],
        
        }
    
    edt = distance_transform_edt(mskb == 0)
    for props in regionprops(label(mskv)):
        coords = props.coords
        idxv = props.label
        area = props.area * ((pix_ref * 1e-3) ** 2)
        ints = np.mean(prp[tuple(coords.T)])
        dist = np.mean(edt[tuple(coords.T)]) * pix_ref * 1e-3
        idxc = np.max(mskl[tuple(coords.T)])
        resv["idxv"].append(idxv)
        resv["area"].append(area)
        resv["ints"].append(ints)
        resv["dist"].append(dist)
        resv["idxc"].append(idxc)
    
    df_resv = pd.DataFrame(resv)
    
    return resv, df_resv

def get_cell_results(prp, mskl, df_resv, pix_ref=27.2):
    
    resc = {
        
        "idxc"      : [],
        "area"      : [],
        "numbv"     : [],
        "densv"     : [],
        "areav_avg" : [],
        "intsv_avg" : [],
        "distv_avg" : [],
        
        }
    
    for props in regionprops(label(mskl)):
        idxc = props.label
        df = df_resv[df_resv["idxc"] == idxc]
        area = props.area * ((pix_ref * 1e-3) ** 2)
        numbv = len(df)
        densv = numbv / area
        areav_avg = df["area"].mean()
        intsv_avg = df["ints"].mean()
        distv_avg = df["dist"].mean()
        resc["idxc"     ].append(idxc)
        resc["area"     ].append(area)
        resc["numbv"    ].append(numbv)
        resc["densv"    ].append(densv)
        resc["areav_avg"].append(areav_avg)
        resc["intsv_avg"].append(intsv_avg)
        resc["distv_avg"].append(distv_avg)    
        
    df_resc = pd.DataFrame(resc)
    
    return resc, df_resc
    