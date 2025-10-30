#%% Imports -------------------------------------------------------------------

import time
import shutil
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
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_objects, remove_small_holes

#%% Function(s) ---------------------------------------------------------------

def clear_directory(dir_path):
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        return
    
def custom_normalization(imgs):
    imgs = np.stack(imgs)
    imgs = imgs.astype("float32")
    imgs = norm_pct(imgs, pct_low=1, pct_high=99, mask=imgs > 0)
    return imgs

#%% Function : downscale_images() ---------------------------------------------

def downscale_images(data_path, df=16):
    
    # Nested function(s) ------------------------------------------------------
        
    def _downscale_images(img_path, level_path, df=16):
        
        # Load image
        img = io.imread(img_path)
            
        # Downscale image
        img = downscale_local_mean(img, df).astype("uint16")

        # Save downscaled image
        save_path = level_path / f"{img_path.stem}_level-{df}.tif"
        io.imsave(save_path, img, check_contrast=False)
        
    # Execute -----------------------------------------------------------------
    
    # Setup level directory
    level_path = data_path / f"level-{df}"
    if not level_path.exists():
        level_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("downscale_images() :", end=" ", flush=True)
    
    # Load & downscale images
    img_paths = list(data_path.glob("*.tif"))
    Parallel(n_jobs=-1)(
        delayed(_downscale_images)(img_path, level_path, df=df)
            for img_path in img_paths
            )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

#%% Function : load_images() --------------------------------------------------

def load_images(data_path, df=16, suffix="", return_metadata=False):

    if df == 1:
        level_path = data_path
    else:
        level_path = data_path / f"level-{df}"
    img_paths = list(level_path.glob(f"*level-{df}{suffix}.tif"))
    
    t0 = time.time()
    print("load_images() :", end=" ", flush=True)
    
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
        
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
            
    return imgs, mtds
        
#%% Function : get_shift() ----------------------------------------------------

def get_shift(imgs, mtds):
    
    # Nested function(s) ------------------------------------------------------
    
    def preprocess_image(img):
        prp = sobel(img)
        return prp
    
    def _get_shift(img_0, img_1):
        
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
    
    t0 = time.time()
    print("get_shift() :", end=" ", flush=True)
    
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
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    return mtds

#%% Function : stich() --------------------------------------------------------

def stich(imgs, mtds, scaling_coeff=1):
    
    # Nested function(s) ------------------------------------------------------
    
    def get_mode(arr):
        arr = arr[~np.isnan(arr)]
        arr = arr[arr < 0]
        values, counts = np.unique(arr, return_counts=True)
        return int(values[np.argmax(counts)])
    
    # Execute -----------------------------------------------------------------
    
    t0 = time.time()
    print("stich() :", end=" ", flush=True)
    
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
    imgs_s = np.zeros((nR * nY, nC * nX), dtype="float32") 
    for i, mtd in enumerate(mtds):
        row, col = mtd["row"], mtd["col"]
        tdy = row * tdy_mode
        ldx = col * ldx_mode
        y0r = mtd["y0"] * scaling_coeff + tdy 
        y1r = mtd["y1"] * scaling_coeff + tdy 
        x0r = mtd["x0"] * scaling_coeff + ldx
        x1r = mtd["x1"] * scaling_coeff + ldx
        imgs_s[y0r:y1r, x0r:x1r] = imgs[i]    
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    return imgs_s

#%% Function : predict() ------------------------------------------------------

def predict(img, model_type="cells"):
    load_name = list(Path.cwd().glob(f"model-{model_type}*"))[0]
    unet = UNet(load_name=load_name)
    prd = unet.predict(img, verbose=1)
    return (prd * 255).astype("uint8")

#%% Function : get_mask() -----------------------------------------------------

def get_mask(prd, thresh, min_size_o, min_size_h):
    msk = prd > thresh * 255
    msk = remove_small_objects(msk, min_size=min_size_o)
    msk = remove_small_holes(msk, area_threshold=min_size_h)
    return (msk * 255).astype("uint8")

def sync_masks(mskc, mskn, mskv):
    mskn[mskc == 0  ] = 0
    mskv[mskc == 0  ] = 0
    mskv[mskn == 255] = 0

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Parameters
    df0, df1 = 16, 8
    model_types = ["cells", "nuclei", "vesicles"]
    
    # Paths
    img_name = "Ins1e_wt_1.7nm_00"
    data_path = Path(f"D:\local_Mayrhofer\data\{img_name}")
    
    # Log
    log_str = f"{img_name} - df{df0}"
    print(log_str); print('-' * len(log_str))
            
    # downscale_images() ------------------------------------------------------
    
    # downscale_images(data_path, df=df0)
    # downscale_images(data_path, df=df1)

    # load_images() -----------------------------------------------------------
    
    imgs0, mtds = load_images(
        data_path, df=df0, suffix="", return_metadata=True)
    imgs1, _ = load_images(
        data_path, df=df1, suffix="", return_metadata=True)
        
    # get_shifts() ------------------------------------------------------------
    
    mtds = get_shift(imgs0, mtds)
        
    # stich() -----------------------------------------------------------------
    
    # Custom normalization
    imgs0 = custom_normalization(imgs0)
    imgs1 = custom_normalization(imgs1)
    
    # Stich 
    imgs0_s = stich(imgs0, mtds)
    imgs1_s = stich(imgs1, mtds, scaling_coeff=2)
    
    import napari
    vwr = napari.Viewer()
    vwr.add_image(imgs1_s, opacity=0.5)
    
    # predict() ---------------------------------------------------------------
    
    # prd_c = predict(imgs0_s, model_type="cells")
    # prd_n = predict(imgs0_s, model_type="nuclei")
    # prd_v = predict(imgs0_s, model_type="vesicles")
        
    # display -----------------------------------------------------------------

    # prd_params = {
    #     "cells"    : {"colormap" : "red"     , "opacity" : 0.1},
    #     "nuclei"   : {"colormap" : "bop blue", "opacity" : 0.2},
    #     "vesicles" : {"colormap" : "yellow"  , "opacity" : 1.0},
    #     }
    
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(imgs0_s, opacity=0.5)
    # for i, prd in enumerate([prd_c, prd_n, prd_v]):
    #     vwr.add_image(
    #         prd, name=model_types[i], 
    #         blending="additive", **prd_params[model_types[i]]
    #         ) 
