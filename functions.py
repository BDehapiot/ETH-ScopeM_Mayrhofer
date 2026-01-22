#%% Imports -------------------------------------------------------------------

import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
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
from skimage.segmentation import find_boundaries
from skimage.morphology import (
    remove_small_objects, remove_small_holes, skeletonize, binary_dilation)

# scipy
from scipy.ndimage import distance_transform_edt

# colors
from colors import fcolors  
                        
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

#%% Function(s) : mask() ------------------------------------------------------

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
    
def skeletonize_junctions(mskj, pad_width):
    mskj = np.pad(mskj > 0, pad_width, mode="constant", constant_values=1)
    mskj = skeletonize(mskj, method="lee")
    mskj = mskj[pad_width:-pad_width, pad_width:-pad_width]
    return mskj   

#%% Function(s) : measure() ---------------------------------------------------

def get_vesicle_results(prp, mskv, mskj, mskl, view, dataset="", pix_ref=27.2):
    
    df_v = {
        
        "dataset"   : [],
        "view"      : [],
        "idx_c"     : [],
        "idx_v"     : [],
        "area_v"    : [],
        "ints_v"    : [],
        "dist_v_j"  : [],
        "dist_v_m"  : [],
        "dist_v_a"  : [],

        }
    
    # Process masks
    msk1 = remove_small_holes(
        mskl > 0, area_threshold=4096, connectivity=2)
    msk2 = find_boundaries(msk1, mode="inner")
    msk3 = binary_dilation(mskj)
    mskm = msk2 & ~msk3
    mska = mskj | mskm
    mskv[mskl == 0] = 0
    
    # Distance transforms
    edtj = distance_transform_edt(mskj == 0)
    edtm = distance_transform_edt(mskm == 0)
    edta = distance_transform_edt(mska == 0)
    
    # Clear "no junctions" cells
    for props in regionprops(label(mskl), intensity_image=msk3):
        if props.intensity_max == 0:
            coords = props.coords
            edtj[tuple(coords.T)] = np.nan
    
    # Measure vesicles
    for props in regionprops(label(mskv)):
        coords = props.coords
        idx_v = props.label
        idx_c = np.max(mskl[tuple(coords.T)])
        area_v = props.area * ((pix_ref * 1e-3) ** 2)
        ints_v = np.mean(prp[tuple(coords.T)])
        dist_v_j = np.mean(edtj[tuple(coords.T)]) * pix_ref * 1e-3
        dist_v_m = np.mean(edtm[tuple(coords.T)]) * pix_ref * 1e-3
        dist_v_a = np.mean(edta[tuple(coords.T)]) * pix_ref * 1e-3
        df_v["dataset"  ].append(dataset)
        df_v["view"     ].append(view)
        df_v["idx_v"    ].append(idx_v)
        df_v["idx_c"    ].append(idx_c)
        df_v["area_v"   ].append(area_v)
        df_v["ints_v"   ].append(ints_v)
        df_v["dist_v_j" ].append(dist_v_j)
        df_v["dist_v_m" ].append(dist_v_m)
        df_v["dist_v_a" ].append(dist_v_a)

    return pd.DataFrame(df_v)

def get_cell_results(prp, mskl, df_resv, view, dataset="", pix_ref=27.2):
    
    df_c = {
        
        "dataset"      : [],
        "view"         : [],
        "idx_c"        : [],
        "area_c"       : [],
        "numb_v"       : [],
        "dens_v"       : [],
        "area_v_avg"   : [],
        "ints_v_avg"   : [],
        "dist_v_j_avg" : [],
        "dist_v_m_avg" : [],
        "dist_v_a_avg" : [],
        
        }
    
    # Measure cells
    for props in regionprops(label(mskl)):
        idx_c = props.label
        df = df_resv[df_resv["idx_c"] == idx_c]
        area_c = props.area * ((pix_ref * 1e-3) ** 2)
        numb_v = len(df)
        dens_v = numb_v / area_c
        area_v_avg = df["area_v"].mean()
        ints_v_avg = df["ints_v"].mean()
        dist_v_j_avg = df["dist_v_j"].mean()
        dist_v_m_avg = df["dist_v_m"].mean()
        dist_v_a_avg = df["dist_v_a"].mean()
        df_c["dataset"     ].append(dataset)
        df_c["view"        ].append(view)
        df_c["idx_c"       ].append(idx_c)
        df_c["area_c"      ].append(area_c)
        df_c["numb_v"      ].append(numb_v)
        df_c["dens_v"      ].append(dens_v)
        df_c["area_v_avg"  ].append(area_v_avg)
        df_c["ints_v_avg"  ].append(ints_v_avg)
        df_c["dist_v_j_avg"].append(dist_v_j_avg)
        df_c["dist_v_m_avg"].append(dist_v_m_avg) 
        df_c["dist_v_a_avg"].append(dist_v_a_avg) 
    
    return pd.DataFrame(df_c)

#%% Function(s) : analyse() ---------------------------------------------------

def concatenate_df(df_paths):
    df = []
    for path in df_paths:
        df.append(pd.read_csv(path))
    return pd.concat(df)

def condition_avg_df(df_all, conditions):
    df_cnd_avg = pd.DataFrame()
    for cnd in conditions:
        df_cnd = df_all[df_all["dataset"].str.contains(
            cnd, case=False, na=False)]
        df_cnd = df_cnd.loc[:, ~df_cnd.columns.str.contains("idx")]
        df_cnd_avg[f"{cnd}_avg"] = df_cnd.mean(numeric_only=True)
        df_cnd_avg[f"{cnd}_std"] = df_cnd.std (numeric_only=True)
        df_cnd_avg[f"{cnd}_sem"] = df_cnd.sem (numeric_only=True)
    return df_cnd_avg.T

def plot_results(
        df_all_v, df_all_c,
        df_cnd_avg_v, df_cnd_avg_c,
        conditions, conds_color,
        ):
    
    tags = [
        "area_c", "numb_v", "dens_v",
        "area_v_avg", "ints_v_avg", 
        "dist_v_j_avg", "dist_v_m_avg", "dist_v_a_avg",
        "dist_v_j", "dist_v_m", "dist_v_a",
        ]
    
    titles = [
        "cell area", "vesicle number", "vesicle density",
        "vesicle area", "vesicle intensity", 
        "vesicle dist. j", "vesicle dist. m", "vesicle dist. a",
        "vesicle dist. j distribution",
        "vesicle dist. m distribution",
        "vesicle dist. a distribution",
        ]
    
    labels = [
        "area (µm²)", "count", "count.µm-2",
        "area (µm²)", "intensity (A.U.)", 
        "distance (µm)", "distance (µm)", "distance (µm)",
        "count", "count", "count",
        ]
    
    # Initialize plot
    fig = plt.figure(figsize=(6, 12))  
    fig.suptitle(
        (
        f"n cells = {len(df_all_c)}\n"
        f"n vesicles = {len(df_all_v)}"
        ), 
        fontsize=12, x=0.03, ha="left"
        )
    
    gs = fig.add_gridspec(6, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[3, :]),
        fig.add_subplot(gs[4, :]),
        fig.add_subplot(gs[5, :]),
        ]
    
    for t, (ax, tag) in enumerate(zip(axes, tags)):
        for c, cnd in enumerate(conditions):    
            
            if t < 8:

                # Data
                avg, sem = (
                    df_cnd_avg_c.loc[f"{cnd}_avg", tag],
                    df_cnd_avg_c.loc[f"{cnd}_sem", tag],
                    )
                
                # Bar plot
                ax.bar(
                    c, avg, yerr=sem, capsize=5, alpha=1, width=0.8,
                    color=fcolors[f"{conds_color[c]}_40"],
                    )
                
                # Formatting
                ax.set_xticks(np.arange(len(conditions)))
                ax.set_xticklabels(conditions, rotation=0)
                ax.set_ylabel(labels[t])
                ax.set_title(titles[t])
                
            else:
                
                # Data
                val = df_all_v[df_all_v["dataset"].str.contains(
                    cnd, case=False, na=False)][tag]
                
                # Hist. plot
                ax.hist(
                    val, label=cnd, bins=300, density=True, alpha=0.5, 
                    color=fcolors[f"{conds_color[c]}_40"]
                    )
                
                # Formatting
                ax.set_xlim(-0.2, 10.2)
                ax.set_xlabel("distance (µm)")
                ax.set_ylabel(labels[t])
                ax.set_title(titles[t])
                ax.legend(loc="upper right")
    
    plt.tight_layout() 
    
    return fig

def plot_distributions(
        df_all_v, df_all_c,
        conditions, conds_color,
        ):
    
    tags = [
        "area_c", "numb_v", "dens_v",
        "area_v_avg", "ints_v_avg", 
        "dist_v_j", "dist_v_m", "dist_v_a",
        ]
    
    titles = [
        "cell area", "ves. num.", "ves. dens.",
        "ves. area", "ves. int.", 
        "ves. dist. j", "ves. dist. m", "ves. dist. a",
        ]
    
        
    labels = [
        "area (µm²)", "count", "count.µm-2",
        "area (µm²)", "intensity (A.U.)", 
        "distance (µm²)", "distance (µm²)", "distance (µm²)",
        ]
    
    # Initialize plot
    fig = plt.figure(figsize=(6, 12))  
    fig.suptitle(
        (
        f"n cells = {len(df_all_c)}\n"
        f"n vesicles = {len(df_all_v)}"
        ), 
        fontsize=12, x=0.03, ha="left"
        )
    
    gs = fig.add_gridspec(6, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        ]
    
    for t, (ax, tag) in enumerate(zip(axes, tags)):
        for c, cnd in enumerate(conditions):  
            
            if t < 5:
                
                # Data
                val = df_all_c[df_all_c["dataset"].str.contains(
                    cnd, case=False, na=False)][tag]
                weights = np.ones_like(val) / len(val)
                
                # Hist. plot
                ax.hist(
                    val, label=cnd, bins=30, weights=weights, alpha=0.5, 
                    color=fcolors[f"{conds_color[c]}_40"]
                    )
                
            else:
                
                # Data
                val = df_all_v[df_all_v["dataset"].str.contains(
                    cnd, case=False, na=False)][tag]
                weights = np.ones_like(val) / len(val)
                
                # Hist. plot
                ax.hist(
                    val, label=cnd, bins=300, weights=weights, alpha=0.5, 
                    color=fcolors[f"{conds_color[c]}_40"]
                    )
            
            # Formatting
            # ax.set_xlim(-0.2, 10.2)
            ax.set_xlabel(labels[t])
            ax.set_ylabel("count")
            ax.set_title(titles[t])
            # ax.legend(loc="upper right")

    plt.tight_layout() 

    return fig

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from main import Main
    from run import parameters, procedure
    main = Main(procedure=procedure, parameters=parameters)