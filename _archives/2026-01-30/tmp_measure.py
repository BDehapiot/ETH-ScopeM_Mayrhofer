#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

# skimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes

# scipy
from scipy.ndimage import binary_erosion, binary_dilation

#%% Function(s) ---------------------------------------------------------------

def get_vesicle_results(
        prp, mskv, mskj, mskl, view, dataset="", pix_ref=27.2):
    
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
    msk2 = binary_erosion(msk1)
    np.logical_xor(msk1, msk2, out=msk2)
    msk3 = binary_dilation(mskj)
    mskm = msk2 & ~msk3
    mska = mskj | mskm
    mskv_clean = mskv.copy()
    mskv_clean[mskl == 0] = 0
    
    # Distance transforms
    edtj = cv2.distanceTransform((mskj == 0).astype("uint8"), cv2.DIST_L2, 5)
    edtm = cv2.distanceTransform((mskm == 0).astype("uint8"), cv2.DIST_L2, 5)
    edta = cv2.distanceTransform((mska == 0).astype("uint8"), cv2.DIST_L2, 5)
    
    # Clear "no junctions" cells
    for props in regionprops(label(mskl), intensity_image=msk3):
        if props.intensity_max == 0:
            coords = props.coords
            edtj[tuple(coords.T)] = np.nan
    
    # Measure vesicles
    for props in regionprops(label(mskv_clean)):
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

def get_cell_results(
        prp, mskl, df_v, view, dist_thresh, dataset="", pix_ref=27.2):
    
    df_c = {
        
        "dataset"        : [],
        "view"           : [],
        "idx_c"          : [],
        "area_c"         : [],
        "numb_v"         : [],
        "dens_v"         : [],
        "area_v_avg"     : [],
        "ints_v_avg"     : [],
        "dist_v_j_avg"   : [],
        "dist_v_m_avg"   : [],
        "dist_v_a_avg"   : [],
        "dist_v_j_ratio" : [],
        "dist_v_m_ratio" : [],
        "dist_v_a_ratio" : [],
        
        }
    
    # Measure cells
    for props in regionprops(label(mskl)):
        idx_c = props.label
        df = df_v[df_v["idx_c"] == idx_c]
        area_c = props.area * ((pix_ref * 1e-3) ** 2)
        numb_v = len(df)
        dens_v = numb_v / area_c
        area_v_avg = df["area_v"].mean()
        ints_v_avg = df["ints_v"].mean()
        dist_v_j_avg = df["dist_v_j"].mean()
        dist_v_m_avg = df["dist_v_m"].mean()
        dist_v_a_avg = df["dist_v_a"].mean()
        dist_v_j_ratio = (df["dist_v_j"] < dist_thresh).mean()
        dist_v_m_ratio = (df["dist_v_m"] < dist_thresh).mean()
        dist_v_a_ratio = (df["dist_v_a"] < dist_thresh).mean()
        df_c["dataset"       ].append(dataset)
        df_c["view"          ].append(view)
        df_c["idx_c"         ].append(idx_c)
        df_c["area_c"        ].append(area_c)
        df_c["numb_v"        ].append(numb_v)
        df_c["dens_v"        ].append(dens_v)
        df_c["area_v_avg"    ].append(area_v_avg)
        df_c["ints_v_avg"    ].append(ints_v_avg)
        df_c["dist_v_j_avg"  ].append(dist_v_j_avg)
        df_c["dist_v_m_avg"  ].append(dist_v_m_avg) 
        df_c["dist_v_a_avg"  ].append(dist_v_a_avg) 
        df_c["dist_v_j_ratio"].append(dist_v_j_ratio)
        df_c["dist_v_m_ratio"].append(dist_v_m_ratio) 
        df_c["dist_v_a_ratio"].append(dist_v_a_ratio)
    
    return pd.DataFrame(df_c)


#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    
    # Parameters
    view = 3
    pix_ref = 27.2
    dist_thresh = 1
    dataset = "gigyf12_dko_3.25nm_00"
    data_path = Path(f"D:\local_Mayrhofer\data\{dataset}")
    
    # Load
    prp =  io.imread(Path(data_path, "prp", f"prp_{view:02d}.tif"))
    mskv = io.imread(Path(data_path, "out", f"msk_vesicles_hc_{view:02d}.tif"))
    mskj = io.imread(Path(data_path, "out", f"msk_junctions_hc_{view:02d}.tif"))
    mskl = io.imread(Path(data_path, "out", f"msk_labels_hc_{view:02d}.tif"))
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("get_vesicle_results() : ", end="", flush=True)
    
    df_v = get_vesicle_results(
        prp, mskv, mskj, mskl, view, 
        dataset=dataset, pix_ref=pix_ref
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    print("get_cell_results() : ", end="", flush=True)
    
    df_c = get_cell_results(
        prp, mskl, df_v, view, dist_thresh,
        dataset=dataset, pix_ref=pix_ref
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
        
    # -------------------------------------------------------------------------
    
    # Display
    # vwr = napari.Viewer()
    # vwr.add_image(msk1a, blending="additive", colormap="green", visible=1)
    # vwr.add_image(msk1b, blending="additive", colormap="magenta", visible=1)
