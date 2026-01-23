#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# colors
from colors import fcolors  

#%% Function(s) ---------------------------------------------------------------

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
    
    # Parameters
    bin_num = 50
    bin_pc  = 99
    
    for t, (ax, tag) in enumerate(zip(axes, tags)):
        for c, cnd in enumerate(conditions):  

            df = df_all_c if t < 5 else df_all_v
                
            # Data
            val_all = df[tag]
            # bin_max = np.nanmax(val_all)
            bin_max = np.nanpercentile(val_all, bin_pc)
            bins = np.linspace(0, bin_max, bin_num + 1)
            val = df[df["dataset"].str.contains(
                cnd, case=False, na=False)][tag]
            counts, _ = np.histogram(val, bins=bins)
            counts = counts.astype(float) / np.nansum(counts)
            
            # Bar plot
            ax.bar(
                bins[1:], counts, width=bin_max / bin_num, alpha=0.5, 
                color=fcolors[f"{conds_color[c]}_40"]
                )
            
            # Formatting
            ax.set_xlim(0, np.nanpercentile(val_all, bin_pc))
            ax.set_xlabel(labels[t])
            ax.set_ylabel("count")
            ax.set_title(titles[t])

    plt.tight_layout() 

    return fig

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    
    # Paths
    root_path = Path("D:\local_Mayrhofer\data")

    # Load
    df_all_v = pd.read_csv(root_path / "results_vesicles_all.csv")
    df_all_c = pd.read_csv(root_path / "results_cells_all.csv")
    conditions = ["ins1e", "gigyf12"]
    conds_color = ["red", "blue"]
    
    # -------------------------------------------------------------------------
    
    plot_distributions(
        df_all_v, df_all_c,
        conditions, conds_color,
        )
    
    # -------------------------------------------------------------------------
    
    # cnd = conditions[0]
    # tag = "area_v_avg"
    
    # bin_num = 100
    # bin_pc  = 99
    
    # val_all = df_all_c[tag]
    # bin_max = np.nanmax(val_all)
    # bin_max_pc = np.percentile(val_all, 99)
    # bins = np.linspace(0, bin_max, bin_num + 1)
    
    # val0 = df_all_c[df_all_c["dataset"].str.contains(
    #     conditions[0], case=False, na=False)][tag]
    # val1 = df_all_c[df_all_c["dataset"].str.contains(
    #     conditions[1], case=False, na=False)][tag]
    # counts0, _ = np.histogram(val0, bins=bins)
    # counts0 = counts0.astype(float) / np.sum(counts0)
    # counts1, _ = np.histogram(val1, bins=bins)
    # counts1 = counts1.astype(float) / np.sum(counts1)
    
    # # Hist. plot
    # plt.bar(
    #     bins[1:], counts0, width=bin_max / bin_num, alpha=0.5, 
    #     color=fcolors[f"{conds_color[0]}_40"]
    #     )
    # plt.bar(
    #     bins[1:], counts1, width=bin_max / bin_num, alpha=0.5, 
    #     color=fcolors[f"{conds_color[1]}_40"]
    #     )
    
    # plt.xlim(0, np.percentile(val_all, bin_pc))
    
    # -------------------------------------------------------------------------
