#%% Imports -------------------------------------------------------------------

from main import Main
from pathlib import Path

#%% Inputs (Main) -------------------------------------------------------------

procedure = {
    
    # one dataset
    "rescale" : 0,
    "prepare" : 0,
    "predict" : 0,
    "mask"    : 0,
    "correct" : 0,
    
    # all dataset
    "measure" : 0,
    "analyse" : 1,

    }

parameters = {
    
    "dataset" : 
        # "gigyf12_dko_1.7nm",
        # "gigyf12_dko_3.25nm_00",
        # "gigyf12_dko_3.25nm_01",
        # "ins1e_wt_1.7nm",
        # "ins1e_wt_3.25nm_00",
        "ins1e_wt_3.25nm_01",
    
    # Paths
    "root_path" : 
        # Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data"),
        Path("D:\local_Mayrhofer\data"),

    # Rescale
    "parallel"     : False,

    # Prepare
    "pix_ref"      : 27.2, # nm
    "tiles_hw"     : 24,
    "tiles_ratio"  : 0.5,
    
    # Mask
    "mask_params"  : {
        "cells"    : (0.5, 1e4, 128),
        "nuclei"   : (0.5, 1e3, 128),
        "vesicles" : (0.25, 8, 4),
        },
    
    # Correct
    "load_prd"     : False,
    
    # Measure
    "dist_thresh"  : 0.5, # µm
    
    # Analyse
    "conditions"   : ["ins1e", "gigyf12"],
    "conds_color"  : ["red", "blue"], 
    
    }
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)