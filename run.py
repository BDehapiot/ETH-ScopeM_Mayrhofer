#%% Imports -------------------------------------------------------------------

from main import Main
from pathlib import Path

#%% Inputs (Main) -------------------------------------------------------------

# dataset = "gigyf12_dko_1.7nm"
# dataset = "gigyf12_dko_3.25nm_00"
# dataset = "gigyf12_dko_3.25nm_01"
# dataset = "ins1e_wt_1.7nm"
# dataset = "ins1e_wt_3.25nm_00"
dataset = "ins1e_wt_3.25nm_01"

procedure = {
    
    "rescale" : 0,
    "prepare" : 0,
    "predict" : 0,
    "mask"    : 0,
    "correct" : 1,
    "analyse" : 0,
    
    }

parameters = {
        
    # Paths
    "root_path" : 
        Path(__file__).resolve().parent,
    "data_path" : 
        Path(rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}"),

    # Rescale
    "parallel"     : False,

    # Prepare
    "pix_ref"      : 27.2, # nm
    "tiles_hw"     : 24,
    "tiles_ratio"  : 0.5,
    
    # Process
    "mask_params"  : {
        "cells"    : (0.5, 1e4, 128),
        "nuclei"   : (0.5, 1e3, 128),
        "vesicles" : (0.25, 8, 4),
        },
    
    }
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    # imgs = main.imgs
    # mtds = main.mtds