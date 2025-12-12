#%% Imports -------------------------------------------------------------------

from main import Main
from pathlib import Path

#%% Inputs (Main) -------------------------------------------------------------

dataset = "gigyf12_dko_1.7nm"

procedure = {
    
    "rescale" : 0,
    "prepare" : 0,
    "predict" : 0,
    "process" : 0,
    "correct" : 1,
    "analyse" : 0,
    
    }

parameters = {
        
    # Paths
    "root_path" : 
        Path(__file__).resolve().parent,
    "data_path" : 
        Path(rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}"),

    # Prepare
    "pix_ref" : 27.2, # nm
    "ntiles"  : 24,
    
    # Process
    "mask_params"  : {
        "cells"    : (0.5, 4096, 32),
        "nuclei"   : (0.5, 512, 32),
        "vesicles" : (0.25, 8, 4),
        },
    
    }
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)