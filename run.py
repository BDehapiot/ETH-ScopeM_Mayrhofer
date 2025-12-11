#%% Imports -------------------------------------------------------------------

from pathlib import Path
from main import Main
from correct import Correct

#%% Inputs (Main) -------------------------------------------------------------

dataset = "gigyf12_dko_1.7nm"

procedure = {
    
    "rescale" : 0,
    "prepare" : 0,
    "predict" : 0,
    "process" : 0,
    "correct" : 1,
    
    }

parameters = {
        
    # Paths
    "root_path" : 
        Path(__file__).resolve().parent,
    "data_path" : 
        Path(rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}"),

    # Prepare
    "pix_ref" : 27.2,
    "ntiles"  : 24,
    
    # Process
    "mask_params"  : {
        "cells"    : (0.5, 4096, 32),
        "nuclei"   : (0.5, 512, 32),
        "vesicles" : (0.25, 8, 4),
        },
    
    # Correct
    
    
    }
    
#%% Inputs (Correct) ----------------------------------------------------------

layer_parameters = {
    
    "prp" : {
        "name"     : "prp",
        "visible"  : 1,
        "opacity"  : 0.6,
        },
    
    "prdc" : {
        "name"     : "prdc",
        "colormap" : "gist_earth",
        "blending" : "additive",
        "visible"  : 0,
        "opacity"  : 1.0,
        },
    
    "prdn" : {
        "name"     : "prdn",
        "colormap" : "gist_earth",
        "blending" : "additive",
        "visible"  : 0,
        "opacity"  : 1.0,
        },
    
    "prdv" : {
        "name"     : "prdv",
        "colormap" : "gist_earth",
        "blending" : "additive",
        "visible"  : 0,
        "opacity"  : 1.0,
        },
    
    "mskc" : {
        "name"     : "mskc",
        "blending" : "additive",
        "visible"  : 1,
        "opacity"  : 0.2,
        },
    
    "mskn" : {
        "name"     : "mskn",
        "blending" : "additive",
        "visible"  : 1,
        "opacity"  : 0.4,
        },
    
    "mskv" : {
        "name"     : "mskv",
        "blending" : "additive",
        "visible"  : 1,
        "opacity"  : 0.6,
        },
    
    "mskb" : {
        "name"     : "mskb",
        "blending" : "additive",
        "visible"  : 0,
        "opacity"  : 0.6,
        },
    
    "mskl" : {
        "name"     : "mskl",
        "blending" : "additive",
        "visible"  : 0,
        "opacity"  : 0.2,
        },
    
    }
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    correct = Correct(procedure=procedure, parameters=parameters)