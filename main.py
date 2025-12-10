#%% Imports -------------------------------------------------------------------

import time
import shutil
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Functions
from functions import (
    get_rescaling_factor, rescale_image, 
    load_images, normalize_images, split_images, get_shifts, stich_images,
    predict_images, get_mask,
    )

#%% Inputs --------------------------------------------------------------------

dataset = "gigyf12_dko_1.7nm"

# Procedure
procedure = {
    
    "rescale" : 0,
    "prepare" : 0,
    "predict" : 0,
    "process" : 1,
    
    }

# Parameters
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
    
    }
    
#%% Class(Main) ---------------------------------------------------------------

class Main:
    
    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        
        # Execute
        self.initialize()
        if self.procedure["rescale"]: self.rescale()
        if self.procedure["prepare"]: self.prepare()
        if self.procedure["predict"]: self.predict()
        if self.procedure["process"]: self.process()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        self.rsc_path = self.data_path / "rsc"
        self.prp_path = self.data_path / "prp"
        self.prd_path = self.data_path / "prd"
        self.prc_path = self.data_path / "prc"
        self.img_paths = list(self.data_path.glob("*.tif")) 
        if self.prp_path.exists():
            self.prp_paths = list(self.prp_path.glob("*.tif"))
        if self.prd_path.exists():
            self.prd_paths = list(self.prd_path.glob("*.tif"))
        
#%% Class(Main) : rescale() ---------------------------------------------------

    def rescale(self):
        
        print(f"rescale() - {self.data_path.name}")
        
        # Setup "rsc" directory
        if self.rsc_path.exists():
            shutil.rmtree(self.rsc_path)
        self.rsc_path.mkdir(parents=True)
            
        # Get rescaling factor
        rf = get_rescaling_factor(self.data_path.name, self.pix_ref)
        
        # Rescale -------------------------------------------------------------
        
        t0 = time.time()
        print(" - rescale : ", end="", flush=True)
        
        imgs = Parallel(n_jobs=-1)(
            delayed(rescale_image)(img_path, rf)
                for img_path in self.img_paths
                )
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

        # Save images ---------------------------------------------------------
        
        t0 = time.time()
        print(" - save    : ", end="", flush=True)
        
        for i, img_path in enumerate(self.img_paths):
            save_path = self.rsc_path / f"{img_path.stem}_rsc.tif"
            io.imsave(save_path, imgs[i], check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : prepare() ---------------------------------------------------

    def prepare(self):
        
        print(f"prepare() - {self.data_path.name}")
        
        # Setup "prp" directory
        if self.prp_path.exists():
            shutil.rmtree(self.prp_path)
        self.prp_path.mkdir(parents=True)
        
        # Prepare -------------------------------------------------------------
        
        t0 = time.time()
        print(" - prepare       : ", end="", flush=True)
        
        # Load
        imgs, mtds = load_images(self.rsc_path)
        
        # Normalize
        imgs = normalize_images(imgs)
        
        # Split
        imgs, mtds = split_images(imgs, mtds, ntiles=self.ntiles)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Shift & stich -------------------------------------------------------
        
        t0 = time.time()
        print(" - shift & stich : ", end="", flush=True)
        
        prps = []
        for i in range(len(imgs)):
            mtds[i] = get_shifts(imgs[i], mtds[i])
            prps.append(stich_images(imgs[i], mtds[i]))
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Save ----------------------------------------------------------------
        
        t0 = time.time()
        print(" - save          : ", end="", flush=True)
        
        for i in range(len(prps)):
            save_path = self.prp_path / f"prp_{i:02d}.tif"
            io.imsave(save_path, prps[i], check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : predict() ---------------------------------------------------

    def predict(self):
        
        print(f"predict() - {self.data_path.name}")
        
        # Setup "prd" directory
        if self.prd_path.exists():
            shutil.rmtree(self.prd_path)
        self.prd_path.mkdir(parents=True)
        
        # Predict & save ------------------------------------------------------

        t0 = time.time()
        print(" - predict & save : ", end="", flush=True)

        for i, prp_path in enumerate(self.prp_paths):
            for model_type in ["cells", "nuclei", "vesicles"]:
                prd = predict_images(
                    io.imread(prp_path), model_type=model_type)
                save_path = self.prd_path / f"prd_{model_type}_{i:02d}.tif"
                io.imsave(save_path, prd, check_contrast=False)

        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : process() ---------------------------------------------------
        
    def process(self):
        
        print(f"process() - {self.data_path.name}")
        
        # Setup "prc" directory
        if self.prc_path.exists():
            shutil.rmtree(self.prc_path)
        self.prc_path.mkdir(parents=True)
        
        # Process & save ------------------------------------------------------
        
        t0 = time.time()
        print(" - process & save : ", end="", flush=True)
        
        for prd_path in self.prd_paths:
            model_type = prd_path.stem.split("_")[-2]
            prd = io.imread(prd_path)
            msk = get_mask(prd, *self.parameters["mask_params"][model_type])
            save_path = self.prc_path / str(prd_path.name).replace("prd", "msk")
            io.imsave(save_path, msk, check_contrast=False)
            
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)