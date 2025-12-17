#%% Imports -------------------------------------------------------------------

import time
import pickle
import shutil
import numpy as np
from skimage import io
from joblib import Parallel, delayed

# Classes
from correct import Correct

# Functions
from functions import (
    get_rescaling_factor, rescale_image, 
    load_images, normalize_images, split_images, get_shifts, stich_images,
    predict_images, get_mask,
    get_vesicle_results, get_cell_results,
    )
    
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
        if self.procedure["correct"]: self.correct()
        if self.procedure["analyse"]: self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        self.raw_img_paths = list(self.data_path.glob("*.tif")) 
        for tag in ["rsc", "prp", "prd", "prc", "out"]:
            setattr(self, f"{tag}_path", self.data_path / f"{tag}") 
            img_paths = list(getattr(self, f"{tag}_path").glob("*.tif"))
            setattr(self, f"{tag}_img_paths", img_paths) 
        
#%% Class(Main) : rescale() ---------------------------------------------------

    def rescale(self):
                
        # Setup "rsc" directory
        if self.rsc_path.exists():
            shutil.rmtree(self.rsc_path)
        self.rsc_path.mkdir(parents=True, exist_ok=True)
            
        # Get rescaling factor
        rf = get_rescaling_factor(self.data_path.name, self.pix_ref)
        
        # Rescale -------------------------------------------------------------
        
        if self.parallel:
        
            print(f"rescale() - {self.data_path.name}")    
        
            t0 = time.time()
            print(" - rescale : ", end="", flush=True)
                    
            imgs = Parallel(n_jobs=-1)(
                delayed(rescale_image)(img_path, rf)
                    for img_path in self.raw_img_paths
                    )
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")
            
        else:
            
            imgs = []
            for img_path in self.raw_img_paths:
                t0 = time.time()
                print(f"rescale() - {img_path.name}") 
                imgs.append(rescale_image(img_path, rf))

        # Save images ---------------------------------------------------------
        
        t0 = time.time()
        print(" - save    : ", end="", flush=True)
        
        for i, path in enumerate(self.raw_img_paths):
            save_path = self.rsc_path / f"{path.stem}_rsc.tif"
            io.imsave(save_path, imgs[i], check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : prepare() ---------------------------------------------------

    def prepare(self):
        
        print(f"prepare() - {self.data_path.name}")
        
        # Setup "prp" directory
        if self.prp_path.exists():
            shutil.rmtree(self.prp_path)
        self.prp_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare -------------------------------------------------------------
        
        t0 = time.time()
        print(" - prepare       : ", end="", flush=True)
        
        # Load
        imgs, mtds = load_images(self.rsc_path)
        
        # Normalize
        imgs = normalize_images(imgs)
        
        # Split
        imgs, mtds = split_images(
            imgs, mtds, tiles_hw=self.tiles_hw, tiles_ratio=self.tiles_ratio)
                
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        self.imgs = imgs
        self.mtds = mtds
        
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
        self.prd_path.mkdir(parents=True, exist_ok=True)
        
        # Predict & save ------------------------------------------------------

        t0 = time.time()
        print(" - predict & save : ", end="", flush=True)

        for i, path in enumerate(self.prp_img_paths):
            for model_type in ["cells", "nuclei", "vesicles"]:
                prd = predict_images(
                    io.imread(path), model_type=model_type)
                save_path = self.prd_path / f"prd_{model_type}_{i:02d}.tif"
                io.imsave(save_path, prd, check_contrast=False)

        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : mask() ------------------------------------------------------
        
    def mask(self):
        
        print(f"mask() - {self.data_path.name}")
        
        # Setup "msk" directory
        if self.msk_path.exists():
            shutil.rmtree(self.msk_path)
        self.msk_path.mkdir(parents=True, exist_ok=True)
        
        # Mask & save ---------------------------------------------------------
        
        t0 = time.time()
        print(" - mask & save : ", end="", flush=True)
        
        for path in self.prd_paths:
            model_type = path.stem.split("_")[-2]
            prd = io.imread(path)
            msk = get_mask(prd, *self.parameters["mask_params"][model_type])
            save_path = self.msk_path / str(path.name).replace("prd", "msk")
            io.imsave(save_path, msk, check_contrast=False)
            
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : correct() ---------------------------------------------------

    def correct(self):
        
        print(f"correct() - {self.data_path.name}")
        Correct(procedure=self.procedure, parameters=self.parameters)
        
#%% Class(Main) : analyse() ---------------------------------------------------

    def analyse(self):
        
        self.nviews = len(self.prp_img_paths)
        
        # Load data
        prps, outs = [], []
        for view in range(self.nviews):
            tmp_dict = {}
            for path in self.prp_img_paths:
                if f"{view:02d}" in path.name:
                    prps.append(io.imread(path))
            for path in self.out_img_paths:
                if f"{view:02d}" in path.name:
                    tmp_dict[f"{path.stem.split('_')[1]}"] = io.imread(path)
            outs.append(tmp_dict)
        self.prps = prps
        self.outs = outs
        
        # Analyse
        for view in range(self.nviews):
                                    
            if self.outs[view]:
            
                # Fetch data
                prp  = self.prps[view]
                mskv = self.outs[view]["vesicles"]
                mskb = self.outs[view]["bounds"]
                mskl = self.outs[view]["labels"]
                
                if not np.all(mskb == 0):
                    
                    resv, df_resv = get_vesicle_results(
                        prp, mskv, mskb, mskl, pix_ref=self.pix_ref)
                    resc, df_resc = get_cell_results(
                        prp, mskl, df_resv, pix_ref=self.pix_ref)
                        
                    # Save
                    save_namev = f"results_vesicles_{view:02d}"
                    save_namec = f"results_cells_{view:02d}"
                    with open(self.out_path / (save_namev + ".pkl"), "wb") as f:
                        pickle.dump(resv, f)
                    with open(self.out_path / (save_namec + ".pkl"), "wb") as f:
                        pickle.dump(resc, f)
                    df_resv.to_csv(
                        self.out_path / (save_namev + ".csv"), index=False)
                    df_resc.to_csv(
                        self.out_path / (save_namec + ".csv"), index=False)
                 
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from run import parameters, procedure
    main = Main(procedure=procedure, parameters=parameters)