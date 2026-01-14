#%% Imports -------------------------------------------------------------------

import time
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
    concatenate_df, condition_avg_df, plot_results,
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
        if self.procedure["mask"   ]: self.mask()
        if self.procedure["correct"]: self.correct()
        if self.procedure["measure"]: self.measure()
        if self.procedure["analyse"]: self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        self.root_path = self.data_path.parent
                
        self.raw_img_paths = list(self.data_path.glob("*.tif")) 
        for tag in ["rsc", "prp", "prd", "msk", "out"]:
            setattr(self, f"{tag}_path", self.data_path / f"{tag}") 
            img_paths = list(getattr(self, f"{tag}_path").glob("*.tif"))
            setattr(self, f"{tag}_img_paths", img_paths) 
        
        self.resv_paths = [
            p for p in self.root_path.rglob("results_v*.csv") 
            if p.parent != self.root_path
            ]
        self.resc_paths = [
            p for p in self.root_path.rglob("results_c*.csv") 
            if p.parent != self.root_path
            ]
        
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
        
        for path in self.prd_img_paths:
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
        
#%% Class(Main) : measure() ---------------------------------------------------

    def measure(self):
        
        print(f"measure() - {self.data_path.name}")
        
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
        
        t0 = time.time()
        print(" - measure & save : ", end="", flush=True)
        
        # Analyse
        for view in range(self.nviews):
                                    
            if self.outs[view]:
            
                # Fetch data
                prp  = self.prps[view]
                mskv = self.outs[view]["vesicles"]
                mskb = self.outs[view]["bounds"]
                mskl = self.outs[view]["labels"]
                
                if not np.all(mskb == 0):
                    
                    df_v = get_vesicle_results(
                        prp, mskv, mskb, mskl, 
                        dataset=self.data_path.name, pix_ref=self.pix_ref
                        )
                    df_c = get_cell_results(
                        prp, mskl, df_v, 
                        dataset=self.data_path.name, pix_ref=self.pix_ref
                        )
                        
                    # Save
                    df_v.to_csv(
                        self.out_path / f"results_vesicles_{view:02d}.csv",
                        index=False
                        )
                    df_c.to_csv(
                        self.out_path / f"results_cells_{view:02d}.csv",
                        index=False
                        )
                    
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

#%% Class(Main) : analyse() ---------------------------------------------------
    
    def analyse(self):
        
        print("analyse() - all datasets")

        # Concatenate results
        df_all_v = concatenate_df(self.resv_paths)
        df_all_c = concatenate_df(self.resc_paths)        
        
        # Condition average
        df_cnd_avg_v = condition_avg_df(df_all_v, self.conditions)
        df_cnd_avg_c = condition_avg_df(df_all_c, self.conditions)
        
        # Plot results
        fig = plot_results(
            df_all_v, df_all_c, 
            df_cnd_avg_v, df_cnd_avg_c, 
            self.conditions, self.conds_color,
            )
        
        # Save
        df_all_v.to_csv(
            self.root_path / "results_vesicles_all.csv", index=False)
        df_all_c.to_csv(
            self.root_path / "results_cells_all.csv", index=False)
        df_cnd_avg_v.to_csv(
            self.root_path / "results_vesicles_cnd_avg.csv", index=True)
        df_cnd_avg_c.to_csv(
            self.root_path / "results_cells_cnd_avg.csv", index=True)
        fig.savefig(
            self.root_path / "plot_results.png", format="png")
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from run import parameters, procedure
    main = Main(procedure=procedure, parameters=parameters)
    # resv_paths = main.resv_paths
    # resc_paths = main.resc_paths
    # df_all_v = main.df_all_v
    # df_all_c = main.df_all_c
    # df_cnd_avg_v = main.df_cnd_avg_v
    # df_cnd_avg_c = main.df_cnd_avg_c