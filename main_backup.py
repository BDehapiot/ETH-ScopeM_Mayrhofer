#%% Imports -------------------------------------------------------------------

import time
import pickle
from skimage import io
from pathlib import Path

# functions
from functions import (
    clear_directory,
    downscale_images, load_images, custom_normalization, 
    get_shift, stich, predict, get_mask,
    )

#%% Inputs --------------------------------------------------------------------

procedure = {
    "downscale"  : 1,
    "preprocess" : 2,
    "predict"    : 0,
    "process"    : 0,
    }

parameters = {
    
    # Paths
    "img_name"  : "Ins1e_wt_1.7nm_00",
    "data_path" : Path("D:\local_Mayrhofer\data"),
    
    # Downscale
    "df0" : 16, # processing
    "df1" :  8, # display
    
    # Predict
    "model_types" : ["cells", "nuclei", "vesicles"],
    
    # Process
    "mask_parameters" : [
        (0.5, 4096, 32),
        (0.5, 512, 32),
        (0.5, 32, 8),
        ],   
    
    }

#%% Class(Main) : -------------------------------------------------------------

class Main:

    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.df0 = parameters["df0"]
        self.df1 = parameters["df1"]
        
        # Run
        self.initialize()
        if self.procedure["downscale" ]: self.downscale() 
        if self.procedure["preprocess"]: self.preprocess() 
        if self.procedure["predict"   ]: self.predict()
        if self.procedure["process"   ]: self.process()
        
#%% Class(Main) : initialize() ------------------------------------------------
        
    def initialize(self):
        
        # Paths
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level0_path = self.data_path / f"level-{parameters['df0']}"
        self.level1_path = self.data_path / f"level-{parameters['df1']}"
        self.outputs0_path = self.level0_path / "outputs"
        self.outputs1_path = self.level1_path / "outputs"
        
        # Files
        file_map = {
            "imgs": ("imgs.tif", io.imread),
            "mtds": ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            "prdc": ("prdc.tif", io.imread),
            "prdn": ("prdn.tif", io.imread),
            "prdv": ("prdv.tif", io.imread),
            "mskc": ("mskc.tif", io.imread),
            "mskn": ("mskn.tif", io.imread),
            "mskv": ("mskv.tif", io.imread),
            }
        
        for attr, (filename, loader) in file_map.items():
            path = self.outputs0_path / filename
            if path.is_file():
                setattr(self, attr, loader(path))
                
#%% Class(Main) : downscale() -------------------------------------------------

    def downscale(self):

        if not self.level0_path.exists() or self.procedure["downscale"] == 2:
            log_str = f"\ndownscale() - {self.img_name} - df{self.df0}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.level0_path)        
            downscale_images(self.data_path, df=self.df0)
            
        if not self.level1_path.exists() or self.procedure["downscale"] == 2:
            log_str = f"\ndownscale() - {self.img_name} - df{self.df1}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.level1_path)        
            downscale_images(self.data_path, df=self.df1)

#%% Class(Main) : preprocess() ------------------------------------------------

    def preprocess(self):
        
        # Nested funtion(s) ---------------------------------------------------

        def _preprocess():
            
            # Setup outputs path
            self.outputs0_path.mkdir(parents=True, exist_ok=True)
            self.outputs1_path.mkdir(parents=True, exist_ok=True)
            
            # Load images
            imgs0, mtds = load_images(
                self.data_path, df=self.df0, suffix="", return_metadata=True)
            imgs1, _ = load_images(
                self.data_path, df=self.df1, suffix="", return_metadata=True)
                
            # Get shifts
            mtds = get_shift(imgs0, mtds)
            
            # Custom normalization
            imgs0 = custom_normalization(imgs0)
            imgs1 = custom_normalization(imgs1)
            
            # Stich
            imgs0_s = stich(imgs0, mtds, scaling_coeff=1)  
            imgs1_s = stich(imgs1, mtds, scaling_coeff=self.df0 // self.df1)  
            
            # Save
            with open(self.outputs0_path / "mtds.pkl", "wb") as f:
                pickle.dump(mtds, f)
            io.imsave(
                self.outputs0_path / "imgs.tif", imgs0_s, check_contrast=False)
            io.imsave(
                self.outputs1_path / "imgs.tif", imgs1_s, check_contrast=False)
            
        # Execute -------------------------------------------------------------
               
        if not self.outputs0_path.exists() or self.procedure["preprocess"] == 2:
            log_str = f"\npreprocess() - {self.img_name} - df{self.df0}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.outputs0_path)
            _preprocess()  
                        
#%% Class(Main) : predict() ---------------------------------------------------
        
    def predict(self):
        
        # Predict
        for m, model_type in enumerate(self.parameters["model_types"]):
            prd_path = self.outputs0_path / f"prd{model_type[0]}.tif"
            if not prd_path.exists() or self.procedure["predict"] == 2:
                if m == 0:
                    log_str = (f"predict() - {self.img_name} - df0{self.df0}")
                    print(log_str); print('-' * len(log_str))
                prd = predict(self.imgs, model_type=model_type)
                io.imsave(
                    self.outputs0_path / f"prd{model_type[0]}.tif",
                    prd, check_contrast=False
                    )

#%% Class(Main) : process() ---------------------------------------------------

    def process(self):

        # Get masks        
        msk_path = self.outputs0_path / "mskc.tif"
        if not msk_path.exists() or self.procedure["process"] == 2:
            log_str = f"process() - {self.img_name} - df0{self.df0}"
            print(log_str); print('-' * len(log_str))
            t0 = time.time()
            print("get_mask() :", end=" ", flush=True)
            mskc = get_mask(self.prdc, *self.parameters["mask_parameters"][0])
            mskn = get_mask(self.prdn, *self.parameters["mask_parameters"][1])
            mskv = get_mask(self.prdv, *self.parameters["mask_parameters"][2])
            mskn[mskc == 0  ] = 0
            mskv[mskc == 0  ] = 0
            mskv[mskn == 255] = 0

            # Save
            io.imsave(
                self.outputs0_path / "mskc.tif", mskc, check_contrast=False)
            io.imsave(
                self.outputs0_path / "mskn.tif", mskn, check_contrast=False)
            io.imsave(
                self.outputs0_path / "mskv.tif", mskv, check_contrast=False)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")

#%% Class(Display) : ----------------------------------------------------------



#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    
#%% Development ---------------------------------------------------------------

    # # Imports
    # from skimage.transform import rescale
        
    # # Fetch
    # imgs = main.imgs
    # prdc = main.prdc
    # prdn = main.prdn
    # prdv = main.prdv
    # mskc = main.mskc
    # mskn = main.mskn
    # mskv = main.mskv
    # model_types = parameters["model_types"]
        
    # # -------------------------------------------------------------------------
    
    # # Display
    # prd_params = {
    #     "cells"    : {"colormap" : "red"     , "opacity" : 0.1},
    #     "nuclei"   : {"colormap" : "bop blue", "opacity" : 0.2},
    #     "vesicles" : {"colormap" : "yellow"  , "opacity" : 0.4},
    #     }
    # msk_params = {
    #     "cells"    : {"colormap" : "red"     , "opacity" : 0.2},
    #     "nuclei"   : {"colormap" : "bop blue", "opacity" : 0.4},
    #     "vesicles" : {"colormap" : "yellow"  , "opacity" : 0.8},
    #     }
    
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(imgs, opacity=0.33)
    # for i, prd in enumerate([prdc, prdn, prdv]):
    #     vwr.add_image(
    #         prd, name=f"prd_{model_types[i]}", visible=0,
    #         blending="additive", **prd_params[model_types[i]]
    #         ) 
    # for i, msk in enumerate([mskc, mskn, mskv]):
    #     vwr.add_image(
    #         msk, name=f"msk_{model_types[i]}", visible=1,
    #         blending="additive", **msk_params[model_types[i]]
    #         ) 
    
    