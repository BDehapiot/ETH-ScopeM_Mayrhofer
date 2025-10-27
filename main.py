#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import (
    clear_directory,
    downscale_images, load_images, get_shift, stich, predict
    )

# bdtools
from bdtools.norm import norm_pct

#%% Inputs --------------------------------------------------------------------

procedure = {
    "downscale"  : 0,
    "preprocess" : 0,
    "predict"    : 0,
    "process"    : 0,
    }

parameters = {
    
    # Paths
    "img_name"  : "Ins1e_wt_1.7nm_00",
    "data_path" : Path("D:\local_Mayrhofer\data"),
    
    # Downscale
    "df" : 16,
    
    # Predict
    "model_types" : ["cells", "nuclei", "vesicles"],
    
    # Process
    "threshs" : [0.5, 0.5, 0.5],
    
    }

#%% Class(Main) : -------------------------------------------------------------

class Main:

    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.df = parameters["df"]

        # Initialize
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level_path = self.data_path / f"level-{parameters['df']}"
        self.outputs_path = self.level_path / "outputs"
        
        # Run
        if self.procedure["downscale" ]: self.downscale() 
        if self.procedure["preprocess"]: self.preprocess() 
        if self.procedure["predict"   ]: self.predict()
        
#%% Class(Main) : downscale() -------------------------------------------------

    def downscale(self):

        if not self.level_path.exists() or self.procedure["downscale"] == 2:
            log_str = f"downscale() - {self.img_name} - df{self.df}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.level_path)        
            downscale_images(self.data_path, df=self.df)

#%% Class(Main) : preprocess() ------------------------------------------------

    def preprocess(self):
        
        # Nested funtion(s) ---------------------------------------------------
        
        def _preprocess():
            
            # Setup outputs path
            self.outputs_path.mkdir(parents=True, exist_ok=True)
            
            # Load images
            imgs, mtds = load_images(
                self.data_path, df=self.df, suffix="", return_metadata=True)
                
            # Get shifts
            mtds = get_shift(imgs, mtds)
            
            # Manual normalization
            imgs = np.stack(imgs)
            imgs = imgs.astype("float32")
            imgs = norm_pct(imgs, pct_low=1, pct_high=99, mask=imgs > 0)
            
            # Stich
            imgs_s = stich(imgs, mtds)  
            
            # Save
            with open(self.outputs_path / "mtds.pkl", "wb") as f:
                pickle.dump(mtds, f)
            io.imsave(
                self.outputs_path / "imgs_s.tif", imgs_s, check_contrast=False)
            
        # Execute -------------------------------------------------------------
               
        if not self.outputs_path.exists() or self.procedure["preprocess"] == 2:
            log_str = f"preprocess() - {self.img_name} - df{self.df}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.outputs_path)
            _preprocess()       
                        
#%% Class(Main) : predict() ---------------------------------------------------
        
    def predict(self):

        # Load image
        imgs_s = io.imread(self.outputs_path / "imgs_s.tif")
        
        # Predict
        for m, model_type in enumerate(self.parameters["model_types"]):
            prd_path = self.outputs_path / f"prd-{model_type}.tif"
            if not prd_path.exists() or self.procedure["predict"] == 2:
                if m == 0:
                    log_str = (f"predict() - {self.img_name} - df{self.df}")
                    print(log_str); print('-' * len(log_str))
                prd = predict(imgs_s, model_type=model_type)
                io.imsave(
                    self.outputs_path / f"prd_{model_type}.tif",
                    prd, check_contrast=False
                    )

#%% Class(Main) : predict() ---------------------------------------------------

    def process(self):
        
        # load predictions
        prd_c = io.imread(self.outputs_path / "prd_cells.tif")
        prd_n = io.imread(self.outputs_path / "prd_nuclei.tif")
        prd_v = io.imread(self.outputs_path / "prd_vesicles.tif")
            
        # Get masks
        msk_c = prd_c > self.parameters["threshs"][0]
        msk_n = prd_n > self.parameters["threshs"][1]
        msk_v = prd_v > self.parameters["threshs"][2]
        
        pass

#%% Class(Display) : ----------------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    Main(procedure=procedure, parameters=parameters)
    