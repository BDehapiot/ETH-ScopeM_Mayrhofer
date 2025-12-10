#%% Imports -------------------------------------------------------------------

import time
import shutil
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# Functions
from functions import (
    get_rescaling_factor, rescale_image, 
    load_images, normalize_images, get_shifts, stich,
    )

#%% Inputs --------------------------------------------------------------------

dataset = "gigyf12_dko_1.7nm"

# Procedure
procedure = {
    
    "rescale" : 0,
    "prepare" : 1,
    
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
        
#%% Class(Main) : initialize() ------------------------------------------------

    def initialize(self):
        
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        self.img_paths = list(self.data_path.glob("*.tif")) 
        self.rsc_path = self.data_path / "rsc"
        
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
        print(" - Save    : ", end="", flush=True)
        
        for i, img_path in enumerate(self.img_paths):
            save_path = self.rsc_path / f"{img_path.stem}_rsc.tif"
            io.imsave(save_path, imgs[i], check_contrast=False)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class(Main) : prepare() ---------------------------------------------------

    def prepare(self):
        
        print(f"prepare() - {self.data_path.name}")
        
        # Load
        self.imgs, self.mtds = load_images(self.rsc_path)
        
        # Normalize
        self.imgs = normalize_images(self.imgs)
        
        # Get shifts ----------------------------------------------------------
        
        t0 = time.time()
        print(" - Get shifts : ", end="", flush=True)
        
        self.mtds = get_shifts(self.imgs, self.mtds)      
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Stich ---------------------------------------------------------------
        
        t0 = time.time()
        print(" - Stich      : ", end="", flush=True)
        
        self.imgs_s = stich(self.imgs, self.mtds)      
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    
#%% 

    # Fetch
    imgs = main.imgs
    mtds = main.mtds
    imgs_s = main.imgs_s
    imgs_s[imgs_s == 0] = 50
    
    # Display
    import napari
    vwr = napari.Viewer()
    vwr.add_image(imgs_s)