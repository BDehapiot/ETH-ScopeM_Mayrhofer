#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path
 
# skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

# dataset = "gigyf12_dko_1.7nm"
# dataset = "ins1e_wt_1.7nm"
# dataset = "gigyf12_dko_3.25nm_00"
# dataset = "gigyf12_dko_3.25nm_01"
# dataset = "ins1e_wt_3.25nm_00"
dataset = "ins1e_wt_3.25nm_01"
data_path = Path(
    rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")
rsc_path = Path(data_path, "rsc")
rsc_img_paths = rsc_path.glob("*.tif")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
        
    pass
    
    
    