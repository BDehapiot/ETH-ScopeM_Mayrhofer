#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# functions
from functions import prepare

#%% Inputs --------------------------------------------------------------------

# Paths
dataset = "gigyf12_dko_1.7nm"
data_path = Path(
    rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")

# Parameters
pix_ref = 27.2

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    prepare(data_path, pix_ref=pix_ref, parallel=True)

#%%    

    # # Initialize
    # img_paths = list(data_path.rglob("*.tif"))
    
    # # Grid positions
    # grid = []
    # for path in img_paths:   
    #     stem = path.stem
    #     row = int(stem[5:8])
    #     col = int(stem[9:12])
    #     grid.append((row, col))
    # rows = np.array([t[0] for t in grid])
    # cols = np.array([t[1] for t in grid])
    # min_row, max_row = rows.min(), rows.max()
    # min_col, max_col = cols.min(), cols.max()
    # nrows = max_row - min_row
    # ncols = max_col - min_col
    # if len(grid) > len(set(grid)):
    #     print("duplicated positions found")
        
    # # 
    # r0s = np.arange(min_row, max_row, ntiles)
    # c0s = np.arange(min_col, max_col, ntiles)
    # for r, r0 in enumerate(r0s):
    #     for c, c0 in enumerate(c0s):
    #         pass                
