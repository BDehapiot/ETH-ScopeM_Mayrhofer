#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

dataset = "gigyf12_dko_3.25nm"
data_path = Path(
    rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Imports
    from collections import defaultdict
    
    # Initialize
    img_paths = list(data_path.rglob("*.tif"))
    
    # Fetch grid position
    grid = []
    for path in img_paths:   
        stem = path.stem
        row = int(stem[5:8])
        col = int(stem[9:12])
        grid.append((row, col))
    rows = np.array([t[0] for t in grid])
    cols = np.array([t[1] for t in grid])
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    unique_grid = len(set(grid))
    
    # Get duplicated positions

    # Build index map
    index_map = defaultdict(list)
    for idx, t in enumerate(grid):
        index_map[t].append(idx)
    
    # Build duplicates list
    duplicates = []
    for t, idxs in index_map.items():
        if len(idxs) > 1:
            # Pair duplicates into tuples
            for i in range(len(idxs) - 1):
                duplicates.append((t, idxs[i], idxs[i + 1]))
    
    # Now you can access duplicates like this:
    for d in range(len(duplicates)):
        print(
            img_paths[duplicates[d][1]].parent.name,
            img_paths[duplicates[d][2]].parent.name,
            )



