#%% Imports -------------------------------------------------------------------

from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# dataset = "gigyf12_dko_1.7nm"
# dataset = "ins1e_wt_1.7nm"
dataset = "gigyf12_dko_3.25nm_00"
# dataset = "gigyf12_dko_3.25nm_01"
# dataset = "ins1e_wt_3.25nm_00"
# dataset = "ins1e_wt_3.25nm_01"
data_path = Path(rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
        
    for dir_path in data_path.iterdir():
        if dir_path.is_dir():
            img_paths = list(dir_path.glob("*.tif"))
            rows, cols = [], []
            for img_path in img_paths:
                rows.append(int(img_path.stem[5:8]))
                cols.append(int(img_path.stem[9:12]))
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)
            print(
                f"{dir_path.stem}\n"
                f"min_row = {min_row} ; max_row = {max_row}\n"
                f"min_col = {min_col} ; max_col = {max_col}\n"
                )
    
    
    