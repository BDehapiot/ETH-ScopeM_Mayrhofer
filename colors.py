#%% Imports -------------------------------------------------------------------

import colorsys
import numpy as np

#%% Inputs --------------------------------------------------------------------

colors = {
    "gray"   : (128, 128, 128), # Gray
    "red"    : (160,  96,  64), # Red
    "green"  : (160, 192,  64), # Green
    "blue"   : ( 64, 128, 192), # Blue
    "yellow" : (160, 160,   0), # Yellow
    }

irange = np.arange(10, 91, 10)
irange = np.insert(irange, 0, 5)

#%% Function : get_icolors() --------------------------------------------------

def get_icolors(name, irange):
    r, g, b = colors[name]
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    color_range = {}
    for i in irange:
        lightness = (100 - i) / 100
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, lightness, s)
        color_range[f"{name}_{i:02d}"] = (
            int(new_r * 255), 
            int(new_g * 255), 
            int(new_b * 255),
            )
    return color_range

#%% Execute -------------------------------------------------------------------

icolors = {}
for key in colors:
    icolors.update(get_icolors(key, irange))
    fcolors = {k: (r/255, g/255, b/255) for k, (r, g, b) in icolors.items()}