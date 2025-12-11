#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import sync_masks

# config
from config import layer_config

# Skimage
from skimage.morphology import skeletonize
from skimage.measure import label

# napari
import napari
from napari.layers.labels.labels import Labels

# qt
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox, QVBoxLayout

#%% Class(Correct) ------------------------------------------------------------

class Correct:
    
    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters

        # Timers
        # self.next_brush_size_timer = QTimer()
        # self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        # self.prev_brush_size_timer = QTimer()
        # self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)  
        
        # Execute
        if self.procedure["correct"]:
            self.initialize()
            self.init_viewer()
            self.init_view()
            self.init_layers()
            
#%% Class(Correct) : function(s) ----------------------------------------------
        
    def change_view(self, direction):
        if direction == "next":
            if self.view < self.nviews - 1:
                self.view += 1
        if direction == "prev":
            if self.view > 0:
                self.view -= 1
        self.update_views()
        self.update_layers()
        self.vwr.reset_view()
        
    def paint(self):
        self.vwr.layers[self.active].mode = "paint"
                    
    def erase(self):
        self.vwr.layers[self.active].mode = "erase"
        
    def fill(self):
        self.vwr.layers[self.active].mode = "fill"
        
    def prev_brush_size(self):
        if self.vwr.layers[self.active].brush_size > 1:
            self.vwr.layers[self.active].brush_size -= 1
        
    def next_brush_size(self): 
        self.vwr.layers[self.active].brush_size += 1
        
    def set_active(self):
        self.vwr.layers.selection.active = self.vwr.layers[self.active]
    
    def set_label(self, value=None):
        if value is None:
            self.vwr.layers[self.active].selected_label = self.labels[self.active]
        else:
            self.vwr.layers[self.active].selected_label = value

#%% Class(Correct) : initialize() ---------------------------------------------

    def initialize(self):
        
        self.view = 0
        self.active = "prp"
        self.labels = {
            "prp"  : 1,
            "prdc" : 1, "prdn" : 1, "prdv" : 1,
            "mskc" : 1, "mskn" : 2, "mskv" : 6, "mskb" : 231, "mskl" : 1,
            }
        
        for key, val in self.parameters.items():
            if not isinstance(val, dict):
                setattr(self, key, val)
                
        # Paths
        self.out_path = self.data_path / "out"
        if not self.out_path.exists():
            self.out_path.mkdir(parents=True, exist_ok=True)
        self.raw_img_paths = list(self.data_path.glob("*.tif")) 
        for tag in ["prp", "prd", "msk", "out"]:
            setattr(self, f"{tag}_path", self.data_path / f"{tag}") 
            img_paths = list(getattr(self, f"{tag}_path").glob("*.tif"))
            setattr(self, f"{tag}_img_paths", img_paths) 

#%% Class(Correct) : viewer ---------------------------------------------------

    def init_viewer(self):

        # Create viewer
        self.vwr = napari.Viewer()
        
        # Create "selection" menu
        slc_group_box = QGroupBox("selection")
        slc_group_layout = QVBoxLayout()
        for tag in ["prev", "next"]:
            setattr(self, f"btn_{tag}", QPushButton(f"{tag}. view"))
            slc_group_layout.addWidget(getattr(self, f"btn_{tag}"))
            getattr(self, f"btn_{tag}").clicked.connect(
                lambda: self.change_view(tag))
        
        # Create "action" menu
        act_group_box = QGroupBox("action")
        act_group_layout = QVBoxLayout()
        for tag0 in ["update", "correct"]:
            if tag0 == "update":
                setattr(self, f"btn_{tag0}", QPushButton(f"{tag0}"))
                act_group_layout.addWidget(getattr(self, f"btn_{tag0}"))
                getattr(self, f"btn_{tag0}").clicked.connect(
                    lambda: self.update())
            else:
                for tag1 in ["cells", "nuclei", "vesicles", "bounds"]:
                    setattr(self, f"btn_{tag0}{tag1[0]}", QPushButton(f"{tag0} {tag1}"))
                    act_group_layout.addWidget(getattr(self, f"btn_{tag0}{tag1[0]}"))
                    getattr(self, f"btn_{tag0}{tag1[0]}").clicked.connect(
                        lambda: self.correct_mask(f"{tag0}{tag1[0]}"))
        
        # Create layout
        self.layout = QVBoxLayout()
        slc_group_box.setLayout(slc_group_layout)
        act_group_box.setLayout(act_group_layout)
        self.layout.addWidget(slc_group_box)
        self.layout.addWidget(act_group_box)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter")    
        
        # Shortcuts -----------------------------------------------------------

#%% Class(Correct) : view -----------------------------------------------------

    def init_view(self):

        self.nviews = len(self.prp_img_paths)
        
        # Load data
        for tag in ["prp", "prd", "msk", "out"]:
            setattr(self, f"{tag}s", []) 
            for view in range(self.nviews):
                if tag == "prp":
                    for path in getattr(self, f"{tag}_img_paths"):
                        if f"{view:02d}" in path.name:
                            getattr(self, f"{tag}s").append(io.imread(path))
                else:
                    tmp_dict = {}
                    for path in getattr(self, f"{tag}_img_paths"):
                        if f"{view:02d}" in path.name:
                            tmp_dict[f"{path.stem.split('_')[1]}"] = io.imread(path)
                    getattr(self, f"{tag}s").append(tmp_dict)
         
        # Fill out data if empty
        for view in range(self.nviews):
            if not self.outs[view]:
                for tag in ["cells", "nuclei", "vesicles"]:
                    self.outs[view][tag] = self.msks[view][tag]
                self.outs[view]["bounds"] = np.zeros_like(self.msks[view]["cells"])
                self.outs[view]["labels"] = label(self.msks[view]["cells"])
        
        self.update_views()
        
    def update_views(self):
        
        self.prp = self.prps[self.view]
        for tag0 in ["prd", "out"]:
            if tag0 == "prd":
                for tag1 in ["cells", "nuclei", "vesicles"]:
                    setattr(
                        self, f"{tag0}{tag1[0]}", 
                        getattr(self, f"{tag0}s")[self.view][tag1],
                        )
            elif tag0 == "out":
                for tag1 in ["cells", "nuclei", "vesicles", "bounds", "labels"]:
                    setattr(
                        self, f"msk{tag1[0]}", 
                        getattr(self, f"{tag0}s")[self.view][tag1],
                        )


                
#%% Class(Correct) : init_layers() --------------------------------------------

    def init_layers(self):  
        
        for tag0 in ["prp", "prd", "msk"]:
            if tag0 == "prp":
                self.vwr.add_image(
                    getattr(self, tag0), 
                    **layer_config[tag0]
                    ) 
            elif tag0 == "prd":
                for tag1 in ["cells", "nuclei", "vesicles"]:
                    self.vwr.add_image(
                        getattr(self, f"{tag0}{tag1[0]}"), 
                        **layer_config[f"{tag0}{tag1[0]}"]
                        ) 
            elif tag0 == "msk":
                for tag1 in ["cells", "nuclei", "vesicles", "bounds", "labels"]:
                    self.vwr.add_image(
                        getattr(self, f"{tag0}{tag1[0]}") * self.labels[f"{tag0}{tag1[0]}"], 
                        **layer_config[f"{tag0}{tag1[0]}"]
                        ) 

        # Set default brush size
        for layer in self.vwr.layers:
            if layer.__class__.__name__ == "Labels":
                layer.brush_size = 20

        self.set_active()
        
    def update_layers(self):
        
        for tag0 in ["prp", "prd", "msk"]:
            if tag0 == "prp":
                self.vwr.layers["prp"].data = self.prp
            
        
        self.vwr.layers["prp" ].data = self.prp
        self.vwr.layers["prdc"].data = self.prdc
        self.vwr.layers["prdn"].data = self.prdn
        self.vwr.layers["prdv"].data = self.prdv
        self.vwr.layers["mskl"].data = self.mskl
        self.vwr.layers["mskc"].data = self.mskc * self.labels["mskc"]
        self.vwr.layers["mskn"].data = self.mskn * self.labels["mskn"]
        self.vwr.layers["mskv"].data = self.mskv * self.labels["mskv"]
        self.vwr.layers["mskb"].data = self.mskb * self.labels["mskb"]
        
        
        pass
        
                    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from run import parameters, procedure
    correct = Correct(procedure=procedure, parameters=parameters)
    prps = correct.prps
    prds = correct.prds
    msks = correct.msks
    outs = correct.outs