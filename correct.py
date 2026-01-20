#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from functools import partial

# functions
from functions import sync_masks, skeletonize_junctions

# config
from config import label_config, layer_config

# Skimage
from skimage.measure import label
from skimage.morphology import remove_small_objects

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
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)  
        
        # Execute
        if self.procedure["correct"]:
            self.initialize()
            self.init_viewer()
            self.init_view()
            self.init_layers()
            self.update()
            
#%% Class(Correct) : function(s) ----------------------------------------------
        
    def change_view(self, direction):
        if direction == "prev":
            if self.view > 0:
                self.view -= 1
        if direction == "next":
            if self.view < self.nviews - 1:
                self.view += 1
        self.update_views()
        self.update_layers()
        self.update()
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
            self.vwr.layers[self.active].selected_label = label_config[self.active]
        else:
            self.vwr.layers[self.active].selected_label = value
                      
    def load_discard(self):
        path = self.out_path / f"discard_{self.view:02d}.pkl"
        if not path.exists():
            self.discard = []
            self.save_discard()
        else:
            with open(path, "rb") as file:
                self.discard = pickle.load(file)
            
    def save_discard(self):
        path = self.out_path / f"discard_{self.view:02d}.pkl"
        with open(path, "wb") as file:
            pickle.dump(self.discard, file)
            
    def correct_mask(self, target):
        self.update()
        self.active = target
        for name in self.vwr.layers:
            name = str(name)
            if name in ["prp", self.active]:
                self.vwr.layers[name].visible = 1
            else:
                self.vwr.layers[name].visible = 0
        if target == "mskj":
            self.vwr.layers["mskl"].visible = 1
        self.set_active()
        self.set_label()
        self.paint()

#%% Class(Correct) : initialize() ---------------------------------------------

    def initialize(self):
        
        self.view = 0
        self.active = "prp"

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
                partial(self.change_view, tag))
        
        # Create "action" menu
        act_group_box = QGroupBox("action")
        act_group_layout = QVBoxLayout()
        for tag0 in ["update", "correct"]:
            if tag0 == "update":
                setattr(self, f"btn_{tag0}", QPushButton(f"{tag0}"))
                act_group_layout.addWidget(getattr(self, f"btn_{tag0}"))
                getattr(self, f"btn_{tag0}").clicked.connect(
                    partial(self.update))
            else:
                for tag1 in ["cells", "nuclei", "vesicles", "junctions"]:
                    setattr(self, f"btn_{tag0}{tag1[0]}", QPushButton(f"{tag0} {tag1}"))
                    act_group_layout.addWidget(getattr(self, f"btn_{tag0}{tag1[0]}"))
                    getattr(self, f"btn_{tag0}{tag1[0]}").clicked.connect(
                        partial(self.correct_mask, f"msk{tag1[0]}"))
        
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

        @self.vwr.bind_key("PageDown", overwrite=True)
        def prev_view_key(viewer):
            self.change_view("prev")
        
        @self.vwr.bind_key("PageUp", overwrite=True)
        def next_view_key(viewer):
            self.change_view("next")
        
        @self.vwr.bind_key("Right", overwrite=True)
        def next_brush_size_key(viewer):
            self.next_brush_size() 
            self.next_brush_size_timer.start(30) 
            yield
            self.next_brush_size_timer.stop()
            
        @self.vwr.bind_key("Left", overwrite=True)
        def prev_brush_size_key(viewer):
            self.prev_brush_size() 
            self.prev_brush_size_timer.start(30) 
            yield
            self.prev_brush_size_timer.stop()
            
        @Labels.bind_key("Enter", overwrite=True)
        def update_key(viewer):
            self.update() 
            
        @self.vwr.mouse_drag_callbacks.append
        def mouse_actions(vwr, event):
            if "Control" in event.modifiers:
                if event.button == 1:
                    if self.active == "mskj":
                        self.vwr.layers["mskj"].mode = "pan_zoom"
                        coords = np.round(event.position).astype(int)
                        lbl = self.mskl_all[tuple(coords)]
                        if lbl in self.discard:
                            self.discard.remove(lbl)
                        self.save_discard()
                    else:
                        self.fill()
                    self.update()
                    yield
                    self.paint()
                if event.button == 2:
                    if self.active == "mskj":
                        self.vwr.layers["mskj"].mode = "pan_zoom"
                        coords = np.round(event.position).astype(int)
                        lbl = self.mskl_all[tuple(coords)]
                        if lbl not in self.discard:
                            self.discard.append(lbl)
                        self.save_discard()
                    else:
                        self.set_label(0)
                        self.fill()
                    self.update()
                    yield
                    self.set_label()
                    self.paint()
            else:
                if event.button == 2:
                    self.erase()
                    yield
                    self.paint()

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
                elif tag == "prd" and self.load_prd:
                    tmp_dict = {}
                    for path in getattr(self, f"{tag}_img_paths"):
                        if f"{view:02d}" in path.name:
                            tmp_dict[f"{path.stem.split('_')[1]}"] = io.imread(path)
                    getattr(self, f"{tag}s").append(tmp_dict)
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
                self.outs[view]["junctions"] = np.zeros_like(self.msks[view]["cells"])
                self.outs[view]["labels"] = label(self.msks[view]["cells"])
        
        self.update_views()
        
    def update_views(self):
        
        self.prp = self.prps[self.view]
        for tag0 in ["prd", "out"]:
            if tag0 == "prd" and self.load_prd:
                for tag1 in ["cells", "nuclei", "vesicles"]:
                    setattr(
                        self, f"{tag0}{tag1[0]}", 
                        getattr(self, f"{tag0}s")[self.view][tag1],
                        )
            elif tag0 == "out":
                for tag1 in ["cells", "nuclei", "vesicles", "junctions", "labels"]:
                    setattr(
                        self, f"msk{tag1[0]}", 
                        getattr(self, f"{tag0}s")[self.view][tag1],
                        )
             
#%% Class(Correct) : layers ---------------------------------------------------

    def init_layers(self):  
        
        for tag0 in ["prp", "prd", "msk"]:
            if tag0 == "prp":
                self.vwr.add_image(
                    getattr(self, tag0), 
                    **layer_config[tag0]
                    ) 
            elif tag0 == "prd" and self.load_prd:
                for tag1 in ["cells", "nuclei", "vesicles"]:
                    self.vwr.add_image(
                        getattr(self, f"{tag0}{tag1[0]}"),
                        **layer_config[f"{tag0}{tag1[0]}"]
                        ) 
            elif tag0 == "msk":
                for tag1 in ["cells", "nuclei", "vesicles", "junctions", "labels"]:
                    self.vwr.add_labels(
                        getattr(self, f"{tag0}{tag1[0]}") 
                        * label_config[f"{tag0}{tag1[0]}"], 
                        **layer_config[f"{tag0}{tag1[0]}"],
                        ) 

        # Set default brush size
        for layer in self.vwr.layers:
            if layer.__class__.__name__ == "Labels":
                layer.brush_size = 20

        self.vwr.reset_view()
        self.set_active()
        
    def update_layers(self):
        
        for tag0 in ["prp", "prd", "msk"]:
            if tag0 == "prp":
                self.vwr.layers["prp"].data = self.prp
            elif tag0 == "prd" and self.load_prd:
                for tag1 in ["cells", "nuclei", "vesicles"]:
                    self.vwr.layers[f"{tag0}{tag1[0]}"].data = (
                        getattr(self, f"{tag0}{tag1[0]}"))
            elif tag0 == "msk":
                for tag1 in ["cells", "nuclei", "vesicles", "junctions", "labels"]:
                    self.vwr.layers[f"{tag0}{tag1[0]}"].data = (
                        getattr(self, f"{tag0}{tag1[0]}") 
                        * label_config[f"{tag0}{tag1[0]}"]
                        )
                    
#%% Class(Correct) : update ---------------------------------------------------

    def update(self):
        self.update_masks()
        self.update_labels()
        self.save_hc()

    def update_masks(self):
        for name in self.vwr.layers:
            name = str(name)
            if name in ["mskn", "mskv", "mskj"]:
                setattr(self, name, self.vwr.layers[name].data > 0)
            else:
                setattr(self, name, self.vwr.layers[name].data)
        sync_masks(self.mskc, self.mskn, self.mskv)
        self.update_layers()
    
    def update_labels(self):
        
        self.mskl = self.mskc > 0
        self.mskj = self.vwr.layers["mskj"].data
        self.mskj = skeletonize_junctions(self.mskj, 10) * label_config["mskj"]                                 
        self.mskj[self.mskc == 0] = 0
        self.mskl[self.mskj != 0] = 0
        self.mskl = label(self.mskl > 0, connectivity=1).astype("uint8")
        self.mskl = remove_small_objects(
            self.mskl, min_size=self.parameters["mask_params"]["cells"][1])
        self.mskl_all = self.mskl.copy()
                
        self.load_discard()
        for lbl in np.unique(self.mskl):
            if lbl in self.discard:
                self.mskl[self.mskl == lbl] = 0
        
        self.vwr.layers["mskj"].data = self.mskj
        self.vwr.layers["mskl"].data = self.mskl

    def save_hc(self):
        for name in self.vwr.layers:
            name = str(name)
            view = f"{self.view:02d}"
            for tag in ["cells", "nuclei", "vesicles", "junctions", "labels"]:
                if name == f"msk{tag[0]}":
                    save_path = self.out_path / f"msk_{tag}_hc_{view}.tif"
                    if name == "mskl":
                        self.outs[self.view][tag] = getattr(self, name)
                    else:
                        self.outs[self.view][tag] = (getattr(self, name) > 0).astype("uint8")
                    io.imsave(
                        save_path, self.outs[self.view][tag], 
                        check_contrast=False,
                        )
           
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from run import parameters, procedure
    correct = Correct(procedure=procedure, parameters=parameters)
    prps = correct.prps
    prds = correct.prds
    msks = correct.msks
    outs = correct.outs
    discard = correct.discard