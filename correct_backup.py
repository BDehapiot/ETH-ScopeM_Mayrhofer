#%% Imports -------------------------------------------------------------------

import shutil
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import sync_masks

# Skimage
from skimage.morphology import skeletonize
from skimage.measure import label

# napari
import napari
from napari.layers.labels.labels import Labels

# qt
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox, QVBoxLayout

#%% Inputs --------------------------------------------------------------------

# Paths
dataset = "gigyf12_dko_1.7nm"
data_path = Path(
    rf"\\scopem-idadata.ethz.ch\BDehapiot\remote_Mayrhofer\data\{dataset}")

# Parameters
pix_ref = 27.2

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
            self.init_view()
            self.init_viewer()
            self.init_layers()
        
#%% Class(Correct) : initialize() ---------------------------------------------

    def initialize(self):
        
        self.view = 0
        self.active = "prp"
        self.labels = {"mskc" : 1, "mskn" : 2, "mskv" : 6, "mskb" : 231}
        
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

#%% Class(Correct) : init_view() ----------------------------------------------

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
        
#%% Class(Correct) : update() -------------------------------------------------
                
    def update(self):
        self.update_views()
        self.update_masks()
        self.update_labels()
        self.save()
            
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
                    
    def update_layers(self):
        self.vwr.layers["prp" ].data = self.prp
        self.vwr.layers["prdc"].data = self.prdc
        self.vwr.layers["prdn"].data = self.prdn
        self.vwr.layers["prdv"].data = self.prdv
        self.vwr.layers["mskl"].data = self.mskl
        self.vwr.layers["mskc"].data = self.mskc * self.labels["mskc"]
        self.vwr.layers["mskn"].data = self.mskn * self.labels["mskn"]
        self.vwr.layers["mskv"].data = self.mskv * self.labels["mskv"]
        self.vwr.layers["mskb"].data = self.mskb * self.labels["mskb"]
                    
    def update_masks(self):
        
        # Fetch attibutes
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                setattr(self, name, self.vwr.layers[name].data)
        
        # Synchronise masks
        sync_masks(self.mskc, self.mskn, self.mskv)
        
        # Update viewer
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                self.vwr.layers[name].data = getattr(self, name)
    
    def update_labels(self):
        self.mskl = self.mskc > 0
        mskb = self.vwr.layers["mskb"].data
        mskb = skeletonize(mskb, method="lee") * self.labels["mskb"]
        mskb[self.mskc == 0] = 0
        self.mskl[mskb != 0] = 0
        self.mskl = label(self.mskl > 0, connectivity=1).astype("uint8")
        self.vwr.layers["mskb"].data = mskb
        self.vwr.layers["mskl"].data = self.mskl
        
    def save(self):
        for name in self.vwr.layers:
            name = str(name)
            
            if "msk" in name:
            
                if   name == "mskc": 
                    save_path = self.out_path / f"msk_cells_hc_{self.view:02d}.tif"
                elif name == "mskn":
                    save_path = self.out_path / f"msk_nuclei_hc_{self.view:02d}.tif"
                elif name == "mskv":
                    save_path = self.out_path / f"msk_vesicles_hc_{self.view:02d}.tif"
                elif name == "mskb":
                    save_path = self.out_path / f"msk_bounds_hc_{self.view:02d}.tif"
                elif name == "mskl":
                    save_path = self.out_path / f"msk_labels_hc_{self.view:02d}.tif"
                
                if name == "mskl":
                    io.imsave(
                        save_path, getattr(self, name), 
                        check_contrast=False
                        )
                else:
                    io.imsave(
                        save_path, (getattr(self, name) > 0).astype("uint8"), 
                        check_contrast=False,
                        )

#%% Class(Correct) : function(s) ----------------------------------------------
                         
    def prev_view(self):
        if self.view > 0:
            self.view -= 1
            self.update_views()
            self.update()
            self.update_layers()
            # self.vwr.reset_view()
        
    def next_view(self):
        if self.view < self.nviews - 1:
            self.view += 1
            self.update_views()
            self.update()
            self.update_layers()
            # self.vwr.reset_view()
        
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
        
    # Correct -----------------------------------------------------------------
        
    def correct_mask(self, target):
        self.update()
        self.active = target
        for name in self.vwr.layers:
            name = str(name)
            if name in ["prp", self.active]:
                self.vwr.layers[name].visible = 1
                if name == "mskb":
                    self.vwr.layers["mskl"].visible = 1
            else:
                self.vwr.layers[name].visible = 0
        self.set_active()
        self.set_label()
        self.paint()
                
#%% Class(Correct) : init_viewer() --------------------------------------------

    def init_viewer(self):
                
        # Create viewer
        self.vwr = napari.Viewer()
        
        # Cteate "selection" menu
        self.slc_group_box = QGroupBox("Selection")
        slc_group_layout = QVBoxLayout()
        self.btn_prev = QPushButton("prev. view")
        self.btn_next = QPushButton("next. view")
        slc_group_layout.addWidget(self.btn_prev)
        slc_group_layout.addWidget(self.btn_next)
        self.slc_group_box.setLayout(slc_group_layout)

        # Create "actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_update = QPushButton("Update")
        self.btn_correct_c = QPushButton("Correct cell")
        self.btn_correct_n = QPushButton("Correct nuclei")
        self.btn_correct_v = QPushButton("Correct vesicles")
        self.btn_correct_b = QPushButton("Correct bounds")
        act_group_layout.addWidget(self.btn_update)
        act_group_layout.addWidget(self.btn_correct_c)
        act_group_layout.addWidget(self.btn_correct_n)
        act_group_layout.addWidget(self.btn_correct_v)
        act_group_layout.addWidget(self.btn_correct_b)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_update.clicked.connect(lambda: self.update())
        self.btn_correct_c.clicked.connect(lambda: self.correct_mask("mskc"))
        self.btn_correct_n.clicked.connect(lambda: self.correct_mask("mskn"))
        self.btn_correct_v.clicked.connect(lambda: self.correct_mask("mskv"))
        self.btn_correct_b.clicked.connect(lambda: self.correct_mask("mskb"))
                
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.slc_group_box)
        self.layout.addWidget(self.act_group_box)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter")    
        
        # Shortcuts -----------------------------------------------------------
        
        @self.vwr.bind_key("PageDown", overwrite=True)
        def prev_view_key(viewer):
            self.prev_view()
        
        @self.vwr.bind_key("PageUp", overwrite=True)
        def next_view_key(viewer):
            self.next_view()
        
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
                    self.fill()
                    yield
                    self.paint()     
                if event.button == 2:
                    self.set_label(0)
                    self.fill()
                    yield
                    self.set_label()
                    self.paint()   
            else:
                if event.button == 2:
                    self.erase()
                    yield
                    self.paint()
                    
#%% Class(Correct) : init_layers() --------------------------------------------

    def init_layers(self):  

        parameters = {
            
            "prp" : {
                "name"     : "prp",
                "visible"  : 1,
                "opacity"  : 0.6,
                },
            
            "prdc" : {
                "name"     : "prdc",
                "colormap" : "gist_earth",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 1.0,
                },
            
            "prdn" : {
                "name"     : "prdn",
                "colormap" : "gist_earth",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 1.0,
                },
            
            "prdv" : {
                "name"     : "prdv",
                "colormap" : "gist_earth",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 1.0,
                },
            
            "mskc" : {
                "name"     : "mskc",
                "blending" : "additive",
                "visible"  : 1,
                "opacity"  : 0.2,
                },
            
            "mskn" : {
                "name"     : "mskn",
                "blending" : "additive",
                "visible"  : 1,
                "opacity"  : 0.4,
                },
            
            "mskv" : {
                "name"     : "mskv",
                "blending" : "additive",
                "visible"  : 1,
                "opacity"  : 0.6,
                },
            
            "mskb" : {
                "name"     : "mskb",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 0.6,
                },
            
            "mskl" : {
                "name"     : "mskl",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 0.2,
                },
            
            }
        
        self.vwr.add_image(self.prp  , **parameters["prp" ]) 
        self.vwr.add_image(self.prdc , **parameters["prdc"])  
        self.vwr.add_image(self.prdn , **parameters["prdn"])  
        self.vwr.add_image(self.prdv , **parameters["prdv"])  
        self.vwr.add_labels(self.mskl, **parameters["mskl"])
        self.vwr.add_labels(self.mskc * self.labels["mskc"], **parameters["mskc"]) 
        self.vwr.add_labels(self.mskn * self.labels["mskn"], **parameters["mskn"])
        self.vwr.add_labels(self.mskv * self.labels["mskv"], **parameters["mskv"])
        self.vwr.add_labels(self.mskb * self.labels["mskb"], **parameters["mskb"])

        # Set default brush size
        for layer in self.vwr.layers:
            if layer.__class__.__name__ == "Labels":
                layer.brush_size = 20

        self.set_active()
        self.update()
                
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    from run import parameters, procedure
    correct = Correct(procedure=procedure, parameters=parameters)
    prps = correct.prps
    prds = correct.prds
    msks = correct.msks
    outs = correct.outs
    # mskls = correct.mskls
