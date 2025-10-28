#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import (
    clear_directory,
    downscale_images, load_images, custom_normalization, 
    get_shift, stich, predict, get_mask,
    )

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QGroupBox, QVBoxLayout,
    )

#%% Inputs --------------------------------------------------------------------

procedure = {
    "downscale"  : 1,
    "preprocess" : 1,
    "predict"    : 1,
    "process"    : 1,
    "annotate"   : 1,
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
    "mask_parameters" : [
        (0.5, 4096, 32),
        (0.5, 512, 32),
        (0.5, 16, 8),
        ],   
    
    # Annotate
    "labels" : [1, 2, 6, 8],
    
    }

#%% Class(Main) : -------------------------------------------------------------

class Main:

    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.df = parameters["df"]
        
        # Run
        self.initialize()
        if self.procedure["downscale" ]: self.downscale() 
        if self.procedure["preprocess"]: self.preprocess() 
        if self.procedure["predict"   ]: self.predict()
        if self.procedure["process"   ]: self.process()
        
#%% Class(Main) : initialize() ------------------------------------------------
        
    def initialize(self):
        
        # Paths
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level_path = self.data_path / f"level-{parameters['df']}"
        self.outputs_path = self.level_path / "outputs"
        
        # Files
        file_map = {
            "imgs": ("imgs.tif", io.imread),
            "mtds": ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            "prdc": ("prdc.tif", io.imread),
            "prdn": ("prdn.tif", io.imread),
            "prdv": ("prdv.tif", io.imread),
            "mskc": ("mskc.tif", io.imread),
            "mskn": ("mskn.tif", io.imread),
            "mskv": ("mskv.tif", io.imread),
            }
        
        for attr, (filename, loader) in file_map.items():
            path = self.outputs_path / filename
            if path.is_file():
                setattr(self, attr, loader(path))
                
#%% Class(Main) : downscale() -------------------------------------------------

    def downscale(self):

        self.initialize()
        
        if not self.level_path.exists() or self.procedure["downscale"] == 2:
            log_str = f"\ndownscale() - {self.img_name} - df{self.df}"
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
            
            # Custom normalization
            imgs = custom_normalization(imgs)
            
            # Stich
            imgs_s = stich(imgs, mtds, scaling_coeff=1)  
            
            # Save
            with open(self.outputs_path / "mtds.pkl", "wb") as f:
                pickle.dump(mtds, f)
            io.imsave(
                self.outputs_path / "imgs.tif", imgs_s, check_contrast=False)
            
        # Execute -------------------------------------------------------------
               
        self.initialize()
        
        if not self.outputs_path.exists() or self.procedure["preprocess"] == 2:
            log_str = f"\npreprocess() - {self.img_name} - df{self.df}"
            print(log_str); print('-' * len(log_str))
            clear_directory(self.outputs_path)
            _preprocess()  
                        
#%% Class(Main) : predict() ---------------------------------------------------
        
    def predict(self):
        
        self.initialize()
        
        # Predict
        for m, model_type in enumerate(self.parameters["model_types"]):
            prd_path = self.outputs_path / f"prd{model_type[0]}.tif"
            if not prd_path.exists() or self.procedure["predict"] == 2:
                if m == 0:
                    log_str = (f"predict() - {self.img_name} - df{self.df}")
                    print(log_str); print('-' * len(log_str))
                prd = predict(self.imgs, model_type=model_type)
                io.imsave(
                    self.outputs_path / f"prd{model_type[0]}.tif",
                    prd, check_contrast=False
                    )

#%% Class(Main) : process() ---------------------------------------------------

    def process(self):

        self.initialize()
        
        # Get masks        
        msk_path = self.outputs_path / "mskc.tif"
        if not msk_path.exists() or self.procedure["process"] == 2:
            log_str = f"process() - {self.img_name} - df{self.df}"
            print(log_str); print('-' * len(log_str))
            t0 = time.time()
            print("get_mask() :", end=" ", flush=True)
            mskc = get_mask(self.prdc, *self.parameters["mask_parameters"][0])
            mskn = get_mask(self.prdn, *self.parameters["mask_parameters"][1])
            mskv = get_mask(self.prdv, *self.parameters["mask_parameters"][2])
            mskn[mskc == 0  ] = 0
            mskv[mskc == 0  ] = 0
            mskv[mskn == 255] = 0

            # Save
            io.imsave(
                self.outputs_path / "mskc.tif", mskc, check_contrast=False)
            io.imsave(
                self.outputs_path / "mskn.tif", mskn, check_contrast=False)
            io.imsave(
                self.outputs_path / "mskv.tif", mskv, check_contrast=False)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")

#%% Class(Annotate) : ---------------------------------------------------------

class Annotate:
    
    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.df = parameters["df"]
        
        # Timers
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)        
        
        # Run
        self.initialize()
        self.init_viewer()
        self.init_layers()
        
#%% Class(Annotate) : initialize() --------------------------------------------

    def initialize(self):
        
        # Paths
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level_path = self.data_path / f"level-{parameters['df']}"
        self.outputs_path = self.level_path / "outputs"
                
        # Files
        file_map = {
            "imgs": ("imgs.tif", io.imread),
            "mtds": ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            "prdc": ("prdc.tif", io.imread),
            "prdn": ("prdn.tif", io.imread),
            "prdv": ("prdv.tif", io.imread),
            "mskc": ("mskc.tif", io.imread),
            "mskn": ("mskn.tif", io.imread),
            "mskv": ("mskv.tif", io.imread),
            }
        
        for attr, (filename, loader) in file_map.items():
            path = self.outputs_path / filename
            if path.is_file():
                setattr(self, attr, loader(path))
                
        # Variables
        self.active = "imgs"
        self.labels = {
            "mskc" : parameters["labels"][0], 
            "mskn" : parameters["labels"][1], 
            "mskv" : parameters["labels"][2],
            "mskb" : parameters["labels"][3],
            }
        self.mskb = np.zeros_like(self.mskc)
                
#%% Class(Annotate) : function(s) ---------------------------------------------
                               
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
        
    def correct_mask(self, target):
        self.active = target
        for name in self.vwr.layers:
            name = str(name)
            if name in ["imgs", self.active]:
                self.vwr.layers[name].visible = 1
            else:
                self.vwr.layers[name].visible = 0
        self.set_active()
        self.set_label()
        self.paint()

#%% Class(Annotate) : init_viewer() -------------------------------------------

    def init_viewer(self):
                
        # Create viewer
        self.vwr = napari.Viewer()

        # Create "actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_correct_c = QPushButton("Correct cell")
        self.btn_correct_n = QPushButton("Correct nuclei")
        self.btn_correct_v = QPushButton("Correct vesicles")
        self.btn_correct_b = QPushButton("Correct bounds")
        act_group_layout.addWidget(self.btn_correct_c)
        act_group_layout.addWidget(self.btn_correct_n)
        act_group_layout.addWidget(self.btn_correct_v)
        act_group_layout.addWidget(self.btn_correct_b)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_correct_c.clicked.connect(lambda: self.correct_mask("mskc"))
        self.btn_correct_n.clicked.connect(lambda: self.correct_mask("mskn"))
        self.btn_correct_v.clicked.connect(lambda: self.correct_mask("mskv"))
        self.btn_correct_b.clicked.connect(lambda: self.correct_mask("mskb"))
                
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.act_group_box)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.vwr.window.add_dock_widget(
            self.widget, area="right", name="Painter")    
        
        # Shortcuts
        
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
        
#%% Class(Annotate) : init_layers() -------------------------------------------

    def init_layers(self):  

        parameters = {
            
            "imgs" : {
                "name"     : "imgs",
                "visible"  : 1,
                "opacity"  : 0.5,
                },
            
            "mskc" : {
                "name"     : "mskc",
                "visible"  : 1,
                "opacity"  : 0.2,
                "blending" : "additive",
                },
            
            "mskn" : {
                "name"     : "mskn",
                "visible"  : 1,
                "opacity"  : 0.4,
                "blending" : "additive",
                },
            
            "mskv" : {
                "name"     : "mskv",
                "visible"  : 1,
                "opacity"  : 0.8,
                "blending" : "additive",
                },
            
            "mskb" : {
                "name"     : "mskb",
                "visible"  : 1,
                "opacity"  : 0.8,
                "blending" : "additive",
                },
            
            }
        
        self.vwr.add_image(self.imgs , **parameters["imgs"])  
        self.vwr.add_labels((self.mskc // 255 ) * 1, **parameters["mskc"]) 
        self.vwr.add_labels((self.mskn // 255 ) * 2, **parameters["mskn"])
        self.vwr.add_labels((self.mskv // 255 ) * 6, **parameters["mskv"])
        self.vwr.add_labels(self.mskb, **parameters["mskb"])
        self.set_active()
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    annotate = Annotate(procedure=procedure, parameters=parameters)
    
#%% Development ---------------------------------------------------------------

    # # Imports
    # from skimage.transform import rescale
        
    # # Fetch
    # imgs = main.imgs
    # prdc = main.prdc
    # prdn = main.prdn
    # prdv = main.prdv
    # mskc = main.mskc
    # mskn = main.mskn
    # mskv = main.mskv
    # model_types = parameters["model_types"]
        
    # # -------------------------------------------------------------------------
    
    # # Display
    # prd_params = {
    #     "cells"    : {"colormap" : "red"     , "opacity" : 0.1},
    #     "nuclei"   : {"colormap" : "bop blue", "opacity" : 0.2},
    #     "vesicles" : {"colormap" : "yellow"  , "opacity" : 0.4},
    #     }
    # msk_params = {
    #     "cells"    : {"colormap" : "red"     , "opacity" : 0.2},
    #     "nuclei"   : {"colormap" : "bop blue", "opacity" : 0.4},
    #     "vesicles" : {"colormap" : "yellow"  , "opacity" : 0.8},
    #     }
    
    # vwr = napari.Viewer()
    # vwr.add_image(imgs, opacity=0.33)
    # for i, prd in enumerate([prdc, prdn, prdv]):
    #     vwr.add_image(
    #         prd, name=f"prd_{model_types[i]}", visible=0,
    #         blending="additive", **prd_params[model_types[i]]
    #         ) 
    # for i, msk in enumerate([mskc, mskn, mskv]):
    #     vwr.add_image(
    #         msk, name=f"msk_{model_types[i]}", visible=1,
    #         blending="additive", **msk_params[model_types[i]]
    #         ) 
    
    