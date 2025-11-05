#%% Imports -------------------------------------------------------------------

import time
import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path

# functions
from functions import (
    clear_directory,
    downscale_images, load_images, custom_normalization, 
    get_shift, stich, predict, get_mask, sync_masks,
    binned_distribution,
    )

# skimage
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

# napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QGroupBox, QVBoxLayout,
    )

# matplot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Inputs --------------------------------------------------------------------

procedure = {
    "downscale"  : 0,
    "preprocess" : 0,
    "predict"    : 0,
    "process"    : 0,
    "correct"    : 0,
    "analyse"    : 0,
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
        # 1) prediction threshold
        # 2) minimum object size
        # 3) minimum hole size
        (0.5, 4096, 32), # cells
        (0.5, 512, 32),  # nuclei
        (0.5, 16, 8),    # vesicles
        ],   
    
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
        if self.procedure["correct"   ]: self.correct()
        if self.procedure["analyse"   ]: self.analyse()
        
#%% Class(Main) : initialize() ------------------------------------------------
        
    def initialize(self):
        
        # Paths
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level_path = self.data_path / f"level-{parameters['df']}"
        self.outputs_path = self.level_path / "outputs"
        
        # Variables
        parts = self.img_name.split("_")
        for part in parts:
            if "nm" in part:
                self.pixel_size = float(part.replace("nm", ""))
                self.pixel_size = self.pixel_size * self.df * 1e-3
        
        # Files
        file_map = {
            
            # Image
            "imgs": ("imgs.tif", io.imread),
            "mtds": ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            
            # Predictions
            "prdc": ("prdc.tif", io.imread),
            "prdn": ("prdn.tif", io.imread),
            "prdv": ("prdv.tif", io.imread),
            
            # Masks
            "mskc": ("mskc.tif", io.imread),
            "mskn": ("mskn.tif", io.imread),
            "mskv": ("mskv.tif", io.imread),
            
            # Hand corrections
            "mskc_hc": ("mskc_hc.tif", io.imread),
            "mskn_hc": ("mskn_hc.tif", io.imread),
            "mskv_hc": ("mskv_hc.tif", io.imread),
            "mskb_hc": ("mskb_hc.tif", io.imread),
            "lblc_hc": ("lblc_hc.tif", io.imread),
            
            # Results
            "results": ("results.pkl", lambda p: pickle.load(open(p, "rb"))),
            
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
            imgs, mtds = load_images(self.data_path, df=self.df)
                
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
            sync_masks(mskc, mskn, mskv)
            
            t1 = time.time()
            print(f"{t1 - t0:.3f}s")

            # Save
            io.imsave(
                self.outputs_path / "mskc.tif", mskc, check_contrast=False)
            io.imsave(
                self.outputs_path / "mskn.tif", mskn, check_contrast=False)
            io.imsave(
                self.outputs_path / "mskv.tif", mskv, check_contrast=False)

#%% Class(Main) : Correct() ---------------------------------------------------

    def correct(self):
        Correct(procedure=self.procedure, parameters=self.parameters)

#%% Class(Main) : analyse() --------------------------------------------------- 
    
    def analyse(self):
        
        # Nested funtion(s) ---------------------------------------------------

        def plot(results):
            
            # # Get data
            # area_bdist = binned_distribution(
            #     results["dist"], y=results["area"], bin_width=2)
            # int_bdist  = binned_distribution(
            #     results["dist"], y=results["int" ], bin_width=2)
            
            # Initialize plot
            fig = plt.figure(figsize=(6, 9))
            gs = fig.add_gridspec(3, 1)
            ax0 = fig.add_subplot(gs[0, 0])  # Distances
            ax1 = fig.add_subplot(gs[1, 0])  # Areas
            ax2 = fig.add_subplot(gs[2, 0])  # Intensities
            
            # Distances (ax0)
            ax0.hist(results["dist"], bins=64, color="lightgray") 

            # Areas (ax1)
            ax1.hist(results["area"], bins=64, color="lightgray") 
            ax1_inset = inset_axes(
                ax1, width="40%", height="40%", loc="upper right")
            
            # Intensities (ax2)
            ax2.hist(results["int" ], bins=64, color="lightgray") 
            
            pass
        
        # Execute -------------------------------------------------------------
        
        self.initialize()
        
        # Extract measurments
        self.results = {
            "name"       : [],
            "df"         : [],
            "pixel_size" : [],
            "label"      : [],
            "area"       : [], 
            "dist"       : [],
            "int"        : [],
            }
        edt = distance_transform_edt(self.mskb_hc == 0)
        for props in regionprops(label(self.mskv_hc)):
            coords = props.coords
            self.results["name"      ].append(self.img_name)
            self.results["df"        ].append(self.df)
            self.results["pixel_size"].append(self.pixel_size)
            self.results["label"     ].append(props.label)
            self.results["area"      ].append(props.area * (self.pixel_size ** 2))
            self.results["dist"      ].append(
                np.mean(edt[tuple(coords.T)]) * self.pixel_size)
            self.results["int"  ].append(np.mean(self.imgs[tuple(coords.T)]))
        
        # Plot
        plot(self.results)
        
        # Save
        with open(self.outputs_path / "results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.outputs_path / "results.csv", index=False)

#%% Class(Correct) : ----------------------------------------------------------

class Correct:
    
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
        if self.procedure["correct"]:
            self.initialize()
            self.init_viewer()
            self.init_layers()
        
#%% Class(Correct) : initialize() ---------------------------------------------

    def initialize(self):
        
        # Paths
        self.img_name = parameters["img_name"]
        self.data_path = Path(parameters["data_path"] / self.img_name)
        self.level_path = self.data_path / f"level-{parameters['df']}"
        self.outputs_path = self.level_path / "outputs"
              
        # Images
        filemap = {
            "imgs" : ("imgs.tif", io.imread),
            "mtds" : ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            "mskc" : ("mskc_hc.tif", io.imread),
            "mskn" : ("mskn_hc.tif", io.imread),
            "mskv" : ("mskv_hc.tif", io.imread),
            "mskb" : ("mskb_hc.tif", io.imread),
            "lblc" : ("lblc_hc.tif", io.imread),
            }
        
        for attr, (name, loader) in filemap.items():
            path = self.outputs_path / name
            if path.exists():
                setattr(self, attr, loader(path))
            else:
                if name == "mskb_hc.tif":
                    self.mskb = np.zeros_like(self.mskc)
                elif name == "lblc_hc.tif":
                    self.lblc = label(self.mskc).astype("uint8")
                else:
                    setattr(self, attr, loader(str(path).replace("_hc", "")))
            
        # Variables
        self.active = "imgs"
        self.labels = {
            "mskc" : 1, 
            "mskn" : 2, 
            "mskv" : 6,
            "mskb" : 231,
            }
        
#%% Class(Correct) : function(s) ----------------------------------------------
                               
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
        
    # Updates
            
    def update_masks(self):
        
        # Fetch attibutes
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                setattr(self, name, self.vwr.layers[name].data)
        
        # Synchronise masks
        sync_masks(self.mskc, self.mskn, self.mskv)
        self.mskb[self.mskc == 0] = 0
        
        # Update viewer
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                self.vwr.layers[name].data = getattr(self, name)
    
    def update_labels(self):
        self.lblc = self.mskc > 0
        mskb = self.vwr.layers["mskb"].data
        mskb = skeletonize(mskb) * self.labels["mskb"]
        self.lblc[mskb != 0] = 0
        self.lblc = label(self.lblc > 0, connectivity=1).astype("uint8")
        self.vwr.layers["mskb"].data = mskb
        self.vwr.layers["lblc"].data = self.lblc
        
    def save(self):
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                io.imsave(
                    self.outputs_path / (f"{name}_hc.tif"),
                    (getattr(self, name) > 0).astype("uint8"), check_contrast=False
                    )
            elif name == "lblc":
                io.imsave(
                    self.outputs_path / (f"{name}_hc.tif"),
                    getattr(self, name), check_contrast=False
                    )
        
    def update(self):
        self.update_masks()
        self.update_labels()
        self.save()
                
    # Correct
        
    def correct_mask(self, target):
        self.update()
        self.active = target
        for name in self.vwr.layers:
            name = str(name)
            if name in ["imgs", self.active]:
                self.vwr.layers[name].visible = 1
                if name == "mskb":
                    self.vwr.layers["lblc"].visible = 1
            else:
                self.vwr.layers[name].visible = 0
        self.set_active()
        self.set_label()
        self.paint()

#%% Class(Correct) : init_viewer() --------------------------------------------

    def init_viewer(self):
                
        # Create viewer
        self.vwr = napari.Viewer()

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
            
            "imgs" : {
                "name"     : "imgs",
                "visible"  : 1,
                "opacity"  : 0.6,
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
                "opacity"  : 0.6,
                "blending" : "additive",
                },
            
            "mskb" : {
                "name"     : "mskb",
                "visible"  : 0,
                "opacity"  : 0.6,
                "blending" : "additive",
                },
            
            "lblc" : {
                "name"     : "lblc",
                "visible"  : 0,
                "opacity"  : 0.2,
                "blending" : "additive",
                },
            
            }
        
        self.vwr.add_image(self.imgs , **parameters["imgs"])  
        self.vwr.add_labels(self.lblc, **parameters["lblc"])
        self.vwr.add_labels(
            self.mskc * self.labels["mskc"], **parameters["mskc"]) 
        self.vwr.add_labels(
            self.mskn * self.labels["mskn"], **parameters["mskn"])
        self.vwr.add_labels(
            self.mskv * self.labels["mskv"], **parameters["mskv"])
        self.vwr.add_labels(
            self.mskb * self.labels["mskb"], **parameters["mskb"])
        
        # Set default brush size
        for layer in self.vwr.layers:
            if layer.__class__.__name__ == "Labels":
                layer.brush_size = 60

        self.set_active()
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    
#%% Analyse -------------------------------------------------------------------
        
    # Fetch
    imgs = main.imgs
    prdc = main.prdc
    prdn = main.prdn
    prdv = main.prdv
    mskc = main.mskc
    mskn = main.mskn
    mskv = main.mskv
    if hasattr(main, "mskc_hc"):
        mskc_hc = main.mskc_hc
        mskn_hc = main.mskn_hc
        mskv_hc = main.mskv_hc
        mskb_hc = main.mskb_hc
        lblc_hc = main.lblc_hc 
        results = main.results
    
    model_types = parameters["model_types"]
        
    # -------------------------------------------------------------------------
    
    # # Imports
    # from skimage.measure import regionprops
    # from scipy.ndimage import distance_transform_edt
    
    # def binned_distribution(x, y=None, bin_width=10):
    #     bin_half_width = bin_width // 2
    #     bin_max = np.max(x)
    #     bin_centers = np.arange(bin_half_width, bin_max, bin_width, dtype=int)
    #     distribution = []
    #     for bin_center in bin_centers:
    #         idx = np.where(
    #             (distance >= (bin_center - bin_half_width)) &
    #             (distance <  (bin_center + bin_half_width))
    #             )[0]
    #         if y is None:
    #             distribution.append(
    #                 (bin_center, len(idx)))
    #         else:
    #             distribution.append(
    #                 (bin_center, np.mean(y[idx])))
    #     return np.stack(distribution)
    
    # # Extract measurments
    # area, distance, intensity = [], [], []
    # edt = distance_transform_edt(mskb_hc == 0)
    # for props in regionprops(label(mskv_hc)):
    #     coords = props.coords
    #     area.append(props.area)
    #     distance.append(np.mean(edt[tuple(coords.T)]))
    #     intensity.append(np.mean(imgs[tuple(coords.T)]))
    # area = np.stack(area)
    # distance = np.stack(distance)
    # intensity = np.stack(intensity)
    
    # # Binned distributions
    # distance_dist = binned_distribution(distance, y=None, bin_width=10)
    # area_dist = binned_distribution(distance, y=area, bin_width=10)
    # intensity_dist = binned_distribution(distance, y=intensity, bin_width=10)
                
#%% Plot ---------------------------------------------------------------------- 
    
    # # Initialize plot
    # fig = plt.figure(figsize=(6, 9))
    # gs = fig.add_gridspec(3, 1)
    # ax0 = fig.add_subplot(gs[0, 0])  # bar plot #1
    # ax1 = fig.add_subplot(gs[1, 0])  # bar plot #2
    # ax2 = fig.add_subplot(gs[2, 0])  # bar plot #3
    
    # ax0.bar(
    #     distance_dist[:, 0], distance_dist[:, 1],
    #     width=8, alpha=1, color="lightgray",
    #     )
    # ax0.set_title("Distance")
    # ax0.set_ylabel("Count")
    # ax0.set_xlabel("Dist. to cell junction")
    # ax0.set_xlim(-5, 500)
    
    # ax1.bar(
    #     area_dist[:, 0], area_dist[:, 1],
    #     width=8, alpha=1, color="lightgray",
    #     )
    # ax1.set_title("Area")
    # ax1.set_ylabel("Area")
    # ax1.set_xlabel("Dist. to cell junction")
    # ax1.set_xlim(-5, 500)
    
    # ax2.bar(
    #     intensity_dist[:, 0], intensity_dist[:, 1],
    #     width=8, alpha=1, color="lightgray",
    #     )
    # ax2.set_title("Intensity")
    # ax2.set_ylabel("Intensity (A.U.)")
    # ax2.set_xlabel("Dist. to cell junction")
    # ax2.set_xlim(-5, 500)
    
    # plt.tight_layout()
    
    #%%
    
    # plt.hist(datav, bins=1000)
    # plt.xlim(0, 100)
        
    # # Display
    # vwr = napari.Viewer()
    # vwr.add_image(
    #     (mskc_hc * 255).astype("uint8"), 
    #     blending="additive", opacity=0.25, visible=0,
    #     )
    # vwr.add_image(
    #     (mskb_hc * 255).astype("uint8"), 
    #     blending="additive", opacity=1.00, visible=1,
    #     )
    # vwr.add_image(
    #     mskb_hc_edt, 
    #     blending="additive", opacity=0.25, visible=1,
    #     )
    # vwr.add_labels(
    #     lblv, 
    #     blending="additive", opacity=0.5, visible=1,
    #     )
    
    # -------------------------------------------------------------------------
    
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
    
    