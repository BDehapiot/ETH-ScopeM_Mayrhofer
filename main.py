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

# qt
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QWidget, QPushButton, QGroupBox, QVBoxLayout

# matplot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tensorflow as tf
import gc

#%% Inputs --------------------------------------------------------------------

procedure = {
    "downscale"  : 1,
    "preprocess" : 1,
    "predict"    : 1,
    "process"    : 2,
    "correct"    : 1,
    "analyse"    : 0,
    }

parameters = {
    
    # Paths
    # "img_name"    : "Ins1e_wt_1.7nm_00",
    # "img_name"    : "Gigyf12d_ko_1.7nm_00",
    "img_name"    : "Ins1e_wt_3.25nm_00",
    "data_path"   : Path("D:\local_Mayrhofer\data"),
    
    # Downscale
    "psize_ref"   : 1.7, # nm
    "df_ref"      : 16,
    
    # Predict
    "model_types" : ["cells", "nuclei", "vesicles"],
    
    # Process
    
    # For df=16
    # 1) prediction threshold
    # 2) minimum object size
    # 3) minimum hole size
    
    "mskc_params" : (0.5, 4096, 32),
    "mskn_params" : (0.5, 512, 32),
    "mskv_params" : (0.25, 8, 4),
    
    }

#%% Class(Main) : -------------------------------------------------------------

class Main:

    def __init__(self, procedure=None, parameters=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.img_name = self.parameters["img_name"]
        self.data_path = Path(self.parameters["data_path"] / self.img_name)
        
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
        
        # Scaling variables
        parts = self.img_name.split("_")
        for part in parts:
            if "nm" in part:
                self.psize_0 = float(part.replace("nm", ""))
                rf = self.psize_0 / self.parameters["psize_ref"]
                self.df = round(self.parameters["df_ref"] / rf, 3)
                self.psize_1 = self.psize_0 * self.df
        
        # Paths
        self.level_path = self.data_path / f"level-{self.df}"
        self.outputs_path = self.level_path / "outputs"

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
                
                if self.img_name == "Ins1e_wt_3.25nm_00":
                    nY, nX = self.imgs.shape
                    overlap = 256
                    imgs0 = self.imgs[:, :nX // 2 + overlap]
                    imgs1 = self.imgs[:, nX // 2 - overlap:]
                    prd0 = predict(imgs0, model_type=model_type)
                    prd1 = predict(imgs1, model_type=model_type)
                    prd = np.hstack((
                        prd0[:, :nX // 2],
                        prd1[:, overlap:],
                        ))
                                        
                else:
                    prd = predict(self.imgs, model_type=model_type)
                
                io.imsave(
                    self.outputs_path / f"prd{model_type[0]}.tif",
                    prd, check_contrast=False
                    )
                
                del prd, prd0, prd1, imgs0, imgs1
                gc.collect()
                tf.keras.backend.clear_session()
                    
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
            
            mskc = get_mask(self.prdc, *self.parameters["mskc_params"])
            mskn = get_mask(self.prdn, *self.parameters["mskn_params"])
            mskv = get_mask(self.prdv, *self.parameters["mskv_params"])
            # sync_masks(mskc, mskn, mskv)
            
            for msk in ["mskc", "mskn", "mskv"]:
                msk_hc_path = self.outputs_path / f"{msk}_hc.tif"
                if msk_hc_path.exists():
                    msk_hc_path.unlink()
            
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
        Correct(
            procedure=self.procedure, 
            parameters=self.parameters,
            df=self.df
            )

#%% Class(Main) : analyse() --------------------------------------------------- 
    
    def analyse(self):
        
        # Nested funtion(s) ---------------------------------------------------

        def plot(results):
            
            # Parameters
            nbins = 128
            bdist_width = 0.5
            inset_ratio = "50%"

            # Get data
            area_bdist = binned_distribution(
                results["dist"], y=results["area"], bin_width=bdist_width)
            int_bdist  = binned_distribution(
                results["dist"], y=results["int" ], bin_width=bdist_width)

            # Initialize plot
            fig = plt.figure(figsize=(6, 9), constrained_layout=True)
            gs = fig.add_gridspec(3, 1)
            ax0 = fig.add_subplot(gs[0, 0])  # Distances
            ax1 = fig.add_subplot(gs[1, 0])  # Areas
            ax2 = fig.add_subplot(gs[2, 0])  # Intensities
            
            # Distance (ax0)
            ax0.hist(results["dist"], bins=nbins, color="lightgray") 
            ax0.set_title("Vesicle distance to cell junctions", loc="left")
            ax0.set_xlabel("Distance (µm)")
            ax0.set_ylabel("count")
            
            # Distance inset (axi0)
            axi0 = inset_axes(
                ax0, width="50%", height="50%", loc="upper right")
            axi0.hist(results["dist"], bins=nbins * 16, color="lightgray") 
            axi0.set_title("Distance (zoomed)", y=0.75)
            axi0.set_xlabel("Distance (µm)")
            axi0.set_ylabel("count")
            axi0.set_xlim(-0.1, 2.1)
            
            # Areas (ax1)
            ax1.hist(results["area"], bins=nbins, color="lightgray") 
            ax1.set_title("Vesicle area", loc="left")
            ax1.set_xlabel("Area (µm²)")
            ax1.set_ylabel("count")
            
            # Areas inset (axi1)
            axi1 = inset_axes(
                ax1, width=inset_ratio, height=inset_ratio, loc="upper right")
            axi1.bar(area_bdist[:, 0], area_bdist[:, 1], color="lightgray")
            axi1.set_title("Areas acc. to distance", y=0.75)
            axi1.set_xlabel("Distance (µm)")
            axi1.set_ylabel("Area (µm²)")
            
            # Intensities (ax2)
            ax2.hist(results["int" ], bins=nbins, color="lightgray")
            ax2.set_title("Vesicle mean intensity", loc="left")
            ax2.set_xlabel("Intensity (A.U.)")
            ax2.set_ylabel("count")
            
            # Intensities (axi2)
            axi2 = inset_axes(
                ax2, width=inset_ratio, height=inset_ratio, loc="upper right")
            axi2.bar(int_bdist[:, 0], int_bdist[:, 1], color="lightgray")
            axi2.set_title("Intensity acc. to distance", y=0.75)
            axi2.set_xlabel("Distance (µm)")
            axi2.set_ylabel("Intensity (A.U.)")   

            return fig        

        # Execute -------------------------------------------------------------
        
        self.initialize()
        
        # Extract measurments
        self.results = {
            "name"       : [],
            "df"         : [],
            "psize" : [],
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
            self.results["psize_0"].append(self.psize_0 * 1e-3)
            self.results["psize_1"].append(self.psize_1 * 1e-3)
            self.results["label"     ].append(props.label)
            self.results["area"      ].append(
                props.area * ((self.psize_1 * 1e-3) ** 2))
            self.results["dist"      ].append(
                np.mean(edt[tuple(coords.T)]) * self.psize)
            self.results["int"  ].append(np.mean(self.imgs[tuple(coords.T)]))
        
        # Plot
        fig = plot(self.results)
        
        # Save
        with open(self.outputs_path / "results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.outputs_path / "results.csv", index=False)
        fig.savefig(self.outputs_path  / "results_plot.png", format="png")

#%% Class(Correct) : ----------------------------------------------------------

class Correct:
    
    def __init__(self, procedure=None, parameters=None, df=None):
        
        # Fetch
        self.procedure  = procedure
        self.parameters = parameters
        self.df = df
        
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
        self.level_path = self.data_path / f"level-{self.df}"
        self.outputs_path = self.level_path / "outputs"
              
        # Images
        filemap = {
            "imgs" : ("imgs.tif", io.imread),
            "mtds" : ("mtds.pkl", lambda p: pickle.load(open(p, "rb"))),
            "prdc" : ("prdc_hc.tif", io.imread),
            "prdn" : ("prdn_hc.tif", io.imread),
            "prdv" : ("prdv_hc.tif", io.imread),
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
        
        # Update viewer
        for name in self.vwr.layers:
            name = str(name)
            if "msk" in name:
                self.vwr.layers[name].data = getattr(self, name)
    
    def update_labels(self):
        self.lblc = self.mskc > 0
        mskb = self.vwr.layers["mskb"].data
        mskb = skeletonize(mskb, method="lee") * self.labels["mskb"]
        mskb[self.mskc == 0] = 0
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
            
            "lblc" : {
                "name"     : "lblc",
                "blending" : "additive",
                "visible"  : 0,
                "opacity"  : 0.2,
                },
            
            }
        
        self.vwr.add_image(self.imgs , **parameters["imgs"]) 
        self.vwr.add_image(self.prdc , **parameters["prdc"])  
        self.vwr.add_image(self.prdn , **parameters["prdn"])  
        self.vwr.add_image(self.prdv , **parameters["prdv"])  
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
                layer.brush_size = 20

        self.set_active()
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main(procedure=procedure, parameters=parameters)
    
#%% Analyse -------------------------------------------------------------------
        
    # # Fetch
    # psize_0 = main.psize_0
    # psize_1 = main.psize_1
    # df = main.df

    # if hasattr(main, "imgs"):
    #     imgs = main.imgs
    # if hasattr(main, "prdc"):
    #     prdc = main.prdc
    #     prdn = main.prdn
    #     prdv = main.prdv
    # if hasattr(main, "mskc"):
    #     mskc = main.mskc
    #     mskn = main.mskn
    #     mskv = main.mskv
    # if hasattr(main, "mskc_hc"):
    #     mskc_hc = main.mskc_hc
    #     mskn_hc = main.mskn_hc
    #     mskv_hc = main.mskv_hc
    #     mskb_hc = main.mskb_hc
    #     lblc_hc = main.lblc_hc 
    # if hasattr(main, "results"):
    #     results = main.results 
        
#%% Development ---------------------------------------------------------------
    
    # nY, nX = imgs.shape
    # overlap = 256
    # imgs0 = imgs[:, :nX // 2 + overlap]
    # imgs1 = imgs[:, nX // 2 - overlap:]
    # imgs2 = np.hstack((
    #     imgs0[:, :nX // 2],
    #     imgs1[:, overlap:],
    #     ))
