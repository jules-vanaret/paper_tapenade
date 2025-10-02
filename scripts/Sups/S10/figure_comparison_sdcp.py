# This script is used to generate the 2D maps of cell density, volume fraction, nuclear volume,
# true strain magnitude, and dot product between density gradient and true strain major axis
# All fields are displayed in Napari windows using precomputed data located in the folder
# "4acd_data_morphology/all_quantities_midplane"

import napari
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib
from magicgui import magicgui
from pathlib import Path
from napari.utils import DirectLabelColormap


path_to_data = Path(__file__).parents[3] / 'data/S10_comparison_sdcp/all_quantities_midplane'
sigmas = [10,20,40] # pixels



def get_napari_angles_cmap():
    cmap_class = napari.utils.colormaps.colormap.Colormap

    colors_strs = [
        (195, 163,75),
        (163, 103, 44),
        (135, 64, 55),
        (115, 57, 87),
        (92, 83, 139),
        (79, 136, 185),
        (116, 187, 205),
        (180, 222, 198),
        (214, 216, 147),
        (195, 163,75),
        (163, 103, 44),
        (135, 64, 55),
        (115, 57, 87),
        (92, 83, 139),
        (79, 136, 185),
        (116, 187, 205),
        (180, 222, 198),
        (214, 216, 147)
    ]

    for i, elem in enumerate(colors_strs):
        colors_strs[i] = tuple([x / 255 for x in elem])


    cmap = cmap_class(
        colors_strs, controls=np.linspace(0, 1, 1+len(colors_strs)),
        interpolation='zero'
    )

    return cmap


cmap_vectors_nematic = get_napari_angles_cmap()
cmap_vectors_gradient = get_napari_angles_cmap()


    


labels_sd = tifffile.imread(
    path_to_data / f'labels/ag6_norm_labels.tif'
)
labels_cp = tifffile.imread(
    path_to_data / f'labels/ag6_norm_labels_cellposesam.tif'
)
data = tifffile.imread(
    path_to_data / f'data/ag6_norm.tif'
)
mask = tifffile.imread(
    path_to_data / f'mask/ag6_mask.tif'
)


viewer1 = napari.Viewer()
viewer2 = napari.Viewer()
viewer3 = napari.Viewer()


viewer1.add_image(data)
viewer1.add_labels(labels_sd, opacity=1)
viewer1.add_labels(labels_cp, opacity=1)

# pixels in labels_sd that are not in labels_cp
mask_sd_not_cp = (labels_sd > 0) & (labels_cp == 0)
# pixels in labels_cp that are not in labels_sd
mask_cp_not_sd = (labels_cp > 0) & (labels_sd == 0)
# pixels in both labels_sd and labels_cp
mask_both = (labels_sd > 0) & (labels_cp > 0)

labels_diff = mask_both + 2*mask_sd_not_cp + 3*mask_cp_not_sd

cmap_labels = DirectLabelColormap(
    color_dict={
        1: (0,0,0,1),
        0: (1,1,1,1),
        2: (0,1,1,1),
        3: (1,0,1,1) 
    },
)

viewer1.add_labels(labels_diff, opacity=1, name='labels_diff', colormap=cmap_labels)

viewer1.grid.enabled = True
viewer1.grid.shape = (1, -1)




###
cell_density = tifffile.imread(
    path_to_data / f'cell_density/ag6_sigma20.tif'
)
cell_density[~mask] = np.nan
volume_fraction = tifffile.imread(
    path_to_data / f'volume_fraction/ag6_sigma20.tif'
)
volume_fraction[~mask] = np.nan
nuclear_volume = tifffile.imread(
    path_to_data / f'nuclear_volume/ag6_sigma20.tif'
)
nuclear_volume[~mask] = np.nan


percs_density = [0.5/(10/0.621)**3,1.62/(10/0.621)**3]#np.percentile(cell_density[mask], [1,99])
# print(f'index organoid 6 percs_density {percs_density*(10/0.621)**3}')
percs_volume_fraction = [0.26,0.61]#np.percentile(volume_fraction[mask], [1,99])
# print(f'index organoid 6 percs_volume_fraction {percs_volume_fraction}')
percs_nuclear_volume = [310/0.621**3,690/0.621**3]#np.percentile(nuclear_volume[mask], [1,99])
# print(f'index organoid 6 percs_nuclear_volume {percs_nuclear_volume /* 0.621**3}')
        
napari_true_strain = np.load(
    path_to_data / f'true_strain_tensor/ag6_sigma{sigmas[0]}.npy'
)
lengths = np.linalg.norm(napari_true_strain[:,1], axis=1)
napari_true_strain[:,1] = napari_true_strain[:,1]/np.median(lengths[:,None]) * 5

napari_true_strain_angles = np.load(
    path_to_data / f'true_strain_tensor/ag6_sigma{sigmas[0]}_angles.npy'
)

viewer2.add_image(cell_density, name='cell_density', colormap='inferno', contrast_limits=percs_density)
viewer2.add_labels(mask*1, name='mask', opacity=1)
viewer2.layers[-1].contour=4
viewer2.add_image(volume_fraction, name='volume_fraction', colormap='cividis', contrast_limits=percs_volume_fraction)
viewer2.add_labels(mask*1, name='mask', opacity=1)
viewer2.layers[-1].contour=4
viewer2.add_image(nuclear_volume, name='nuclear_volume', colormap='viridis', contrast_limits=percs_nuclear_volume)
viewer2.add_labels(mask*1, name='mask', opacity=1)
viewer2.layers[-1].contour=4
viewer2.add_vectors(napari_true_strain, name='true_strain',
    properties={'angles': napari_true_strain_angles},
    vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
    length=1.5, edge_width=5)
viewer2.grid.enabled = True
viewer2.grid.shape = (1, -1)
viewer2.grid.stride=2

###

density_gradient = np.load(
    path_to_data / f'cell_density_gradient/ag6.npy'
)
lengths = np.linalg.norm(density_gradient[:,1], axis=1)
density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

density_gradient_angles = np.load(
    path_to_data / f'cell_density_gradient/ag6_angles.npy'
)

gradient_magnitude = tifffile.imread(
    path_to_data / f'cell_density_gradient_mag/ag6.tif'
)

ts_maxeig = tifffile.imread(
    path_to_data / f'true_strain_maxeig/ag6_sigma40.tif'
)

ts_maxeig[ts_maxeig==0] = np.nan

dot_product_map = tifffile.imread(
    path_to_data / f'dot_product_map/ag6.tif'
)
dot_product_map[~mask] = np.nan



viewer3.add_vectors(density_gradient, name='density_gradient',
                    properties={'angles': density_gradient_angles},
                    vector_style='triangle', blending='translucent', opacity=1, 
                    edge_colormap=cmap_vectors_gradient,
                    length=2, edge_width=10)


gradient_magnitude[gradient_magnitude==0] = np.nan
viewer3.add_image(gradient_magnitude, name='gradient_magnitude', colormap='Reds',)
                #   contrast_limits=[0.0042/(10/0.621)**4, 0.064/(10/0.621)**4])

viewer3.add_image(ts_maxeig, name='ts_maxeig', colormap='Blues')#, contrast_limits=[0.018,0.22])
viewer3.add_image(dot_product_map, name='dot_product_map', colormap='turbo', contrast_limits=[0, 0.5])


viewer3.grid.enabled = True
viewer3.grid.shape = (1, -1)
# 
def remap_angles(angles, angle_shift):
    angles = angles + angle_shift
    angles %= 2*np.pi
    return angles - 2*np.pi*(angles >= np.pi)

@magicgui(auto_call=True,
            angle_shift={'widget_type': 'FloatSlider', 'min': 0, 'max': 2*np.pi, 'step': 1},
        )
def widget_angle_shift(viewer: napari.Viewer, angle_shift: float = 0):

    for vectors in viewer.layers:
        if hasattr(vectors, 'properties') and 'angles' in vectors.properties:

            if not('old_angles' in vectors.properties):
                old_dict = {'old_angles': vectors.properties['angles']}
                vectors.properties = {**vectors.properties, **old_dict}

            angles = vectors.properties['old_angles']
            angles = remap_angles(angles, angle_shift)

            vectors.properties = {**vectors.properties, **{'angles': angles}}
            vectors.refresh()
@magicgui(auto_call=True,
            angle_shift={'widget_type': 'FloatSlider', 'min': 0, 'max': 2*np.pi, 'step': 1},
        )
def widget_angle_shift2(viewer: napari.Viewer, angle_shift: float = 0):

    for vectors in viewer.layers:
        if hasattr(vectors, 'properties') and 'angles' in vectors.properties:

            if not('old_angles' in vectors.properties):
                old_dict = {'old_angles': vectors.properties['angles']}
                vectors.properties = {**vectors.properties, **old_dict}

            angles = vectors.properties['old_angles']
            angles = remap_angles(angles, angle_shift)

            vectors.properties = {**vectors.properties, **{'angles': angles}}
            vectors.refresh()
            

viewer2.window.add_dock_widget(widget_angle_shift, area='left')
viewer3.window.add_dock_widget(widget_angle_shift2, area='left')

napari.run()