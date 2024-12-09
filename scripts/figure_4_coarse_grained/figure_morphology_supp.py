import napari
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib
from magicgui import magicgui

path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed/all_quantities_midplane'

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
inds_organoids = [2, 4, 5]



viewer = napari.Viewer()

for index_organoid in tqdm(inds_organoids):
    

    labels = tifffile.imread(
        f'{path_to_data}/labels/ag{index_organoid}_norm_labels.tif'
    )
    data = tifffile.imread(
        f'{path_to_data}/data/ag{index_organoid}_norm.tif'
    )
    mask = tifffile.imread(
        f'{path_to_data}/mask/ag{index_organoid}_mask.tif'
    )

    ### -->
    viewer.add_image(data, colormap='gray_r', contrast_limits=[0.14,0.5],name='data')
    viewer.layers[-1].gamma = 0.75

    ### -->
    cell_density_10 = tifffile.imread(
            f'{path_to_data}/cell_density/ag{index_organoid}_sigma10.tif'
        )
    cell_density_40 = tifffile.imread(
        f'{path_to_data}/cell_density/ag{index_organoid}_sigma40.tif'
    )

    cmap = matplotlib.colormaps['inferno']
    imrgb = cell_density_40.copy()
    percs_density = np.percentile(cell_density_10[mask], [1,99])
    imrgb = np.clip((imrgb - percs_density[0])/(percs_density[1] - percs_density[0]), 0.001, 0.999)
    imrgb[mask.copy() == 0] = 0
    im_cmap_density = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
    for i in range(imrgb.shape[0]):
        for j in range(imrgb.shape[1]):
            if imrgb[i,j] > 0:
                im_cmap_density[i,j] = cmap(imrgb[i,j])
    
    viewer.add_image(im_cmap_density, opacity=1)

    ### -->
    density_gradient = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}.npy'
    )
    lengths = np.linalg.norm(density_gradient[:,1], axis=1)
    density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

    density_gradient_angles = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}_angles.npy'
    )

    viewer.add_vectors(density_gradient, name='density_gradient',
                        properties={'angles': density_gradient_angles},
                        vector_style='triangle', blending='translucent', opacity=1, 
                        edge_colormap=cmap_vectors_gradient,
                        length=2, edge_width=10)

    ### -->
    napari_true_strain_resampled = np.load(
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma10_resampled.npy'
    )
    lengths = np.linalg.norm(napari_true_strain_resampled[:,1], axis=1)
    napari_true_strain_resampled[:,1] = napari_true_strain_resampled[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_resampled_angles = np.load(
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma10_resampled_angles.npy'
    )

    viewer.add_vectors(napari_true_strain_resampled, name='true_strain_resampled',
                        properties={'angles': napari_true_strain_resampled_angles},
                        vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
                        length=1.5, edge_width=5)

    ### -->
    dot_product_map = tifffile.imread(
        f'{path_to_data}/dot_product_map/ag{index_organoid}.tif'
    )
    dot_product_map[~mask] = np.nan

    viewer.add_image(dot_product_map, name='dot_product_map', colormap='turbo')

    ### -->
    viewer.grid.enabled = True
    viewer.grid.shape = (len(inds_organoids), -1)
    viewer.grid.stride = -1


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

viewer.window.add_dock_widget(widget_angle_shift, area='right')

napari.run()