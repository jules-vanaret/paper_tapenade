import napari
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib
from magicgui import magicgui
from pathlib import Path


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



path_to_data = Path(__file__).parents[2] / 'data/4acd_data_morphology/all_quantities_midplane'

cmap_vectors_nematic = get_napari_angles_cmap()
cmap_vectors_gradient = get_napari_angles_cmap()
inds_organoids = [6, 2, 4, 5] # add 6 as first to showcase orthogonal views next to XY views


def remap_angles(angles, angle_shift):
    angles = angles + angle_shift
    angles %= 2*np.pi
    return angles - 2*np.pi*(angles >= np.pi)


viewer_list = []

for index_organoid in tqdm(inds_organoids):
    
    # Create a new viewer for each organoid
    viewer = napari.Viewer(title=f'Organoid {index_organoid}')
    

    labels = tifffile.imread(
        path_to_data / f'labels/ag{index_organoid}_norm_labels.tif'
    )
    data = tifffile.imread(
        path_to_data / f'data/ag{index_organoid}_norm.tif'
    )
    mask = tifffile.imread(
        path_to_data / f'mask/ag{index_organoid}_mask.tif'
    )

    ### -->
    viewer.add_image(data, colormap='gray_r', contrast_limits=[0.14,0.5],name='data')
    viewer.layers[-1].gamma = 0.75

    ### -->
    cell_density_10 = tifffile.imread(
            path_to_data / f'cell_density/ag{index_organoid}_sigma10.tif'
        )
    cell_density_40 = tifffile.imread(
        path_to_data / f'cell_density/ag{index_organoid}_sigma40.tif'
    )

    cmap = matplotlib.colormaps['inferno']
    imrgb = cell_density_40.copy()
    percs_density = np.percentile(cell_density_10[mask], [1,99])
    imrgb = np.clip((imrgb - percs_density[0])/(percs_density[1] - percs_density[0]), 0.001, 0.999)
    imrgb[mask.copy() == 0] = 0

    # Vectorized colormap application
    im_cmap_density = cmap(imrgb)
    # Set pixels outside mask to transparent (alpha=0)
    im_cmap_density[mask == 0] = [0, 0, 0, 0]
    
    viewer.add_image(im_cmap_density, opacity=1)

    ### -->
    density_gradient = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}.npy'
    )
    lengths = np.linalg.norm(density_gradient[:,1], axis=1)
    density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

    density_gradient_angles = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}_angles.npy'
    )

    viewer.add_vectors(density_gradient, name='density_gradient',
                        properties={'angles': density_gradient_angles},
                        vector_style='triangle', blending='translucent', opacity=1, 
                        edge_colormap=cmap_vectors_gradient,
                        length=2, edge_width=10)

    ### -->
    napari_true_strain_resampled = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma10_resampled.npy'
    )
    lengths = np.linalg.norm(napari_true_strain_resampled[:,1], axis=1)
    napari_true_strain_resampled[:,1] = napari_true_strain_resampled[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_resampled_angles = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma10_resampled_angles.npy'
    )

    viewer.add_vectors(napari_true_strain_resampled, name='true_strain_resampled',
                        properties={'angles': napari_true_strain_resampled_angles},
                        vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
                        length=1.5, edge_width=5)

    ### -->
    dot_product_map = tifffile.imread(
        path_to_data / f'dot_product_map/ag{index_organoid}.tif'
    )
    dot_product_map[~mask] = np.nan

    viewer.add_image(dot_product_map, name='dot_product_map', colormap='turbo', contrast_limits=[0, 0.5])

    ### ORTHOGONAL VIEWS FOR THE SAME ORGANOID -->
    labels_ortho = tifffile.imread(
        path_to_data / f'labels/ag{index_organoid}_norm_labels_ortho.tif'
    )
    data_ortho = tifffile.imread(
        path_to_data / f'data/ag{index_organoid}_norm_ortho.tif'
    )
    mask_ortho = tifffile.imread(
        path_to_data / f'mask/ag{index_organoid}_mask_ortho.tif'
    )

    # Reverse x-direction for organoid 6
    if index_organoid == 6:
        labels_ortho = labels_ortho[:, ::-1]
        data_ortho = data_ortho[:, ::-1]
        mask_ortho = mask_ortho[:, ::-1]

    ### -->
    viewer.add_image(data_ortho, colormap='gray_r', contrast_limits=[0.14,0.5],name='data_ortho')
    viewer.layers[-1].gamma = 0.75

    ### -->
    cell_density_10_ortho = tifffile.imread(
            path_to_data / f'cell_density/ag{index_organoid}_sigma10_ortho.tif'
        )
    cell_density_40_ortho = tifffile.imread(
        path_to_data / f'cell_density/ag{index_organoid}_sigma40_ortho.tif'
    )

    # Reverse x-direction for organoid 6
    if index_organoid == 6:
        cell_density_10_ortho = cell_density_10_ortho[:, ::-1]
        cell_density_40_ortho = cell_density_40_ortho[:, ::-1]

    cmap = matplotlib.colormaps['inferno']
    imrgb_ortho = cell_density_40_ortho.copy()
    percs_density_ortho = np.percentile(cell_density_10_ortho[mask_ortho], [1,99])
    imrgb_ortho = np.clip((imrgb_ortho - percs_density_ortho[0])/(percs_density_ortho[1] - percs_density_ortho[0]), 0.001, 0.999)
    imrgb_ortho[mask_ortho.copy() == 0] = 0

    # Vectorized colormap application
    im_cmap_density_ortho = cmap(imrgb_ortho)
    # Set pixels outside mask to transparent (alpha=0)
    im_cmap_density_ortho[mask_ortho == 0] = [0, 0, 0, 0]
    
    viewer.add_image(im_cmap_density_ortho, opacity=1)

    ### -->
    density_gradient_ortho = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}_ortho.npy'
    )
    lengths_ortho = np.linalg.norm(density_gradient_ortho[:,1], axis=1)
    density_gradient_ortho[:,1] = density_gradient_ortho[:,1]/np.median(lengths_ortho[:,None]) * 5

    density_gradient_angles_ortho = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}_angles_ortho.npy'
    )

    # Reverse x-direction for organoid 6
    if index_organoid == 6:
        # Flip x-coordinates of vector positions
        density_gradient_ortho[:, 0, 1] = data_ortho.shape[1] - 1 - density_gradient_ortho[:, 0, 1]
        # Flip x-component of vector directions
        density_gradient_ortho[:, 1, 1] = -density_gradient_ortho[:, 1, 1]

    viewer.add_vectors(density_gradient_ortho, name='density_gradient_ortho',
                        properties={'angles': density_gradient_angles_ortho},
                        vector_style='triangle', blending='translucent', opacity=1, 
                        edge_colormap=cmap_vectors_gradient,
                        length=2, edge_width=10)

    ### -->
    napari_true_strain_resampled_ortho = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma10_resampled_ortho.npy'
    )
    lengths_ortho = np.linalg.norm(napari_true_strain_resampled_ortho[:,1], axis=1)
    napari_true_strain_resampled_ortho[:,1] = napari_true_strain_resampled_ortho[:,1]/np.median(lengths_ortho[:,None]) * 5

    napari_true_strain_resampled_angles_ortho = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma10_resampled_angles_ortho.npy'
    )

    # Reverse x-direction for organoid 6
    if index_organoid == 6:
        # Flip x-coordinates of vector positions
        napari_true_strain_resampled_ortho[:, 0, 1] = data_ortho.shape[1] - 1 - napari_true_strain_resampled_ortho[:, 0, 1]
        # Flip x-component of vector directions
        napari_true_strain_resampled_ortho[:, 1, 1] = -napari_true_strain_resampled_ortho[:, 1, 1]

    viewer.add_vectors(napari_true_strain_resampled_ortho, name='true_strain_resampled_ortho',
                        properties={'angles': napari_true_strain_resampled_angles_ortho},
                        vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
                        length=1.5, edge_width=5)

    ### -->
    dot_product_map_ortho = tifffile.imread(
        path_to_data / f'dot_product_map/ag{index_organoid}_ortho.tif'
    )
    dot_product_map_ortho[~mask_ortho] = np.nan

    # Reverse x-direction for organoid 6
    if index_organoid == 6:
        dot_product_map_ortho = dot_product_map_ortho[:, ::-1]
        dot_product_map_ortho = np.where(dot_product_map_ortho==0, np.nan, dot_product_map_ortho)

    viewer.add_image(dot_product_map_ortho, name='dot_product_map_ortho', colormap='turbo', contrast_limits=[0, 0.5])

    ### -->
    viewer.grid.enabled = True
    viewer.grid.shape = (2, -1)  # 2 rows: one for regular views, one for orthogonal views
    
    # Add the viewer to our list
    viewer_list.append(viewer)

    # Add angle shift widget to each viewer
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

    viewer.window.add_dock_widget(widget_angle_shift, area='left')

napari.run()