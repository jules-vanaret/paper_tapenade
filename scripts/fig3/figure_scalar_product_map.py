
import numpy as np
import tifffile
import napari
from tqdm import tqdm
from magicgui import magicgui




path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed/all_quantities_midplane'



# !!!!!!!!!! CHANGE THIS !!!!!!!!!!
indices_to_display = range(1,9)#range(2, 8)#[2, 6] # from 1 to 8
# !!!!!!!!!! CHANGE THIS !!!!!!!!!!




def get_napari_angles_cmap(colors=None, color_mode="recirculation"):
    cmap_class = napari.utils.colormaps.colormap.Colormap

    if colors is None:

        if color_mode == "recirculation":
            colors_strs = [
                "white",
                "white",
                "blue",
                "blue",
                "white",
                "white",
                "red",
                "red",
                "white",
            ]
        elif color_mode == "nematic":
            # colors_strs = [
            #     "red",
            #     "white",
            #     "blue",
            #     "blue",
            #     "white",
            #     "red",
            #     "red",
            #     "white",
            #     "blue",
            #     "blue",
            #     "white",
            #     "red",
            # ]
            colors_strs = [
                "red",
                "red",
                "lightcoral",
                "white",
                "cornflowerblue",
                "blue",
                "blue",
                "cornflowerblue",
                "white",
                "lightcoral",
                "red",
                "red",
                "lightcoral",
                "white",
                "cornflowerblue",
                "blue",
                "blue",
                "cornflowerblue",
                "white",
                "lightcoral",
                "red",
            ]

        cmap = cmap_class(
            colors_strs, controls=np.linspace(0, 1, len(colors_strs))
        )
    else:
        cmap = cmap_class(colors, controls=np.linspace(0, 1, len(colors)))

    return cmap
cmap = get_napari_angles_cmap(color_mode='nematic')


for index_organoid in tqdm(indices_to_display):

    mask = tifffile.imread(
        f'{path_to_data}/mask/ag{index_organoid}_mask.tif',
    )

    data = tifffile.imread(
        f'{path_to_data}/data/ag{index_organoid}_norm.tif',
    )

    labels = tifffile.imread(
        f'{path_to_data}/labels/ag{index_organoid}_norm_labels.tif',
    )

    density_40 = tifffile.imread(
        f'{path_to_data}/cell_density/ag{index_organoid}_sigma40.tif',
    )
    density_40[~mask] = np.nan

    density_gradient = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}.npy',
    )

    lengths = np.linalg.norm(density_gradient[:,1], axis=1)
    density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

    density_gradient_angles = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}_angles.npy',
    )

    scalar_product = tifffile.imread(
        f'{path_to_data}/dot_product_map/ag{index_organoid}.tif',
    )
    scalar_product[~mask] = np.nan

    viewer = napari.Viewer()

    viewer.add_image(data, name='data', colormap='gray_r')
    viewer.add_labels(labels, name='labels')
    viewer.add_image(density_40, name='density_40', colormap='inferno')

    viewer.add_vectors(density_gradient, name='density_gradient', 
                       properties={'angles': density_gradient_angles}, 
                       vector_style='triangle',blending='opaque', edge_colormap=cmap,
                       length=4, edge_width=8)
    viewer.add_image(scalar_product, name='scalar_product', colormap='turbo')

    def remap_angles(angles, angle_shift):
        angles = angles + angle_shift
        angles %= 2*np.pi
        return angles - 2*np.pi*(angles >= np.pi)

    @magicgui(
        auto_call=True,
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

    viewer.grid.enabled = True  

napari.run()