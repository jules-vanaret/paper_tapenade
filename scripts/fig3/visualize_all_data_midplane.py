
import numpy as np
import tifffile
import napari
from tqdm import tqdm
from magicgui import magicgui




path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'

name_folder_midplane = 'all_quantities_midplane'

# !!!!!!!!!! CHANGE THIS !!!!!!!!!!
indices_to_display = range(2, 8)#[2, 6] # from 1 to 8
# !!!!!!!!!! CHANGE THIS !!!!!!!!!!


it10 = 0.1
itr10 = 0.1
tst10 = 250
tstr10 = 250

it20 = 0.15
itr20 = 0.15
tst20 = 250
tstr20 = 250

it40 = 0.2
itr40 = 0.2
tst40 = 250*2
tstr40 = 250*2

its = {10: it10, 20: it20, 40: it40}
itrs = {10: itr10, 20: itr20, 40: itr40}
tsts = {10: tst10, 20: tst20, 40: tst40}
tstrs = {10: tstr10, 20: tstr20, 40: tstr40}

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
        f'{path_to_data}/{name_folder_midplane}/mask/ag{index_organoid}_mask.tif',
    )

    data = tifffile.imread(
        f'{path_to_data}/{name_folder_midplane}/data/ag{index_organoid}_norm.tif',
    )

    labels = tifffile.imread(
        f'{path_to_data}/{name_folder_midplane}/labels/ag{index_organoid}_norm_labels.tif',
    )

    viewer = napari.Viewer()


    for i,sigma in enumerate([10, 20, 40]):

        if i == 0:
            viewer.add_image(mask,name='mask',colormap='gray')
        elif i == 1:
            viewer.add_image(data,name='data',colormap='gray')
        else:
            viewer.add_labels(labels,name='labels')

        cell_density = tifffile.imread(
                f'{path_to_data}/{name_folder_midplane}/cell_density/ag{index_organoid}_sigma{sigma}.tif',
            )
        cell_density[~mask] = np.nan
        
        volume_fraction = tifffile.imread(
                f'{path_to_data}/{name_folder_midplane}/volume_fraction/ag{index_organoid}_sigma{sigma}.tif',
            )
        volume_fraction[~mask] = np.nan
        
        nuclear_volume = tifffile.imread(
                f'{path_to_data}/{name_folder_midplane}/nuclear_volume/ag{index_organoid}_sigma{sigma}.tif',
            )
        nuclear_volume[~mask] = np.nan
        
        inertia_tensor_napari = np.load(
                f'{path_to_data}/{name_folder_midplane}/inertia_tensor/ag{index_organoid}_sigma{sigma}.npy',
            )
        
        inertia_tensor_napari_angles = np.load(
                f'{path_to_data}/{name_folder_midplane}/inertia_tensor/ag{index_organoid}_sigma{sigma}_angles.npy',
            )
        
        inertia_tensor_resampled = np.load(
                f'{path_to_data}/{name_folder_midplane}/inertia_tensor/ag{index_organoid}_sigma{sigma}_resampled.npy',
            )
        
        inertia_tensor_resampled_angles = np.load(
                f'{path_to_data}/{name_folder_midplane}/inertia_tensor/ag{index_organoid}_sigma{sigma}_resampled_angles.npy',
            )
        
        true_strain_tensor = np.load(
                f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}.npy',
            )
        
        true_strain_tensor_angles = np.load(
                f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_angles.npy',
            )
        
        true_strain_tensor_resampled = np.load(
                f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled.npy',
            )
        
        true_strain_tensor_resampled_angles = np.load(
                f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled_angles.npy',
            )
        
    

        viewer.add_image(cell_density,name=f'cell_density_{sigma}',colormap='inferno')
        # viewer.add_image(volume_fraction,name=f'volume_fraction_{sigma}',colormap='cividis')
        # viewer.add_image(nuclear_volume,name=f'nuclear_volume_{sigma}',colormap='viridis')

        # viewer.add_vectors(inertia_tensor_napari,name=f'inertia_tensor_{sigma}',vector_style='line',blending='opaque', 
        #              edge_colormap=cmap, properties={'angles': inertia_tensor_napari_angles}, length=its[sigma])
        
        # viewer.add_vectors(inertia_tensor_resampled,name=f'inertia_tensor_resampled_{sigma}',vector_style='line',blending='opaque',
        #                 edge_colormap=cmap, properties={'angles': inertia_tensor_resampled_angles}, length=itrs[sigma])
        
        # viewer.add_vectors(true_strain_tensor,name=f'true_strain_tensor_{sigma}',vector_style='line',blending='opaque',
        #                 edge_colormap=cmap, properties={'angles': true_strain_tensor_angles}, length=tsts[sigma])
        
        viewer.add_vectors(true_strain_tensor_resampled,name=f'true_strain_tensor_resampled_{sigma}',vector_style='line',blending='opaque',
                        edge_colormap=cmap, properties={'angles': true_strain_tensor_resampled_angles}, length=tstrs[sigma])
        

    viewer.grid.enabled = True
    viewer.grid.shape = (3, -1)


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
