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

path_to_data = path_to_data = Path(__file__).parents[2] / 'data/4acd_data_morphology/all_quantities_midplane'

cmap_vectors_nematic = get_napari_angles_cmap()
cmap_vectors_gradient = get_napari_angles_cmap()


inds_organoids = [6]


sigmas = [10,20,40]




for index_organoid in tqdm(inds_organoids):
    
    viewer1 = napari.Viewer()
    viewer2 = napari.Viewer()
    viewer3 = napari.Viewer()
    viewer4 = napari.Viewer()
    # viewer5 = napari.Viewer()
    # viewer6 = napari.Viewer()

    labels = tifffile.imread(
        path_to_data / f'labels/ag{index_organoid}_norm_labels.tif'
    )
    data = tifffile.imread(
        path_to_data / f'data/ag{index_organoid}_norm.tif'
    )
    mask = tifffile.imread(
        path_to_data / f'mask/ag{index_organoid}_mask.tif'
    )

    

    sum_sigmas = sum(sigmas)
    im = np.zeros(
        (
            4*sum_sigmas+len(sigmas)+10*(len(sigmas)-1), 
            4*max(sigmas)+1
        )
    )

    for ind, (sigma, gamma) in enumerate(zip(sigmas, [1,1,1])):
        for i in range(4*sigma+1):
            for j in range(4*sigma+1):
                im[
                    i+np.cumsum([0]+(4*np.array(sigmas)+1).tolist())[ind] + 10*ind, 
                    j
                        ] = np.exp(-((i-2*sigma)**2 + (j-2*sigma)**2)/(2*sigma**2))
                # im[i,j] = np.exp(-((i-2*sigma)**2 + (j-2*sigma)**2)/(2*sigma**2))



        cell_density = tifffile.imread(
            path_to_data / f'cell_density/ag{index_organoid}_sigma{sigma}.tif'
        )
        volume_fraction = tifffile.imread(
            path_to_data / f'volume_fraction/ag{index_organoid}_sigma{sigma}.tif'
        )
        nuclear_volume = tifffile.imread(
            path_to_data / f'nuclear_volume/ag{index_organoid}_sigma{sigma}.tif'
        )
        if sigma == sigmas[0]:
            percs_density = np.percentile(cell_density[mask], [1,99])
            # print(f'index organoid {index_organoid} percs_density {percs_density*(10/0.621)**3}')
            # print(volume_fraction[mask].min(), volume_fraction[mask].max())
            percs_volume_fraction = np.percentile(volume_fraction[mask], [1,99])
            # print(f'index organoid {index_organoid} percs_volume_fraction {percs_volume_fraction}')
            percs_nuclear_volume = np.percentile(nuclear_volume[mask], [1,99])
            # print(f'index organoid {index_organoid} percs_nuclear_volume {percs_nuclear_volume * 0.621**3}')
        
        ### -->
        cmap = matplotlib.colormaps['inferno']
        imrgb = cell_density.copy()
        imrgb = np.clip((imrgb - percs_density[0])/(percs_density[1] - percs_density[0]), 0.001, 0.999)
        imrgb[mask.copy() == 0] = 0
        im_cmap_density = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap_density[i,j] = cmap(imrgb[i,j])
        
        viewer2.add_image(im_cmap_density, opacity=1)
        # viewer2.add_image(cell_density.copy(), colormap='blues', 
        #     opacity=1, contrast_limits=percs_density
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask*1)
        viewer2.layers[-1].contour=4
        ### -->
        cmap = matplotlib.colormaps['cividis']
        imrgb = volume_fraction.copy()
        imrgb = np.clip((imrgb - percs_volume_fraction[0])/(percs_volume_fraction[1] - percs_volume_fraction[0]), 0.001, 0.999)
        imrgb[mask.copy() == 0] = 0
        im_cmap_vf = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap_vf[i,j] = cmap(imrgb[i,j])

        viewer2.add_image(im_cmap_vf, opacity=1)
        # viewer2.add_image(volume_fraction.copy(), colormap='greens', 
        #     opacity=1, contrast_limits=percs_volume_fraction
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask*1)
        viewer2.layers[-1].contour=4
        ### -->
        cmap = matplotlib.colormaps['viridis']
        imrgb = nuclear_volume.copy()
        imrgb = np.clip((imrgb - percs_nuclear_volume[0])/(percs_nuclear_volume[1] - percs_nuclear_volume[0]), 0.001, 0.999)
        imrgb[mask.copy() == 0] = 0
        im_cmap_nv = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap_nv[i,j] = cmap(imrgb[i,j])

        viewer2.add_image(im_cmap_nv, opacity=1)
        # viewer2.add_image(nuclear_volume.copy(), colormap='reds', 
        #     opacity=1, contrast_limits=percs_nuclear_volume
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask*1)
        viewer2.layers[-1].contour=4

    viewer1.add_image(data, colormap='gray_r', contrast_limits=[0.14,0.5],name='data')
    viewer1.layers[-1].gamma = 0.75
    viewer1.add_labels(labels, opacity=1,name='labels')



    cmap = matplotlib.colormaps['inferno']
    im_cmap = np.zeros((im.shape[0], im.shape[1], 4))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] > 0:
                im_cmap[i,j] = cmap(im[i,j])

    im_cmap = im_cmap[:,::-1]

    im_cmap_big = np.zeros((mask.shape[0], mask.shape[1], 4))
    im_cmap_big[-im_cmap.shape[0]:, -im_cmap.shape[1]:] = im_cmap

    viewer1.add_image(im_cmap_big, opacity=1, colormap='inferno')

    viewer1.grid.enabled = True
    viewer1.grid.shape = (len(sigmas), 1)
    viewer1.grid.stride = -1
    viewer2.grid.enabled = True
    viewer2.grid.shape = (len(sigmas), -1)
    viewer2.grid.stride=-2


    ### VIEWER 3
    napari_true_strain = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}.npy'
    )
    lengths = np.linalg.norm(napari_true_strain[:,1], axis=1)
    napari_true_strain[:,1] = napari_true_strain[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_angles = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_angles.npy'
    )

    napari_true_strain_resampled = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_resampled.npy'
    )
    lengths = np.linalg.norm(napari_true_strain_resampled[:,1], axis=1)
    napari_true_strain_resampled[:,1] = napari_true_strain_resampled[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_resampled_angles = np.load(
        path_to_data / f'true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_resampled_angles.npy'
    )

    viewer3.add_image(data, colormap='gray_r', contrast_limits=[0.14,0.5],name='data')
    viewer3.layers[-1].gamma = 0.75
    viewer3.add_points(ndim=2)

    viewer3.add_points(ndim=2)
    viewer3.add_points(ndim=2)


    viewer3.add_vectors(napari_true_strain, name='true_strain',
                          properties={'angles': napari_true_strain_angles},
                          vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
                          length=1.5, edge_width=5)
    viewer3.add_points(ndim=2)
    
    viewer3.add_vectors(napari_true_strain_resampled, name='true_strain_resampled',
                            properties={'angles': napari_true_strain_resampled_angles},
                            vector_style='line', blending='translucent', opacity=1, edge_colormap=cmap_vectors_nematic,
                            length=1.5, edge_width=5)
    viewer3.add_points(ndim=2)

    viewer3.grid.enabled = True
    viewer3.grid.shape = (2, -1)
    viewer3.grid.stride = -2

    ### VIEWER 5 (snapshots of true strain)
    # viewer5.add_image(data[162:392, 65:291])


    ### VIEWER 4
    density_gradient = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}.npy'
    )
    lengths = np.linalg.norm(density_gradient[:,1], axis=1)
    density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

    density_gradient_angles = np.load(
        path_to_data / f'cell_density_gradient/ag{index_organoid}_angles.npy'
    )

    gradient_magnitude = tifffile.imread(
        path_to_data / f'cell_density_gradient_mag/ag{index_organoid}.tif'
    )

    ts_maxeig = tifffile.imread(
        path_to_data / f'true_strain_maxeig/ag{index_organoid}_sigma40.tif'
    )

    ts_maxeig_cellular = tifffile.imread(
        path_to_data / f'true_strain_maxeig/ag{index_organoid}.tif'
    )

    dot_product_map = tifffile.imread(
        path_to_data / f'dot_product_map/ag{index_organoid}.tif'
    )
    dot_product_map[~mask] = np.nan

    cls = np.percentile(ts_maxeig[mask], [1,99])
    # print(f'index organoid {index_organoid} percs_ts_maxeig {cls}')
    ts_maxeig[ts_maxeig==0] = np.nan

    viewer4.add_vectors(density_gradient, name='density_gradient',
                        properties={'angles': density_gradient_angles},
                        vector_style='triangle', blending='translucent', opacity=1, 
                        edge_colormap=cmap_vectors_gradient,
                        length=2, edge_width=10)
    viewer4.add_points(ndim=2)

    viewer4.add_points(ndim=2)
    viewer4.add_points(ndim=2)

    cls = np.percentile(gradient_magnitude[mask], [1,99])
    # print(f'index organoid {index_organoid} percs_gradient_magnitude {cls}')
    gradient_magnitude[gradient_magnitude==0] = np.nan
    viewer4.add_image(gradient_magnitude, name='gradient_magnitude', colormap='Reds')
    viewer4.add_points(ndim=2)

    viewer4.add_image(ts_maxeig, name='ts_maxeig', colormap='Blues')
    viewer4.add_points(ndim=2)


    viewer4.add_points(ndim=2)
    viewer4.add_points(ndim=2)


    
    viewer4.add_image(dot_product_map, name='dot_product_map', colormap='turbo')
    viewer4.add_points(ndim=2)

    viewer4.grid.enabled = True
    viewer4.grid.shape = (-1, 2)
    viewer4.grid.stride = -2


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
                


    viewer3.window.add_dock_widget(widget_angle_shift, area='right')
    viewer4.window.add_dock_widget(widget_angle_shift2, area='right')




napari.run()