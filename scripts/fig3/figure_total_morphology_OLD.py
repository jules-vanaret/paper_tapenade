import napari
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib
from pyngs.utils import get_napari_angles_cmap
from magicgui import magicgui
from skimage.morphology import binary_erosion

path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed/all_quantities_midplane'

cmap_vectors_nematic = get_napari_angles_cmap(color_mode='nematic')
cmap_vectors_gradient = get_napari_angles_cmap(color_mode='cyclic')
# inds_organoids = [1]
# zs = [173]
# inds_organoids = [2, 4, 5]
inds_organoids = [6]


sigmas = [10,20,40]




for index_organoid in tqdm(inds_organoids):
    
    viewer1 = napari.Viewer()
    viewer2 = napari.Viewer()
    viewer3 = napari.Viewer()
    viewer4 = napari.Viewer()

    labels = tifffile.imread(
        f'{path_to_data}/labels/ag{index_organoid}_norm_labels.tif'
    )
    data = tifffile.imread(
        f'{path_to_data}/data/ag{index_organoid}_norm.tif'
    )
    mask = tifffile.imread(
        f'{path_to_data}/mask/ag{index_organoid}_mask.tif'
    )

    

    sum_sigmas = sum(sigmas)
    im = np.zeros(
        (
            4*sum_sigmas+len(sigmas)+10*(len(sigmas)-1), 
            4*max(sigmas)+1
        )
    )
    print(im.shape)
    for ind, (sigma, gamma) in enumerate(zip(sigmas, [1,1,1])):
        for i in range(4*sigma+1):
            for j in range(4*sigma+1):
                im[
                    i+np.cumsum([0]+(4*np.array(sigmas)+1).tolist())[ind] + 10*ind, 
                    j
                        ] = np.exp(-((i-2*sigma)**2 + (j-2*sigma)**2)/(2*sigma**2))
                # im[i,j] = np.exp(-((i-2*sigma)**2 + (j-2*sigma)**2)/(2*sigma**2))



        cell_density = tifffile.imread(
            f'{path_to_data}/cell_density/ag{index_organoid}_sigma{sigma}.tif'
        )
        volume_fraction = tifffile.imread(
            f'{path_to_data}/volume_fraction/ag{index_organoid}_sigma{sigma}.tif'
        )
        nuclear_volume = tifffile.imread(
            f'{path_to_data}/nuclear_volume/ag{index_organoid}_sigma{sigma}.tif'
        )
        if sigma == sigmas[0]:
            percs_density = np.percentile(cell_density[mask], [1,99])
            print(f'index organoid {index_organoid} percs_density {percs_density*(10/0.621)**3}')
            print(volume_fraction[mask].min(), volume_fraction[mask].max())
            percs_volume_fraction = np.percentile(volume_fraction[mask], [1,99])
            print(f'index organoid {index_organoid} percs_volume_fraction {percs_volume_fraction}')
            percs_nuclear_volume = np.percentile(nuclear_volume[mask], [1,99])
            print(f'index organoid {index_organoid} percs_nuclear_volume {percs_nuclear_volume * 0.621**3}')
        
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
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}.npy'
    )
    lengths = np.linalg.norm(napari_true_strain[:,1], axis=1)
    napari_true_strain[:,1] = napari_true_strain[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_angles = np.load(
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_angles.npy'
    )

    napari_true_strain_resampled = np.load(
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_resampled.npy'
    )
    lengths = np.linalg.norm(napari_true_strain_resampled[:,1], axis=1)
    napari_true_strain_resampled[:,1] = napari_true_strain_resampled[:,1]/np.median(lengths[:,None]) * 5

    napari_true_strain_resampled_angles = np.load(
        f'{path_to_data}/true_strain_tensor/ag{index_organoid}_sigma{sigmas[0]}_resampled_angles.npy'
    )

    viewer3.add_vectors(napari_true_strain, name='true_strain',
                          properties={'angles': napari_true_strain_angles},
                          vector_style='line', blending='opaque', edge_colormap=cmap_vectors_nematic,
                          length=1.5, edge_width=5)
    viewer3.add_points(ndim=2)
    viewer3.add_vectors(napari_true_strain_resampled, name='true_strain_resampled',
                            properties={'angles': napari_true_strain_resampled_angles},
                            vector_style='line', blending='opaque', edge_colormap=cmap_vectors_nematic,
                            length=1.5, edge_width=5)
    viewer3.add_points(ndim=2)
    viewer3.add_image(data, name='data', colormap='gray_r', contrast_limits=[0.14,0.5], opacity=0.5)
    viewer3.add_vectors(napari_true_strain_resampled, name='true_strain_resampled',
                            properties={'angles': napari_true_strain_resampled_angles},
                            vector_style='line', blending='opaque', edge_colormap=cmap_vectors_nematic,
                            length=1.5, edge_width=5)
    
    viewer3.grid.enabled = True
    viewer3.grid.shape = (1, -1)
    viewer3.grid.stride = -2

    ### VIEWER 4
    density_gradient = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}.npy'
    )
    lengths = np.linalg.norm(density_gradient[:,1], axis=1)
    density_gradient[:,1] = density_gradient[:,1]/np.median(lengths[:,None]) * 5

    density_gradient_angles = np.load(
        f'{path_to_data}/cell_density_gradient/ag{index_organoid}_angles.npy'
    )

    gradient_magnitude = tifffile.imread(
        f'{path_to_data}/cell_density_gradient_mag/ag{index_organoid}.tif'
    )

    anisotropy_field = tifffile.imread(
        f'{path_to_data}/anisotropy_coefficient/ag{index_organoid}_sigma40.tif'
    )

    anisotropy_field_cellular = tifffile.imread(
        f'{path_to_data}/anisotropy_coefficient/ag{index_organoid}.tif'
    )

    ts_maxeig = tifffile.imread(
        f'{path_to_data}/true_strain_maxeig/ag{index_organoid}_sigma40.tif'
    )

    ts_maxeig_cellular = tifffile.imread(
        f'{path_to_data}/true_strain_maxeig/ag{index_organoid}.tif'
    )

    dot_product_map = tifffile.imread(
        f'{path_to_data}/dot_product_map/ag{index_organoid}.tif'
    )
    dot_product_map[~mask] = np.nan

    viewer4.add_image(im_cmap_density, name='density', opacity=0.7)
    viewer4.add_vectors(density_gradient, name='density_gradient',
                        properties={'angles': density_gradient_angles},
                        vector_style='triangle', blending='opaque', edge_colormap=cmap_vectors_gradient,
                        length=2, edge_width=10)
    
    cls = np.percentile(gradient_magnitude[mask], [1,99])
    print(f'index organoid {index_organoid} percs_gradient_magnitude {cls}')
    gradient_magnitude[gradient_magnitude==0] = np.nan
    viewer4.add_image(gradient_magnitude, name='gradient_magnitude', colormap='Reds')
    viewer4.add_points(ndim=2)
    
    # # cls = np.percentile(anisotropy_field[mask], [1,99])
    # cls = np.percentile(anisotropy_field_cellular[labels.astype(bool)], [1,99])
    # print(f'index organoid {index_organoid} percs_anisotropy_field {cls}')
    # anisotropy_field_cellular[~labels.astype(bool)] = np.nan
    # viewer4.add_image(anisotropy_field_cellular, name='anisotropy_field', colormap='Blues')
    # # viewer4.add_image(anisotropy_field, name='anisotropy_field', colormap='Blues')
    # viewer4.add_points(ndim=2)

    cls = np.percentile(ts_maxeig[mask], [1,99])
    print(f'index organoid {index_organoid} percs_ts_maxeig {cls}')
    ts_maxeig[ts_maxeig==0] = np.nan
    viewer4.add_image(ts_maxeig, name='ts_maxeig', colormap='Blues')
    cls = np.percentile(ts_maxeig_cellular[labels.astype(bool)], [1,99])
    print(f'index organoid {index_organoid} percs_ts_maxeig_cellular {cls}')
    ts_maxeig_cellular[~labels.astype(bool)] = np.nan
    viewer4.add_image(ts_maxeig_cellular, name='ts_maxeig_cellular', colormap='Blues')
    

    viewer4.add_points(ndim=2)
    viewer4.add_points(ndim=2)

    
    viewer4.add_image(dot_product_map, name='dot_product_map', colormap='turbo')
    viewer4.add_points(ndim=2)

    viewer4.add_image(data, name='data', colormap='gray_r', contrast_limits=[0.14,0.5], opacity=0.5)
    viewer4.layers[-1].gamma = 0.75
    viewer4.add_image(dot_product_map, name='dot_product_map', colormap='turbo', opacity=0.5)

    viewer4.grid.enabled = True
    viewer4.grid.shape = (2, -1)
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


# for l in viewer.layers:
#     if hasattr(l, 'gamma'):
#         l.gamma = float(l.gamma)
