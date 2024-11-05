import napari
import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib

path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'


# inds_organoids = [1]
# zs = [173]
inds_organoids = [6]
zs = [110]
# inds_organoids = [7]
# zs = [117]

sigmas = [10,20,40]
# gammas = [1, 1.25, 1.5]
gammas = [1, 1, 1]


for index_organoid, z in tqdm(zip(inds_organoids, zs), total=len(inds_organoids)):
    viewer1 = napari.Viewer()
    viewer2 = napari.Viewer()

    labels = tifffile.imread(
        f'{path_to_data}/ag{index_organoid}_norm_labels.tif'
    )
    data = tifffile.imread(
        f'{path_to_data}/ag{index_organoid}_norm.tif'
    )
    mask = tifffile.imread(
        f'{path_to_data}/ag{index_organoid}_mask.tif'
    )

    

    sum_sigmas = sum(sigmas)
    im = np.zeros(
        (
            4*sum_sigmas+len(sigmas)+10*(len(sigmas)-1), 
            4*max(sigmas)+1
        )
    )
    print(im.shape)
    for ind, (sigma, gamma) in enumerate(zip(sigmas, gammas[::-1])):
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
        imrgb = cell_density[z]
        imrgb = np.clip((imrgb - percs_density[0])/(percs_density[1] - percs_density[0]), 0.001, 0.999)
        imrgb[mask[z] == 0] = 0
        im_cmap = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap[i,j] = cmap(imrgb[i,j])
        
        viewer2.add_image(im_cmap, opacity=1)
        # viewer2.add_image(cell_density[z], colormap='blues', 
        #     opacity=1, contrast_limits=percs_density
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask[z]*1)
        viewer2.layers[-1].contour=1
        ### -->
        cmap = matplotlib.colormaps['cividis']
        imrgb = volume_fraction[z]
        imrgb = np.clip((imrgb - percs_volume_fraction[0])/(percs_volume_fraction[1] - percs_volume_fraction[0]), 0.001, 0.999)
        imrgb[mask[z] == 0] = 0
        im_cmap = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap[i,j] = cmap(imrgb[i,j])

        viewer2.add_image(im_cmap, opacity=1)
        # viewer2.add_image(volume_fraction[z], colormap='greens', 
        #     opacity=1, contrast_limits=percs_volume_fraction
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask[z]*1)
        viewer2.layers[-1].contour=1
        ### -->
        cmap = matplotlib.colormaps['viridis']
        imrgb = nuclear_volume[z]
        imrgb = np.clip((imrgb - percs_nuclear_volume[0])/(percs_nuclear_volume[1] - percs_nuclear_volume[0]), 0.001, 0.999)
        imrgb[mask[z] == 0] = 0
        im_cmap = np.zeros((imrgb.shape[0], imrgb.shape[1], 4))
        for i in range(imrgb.shape[0]):
            for j in range(imrgb.shape[1]):
                if imrgb[i,j] > 0:
                    im_cmap[i,j] = cmap(imrgb[i,j])

        viewer2.add_image(im_cmap, opacity=1)
        # viewer2.add_image(nuclear_volume[z], colormap='reds', 
        #     opacity=1, contrast_limits=percs_nuclear_volume
        # )
        viewer2.layers[-1].gamma = gamma
        viewer2.add_labels(mask[z]*1)
        viewer2.layers[-1].contour=1

    viewer1.add_image(data[z], colormap='gray_r', contrast_limits=[0.14,1],name='data')
    viewer1.layers[-1].gamma = 0.75
    viewer1.add_labels(labels[z], opacity=1,name='labels')



    cmap = matplotlib.colormaps['inferno']
    im_cmap = np.zeros((im.shape[0], im.shape[1], 4))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] > 0:
                im_cmap[i,j] = cmap(im[i,j])

    viewer1.add_image(im_cmap[::-1], opacity=1, colormap='inferno')




    viewer1.grid.enabled = True
    viewer1.grid.shape = (len(sigmas), 1)
    viewer2.grid.enabled = True
    viewer2.grid.shape = (len(sigmas), -1)
    viewer2.grid.stride=2

napari.run()


# for l in viewer.layers:
#     if hasattr(l, 'gamma'):
#         l.gamma = float(l.gamma)
