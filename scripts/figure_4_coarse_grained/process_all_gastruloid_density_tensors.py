import numpy as np
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm
from tapenade.analysis.deformation.additional_regionprops_properties import (
    add_tensor_inertia,
    add_principal_lengths,
    add_true_strain_tensor,
)
from tapenade.preprocessing import masked_gaussian_smoothing_sparse, masked_gaussian_smoothing
from skimage.morphology import binary_erosion
import os


def func(index_organoid, path_to_data, name_folder_midplane):

    def napari_vectors_from_tensors(smoothed_tensors, apply_decoupling, nematic=False, return_angles=False):

        positions = smoothed_tensors[:, :3]
        smoothed_tensors = smoothed_tensors[:, 3:]
        smoothed_tensors = smoothed_tensors.reshape(-1, 3, 3)


        eigen_values, principal_vectors = np.linalg.eigh(smoothed_tensors)
        principal_vectors = principal_vectors.transpose(0,2,1)

        if apply_decoupling:
            axis_decoupling_matrix = np.ones((3,3))/2 - np.eye(3)
            # principal_lengths = np.sqrt(axis_decoupling_matrix @ eigen_values)
            eigen_values = np.sqrt(
                np.einsum('ij,lj->li', axis_decoupling_matrix, eigen_values)
            )

        napari_vectors = np.zeros((len(positions), 2, 3))
        indices_maxs = np.nanargmax(eigen_values, axis=1)
        principal_vectors = principal_vectors[np.arange(len(eigen_values)), indices_maxs]
        eigen_values = eigen_values[np.arange(len(eigen_values)), indices_maxs]
        napari_vectors[:,0] = positions
        napari_vectors[:,1] = principal_vectors * eigen_values.reshape(-1,1)

        if nematic:
            N_vec = napari_vectors.shape[0]
            napari_vectors = np.concatenate([napari_vectors, napari_vectors], axis=0)
            napari_vectors[N_vec:, 1] *= -1

        if return_angles:
            angles = np.arctan2(*(napari_vectors[:,1, -2:].reshape(-1, 2).T))

            angles = np.arctan2(np.sin(angles-1), np.cos(angles-1))

            return napari_vectors, angles



        return napari_vectors
    
    
    mid_plane_ind = [181, 80, 80, 199, 98, 81, 81, 81][index_organoid-1]

  
    density_dp = None
    true_strain_dp = None

    sigmas = [10, 20, 40]
    

    mask = tifffile.imread(
        f'{path_to_data}/ag{index_organoid}_mask.tif'
    )

    labels = tifffile.imread(
        f'{path_to_data}/ag{index_organoid}_norm_labels.tif'
    )


    positions_on_grid = np.mgrid[
        [slice(0, labels.shape[0], 10)]+[slice(0, dim, 20) for dim in labels.shape[1:]]
    ].reshape(labels.ndim, -1).T

    positions_on_grid = positions_on_grid[mask[positions_on_grid[:,0], positions_on_grid[:,1], positions_on_grid[:,2]]]


    # Compute the cell density
    props = regionprops(labels)
    centroids = np.array([prop.centroid for prop in props])

    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[
        centroids[:, 0].astype(int), 
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int)
    ] = 1

    # Compute volume fraction
    volume_data = labels.astype(bool).astype(np.float32)
    true_strain_maxeig = np.zeros_like(mask, dtype=np.float32)
    true_strain_maxeig_dense = np.zeros_like(mask, dtype=np.float32)

    nuclear_volume = np.zeros_like(mask, dtype=np.float32)

    centroids_data_volumes = np.zeros_like(mask, dtype=np.float32)
    volumes = np.array([prop.area for prop in props])

    centroids_data_volumes[
        centroids[:, 0].astype(int),
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int)
    ] = volumes

    sparse_inertia_tensors_data = np.zeros((len(props), 3+9))
    sparse_true_strain_tensors_data = np.zeros((len(props), 3+9))
    vectors = np.zeros((len(props), 2, 3))

    for i, prop in enumerate(props):
        add_tensor_inertia(prop, scale=(1,1,1))
        add_principal_lengths(prop, scale=(1,1,1), add_principal_vectors=True)
        add_true_strain_tensor(prop, scale=(1,1,1))

        centroid = prop.centroid
        principal_lengths = prop.principal_lengths

        nuclear_volume[prop.slice][prop.image] = prop.area

        denominator = np.power(np.product(principal_lengths),1/3)
        max_eig = np.log(principal_lengths[0]/denominator)

        true_strain_maxeig[prop.slice][prop.image] = max_eig
        true_strain_maxeig_dense[
            centroid[0].astype(int),
            centroid[1].astype(int),
            centroid[2].astype(int)
        ] = max_eig

        sparse_inertia_tensors_data[i, :3] = prop.centroid
        sparse_inertia_tensors_data[i, -9:] = prop.tensor_inertia.ravel()
        sparse_true_strain_tensors_data[i, :3] = prop.centroid
        sparse_true_strain_tensors_data[i, -9:] = prop.true_strain_tensor.ravel()
        vectors[i,1] = prop.principal_vectors[0] * prop.principal_lengths[0]
        vectors[i,0] = prop.centroid

    tifffile.imwrite(
        f'{path_to_data}/{name_folder_midplane}/true_strain_maxeig/ag{index_organoid}.tif',
        true_strain_maxeig[mid_plane_ind]
    )


    tifffile.imwrite(
        f'{path_to_data}/{name_folder_midplane}/nuclear_volume/ag{index_organoid}.tif',
        nuclear_volume[mid_plane_ind]
    )


    for sigma in sigmas:

        density = masked_gaussian_smoothing(
            centroids_data, sigmas=sigma, mask=mask, dim_space=3,
            is_temporal=False,
            n_job=-1
        )

        tifffile.imwrite(
            f'{path_to_data}/{name_folder_midplane}/cell_density/ag{index_organoid}_sigma{sigma}.tif',
            density[mid_plane_ind]
        )

        if sigma == 40:

            density_dp = density.copy()

            
            gradient_field = np.gradient(density_dp)
            gradient_field = np.array(gradient_field).transpose(1,2,3,0)
            gradient_field[~binary_erosion(mask)] = 0

            gradient_magnitude_field = np.linalg.norm(gradient_field, axis=-1)

            tifffile.imwrite(
                f'{path_to_data}/{name_folder_midplane}/cell_density_gradient_mag/ag{index_organoid}.tif',
                gradient_magnitude_field[mid_plane_ind]
            )

            gradient_on_grid = gradient_field[
                positions_on_grid[:,0].astype(int),
                positions_on_grid[:,1].astype(int),
                positions_on_grid[:,2].astype(int)
            ]

            napari_gradient_on_grid = np.zeros((len(positions_on_grid), 2, 3))

            napari_gradient_on_grid[:,0] = positions_on_grid
            napari_gradient_on_grid[:,1] = gradient_on_grid

            angles = np.arctan2(*(napari_gradient_on_grid[:,1, -2:].reshape(-1, 2).T))
            angles = np.arctan2(np.sin(angles-1), np.cos(angles-1))

            zs = napari_gradient_on_grid[:,0,0]
            zs_mask = np.abs(zs - mid_plane_ind) < 5

            napari_gradient_on_grid_midplane = napari_gradient_on_grid[zs_mask]

            np.save(
                f'{path_to_data}/{name_folder_midplane}/cell_density_gradient/ag{index_organoid}.npy',
                napari_gradient_on_grid_midplane[:,:,1:]
            )

            np.save(
                f'{path_to_data}/{name_folder_midplane}/cell_density_gradient/ag{index_organoid}_angles.npy',
                angles[zs_mask]
            )


            
        volume_fraction = masked_gaussian_smoothing(
            volume_data, sigmas=sigma, mask=mask, dim_space=3,
            is_temporal=False,
            n_job=-1
        )

        tifffile.imwrite(
            f'{path_to_data}/{name_folder_midplane}/volume_fraction/ag{index_organoid}_sigma{sigma}.tif',
            volume_fraction[mid_plane_ind]
        )
    

        data_true_strain_maxeig_dense = masked_gaussian_smoothing(
            data_dense = true_strain_maxeig_dense,
            is_temporal=False,
            dim_space=3,
            mask=mask,
            mask_for_volume=centroids_data.astype(bool),
            sigmas=sigma,
        )

        tifffile.imwrite(
            f'{path_to_data}/{name_folder_midplane}/true_strain_maxeig/ag{index_organoid}_sigma{sigma}.tif',
            data_true_strain_maxeig_dense[mid_plane_ind]
        )

        nuclear_volume_smoothed = masked_gaussian_smoothing(
            centroids_data_volumes, sigmas=sigma, mask=mask, mask_for_volume=centroids_data_volumes.astype(bool),
            dim_space=3, is_temporal=False, n_job=-1
        )

        tifffile.imwrite(
            f'{path_to_data}/{name_folder_midplane}/nuclear_volume/ag{index_organoid}_sigma{sigma}.tif',
            nuclear_volume_smoothed[mid_plane_ind]
        )

            

        ### TRUE STRAIN TENSOR

        smoothed_true_strain_tensors_data = masked_gaussian_smoothing_sparse(
            sparse_true_strain_tensors_data, is_temporal=False, sigmas=sigma,
            dim_space=3#, positions=positions
        ) # remove the positions

        napari_vectors_true_strain, angles_vectors_true_strain = napari_vectors_from_tensors(
            smoothed_true_strain_tensors_data,
            apply_decoupling=False,
            nematic=True,
            return_angles=True
        )

        if sigma == 10:
            true_strain_dp = napari_vectors_true_strain.copy()
            true_strain_dp = true_strain_dp[:int(true_strain_dp.shape[0]//2)]

        smoothed_true_strain_tensors_data_midplane = smoothed_true_strain_tensors_data.copy()
        zs = smoothed_true_strain_tensors_data_midplane[:,0]
        zs_mask = np.abs(zs - mid_plane_ind) < 5
        smoothed_true_strain_tensors_data_midplane = smoothed_true_strain_tensors_data_midplane[zs_mask]
        
        napari_vectors_true_strain_midplane, angles_vectors_true_strain_midplane = napari_vectors_from_tensors(
            smoothed_true_strain_tensors_data_midplane,
            apply_decoupling=False,
            nematic=True,
            return_angles=True
        )

        np.save(
            f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}.npy',
            napari_vectors_true_strain_midplane[:,:,1:]
        )

        np.save(
            f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_angles.npy',
            angles_vectors_true_strain_midplane
        )


        ### TRUE STRAIN TENSOR RESAMPLED

        sparse_true_strain_tensors_data_2d = sparse_true_strain_tensors_data.copy()

        smoothed_true_strain_tensors_data_2d = masked_gaussian_smoothing_sparse(
            sparse_true_strain_tensors_data_2d, is_temporal=False, sigmas=sigma,
            dim_space=3, positions=positions_on_grid
        )

        smoothed_true_strain_tensors_data_2d_midplane = smoothed_true_strain_tensors_data_2d.copy()
        zs = smoothed_true_strain_tensors_data_2d_midplane[:,0]
        zs_mask = np.abs(zs - mid_plane_ind) < 5
        smoothed_true_strain_tensors_data_2d_midplane = smoothed_true_strain_tensors_data_2d_midplane[zs_mask]

        napari_vectors_true_strain_2d_midplane, angles_vectors_true_strain_2d_midplane = napari_vectors_from_tensors(
            smoothed_true_strain_tensors_data_2d_midplane,
            apply_decoupling=False,
            nematic=True,
            return_angles=True
        )

        np.save(
            f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled.npy',
            napari_vectors_true_strain_2d_midplane[:,:,1:]
        )

        np.save(
            f'{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled_angles.npy',
            angles_vectors_true_strain_2d_midplane
        )
        ###

    try:
        del density
    except:
        pass


    gradient_field = np.gradient(density_dp)
    gradient_field = np.array(gradient_field).transpose(1,2,3,0)
    gradient_field[~binary_erosion(mask)] = 0

    positions = true_strain_dp[:, 0]
    true_strain_dp_values = true_strain_dp[:, 1]

    gradient_field_at_centroids = gradient_field[
        positions[:,0].astype(int),
        positions[:,1].astype(int),
        positions[:,2].astype(int)
    ]

    del gradient_field

    # normalize
    gradient_field_at_centroids /= np.linalg.norm(gradient_field_at_centroids, axis=1).reshape(-1,1)
    true_strain_dp_values /= np.linalg.norm(true_strain_dp_values, axis=1).reshape(-1,1)


    dot_product_map_sparse = np.sum(
        gradient_field_at_centroids * true_strain_dp_values, axis=1
    )
    dot_product_map_sparse = dot_product_map_sparse**2

    dot_product_map = np.zeros_like(mask, dtype=np.float32)

    dot_product_map[
        positions[:,0].astype(int),
        positions[:,1].astype(int),
        positions[:,2].astype(int)
    ] = dot_product_map_sparse

    dot_product_map_dense = masked_gaussian_smoothing(
        dot_product_map, sigmas=30, mask=mask, mask_for_volume=centroids_data.astype(bool),
        dim_space=3, is_temporal=False, n_job=-1
    )

    tifffile.imwrite(
        f'{path_to_data}/{name_folder_midplane}/dot_product_map/ag{index_organoid}.tif',
        dot_product_map_dense[mid_plane_ind]
    )


if __name__ == '__main__':

    path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'
    name_folder_midplane = 'all_quantities_midplane'

    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'cell_density'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'cell_density_gradient'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'cell_density_gradient_mag'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'true_strain_maxeig'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'nuclear_volume'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'volume_fraction'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'true_strain_tensor'), exist_ok=True
    )
    os.mkdir(
        os.path.join(path_to_data,name_folder_midplane,'dot_product_map'), exist_ok=True
    )

    # from tqdm.contrib.concurrent import process_map
    # process_map(func, range(1, 9), max_workers=8, smoothing=0)
    map(func, tqdm(range(1, 9)))






