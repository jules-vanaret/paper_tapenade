# This script precomputes data to be displayed by the script 
# "figure_4_coarse_grained/figure_morphology.py"
# The data is stored as .tif and .npy files in the folder
# "4acd_data_morphology/all_quantities_midplane"
# The Zenodo dataset already contains the precomputed data, so this script does not need to be run again.

import numpy as np
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm
from pathlib import Path
from tapenade.analysis.deformation.additional_regionprops_properties import (
    add_tensor_inertia,
    add_principal_lengths,
    add_true_strain_tensor,
)
from tapenade.preprocessing import (
    masked_gaussian_smoothing,
    masked_gaussian_smooth_sparse,
)
from skimage.morphology import binary_erosion


path_to_data = Path(__file__).parents[2] / "data/4acd_data_morphology"
name_folder_midplane = "all_quantities_midplane"
# CHANGE THE INDICES IN THE LIST (from 1 to 8) IF YOU WANT TO DISPLAY MORE ORGANOIDS
indices_organoids = [6]
sigmas = [10, 20, 40]  # voxels


def napari_vectors_from_tensors(
    smoothed_tensors, apply_decoupling, nematic=False, return_angles=False
):

    positions = smoothed_tensors[:, :3]
    smoothed_tensors = smoothed_tensors[:, 3:]
    smoothed_tensors = smoothed_tensors.reshape(-1, 3, 3)

    eigen_values, principal_vectors = np.linalg.eigh(smoothed_tensors)
    principal_vectors = principal_vectors.transpose(0, 2, 1)

    if (
        apply_decoupling
    ):  # specifically for the inertia tensor, not the true strain tensor
        axis_decoupling_matrix = np.ones((3, 3)) / 2 - np.eye(3)
        eigen_values = np.sqrt(
            np.einsum("ij,lj->li", axis_decoupling_matrix, eigen_values)
        )

    napari_vectors = np.zeros((len(positions), 2, 3))
    indices_maxs = np.nanargmax(eigen_values, axis=1)
    principal_vectors = principal_vectors[np.arange(len(eigen_values)), indices_maxs]
    eigen_values = eigen_values[np.arange(len(eigen_values)), indices_maxs]
    napari_vectors[:, 0] = positions
    napari_vectors[:, 1] = principal_vectors * eigen_values.reshape(-1, 1)

    if nematic:  # then double the vectors, and flip the second half, just for display
        N_vec = napari_vectors.shape[0]
        napari_vectors = np.concatenate([napari_vectors, napari_vectors], axis=0)
        napari_vectors[N_vec:, 1] *= -1

    if return_angles:  # correctly unwrap the angles
        angles = np.arctan2(*(napari_vectors[:, 1, -2:].reshape(-1, 2).T))
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        return napari_vectors, angles

    return napari_vectors


def process_true_strain_maxeig(
    mask, props, path_to_data, name_folder_midplane, index_organoid, mid_plane_ind
):

    true_strain_maxeig = np.zeros_like(mask, dtype=np.float32)

    for prop in props:
        principal_lengths = prop.principal_lengths

        denominator = np.power(np.prod(principal_lengths), 1 / 3)
        max_eig = np.log(principal_lengths[0] / denominator)

        true_strain_maxeig[prop.slice][prop.image] = max_eig

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/true_strain_maxeig/ag{index_organoid}.tif",
        true_strain_maxeig[mid_plane_ind],
    )

    return true_strain_maxeig


def process_cell_density_sigma(
    mask,
    centroids,
    sigma,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):

    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[
        centroids[:, 0].astype(int),
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int),
    ] = 1

    density = masked_gaussian_smoothing(
        centroids_data,
        sigmas=sigma,
        mask=mask,
        n_jobs=-1,
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/cell_density/ag{index_organoid}_sigma{sigma}.tif",
        density[mid_plane_ind],
    )

    return density


def process_density_gradient(
    mask,
    density,
    positions_on_grid,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):
    gradient_field = np.gradient(density)
    gradient_field = np.array(gradient_field).transpose(1, 2, 3, 0)
    gradient_field[~binary_erosion(mask)] = 0

    gradient_magnitude_field = np.linalg.norm(gradient_field, axis=-1)

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/cell_density_gradient_mag/ag{index_organoid}.tif",
        gradient_magnitude_field[mid_plane_ind],
    )

    gradient_on_grid = gradient_field[
        positions_on_grid[:, 0].astype(int),
        positions_on_grid[:, 1].astype(int),
        positions_on_grid[:, 2].astype(int),
    ]

    napari_gradient_on_grid = np.zeros((len(positions_on_grid), 2, 3))

    napari_gradient_on_grid[:, 0] = positions_on_grid
    napari_gradient_on_grid[:, 1] = gradient_on_grid

    angles = np.arctan2(*(napari_gradient_on_grid[:, 1, -2:].reshape(-1, 2).T))
    angles = np.arctan2(np.sin(angles - 1), np.cos(angles - 1))

    zs = napari_gradient_on_grid[:, 0, 0]
    zs_mask = np.abs(zs - mid_plane_ind) < 5

    napari_gradient_on_grid_midplane = napari_gradient_on_grid[zs_mask]

    np.save(
        f"{path_to_data}/{name_folder_midplane}/cell_density_gradient/ag{index_organoid}.npy",
        napari_gradient_on_grid_midplane[:, :, 1:],
    )

    np.save(
        f"{path_to_data}/{name_folder_midplane}/cell_density_gradient/ag{index_organoid}_angles.npy",
        angles[zs_mask],
    )

    return gradient_field


def process_volume_fraction_sigma(
    mask,
    labels,
    sigma,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):
    volume_data = labels.astype(bool).astype(np.float32)
    volume_fraction = masked_gaussian_smoothing(
        volume_data, sigmas=sigma, mask=mask, n_jobs=-1
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/volume_fraction/ag{index_organoid}_sigma{sigma}.tif",
        volume_fraction[mid_plane_ind],
    )


def process_nuclear_volume_sigma(
    mask,
    centroids,
    volumes,
    sigma,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):
    centroids_data_volumes = np.zeros_like(mask, dtype=np.float32)

    centroids_data_volumes[
        centroids[:, 0].astype(int),
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int),
    ] = volumes

    nuclear_volume_smoothed = masked_gaussian_smoothing(
        centroids_data_volumes,
        sigmas=sigma,
        mask=mask,
        mask_for_volume=centroids_data_volumes.astype(bool),
        n_jobs=-1,
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/nuclear_volume/ag{index_organoid}_sigma{sigma}.tif",
        nuclear_volume_smoothed[mid_plane_ind],
    )


def process_true_strain(
    props,
    positions_on_grid,
    sigma,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):
    sparse_true_strain_tensors_data = np.zeros((len(props), 3 + 9))
    vectors = np.zeros((len(props), 2, 3))

    for i, prop in enumerate(props):
        sparse_true_strain_tensors_data[i, :3] = prop.centroid
        sparse_true_strain_tensors_data[i, -9:] = prop.true_strain_tensor.ravel()
        vectors[i, 1] = prop.principal_vectors[0] * prop.principal_lengths[0]
        vectors[i, 0] = prop.centroid

    smoothed_true_strain_tensors_data = masked_gaussian_smooth_sparse(
        sparse_true_strain_tensors_data,
        is_temporal=False,
        sigmas=sigma,
        dim_space=3,
    )

    napari_vectors_true_strain, _ = napari_vectors_from_tensors(
        smoothed_true_strain_tensors_data,
        apply_decoupling=False,
        nematic=True,
        return_angles=True,
    )

    smoothed_true_strain_tensors_data_midplane = (
        smoothed_true_strain_tensors_data.copy()
    )
    zs = smoothed_true_strain_tensors_data_midplane[:, 0]
    zs_mask = np.abs(zs - mid_plane_ind) < 5
    smoothed_true_strain_tensors_data_midplane = (
        smoothed_true_strain_tensors_data_midplane[zs_mask]
    )

    napari_vectors_true_strain_midplane, angles_vectors_true_strain_midplane = (
        napari_vectors_from_tensors(
            smoothed_true_strain_tensors_data_midplane,
            apply_decoupling=False,
            nematic=True,
            return_angles=True,
        )
    )

    np.save(
        f"{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}.npy",
        napari_vectors_true_strain_midplane[:, :, 1:],
    )

    np.save(
        f"{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_angles.npy",
        angles_vectors_true_strain_midplane,
    )

    ### TRUE STRAIN TENSOR RESAMPLED
    sparse_true_strain_tensors_data_2d = sparse_true_strain_tensors_data.copy()

    smoothed_true_strain_tensors_data_2d = masked_gaussian_smooth_sparse(
        sparse_true_strain_tensors_data_2d,
        is_temporal=False,
        sigmas=sigma,
        dim_space=3,
        positions=positions_on_grid,  # resample the tensors on the grid
    )

    smoothed_true_strain_tensors_data_2d_midplane = (
        smoothed_true_strain_tensors_data_2d.copy()
    )
    zs = smoothed_true_strain_tensors_data_2d_midplane[:, 0]
    zs_mask = np.abs(zs - mid_plane_ind) < 5
    smoothed_true_strain_tensors_data_2d_midplane = (
        smoothed_true_strain_tensors_data_2d_midplane[zs_mask]
    )

    napari_vectors_true_strain_2d_midplane, angles_vectors_true_strain_2d_midplane = (
        napari_vectors_from_tensors(
            smoothed_true_strain_tensors_data_2d_midplane,
            apply_decoupling=False,
            nematic=True,
            return_angles=True,
        )
    )

    np.save(
        f"{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled.npy",
        napari_vectors_true_strain_2d_midplane[:, :, 1:],
    )

    np.save(
        f"{path_to_data}/{name_folder_midplane}/true_strain_tensor/ag{index_organoid}_sigma{sigma}_resampled_angles.npy",
        angles_vectors_true_strain_2d_midplane,
    )

    return napari_vectors_true_strain


def process_dotproduct(
    mask,
    density_gradient_dp,
    true_strain_dp,
    centroids,
    path_to_data,
    name_folder_midplane,
    index_organoid,
    mid_plane_ind,
):

    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[
        centroids[:, 0].astype(int),
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int),
    ] = 1

    positions = true_strain_dp[:, 0]
    true_strain_dp_values = true_strain_dp[:, 1]

    gradient_field_at_centroids = density_gradient_dp[
        positions[:, 0].astype(int),
        positions[:, 1].astype(int),
        positions[:, 2].astype(int),
    ]

    del density_gradient_dp

    # normalize
    gradient_field_at_centroids /= np.linalg.norm(
        gradient_field_at_centroids, axis=1
    ).reshape(-1, 1)
    true_strain_dp_values /= np.linalg.norm(true_strain_dp_values, axis=1).reshape(-1, 1)

    dot_product_map_sparse = np.sum(
        gradient_field_at_centroids * true_strain_dp_values, axis=1
    )

    # the dot product is already the cosine of the angle between the two vectors
    dot_product_map_sparse = dot_product_map_sparse**2

    dot_product_map = np.zeros_like(mask, dtype=np.float32)

    dot_product_map[
        positions[:, 0].astype(int),
        positions[:, 1].astype(int),
        positions[:, 2].astype(int),
    ] = dot_product_map_sparse

    dot_product_map_dense = masked_gaussian_smoothing(
        dot_product_map,
        sigmas=sigmas[-1] - sigmas[0],
        mask=mask,
        mask_for_volume=centroids_data.astype(bool),
        n_jobs=-1,
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/dot_product_map/ag{index_organoid}.tif",
        dot_product_map_dense[mid_plane_ind],
    )


def main_processing_function(index_organoid):
    # mid_plane_ind = int(mask.shape[0] // 2)
    mid_plane_ind = [181, 80, 80, 199, 98, 81, 81, 81][index_organoid - 1]

    save_kwargs = dict(
        path_to_data=path_to_data,
        name_folder_midplane=name_folder_midplane,
        index_organoid=index_organoid,
        mid_plane_ind=mid_plane_ind,
    )

    mask = tifffile.imread(f"{path_to_data}/ag{index_organoid}_mask.tif")

    labels = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm_labels.tif")

    positions_on_grid = (
        np.mgrid[
            [slice(0, labels.shape[0], 10)]
            + [slice(0, dim, 20) for dim in labels.shape[1:]]
        ]
        .reshape(labels.ndim, -1)
        .T
    )

    positions_on_grid = positions_on_grid[
        np.where(mask[(positions_on_grid[:, 0], positions_on_grid[:, 1], positions_on_grid[:, 2])])
    ]

    # Compute the cell density
    props = regionprops(labels)

    for prop in props:
        add_principal_lengths(prop, scale=(1, 1, 1), add_principal_vectors=True)
        add_tensor_inertia(prop, scale=(1, 1, 1))
        add_true_strain_tensor(prop, scale=(1, 1, 1))

    process_true_strain_maxeig(mask, props, **save_kwargs)

    volumes = np.array([prop.area for prop in props])
    centroids = np.array([prop.centroid for prop in props])
    

    # loop over all scales
    for sigma in sigmas:
        density = process_cell_density_sigma(mask, centroids, sigma, **save_kwargs)

        if sigma == sigmas[-1]:  # tissue scale
            # store density field gradient to be used for the dot product map
            density_gradient_dp = process_density_gradient(
                mask, density, positions_on_grid, **save_kwargs
            )
        del density

        process_volume_fraction_sigma(mask, labels, sigma, **save_kwargs)

        process_nuclear_volume_sigma(mask, centroids, volumes, sigma, **save_kwargs)

        true_strain = process_true_strain(
            props, positions_on_grid, sigma, **save_kwargs
        )

        if sigma == sigmas[0]:  # cell scale
            true_strain_dp = true_strain.copy()
            # remove the second half, which is just a flipped version of the first half for display purposes
            true_strain_dp = true_strain_dp[: int(true_strain_dp.shape[0] // 2)]
        del true_strain

    process_dotproduct(
        mask, density_gradient_dp, true_strain_dp, centroids, **save_kwargs
    )


list(map(main_processing_function, tqdm(indices_organoids)))
