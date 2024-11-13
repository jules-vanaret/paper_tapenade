from pyngs.dense_smoothing import gaussian_smooth_dense
from pyngs.sparse_smoothing import gaussian_smooth_sparse
import numpy as np
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm
from tapenade.analysis.additional_regionprops_properties import (
    add_ellipsoidal_nature_bool,
)
from tqdm.contrib.concurrent import process_map

path_to_data = "/home/jvanaret/data/data_paper_tapenade/morphology/processed"

# for index_organoid in tqdm(range(1, 9)):
compute_nuclear_volume = True
compute_cell_density = False
compute_volume_fraction = False
compute_deformation = False


def func(index_organoid):

    mask = tifffile.imread(f"{path_to_data}/ag{index_organoid}_mask.tif")

    labels = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm_labels.tif")

    # Compute the cell density
    props = regionprops(labels)
    centroids = np.array([prop.centroid for prop in props])

    centroids_data = np.zeros_like(mask, dtype=np.float32)
    centroids_data[
        centroids[:, 0].astype(int),
        centroids[:, 1].astype(int),
        centroids[:, 2].astype(int),
    ] = 1

    # Compute volume fraction
    volume_data = labels.astype(bool).astype(np.float32)

    # Compute deformation
    if compute_deformation:
        props = [add_ellipsoidal_nature_bool(prop, [1, 1, 1]) for prop in props]
        sparse_data = np.zeros((len(props), 4), dtype=np.float32)
        for i, prop in enumerate(props):
            sparse_data[i, :3] = prop.centroid
            sparse_data[i, 3] = prop.ellipsoidal_nature_bool * 1.0

        deformation_data = np.zeros_like(mask, dtype=np.float32)
        for prop in props:
            deformation_data[prop.slice][prop.image] = (
                prop.ellipsoidal_nature_bool * 1.0
            )

        tifffile.imwrite(
            f"{path_to_data}/prolate_vs_oblate/ag{index_organoid}.tif", deformation_data
        )

    if compute_nuclear_volume:
        nuclear_volume = np.zeros_like(mask, dtype=np.float32)
        for prop in props:
            nuclear_volume[prop.slice][prop.image] = prop.area

        tifffile.imwrite(
            f"{path_to_data}/nuclear_volume/ag{index_organoid}.tif", nuclear_volume
        )

        centroids_data_volumes = np.zeros_like(mask, dtype=np.float32)
        volumes = np.array([prop.area for prop in props])

        centroids_data_volumes[
            centroids[:, 0].astype(int),
            centroids[:, 1].astype(int),
            centroids[:, 2].astype(int),
        ] = volumes

    for sigma in [10, 20, 40]:

        if compute_cell_density:
            density = gaussian_smooth_dense(
                centroids_data,
                sigmas=sigma,
                mask=mask,
                dim_space=3,
                is_temporal=False,
                n_job=1,
            )
            tifffile.imwrite(
                f"{path_to_data}/cell_density/ag{index_organoid}_sigma{sigma}.tif",
                density,
            )

        if compute_volume_fraction:
            volume_fraction = gaussian_smooth_dense(
                volume_data,
                sigmas=sigma,
                mask=mask,
                dim_space=3,
                is_temporal=False,
                n_job=1,
            )
            tifffile.imwrite(
                f"{path_to_data}/volume_fraction/ag{index_organoid}_sigma{sigma}.tif",
                volume_fraction,
            )

        if compute_deformation:
            sparse_data_smoothed = gaussian_smooth_sparse(
                sparse_data, sigmas=sigma, dim_space=3, is_temporal=False, n_job=1
            )

            deformation_data = np.zeros_like(mask, dtype=np.float32)
            for i, prop in enumerate(props):
                deformation_data[prop.slice][prop.image] = sparse_data_smoothed[i, 3]

            tifffile.imwrite(
                f"{path_to_data}/prolate_vs_oblate/ag{index_organoid}_sigma{sigma}.tif",
                deformation_data,
            )

        if compute_nuclear_volume:
            nuclear_volume_smoothed = gaussian_smooth_dense(
                centroids_data_volumes,
                sigmas=sigma,
                mask=mask,
                mask_for_volume=centroids_data_volumes.astype(bool),
                dim_space=3,
                is_temporal=False,
                n_job=1,
            )
            tifffile.imwrite(
                f"{path_to_data}/nuclear_volume/ag{index_organoid}_sigma{sigma}.tif",
                nuclear_volume_smoothed,
            )


process_map(func, range(1, 9), max_workers=8, smoothing=0)
