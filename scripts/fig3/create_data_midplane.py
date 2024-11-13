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
from tapenade.analysis.additional_regionprops_properties import (
    add_tensor_inertia,
    add_principal_lengths,
    add_true_strain_tensor,
)


path_to_data = "/home/jvanaret/data/data_paper_tapenade/morphology/processed"
name_folder = "all_quantities"
name_folder_midplane = "all_quantities_midplane"


for index_organoid in tqdm(range(1, 9)):

    mid_plane_ind = [181, 80, 80, 199, 98, 81, 81, 81][index_organoid - 1]

    mask = tifffile.imread(f"{path_to_data}/ag{index_organoid}_mask.tif")
    data = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm.tif")
    labels = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm_labels.tif")

    # mid_plane_ind = int(mask.shape[0] // 2)

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/mask/ag{index_organoid}_mask.tif",
        mask[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/data/ag{index_organoid}_norm.tif",
        data[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag{index_organoid}_norm_labels.tif",
        labels[mid_plane_ind],
    )
