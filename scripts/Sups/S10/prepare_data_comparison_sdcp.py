# T

import tifffile
from tqdm import tqdm
from pathlib import Path


path_to_data = Path(__file__).parents[3] / "data/S10_comparison_sdcp"
name_folder_midplane = "all_quantities_midplane"



def main_processing_function():

    mid_plane_ind = 81
    orthogonal_mid_plane_slice = (slice(None), slice(None), 245)


    data = tifffile.imread(f"{path_to_data}/ag6_norm.tif")

    mask = tifffile.imread(f"{path_to_data}/ag6_mask.tif")

    labels = tifffile.imread(f"{path_to_data}/ag6_norm_labels.tif")
    labels_cellpose = tifffile.imread(f"{path_to_data}/ag6_norm_labels_cellposesam.tif")

    ### XY PLANE
    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/data/ag6_norm.tif",
        data[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/mask/ag6_mask.tif",
        mask[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag6_norm_labels.tif",
        labels[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag6_norm_labels_cellposesam.tif",
        labels_cellpose[mid_plane_ind],
    )

    ### ORTHOGONAL PLANE
    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/data/ag6_norm_ortho.tif",
        data[orthogonal_mid_plane_slice],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/mask/ag6_mask_ortho.tif",
        mask[orthogonal_mid_plane_slice],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag6_norm_labels_ortho.tif",
        labels[orthogonal_mid_plane_slice],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag6_norm_labels_cellposesam_ortho.tif",
        labels_cellpose[orthogonal_mid_plane_slice],
    )
        
main_processing_function()