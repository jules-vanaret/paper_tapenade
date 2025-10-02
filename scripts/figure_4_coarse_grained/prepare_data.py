import tifffile
from tqdm import tqdm
from pathlib import Path


path_to_data = Path(__file__).parents[2] / "data/4acd_data_morphology"
name_folder_midplane = "all_quantities_midplane"
# CHANGE THE INDICES IN THE LIST (from 1 to 8) IF YOU WANT TO DISPLAY MORE ORGANOIDS
indices_organoids = range(1, 9)#[6]


def main_processing_function(index_organoid):   

    mid_plane_ind = [181, 80, 80, 199, 98, 81, 81, 81][index_organoid - 1]
    orthogonal_mid_plane_slice = [
        (slice(None), 190), 
        (slice(None), 280), 
        (slice(None), 255), 
        (slice(None), 308), 
        (slice(None), 217), 
        (slice(None), slice(None), 245), 
        (slice(None), slice(None), 286), 
        (slice(None), 300)
    ][index_organoid - 1]


    data = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm.tif")

    mask = tifffile.imread(f"{path_to_data}/ag{index_organoid}_mask.tif")

    labels = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm_labels.tif")

    ### XY PLANE
    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/data/ag{index_organoid}_norm.tif",
        data[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/mask/ag{index_organoid}_mask.tif",
        mask[mid_plane_ind],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag{index_organoid}_norm_labels.tif",
        labels[mid_plane_ind],
    )

    ### ORTHOGONAL PLANE
    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/data/ag{index_organoid}_norm_ortho.tif",
        data[orthogonal_mid_plane_slice],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/mask/ag{index_organoid}_mask_ortho.tif",
        mask[orthogonal_mid_plane_slice],
    )

    tifffile.imwrite(
        f"{path_to_data}/{name_folder_midplane}/labels/ag{index_organoid}_norm_labels_ortho.tif",
        labels[orthogonal_mid_plane_slice],
    )
        

list(map(main_processing_function, tqdm(indices_organoids)))