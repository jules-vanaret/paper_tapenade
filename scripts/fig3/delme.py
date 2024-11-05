
import numpy as np
import tifffile
import napari
from tqdm import tqdm
from magicgui import magicgui



path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'

name_folder = 'all_quantities'
name_folder_midplane = 'all_quantities_midplane'



for index_gastruloid in tqdm(range(1,9)):

    dpm = tifffile.imread(
        f'{path_to_data}/{name_folder_midplane}/dot_product_map/ag{index_gastruloid}.tif'
    )
    dpm[dpm==0.5] = 0
    tifffile.imwrite(
        f'{path_to_data}/{name_folder_midplane}/dot_product_map/ag{index_gastruloid}.tif',
        dpm
    )

    dpm = tifffile.imread(
        f'{path_to_data}/{name_folder}/dot_product_map/ag{index_gastruloid}.tif'
    )
    dpm[dpm==0.5] = 0

    tifffile.imwrite(
        f'{path_to_data}/{name_folder}/dot_product_map/ag{index_gastruloid}.tif',
        dpm
    )
    