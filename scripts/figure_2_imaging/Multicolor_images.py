import napari
from pathlib import Path
import tifffile

folder = ...
viewer = napari.Viewer()
z_midplane = [155, 150, 135, 135, 110, 120, 130, 120]  # midplane of each sample
# samples = [1, 4, 5, 6, 8, 9, 10, 11]
samples = [1]
for index, num_sample in enumerate(samples):
    image = tifffile.imread(
        Path(folder) / f"2k_Hoechst_FoxA2_Oct4_Bra_78h/small/data/{num_sample}.tif"
    )
    hoechst = image[z_midplane[index], 0, :, :]
    foxa2 = image[z_midplane[index], 1, :, :]
    oct4 = image[z_midplane[index], 2, :, :]
    bra = image[z_midplane[index], 3, :, :]

    viewer.add_image(
        hoechst, name=f"hoechst_{num_sample}", colormap="gray", blending="additive"
    )
    viewer.add_image(
        foxa2, name=f"foxa2_{num_sample}", colormap="blue", blending="additive"
    )
    viewer.add_image(
        oct4, name=f"oct4_{num_sample}", colormap="green", blending="additive"
    )
    viewer.add_image(bra, name=f"bra_{num_sample}", colormap="red", blending="additive")

viewer.grid.enabled = True
viewer.grid.stride = -4  # nb of channels
viewer.grid.shape = (2, 4)  # 2 rows, 4 columns
napari.run()
