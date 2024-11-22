# Compute and plot radial profile of intensity of T-Bra during the differentiation wave, in 3D

import tifffile
import napari
from scipy import ndimage as ndi
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np
import pandas as pd
import seaborn as sns
from tapenade.preprocessing import isotropize_and_normalize

folder = ...
# plot 5d
path_dataset1 = Path(folder) / "2k_Hoechst_FoxA2_Oct4_Bra_78h/big"
path_dataset2 = Path(folder) / "2k_Hoechst_FoxA2_Oct4_Bra_78h/small"
path_dataset3 = Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re"


def compute_edt_from_mask(mask, scale):
    """
    Compute the euclidean distance transform from a mask, that will be used to compute distances from each point to the border of the gastruloid
    mask : mask of the gastruloid
    scale : scale of the image
    """
    rg = regionprops(mask.astype(int))
    for props in rg:
        volume = props.area
    if mask.ndim == 3:
        diameter = 2 * ((3 * volume / (4 * np.pi)) ** (1 / 3))
    else:
        diameter = 2 * ((volume / (np.pi)) ** (1 / 2))
    edt = ndi.distance_transform_edt(mask, return_distances=True, sampling=scale)
    edt = edt * mask
    return (diameter, edt)


def radial_intensities_one_image(
    signal,
    mask,
    scale=(1, 1, 1),
    number_layers=5,
    return_df: bool = False,
    num_sample: int = 0,
):
    """
    Compute the radial distribution of the intensity of a given image and return it into a dataframe, one intensity per layer
    signal : 3D map obtained with the proliferation code. nans outside and isotropic (1,1,1)
    scale : scale of the image
    number_layers : number of layers in which the image will be divided
    return_df : if True, return a dataframe with the mean intensity of each layer
    num_sample : number of the sample (needed to classify in the dataframe)
    """
    df = pd.DataFrame(columns=["Sample", "Layer", "Mean"])
    diam, edt = compute_edt_from_mask(mask, scale=scale)
    radius = np.max(edt)
    layers = np.linspace(0, radius, number_layers + 1, dtype=int)
    Intensity = []
    # viewer=napari.Viewer()
    for ind in range(len(layers) - 1):
        layer_mask = np.logical_and(edt > layers[ind], edt < layers[ind + 1])
        # viewer.add_labels(layer_mask)
        signal_masked = (np.copy(signal)).astype(float)
        signal_masked[layer_mask == 0] = np.nan
        # viewer.add_image(signal_masked)
        Intensity.append(np.nanmean(signal_masked))
        if return_df:
            radial_pos = ind / number_layers
            df = df._append(
                {
                    "Sample": num_sample,
                    "Layer": radial_pos,
                    "Mean": np.nanmean(signal_masked),
                },
                ignore_index=True,
            )
    if return_df:
        return df
    else:
        return Intensity


def compute_df_fromfolder(
    folder_data,
    df,
    scale=(1, 1, 1),
    sigma: float = 25,
    number_layers: int = 5,
):
    """
    Compute the radial distribution of the intensity of a given image and add it to a dataframe
    folder_data : path to the folder containing the images
    df : dataframe to add the results
    scale : scale of the input image (they will be isotropized)
    number_layers : number of layers to compute the radial intensity
    """
    paths = sorted(glob(str(Path(folder_data) / "*.tif")))
    samples = []
    for path in paths:
        samples.append(Path(path).stem)
    for indnum, num in enumerate(samples):
        im = tifffile.imread(Path(folder_data) / f"{num}.tif")
        mask = np.invert(np.isnan(im))
        radial_distrib = radial_intensities_one_image(
            im,
            mask=mask,
            scale=scale,
            number_layers=number_layers,
            return_df=True,
            num_sample=num,
        )
        df = pd.concat([df, radial_distrib])
    return df


scale = (1,1,1)
number_layers = 5

#input data : proliferation maps from the proliferation code : nan outside and isotropic
df_1 = compute_df_fromfolder(
    folder_data=Path(path_dataset1) / "foxa2_fraction",
    df=pd.DataFrame(columns=["Sample", "Layer", "Mean"]),
    scale=scale,
    sigma=25,
    number_layers=number_layers,
)
df_2 = compute_df_fromfolder(
    folder_data=Path(path_dataset2) / "foxa2_fraction",
    df=pd.DataFrame(columns=["Sample", "Layer", "Mean"]),
    scale=scale,
    sigma=25,
    number_layers=number_layers,
)

fig = plt.figure(figsize=(14, 10))
sns.lineplot(data=df_1, x="Layer", y="Mean", ci="sd", linewidth=3, color="plum")
sns.lineplot(
    data=df_2, x="Layer", y="Mean", linewidth=3, ci="sd", color="mediumaquamarine"
)

plt.xlabel("Relative distance to the border", fontsize=30)
plt.ylabel("Intensity", fontsize=30)
plt.legend(fontsize=30)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=30)
plt.yticks([0.03,0.04,0.05,0.06,0.07,0.08], fontsize=30)
plt.show()
fig.savefig(Path(folder) / "5d_plot.svg")
