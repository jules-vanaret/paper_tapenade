# Plot intensity as a function of depth for Hoechst, non normalized and normalized by Hoechst intensity.

import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tapenade.preprocessing._preprocessing import (
    change_array_pixelsize,
    normalize_intensity,
)
from tqdm import tqdm





folder = Path(__file__).parents[3] / "data"
list_name = ['#1_Hoechst_FoxA2_Oct4_Bra_78h_big_5','#2_Hoechst_FoxA2_Oct4_Bra_78h_small_4','#3_Dapi_Ecad_bra_sox2_725h_re_6_reg_0.6','#4_48_12h_Hoechst_Ecad_Bra_Sox2_2']
for name in tqdm(list_name):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    hoechst = tifffile.imread(Path(folder) / f"S8d_robustness_normalization/{name}_hoechst.tif")
    mask = (tifffile.imread(Path(folder) / f"S8d_robustness_normalization/{name}_mask.tif")).astype(bool)

    sigma = 20
    scale = (1, 0.6, 0.6)

    hoechst_iso = change_array_pixelsize(array=hoechst, input_pixelsize=scale)
    mask_iso = change_array_pixelsize(array=mask, input_pixelsize=scale, order=0)

    hoechst_norm = normalize_intensity(
        image=hoechst_iso,
        ref_image=hoechst_iso,
        mask=mask_iso.astype(bool),
        sigma=sigma,
        image_wavelength=405,
    )

    hoechst_nan = np.where(mask_iso == 1, hoechst_iso, np.nan).astype(float)

    Int_hoechst_non_norm = []
    Int_hoechst_norm = []
    for z in range(len(mask)):
        Int_hoechst_non_norm.append(np.nanmedian(hoechst_nan[z, :, :]))
        Int_hoechst_norm.append(np.nanmedian(hoechst_norm[z, :, :]))
    Int_hoechst_non_norm = [i for i in Int_hoechst_non_norm if np.isnan(i) == False]
    Int_hoechst_norm = [i for i in Int_hoechst_norm if np.isnan(i) == False]

    ax.plot(Int_hoechst_norm, label="normalized", color="#009988", linewidth=4)
    ax.plot(
        Int_hoechst_non_norm,
        label="not normalized",
        color="#009988",
        linewidth=4,
        linestyle="dashed",
    )

    ax.set_xlabel("Depth (µm)", fontsize=30)
    ax.set_ylabel("Median intensity in \n Hoechst (A.U)", fontsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.tick_params(axis="x", labelsize=30)
    ax.legend(fontsize=25)
    plt.legend()
    plt.tight_layout()
    # fig.savefig(Path(folder)/f"S8d_{name}_plot.svg")
plt.show()
