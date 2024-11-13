import numpy as np
import napari


colormaps = [
    "GrBu",
    "GrBu_d",
    "PiYG",
    "PuGr",
    "RdBu",
    "RdYeBuCy",
    "autumn",
    "blues",
    "cool",
    "coolwarm",
    "cubehelix",
    "diverging",
    "fire",
    "gist_earth",
    "gray",
    "gray_r",
    "greens",
    "hot",
    "hsl",
    "hsv",
    "husl",
    "inferno",
    "light_blues",
    "magma",
    "orange",
    "plasma",
    "reds",
    "single_hue",
    "spring",
    "summer",
    "turbo",
    "twilight",
    "twilight_shifted",
    "viridis",
    "winter",
]


img, _ = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))

viewer = napari.Viewer()

for cmap in colormaps:
    viewer.add_image(img, colormap=cmap, name=cmap)
    print(cmap)
viewer.grid.enabled = True

napari.run()
