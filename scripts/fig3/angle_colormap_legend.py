import napari
import numpy as np
from pyngs.utils import get_napari_angles_cmap


angles = np.linspace(-np.pi, np.pi, 100)
vec = np.array([[np.cos(th), np.sin(th)] for th in angles])
# print(angles)
angles = np.arctan2(np.sin(angles), np.cos(angles))
# print(angles)

napari_vec = 10 * np.concatenate([vec[:, None], vec[:, None]], axis=1)

cmap = get_napari_angles_cmap(
    # color_mode='recirculation',
    # color_mode='nematic',
    color_mode="cyclic",
)

viewer = napari.Viewer()
viewer.add_vectors(
    napari_vec,
    edge_colormap=cmap,
    edge_width=2,
    length=2,
    properties={"angles": angles},
    vector_style="line",
)

napari.run()
