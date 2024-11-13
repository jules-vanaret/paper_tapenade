from pyngs.utils import get_napari_angles_cmap
import numpy as np
import napari


cmap = get_napari_angles_cmap(color_mode="nematic")


pos = np.zeros((100, 2))
angles = np.linspace(-np.pi, np.pi, 100)
vec = np.array([[np.cos(th), np.sin(th)] for th in angles])
# print(angles)
angles = np.arctan2(np.sin(angles + 2.3), np.cos(angles + 2.3))
# print(angles)

napari_vec = np.concatenate([vec[:, None], vec[:, None]], axis=1)

viewer = napari.Viewer()
viewer.add_vectors(
    napari_vec,
    edge_colormap=cmap,
    edge_width=0.1,
    length=2,
    properties={"angles": angles},
    vector_style="line",
)

napari.run()
