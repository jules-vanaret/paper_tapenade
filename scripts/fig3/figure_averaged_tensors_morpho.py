import tifffile
import numpy as np
import napari
from magicgui import magicgui
from tqdm import tqdm
from pyngs.sparse_smoothing import gaussian_smooth_sparse
from pyngs.dense_smoothing import gaussian_smooth_dense
from pyngs.formatting import dense_array_to_napari_vectors
from pyngs.utils import get_napari_angles_cmap
from skimage.measure import regionprops
from tapenade.analysis.additional_regionprops_properties import (
    add_tensor_inertia,
    add_principal_lengths,
    add_true_strain_tensor
)
from napari.utils import DirectLabelColormap
import matplotlib
import matplotlib.pyplot as plt



path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'

index = 6

if index == 7:
    angle_shift = 1.9
elif index == 6:
    angle_shift = 2.3
else:
    angle_shift = 0

data = tifffile.imread(f'{path_to_data}/ag{index}_norm.tif')
labels = tifffile.imread(f'{path_to_data}/ag{index}_norm_labels.tif')
mask = tifffile.imread(f'{path_to_data}/ag{index}_mask.tif')

mid_z = 108#126#int(labels.shape[0]/2)
sigma=15

# positions_2d = np.mgrid[
#    [slice(mid_z, mid_z+1)] + [slice(0, dim, 20) for dim in labels.shape[1:]]
# ].reshape(labels.ndim, -1).T
positions_2d = np.mgrid[
   [slice(0, labels.shape[0], 10)]+[slice(0, dim, 20) for dim in labels.shape[1:]]
].reshape(labels.ndim, -1).T

positions_2d = positions_2d[mask[positions_2d[:,0], positions_2d[:,1], positions_2d[:,2]]]

props = regionprops(labels)
sparse_inertia_tensors_data = np.zeros((len(props), 3+9))
sparse_true_strain_tensors_data = np.zeros((len(props), 3+9))
vectors = np.zeros((len(props), 2, 3))

for i,prop in enumerate(tqdm(props[::], smoothing=0)):

    add_tensor_inertia(prop, scale=(1,1,1))
    add_principal_lengths(prop, scale=(1,1,1), add_principal_vectors=True)
    add_true_strain_tensor(prop, scale=(1,1,1))
    
    sparse_inertia_tensors_data[i, :3] = prop.centroid
    sparse_inertia_tensors_data[i, -9:] = prop.tensor_inertia.ravel()
    sparse_true_strain_tensors_data[i, :3] = prop.centroid
    sparse_true_strain_tensors_data[i, -9:] = prop.true_strain_tensor.ravel()
    vectors[i,1] = prop.principal_vectors[0] * prop.principal_lengths[0]
    vectors[i,0] = prop.centroid



def napari_vectors_from_tensors(smoothed_tensors, apply_decoupling, nematic=False, return_angles=False):

    positions = smoothed_tensors[:, :3]
    smoothed_tensors = smoothed_tensors[:, 3:]
    smoothed_tensors = smoothed_tensors.reshape(-1, 3, 3)


    eigen_values, principal_vectors = np.linalg.eigh(smoothed_tensors)
    principal_vectors = principal_vectors.transpose(0,2,1)

    if apply_decoupling:
        axis_decoupling_matrix = np.ones((3,3))/2 - np.eye(3)
        # principal_lengths = np.sqrt(axis_decoupling_matrix @ eigen_values)
        eigen_values = np.sqrt(
            np.einsum('ij,lj->li', axis_decoupling_matrix, eigen_values)
        )

    napari_vectors = np.zeros((len(positions), 2, 3))
    indices_maxs = np.nanargmax(eigen_values, axis=1)
    principal_vectors = principal_vectors[np.arange(len(eigen_values)), indices_maxs]
    eigen_values = eigen_values[np.arange(len(eigen_values)), indices_maxs]
    napari_vectors[:,0] = positions
    napari_vectors[:,1] = principal_vectors * eigen_values.reshape(-1,1)

    if nematic:
        N_vec = napari_vectors.shape[0]
        napari_vectors = np.concatenate([napari_vectors, napari_vectors], axis=0)
        napari_vectors[N_vec:, 1] *= -1

    if return_angles:
        angles = np.arctan2(*(napari_vectors[:,1, -2:].reshape(-1, 2).T))

        angles = np.arctan2(np.sin(angles-1), np.cos(angles-1))

        return napari_vectors, angles



    return napari_vectors


def smooth_labels_values(labels, values, sigma):
    values = np.array(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    dim_space = labels.ndim
    dim_data = values.shape[1]

    props = regionprops(labels)
    centroids = np.array([prop.centroid for prop in props])

    sparse_array = np.zeros((len(props), dim_space + dim_data))
    sparse_array[:, :labels.ndim] = centroids
    sparse_array[:, labels.ndim:] = values

    smoothed_array = gaussian_smooth_sparse(
        sparse_array, is_temporal=False, sigmas=sigma, dim_space=dim_space
    )

    smoothed_values = smoothed_array[:, dim_space:]

    return smoothed_values

def smooth_labels_values2(labels, values, sigma, mask, mask_for_volume):

    props = regionprops(labels)
    centroids = np.round(np.array([prop.centroid for prop in props])).astype(int)

    array_centroids = np.zeros(labels.shape)
    array_centroids[centroids[:,0], centroids[:,1], centroids[:,2]] = values

    smoothed_array = gaussian_smooth_dense(
        array_centroids, is_temporal=False, dim_space=3,
        sigmas=sigma, mask=mask, mask_for_volume=array_centroids.astype(bool)
    )

    return smoothed_array


# positions = np.mgrid[
#     [slice(0, dim, 20) for dim in labels.shape]
# ].reshape(labels.ndim, -1).T

# positions = positions[mask[positions[:,0], positions[:,1], positions[:,2]]]

### INERTIA TENSOR
smoothed_inertia_tensors_data = gaussian_smooth_sparse(
    sparse_inertia_tensors_data, is_temporal=False, sigmas=sigma,
    dim_space=3#, positions=positions
) # remove the positions

napari_vectors_inertia, angles_vectors_inertia = napari_vectors_from_tensors(
    smoothed_inertia_tensors_data,
    apply_decoupling=True,
    nematic=True,
    return_angles=True
)
# napari_vectors_inertia_cellular = napari_vectors_from_tensors(
#     sparse_inertia_tensors_data,
#     apply_decoupling=True
# )
### INERTIA TENSOR AVERAGED IN THE Z DIRECTION

sparse_inertia_tensors_data_2d = sparse_inertia_tensors_data.copy()
# sparse_inertia_tensors_data_2d[:, 0] = mid_z
# sparse_inertia_tensors_data_2d[:, [3,4,5,6,9]] = 0

smoothed_inertia_tensors_data_2d = gaussian_smooth_sparse(
    sparse_inertia_tensors_data_2d, is_temporal=False, sigmas=sigma,
    dim_space=3, positions=positions_2d
)

napari_vectors_inertia_2d, angles_vectors_inertia_2d = napari_vectors_from_tensors(
    smoothed_inertia_tensors_data_2d,
    apply_decoupling=True,
    nematic=True,
    return_angles=True
)

### TRUE STRAIN TENSOR

smoothed_true_strain_tensors_data = gaussian_smooth_sparse(
    sparse_true_strain_tensors_data, is_temporal=False, sigmas=sigma,
    dim_space=3#, positions=positions
) # remove the positions

napari_vectors_true_strain, angles_vectors_true_strain = napari_vectors_from_tensors(
    smoothed_true_strain_tensors_data,
    apply_decoupling=False,
    nematic=True,
    return_angles=True
)

# napari_vectors_true_strain_cellular = napari_vectors_from_tensors(
#     sparse_true_strain_tensors_data,
#     apply_decoupling=False
# )

### TRUE STRAIN TENSOR AVERAGED IN THE Z DIRECTION

sparse_true_strain_tensors_data_2d = sparse_true_strain_tensors_data.copy()
# sparse_true_strain_tensors_data_2d[:, 0] = mid_z
# sparse_true_strain_tensors_data_2d[:, [3,4,5,6,9]] = 0

smoothed_true_strain_tensors_data_2d = gaussian_smooth_sparse(
    sparse_true_strain_tensors_data_2d, is_temporal=False, sigmas=sigma,
    dim_space=3, positions=positions_2d
)

napari_vectors_true_strain_2d, angles_vectors_true_strain_2d = napari_vectors_from_tensors(
    smoothed_true_strain_tensors_data_2d,
    apply_decoupling=False,
    nematic=True,
    return_angles=True
)

###





viewer = napari.Viewer()



### NORMALIZE VECTORS
vector_lenghts_inertia = np.linalg.norm(napari_vectors_inertia[:,1], axis=1)
napari_vectors_inertia[:,1] /= np.median(vector_lenghts_inertia)
vector_lenghts_inertia_2d = np.linalg.norm(napari_vectors_inertia_2d[:,1], axis=1)
napari_vectors_inertia_2d[:,1] /= np.median(vector_lenghts_inertia_2d)

# fig = plt.figure()
# plt.hist(vector_lenghts_inertia, bins=100)

vector_lenghts_true_strain = np.linalg.norm(napari_vectors_true_strain[:,1], axis=1)
napari_vectors_true_strain[:,1] /= np.median(vector_lenghts_true_strain)
vector_lenghts_true_strain_2d = np.linalg.norm(napari_vectors_true_strain_2d[:,1], axis=1)
napari_vectors_true_strain_2d[:,1] /= np.median(vector_lenghts_true_strain_2d)

# fig = plt.figure()
# plt.hist(vector_lenghts_true_strain, bins=100)

# plt.show()


# ### REMOVE ALL Z COMPONENTS SO THAT THEY DONT 
# ### DISPLAY IN NAPARI
# napari_vectors_inertia[:,1,0] = 0
# napari_vectors_inertia_2d[:,1,0] = 0

# # napari_vectors_true_strain[:,1,0] = 0
# napari_vectors_true_strain_2d[:,1,0] = 0

# # napari_vectors_inertia[:,0,0] = np.round(napari_vectors_inertia[:,0,0])
# napari_vectors_inertia_2d[:,0,0] = np.round(napari_vectors_inertia_2d[:,0,0])

# # napari_vectors_true_strain[:,0,0] = np.round(napari_vectors_true_strain[:,0,0])
# napari_vectors_true_strain_2d[:,0,0] = np.round(napari_vectors_true_strain_2d[:,0,0])


cmap = get_napari_angles_cmap(color_mode='nematic')


### INERTIA TENSOR
viewer.add_image(data, opacity=1, colormap='gray_r', contrast_limits=(0,0.6))
viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)

viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)
viewer.add_vectors(napari_vectors_inertia, edge_color='blue', edge_width=4, length=15,
                   out_of_slice_display=True, opacity=1, vector_style='line', 
                   edge_colormap=cmap, properties={'angles': angles_vectors_inertia})
# viewer.add_vectors(napari_vectors_inertia_cellular, edge_color='blue', edge_width=4, length=15,
#                    out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque')

viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)
viewer.add_vectors(napari_vectors_inertia_2d, edge_color='blue', edge_width=4, length=15,
                     out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque', 
                     edge_colormap=cmap, properties={'angles': angles_vectors_inertia_2d})
# viewer.add_labels(np.zeros(data.shape, dtype=np.uint8))

viewer.add_image(data, opacity=0.4, colormap='gray_r', contrast_limits=(0,0.6))
viewer.add_vectors(napari_vectors_inertia_2d, edge_color='blue', edge_width=4, length=15,
                     out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque', 
                     edge_colormap=cmap, properties={'angles': angles_vectors_inertia_2d})
###

### TRUE STRAIN TENSOR
viewer.add_image(data, opacity=1, colormap='gray_r', contrast_limits=(0,0.6))
viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)

viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)
viewer.add_vectors(napari_vectors_true_strain, edge_color='magenta', edge_width=4, length=15,
                   out_of_slice_display=True, opacity=1, vector_style='line', 
                   edge_colormap=cmap, properties={'angles': angles_vectors_true_strain})
# viewer.add_vectors(napari_vectors_true_strain_cellular, edge_color='magenta', edge_width=4, length=15,
#                    out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque')

viewer.add_image(data*0, opacity=0.5, colormap='gray_r', contrast_limits=(0,0.6), visible=False)
viewer.add_vectors(napari_vectors_true_strain_2d, edge_color='magenta', edge_width=4, length=15,
                     out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque', 
                     edge_colormap=cmap, properties={'angles': angles_vectors_true_strain_2d})
# viewer.add_labels(np.zeros(data.shape, dtype=np.uint8))

viewer.add_image(data, opacity=0.4, colormap='gray_r', contrast_limits=(0,0.6))
viewer.add_vectors(napari_vectors_true_strain_2d, edge_color='magenta', edge_width=4, length=15,
                     out_of_slice_display=True, opacity=1, vector_style='line',blending='opaque', 
                     edge_colormap=cmap, properties={'angles': angles_vectors_true_strain_2d})
###


###

volumes = [prop.area for prop in props]
do_sparse = False

if do_sparse:
    smoothed_volumes = smooth_labels_values(labels, volumes, sigma)
    percs = np.percentile(smoothed_volumes, [10, 90])
    smoothed_volumes = np.clip(smoothed_volumes, percs[0], percs[1])
    volume_max = np.max(smoothed_volumes)
    volume_min = np.min(smoothed_volumes)

    print('volume min', volume_min, 'volume max', volume_max)
    smoothed_volumes = (smoothed_volumes - volume_min) / (volume_max - volume_min)
    cmap = matplotlib.colormaps['viridis']
    cmap_dict = {prop.label: cmap(smoothed_volumes[i]) for i, prop in enumerate(props)}
    # take care of background label
    cmap_dict[None] = [0, 0, 0, 0]

    napari_cmap = DirectLabelColormap(color_dict=cmap_dict)

    viewer.add_labels(labels, opacity=1, colormap=napari_cmap)
else:
    smoothed_volumes = smooth_labels_values2(labels, volumes, sigma, 
                                             mask=mask, 
                                             mask_for_volume=labels.astype(bool))
    smoothed_volumes[~mask] = np.nan
    percs = np.nanpercentile(smoothed_volumes, [1, 99])
    viewer.add_image(smoothed_volumes, colormap='viridis', contrast_limits=percs)
###



for l in viewer.layers:
    l.scale = (0.621, 0.621, 0.621)

im = np.zeros((4*sigma+1, 4*sigma+1))
for i in range(4*sigma+1):
    for j in range(4*sigma+1):
        im[i,j] = np.exp(-((i-2*sigma)**2 + (j-2*sigma)**2)/(2*sigma**2))

viewer.add_image(im, opacity=1, colormap='inferno', scale=(0.621, 0.621))

def update(event):
    z = viewer.dims.current_step[0]
    viewer.text_overlay.text = f"z={0.621*(z-4):1.1f} um"

viewer.text_overlay.visible = True
viewer.text_overlay.font_size = 30
viewer.text_overlay.position = 'top_left'
viewer.dims.events.current_step.connect(update)

viewer.scale_bar.visible = True




viewer.grid.enabled = True
viewer.grid.shape = (3, 4)
viewer.grid.stride = -2
viewer.dims.set_current_step(0,108)

viewer.scale_bar.visible = True

### ANGLE LEGEND
angles = np.linspace(-np.pi, np.pi, 100)
vec = np.array([[np.cos(th), np.sin(th)] for th in angles])
# print(angles)
angles = np.arctan2(np.sin(angles+angle_shift), np.cos(angles+angle_shift))
# print(angles)

napari_vec = 10 * np.concatenate([vec[:, None], vec[:, None]], axis=1)
cmap = get_napari_angles_cmap(color_mode='nematic')
viewer.add_vectors(napari_vec, edge_colormap=cmap, edge_width=2, 
                   length=2, properties={'angles': angles}, vector_style='line')


def remap_angles(angles, angle_shift):
    angles = angles + angle_shift
    angles %= 2*np.pi
    return angles - 2*np.pi*(angles >= np.pi)

@magicgui(auto_call=True,
            angle_shift={'widget_type': 'FloatSlider', 'min': 0, 'max': 2*np.pi, 'step': 1},
          )
def widget_angle_shift(viewer: napari.Viewer, angle_shift: float = 0):

    for vectors in viewer.layers:
        if hasattr(vectors, 'properties') and 'angles' in vectors.properties:

            if not('old_angles' in vectors.properties):
                old_dict = {'old_angles': vectors.properties['angles']}
                vectors.properties = {**vectors.properties, **old_dict}

            angles = vectors.properties['old_angles']
            angles = remap_angles(angles, angle_shift)

            vectors.properties = {**vectors.properties, **{'angles': angles}}
            vectors.refresh()
            


viewer.window.add_dock_widget(widget_angle_shift, area='right')


napari.run()
