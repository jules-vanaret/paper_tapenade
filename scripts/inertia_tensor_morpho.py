import tifffile
import numpy as np
import napari

from puns.sparse_smoothing import gaussian_smooth_sparse
from skimage.measure import regionprops
from organoid.analysis._additional_regionprops_properties import (
    add_tensor_moment,
    add_tensor_inertia,
    add_principal_lengths,
)


### Import labels
path_to_data = '/home/jvanaret/data/data_paper_valentin/morphology/processed'

index = 7

data = tifffile.imread(f'{path_to_data}/ag{index}_norm.tif')
labels = tifffile.imread(f'{path_to_data}/ag{index}_norm_labels.tif')
mask = tifffile.imread(f'{path_to_data}/ag{index}_mask.tif')

### Create regionprops with extra properties
props = regionprops(labels)
for prop in props:
    add_tensor_inertia(prop, scale=(1,1,1))
    add_principal_lengths(prop, scale=(1,1,1), add_principal_vectors=True)

### Accumulate inertia tensor, then transform to vector
sparse_inertia_tensors_data = np.zeros((len(props), 3+9))
for i, prop in enumerate(props):
    sparse_inertia_tensors_data[i, :3] = prop.centroid
    sparse_inertia_tensors_data[i, -9:] = prop.tensor_inertia.ravel()

### Transfom into napari vectors
vectors = np.zeros((len(props), 2, 3))
for i, prop in enumerate(props):
    vectors[i,0] = prop.centroid
    vectors[i,1] = prop.principal_vectors[0] * prop.principal_lengths[0]


### Plot on one slice to confirm
viewer = napari.Viewer()
viewer.add_image(data)
viewer.add_labels(labels, visible=False)
viewer.add_vectors(vectors, edge_width=2, length=3,out_of_slice_display=True,opacity=1)

# napari.run()
### Average on labels positions
positions = np.mgrid[
    [slice(0, dim, 20) for dim in labels.shape]
].reshape(labels.ndim, -1).T

positions = positions[mask[positions[:,0], positions[:,1], positions[:,2]]]

smoothed_inertia_tensors_data = gaussian_smooth_sparse(
    sparse_inertia_tensors_data, is_temporal=False, sigmas=15,
    dim_space=3#, positions=positions
) # remove the positions

positions = smoothed_inertia_tensors_data[:, :3]
smoothed_inertia_tensors_data = smoothed_inertia_tensors_data[:, 3:]

### Turn tensors into directions
smoothed_inertia_tensors_data = smoothed_inertia_tensors_data.reshape(-1, 3, 3)


eigen_values, principal_vectors = np.linalg.eigh(smoothed_inertia_tensors_data)
principal_vectors = principal_vectors.transpose(0,2,1)

axis_decoupling_matrix = np.ones((3,3))/2 - np.eye(3)
# principal_lengths = np.sqrt(axis_decoupling_matrix @ eigen_values)
principal_lengths = np.sqrt(
    np.einsum('ij,lj->li', axis_decoupling_matrix, eigen_values)
)

napari_vectors_inertia = np.zeros((len(positions), 2, 3))
napari_vectors_inertia[:,0] = positions
napari_vectors_inertia[:,1] = principal_vectors[:,0] * principal_lengths[:,0].reshape(-1,1)

### Plot on one slice to confirm
viewer.add_vectors(napari_vectors_inertia, edge_color='blue', edge_width=2, length=0.1,
                   out_of_slice_display=True, opacity=1)

napari.run()
