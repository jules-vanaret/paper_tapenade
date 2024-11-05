import tifffile
import numpy as np
import napari

from tqdm import tqdm
from pyngs.sparse_smoothing import gaussian_smooth_sparse
from skimage.measure import regionprops
from tapenade.analysis.additional_regionprops_properties import (
    add_tensor_inertia,
    add_principal_lengths,
    add_true_strain_tensor
)



path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed'

index = 8

data = tifffile.imread(f'{path_to_data}/ag{index}_norm.tif')
labels = tifffile.imread(f'{path_to_data}/ag{index}_norm_labels.tif')
mask = tifffile.imread(f'{path_to_data}/ag{index}_mask.tif')


props = regionprops(labels)
sparse_inertia_tensors_data = np.zeros((len(props), 3+9))
sparse_true_strain_tensors_data = np.zeros((len(props), 3+9))
vectors = np.zeros((len(props), 2, 3))

for i,prop in enumerate(tqdm(props[::])):

    add_tensor_inertia(prop, scale=(1,1,1))
    add_principal_lengths(prop, scale=(1,1,1), add_principal_vectors=True)
    add_true_strain_tensor(prop, scale=(1,1,1))
    
    sparse_inertia_tensors_data[i, :3] = prop.centroid
    sparse_inertia_tensors_data[i, -9:] = prop.tensor_inertia.ravel()
    sparse_true_strain_tensors_data[i, :3] = prop.centroid
    sparse_true_strain_tensors_data[i, -9:] = prop.true_strain_tensor.ravel()
    vectors[i,1] = prop.principal_vectors[0] * prop.principal_lengths[0]
    vectors[i,0] = prop.centroid



def napari_vectors_from_tensors(smoothed_tensors, apply_decoupling):

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


    return napari_vectors


# positions = np.mgrid[
#     [slice(0, dim, 20) for dim in labels.shape]
# ].reshape(labels.ndim, -1).T

# positions = positions[mask[positions[:,0], positions[:,1], positions[:,2]]]

### INERTIA TENSOR
smoothed_inertia_tensors_data = gaussian_smooth_sparse(
    sparse_inertia_tensors_data, is_temporal=False, sigmas=15,
    dim_space=3#, positions=positions
) # remove the positions

napari_vectors_inertia = napari_vectors_from_tensors(
    smoothed_inertia_tensors_data, 
    apply_decoupling=True

)
### INERTIA TENSOR AVERAGED IN THE Z DIRECTION

mid_z = int(labels.shape[0]/2)

positions_2d = np.mgrid[
   [slice(mid_z, mid_z+1)] + [slice(0, dim, 20) for dim in labels.shape[1:]]
].reshape(labels.ndim, -1).T

positions_2d = positions_2d[mask[positions_2d[:,0], positions_2d[:,1], positions_2d[:,2]]]

sparse_inertia_tensors_data_2d = sparse_inertia_tensors_data.copy()
sparse_inertia_tensors_data_2d[:, 0] = mid_z
# sparse_inertia_tensors_data_2d[:, [3,4,5,6,9]] = 0

smoothed_inertia_tensors_data_2d = gaussian_smooth_sparse(
    sparse_inertia_tensors_data_2d, is_temporal=False, sigmas=15,
    dim_space=3, positions=positions_2d
)

napari_vectors_inertia_2d = napari_vectors_from_tensors(
    smoothed_inertia_tensors_data_2d,
    apply_decoupling=True
)

### TRUE STRAIN TENSOR

smoothed_true_strain_tensors_data = gaussian_smooth_sparse(
    sparse_true_strain_tensors_data, is_temporal=False, sigmas=15,
    dim_space=3#, positions=positions
) # remove the positions

napari_vectors_true_strain = napari_vectors_from_tensors(
    smoothed_true_strain_tensors_data,
    apply_decoupling=False
)

### TRUE STRAIN TENSOR AVERAGED IN THE Z DIRECTION

sparse_true_strain_tensors_data_2d = sparse_true_strain_tensors_data.copy()
sparse_true_strain_tensors_data_2d[:, 0] = mid_z
# sparse_true_strain_tensors_data_2d[:, [3,4,5,6,9]] = 0

smoothed_true_strain_tensors_data_2d = gaussian_smooth_sparse(
    sparse_true_strain_tensors_data_2d, is_temporal=False, sigmas=15,
    dim_space=3, positions=positions_2d
)

napari_vectors_true_strain_2d = napari_vectors_from_tensors(
    smoothed_true_strain_tensors_data_2d,
    apply_decoupling=False
)

###
viewer = napari.Viewer()
viewer.add_image(data, opacity=0.5)
viewer.add_labels(labels, visible=False)
viewer.add_vectors(vectors, edge_width=2, length=3,out_of_slice_display=True,opacity=1)

viewer.add_vectors(napari_vectors_inertia, edge_color='blue', edge_width=2, length=0.1,
                   out_of_slice_display=True, opacity=1)

viewer.add_vectors(napari_vectors_inertia_2d, edge_color='blue', edge_width=2, length=0.1,
                     out_of_slice_display=True, opacity=1)

viewer.add_vectors(napari_vectors_true_strain, edge_color='green', edge_width=2, length=150,
                   out_of_slice_display=True, opacity=1)

viewer.add_vectors(napari_vectors_true_strain_2d, edge_color='green', edge_width=2, length=150,
                        out_of_slice_display=True, opacity=1)

napari.run()
