import numpy as np
import tifffile
import napari

from tapenade.analysis.additional_regionprops_properties import (
    add_principal_lengths,
    add_tensor_inertia,
    add_true_strain_tensor,
)

from skimage.measure import regionprops


labels = tifffile.imread("/home/jvanaret/data/data_paper_tapenade/delme.tif")

props = regionprops(labels)

n_points = len(props)
n_dim_space = 3
n_dim_tensor = 9

sparse_inertia_tensor = np.zeros((n_points, n_dim_space + n_dim_tensor))
sparse_true_strain_tensor = np.zeros((n_points, n_dim_space + n_dim_tensor))

# store volumes to use later
volumes = np.zeros(n_points)

for index_label, prop in enumerate(props):
    add_tensor_inertia(prop, scale=(1, 1, 1))
    add_principal_lengths(prop, scale=(1, 1, 1))
    # add_true_strain_tensor(prop, scale=(1,1,1))
    volumes[index_label] = prop.area

    axis_decoupling_matrix = np.ones((3, 3)) / 2 - np.eye(3)
    print(prop.principal_lengths)
    print(prop.tensor_inertia)
    print(prop.tensor_inertia * 5 / prop.area)
    print(
        np.sqrt(
            axis_decoupling_matrix
            @ np.linalg.eigh(prop.tensor_inertia * 5 / prop.area)[0]
        )
    )

    print("\n")
    # print(prop.principal_lengths)
    # print(prop.principal_vectors)

    sparse_inertia_tensor[index_label, :n_dim_space] = prop.centroid
    sparse_inertia_tensor[index_label, n_dim_space:] = prop.tensor_inertia.reshape(-1)

    sparse_true_strain_tensor[index_label, :n_dim_space] = prop.centroid
    # sparse_true_strain_tensor[index_label, n_dim_space:] = prop.true_strain_tensor.flatten()


def tensors_to_napari_vectors(
    sparse_tensors,
    is_inertia_tensor: bool,
    volumes: np.ndarray = None,
    return_angles: bool = False,
):

    positions = sparse_tensors[:, :3]
    tensors = sparse_tensors[:, 3:]
    tensors = tensors.reshape(-1, 3, 3)

    # diagonalize the tensors

    eigen_values, principal_vectors = np.linalg.eigh(tensors)
    if is_inertia_tensor:
        # the principal lengths are a mix of the eigenvalues and the volumes
        axis_decoupling_matrix = np.ones((3, 3)) / 2 - np.eye(3)
        eigen_values = np.sqrt(
            np.einsum("ij,lj->li", axis_decoupling_matrix, eigen_values)
        )
        eigen_values = eigen_values * np.sqrt(5 / volumes.reshape(-1, 1, 1))
        print("HEY", eigen_values)

    # WARNING
    principal_vectors = principal_vectors.transpose(0, 2, 1)

    napari_vectors = np.zeros((len(positions), 2, 3))
    indices_maxs = np.nanargmax(eigen_values, axis=1)
    principal_vectors = principal_vectors[np.arange(len(eigen_values)), indices_maxs]
    eigen_values = eigen_values[np.arange(len(eigen_values)), indices_maxs]
    napari_vectors[:, 0] = positions
    napari_vectors[:, 1] = principal_vectors * eigen_values.reshape(-1, 1)

    if return_angles:
        angles = np.arctan2(*(napari_vectors[:, 1, -2:].reshape(-1, 2).T))
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        return napari_vectors, angles

    return napari_vectors


napari_vectors, angles = tensors_to_napari_vectors(
    sparse_inertia_tensor, is_inertia_tensor=True, volumes=volumes, return_angles=True
)
napari_vectors_true_strain, angles_true_strain = tensors_to_napari_vectors(
    sparse_true_strain_tensor,
    is_inertia_tensor=False,
    volumes=volumes,
    return_angles=True,
)


viewer = napari.Viewer()

viewer.add_labels(labels)

napari.run()
