# This script visualizes the XY and orthogonal plane cuts for all organoids
# with lines showing where the complementary cuts are located

import napari
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path


def argwhere_int(tup):
    """Helper function to identify which axis is being sliced in orthogonal cuts"""
    is_int = [isinstance(x, int) for x in tup]
    index = is_int.index(True)
    value = tup[index]
    other_indices = [i for i in range(3) if i != index]
    return index, value, other_indices


def create_cut_line_vectors(image_shape, cut_position, axis):
    """Create vectors to show where cuts are made"""
    if axis == 0:  # vertical line (constant x)
        start_y, end_y = 0, image_shape[0] - 1
        positions = np.array([[start_y, cut_position], [end_y, cut_position]])
        directions = np.array([[end_y - start_y, 0], [0, 0]])
    elif axis == 1:  # horizontal line (constant y)
        start_x, end_x = 0, image_shape[1] - 1
        positions = np.array([[cut_position, start_x], [cut_position, end_x]])
        directions = np.array([[0, end_x - start_x], [0, 0]])
    
    # Create napari vectors format: [position, direction]
    vectors = np.stack([positions, directions], axis=1)
    return vectors


path_to_data = Path(__file__).parents[2] / 'data/4acd_data_morphology'
indices_organoids = [6, 2, 4, 5]#range(1, 9)

# Define the cut positions for each organoid

# Create viewer
viewer = napari.Viewer(title='Plane Cuts Visualization')

for i, index_organoid in enumerate(tqdm(indices_organoids)):
    mid_plane_ind = [181, 80, 80, 199, 98, 81, 81, 81][index_organoid - 1]
    orthogonal_mid_plane_slice = [
        (slice(None), 190), 
        (slice(None), 280), 
        (slice(None), 255), 
        (slice(None), 308), 
        (slice(None), 217), 
        (slice(None), slice(None), 245), 
        (slice(None), slice(None), 286), 
        (slice(None), 300)
    ][index_organoid - 1]
    
    # Load 3D data
    mask_3d = tifffile.imread(f"{path_to_data}/ag{index_organoid}_mask.tif")
    data_3d = tifffile.imread(f"{path_to_data}/ag{index_organoid}_norm.tif")
    
    # Get XY slice (midplane)
    xy_data = data_3d[mid_plane_ind]
    xy_mask = mask_3d[mid_plane_ind]
    
    # Get orthogonal slice
    ortho_data = data_3d[orthogonal_mid_plane_slice]
    ortho_mask = mask_3d[orthogonal_mid_plane_slice]
    
    # Determine which axis is being sliced for orthogonal view
    slice_axis, slice_value, remaining_axes = argwhere_int(orthogonal_mid_plane_slice)
    
    
    # Create cut line vectors for XY view showing where orthogonal cut is made
    if slice_axis == 1:  # orthogonal cut is at constant Y, so draw horizontal line on XY
        xy_cut_vectors = create_cut_line_vectors(xy_data.shape, slice_value, axis=1)
        # For orthogonal view (ZX plane), draw horizontal line at Z=mid_plane_ind
        ortho_cut_vectors = create_cut_line_vectors(ortho_data.shape, mid_plane_ind, axis=1)
        xy_line_name = f'ag{index_organoid}_xy_cut_y={slice_value}'
        ortho_line_name = f'ag{index_organoid}_ortho_cut_z={mid_plane_ind}'
        
    elif slice_axis == 2:  # orthogonal cut is at constant X, so draw vertical line on XY  
        xy_cut_vectors = create_cut_line_vectors(xy_data.shape, slice_value, axis=0)
        # For orthogonal view (ZY plane), draw horizontal line at Z=mid_plane_ind
        ortho_cut_vectors = create_cut_line_vectors(ortho_data.shape, mid_plane_ind, axis=1)
        xy_line_name = f'ag{index_organoid}_xy_cut_x={slice_value}'
        ortho_line_name = f'ag{index_organoid}_ortho_cut_z={mid_plane_ind}'
        
    else:  # slice_axis == 0, orthogonal cut is at constant Z
        # This case means we're looking at a different Z slice, 
        # so we show where the XY midplane intersects
        xy_cut_vectors = create_cut_line_vectors(xy_data.shape, slice_value, axis=1)
        # For orthogonal view (YX plane), draw horizontal line  
        ortho_cut_vectors = create_cut_line_vectors(ortho_data.shape, mid_plane_ind, axis=1)
        xy_line_name = f'ag{index_organoid}_xy_cut_y={slice_value}'
        ortho_line_name = f'ag{index_organoid}_ortho_cut_y={mid_plane_ind}'
    
    # Add XY view
    viewer.add_image(xy_data, 
                    name=f'ag{index_organoid}_xy_data',
                    colormap='gray_r', 
                    contrast_limits=[0.14, 0.5])
    viewer.layers[-1].gamma = 0.75
    # Add cut line vectors
    viewer.add_vectors(xy_cut_vectors,
                      name=xy_line_name,
                      edge_color='red',
                      edge_width=3,
                      length=1.0,
                      vector_style='line')
    
    # Add orthogonal view
    viewer.add_image(ortho_data,
                    name=f'ag{index_organoid}_ortho_data', 
                    colormap='gray_r',
                    contrast_limits=[0.14, 0.5])
    viewer.layers[-1].gamma = 0.75
    
    viewer.add_vectors(ortho_cut_vectors,
                      name=ortho_line_name, 
                      edge_color='red',
                      edge_width=3,
                      length=1.0,
                      vector_style='line')
    

# Set up grid layout: 2 columns (XY, orthogonal), with each organoid as a row
viewer.grid.enabled = True
viewer.grid.stride = 2
napari.run()
