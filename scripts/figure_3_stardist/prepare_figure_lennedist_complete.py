import tifffile
import napari
from pathlib import Path


path_to_data = Path(__file__).parents[2] / 'data/stardist_training_datasets'

data_dict = {
    'annotator1': ['image1.tif', 'label1.tif'],
    'annotator2': ['square.tif', 'square_annotations_gt.tif', 'data_block.tif', 'annotated_block.tif']
}

viewer = napari.Viewer()


for name in ['annotator1', 'annotator2']:
    for name_data, name_labels in zip(data_dict[name][::2], data_dict[name][1::2]):

        data = tifffile.imread(
            f'{path_to_data}/{name}/{name_data}'
        )

        labels = tifffile.imread(
            f'{path_to_data}/{name}/{name_labels}'
        )
        if data.ndim == 4:
            data = data[14]
            labels = labels[14]

        mids = [s/2 for s in data.shape]
        cuts = [min(s,s_)/2 for s,s_ in zip(data.shape, (64, 64,64))]

        slices = [
            slice(int(m-c), int(m+c)) for m,c in zip(mids, cuts)
        ]

        if name_data == 'square.tif':
            slices[2] = slice(64+14, 128+14)

        data = data[tuple(slices)]
        labels = labels[tuple(slices)]

        data = data[26]
        labels = labels[26]


        viewer.add_image(data, colormap='gray_r', name=f'{name}_{name_data}')
        viewer.add_labels(labels, opacity=1, name=f'{name}_{name_data}')

viewer.grid.enabled = True
viewer.grid.shape = (-1, 2)



napari.run()