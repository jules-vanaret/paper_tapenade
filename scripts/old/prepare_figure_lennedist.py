import numpy as np
import tifffile
import napari

path_to_data = '/home/jvanaret/data/lennedist_data/raw'

data_dict = {
    'alice': ['image1.tif', 'label1.tif', 'image2.tif', 'label2.tif', 'image3.tif', 'label3.tif'],
    'mine': ['square.tif', 'square_annotations_gt.tif', 'data_block.tif', 'annotated_block.tif']
}

viewer = napari.Viewer()

def normalize_percentiles(data):

        perc_low, perc_high = np.percentile(data, (1,99))

        data = (data - perc_low)/(perc_high - perc_low)
        data = np.clip(data, 0, 1)

        return data


for name in ['alice', 'mine']:

    for name_data, name_labels in zip(data_dict[name][::2], data_dict[name][1::2]):
        print(name_data)
        print(data_dict[name][::2])
        data = tifffile.imread(f'{path_to_data}/{name}/{name_data}')
        labels = tifffile.imread(f'{path_to_data}/{name}/{name_labels}')
        if data.ndim == 4:
            data = data[0]
            labels = labels[0]

        data = normalize_percentiles(data)
        slices = tuple([
             slice(int((s-s_)/2), int((s+s_)/2)) for s,s_ in zip(data.shape, (48, 60, 60))
        ])

        data = data[slices]
        labels = labels[slices]

        data = data[0]
        labels = labels[0]


        viewer.add_image(data)
        viewer.add_labels(labels)

viewer.grid.enabled = True
viewer.grid.shape = (-1, 2)

napari.run()
