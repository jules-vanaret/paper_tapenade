import numpy as np
import tifffile
import napari

path_to_data = "/home/jvanaret/data/data_paper_tapenade/figure_lennedist/"

data_dict = {
    "raw": {
        "alice": [
            "image1.tif",
            "label1.tif",
            "image2.tif",
            "label2.tif",
            "image3.tif",
            "label3.tif",
        ],
        "mine": [
            "square.tif",
            "square_annotations_gt.tif",
            "data_block.tif",
            "annotated_block.tif",
        ],
    },
    "isoscaled_norm": {
        "alice": [
            "image1.tif",
            "label1.tif",
            "image2.tif",
            "label2.tif",
            "image3.tif",
            "label3.tif",
        ],
        "mine": [
            "square.tif",
            "square_annotations_gt.tif",
            "data_block.tif",
            "annotated_block.tif",
        ],
    },
    "new_datasets": {
        "alice": ["ag8_norm.tif", "ag8_norm_labels.tif"],
        "mine": [
            "data_registered_firsttenframes.tif",
            "labels_registered_firsttenframes.tif",
        ],
    },
}

viewer = napari.Viewer()


for name in ["alice", "mine"]:
    for name_data, name_labels in zip(
        data_dict["raw"][name][::2], data_dict["raw"][name][1::2]
    ):

        for modality in ["raw", "isoscaled_norm"]:

            print(name_data)
            data = tifffile.imread(f"{path_to_data}/{modality}/{name}/{name_data}")
            print(data.shape)
            labels = tifffile.imread(f"{path_to_data}/{modality}/{name}/{name_labels}")
            if data.ndim == 4:
                data = data[14]
                labels = labels[14]

            mids = [s / 2 for s in data.shape]
            cuts = [min(s, s_) / 2 for s, s_ in zip(data.shape, (64, 64, 64))]

            slices = tuple([slice(int(m - c), int(m + c)) for m, c in zip(mids, cuts)])
            print(slices)

            data = data[slices]
            labels = labels[slices]

            data = data[26]
            labels = labels[26]

            viewer.add_image(data, colormap="gray_r")
            viewer.add_labels(labels, opacity=1)

viewer.grid.enabled = True
viewer.grid.shape = (-1, 4)

viewer = napari.Viewer()

for name in ["alice", "mine"]:
    for modality in ["new_datasets"]:
        for name_data, name_labels in zip(
            data_dict[modality][name][::2], data_dict[modality][name][1::2]
        ):

            print(name_data)
            data = tifffile.imread(f"{path_to_data}/{modality}/{name}/{name_data}")
            print(data.shape)
            labels = tifffile.imread(f"{path_to_data}/{modality}/{name}/{name_labels}")
            if data.ndim == 4:
                data = data[0]
                labels = labels[0]

            mids = [s / 2 for s in data.shape]
            cuts = [min(s, s_) / 2 for s, s_ in zip(data.shape, (64, 64, 64))]

            slices = tuple([slice(int(m - c), int(m + c)) for m, c in zip(mids, cuts)])
            print(slices)

            data = data[slices]
            labels = labels[slices]

            data = data[19]
            labels = labels[19]

            viewer.add_image(data, colormap="gray_r")
            viewer.add_labels(labels, opacity=1)

viewer.grid.enabled = True
viewer.grid.shape = (-1, 2)


napari.run()
