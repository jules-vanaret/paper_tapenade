import numpy as np
import napari
import morphsnakes
import tifffile
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

path_to_data = "/home/jvanaret/data/data_paper_tapenade/comparison_12views_globlocnorm"

input_image = tifffile.imread(f"{path_to_data}/Hoechst_FoxA2_Oct4_Bra_78h_big_1.tif")

viewer = napari.Viewer()
image_layer = viewer.add_image(input_image)


labels_layer = viewer.add_labels(
    np.zeros(image_layer.data.shape, dtype=int), name="blobs labels"
)

shapes_layer = viewer.add_shapes(
    shape_type=["ellipse"], face_color=np.array([1, 1, 1, 0])
)

viewer.layers.selection.active = labels_layer


global buffer
global count
count = [0]
global size
size = 30

append_each = 1


def iter_global(append_each):
    global buffer
    buffer = []
    # count = 0
    global count

    def iter_callback(u, count=count):
        print("hey", count)
        # if count[0]%append_each == 0:
        #
        buffer.append(u)
        count[0] = count[0] + 1

    return iter_callback


iter_callback = iter_global(append_each)


@labels_layer.mouse_drag_callbacks.append
def profile_lines_drag(layer, event):
    print("pushed clic")

    movable = False

    global buffer
    global mpl_widget

    if "Shift" in event.modifiers:
        global image_layer
        global shapes_layer
        global size

        shapes_layer.data = []
        init_pos = np.array(list(event.position), dtype=int)

        layer_shape = layer.data.shape

        roi_slices = tuple(
            slice(max(0, pos - size), min(layer_s, pos + size))
            for pos, layer_s in zip(init_pos[1:], layer_shape[1:])
        )
        roi_slices = (slice(init_pos[0], init_pos[0] + 1), *roi_slices)

        center_pos = np.array(tuple(size - max(size - pos, 0) for pos in init_pos[1:]))

        roi_data = image_layer.data[roi_slices][0]

        init_ls = morphsnakes.circle_level_set(roi_data.shape, center_pos, 3)

        snake = morphsnakes.morphological_chan_vese(
            roi_data,
            iterations=100,
            init_level_set=init_ls,
            smoothing=1,
            lambda1=2,
            lambda2=0.1,
            iter_callback=iter_callback,
        )

        shapes_layer.data = []
        shapes_layer.add_ellipses(np.array([init_pos[1:], [size, size]]))

    yield
    print("released clic 1")

    try:
        labels_layer.data[roi_slices] = buffer[-1] * labels_layer.selected_label
        labels_layer.refresh()
    except:
        pass

    while event.type == "mouse_move" and "Shift" in event.modifiers:

        # the yield statement allows the mouse UI to keep working while
        # this loop is executed repeatedly
        print(event.modifiers)

        print(len(buffer))

        dist = np.linalg.norm(np.subtract(event.position, init_pos))
        if dist > size:
            movable = True

        if movable:
            shapes_layer.data = []
            shapes_layer.add_ellipses(np.array([init_pos[1:], [dist, dist]]))

            print("accessing ", int(dist), " out of ", len(buffer) - 1)

            new_labels = buffer[min(len(buffer) - 1, int(dist))]

            labels_layer.data[roi_slices] = new_labels
            labels_layer.refresh()

        yield
        print("released clic 2")

    labels_layer.selected_label += 1

    buffer = []
    shapes_layer.data = []

    print("reset buffer")


if __name__ == "__main__":
    napari.run()
