## Comparison of 2 predictions ('corrected' for the version that is supposed to give the better performance, 'uncorrected' for the other, for example 2 views/1view or local normalization/global normalization)
## Assign cells from the prediction with cells from the GT, based on their IoU
## Discard the false detections based on IoU threshold.
## Compute the recall, precision and F1 score.
## Plot the results as a function of depth.

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from skimage.measure import regionprops

io.use_plugin("pil")


def build_iou_matrix(gt_segmentation, pred_segmentation):
    """
    Build a matrix of IoU values between the ground truth and predicted segmentations.

    Parameters
    ----------
    gt_segmentation : np.ndarray
        The ground truth segmentation.
    pred_segmentation : np.ndarray
        The predicted segmentation.

    Returns
    -------
    iou_matrix : np.ndarray
        A matrix of IoU values between the ground truth and predicted segmentations.
    """

    ndim = gt_segmentation.ndim

    gt_props = regionprops(gt_segmentation)
    pred_props = regionprops(pred_segmentation)

    iou_matrix = np.zeros((len(gt_props), len(pred_props)))

    for i_gt, gt_prop in enumerate(gt_props):

        gt_bb = gt_prop.bbox
        a = np.array(gt_bb[:ndim])
        b = np.array(gt_bb[ndim:])

        for i_pred, pred_prop in enumerate(pred_props):

            pred_bb = pred_prop.bbox
            c = np.array(pred_bb[:ndim])
            d = np.array(pred_bb[ndim:])

            if np.dot(a - d, b - c) < 0:  # if the bounding boxes intersect
                roi_slices = tuple(
                    slice(min(a[i], c[i]), max(b[i], d[i])) for i in range(ndim)
                )

                gt_roi_bool = gt_segmentation[roi_slices] == gt_prop.label
                pred_roi_bool = pred_segmentation[roi_slices] == pred_prop.label

                intersection = np.count_nonzero(gt_roi_bool * pred_roi_bool)

                if intersection > 0:
                    union = np.count_nonzero(gt_roi_bool + pred_roi_bool)
                    iou = intersection / union

                    iou_matrix[i_gt, i_pred] = iou

    return iou_matrix


def relabel_segmentation(segmentation):
    """
    Relabel a segmentation so that all numbers are consecutive integers starting from 0.
    Necessary for the iou matrix computation.
    """
    labels, label_counts = np.unique(segmentation, return_counts=True)
    temp = np.zeros(segmentation.shape).astype("uint16")
    for i, label in enumerate(labels):
        temp[segmentation == label] = i
    return temp


def filter_tiny_volumes(segmentation, thresh=50):
    """
    Filter out small volumes in a segmentation

    """
    new_segmentation = np.copy(segmentation)
    props = regionprops(segmentation)
    for prop in props:
        if prop.area < thresh:
            new_segmentation[new_segmentation == prop.label] = 0
    return new_segmentation


def array_tp(annotation, segmentation, listTP):
    """
    Create arrays of True Positives cells based on the IoU matrix and the IoU threshold.

    Parameters
    ----------
    annotation : np.ndarray
    segmentation : np.ndarray
    listTP : list
        A list of True Positives cells, with the format [gt_label, pred_label, IoU].

    """
    array_TP_annotation = np.zeros_like(annotation)
    array_TP_prediction = np.zeros_like(segmentation)
    for label in listTP[0]:
        array_TP_annotation[annotation == label] = label
    for label in listTP[1]:
        array_TP_prediction[segmentation == label] = label

    return array_TP_annotation, array_TP_prediction


def save_matrix_in_txt(matrix, filename):
    """
    Save the whole matrix into a txt file (to check the iou matrix)
    """
    with open(filename, "w") as file:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                file.write(str(i) + " " + str(j) + " " + str(matrix[i, j]) + "\n")


def list_tp_from_iou_matrix(iou_matrix, thresh_IoU):
    """
    From the iou matrix, maximizes the iou to assign pairs. Returns the list of True Positives cells only if they have an iou>chosen threshold

    Parameters
    ----------
    iou_matrix : np.ndarray
    thresh_IoU : float

    Returns
    -------
    listTP : list
        A list of True Positives cells, with the format [gt_label, pred_label, IoU].
    """
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

    listTP = [
        [gt_ind + 1, pred_ind + 1, iou_matrix[gt_ind, pred_ind]]
        for gt_ind, pred_ind in zip(row_ind, col_ind)
        if iou_matrix[gt_ind, pred_ind] > thresh_IoU
    ]

    return np.array(listTP).T


folder = Path(__file__).parents[3] / 'data'
list_z = [10, 50, 90, 130, 170, 210, 250]
image = tifffile.imread(Path(folder)/"S2_segmentation_performances/S2f_1_2views/image.tif")

Z = []
ious = []
Precision_u = [] #uncorrected (1 view)
Recall_u = []
Precision_c = []#corrected (2 views)
Recall_c = []

thresh_IoU = 0.5
# viewer=napari.Viewer()
annotation_3D = tifffile.imread(Path(folder)/"S2_segmentation_performances/S2f_1_2views/annotation_1_2views.tif")
segmentation_corrected_3D = tifffile.imread(Path(folder)/"S2_segmentation_performances/S2f_1_2views/labels_2views.tif")
segmentation_uncorrected_3D = tifffile.imread(Path(folder)/"S2_segmentation_performances/S2f_1_2views/labels_1view.tif")
for z in list_z:
    dapi = image[
        z, image.shape[1] // 2 :, :
    ]  # annotated only in the bottom part of the image
    annotation = annotation_3D[
        z, image.shape[1] // 2 :, :
    ]
    segmentation_corrected = (segmentation_corrected_3D[
            z, image.shape[1] // 2 :, :
        ]
    ).astype(int)


    segmentation_corrected = filter_tiny_volumes(
        segmentation_corrected
    )  # the 3D segmentation after cuting 1 slice shows small volumes that we dont want to assign, we filter them out
    annotation = filter_tiny_volumes(annotation)
    annotation = relabel_segmentation(annotation)
    segmentation_corrected = relabel_segmentation(segmentation_corrected)

    iou_matrix = build_iou_matrix(annotation, segmentation_corrected)
    listTP = list_tp_from_iou_matrix(iou_matrix, thresh_IoU)
    # array_tp_annotation,array_tp_seg_corrected=array_tp(annotation,segmentation_corrected,listTP) #possibility to save and plot this to check the assignation
    nb_cells_annotated = len(np.unique(annotation)) - 1
    nb_cells_predicted = len(np.unique(segmentation_corrected)) - 1

    if len(listTP) == 0:
        recall, precision, avg_iou = 0, 0, 0
    else:
        nb_tp = len(listTP[0])
        recall = nb_tp / nb_cells_annotated
        precision = nb_tp / nb_cells_predicted
        avg_iou = np.median(listTP[2])

    Z.append(z)
    Precision_c.append(precision)
    Recall_c.append(recall)

    annotation = tifffile.imread(Path(folder)/"S2_segmentation_performances/S2f_1_2views/annotation_1_2views.tif")[
        z, image.shape[1] // 2 :, :
    ]
    segmentation_uncorrected = (
        segmentation_uncorrected_3D[
            z, image.shape[1] // 2 :, :
        ]
    ).astype(int)

    segmentation_uncorrected = filter_tiny_volumes(segmentation_uncorrected)
    segmentation_uncorrected = relabel_segmentation(segmentation_uncorrected)
    iou_matrix = build_iou_matrix(annotation, segmentation_uncorrected)
    listTP = list_tp_from_iou_matrix(iou_matrix, thresh_IoU)

    nb_cells_annotated = len(np.unique(annotation)) - 1
    nb_cells_predicted = len(np.unique(segmentation_uncorrected)) - 1
    if len(listTP) == 0:
        recall, precision, avg_iou = 0, 0, 0
    else:
        nb_tp = len(listTP[0])
        recall = nb_tp / nb_cells_annotated
        precision = nb_tp / nb_cells_predicted
        avg_iou = np.median(listTP[2])
        # array_tp_annotation,array_tp_seg_uncorrected=array_tp(annotation,segmentation_corrected,listTP)

    Precision_u.append(precision)
    Recall_u.append(recall)

f1_corrected = [(p + r) / 2 for p, r in zip(Precision_c, Recall_c)]
f1_uncorrected = [(p + r) / 2 for p, r in zip(Precision_u, Recall_u)]

fig, ax = plt.subplots()

# plot recall and precision
# ax.plot(Z,Precision_c,"o-",color=cset.rose,label='precision with two views',linewidth=4)
# ax.plot(Z,Recall_c,"o-",color=cset.cyan,label='recall with two views',linewidth=4)
# ax.plot(Z,Precision_u,"x--",color=cset.rose,label='precision with one view',linewidth=4)
# ax.plot(Z,Recall_u,"x--",color=cset.cyan,label='recall with one view',linewidth=4)

# plot f1
ax.plot(Z, f1_corrected, "o-", color="cyan", label="two views", linewidth=3)
ax.plot(
    Z, f1_uncorrected, "o--", color="cyan", label="one view", linewidth=3
)


ax.set_xlabel("depth(µm)", fontsize=25)
ax.set_xticks([0, 100, 200])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.tick_params(axis="y", labelsize=25)
ax.tick_params(axis="x", labelsize=25)
ax.set_ylabel("f1 score", fontsize=25)
lines_1, labels_1 = ax.get_legend_handles_labels()
plt.legend()
plt.tight_layout()
# plt.savefig(Path(folder) / "S2d_plot2.svg")
plt.show()
