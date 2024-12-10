## Score of a prediction compared to a ground truth, as a function of IoU thresholfd
## Assign cells from the prediction with cells from the GT, based on their IoU
## Discard the false detections based on IoU threshold.
## Compute the recall, precision and F1 score.
## Plot the results for each depth as a function of IoU threshold.

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from scipy.optimize import linear_sum_assignment
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


def array_tp(annotation, prediction, listTP):
    """
    Create arrays of True Positives cells based on the IoU matrix and the IoU threshold.

    Parameters
    ----------
    annotation : np.ndarray
    prediction : np.ndarray
    listTP : list
        A list of True Positives cells, with the format [gt_label, pred_label, IoU].

    """
    array_TP_annotation = np.zeros_like(annotation)
    array_TP_prediction = np.zeros_like(prediction)
    for label in listTP[0]:
        array_TP_annotation[annotation == label] = label
    for label in listTP[1]:
        array_TP_prediction[prediction == label] = label

    return array_TP_annotation, array_TP_prediction


folder = Path(__file__).parents[2] / 'data'
Score = []

list_IOUs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
list_z = [17, 30, 60, 89, 120, 150, 180, 210]
Scores = np.zeros((len(list_IOUs), len(list_z)))
image = tifffile.imread(
    Path(folder) / "3c_segmentation_performances/2_registered.tif"
)
colors = [
    "#77AADD",
    "#99DDFF",
    "#44BB99",
    "#BBCC33",
    "#AAAA00",
    "#EEDD88",
    "#EE8866",
    "#FFAABB",
]

for ind_thresh, thresh_IoU in enumerate(list_IOUs):
    for indz, z in enumerate(list_z):
        dapi = image[z, 0, :, :]
        annotation = tifffile.imread(
            Path(folder) / "3c_segmentation_performances/2_annotation.tif"
        )[z, :, :]
        prediction = tifffile.imread(
            Path(folder) / "3c_segmentation_performances/2_seg.tif"
        )[z, :, :]

        annotation = relabel_segmentation(annotation)
        prediction = relabel_segmentation(prediction)
        annotation = filter_tiny_volumes(annotation)
        prediction = filter_tiny_volumes(prediction)

        iou_matrix = build_iou_matrix(annotation, prediction)
        listTP = list_tp_from_iou_matrix(iou_matrix, thresh_IoU)
        nb_cells_annotated = len(np.unique(annotation)) - 1
        nb_cells_predicted = len(np.unique(prediction)) - 1
        if len(listTP) == 0:
            recall, precision, avg_iou = 0, 0, 0
        else:
            nb_tp = len(listTP[0])
            # array_tp_annotation,array_tp_seg=array_tp(annotation,prediction,listTP)
            Scores[ind_thresh, indz] = (
                nb_tp / nb_cells_annotated + nb_tp / nb_cells_predicted
            ) / 2

Scores = Scores.T


fig, ax = plt.subplots()
for ind_z, z in enumerate(list_z):
    ax.plot(
        list_IOUs, Scores[ind_z, :], color=colors[ind_z], label=f"z={z}µm", linewidth=2
    )
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.tick_params(axis="y", labelsize=25)
ax.tick_params(axis="x", labelsize=25)
ax.set_xlabel("IoU threshold", fontsize=25)
ax.set_ylabel("f1 score", fontsize=25)
plt.legend()
plt.savefig(Path(folder) / "3d_2_plot.svg")
plt.show()